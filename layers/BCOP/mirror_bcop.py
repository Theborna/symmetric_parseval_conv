from layers.BCOP.bcop import BCOP, convolution_orthogonal_generator_projs
import torch
from layers.BCOP.utils import StreamlinedModule, conv_clip_2_norm_numpy, conv_singular_values_numpy, conv2d_cyclic_pad, bjorck_orthonormalize
import torch.nn as nn
import numpy as np

depth = 0

class MirrorBCOP(BCOP):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        bias=False,
        bjorck_iters=16,
        power_iteration_scaling=True,
        frozen=False,
        middle_network=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            bjorck_iters,
            power_iteration_scaling,
            frozen,
        )

        self.middle_network = middle_network
        
        # Store only half the matrices
        self.param_matrices = nn.Parameter(
            torch.Tensor(self.num_kernels - 1, self.max_channels, self.max_channels),
            requires_grad=not self.frozen,
        )

        # Generate only half the mask
        self.mask = nn.Parameter(
            torch.cat(
                (
                    torch.ones(self.num_kernels - 1, 1, self.max_channels // 2),
                    torch.zeros(
                        self.num_kernels - 1,
                        1,
                        self.max_channels - self.max_channels // 2,
                    ),
                ),
                dim=-1,
            ).float(),
            requires_grad=False,
        )

        self.order = nn.Parameter(
            torch.zeros((self.num_kernels - 1) // 2, 1),
            requires_grad=False
        )

        # Initialize the half parameters
        self.reset_parameters()
                
    def reset_parameters(self):
        ortho_weights = [
            torch.empty(self.max_channels, self.max_channels)
            for _ in range(self.num_kernels - 1)
        ]
        stdv = 1.0 / (self.max_channels ** 0.5)
        for index, ortho_weight in enumerate(ortho_weights):
            nn.init.orthogonal_(ortho_weight, gain=stdv)
            self.param_matrices.data[index] = ortho_weight

        std = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def _gen_weights(self):
        if not self.streamline or self.weight is None:
            # Orthogonalize the matrices using Bjorck
            ortho = bjorck_orthonormalize(
                self.param_matrices,
                iters=self.bjorck_iters,
                power_iteration_scaling=self.power_iteration_scaling,
                default_scaling=not self.power_iteration_scaling,
            )

            # Generate the mirrored versions
            ortho_mirrored = ortho.flip(0)
            mask_mirrored  = self.mask.flip(0).flip(-1)
            order_mirrored = 1 - self.order.flip(0)
            
            PQ          = ortho * self.mask
            PQ_mirrored = ortho_mirrored * mask_mirrored
            
            PQ          = PQ @ PQ.transpose(-1, -2)
            PQ_mirrored = PQ_mirrored @ PQ_mirrored.transpose(-1, -2)

            # Compute the resulting convolution kernels using block convolutions
            self.weight = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, None, PQ, order=self.order
            )
            self.weight_mirrored = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, None, PQ_mirrored, order=order_mirrored
            )

        self.buffer_weight = self.weight
        self.buffer_weight_mirrored = self.weight_mirrored
        
    
        
    def forward(self, x):
        self._input_shape = x.shape[2:]  # cache the input shape for self.singular_values()

        self._gen_weights()

        # detach the weights when we are using the cached weights from previous steps
        if self.streamline:
            weight = self.weight.detach()
            weight_mirrored = self.weight_mirrored.detach()
        else:
            weight = self.weight
            weight_mirrored = self.weight_mirrored

        # apply cyclic padding to the input and perform two consecutive convolutions
        x = conv2d_cyclic_pad(x, weight, None, dilation=self.dilation)
        x = self.middle_network(x) if self.middle_network is not None else x
        x = conv2d_cyclic_pad(x, weight_mirrored, self.bias, dilation=self.dilation)
                
        return x
    
    # Assuming the structure from before
    def forward_half(self, x, apply_first=False):
        self._gen_weights()

        # apply cyclic padding to the input and perform two consecutive convolutions
        x = conv2d_cyclic_pad(x, self.weight, None, dilation=self.dilation)

        global depth
        print('at depth: ', depth + 1)
        
        depth += 1
        if not hasattr(self.middle_network, 'forward_half'):
            depth = 0
            return x
        return self.middle_network.forward_half(x, apply_first=apply_first)
    
    def forward_middle(self, x):
        self._gen_weights()
    
        # apply cyclic padding to the input and perform two consecutive convolutions
        x = conv2d_cyclic_pad(x, self.weight, None, dilation=self.dilation)
        x = self.middle_network.forward_middle(x) if hasattr(self.middle_network, 'forward_middle') else x
        x = conv2d_cyclic_pad(x, self.weight_mirrored, self.bias, dilation=self.dilation)
                
        return x
    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, bias={self.bias is not None}"
    

class SymMirrorBCOP(MirrorBCOP):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        bias=False,
        bjorck_iters=25,
        power_iteration_scaling=True,
        frozen=False,
        middle_network=None,
        first=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias,
            bjorck_iters,
            power_iteration_scaling,
            frozen,
            middle_network,
        )
        
        self.first = first
        self.max_channels = max(self.in_channels, self.out_channels)

        # Replace param_matrices with U0 and U1
        del self.param_matrices
        self.U0 = nn.Parameter(
            torch.Tensor(self.num_kernels - 1, self.max_channels // 2, self.max_channels // 2),
            requires_grad=not self.frozen,
        )
        self.U1 = nn.Parameter(
            torch.Tensor(self.num_kernels - 1, self.max_channels - self.max_channels // 2, self.max_channels - self.max_channels // 2),
            requires_grad=not self.frozen,
        )
        
        # Initial exchange matrix
        I = torch.eye(self.max_channels // 2)
        J = torch.flip(torch.eye(self.max_channels - self.max_channels // 2), dims=[1])
        self.perm_matrix = nn.Parameter(
            torch.block_diag(I, J).float(),
            requires_grad=False,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = np.sqrt(2) / (self.max_channels ** 0.5)
        nn.init.orthogonal_(self.U0, gain=stdv)
        nn.init.orthogonal_(self.U1, gain=stdv)

        if self.bias is not None:
            std = 1.0 / np.sqrt(self.out_channels)
            nn.init.uniform_(self.bias, -std, std)

    def construct_U_matrix(self, U0, U1):
        U = torch.empty(self.num_kernels - 1, self.max_channels, self.max_channels, device=U0.device)
        for k in range(self.num_kernels - 1):
            u0, u1 = U0[k, :, :], U1[k, :, :]
            x, y = torch.cat((u0, u0, u1, -u1), dim=1).t().chunk(2)
            U[k, :, :] = torch.cat((x, y), dim=1).t() / np.sqrt(2.0)
        return U

    def _gen_weights(self):
        if not self.streamline or self.weight is None:
            # Orthogonalize U0 and U1 using Bjorck
            U0_ortho = bjorck_orthonormalize(
                self.U0,
                iters=self.bjorck_iters,
                power_iteration_scaling=self.power_iteration_scaling,
                default_scaling=not self.power_iteration_scaling,
            )
            U1_ortho = bjorck_orthonormalize(
                self.U1,
                iters=self.bjorck_iters,
                power_iteration_scaling=self.power_iteration_scaling,
                default_scaling=not self.power_iteration_scaling,
            )

            # Convert list to tensor
            ortho = self.construct_U_matrix(U0_ortho, U1_ortho)

            # Generate the mirrored versions
            ortho_mirrored = ortho.flip(0)
            mask_mirrored  = self.mask.flip(0).flip(-1)
            order_mirrored = 1 - self.order.flip(0)
            
            PQ          = ortho * self.mask
            PQ_mirrored = ortho_mirrored * mask_mirrored
            
            PQ          = PQ @ PQ.transpose(-1, -2)
            PQ_mirrored = PQ_mirrored @ PQ_mirrored.transpose(-1, -2)
            
            # Compute the resulting convolution kernels using block convolutions
            self.weight = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, None, PQ,
                p_init={(0, 0): self.perm_matrix} if self.first else None,
                order=self.order
            )
            self.weight_mirrored = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, None, PQ_mirrored,
                p_init={(0, 0): self.perm_matrix} if self.first else None,
                order=order_mirrored
            )

        self.buffer_weight = self.weight
        self.buffer_weight_mirrored = self.weight_mirrored