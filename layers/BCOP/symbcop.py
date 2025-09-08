from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from layers.BCOP.utils import conv_clip_2_norm_numpy, conv_singular_values_numpy, conv2d_cyclic_pad, bjorck_orthonormalize
from layers.BCOP.invertible_downsampling import PixelUnshuffle2d
from layers.BCOP.linear import BjorckLinear
from layers.BCOP.bcop import *

class SymBCOP(StreamlinedModule, LipschitzModuleL2):
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
        first=False,
    ):
        super().__init__()
        assert stride == 1, "BCOP convolution only supports stride 1."
        assert padding is None or padding == (dilation * kernel_size) // 2, "BCOP convolution only supports d * k // 2 padding. actual - {}, required - {}".format(padding, kernel_size // 2)

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation if type(dilation) == tuple else (dilation, dilation)           

        self.max_channels = max(self.in_channels, self.out_channels)
        self.num_kernels = 2 * (kernel_size - 1) + 1
        self.bjorck_iters = bjorck_iters
        self.power_iteration_scaling = power_iteration_scaling
        self.frozen = frozen
        self.first = first

        # Define the unconstrained matrices U0 and U1 for constructing U matrices
        self.U0 = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels // 2, self.max_channels // 2),
            requires_grad=not self.frozen,
        )
        self.U1 = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels - self.max_channels // 2, self.max_channels - self.max_channels // 2),
            requires_grad=not self.frozen,
        )

        # The mask controls the rank of the symmetric projectors (full half rank).
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
        
        # Initial exchange matrix
        I  = torch.eye(self.max_channels // 2)
        J = torch.flip(torch.eye(self.max_channels - self.max_channels // 2), dims=[1])
    
        self.perm_matrix = nn.Parameter(
            torch.block_diag(
                I, J    
            ).float(),
            requires_grad=False,
        )

        # Bias parameters in the convolution
        self.enable_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels), requires_grad=not self.frozen
            )
        else:
            self.bias = None

        # Initialize the weights (self.weight is set to zero for streamline module)
        self.reset_parameters()
        self.weight = None
        self.mirror = False

    def set_streamline(self, streamline=False):
        # Implements interface required by StreamlineModule
        super().set_streamline(streamline=streamline)
        if streamline:
            self.weight = None

    def singular_values(self):
        # Implements interface required by LipschitzModuleL2
        svs = torch.from_numpy(
            conv_singular_values_numpy(
                self.buffer_weight.detach().cpu().numpy(), self._input_shape
            )
        ).to(device=self.buffer_weight.device)
        return svs

    def reset_parameters(self):
        stdv = np.sqrt(2) / (self.max_channels ** 0.5)
        nn.init.orthogonal_(self.U0, gain=stdv)
        nn.init.orthogonal_(self.U1, gain=stdv)

        std = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def construct_U_matrix(self, U0, U1):
        """
        Construct the U matrix as [U0, U0; U1, -U1]
        
        Note: The last matrix is constructed differently with a block diagonal structured diag([U0, U1])
        """
        U = torch.Tensor(self.num_kernels, self.max_channels, self.max_channels).to("cuda")
        for k in range(self.num_kernels):
            u0, u1 = U0[k, :, :], U1[k, :, :]
            x, y = torch.cat((u0, u0, u1, -u1), dim=1).t().chunk(2)
            U[k, :, :]    = torch.cat((x,y),dim=1).t() / np.sqrt(2.0)
        U[-1, :, :] = torch.block_diag(u0, u1)
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

            # Compute the symmetric projectors
            H = ortho[-1, :self.in_channels, :self.out_channels]
            PQ = ortho[:-1]
            PQ = PQ * self.mask
            PQ = PQ @ PQ.transpose(-1, -2)

            # Compute the resulting convolution kernel using block convolutions
            self.weight = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, H, PQ, p_init={(0,0): self.perm_matrix} if self.first else None
            )
        self.buffer_weight = self.weight
    
    def forward(self, x):
        self._input_shape = x.shape[2:]  # cache the input shape for self.singular_values()
        
        self._gen_weights()

        if self.streamline:
            weight = self.weight.detach()
        else:
            weight = self.weight

        if self.mirror:
            weight = weight.permute(1, 0, 2, 3).flip([2, 3])

        # Apply cyclic padding to the input and perform a standard convolution
        return conv2d_cyclic_pad(x, weight, self.bias, dilation=self.dilation)

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, bias={enable_bias}, dilation={dilation}".format(
            **self.__dict__
        )
