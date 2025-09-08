"""
BCOP parameterization with block convolution procedure adapted from the official Tensorflow repo:
https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/ops/init_ops.py#L683
"""

from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from layers.BCOP.utils import StreamlinedModule, conv_clip_2_norm_numpy, conv_singular_values_numpy, conv2d_cyclic_pad, bjorck_orthonormalize
from layers.BCOP.core import LipschitzModuleL2
from layers.BCOP.invertible_downsampling import PixelUnshuffle2d
from layers.BCOP.linear import BjorckLinear
from layers.BCOP.bcop import dict_to_tensor, matrix_conv

def convolution_orthogonal_generator_unitaries(ksize, cin, cout, ortho, UV, mask, p_init=None):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
        ortho = ortho.t()
    if ksize == 1:
        return ortho.unsqueeze(-1).unsqueeze(-1)
    p = block_orth_unitary(UV[1], UV[0], mask[1], mask[0])
    if p_init is not None:
        p = matrix_conv(p, p_init)
    for _ in range(1, ksize - 1):
        p = matrix_conv(block_orth_unitary(UV[_ * 2 + 1], UV[_ * 2], mask[_ * 2 + 1], mask[_ * 2]), p)
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = ortho.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)

# def block_orth_unitary(U, V, mask1, mask2):
#     """Construct a 2 x 2 kernel using unitary matrices U and V.
#     Args:
#       U: A unitary matrix.
#       V: A unitary matrix.
#       mask1: A mask vector for U.
#       mask2: A mask vector for V.
#     Returns:
#       A 2 x 2 kernel:
#       [[M2 * V * M1 * U,         M2 * V * (1 - M1) * U],
#        [(1 - M2) * V * M1 * U, (1 - M2) * V * (1 - M1) * U]].
#     Raises:
#       ValueError: If the dimensions of U and V are different.
#     """
#     assert U.shape == V.shape
#     n = U.size(0)
#     kernel2x2 = {}
#     eye = torch.eye(n, device=U.device, dtype=U.dtype)
#     M1, M2  = mask1.diagflat(), mask2.diagflat()
#     M1_, M2_ = eye - M1, eye - M2
#     kernel2x2[0, 0] = M2.mm(V).mm(M1).mm(U)
#     kernel2x2[0, 1] = M2.mm(V).mm(M1_).mm(U)
#     kernel2x2[1, 0] = M2_.mm(V).mm(M1).mm(U)
#     kernel2x2[1, 1] = M2_.mm(V).mm(M1_).mm(U)

#     return kernel2x2

def block_orth_unitary(V, U, mask2, mask1):
    """Construct a 2 x 2 kernel using unitary matrices U and V.
    Args:
      U: A unitary matrix.
      V: A unitary matrix.
      mask1: A mask vector for U.
      mask2: A mask vector for V.
    Returns:
      A 2 x 2 kernel:
      [[M2 * V * M1 * U,             M2 * V * (1 - M1) * U],
       [(1 - M2) * V * M1 * U, (1 - M2) * V * (1 - M1) * U]].
    Raises:
      ValueError: If the dimensions of U and V are different.
    """
    assert U.shape == V.shape, "U and V must have the same dimensions"

    # Apply masks directly through element-wise multiplication
    M1U = U * mask1
    M1_U = U * (1 - mask1)
    M2V = V * mask2
    M2_V = V * (1 - mask2)

    # Compute the 2x2 block orthogonal kernel
    kernel2x2 = {}
    kernel2x2[0, 0] = M2V @ M1U
    kernel2x2[0, 1] = M2V @ M1_U
    kernel2x2[1, 0] = M2_V @ M1U
    kernel2x2[1, 1] = M2_V @ M1_U

    return kernel2x2


class SVD(StreamlinedModule, LipschitzModuleL2):
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

        # Define the unconstrained matrices Ms and Ns for Ps and Qs
        self.param_matrices = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels, self.max_channels),
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
        if streamline == True:
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
        ortho_weights = [
            torch.empty(self.max_channels, self.max_channels)
            for i in range(self.num_kernels)
        ]
        stdv = 1.0 / (self.max_channels ** 0.5)
        for index, ortho_weight in enumerate(ortho_weights):
            nn.init.orthogonal_(ortho_weight, gain=stdv)
            self.param_matrices.data[index] = ortho_weight

        std = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def _gen_weights(self):
        # streamline controls whether the weight from previous steps are being used
        if not self.streamline or self.weight is None:
            # orthognoalize all the matrices using Bjorck
            ortho = bjorck_orthonormalize(
                self.param_matrices,
                iters=self.bjorck_iters,
                power_iteration_scaling=self.power_iteration_scaling,
                default_scaling=not self.power_iteration_scaling,
            )

            # compute the symmetric projectors
            H = ortho[-1, :self.in_channels, :self.out_channels]
            UV = ortho[:-1]

            # compute the resulting convolution kernel using block convolutions
            self.weight = convolution_orthogonal_generator_unitaries(
                self.kernel_size, self.in_channels, self.out_channels, H, UV, self.mask
            )
        self.buffer_weight = self.weight
            
    def forward(self, x):
        self._input_shape = x.shape[
            2:
        ]  # cache the input shape for self.singular_values()

        self._gen_weights()

        # detach the weight when we are using the cached weights from previous steps
        if self.streamline: weight = self.weight.detach()
        else: weight = self.weight

        if self.mirror:
            weight = weight.permute(1, 0, 2, 3).flip([2, 3])

        # apply cyclic padding to the input and perform a standard convolution
        return conv2d_cyclic_pad(x, weight, self.bias, dilation=self.dilation)

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, bias={enable_bias}, dilation={dilation}".format(
            **self.__dict__
        )

