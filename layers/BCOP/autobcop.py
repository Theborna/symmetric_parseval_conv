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
from layers.BCOP.bcop import matrix_conv, dict_to_tensor, block_orth

def directed_block_orth(p, alpha):
    """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
    Args:
      p1: A symmetric projection matrix.
    Returns:
      A 1 x 2 kernel [p1, I-p1] if alpha = 0
      A 2 x 1 kernel [[p1]; [I-p1]] if alpha = 1
      A 2 x 2 kernel [[P                ,  alpha * (I-P)
                      (1-alpha) * (I-P) ,  0]]
    Raises:
      ValueError: If the alpha is not between 0 or 1
    """
    assert alpha <= 1.0 and alpha >= 0.0
    n = p.size(0)
    kernel = {}
    eye = torch.eye(n, device=p.device, dtype=p.dtype)
    kernel[0, 0] = p
    kernel[0, 1] = (1.0 - alpha) * (eye - p)
    kernel[1, 0] = (alpha) * (eye - p)
    kernel[1, 1] = torch.zeros_like(p)

    return kernel

def dilated_directed_block_orth(p, alpha):
    """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
    Args:
      p1: A symmetric projection matrix.
    Returns:
      A 3 x 3 kernel [[0,  alpha * P,  0
                      (1-alpha)*P, 0, (1-alpha)*(I-P)
                      0, alpha * (I-P), 0]]
    Raises:
      ValueError: If the alpha is not between 0 or 1
    """
    assert alpha <= 1.0 and alpha >= 0.0
    n = p.size(0)
    kernel = {}
    eye = torch.eye(n, device=p.device, dtype=p.dtype)
    for i in range(3):
        for j in range(3):
            kernel[i, j] = torch.zeros_like(p)
    kernel[1, 0] = (alpha)       * p
    kernel[0, 1] = (1.0 - alpha) * p
    kernel[2, 1] = (1.0 - alpha) * (eye - p)
    kernel[1, 2] = (alpha)       * (eye - p)

    return kernel

import json

def directed_convolution_orthogonal_generator_projs(ksize, cin, cout, ortho, sym_projs, direction, p_init=None, mode=0):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
        ortho = ortho.t()
    if ksize == 1:
        return ortho.unsqueeze(-1).unsqueeze(-1)
    block = directed_block_orth if mode == 0 else dilated_directed_block_orth    
    
    p = block(sym_projs[0], direction[0])
    
    if p_init is not None:
        p = matrix_conv(p, p_init)
    for _ in range(1, ksize - 1):
        p = matrix_conv(block(sym_projs[_], direction[_]), p)
        
    ksize = 2 * ksize - 1
    zero = torch.zeros_like(p[0, 0])
    for i in range(ksize):
        for j in range(ksize):
            if torch.allclose(p[i, j], zero):
                continue
            p[i, j] = ortho.mm(p[i, j])
            
    # print(p.keys())
    # print("\n")
    # print(json.dumps({f"{k}": torch.sum(v ** 2).item() for k,v  in p.items() if not torch.allclose(v, zero)}, indent = 4))
    # print({f"{k}": v for k,v  in p.items() if not torch.allclose(v, zero)})
    # print("\n")
            
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)

def entropy_loss(dir, natural=True, p=1):
    dir_ = dir.view(-1)
    dir_ = torch.clamp(dir_, 1e-7, 1.0 - 1e-7)
    
    if p == 1:
        entropy = torch.sum(
            - (dir_ * torch.log(dir_) + (1 - dir_) * torch.log(1 - dir_))
        )
    else:
        entropy = np.log(2) * torch.sum(
            (dir_ ** p + (1 - dir_) ** p - 1) / (2.0 ** (1 - p) - 1)
        )

    # Register a hook for natural gradients
    def natural_grad_backward_hook(grad):
        return grad * dir_ * (1 - dir_)

    # Attach the hook to the direction tensor
    if dir.requires_grad and natural:
        handle = dir_.register_hook(natural_grad_backward_hook)

    return entropy


def upsample_with_zeros(x, scale_factor):
    # Create an empty tensor with the shape of the upsampled output
    upsampled_shape = (x.shape[0], x.shape[1], x.shape[2] * scale_factor, x.shape[3] * scale_factor)
    upsampled_x = torch.zeros(upsampled_shape, device=x.device, dtype=x.dtype)

    # Place the original values into the correct positions
    upsampled_x[:, :, ::scale_factor, ::scale_factor] = x
    return upsampled_x

class AutoBCOP(StreamlinedModule, LipschitzModuleL2):
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
        auto_dir=True,
        up_down_sample=True,
        entropic_loss_p=1,  
        natural_grad=False,
        optim_rank=False,
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
        self.auto_dir = auto_dir
        self.optim_rank = optim_rank
        self.up_down_sample = up_down_sample
        
        self.largenum = 1e2
        self.entropic_loss_p = entropic_loss_p
        self.natural_grad = natural_grad
        
        # Define the unconstrained matrices Ms and Ns for Ps and Qs
        self.param_matrices = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels, self.max_channels),
            requires_grad=not self.frozen,
        )

        # The mask controls the rank of the symmetric projectors (full half rank).
        if self.optim_rank:
            self.mask = nn.Parameter(
                torch.Tensor(self.num_kernels - 1 , 1, self.max_channels), requires_grad=(not self.frozen)
            )
        else:
            self.mask = nn.Parameter(
                torch.cat(
                    (
                        self.largenum * torch.ones(self.num_kernels - 1, 1, self.max_channels // 2),
                        -self.largenum * torch.ones(
                            self.num_kernels - 1,
                            1,
                            self.max_channels - self.max_channels // 2,
                        ),
                    ),
                    dim=-1,
                ).float(),
                requires_grad=False,
            )
                    
        # THe directions control the orientation of each kernel
        self.direction = nn.Parameter(
            torch.Tensor(self.num_kernels - 1), requires_grad=(not self.frozen) and self.auto_dir
        )
        if not self.auto_dir:
            for _ in range(len(self.direction)):
                self.direction.data[_] = ((-1) ** _) * self.largenum
            
        self.entropic_loss = torch.tensor(0.0)
        
        # Bias parameters in the convolution
        self.enable_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.num_kernels - 1), requires_grad=not self.frozen
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
            
        if self.auto_dir:
            std = 6.0 / np.sqrt(len(self.direction))
            nn.init.uniform_(self.direction, -std, std)

        if self.optim_rank:
            std = 6.0 / np.sqrt(self.max_channels)
            nn.init.uniform_(self.mask, -std, std)

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
            H = ortho[0, : self.in_channels, : self.out_channels]
            PQ = ortho[1:]
            PQ = PQ * torch.sigmoid(self.mask)
            PQ = PQ @ PQ.transpose(-1, -2)

            # compute the resulting convolution kernel using block convolutions
            self.weight = directed_convolution_orthogonal_generator_projs(
                self.num_kernels, self.in_channels, self.out_channels, H, PQ, torch.sigmoid(self.direction), mode=1
            )
        self.buffer_weight = self.weight

    def forward(self, x):
        self._input_shape = x.shape[
            2:
        ]  # cache the input shape for self.singular_values()

        self._gen_weights()
        
        # print(self.weight.shape)
        # print(self.weight[0, 0, :, :])
        
        entropic_loss =  entropy_loss(torch.sigmoid(self.direction), natural=self.natural_grad, p=self.entropic_loss_p)
        entropic_loss += entropy_loss(torch.sigmoid(self.mask), natural=self.natural_grad, p=self.entropic_loss_p)
        self.entropic_loss = entropic_loss if self.auto_dir else entropic_loss.detach() 
        
        # detach the weight when we are using the cached weights from previous steps
        if self.streamline: weight = self.weight.detach()
        else: weight = self.weight

        if self.mirror:
            weight = weight.permute(1, 0, 2, 3).flip([2, 3])

        # apply cyclic padding to the input and perform a standard convolution
        x = upsample_with_zeros(x, scale_factor=2) if self.up_down_sample else x
        # print("x:", x[0, 0, :8, :8])
        y = conv2d_cyclic_pad(x, weight, self.bias, dilation=self.dilation)
        # print("y1:", y[0, 0, :8, :8])
        y = F.avg_pool2d(y, kernel_size=2, divisor_override=1) if self.up_down_sample else y
        # print("y2:", y[0, 0, :8, :8])
        return y

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, bias={enable_bias}, dilation={dilation}, auto_dir={auto_dir}, p={entropic_loss_p}, natural_grads={natural_grad}, optim_rank={optim_rank}".format(
            **self.__dict__
        )

