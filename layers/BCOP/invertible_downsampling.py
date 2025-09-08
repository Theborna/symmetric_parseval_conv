"""
Invertible Downsampling is implemented as described in https://arxiv.org/pdf/1802.07088.pdf.
"""
import torch
import torch.nn as nn
from einops import rearrange

class PixelUnshuffle2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

    def lipschitz_constant(self):
        return 1

    def forward(self, x):
        return rearrange(
            x,
            "b c (w k1) (h k2) -> b (c k1 k2) w h",
            k1=self.kernel_size[0],
            k2=self.kernel_size[1],
        )

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)
    
""""
Code from https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/tests.py
"""
class psi_legacy(nn.Module):  # Original implementation of class model_utils.psi
    def __init__(self, block_size):
        super(psi_legacy, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()
    
class psi(nn.Module):
    def __init__(self, block_size, mode="up"):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size
        self.mode = mode

    def _inverse(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = input.shape[0], input.shape[1] // bl_sq, input.shape[2], input.shape[3]
        return input.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)

    def _forward(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl
        return input.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)
    
    def forward(self, input):
        if self.mode == "up":
            return self._forward(input)
        elif self.mode == "down":
            return self._inverse(input)
        else:
            raise ValueError("not a valid mode of operation")
        
    def __repr__(self):
        return f"psi(mode={self.mode}, block_size={self.block_size})"
    
if __name__ == "__main__":
    block_size = 3
    t = torch.randn(64, 5, 192, 192, dtype=torch.float32)
    psi_new_result = psi(block_size)._forward(t.clone())
    psi_old_result = psi_legacy(block_size).forward(t.clone())
    assert ((psi_new_result == psi_old_result).all().item()), "nope 1"
    assert (psi_new_result.is_contiguous()), "nope 2"
    
    block_size = 3
    t = torch.randn(64, 45, 64, 64, dtype=torch.float32)
    psi_new_result = psi(block_size)._inverse(t.clone())
    psi_old_result = psi_legacy(block_size).inverse(t.clone())
    assert ((psi_new_result == psi_old_result).all().item()), "nope 3"
    assert (psi_new_result.is_contiguous()), "nope 4"
    
    block_size = 3
    t = torch.nn.Parameter(torch.randn(64, 5, 192, 192, dtype=torch.float32))
    optimizer = torch.optim.Adam(t.data)
    print(t)
    t_ = psi(block_size)._forward(t)
    t_ = psi(block_size)._inverse(t_)
    print(t_)
    loss = t_.sum()
    loss.backward()
    assert (t == t_).all().item(), "nope 4"
    