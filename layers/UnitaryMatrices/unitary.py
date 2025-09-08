import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import parametrizations
import layers.shifts as shifts
from layers.BCOP.utils import bjorck_orthonormalize

class UnitaryMatrix(nn.Module):
    def __init__(self, num_channels, out_channels=None, p=1, initial=None):
        super(UnitaryMatrix, self).__init__()
        self.num_channels = num_channels
        if out_channels is None:
            out_channels = num_channels
        self.out_channels = out_channels
        self.max_channels = max(self.num_channels, self.out_channels)
        self.p = p
        # Initialize orthogonal matrices for channel mixing
        self.orthogonal_matrices = nn.ParameterList(
            [nn.Parameter(torch.empty(out_channels, num_channels)) for _ in range(self.p)]
        )
        
        # Apply orthogonal initialization to each matrix
        for idx, matrix in enumerate(self.orthogonal_matrices):
            if initial and idx < len(initial):
                assert initial[idx].shape == (out_channels, num_channels), (
                    f"Initial matrix at index {idx} has shape {initial[idx].shape}, "
                    f"expected ({out_channels}, {num_channels})"
                )
                matrix.data = initial[idx]
            else:
                nn.init.orthogonal_(matrix, 1.0 / (self.max_channels ** 0.5))
            
    def forward(self, x, transposed=False):
        batch_size, num_channels, height, width = x.size()
        if transposed:
            assert num_channels == self.max_channels, "Number of channels must match the size of the orthogonal matrix"            
        else:
            assert num_channels == self.num_channels, "Number of channels must match the size of the orthogonal matrix"
        
        # Reshape to (batch_size, num_channels, height * width) for matrix multiplication
        x = x.view(batch_size, num_channels, -1)
        # Apply the orthogonal matrices to mix the channels
        y = torch.zeros(batch_size, (self.num_channels if transposed else self.out_channels) * self.p, height * width, device=x.device)
        for i in range(self.p):
            A = self.orthogonal_matrices[i]
            # Q, R = torch.linalg.qr(self.orthogonal_matrices[i].data)
            # U, _, Vh = torch.linalg.svd(A, full_matrices=False)
            # Q = U @ Vh
            power_iteration_scaling = False
            bjorck_iters = 25
            Q = bjorck_orthonormalize(
                A,
                iters=bjorck_iters,
                power_iteration_scaling=power_iteration_scaling,
                default_scaling=not power_iteration_scaling,
            )
            
            self.orthogonal_matrices[i].data = Q
            
            start_idx = i * self.out_channels
            end_idx = (i + 1) * self.out_channels
            y[:, start_idx:end_idx, :] = torch.matmul(self.orthogonal_matrices[i].t() if transposed else self.orthogonal_matrices[i], x)
        
        # Reshape back to (batch_size, num_channels * self.p, height, width)
        y = y.view(batch_size, (self.num_channels if transposed else self.out_channels) * self.p, height, width) / torch.sqrt(torch.tensor(self.p, dtype=torch.float32))
        
        return y
    
    def __repr__(self):
        return (f"UnitaryMatrix(num_channels={self.num_channels}, out_channels={self.out_channels}, p={self.p})")
    
class UnitaryTransposed(nn.Module):
    def __init__(self, unitary):
        super().__init__()
        self.unitary = unitary

    def forward(self, x):
        return self.unitary(x, transposed=True)

    def __repr__(self):
        return f"UnitaryTransposed({self.unitary.__repr__()})"
    
class QModule(nn.Module):
    def __init__(self, J=False):
        super(QModule, self).__init__()
        self.J = J
    
    def forward(self, x, full=False):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % 2 == 0, "Number of channels must be even"
        
        half_channels = num_channels // 2
        first_half = x[:, :half_channels, :, :]
        second_half = x[:, half_channels:, :, :]
        if self.J:
            second_half = torch.flip(second_half, dims=[1])
        
        new_first_half  = (first_half + second_half) / np.sqrt(2)
        new_second_half = (first_half - second_half) / np.sqrt(2)
        
        if full:
            return new_first_half, new_second_half 
        return torch.cat((new_first_half, new_second_half), dim=1)
    
    def __repr__(self):
        return f"QModule()"

class SymAntiSymUnitary(nn.Module):
    def __init__(self, num_channels, projection_dim=None, p=1, J=False, reduced=False):
        super(SymAntiSymUnitary, self).__init__()
        if projection_dim is None:
            projection_dim = num_channels
        
        self.qmodule   = QModule(J=J)
        self.unitary_1 = UnitaryMatrix(num_channels=num_channels // 2, out_channels=projection_dim // 2, p=p)
        if reduced:
            self.unitary_1 = nn.Identity()
        self.unitary_2 = UnitaryMatrix(num_channels=num_channels // 2, out_channels=projection_dim // 2, p=p)
    
    def forward(self, x, full=False):
        first_half, second_half = self.qmodule(x, full=True)

        first_half = self.unitary_1(first_half)
        second_half = self.unitary_2(second_half)
        
        if full:
            return first_half, second_half
        return torch.cat((first_half, second_half), dim=1)
    
    def __repr__(self):
        return (f"SymAntiSymUnitary(num_channels={self.qmodule.num_channels}, "
                f"projection_dim={self.unitary_1.out_channels}, p={self.unitary_1.p})")


class UnitPatchModule(nn.Module):
    def __init__(self, num_channels, kernel_size, projection_dim, k_patch=None, output_dim=None, reduction=None):
        super(UnitPatchModule, self).__init__()
        if output_dim is None:
            output_dim = num_channels
        if k_patch is None:
            k_patch = num_channels
        
        # Initialize the unitary matrices and patch descriptor
        self.unitary_mixer      = UnitaryMatrix(num_channels, projection_dim)
        self.shift_operator     = shifts.SymmetricShift(num_channels=projection_dim, kernel_size=kernel_size, symmetric_reduction=reduction)
        self.patch              = shifts.TailPatch(projection_dim, k=k_patch, shift_operator=self.shift_operator)
        self.unitary_reduction  = UnitaryMatrix(self.patch.out_channels, output_dim)
        
    def forward(self, x):
        x = self.unitary_mixer(x)
        x = self.patch(x)
        x = self.unitary_reduction(x)
        return x
    
class SVDModule(nn.Module):
    def __init__(self, num_channels, kernel_size, projection_dim, k_patch=None, output_dim=None):
        super(SVDModule, self).__init__()
        if output_dim is None:
            output_dim = num_channels
        if k_patch is None:
            k_patch = num_channels
        
        # Initialize the unitary matrices and patch descriptor
        self.unitary_mixer      = UnitaryMatrix(num_channels, projection_dim)
        self.shift_operator     = shifts.GeneralizedShift(shifts.SymmetricShift._generate_shifts(kernel_size))
        self.patch              = shifts.TailPatch(projection_dim, k=kernel_size*kernel_size, shift_operator=self.shift_operator)
        self.unitary_reduction  = UnitaryMatrix(self.patch.out_channels, output_dim)
        
    def forward(self, x):
        x = self.unitary_mixer(x)
        x = self.patch(x)
        x = self.unitary_reduction(x)
        return x
    
class SVDBCOPModule(nn.Module):
    def __init__(self, num_channels, projection_dim, shift=None, kernel_size=None, output_dim=None, init_identity=True, kernel_mode='simple-neg'):
        super(SVDBCOPModule, self).__init__()
        if output_dim is None:
            output_dim = num_channels
        assert shift is not None or kernel_size is not None, "either kernel_size or shifts should be specified"
        
        self.modules = []
        self.num_channels = num_channels
        self.shift = shift
        self.kernel_size = kernel_size
        self.projection_dim = projection_dim
        self.output_dim = output_dim
        
        if self.kernel_size:
            if kernel_mode == 'simple-neg':
                self.shifts = shifts._kernel_shifts(kernel_size, use_negatives=True)[:2 * (kernel_size - 1)]
            elif kernel_mode == 'simple-pos':
                self.shifts = shifts._kernel_shifts(kernel_size, use_negatives=False)       
            elif kernel_mode == 'half-simple':
                self.shifts = shifts._kernel_shifts(kernel_size, use_negatives=False)
                self.shifts.extend([(-shift[0], -shift[1]) for shift in self.shifts[::-1]])
            else:
                self.shifts = shifts.SymmetricShift._generate_shifts(kernel_size)
        else:
            self.shifts = shift
            
            
        for idx, shift in enumerate(self.shifts):
            input_dim  = num_channels if idx == 0 else projection_dim
            self.modules.extend(self._create_module(shift, input_dim, projection_dim))
        self.modules.extend([UnitaryMatrix(projection_dim, output_dim)])
        
        # To make it initially identitiy
        if init_identity:
            for idx in range(0, len(self.modules) // 2, 2):
                initial = self.modules[idx].orthogonal_matrices[-1].data
                self.modules[len(self.modules) - 1 - idx].orthogonal_matrices[-1].data = initial.t()
            if len(self.shifts) % 2 == 0:
                self.modules[len(self.modules) // 2].orthogonal_matrices[-1].data = torch.eye(self.modules[len(self.modules) // 2].orthogonal_matrices[-1].data.shape[0])
                
        self.model = nn.Sequential(*self.modules)
        
        print(self.model)
                
    def forward(self, x):
        x = self.model(x)
        return x

    
    def _create_module(self, shift, input_dim, output_dim):
        return nn.Sequential(
            UnitaryMatrix(input_dim, output_dim),
            shifts.HalfShift(shift),
        )
        
    def __repr__(self):
        return (f"SVDBCOPModule(num_channels={self.num_channels}, projection_dim={self.projection_dim}, "
                f"shift={self.shift}, kernel_size={self.kernel_size}, output_dim={self.output_dim})")


class SymAntiSymModule(nn.Module):
    def __init__(self, num_channels, shift=None, kernel_size=None, kernel_mode='simple-neg', reduced=False):
        super(SymAntiSymModule, self).__init__()
        if kernel_size:
            if kernel_mode == 'simple-neg':
                self.shifts = shifts._kernel_shifts(kernel_size, use_negatives=True)                
            elif kernel_mode == 'simple-pos':
                self.shifts = shifts._kernel_shifts(kernel_size, use_negatives=False)                
            else:
                self.shifts = shifts.SymmetricShift._generate_shifts(kernel_size)
        else:
            self.shifts = [shift] if shift is not None else [(0, 0)]
        
        self.reduced = reduced
        self.num_channels = num_channels
        
        self.modules = []
        for idx, shift in enumerate(self.shifts):
            shift_type = 0 if idx == 0 else 2 if idx == len(self.shifts) - 1 else 1
            self.modules.append(self._create_module(num_channels, shift, shift_type))
        
        self.model = nn.Sequential(*self.modules)
    
    def forward(self, x):
        return self.model(x)
    
    def _create_module(self, num_channels, shift, shift_type):
        if shift_type == 0:
            return nn.Sequential(
                SymAntiSymUnitary(num_channels, J=True, reduced=False),
                QModule(),
                shifts.HalfShift(shift)
            )
        elif shift_type == 1:
            return nn.Sequential(
                SymAntiSymUnitary(num_channels, reduced=self.reduced),
                QModule(),
                shifts.HalfShift(shift)
            )
        elif shift_type == 2:
            return SymAntiSymUnitary(num_channels, reduced=self.reduced)
        else:
            raise KeyError("shift_type should be 0, 1 or 2")
    
    def __repr__(self):
        return (f"SymAntiSymModule(num_channels={self.num_channels}, "
                f"shift={self.shifts})")

        
# Example usage
if __name__ == "__main__":
    input_tensor = torch.randn(10, 4, 32, 32)  # Example input tensor with batch size 1, 3 channels, and 32x32 size
    model = SymAntiSymModule(4, kernel_size=3, kernel_mode='simple-pos')
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be [10, 16, 32, 32] assuming in_channels=3, out_channels=8, and p=2
    print((input_tensor ** 2).sum())
    print((output_tensor ** 2).sum())
    
    input_tensor = torch.randn(10, 50, 32, 32)  # Example input tensor with batch size 1, 3 channels, and 32x32 size
    model = SVDBCOPModule(50, 64, kernel_size=7)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be [10, 16, 32, 32] assuming in_channels=3, out_channels=8, and p=2
    print((input_tensor ** 2).sum())
    print((output_tensor ** 2).sum())
    print(((input_tensor - output_tensor) ** 2).sum())