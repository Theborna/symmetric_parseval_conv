import torch
import torch.nn as nn
import numpy as np

class GeneralizedShift(nn.Module):
    def __init__(self, shifts):
        super(GeneralizedShift, self).__init__()
        self.shifts = shifts
        self.out_channels = 1
        
    def forward(self, x):
        _, num_channels, _, _ = x.size()
        assert len(self.shifts) == num_channels, "Number of shifts must match the number of channels"
        dims = tuple([i - 1 for i in range(len(x.shape))][-2:])

        # Apply shifts to each channel
        shifted_x = torch.zeros_like(x)
        for i in range(num_channels):
            shift = self.shifts[i]
            shifted_x[:, i] = torch.roll(x[:, i], shifts=shift, dims=dims)

        return shifted_x
    
    def __repr__(self):
        return f"GeneralizedShift(shifts={self.shifts})"

class HalfShift(nn.Module):
    def __init__(self, shift):
        super(HalfShift, self).__init__()
        self.shift = shift
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % 2 == 0, "Number of channels must be even"
        
        half_channels = num_channels // 2
        first_half = x[:, :half_channels, :, :]
        second_half = x[:, half_channels:, :, :]
        
        # Apply the shift to the first half
        shifted_first_half = torch.roll(first_half, shifts=self.shift, dims=(-2, -1))
        
        # Concatenate the shifted first half with the unchanged second half
        output = torch.cat((shifted_first_half, second_half), dim=1)
        
        return output
    
    def __repr__(self):
        return f"HalfShift(shift={self.shift})"
class PatchDescriptor(nn.Module):
    def __init__(self, num_channels, shifts):
        super(PatchDescriptor, self).__init__()
        self.num_channels = num_channels
        self.m = len(shifts) 
        self.shifts = shifts
        self.out_channels = self.m * self.num_channels
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        dims = tuple([i for i in range(len(x.shape))][-2:])
        # Apply shifts
        shifted_x = torch.zeros(batch_size, num_channels * self.m, height, width, device=x.device)
        for i in range(self.m):
            start_idx = i * num_channels
            end_idx = (i + 1) * num_channels
            shifts = self.shifts[i]
            shifted_x[:, start_idx:end_idx] = torch.roll(x, shifts=shifts, dims=dims)
        
        return shifted_x / torch.sqrt(torch.tensor(self.m, dtype=torch.float32))
    
    def __repr__(self):
        return (f"PatchDescriptor(num_channels={self.m}, shifts={self.shifts})")
    
class TailPatch(nn.Module):
    def __init__(self, num_channels, k, shift_operator):
        super(TailPatch, self).__init__()
        assert k <= num_channels, "k must be less than or equal to the number of channels"
        self.num_channels = num_channels
        self.k = k
        self.shift_operator = shift_operator
        self.out_channels = shift_operator.out_channels * k + num_channels - k

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels == self.num_channels, "Number of channels must match the initialized number of channels"
        
        # Split the input tensor into two parts: unchanged and to be processed by PatchDescriptor
        unchanged_part = x[:, :-self.k, :, :]
        tail_part = x[:, -self.k:, :, :]

        # Apply PatchDescriptor to the tail part
        processed_tail_part = self.shift_operator(tail_part)
        
        # Concatenate the unchanged part with the processed tail part
        output = torch.cat((unchanged_part, processed_tail_part), dim=1)
        
        return output

    def __repr__(self):
        return (f"TailPatch(num_channels={self.num_channels}, k={self.k}, "
                f"shift_operator={self.shift_operator})")
        
class SymmetricShift(PatchDescriptor):
    def __init__(self, num_channels, kernel_size, symmetric_reduction=None):
        shifts = SymmetricShift._generate_shifts(kernel_size)
        super(SymmetricShift, self).__init__(num_channels, shifts)
        self.kernel_size = kernel_size
        self.symmetric_reduction = symmetric_reduction
        self.symmetric_pairs = self.get_symmetric_pairs()
        self.out_channels = len(self.symmetric_pairs)
        self.m = len(shifts) 
        self.shifts = shifts
        
    @staticmethod
    def _generate_shifts(kernel_size):
        shifts = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                shifts.append((i - kernel_size // 2, j - kernel_size // 2))
        return shifts
    
    def _get_duals(self, shift, reduction):
        if reduction == "T1":
            return list(set([(-shift[0], -shift[1]), (-shift[0], shift[1]), (shift[0], -shift[1])]))
        elif reduction == "T2A":
            return [(-shift[0], -shift[1])]
        elif reduction == "T2B":
            return [(-shift[1], -shift[0])]
        elif reduction == "T3":
            return [(shift[1], shift[0])]
        elif reduction == "T4":
            return [(shift[1], shift[0]), (-shift[1], -shift[0])]
        else:
            return [shift]
    
    def get_symmetric_pairs(self):
        if not self.symmetric_reduction:
            return [[i] for i in range(len(self.shifts))]
        
        symmetric_pairs = []
        used = set()
        
        for idx, shift in enumerate(self.shifts):
            if idx in used:
                continue
                
            pos_shift = shift
            
            pos_idx = self.shifts.index(pos_shift)
            duals = self._get_duals(shift, self.symmetric_reduction)
            dual_idxs = [self.shifts.index(dual) for dual in duals]
            
            if pos_idx not in used:
                used.add(pos_idx)
                pair = set([pos_idx])
                for dual in dual_idxs:
                    used.add(dual)
                    pair.add(dual)
                symmetric_pairs.append(pair)
        
        return symmetric_pairs
    
    def _apply_symmetric_reduction(self, x):
        batch_size, num_channels, height, width = x.size()
        num_channels = num_channels // (self.kernel_size ** 2)
        reduced_shifted_x = torch.zeros(batch_size, len(self.symmetric_pairs) * num_channels, height, width, device=x.device)
        
        for idx, pair in enumerate(self.symmetric_pairs):
            for p in pair:
                reduced_shifted_x[:, idx * num_channels:(idx + 1) * num_channels, :, :] += x[:, p * num_channels:(p + 1) * num_channels, :, :]
            reduced_shifted_x[:, idx * num_channels:(idx + 1) * num_channels, :, :] /= torch.sqrt(torch.tensor(len(pair), dtype=torch.float32))
        
        return reduced_shifted_x

    def forward(self, x):
        shifted_x = super(SymmetricShift, self).forward(x)
        if self.symmetric_reduction:
            shifted_x = self._apply_symmetric_reduction(shifted_x)
        return shifted_x
    
    def __repr__(self):
        return (f"SymmetricShift(kernel_size={self.kernel_size}, symmetric_reduction={self.symmetric_reduction}, "
                f"symmetric_pairs={self.symmetric_pairs})")

def _kernel_shifts(kernel_size, use_negatives=False):
    shifts = []
    for _ in range(kernel_size - 1):
        shifts.append((1, 0))
        shifts.append((0, 1))
        if use_negatives:
            shifts.append((0, -1))
            shifts.append((-1, 0))
    return shifts

# Example usage
if __name__ == "__main__":
    shifts = [(1, 0), (0, 1), (-1, -1)]  # Example shifts for each channel
    model = GeneralizedShift(shifts=shifts)
    input_tensor = torch.randn(2, 3, 2, 2)  # Example input tensor with batch size 1, 3 channels, and 32x32 size
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be [1, 3, 32, 32]
    # print("============================================")
    # print(input_tensor)
    # print("============================================")
    # print(output_tensor)
    
    num_channels = 1
    shifts = [
        (1, 0),     # Shifts for the 1st copy
        (1, 1),     # Shifts for the 2nd copy
        (-1, 1),     # Shifts for the 3rd copy
        (-1, 0),     # Shifts for the 4th copy
    ]
    
    model = PatchDescriptor(num_channels=num_channels, shifts=shifts)
    input_tensor = torch.randn(1, num_channels, 32, 32)  # Example input tensor with batch size 1, 3 channels, and 32x32 size
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should be [1, 12, 32, 32] assuming num_channels=3 and num_copies=4
    # print("============================================")
    # print(input_tensor)
    # print("============================================")
    # print(output_tensor)
    
    # Example usage
    num_channels = 10
    k = 10
    shifts = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Example shifts

    # Create an instance of TailPatch
    tail_patch = TailPatch(num_channels, k, PatchDescriptor(num_channels, shifts))

    # Print the TailPatch instance
    print(tail_patch)

    # Example input tensor (batch_size, num_channels, height, width)
    x = torch.randn(2, num_channels, 32, 32)

    # Apply the TailPatch module
    output = tail_patch(x)

    # Print the output shape
    print("Output shape:", output.shape)
    
    # Example usage
    num_channels = 1
    kernel_size = 3

    # Create an instance of SymmetricShift
    symmetric_shift = SymmetricShift(num_channels, kernel_size, "T1")

    # Print the SymmetricShift instance
    print(symmetric_shift)

    # Example input tensor (batch_size, num_channels, height, width)
    x = torch.randn(1, num_channels, 3, 3)

    # Apply the SymmetricShift module
    output = symmetric_shift(x)

    # Print the output shape
    print("Output shape:", output.shape)
    print("============================================")
    print(x)
    print("============================================")
    print(output)
