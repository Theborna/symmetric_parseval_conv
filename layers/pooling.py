import torch
import torch.nn as nn
from layers.UnitaryMatrices.unitary import UnitaryMatrix
from layers.shifts import TailPatch, SymmetricShift

class NormPooling(nn.Module):
    def __init__(self, norm_type=2):
        super(NormPooling, self).__init__()
        self.norm_type = norm_type

    def forward(self, x):
        norm_pooled = torch.norm(x, p=self.norm_type, dim=1, keepdim=False)
        return norm_pooled

class NormPatchModule(nn.Module):
    def __init__(self, num_channels, kernel_size, projection_dim, k_patch=None, output_dim=None, reduction=None):
        super(NormPatchModule, self).__init__()
        if k_patch is None:
            k_patch = num_channels
            
        self.output_dim     = num_channels
        self.k_patch        = k_patch
        self.unitary_mixer  = UnitaryMatrix(num_channels, projection_dim)
        self.shift_operator = SymmetricShift(num_channels=projection_dim, kernel_size=kernel_size, symmetric_reduction=reduction)
        self.patch          = TailPatch(projection_dim, k=k_patch, shift_operator=self.shift_operator)
        self.norm_pooling   = NormPooling(norm_type=2)
        
    def forward(self, x):
        x = self.unitary_mixer(x)
        x = self.patch(x)
        batch_size, num_channels, height, width = x.size()
        pooled_x = torch.zeros(batch_size, self.output_dim, height, width, device=x.device)
        
        i0 = self.output_dim - self.k_patch
        pooled_x[:, 0:i0, :, :] = x[:, 0:i0, :, :]
        step = self.shift_operator.out_channels
        for i in range(self.k_patch):
            pooled_x[:, i0 + i, :, :] = self.norm_pooling(x[:, i0 + i * step:i0 + (i + 1) * step, :, :])
        return pooled_x

    def __repr__(self):
        return (f"NormPatchModule(num_channels={self.unitary_mixer.num_channels}, kernel_size={self.shift_operator.kernel_size}, "
                f"projection_dim={self.unitary_mixer.out_channels}, k_patch={self.patch.k}, output_dim={self.unitary_mixer.num_channels}, "
                f"reduction={self.shift_operator.symmetric_reduction}")
        
        
if __name__ == "__main__":
    # Parameters
    num_channels = 7  # Original number of channels
    kernel_size = 3  # Kernel size for patch descriptor
    projection_dim = 7  # Dimension after unitary projection
    k_patch = 7  # Number of tail channels to apply patch descriptor on
    reduction = None  # Type of symmetric reduction

    # Create a NormPatchModule instance
    norm_patch_module = NormPatchModule(
        num_channels=num_channels,
        kernel_size=kernel_size,
        projection_dim=projection_dim,
        k_patch=k_patch,
        reduction=reduction,
    )
    
    # Print the module to check its structure
    print(norm_patch_module)
    
    # Create synthetic input data (batch_size, num_channels, height, width)
    batch_size = 2
    height = 3
    width = 3
    x = torch.randn(batch_size, num_channels, height, width)
    
    # Forward pass through the NormPatchModule
    output = norm_patch_module(x)
    
    # Print input and output shapes
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
    print(output)
    
    print((x ** 2).sum())
    print((output ** 2).sum())