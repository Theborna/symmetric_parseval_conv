import torch
import torch.nn.functional as F

# filters = torch.randn(8, 4, 3, 3)
# matrix = torch.randn(4, 4)
# inputs = torch.randn(1, 4, 5, 5)
# outputs1 = torch.matmul(matrix, inputs.view(1, 4, -1)).view(1, 4, 5, 5)
# outputs1 = F.conv2d(outputs1, filters, padding=1)
# print(outputs1.shape)

# filters_reshaped = filters.view(8, 4, -1)
# filters_reshaped = matrix @ filters_reshaped
# filters = filters_reshaped.view(8, 4, 3, 3)
# outputs2 = F.conv2d(inputs, filters, padding=1)
# print(outputs2.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.BCOP.autobcop import AutoBCOP
from layers.BCOP.bcop import BCOP

def test_AutoBCOP_vs_BCOP():
    # Define network parameters
    network_parameters = {
        'nb_channels': 16,
        'kernel_size': 3,
        'bias': False
    }
    
    # Initialize the AutoBCOP with auto_dir=False
    auto_bcop = AutoBCOP(
        network_parameters['nb_channels'], 
        network_parameters['nb_channels'], 
        network_parameters['kernel_size'], 
        bias=network_parameters['bias'], 
        auto_dir=False
    )

    # Initialize the BCOP with the same parameters
    bcop = BCOP(
        network_parameters['nb_channels'], 
        network_parameters['nb_channels'], 
        network_parameters['kernel_size'], 
        bias=network_parameters['bias']
    )
    
    # Set the same param_matrices in BCOP as in AutoBCOP
    with torch.no_grad():
        bcop.param_matrices.copy_(auto_bcop.param_matrices)
    
    # Generate a random input tensor
    input_tensor = torch.randn(1, network_parameters['nb_channels'], 32, 32)
    
    # Forward pass through both models
    output_auto_bcop = auto_bcop(input_tensor)
    output_bcop = bcop(input_tensor)
    
    # Check if the outputs are the same    
    assert torch.allclose(output_auto_bcop, output_bcop, atol=1e-3), \
        "Outputs from AutoBCOP and BCOP do not match!"
    
    print("Test passed! AutoBCOP and BCOP outputs match.")

# Run the test
test_AutoBCOP_vs_BCOP()
