import torch
import torch.nn as nn
from layers.BCOP.bcop import BCOP
from linearspline import LinearSpline


class ParsevalCNN(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(BCOP(1, nb_channels, kernel_size, bias=bias))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth-2):
            self.network.append(BCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(BCOP(nb_channels, 1, kernel_size, bias=bias))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)

from layers import shifts
from layers.UnitaryMatrices.unitary import UnitaryMatrix

class ParsevalCNN2(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth-2):
            self.network.append(UnitaryMatrix(nb_channels, nb_channels))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, 1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
from layers.UnitaryMatrices.unitary import UnitPatchModule
    
class ParsevalCNN3(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        k_patch = network_parameters['k_patch']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth-2):
            self.network.append(UnitPatchModule(nb_channels, kernel_size=kernel_size, projection_dim=nb_channels, k_patch=k_patch))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, 1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
class ParsevalCNN4(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        k_patch = network_parameters['k_patch']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth-2):
            self.network.append(UnitPatchModule(nb_channels, kernel_size=kernel_size, projection_dim=nb_channels, k_patch=k_patch, reduction="T1"))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, 1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
from layers.pooling import NormPatchModule
    
class ParsevalCNN5(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        k_patch = network_parameters['k_patch']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth-2):
            self.network.append(NormPatchModule(nb_channels, kernel_size=kernel_size, projection_dim=nb_channels, k_patch=k_patch, reduction=None))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, 1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
from layers.UnitaryMatrices.unitary import SVDModule
    
class ParsevalCNN6(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        k_patch = network_parameters['k_patch']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth-2):
            self.network.append(SVDModule(nb_channels, kernel_size=kernel_size, projection_dim=nb_channels, k_patch=k_patch))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, 1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
from layers.UnitaryMatrices.unitary import SVDBCOPModule
    
class ParsevalCNN7(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(SVDBCOPModule(num_channels=nb_channels, projection_dim=nb_channels, output_dim=nb_channels, kernel_size=kernel_size, kernel_mode='simple-neg', init_identity=False))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for _ in range(depth - 2):
            self.network.append(SVDBCOPModule(num_channels=nb_channels, projection_dim=nb_channels, output_dim=nb_channels, kernel_size=kernel_size, kernel_mode='simple-neg', init_identity=False))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(SVDBCOPModule(num_channels=nb_channels, projection_dim=nb_channels, output_dim=nb_channels, kernel_size=kernel_size, kernel_mode='simple-neg', init_identity=False))
        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        
        self.network[-1].orthogonal_matrices[-1].data = self.network[0].orthogonal_matrices[-1].data.t()
        # self.network[0].orthogonal_matrices[-1].data = torch.Tensor([[1], [0]])
        # self.network[-1].orthogonal_matrices[-1].data = self.network[0].orthogonal_matrices[-1].data.t()
        # self.network[0].orthogonal_matrices[-1].requires_grad = False
        # self.network[-1].orthogonal_matrices[-1].requires_grad = False
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        y = self.network(x)
        # print((x ** 2).sum().item(), (y ** 2).sum().item(), ((x - y) ** 2).sum().item(), (x ** 2).sum().item() - (y ** 2).sum().item())
        return y
    
class ParsevalCNN8(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(SVDBCOPModule(num_channels=1, projection_dim=nb_channels, output_dim=nb_channels, kernel_size=kernel_size, kernel_mode='simple-neg', init_identity=False))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for _ in range(depth - 2):
            self.network.append(SVDBCOPModule(num_channels=nb_channels, projection_dim=nb_channels, output_dim=nb_channels, kernel_size=kernel_size, kernel_mode='simple-neg', init_identity=False))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(SVDBCOPModule(num_channels=nb_channels, projection_dim=nb_channels, output_dim=1, kernel_size=kernel_size, kernel_mode='simple-neg', init_identity=False))
        # self.network[-1].orthogonal_matrices[-1].data = self.network[0].orthogonal_matrices[-1].data.t()
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        y = self.network(x)
        # print((x ** 2).sum().item(), (y ** 2).sum().item(), ((x - y) ** 2).sum().item())
        return y
    
from layers.UnitaryMatrices.unitary import SymAntiSymModule

class ParsevalCNN9(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for _ in range(depth):
            self.network.append(SymAntiSymModule(num_channels=nb_channels, kernel_size=kernel_size, kernel_mode='simple-neg'))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network[-1].orthogonal_matrices[-1].data = self.network[0].orthogonal_matrices[-1].data.t()
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        y = self.network(x)
        # print((x ** 2).sum().item(), (y ** 2).sum().item(), ((x - y) ** 2).sum().item(), (x ** 2).sum().item() - (y ** 2).sum().item())
        return y
    
from layers.BCOP.invertible_downsampling import psi
    
class ParsevalCNN10(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.upsampler, self.downsampler = psi(8, mode="up"), psi(8, mode="down")
        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = 2 * network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=2))
        self.network.append(self.upsampler)

        for _ in range(depth):
            self.network.append(SVDBCOPModule(num_channels=nb_channels, projection_dim=nb_channels, output_dim=nb_channels, kernel_size=kernel_size))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))
        
        self.network.append(self.downsampler)
        self.network.append(UnitaryMatrix(2, out_channels=1))
        
        self.network[0].orthogonal_matrices[-1].data = torch.Tensor([[1], [0]])
        self.network[-1].orthogonal_matrices[-1].data = self.network[0].orthogonal_matrices[-1].data.t()
        self.network[0].orthogonal_matrices[-1].requires_grad = False
        self.network[-1].orthogonal_matrices[-1].requires_grad = False
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        """ """
        y = self.network(x)
        # print((x ** 2).sum().item(), (y ** 2).sum().item(), ((x - y) ** 2).sum().item(), (x ** 2).sum().item() - (y ** 2).sum().item())
        return y
    
class ParsevalCNN11(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.sampler = psi(2)
        self.networks = [nn.ModuleList() for _ in range(5)]
        channels = [4, 16, 64, 16, 4]
        
        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.networks[0].append(UnitaryMatrix(1, out_channels=4))

        for i in range(5):
            for _ in range(4):
                self.networks[i].append(BCOP(channels[i], channels[i], kernel_size, bias=network_parameters['bias']))
                self.networks[i].append(LinearSpline(channels[i], spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))
            self.networks[i] = nn.Sequential(*self.networks[i])
            
        # self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.networks[-1].append(UnitaryMatrix(4, out_channels=1))
        self.networks[0][0].orthogonal_matrices[-1].data = torch.Tensor([[1], [0], [0], [0]])
        self.networks[-1][-1].orthogonal_matrices[-1].data = self.networks[0][0].orthogonal_matrices[-1].data.t()
        self.networks[0][0].orthogonal_matrices[-1].requires_grad = False
        self.networks[-1][-1].orthogonal_matrices[-1].requires_grad = False
        self.networks = nn.Sequential(*self.networks)

    def forward(self, x):
        """ """
        # y = self.sampler._forward(x)
        y = self.networks[0](x)
        y = self.sampler._forward(y)
        y = self.networks[1](y)
        y = self.sampler._forward(y)
        y = self.networks[2](y)
        y = self.sampler._inverse(y)
        y = self.networks[3](y)
        y = self.sampler._inverse(y)
        y = self.networks[4](y)
        # y = self.sampler._inverse(y)
        # print((x ** 2).sum().item(), (y ** 2).sum().item(), ((x - y) ** 2).sum().item(), (x ** 2).sum().item() - (y ** 2).sum().item())
        return y
    

class ParsevalCNN12(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()
        
        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth):
            self.network.append(BCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
from layers.BCOP.symbcop import SymBCOP    

class ParsevalCNN13(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()
        
        # self.upsampler, self.downsampler = psi(8, mode="up"), psi(8, mode="down")

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        # self.network.append(self.upsampler)
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))
    
        for _ in range(depth):
            self.network.append(SymBCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        # self.network.append(self.downsampler)
        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        # self.network[0].orthogonal_matrices[-1].data = torch.Tensor([[1], [0]])
        # self.network[-1].orthogonal_matrices[-1].data = self.network[0].orthogonal_matrices[-1].data.t()
        # self.network[0].orthogonal_matrices[-1].requires_grad = False
        # self.network[-1].orthogonal_matrices[-1].requires_grad = False
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
from layers.UnitaryMatrices.svd import SVD  

class ParsevalCNN14(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))
    
        for i in range(depth):
            self.network.append(SVD(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        # self.network.append(self.downsampler)
        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
from layers.UnitaryMatrices.symsvd import SymSVD  

class ParsevalCNN15(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()
        
        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))
    
        for i in range(depth):
            self.network.append(SymSVD(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
class ParsevalCNN16(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()
        
        self.upsampler, self.downsampler = psi(8, mode="up"), psi(8, mode="down")

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        # self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(self.upsampler)
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))
    
        for i in range(depth):
            self.network.append(SVD(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(self.downsampler)
        # self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        # self.network[0].orthogonal_matrices[-1].data = torch.Tensor([[1], [0]])
        # self.network[-1].orthogonal_matrices[-1].data = self.network[0].orthogonal_matrices[-1].data.t()
        # self.network[0].orthogonal_matrices[-1].requires_grad = False
        # self.network[-1].orthogonal_matrices[-1].requires_grad = False
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
class ParsevalCNN17(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()
        
        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']
        dilations = [1, 1, 2, 4, 8, 16, 1, 1]

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth):
            self.network.append(BCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias'], dilation=dilations[i]))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
    
from layers.BCOP.autobcop import AutoBCOP, upsample_with_zeros
from torch.nn.functional import avg_pool2d

class ParsevalCNN18(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()
        
        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.total_loss = torch.tensor(0.0)
        
        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth):
            self.network.append(AutoBCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias'], auto_dir=False, up_down_sample=False, entropic_loss_p=network_parameters['p'], natural_grad=network_parameters['natural']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        x = upsample_with_zeros(x, scale_factor=2)
        y = self.network(x)
        y = avg_pool2d(y, kernel_size=2, divisor_override=1)
        return y
    
    def entropic_loss(self):
        loss = 0.0
        for module in self.network:
            if type(module) is AutoBCOP:
                loss += module.entropic_loss
        return loss 
    
class ParsevalCNN19(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, activation_params):
        
        super().__init__()

        self.network = nn.ModuleList()
        
        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        self.total_loss = torch.tensor(0.0)
        
        self.network.append(UnitaryMatrix(1, out_channels=nb_channels))
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for i in range(depth):
            self.network.append(AutoBCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias'], auto_dir=network_parameters['auto_dir'], up_down_sample=True, entropic_loss_p=network_parameters['p'], natural_grad=network_parameters['natural'], optim_rank=network_parameters['optim_rank']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)
    
    def entropic_loss(self):
        loss = 0.0
        for module in self.network:
            if type(module) is AutoBCOP:
                loss += module.entropic_loss
        return loss 
    
    def direction_params(self):
        params = []
        loss = 0.0
        for module in self.network:
            if type(module) is AutoBCOP:
                if module.auto_dir:
                    params.append(module.direction)
                if module.optim_rank:
                    params.append(module.mask)
        return params
    
from layers.BCOP.mirror_bcop import MirrorBCOP
from layers.UnitaryMatrices.unitary import UnitaryTransposed

class NestedSequential(nn.Sequential):        
    def forward_half(self, x, apply_first=True):
        y = self[0](x) if apply_first else x
        y = self[1].forward_half(y, apply_first=apply_first)
        return y
    
    def forward_middle(self, x):
        return self[1].forward_middle(x)

    
class ParsevalCNN20(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary = UnitaryMatrix(1, out_channels=nb_channels)
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            return NestedSequential(
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1),
                MirrorBCOP(
                    nb_channels, nb_channels, kernel_size, bias=bias, 
                    middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range)
                ),
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    

    
class ParsevalCNN21(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        dilations = [1, 1, 2, 4]

        unitary = UnitaryMatrix(1, out_channels=nb_channels)
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range, dilations)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range, dilations):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            dilation = dilations.pop(0)
            return NestedSequential(
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1),
                MirrorBCOP(
                    nb_channels, nb_channels, kernel_size, bias=bias, dilation=dilation,
                    middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range, dilations=dilations)
                ),
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
    
from layers.BCOP.mirror_bcop import SymMirrorBCOP
    
class ParsevalCNN22(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary = UnitaryMatrix(1, out_channels=nb_channels)
        unitary.orthogonal_matrices[0] = nn.Parameter(
            torch.zeros(nb_channels, 1), requires_grad=False
        )
        unitary.orthogonal_matrices[0].data[0, 0] = 1
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            return NestedSequential(
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1),
                SymMirrorBCOP(
                    nb_channels, nb_channels, kernel_size, bias=bias, 
                    middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range)
                ),
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
class ParsevalCNN23(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        # dilations = [1, 1, 2, 4, 1, 1]
        dilations = [1, 1, 2, 4, 1, 1, 1, 1]

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary = UnitaryMatrix(1, out_channels=nb_channels)
        unitary.orthogonal_matrices[0] = nn.Parameter(
            torch.zeros(nb_channels, 1), requires_grad=False
        )
        unitary.orthogonal_matrices[0].data[0, 0] = 1
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range, dilations)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range, dilations):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            dilation = dilations.pop(0)
            return NestedSequential(
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1),
                SymMirrorBCOP(
                    nb_channels, nb_channels, kernel_size, bias=bias, dilation=dilation,
                    middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range, dilations)
                ),
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
        
class ParsevalCNN24(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary = UnitaryMatrix(1, out_channels=nb_channels)
        unitary.orthogonal_matrices[0] = nn.Parameter(
            torch.zeros(nb_channels, 1), requires_grad=False
        )
        unitary.orthogonal_matrices[0].data[0, 0] = 1
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            return SymMirrorBCOP(
                nb_channels, nb_channels, kernel_size, bias=bias, 
                middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
class ParsevalCNN25(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary = UnitaryMatrix(1, out_channels=nb_channels)
        unitary.orthogonal_matrices[0] = nn.Parameter(
            torch.zeros(nb_channels, 1), requires_grad=False
        )
        unitary.orthogonal_matrices[0].data[0, 0] = 1
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            return NestedSequential(
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1),
                MirrorBCOP(
                    nb_channels, nb_channels, kernel_size, bias=bias, 
                    middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range)
                ),
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
class ParsevalCNN26(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary = UnitaryMatrix(1, out_channels=nb_channels // 2, p=2)
        for p in [0, 1]:
            unitary.orthogonal_matrices[p] = nn.Parameter(
                torch.zeros(nb_channels // 2, 1), requires_grad=False
            )
            unitary.orthogonal_matrices[p].data[0, 0] = 1
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            return NestedSequential(
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1),
                SymMirrorBCOP(
                    nb_channels, nb_channels, kernel_size, bias=bias, 
                    middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range)
                ),
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
class ParsevalCNN27(nn.Module):
    """Parseval CNN with nested structure like an onion"""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary = UnitaryMatrix(1, out_channels=nb_channels)
        unitary.orthogonal_matrices[0] = nn.Parameter(
            torch.zeros(nb_channels, 1), requires_grad=False
        )
        unitary.orthogonal_matrices[0].data[nb_channels // 2, 0] = 1
        unitary_transposed = UnitaryTransposed(unitary)        
        layers = self._create_nested_structure(depth, nb_channels, kernel_size, bias, spline_size, spline_range)
        
        self.network = NestedSequential(
            unitary, 
            layers, 
            unitary_transposed
        )

    def _create_nested_structure(self, depth, nb_channels, kernel_size, bias, spline_size, spline_range):
        if depth == 0:
            return LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
        else:
            return NestedSequential(
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1),
                SymMirrorBCOP(
                    nb_channels, nb_channels, kernel_size, bias=bias, 
                    middle_network=self._create_nested_structure(depth-1, nb_channels, kernel_size, bias, spline_size, spline_range)
                ),
                LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1)
            )

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)