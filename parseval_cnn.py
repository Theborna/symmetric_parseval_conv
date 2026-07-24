"""Parseval (orthogonal, 1-Lipschitz) CNN denoisers.

This module keeps only the three architectures that matter. The full set of
experimental variants (ParsevalCNN2 .. ParsevalCNN27) lives in
``parseval_cnn_experiments.py``.

The three kept here form a clean ablation:

    ParsevalCNN    -- the original baseline: a plain stack of BCOP orthogonal
                      convolutions with linear-spline activations.
    ParsevalCNN13  -- "symmetry only": same stacked design, but each layer is a
                      symmetric orthogonal convolution (SymBCOP).
    ParsevalCNN22  -- "symmetric mirror": the nested "onion" architecture where
                      each orthogonal layer's transpose is reused on the way out
                      (SymMirrorBCOP).
"""

import torch
import torch.nn as nn

from linearspline import LinearSpline
from layers.BCOP.bcop import BCOP
from layers.BCOP.symbcop import SymBCOP
from layers.BCOP.mirror_bcop import SymMirrorBCOP
from layers.UnitaryMatrices.unitary import UnitaryMatrix, UnitaryTransposed


class ParsevalCNN(nn.Module):
    """Original baseline: a plain stack of BCOP orthogonal convolutions."""
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


class ParsevalCNN13(nn.Module):
    """Symmetry only: a stacked architecture of symmetric orthogonal convs."""
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
            self.network.append(SymBCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(UnitaryMatrix(nb_channels, out_channels=1))
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)


class NestedSequential(nn.Sequential):
    def forward_half(self, x, apply_first=True):
        y = self[0](x) if apply_first else x
        y = self[1].forward_half(y, apply_first=apply_first)
        return y

    def forward_middle(self, x):
        return self[1].forward_middle(x)


class ParsevalCNN22(nn.Module):
    """Symmetric mirror: nested ("onion") Parseval CNN using SymMirrorBCOP."""
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
