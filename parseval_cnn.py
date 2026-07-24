"""Parseval (orthogonal, 1-Lipschitz) CNN denoisers.

This module keeps only the architectures that matter, under descriptive names.
The full set of experimental variants (ParsevalCNN2 .. ParsevalCNN27) lives in
``parseval_cnn_experiments.py``.

Two independent ideas are combined here:

    * "symmetric filters" -- the orthogonal convolution is built to be
      symmetric (SymBCOP, and the U0/U1 construction inside SymMirrorBCOP).
    * "mirror" -- a nested "onion" architecture in which the second half of the
      network reuses the mirrored (flipped) filters of the first half
      (MirrorBCOP / SymMirrorBCOP).

The four kept models form a clean 2x2 ablation over those two ideas:

    class name                    symmetric?  mirror?   was      conv block
    --------------------------    ----------  -------   ------   -------------
    BaselineParsevalCNN               no        no      CNN      BCOP
    SymmetricParsevalCNN              yes        no      CNN13    SymBCOP
    MirrorParsevalCNN                 no        yes      CNN20    MirrorBCOP
    SymmetricMirrorParsevalCNN        yes        yes     CNN22    SymMirrorBCOP

Input/output embedding
----------------------
The unitary-based models all share the *same* fixed input/output embedding
(``_fixed_channel_lift``): a non-trainable one-hot lift from the single input
channel into ``nb_channels``, with its transpose projecting back. Theoretically
the choice of this orthogonal embedding is irrelevant (the network absorbs it),
so fixing it identically across models keeps the ablation to a single variable:
the core convolution block.
"""

import torch
import torch.nn as nn

from linearspline import LinearSpline
from layers.BCOP.bcop import BCOP
from layers.BCOP.symbcop import SymBCOP
from layers.BCOP.mirror_bcop import MirrorBCOP, SymMirrorBCOP
from layers.UnitaryMatrices.unitary import UnitaryMatrix, UnitaryTransposed


def _fixed_channel_lift(nb_channels):
    """Shared input/output orthogonal embedding.

    Returns ``(unitary, unitary_transposed)`` where ``unitary`` is a fixed
    (non-trainable) one-hot lift from the single input channel into
    ``nb_channels`` (channel 0), and ``unitary_transposed`` projects back to one
    channel. The same fixed embedding is used by every unitary-based model so
    the only thing that varies across the ablation is the convolution block.
    """
    unitary = UnitaryMatrix(1, out_channels=nb_channels)
    unitary.orthogonal_matrices[0] = nn.Parameter(
        torch.zeros(nb_channels, 1), requires_grad=False
    )
    unitary.orthogonal_matrices[0].data[0, 0] = 1
    unitary_transposed = UnitaryTransposed(unitary)
    return unitary, unitary_transposed


class NestedSequential(nn.Sequential):
    """Sequential container that also exposes the mirror-network half passes."""
    def forward_half(self, x, apply_first=True):
        y = self[0](x) if apply_first else x
        y = self[1].forward_half(y, apply_first=apply_first)
        return y

    def forward_middle(self, x):
        return self[1].forward_middle(x)


class BaselineParsevalCNN(nn.Module):
    """Baseline: a plain stack of BCOP orthogonal convolutions (was ParsevalCNN)."""
    def __init__(self, network_parameters, activation_params):

        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary, unitary_transposed = _fixed_channel_lift(nb_channels)

        self.network.append(unitary)
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for _ in range(depth):
            self.network.append(BCOP(nb_channels, nb_channels, kernel_size, bias=bias))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(unitary_transposed)
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)


class SymmetricParsevalCNN(nn.Module):
    """Symmetry only: a stacked architecture of symmetric orthogonal convs
    (SymBCOP). Was ParsevalCNN13."""
    def __init__(self, network_parameters, activation_params):

        super().__init__()

        self.network = nn.ModuleList()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary, unitary_transposed = _fixed_channel_lift(nb_channels)

        self.network.append(unitary)
        self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        for _ in range(depth):
            self.network.append(SymBCOP(nb_channels, nb_channels, kernel_size, bias=network_parameters['bias']))
            self.network.append(LinearSpline(nb_channels, spline_size, -spline_range, spline_range, 'identity', slope_min=-1, slope_max=1))

        self.network.append(unitary_transposed)
        self.network = nn.Sequential(*self.network)


    def forward(self, x):
        """ """
        return self.network(x)


class MirrorParsevalCNN(nn.Module):
    """Mirror only: the nested ("onion") architecture whose second half reuses
    the mirrored filters of the first half, without symmetric filters
    (MirrorBCOP). Was ParsevalCNN20."""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary, unitary_transposed = _fixed_channel_lift(nb_channels)
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


class SymmetricMirrorParsevalCNN(nn.Module):
    """Symmetric mirror: the nested ("onion") architecture using symmetric
    mirrored filters (SymMirrorBCOP). Was ParsevalCNN22."""
    def __init__(self, network_parameters, activation_params):
        super().__init__()

        depth = network_parameters['depth']
        nb_channels = network_parameters['nb_channels']
        kernel_size = network_parameters['kernel_size']
        bias = network_parameters['bias']

        spline_size = activation_params['spline_size']
        spline_range = activation_params['spline_range']

        unitary, unitary_transposed = _fixed_channel_lift(nb_channels)
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


# Registry mapping a short config name -> model class, plus a human-readable
# label for tables/plots. Used by trainer.py (config['net_params']['model'])
# and by experiments.py.
MODELS = {
    'baseline': BaselineParsevalCNN,
    'symmetric': SymmetricParsevalCNN,
    'mirror': MirrorParsevalCNN,
    'symmetric_mirror': SymmetricMirrorParsevalCNN,
}

MODEL_LABELS = {
    'baseline': 'Baseline (BCOP)',
    'symmetric': 'Symmetric',
    'mirror': 'Mirror',
    'symmetric_mirror': 'Symmetric Mirror',
}
