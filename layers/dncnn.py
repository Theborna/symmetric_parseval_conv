"""DnCNN: a plain (unconstrained) residual denoiser.

Unlike everything in ``parseval_cnn.py``, this network has no architectural
constraint on its Jacobian -- it is not 1-Lipschitz by construction. That
makes it useful in two unrelated places in this repo, which is why it lives
here rather than inside either of them:

* ``pnp.py`` uses it as the *control* that demonstrates Plug-and-Play has no
  convergence guarantee without a nonexpansive denoiser (see
  ``colab_pnp.ipynb``).
* ``parseval_cnn.py`` registers it under ``MODELS['dncnn']`` as a plain
  accuracy baseline for the denoising table (see
  ``experiment_configs/dncnn.json`` and ``colab_dncnn_baseline.ipynb``): how
  much denoising accuracy is left on the table by enforcing 1-Lipschitzness at
  a comparable depth and width.
"""

import torch.nn as nn


class DnCNN(nn.Module):
    """Standard DnCNN: Conv-ReLU, then (depth-2) x [Conv-BN-ReLU], then Conv,
    with a residual (noise-predicting) output. ``depth`` counts convolutions,
    matching how "depth" is counted for the Parseval models in this repo."""

    def __init__(self, depth=7, channels=64, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        layers = [nn.Conv2d(1, channels, kernel_size, padding=pad), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 1, kernel_size, padding=pad)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)  # residual (noise-predicting) parameterisation
