"""CNN architecture for BEV grid classification (M3).

A small purpose-built network rather than a literature-standard backbone
(ResNet-18 etc.): the train split has ~1140 tiles, far fewer than those
architectures are designed for, and a small network keeps every layer's
role visible end to end instead of hidden inside a library backbone. This
module only defines and initialises the architecture -- training (loss,
optimizer, epochs) is a later step.
"""

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> 2x2 max-pool.

    `ceil_mode=True` on the pool rounds the output size up instead of down,
    so a spatial dimension of 1 stays 1 instead of collapsing to 0. BEV
    grids can be as small as ~7x16 cells for the smallest tiles (see
    src/bev.py); three pools in a row with the default floor rounding would
    reach 0 before the network's last conv layer for a grid that small.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, ceil_mode=True),
    )


class LaneCNN(nn.Module):
    """BEV grid -> 6-way road-scene classification.

    Accepts `(N, in_channels, H, W)` input of any H, W: `src/bev.py` builds
    each grid at the tile's own extent rather than a fixed crop, so the
    network reduces spatial size with global adaptive average pooling right
    before the classifier instead of a flatten + fixed-size linear layer,
    which would only accept one specific H, W.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
