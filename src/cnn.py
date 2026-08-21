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


NORM_GROUPS = 8  # divides every channel count below, and M4's wider variants

# (16, 32, 64, 128) is the M3 architecture: three pooling conv blocks feeding a
# final un-pooled conv. M4 searches over alternatives, e.g. (16, 32, 64) for a
# shallower net or (32, 64, 128, 256) for a wider one. Any value here must be
# divisible by NORM_GROUPS or nn.GroupNorm raises -- loudly, which is what we
# want rather than a silent reshape.
DEFAULT_CHANNELS = (16, 32, 64, 128)


def _norm(channels: int) -> nn.GroupNorm:
    """Per-sample normalisation over `NORM_GROUPS` groups of channels.

    `nn.BatchNorm2d` is the more usual choice, but it cannot be used here.
    BatchNorm normalises using statistics of the *batch*, which forces a
    choice this dataset has no good answer to:

    - Batching >1 tile requires zero-padding to a common H, W, because BEV
      grids vary in size (src/bev.py). That padding is not inert -- the norm
      layer's learned offset turns padded zeros into a non-zero constant that
      the next conv mixes into neighbouring real cells, so a tile's prediction
      depends on which tiles shared its batch. Measured: 136 of 213 val tiles
      changed prediction between batch size 16 and 1.
    - `batch_size=1` avoids padding, but then BatchNorm normalises each tile by
      its own statistics during training and switches to accumulated running
      averages at `eval()`. Measured on an 8-epoch model, the same weights gave
      0.662 val accuracy with per-sample statistics and 0.127 with the running
      averages -- they disagree, so the reported number means nothing.

    GroupNorm has neither problem: it normalises within each sample, across a
    group of channels, so it never consults the batch and behaves identically
    in `train()` and `eval()`. That is what it was introduced for (Wu & He,
    2018) -- small-batch training, where BatchNorm degrades.
    """
    return nn.GroupNorm(NORM_GROUPS, channels)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv -> GroupNorm -> ReLU -> 2x2 max-pool.

    `ceil_mode=True` on the pool rounds the output size up instead of down,
    so a spatial dimension of 1 stays 1 instead of collapsing to 0. BEV
    grids can be as small as ~7x16 cells for the smallest tiles (see
    src/bev.py); three pools in a row with the default floor rounding would
    reach 0 before the network's last conv layer for a grid that small.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        _norm(out_channels),
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

    Any H, W, but **one size per call**: mixing sizes in a batch would require
    zero-padding to a common H, W, and that padding leaks into real cells
    through the conv stack (see `_norm`). `src/train.py` trains one tile at a
    time; a batch of mismatched grids raises in PyTorch's default collate
    rather than being silently padded.

    `channels` sets the per-stage width and, through its length, the depth --
    see `DEFAULT_CHANNELS`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 6,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
    ):
        super().__init__()
        # Every entry but the last becomes a pooling conv block; the last is a
        # plain conv with no pool, because the global average pool right after
        # already collapses the spatial dimensions. `channels` is what M4's
        # depth/width search varies -- the default reproduces the M3 network
        # exactly, so M3's numbers stay reproducible.
        widths = (in_channels, *channels)
        blocks = [_conv_block(widths[i], widths[i + 1]) for i in range(len(channels) - 1)]
        self.features = nn.Sequential(
            *blocks,
            nn.Conv2d(widths[-2], widths[-1], kernel_size=3, padding=1),
            _norm(widths[-1]),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
