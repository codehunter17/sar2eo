"""
models/discriminator.py — PatchGAN Discriminator
==================================================
Architecture: PatchGAN (Isola et al., Pix2Pix 2017)

Why PatchGAN?
--------------
A global discriminator outputs a single real/fake score for the
entire image. This means it only needs to fool the discriminator
once, globally, which is easy and doesn't force local realism.

PatchGAN instead outputs an N×N grid of real/fake scores, where
each score covers a 70×70 pixel patch of the input.  This forces
the generator to produce realistic textures at every local region —
exactly what matters for perceptual quality.

How it works:
  Input:  [SAR (1ch) | EO (3ch)]  concatenated → 4 channels
          (conditioning on SAR prevents the discriminator from
           ignoring it; it must judge "does this EO match THIS SAR?")
  Output: A 30×30 grid of scalars (for 256×256 input, 3 layers)
          Each scalar = probability that the corresponding 70×70
          patch of the image is real.

Architecture (n_layers=3):
  Conv(4→64, k4s2)   LeakyReLU
  Conv(64→128, k4s2) BN  LeakyReLU
  Conv(128→256, k4s2) BN LeakyReLU
  Conv(256→512, k4s1) BN LeakyReLU   (stride=1, padding for spatial size)
  Conv(512→1,  k4s1)                  (final logit map)
"""

import torch
import torch.nn as nn
from .generator import _get_norm


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Args:
        in_channels:   SAR channels + EO channels (1 + 3 = 4 by default).
        ndf:           Base number of discriminator filters.
        n_layers:      Number of conv layers (3 → 70×70 receptive field).
        norm:          Normalisation type: 'batch' | 'instance' | 'none'.
    """

    def __init__(
        self,
        in_channels: int = 4,   # SAR(1) + EO(3)
        ndf:         int = 64,
        n_layers:    int = 3,
        norm:        str = "batch",
    ):
        super().__init__()
        layers = []

        # First layer: no normalisation (standard practice)
        layers += [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate layers: stride-2 convolutions with norm
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)    # cap at 512 channels
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False),
                _get_norm(norm, nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # After n_layers, add one more stride-1 conv to widen receptive field
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=False),
            _get_norm(norm, nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final conv → 1 channel logit map (no sigmoid — BCEWithLogitsLoss handles it)
        layers += [
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        sar: torch.Tensor,
        eo:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sar: (B, 1, H, W) — SAR input (condition)
            eo:  (B, 3, H, W) — EO image (real or generated)
        Returns:
            (B, 1, Ph, Pw) — patch logit map
        """
        # Concatenate along channel dimension for conditional discriminator
        x = torch.cat([sar, eo], dim=1)   # (B, 4, H, W)
        return self.model(x)


def build_discriminator(cfg: dict) -> PatchGANDiscriminator:
    """Instantiate discriminator from config dict."""
    d = cfg["model"]["discriminator"]
    return PatchGANDiscriminator(
        in_channels = cfg["data"]["in_channels"] + cfg["data"]["out_channels"],
        ndf         = d["ndf"],
        n_layers    = d["n_layers"],
        norm        = d["norm"],
    )
