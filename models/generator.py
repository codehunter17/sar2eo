"""
models/generator.py — U-Net Generator
======================================
Architecture: U-Net with skip connections (Ronneberger et al., 2015)
adapted for image-to-image translation (Isola et al., Pix2Pix 2017).

Why U-Net for SAR-to-EO?
--------------------------
A plain encoder-decoder compresses the input into a bottleneck and
then decodes it. The bottleneck forces the network to discard
high-frequency detail — problematic here because we WANT to preserve
structure (roads, coastlines, building edges) that IS detectable in SAR.

Skip connections copy feature maps from each encoder layer to the
matching decoder layer, letting the decoder reuse low-level spatial
features instead of regenerating them from the compressed bottleneck.

Channel accounting (ngf=64, num_downs=8, input=256×256):
──────────────────────────────────────────────────────────────────────
Layer        in_ch  out_ch  spatial
──────────────────────────────────────────────────────────────────────
enc1           1      64    256→128   (no norm on first layer)
enc2          64     128    128→64
enc3         128     256     64→32
enc4         256     512     32→16
enc5         512     512     16→8
enc6         512     512      8→4
enc7         512     512      4→2
bottleneck   512     512      2→1
──────────────────────────────────────────────────────────────────────
dec1  (skip←e7)  512+512=1024  →512    1→2    dropout
dec2  (skip←e6)  512+512=1024  →512    2→4    dropout
dec3  (skip←e5)  512+512=1024  →512    4→8    dropout
dec4  (skip←e4)  512+512=1024  →512    8→16
dec5  (skip←e3)  512+256= 768  →256   16→32
dec6  (skip←e2)  256+128= 384  →128   32→64
dec7  (skip←e1)  128+ 64= 192  → 64   64→128
final            64→3                128→256   Tanh
──────────────────────────────────────────────────────────────────────
Note: each UNetUp concatenates (prev_decoder, skip) BEFORE its
ConvTranspose, so in_ch = prev_decoder_channels + skip_channels.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Norm helper
# ──────────────────────────────────────────────────────────────

def _get_norm(norm: str, num_features: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_features)
    elif norm == "instance":
        return nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
    elif norm == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm}")


# ──────────────────────────────────────────────────────────────
# Encoder block  (H×W → H/2×W/2)
# ──────────────────────────────────────────────────────────────

class UNetDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", use_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm),
        ]
        if use_norm:
            layers.append(_get_norm(norm, out_ch))
        # LeakyReLU(0.2) is standard for GAN encoder blocks —
        # allows small gradients for negative activations.
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ──────────────────────────────────────────────────────────────
# Decoder block  (H×W → 2H×2W, with skip concatenation)
# ──────────────────────────────────────────────────────────────

class UNetUp(nn.Module):
    """
    in_ch must equal prev_decoder_channels + skip_channels
    (the concat happens here, BEFORE the ConvTranspose).
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            _get_norm(norm, out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────────────────────
# U-Net Generator
# ──────────────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    Pix2Pix U-Net generator (8 downsampling layers for 256×256 input).

    Args:
        in_channels:  1 for SAR VV.
        out_channels: 3 for RGB.
        ngf:          Base filter count (64).
        norm:         'batch' | 'instance' | 'none'.
        use_dropout:  Dropout in innermost 3 decoder layers.
    """

    def __init__(
        self,
        in_channels:  int  = 1,
        out_channels: int  = 3,
        ngf:          int  = 64,
        num_downs:    int  = 8,   # kept for API compat; fixed at 8 for 256×256
        norm:         str  = "batch",
        use_dropout:  bool = True,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = UNetDown(in_channels,  ngf,     norm=norm, use_norm=False)  # 256→128
        self.enc2 = UNetDown(ngf,          ngf*2,   norm=norm)                  # 128→64
        self.enc3 = UNetDown(ngf*2,        ngf*4,   norm=norm)                  # 64→32
        self.enc4 = UNetDown(ngf*4,        ngf*8,   norm=norm)                  # 32→16
        self.enc5 = UNetDown(ngf*8,        ngf*8,   norm=norm)                  # 16→8
        self.enc6 = UNetDown(ngf*8,        ngf*8,   norm=norm)                  # 8→4
        self.enc7 = UNetDown(ngf*8,        ngf*8,   norm=norm)                  # 4→2

        # Bottleneck (no norm; this is the innermost 1×1 representation)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1),        # 2→1
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──────────────────────────────────────────
        # in_ch = (previous decoder output) + (skip connection channels)
        self.dec1 = UNetUp(ngf*8  + ngf*8,  ngf*8,  norm=norm, dropout=use_dropout)  # 1→2
        self.dec2 = UNetUp(ngf*8  + ngf*8,  ngf*8,  norm=norm, dropout=use_dropout)  # 2→4
        self.dec3 = UNetUp(ngf*8  + ngf*8,  ngf*8,  norm=norm, dropout=use_dropout)  # 4→8
        self.dec4 = UNetUp(ngf*8  + ngf*8,  ngf*8,  norm=norm, dropout=False)        # 8→16
        self.dec5 = UNetUp(ngf*8  + ngf*4,  ngf*4,  norm=norm, dropout=False)        # 16→32
        self.dec6 = UNetUp(ngf*4  + ngf*2,  ngf*2,  norm=norm, dropout=False)        # 32→64
        self.dec7 = UNetUp(ngf*2  + ngf,    ngf,    norm=norm, dropout=False)        # 64→128

        # Final upsampling: dec7_out(ngf=64) → 3 channels at 256×256
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),   # output in [-1, 1] matches [-1,1] EO normalisation
        )

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        # ── Encode ───────────────────────────────────────────
        e1 = self.enc1(sar)         # (B,  64, 128, 128)
        e2 = self.enc2(e1)          # (B, 128,  64,  64)
        e3 = self.enc3(e2)          # (B, 256,  32,  32)
        e4 = self.enc4(e3)          # (B, 512,  16,  16)
        e5 = self.enc5(e4)          # (B, 512,   8,   8)
        e6 = self.enc6(e5)          # (B, 512,   4,   4)
        e7 = self.enc7(e6)          # (B, 512,   2,   2)
        bn = self.bottleneck(e7)    # (B, 512,   1,   1)

        # ── Decode (skip connections keep structural information) ─
        d1 = self.dec1(bn, e7)      # concat(512,512)=1024 → 512  at 2×2
        d2 = self.dec2(d1, e6)      # concat(512,512)=1024 → 512  at 4×4
        d3 = self.dec3(d2, e5)      # concat(512,512)=1024 → 512  at 8×8
        d4 = self.dec4(d3, e4)      # concat(512,512)=1024 → 512  at 16×16
        d5 = self.dec5(d4, e3)      # concat(512,256)= 768 → 256  at 32×32
        d6 = self.dec6(d5, e2)      # concat(256,128)= 384 → 128  at 64×64
        d7 = self.dec7(d6, e1)      # concat(128, 64)= 192 →  64  at 128×128
        return self.final(d7)       # 64 → 3, 128→256, Tanh


def build_generator(cfg: dict) -> UNetGenerator:
    """Instantiate generator from config dict."""
    g = cfg["model"]["generator"]
    return UNetGenerator(
        in_channels  = cfg["data"]["in_channels"],
        out_channels = cfg["data"]["out_channels"],
        ngf          = g["ngf"],
        num_downs    = g["num_downs"],
        norm         = g["norm"],
        use_dropout  = g["dropout"],
    )
