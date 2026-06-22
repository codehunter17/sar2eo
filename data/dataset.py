"""
data/dataset.py — SAR-to-EO Dataset and DataLoader
====================================================
Loads paired SAR (Sentinel-1 VV) and EO (Sentinel-2 RGB) patches
from the Kaggle Sentinel-1&2 terrain-segregated dataset.

Dataset directory layout expected:
    dataset/
        train/
            s1/   ← SAR patches  (grayscale PNG or TIFF, 256×256)
            s2/   ← EO  patches  (RGB PNG or TIFF,       256×256)
        val/
            s1/
            s2/

The Kaggle dataset organises images by terrain type (urban,
vegetation, water, barren).  We exploit that structure as our
train/val split to avoid adjacent-patch leakage.

SAR preprocessing rationale
----------------------------
SAR amplitude spans a very large dynamic range. Raw backscatter
values are first converted to dB (10·log10) to compress the range,
then min-max normalised to [0, 255] for storage. When we load the
PNG we divide by 255 → [0, 1], then normalise with mean=0.5,
std=0.5 → [-1, 1] to match the generator's tanh output range.

EO preprocessing rationale
---------------------------
Sentinel-2 optical bands are stored as uint16 reflectance values
(0–10 000 typical range). For RGB we extract bands B4/B3/B2 and
scale to [0, 255] uint8.  Again normalise to [-1, 1] to match
the generator's output.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────
# Helper: collect file pairs
# ──────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _collect_pairs(sar_dir: str, eo_dir: str) -> list:
    """
    Returns a sorted list of (sar_path, eo_path) pairs where both
    files share the same filename stem (e.g. patch_0001.png).
    """
    sar_files = {
        p.stem: p
        for p in sorted(Path(sar_dir).iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS
    }
    eo_files = {
        p.stem: p
        for p in sorted(Path(eo_dir).iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS
    }
    common = sorted(sar_files.keys() & eo_files.keys())
    if len(common) == 0:
        raise FileNotFoundError(
            f"No matching SAR/EO pairs found.\n"
            f"  SAR dir: {sar_dir}  ({len(sar_files)} files)\n"
            f"  EO  dir: {eo_dir}  ({len(eo_files)} files)"
        )
    return [(str(sar_files[k]), str(eo_files[k])) for k in common]


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class SARtoEODataset(Dataset):
    """
    Paired SAR–EO dataset.

    Args:
        sar_dir:    Path to the directory of SAR patches.
        eo_dir:     Path to the directory of EO patches.
        image_size: Spatial size to resize patches to (default 256).
        augment:    If True, apply random horizontal flips.
        fraction:   Float in (0, 1] — use only this fraction of the
                    data (useful for quick experiments on Colab).
        sar_mean / sar_std:  Normalisation for SAR channel.
        eo_mean  / eo_std:   Normalisation for EO channels (list of 3).
    """

    def __init__(
        self,
        sar_dir: str,
        eo_dir: str,
        image_size: int = 256,
        augment: bool = False,
        fraction: float = 1.0,
        sar_mean: float = 0.5,
        sar_std: float = 0.5,
        eo_mean: Tuple = (0.5, 0.5, 0.5),
        eo_std: Tuple = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.pairs = _collect_pairs(sar_dir, eo_dir)

        # Optionally sub-sample for fast experiments
        if fraction < 1.0:
            n = max(1, int(len(self.pairs) * fraction))
            self.pairs = random.sample(self.pairs, n)

        self.image_size = image_size
        self.augment = augment

        # SAR: grayscale (L mode) → 1-channel tensor, normalised to [-1,1]
        self.sar_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),                         # [0, 255] → [0.0, 1.0]
            T.Normalize(mean=[sar_mean], std=[sar_std]),  # → [-1, 1]
        ])

        # EO: RGB → 3-channel tensor, normalised to [-1,1]
        self.eo_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=list(eo_mean), std=list(eo_std)),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sar_path, eo_path = self.pairs[idx]

        # ── Load SAR ────────────────────────────────────────
        # The Kaggle dataset stores SAR as single-channel PNG.
        # We open in "L" mode (8-bit grayscale) which PIL converts
        # to a single channel without gamma correction.
        sar_img = Image.open(sar_path).convert("L")

        # ── Load EO ─────────────────────────────────────────
        # EO patches are RGB; open in "RGB" mode.
        eo_img = Image.open(eo_path).convert("RGB")

        # ── Augmentation ────────────────────────────────────
        # Apply the same random flip to both images so they stay
        # aligned.  Vertical flip is disabled — SAR images have a
        # directional look-angle that makes vertical flipping
        # physically implausible.
        if self.augment and random.random() > 0.5:
            sar_img = T.functional.hflip(sar_img)
            eo_img  = T.functional.hflip(eo_img)

        sar_tensor = self.sar_transform(sar_img)   # shape: (1, H, W)
        eo_tensor  = self.eo_transform(eo_img)     # shape: (3, H, W)

        return {
            "sar": sar_tensor,
            "eo":  eo_tensor,
            "sar_path": sar_path,
            "eo_path":  eo_path,
        }


# ──────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train and validation DataLoaders from the config dict.

    Args:
        cfg: the parsed config.yaml as a Python dict.

    Returns:
        (train_loader, val_loader)
    """
    d = cfg["data"]
    root = d["data_root"]

    train_ds = SARtoEODataset(
        sar_dir    = os.path.join(root, "train", "s1"),
        eo_dir     = os.path.join(root, "train", "s2"),
        image_size = d["image_size"],
        augment    = d["augment"]["horizontal_flip"],
        fraction   = d["train_fraction"],
        sar_mean   = d["sar_mean"],
        sar_std    = d["sar_std"],
        eo_mean    = d["eo_mean"],
        eo_std     = d["eo_std"],
    )

    val_ds = SARtoEODataset(
        sar_dir    = os.path.join(root, "val", "s1"),
        eo_dir     = os.path.join(root, "val", "s2"),
        image_size = d["image_size"],
        augment    = False,   # never augment validation data
        fraction   = d["val_fraction"],
        sar_mean   = d["sar_mean"],
        sar_std    = d["sar_std"],
        eo_mean    = d["eo_mean"],
        eo_std     = d["eo_std"],
    )

    t_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size  = t_cfg["batch_size"],
        shuffle     = True,
        num_workers = t_cfg["num_workers"],
        pin_memory  = t_cfg["pin_memory"],
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["eval"]["batch_size"],
        shuffle     = False,
        num_workers = t_cfg["num_workers"],
        pin_memory  = t_cfg["pin_memory"],
        drop_last   = False,
    )

    print(f"[Dataset] Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────
# Tensor ↔ image utilities (used in visualisation + infer.py)
# ──────────────────────────────────────────────────────────────

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalised tensor ([-1,1]) back to a PIL Image.

    Works for both 1-channel (SAR) and 3-channel (EO) tensors.
    Removes the batch dimension if present.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]                  # take first item in batch

    # Denormalise: [-1,1] → [0,1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)

    # Convert to uint8 numpy array
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


def pil_to_sar_tensor(img: Image.Image, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """
    Convert a uint8 grayscale PIL image to a normalised SAR tensor.
    Returns shape (1, 1, H, W) — batch dimension included.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[mean], std=[std]),
    ])
    return transform(img.convert("L")).unsqueeze(0)   # (1, 1, H, W)
