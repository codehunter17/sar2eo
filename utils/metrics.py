"""
utils/metrics.py — Evaluation Metrics
=======================================
Computes the four metrics required by the assignment:

Primary (perceptual — drive ranking):
  LPIPS ↓  Learned Perceptual Image Patch Similarity
  FID   ↓  Fréchet Inception Distance

Secondary (pixel-level):
  SSIM  ↑  Structural Similarity Index
  PSNR  ↑  Peak Signal-to-Noise Ratio

Why perceptual metrics matter for SAR-to-EO
--------------------------------------------
SAR-to-EO is ill-posed: the model cannot recover true colour or
spectral content from a radar backscatter signal.  Any "correct"
solution is one of many plausible EO images consistent with the
SAR input.

PSNR / SSIM penalise every pixel deviation equally.  A model that
predicts the average EO colour (a blurry mean image) will often
score HIGHER on PSNR than a model producing sharp, realistic-looking
images — because averaging reduces squared error.

LPIPS uses features from a pretrained VGG network to measure how
perceptually different two patches are.  It correlates much better
with human judgement of image quality.

FID compares the distribution of generated images to the distribution
of real images in Inception feature space.  A model that merely
memorises the training distribution will score well on PSNR/SSIM but
potentially poorly on FID if it doesn't generalise.

The gap between your PSNR score and your LPIPS/FID score is often
the most informative part of the analysis.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# PSNR & SSIM  (using torchmetrics for reliability)
# ──────────────────────────────────────────────────────────────

def compute_psnr_ssim(
    pred_dir: str,
    gt_dir:   str,
) -> Tuple[float, float]:
    """
    Compute mean PSNR and SSIM over all image pairs in two directories.

    Args:
        pred_dir: Directory of generated EO images (PNG).
        gt_dir:   Directory of ground-truth EO images (PNG).

    Returns:
        (mean_psnr, mean_ssim)
    """
    from pathlib import Path
    from PIL import Image
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    pred_files = sorted(Path(pred_dir).glob("*.png"))
    if len(pred_files) == 0:
        raise FileNotFoundError(f"No PNG files found in {pred_dir}")

    psnr_vals, ssim_vals = [], []

    for pred_path in pred_files:
        gt_path = Path(gt_dir) / pred_path.name
        if not gt_path.exists():
            continue

        # Load as float tensors in [0, 1]
        pred = _load_image_tensor(str(pred_path))
        gt   = _load_image_tensor(str(gt_path))

        psnr_vals.append(psnr_metric(pred, gt).item())
        ssim_vals.append(ssim_metric(pred, gt).item())

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def _load_image_tensor(path: str) -> torch.Tensor:
    """Load a PNG as a (1, C, H, W) float tensor in [0, 1]."""
    from PIL import Image
    import torchvision.transforms.functional as F
    img = Image.open(path).convert("RGB")
    return F.to_tensor(img).unsqueeze(0)   # (1, 3, H, W), [0, 1]


# ──────────────────────────────────────────────────────────────
# LPIPS
# ──────────────────────────────────────────────────────────────

def compute_lpips(
    pred_dir: str,
    gt_dir:   str,
    net:      str = "vgg",
) -> float:
    """
    Compute mean LPIPS over all image pairs.

    Uses the official lpips library (Zhang et al., 2018).
    Lower is better.

    Args:
        pred_dir: Directory of generated EO images.
        gt_dir:   Directory of ground-truth EO images.
        net:      Backbone: 'vgg' | 'alex' | 'squeeze'
    """
    import lpips
    from pathlib import Path

    loss_fn = lpips.LPIPS(net=net)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = loss_fn.to(device)

    pred_files = sorted(Path(pred_dir).glob("*.png"))
    vals = []

    for pred_path in pred_files:
        gt_path = Path(gt_dir) / pred_path.name
        if not gt_path.exists():
            continue

        # LPIPS expects tensors in [-1, 1]
        pred = _load_image_tensor(str(pred_path)).to(device) * 2 - 1
        gt   = _load_image_tensor(str(gt_path)).to(device) * 2 - 1

        with torch.no_grad():
            vals.append(loss_fn(pred, gt).item())

    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────
# FID
# ──────────────────────────────────────────────────────────────

def compute_fid(
    pred_dir:   str,
    gt_dir:     str,
    num_samples: int = 2048,
) -> float:
    """
    Compute FID (Fréchet Inception Distance) between generated and
    real EO images using pytorch-fid.

    FID measures the distance between the Inception feature
    distributions of real vs. generated images.  Lower is better.

    Args:
        pred_dir:    Directory of generated EO images.
        gt_dir:      Directory of ground-truth EO images.
        num_samples: Cap on how many samples to use (≥2048 recommended).
    """
    from pytorch_fid.fid_score import calculate_fid_given_paths
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = calculate_fid_given_paths(
        [pred_dir, gt_dir],
        batch_size=32,
        device=device,
        dims=2048,
    )
    return float(fid)


# ──────────────────────────────────────────────────────────────
# All-in-one
# ──────────────────────────────────────────────────────────────

def compute_all_metrics(pred_dir: str, gt_dir: str) -> dict:
    """
    Compute LPIPS, FID, SSIM, PSNR in one call.
    Returns a dict suitable for printing / saving.
    """
    print("Computing PSNR and SSIM...")
    psnr, ssim = compute_psnr_ssim(pred_dir, gt_dir)

    print("Computing LPIPS...")
    lpips_val = compute_lpips(pred_dir, gt_dir)

    print("Computing FID...")
    fid = compute_fid(pred_dir, gt_dir)

    return {
        "LPIPS": round(lpips_val, 4),
        "FID":   round(fid, 2),
        "SSIM":  round(ssim, 4),
        "PSNR":  round(psnr, 2),
    }
