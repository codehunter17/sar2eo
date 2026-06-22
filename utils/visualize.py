"""
utils/visualize.py — Visualization Utilities
=============================================
Saves:
  - Loss curves (train G loss, train D loss, val L1)
  - Side-by-side triplets: SAR input | Generated EO | Ground Truth EO
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/Colab
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────
# Loss curve
# ──────────────────────────────────────────────────────────────

def save_loss_curve(history: dict, save_path: str, mode: str = "full_gan"):
    """
    Plot and save the training loss curve.

    For full_gan mode: plots G_total, G_L1, D_total, val_L1.
    For l1_only mode: plots G_total (=G_L1) and val_L1.

    Args:
        history:   Dict of lists, keys matching CSV column names.
        save_path: File path for the PNG (e.g. outputs/loss_curve.png).
        mode:      'full_gan' | 'l1_only'
    """
    epochs = list(range(1, len(history["epoch"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    # ── Left: Generator losses ───────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["g_total"], label="G total", color="steelblue", linewidth=2)
    ax.plot(epochs, history["g_l1"],    label="G L1",    color="cornflowerblue", linestyle="--")
    if mode == "full_gan":
        ax.plot(epochs, history["g_adv"], label="G adv", color="tomato", linestyle=":")
    # Val L1 (only logged every val_every_n_epochs, so filter zeros)
    val_epochs = [e for e, v in zip(epochs, history["val_l1"]) if v > 0]
    val_vals   = [v for v in history["val_l1"] if v > 0]
    if val_epochs:
        ax.plot(val_epochs, val_vals, label="Val L1", color="forestgreen",
                marker="o", markersize=4, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Generator Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: Discriminator losses ──────────────────────────
    ax = axes[1]
    if mode == "full_gan":
        ax.plot(epochs, history["d_total"], label="D total", color="darkorange", linewidth=2)
        ax.plot(epochs, history["d_real"],  label="D real",  color="gold",       linestyle="--")
        ax.plot(epochs, history["d_fake"],  label="D fake",  color="coral",      linestyle=":")
        ax.set_title("Discriminator Loss")
    else:
        ax.text(0.5, 0.5, "No discriminator\n(L1-only ablation)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Discriminator Loss (N/A)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Qualitative triplets
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def save_sample_triplets(
    val_loader:  DataLoader,
    generator:   torch.nn.Module,
    device:      torch.device,
    save_dir:    str,
    epoch:       int,
    n_samples:   int = 5,
):
    """
    Save N side-by-side triplets: SAR | Generated EO | Ground Truth EO.
    Covers a range of samples so we capture both successes and failures.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    generator.eval()

    collected_sar, collected_fake, collected_real = [], [], []

    for batch in val_loader:
        sar_t = batch["sar"].to(device)
        eo_t  = batch["eo"].to(device)
        fake  = generator(sar_t)

        for i in range(sar_t.size(0)):
            collected_sar.append(sar_t[i].cpu())
            collected_fake.append(fake[i].cpu())
            collected_real.append(eo_t[i].cpu())
            if len(collected_sar) >= n_samples:
                break
        if len(collected_sar) >= n_samples:
            break

    n = len(collected_sar)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["SAR Input", "Generated EO", "Ground Truth EO"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12, fontweight="bold")

    for row in range(n):
        sar_img  = _tensor_to_np(collected_sar[row],  grayscale=True)
        fake_img = _tensor_to_np(collected_fake[row], grayscale=False)
        real_img = _tensor_to_np(collected_real[row], grayscale=False)

        axes[row][0].imshow(sar_img,  cmap="gray", vmin=0, vmax=1)
        axes[row][1].imshow(fake_img)
        axes[row][2].imshow(real_img)

        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(f"Qualitative Results — Epoch {epoch}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"triplets_epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    generator.train()


def _tensor_to_np(tensor: torch.Tensor, grayscale: bool = False) -> np.ndarray:
    """
    Convert a normalised tensor ([-1,1]) to a numpy uint8 image [0,1] float.
    """
    arr = (tensor * 0.5 + 0.5).clamp(0, 1).numpy()
    if grayscale:
        # shape: (1, H, W) → (H, W)
        return arr[0]
    else:
        # shape: (3, H, W) → (H, W, 3)
        return arr.transpose(1, 2, 0)
