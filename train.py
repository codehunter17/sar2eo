"""
train.py — Training Script for SAR-to-EO Pix2Pix
===================================================
Usage:
    python train.py --config config.yaml

For the ablation experiment (L1 only, no GAN), change config.yaml:
    loss:
      mode: "l1_only"
then re-run:
    python train.py --config config.yaml

Outputs saved to outputs/:
    checkpoints/        model weights every N epochs
    loss_log.csv        per-epoch G and D losses
    loss_curve.png      plot of training + validation curves
    samples/            qualitative image triplets every val epoch
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import LambdaLR

from data import build_dataloaders, tensor_to_pil
from models import build_generator, build_discriminator, Pix2PixLoss
from utils.visualize import save_loss_curve, save_sample_triplets


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Learning rate scheduler: constant then linear decay
# ──────────────────────────────────────────────────────────────

def build_lr_scheduler(optimizer, total_epochs: int, decay_start: int):
    """
    Constant LR for [0, decay_start], then linearly decay to 0
    by epoch total_epochs.  Mirrors the original Pix2Pix schedule.
    """
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ──────────────────────────────────────────────────────────────
# One training step
# ──────────────────────────────────────────────────────────────

def train_step(
    batch, generator, discriminator, criterion, opt_g, opt_d, device, mode
):
    """
    Executes one forward + backward pass for both G and D.
    Returns dict of scalar loss values for logging.
    """
    sar = batch["sar"].to(device)
    real_eo = batch["eo"].to(device)

    # ── (1) Generate fake EO ─────────────────────────────────
    fake_eo = generator(sar)

    # ── (2) Update Discriminator ─────────────────────────────
    if mode != "l1_only":
        opt_d.zero_grad()

        # Detach fake_eo so generator gradients don't flow through D update
        fake_pred = discriminator(sar, fake_eo.detach())
        real_pred = discriminator(sar, real_eo)

        d_losses = criterion.discriminator_loss(real_pred, fake_pred)
        d_losses["d_total"].backward()
        opt_d.step()
    else:
        d_losses = {
            "d_total": torch.tensor(0.0, device=device),
            "d_real":  torch.tensor(0.0, device=device),
            "d_fake":  torch.tensor(0.0, device=device),
        }

    # ── (3) Update Generator ─────────────────────────────────
    opt_g.zero_grad()

    if mode != "l1_only":
        fake_pred_for_g = discriminator(sar, fake_eo)
    else:
        fake_pred_for_g = None

    g_losses = criterion.generator_loss(fake_pred_for_g, fake_eo, real_eo)
    g_losses["g_total"].backward()
    opt_g.step()

    return {**g_losses, **d_losses}


# ──────────────────────────────────────────────────────────────
# Validation (generator only, no GAN loss)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(val_loader, generator, device):
    """
    Run L1 loss on validation set to monitor overfitting.
    Returns mean val L1 loss.
    """
    generator.eval()
    import torch.nn as nn
    l1 = nn.L1Loss()
    total = 0.0
    count = 0

    for batch in val_loader:
        sar = batch["sar"].to(device)
        eo  = batch["eo"].to(device)
        fake = generator(sar)
        total += l1(fake, eo).item() * sar.size(0)
        count += sar.size(0)

    generator.train()
    return total / max(1, count)


# ──────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])

    # ── Directories ──────────────────────────────────────────
    out_dir = Path(cfg["experiment"]["output_dir"])
    ckpt_dir   = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    for d in [ckpt_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Models ───────────────────────────────────────────────
    generator     = build_generator(cfg).to(device)
    discriminator = build_discriminator(cfg).to(device)

    mode = cfg["loss"]["mode"]
    print(f"[Train] Loss mode: {mode}")

    # ── Loss ─────────────────────────────────────────────────
    criterion = Pix2PixLoss(
        lambda_l1=cfg["loss"]["lambda_l1"],
        mode=mode,
    )

    # ── Optimisers ───────────────────────────────────────────
    t = cfg["training"]
    opt_g = optim.Adam(generator.parameters(),
                       lr=t["lr"], betas=(t["beta1"], t["beta2"]))
    opt_d = optim.Adam(discriminator.parameters(),
                       lr=t["lr"], betas=(t["beta1"], t["beta2"]))

    sched_g = build_lr_scheduler(opt_g, t["epochs"], t["lr_decay_start_epoch"])
    sched_d = build_lr_scheduler(opt_d, t["epochs"], t["lr_decay_start_epoch"])

    # ── Loss log (CSV) ───────────────────────────────────────
    csv_path = out_dir / "loss_log.csv"
    csv_fields = ["epoch", "g_total", "g_adv", "g_l1", "d_total", "d_real", "d_fake", "val_l1"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    history = {k: [] for k in csv_fields}

    # ── Training epochs ──────────────────────────────────────
    for epoch in range(1, t["epochs"] + 1):
        generator.train()
        discriminator.train()

        # Running sums for this epoch
        epoch_losses = {k: 0.0 for k in ["g_total","g_adv","g_l1","d_total","d_real","d_fake"]}
        n_steps = 0

        for step, batch in enumerate(train_loader, 1):
            step_losses = train_step(
                batch, generator, discriminator,
                criterion, opt_g, opt_d, device, mode
            )
            for k in epoch_losses:
                epoch_losses[k] += step_losses[k].item()
            n_steps += 1

            if step % t["log_every_n_steps"] == 0:
                g = epoch_losses["g_total"] / n_steps
                d = epoch_losses["d_total"] / n_steps
                print(f"  Epoch {epoch:3d} | Step {step:4d} | G: {g:.4f} | D: {d:.4f}")

        # Average over steps
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_steps)

        # ── Validation ───────────────────────────────────────
        val_l1 = 0.0
        if epoch % t["val_every_n_epochs"] == 0:
            val_l1 = validate(val_loader, generator, device)
            print(f"[Val] Epoch {epoch:3d} | Val L1: {val_l1:.4f}")

            # Save sample triplets for qualitative inspection
            save_sample_triplets(
                val_loader, generator, device,
                save_dir=str(sample_dir),
                epoch=epoch,
                n_samples=5,
            )

        # ── Log to CSV ───────────────────────────────────────
        row = {**epoch_losses, "epoch": epoch, "val_l1": val_l1}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        for k in csv_fields:
            history[k].append(row.get(k, 0.0))

        # ── Checkpoint ───────────────────────────────────────
        if epoch % t["save_every_n_epochs"] == 0:
            ckpt = {
                "epoch":         epoch,
                "generator":     generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g":         opt_g.state_dict(),
                "opt_d":         opt_d.state_dict(),
                "cfg":           cfg,
            }
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"[Ckpt] Saved → {ckpt_path}")

            # Keep only the last N checkpoints
            keep = t.get("keep_last_n_checkpoints", 3)
            all_ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
            for old in all_ckpts[:-keep]:
                old.unlink()

        # ── LR scheduler step ────────────────────────────────
        sched_g.step()
        sched_d.step()

    # ── Save final checkpoint ────────────────────────────────
    torch.save({
        "epoch":         t["epochs"],
        "generator":     generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "cfg":           cfg,
    }, out_dir / "final_weights.pth")
    print(f"[Train] Final weights saved → {out_dir}/final_weights.pth")

    # ── Save loss curve ──────────────────────────────────────
    save_loss_curve(history, save_path=str(out_dir / "loss_curve.png"), mode=mode)
    print(f"[Train] Loss curve saved → {out_dir}/loss_curve.png")
    print("[Train] Done.")


if __name__ == "__main__":
    main()
