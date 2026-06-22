"""
infer.py — Inference Script (GalaxEye I/O Contract)
=====================================================
Conforms exactly to the required interface:

    python infer.py --input_dir <path> --output_dir <path> --weights <path/to/checkpoint>

Input contract:
  - A directory of single-channel Sentinel-1 SAR (VV) patches
  - 256×256, 8-bit PNG, dB-scaled and min-max normalised to [0, 255]

Output contract:
  - A directory of generated 256×256 RGB PNG images
  - Same filenames as the corresponding inputs

Requirements:
  - Runs on a single GPU with ≤16 GB VRAM (Colab/Kaggle free tier)
  - No internet access at inference time (weights loaded locally)
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

from models import build_generator


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_sar_patch(path: str, image_size: int = 256) -> torch.Tensor:
    """
    Load a single SAR PNG patch and convert to a model-ready tensor.

    The input is expected to be:
      - 8-bit grayscale PNG
      - dB-scaled, min-max normalised to [0, 255]

    We apply the same preprocessing as during training:
      [0, 255] → [0, 1]  (ToTensor)
      [0, 1]   → [-1, 1] (Normalise with mean=0.5, std=0.5)

    Returns:
        Tensor of shape (1, 1, H, W) ready for the generator.
    """
    img = Image.open(path).convert("L")   # force single-channel

    # Resize if not already 256×256 (shouldn't be needed but safe)
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)

    transform = T.Compose([
        T.ToTensor(),                              # → (1, H, W) in [0,1]
        T.Normalize(mean=[0.5], std=[0.5]),        # → [-1, 1]
    ])
    return transform(img).unsqueeze(0)             # (1, 1, H, W)


def save_eo_tensor(tensor: torch.Tensor, path: str):
    """
    Convert a generated EO tensor back to a uint8 RGB PNG.

    The generator outputs values in [-1, 1] (tanh activation).
    Denormalise: [-1, 1] → [0, 1] → [0, 255] uint8.
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]                        # (3, H, W)

    # Denormalise
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)    # [0, 1]
    pil_img = F.to_pil_image(tensor.cpu())        # PIL RGB

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    pil_img.save(path)


# ──────────────────────────────────────────────────────────────
# Main inference loop
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAR-to-EO inference")
    parser.add_argument("--input_dir",  required=True,  help="Directory of 256×256 SAR PNG patches")
    parser.add_argument("--output_dir", required=True,  help="Directory to write generated EO PNGs")
    parser.add_argument("--weights",    required=True,  help="Path to model checkpoint (.pth)")
    parser.add_argument("--device",     default="auto", help="'cuda', 'cpu', or 'auto' (default)")
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[infer] Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────
    print(f"[infer] Loading weights from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)

    # The checkpoint stores the config so we can rebuild the exact model
    # that was trained, regardless of what config.yaml currently says.
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        # Fallback: try to load config.yaml from the same directory
        config_path = Path(args.weights).parent.parent / "config.yaml"
        if not config_path.exists():
            config_path = Path("config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[infer] Warning: checkpoint has no embedded config — loaded {config_path}")

    # ── Build and load generator ─────────────────────────────
    generator = build_generator(cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print("[infer] Generator loaded successfully.")

    # ── Collect input files ──────────────────────────────────
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ])

    if len(input_files) == 0:
        print(f"[infer] ERROR: No image files found in {input_dir}")
        return

    print(f"[infer] Processing {len(input_files)} patches...")

    # ── Run inference ─────────────────────────────────────────
    with torch.no_grad():
        for i, sar_path in enumerate(input_files):
            # Load SAR patch
            sar_tensor = load_sar_patch(str(sar_path)).to(device)  # (1, 1, 256, 256)

            # If the model was trained on VV+VH (SEN12MS) but we receive
            # single-channel VV, duplicate the channel to match model input.
            in_ch = cfg["data"]["in_channels"]
            if in_ch == 2 and sar_tensor.shape[1] == 1:
                sar_tensor = sar_tensor.repeat(1, 2, 1, 1)   # (1, 2, 256, 256)

            # Generate EO
            fake_eo = generator(sar_tensor)   # (1, 3, 256, 256) in [-1, 1]

            # Save with the same filename as the input
            out_path = output_dir / (sar_path.stem + ".png")
            save_eo_tensor(fake_eo, str(out_path))

            if (i + 1) % 100 == 0 or (i + 1) == len(input_files):
                print(f"  [{i+1}/{len(input_files)}] {sar_path.name} → {out_path.name}")

    print(f"[infer] Done. Generated EO images saved to: {output_dir}")


if __name__ == "__main__":
    main()
