"""
eval.py — Evaluation Script
=============================
Computes LPIPS, FID, SSIM, PSNR on a set of predicted and ground-truth images.

Usage:
    # Step 1: generate predictions with infer.py
    python infer.py --input_dir dataset/val/s1 --output_dir outputs/val_preds --weights outputs/final_weights.pth

    # Step 2: evaluate
    python eval.py --pred_dir outputs/val_preds --gt_dir dataset/val/s2

    # Optional: save results to JSON
    python eval.py --pred_dir outputs/val_preds --gt_dir dataset/val/s2 --save_json outputs/metrics.json
"""

import argparse
import json
import sys

from utils.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAR-to-EO translation metrics")
    parser.add_argument("--pred_dir",  required=True, help="Directory of generated EO images (PNG)")
    parser.add_argument("--gt_dir",    required=True, help="Directory of ground-truth EO images (PNG)")
    parser.add_argument("--save_json", default=None,  help="Optional path to save results as JSON")
    args = parser.parse_args()

    print("=" * 50)
    print("SAR-to-EO Evaluation")
    print("=" * 50)
    print(f"  Predictions : {args.pred_dir}")
    print(f"  Ground truth: {args.gt_dir}")
    print()

    metrics = compute_all_metrics(args.pred_dir, args.gt_dir)

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"  LPIPS (↓ better): {metrics['LPIPS']}")
    print(f"  FID   (↓ better): {metrics['FID']}")
    print(f"  SSIM  (↑ better): {metrics['SSIM']}")
    print(f"  PSNR  (↑ better): {metrics['PSNR']} dB")
    print("=" * 50)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Results saved → {args.save_json}")


if __name__ == "__main__":
    main()
