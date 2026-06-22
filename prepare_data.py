"""
prepare_data.py — Arrange the Kaggle Sentinel-1&2 (terrain-segregated)
dataset into the train/val/test layout that config.yaml / data/dataset.py expect.

The Kaggle dataset "requiemonk/sentinel12-image-pairs-segregated-by-terrain"
is organised by land-cover terrain (4 classes, e.g. agri / barrenland /
grassland / urban). Each terrain folder contains an `s1` (SAR) and an `s2`
(optical RGB) subfolder of 256x256 PNGs, paired by an `s1`/`s2` token in the
filename.

Our dataloader (`data/dataset.py`) pairs SAR and EO files by IDENTICAL filename
stem, and reads from:

    dataset/
        train/ s1/  s2/
        val/   s1/  s2/
        test/  s1/  s2/   (optional, for the held-out test split)

So this script:
  1. Auto-discovers every terrain folder that contains both an s1 and s2 subdir
     (works regardless of the exact top-level folder name, e.g. `v_2`).
  2. Pairs each SAR file with its EO counterpart (token substitution, with a
     sorted-order fallback).
  3. Copies/symlinks pairs into dataset/{split}/{s1,s2}/ using a UNIFIED stem
     (`<terrain>_<index>.png`) so the dataloader pairs them correctly.
  4. Splits by TERRAIN (whole terrains held out for val/test) — the
     leakage-controlled split described in the technical report.

Usage (on Kaggle):
    python prepare_data.py \
        --src "/kaggle/input/sentinel12-image-pairs-segregated-by-terrain" \
        --dst "dataset" \
        --val_terrain grassland \
        --test_terrain urban

    # Quick subset run (e.g. 800 pairs/terrain) to validate the pipeline fast:
    python prepare_data.py --src <path> --dst dataset \
        --val_terrain grassland --test_terrain urban --max_per_terrain 800

After it finishes it prints the terrains found, the held-out terrains, and the
final train/val/test pair counts. Point `data_root` in config.yaml at --dst.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_terrains(src: Path):
    """
    Return a dict {terrain_name: (s1_dir, s2_dir)} for every folder under `src`
    that contains both an 's1' and an 's2' subdirectory (case-insensitive).
    """
    terrains = {}
    for dirpath, dirnames, _ in os.walk(src):
        lower = {d.lower(): d for d in dirnames}
        if "s1" in lower and "s2" in lower:
            d = Path(dirpath)
            s1 = d / lower["s1"]
            s2 = d / lower["s2"]
            name = d.name
            if name.lower() in {"", ".", "v_2", "v2", "data", "dataset"}:
                name = d.parent.name or name
            base, k = name, 1
            while name in terrains:
                k += 1
                name = f"{base}_{k}"
            terrains[name] = (s1, s2)
    return terrains


def _norm_key(stem: str) -> str:
    """
    Normalise a filename stem so an s1 file and its s2 partner map to the same
    key: drop the s1/s2 sensor token.
    e.g. 'ROIs1868_summer_s1_0_p30' -> 'rois1868_summer_0_p30'
    """
    s = stem.lower()
    s = re.sub(r"_s[12]_", "_", s)
    s = re.sub(r"(^|_)s[12]($|_)", r"\1\2", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s


def pair_files(s1_dir: Path, s2_dir: Path):
    """
    Return (pairs, unmatched).
    Strategy 1: match by normalised key (handles s1/s2 token in names).
    Strategy 2 (fallback): sorted-order pairing when key matching fails.
    """
    s1_files = sorted([p for p in s1_dir.iterdir() if _is_img(p)])
    s2_files = sorted([p for p in s2_dir.iterdir() if _is_img(p)])

    s2_by_key = {}
    for p in s2_files:
        s2_by_key.setdefault(_norm_key(p.stem), p)

    pairs, unmatched = [], []
    for p in s1_files:
        q = s2_by_key.get(_norm_key(p.stem))
        if q is not None:
            pairs.append((p, q))
        else:
            unmatched.append(p)

    if len(pairs) < 0.5 * max(1, min(len(s1_files), len(s2_files))):
        if len(s1_files) == len(s2_files) and len(s1_files) > 0:
            pairs = list(zip(s1_files, s2_files))
            unmatched = []
        else:
            n = min(len(s1_files), len(s2_files))
            pairs = list(zip(s1_files[:n], s2_files[:n]))
            unmatched = s1_files[n:]

    return pairs, unmatched


def place(pairs, dst: Path, split: str, terrain: str, link: bool, limit: int):
    """Copy (or symlink) pairs into dst/split/{s1,s2} with unified stems."""
    out_s1 = dst / split / "s1"
    out_s2 = dst / split / "s2"
    out_s1.mkdir(parents=True, exist_ok=True)
    out_s2.mkdir(parents=True, exist_ok=True)

    if limit and limit > 0:
        pairs = pairs[:limit]

    for i, (sp, ep) in enumerate(pairs):
        stem = f"{terrain}_{i:05d}"
        d1 = out_s1 / f"{stem}.png"
        d2 = out_s2 / f"{stem}.png"
        for src_p, dst_p in [(sp, d1), (ep, d2)]:
            if dst_p.exists() or dst_p.is_symlink():
                dst_p.unlink()
            if link:
                try:
                    os.symlink(os.path.abspath(src_p), dst_p)
                    continue
                except OSError:
                    pass
            shutil.copy2(src_p, dst_p)
    return len(pairs)


def main():
    ap = argparse.ArgumentParser(
        description="Prepare Kaggle Sentinel-1&2 terrain dataset into train/val/test layout."
    )
    ap.add_argument("--src", required=True, help="Root of the Kaggle dataset (read-only on Kaggle).")
    ap.add_argument("--dst", default="dataset", help="Output root (default: dataset).")
    ap.add_argument("--val_terrain", default=None,
                    help="Terrain held out for validation (case-insensitive substring). "
                         "If omitted, the alphabetically-last terrain is used.")
    ap.add_argument("--test_terrain", default=None,
                    help="Optional terrain held out as a separate TEST split. Recommended, since "
                         "the assignment asks for metrics on both val and test splits.")
    ap.add_argument("--max_per_terrain", type=int, default=0,
                    help="Cap pairs per terrain (0 = use all). Useful for fast pipeline checks.")
    ap.add_argument("--copy", action="store_true",
                    help="Force file copy instead of symlink (symlink is default).")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        sys.exit(f"[prepare_data] ERROR: --src does not exist: {src}")

    terrains = find_terrains(src)
    if not terrains:
        sys.exit(f"[prepare_data] ERROR: no folders with both s1/ and s2/ subdirs found under {src}.\n"
                 f"Inspect the dataset layout and pass the correct --src.")

    names = sorted(terrains.keys())
    print(f"[prepare_data] Found {len(names)} terrain folder(s): {names}")

    if args.val_terrain:
        match = [n for n in names if args.val_terrain.lower() in n.lower()]
        if not match:
            sys.exit(f"[prepare_data] ERROR: --val_terrain '{args.val_terrain}' did not match any of {names}.")
        val_terrain = match[0]
    else:
        val_terrain = names[-1]

    test_terrain = None
    if args.test_terrain:
        match = [n for n in names if args.test_terrain.lower() in n.lower() and n != val_terrain]
        if not match:
            sys.exit(f"[prepare_data] ERROR: --test_terrain '{args.test_terrain}' did not match a terrain "
                     f"distinct from val ({val_terrain}). Options: {names}.")
        test_terrain = match[0]

    print(f"[prepare_data] Held-out validation terrain: '{val_terrain}'")
    if test_terrain:
        print(f"[prepare_data] Held-out test terrain:       '{test_terrain}'")
    train_terrains = [n for n in names if n not in {val_terrain, test_terrain}]
    print(f"[prepare_data] Training terrains: {train_terrains}")

    link = not args.copy
    total = {"train": 0, "val": 0, "test": 0}
    for name in names:
        s1_dir, s2_dir = terrains[name]
        pairs, unmatched = pair_files(s1_dir, s2_dir)
        if name == val_terrain:
            split = "val"
        elif name == test_terrain:
            split = "test"
        else:
            split = "train"
        n = place(pairs, dst, split, name, link, args.max_per_terrain)
        total[split] += n
        warn = f"  (WARNING: {len(unmatched)} unmatched s1 files)" if unmatched else ""
        print(f"  - {name:<14} -> {split:<5} : {n} pairs{warn}")

    print("-" * 56)
    print(f"[prepare_data] DONE.  Train: {total['train']} | Val: {total['val']} | Test: {total['test']} pairs")
    print(f"[prepare_data] Layout written under: {dst.resolve()}")
    print(f"[prepare_data] Set  data.data_root: \"{dst}\"  in config.yaml")
    if total["val"] == 0:
        print("[prepare_data] WARNING: validation split is empty — check --val_terrain.")


if __name__ == "__main__":
    main()
