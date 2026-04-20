#!/usr/bin/env python3
"""
Convert MoNuSeg training TIFFs in "Tissue Images" to PNG (same basename).

Default paths (this repo):
  Input:  monuseg/MoNuSeg2018/Tissue Images/*.tif
  Output: monuseg/MoNuSeg2018/Tissue Images PNG/*.png

Examples:
  python convert_monuseg_tif_to_png.py
  python convert_monuseg_tif_to_png.py --out /tmp/monuseg_png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TIF_DIR = PROJECT_ROOT / "monuseg" / "MoNuSeg2018" / "Tissue Images"
DEFAULT_PNG_DIR = PROJECT_ROOT / "monuseg" / "MoNuSeg2018" / "Tissue Images PNG"


def convert_one(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        if im.mode in ("I;16", "I"):
            arr = np.asarray(im, dtype=np.float64)
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            else:
                arr = np.zeros_like(arr)
            im = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L").convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "PNG", compress_level=6)


def main() -> int:
    ap = argparse.ArgumentParser(description="MoNuSeg TIF → PNG batch converter")
    ap.add_argument(
        "--tif-dir",
        type=Path,
        default=DEFAULT_TIF_DIR,
        help=f"Folder with .tif/.tiff (default: {DEFAULT_TIF_DIR})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PNG_DIR,
        help=f"Output folder for .png (default: {DEFAULT_PNG_DIR})",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PNGs",
    )
    args = ap.parse_args()
    tif_dir = args.tif_dir.resolve()
    out_dir = args.out.resolve()

    if not tif_dir.is_dir():
        print(f"ERROR: not a directory: {tif_dir}", file=sys.stderr)
        return 1

    paths = sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.tiff"))
    if not paths:
        print(f"No .tif/.tiff in {tif_dir}", file=sys.stderr)
        return 1

    done = 0
    skipped = 0
    for src in paths:
        dst = out_dir / (src.stem + ".png")
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        convert_one(src, dst)
        done += 1
        print(dst)

    print(f"Converted: {done}  skipped (exists): {skipped}  total tifs: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
