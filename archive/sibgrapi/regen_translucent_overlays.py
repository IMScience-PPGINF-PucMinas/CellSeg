#!/usr/bin/env python3
"""
Regenerate translucent overlays for every slice already processed under
``out_sibgrapi2026/<stem>/``, without re-running Cellpose or SICLE.

Reads:
    out_sibgrapi2026/<stem>/sicle/merged_percell_sicle_masks_int32.npy
    out_sibgrapi2026/<stem>/cp_flow/step04_masks_uint16.npy
    data_sibgrapi2026/data_sibgrapi2026/<stem>.png

Writes:
    out_sibgrapi2026/<stem>/sicle/merged_percell_sicle_translucent_sicle.png
    out_sibgrapi2026/<stem>/sicle/merged_percell_sicle_translucent_cellpose.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from percell_sicle_cellprob_pipeline import _translucent_mask_overlay


def parse_color(s: str) -> tuple[int, int, int]:
    r, g, b = (int(x.strip()) for x in s.split(","))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def main() -> int:
    import numpy as np
    from PIL import Image

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=str, default=str(ROOT / "out_sibgrapi2026"))
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data_sibgrapi2026/data_sibgrapi2026"))
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--sicle-color", type=str, default="0,255,0")
    p.add_argument("--cellpose-color", type=str, default="255,255,0")
    args = p.parse_args()

    out_root = Path(args.out_root)
    data_dir = Path(args.data_dir)
    sicle_rgb = parse_color(args.sicle_color)
    cp_rgb = parse_color(args.cellpose_color)

    n_ok = n_skip = 0
    for case_dir in sorted(out_root.iterdir()):
        if not case_dir.is_dir():
            continue
        stem = case_dir.name
        merged_npy = case_dir / "sicle" / "merged_percell_sicle_masks_int32.npy"
        cp_npy = case_dir / "cp_flow" / "step04_masks_uint16.npy"
        png = data_dir / f"{stem}.png"
        if not (merged_npy.is_file() and cp_npy.is_file() and png.is_file()):
            print(f"[{stem}] skip: missing inputs")
            n_skip += 1
            continue

        rgb = np.asarray(Image.open(png).convert("RGB"))
        merged = np.load(merged_npy)
        cp = np.load(cp_npy)
        if merged.shape[:2] != rgb.shape[:2]:
            import cv2

            rgb = cv2.resize(rgb, (merged.shape[1], merged.shape[0]), interpolation=cv2.INTER_LINEAR)

        out_sicle = _translucent_mask_overlay(rgb, merged, color_rgb=sicle_rgb, alpha=args.alpha)
        out_cp = _translucent_mask_overlay(rgb, cp, color_rgb=cp_rgb, alpha=args.alpha)

        s_path = case_dir / "sicle" / "merged_percell_sicle_translucent_sicle.png"
        c_path = case_dir / "sicle" / "merged_percell_sicle_translucent_cellpose.png"
        Image.fromarray(out_sicle).save(s_path)
        Image.fromarray(out_cp).save(c_path)
        print(f"[{stem}] wrote {s_path.name}, {c_path.name}")
        n_ok += 1

    print(f"\nDone: {n_ok} updated, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
