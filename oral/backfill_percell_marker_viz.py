#!/usr/bin/env python3
"""Backfill markers_object_bg_ignored.png for existing per-cell runs (no re-segmentation)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from _paths import PIPE, REPO, RUNS

RUNS_DEFAULT = [
    RUNS / "percell_idisf_full",
    RUNS / "conquest_exclude_other_full",
]


def _backfill_roi(cat: str, stem: str, root: Path, *, exclude_other: bool, force: bool) -> int:
    sys.path.insert(0, str(PIPE))
    dout = REPO.parent
    if str(dout) not in sys.path:
        sys.path.insert(0, str(dout))

    from cellpose_to_idisf_pipeline import eroded_foreground_mask, mask_to_scribble_coords
    from percell_conquest_viz import (
        background_scribble_mask,
        unconquerable_mask,
        write_percell_marker_figure,
    )
    from percell_sicle_cellprob_pipeline import bbox_for_label, fg_scribble_coords, load_cellprob_masks

    roi_dir = root / cat / stem
    cp_dir = RUNS / "postprocess_ablation_full" / cat / stem / "cp_flow"
    if not (cp_dir / "step04_masks_uint16.npy").is_file():
        return 0

    input_png = RUNS / "postprocess_ablation_full" / cat / stem / f"{stem}.png"
    if not input_png.is_file():
        return 0

    _, masks, _ = load_cellprob_masks(cp_dir)
    h, w = masks.shape
    img_full = np.asarray(Image.open(input_png).convert("RGB"))

    n = 0
    for pco in roi_dir.glob("*/percell_cell_outputs"):
        for cell_dir in sorted(pco.glob("cell_*")):
            out_png = cell_dir / "markers_object_bg_ignored.png"
            if out_png.is_file() and not force:
                continue
            try:
                lab = int(cell_dir.name.split("_")[-1])
            except ValueError:
                continue
            r0, r1, c0, c1 = bbox_for_label(masks, lab, 4, h, w)
            full_crop = masks[r0:r1, c0:c1]
            crop_m = (full_crop == lab).astype(np.uint8)
            ch, cw = crop_m.shape
            crop_rgb = img_full[r0:r1, c0:c1]
            fg_mask = eroded_foreground_mask(crop_m, 1)
            bg_mask = background_scribble_mask(
                ch,
                cw,
                full_crop,
                lab,
                border_px=2,
                use_unconquerable=exclude_other,
            )
            ign = unconquerable_mask(full_crop, lab) if exclude_other else None
            if exclude_other and ign is not None:
                bg_mask = (np.asarray(bg_mask, dtype=bool) & ~np.asarray(ign, dtype=bool)).astype(np.uint8)
            fg_coords = fg_scribble_coords(crop_m.astype(bool), erosion_pixels=0)
            bg_coords = mask_to_scribble_coords(bg_mask)
            write_percell_marker_figure(
                cell_dir,
                crop_rgb,
                fg_mask=fg_mask,
                bg_mask=bg_mask,
                ignored_mask=ign,
                fg_coords=fg_coords,
                bg_coords=bg_coords,
            )
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, action="append", default=[])
    p.add_argument("--no-exclude-other", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite existing marker PNGs")
    args = p.parse_args()

    roots = [Path(r) for r in args.root] if args.root else RUNS_DEFAULT
    exclude_other = not args.no_exclude_other
    total = 0
    for root in roots:
        if not root.is_dir():
            print(f"skip missing {root}")
            continue
        for cat in ("healthy", "severe"):
            cat_dir = root / cat
            if not cat_dir.is_dir():
                continue
            for roi_dir in sorted(cat_dir.iterdir()):
                if not roi_dir.is_dir():
                    continue
                n = _backfill_roi(cat, roi_dir.name, root, exclude_other=exclude_other, force=args.force)
                if n:
                    print(f"{cat}/{roi_dir.name}: {n} cells")
                total += n
    print(f"Backfilled {total} marker figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
