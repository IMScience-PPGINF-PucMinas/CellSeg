#!/usr/bin/env python3
"""
Build a self-contained pack:
  - copy of Oral Epithelium DB (images + annotations)
  - per-ROI Cellpose activation maps (cellprob, flow viz, instance masks)
  - per-cell crops with strict saliency (sigmoid inside mask only, thr+blur)

Output: exports/oral_epithelium_activation_pack/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from _paths import DATA, GT_COLORED, IMAGES_ORIGINAL, PIPE, REPO

OUT_PACK = REPO / "exports" / "oral_epithelium_activation_pack"
CP_ROOT = REPO / "outputs" / "runs" / "postprocess_ablation_full"

MARGIN = 4
SALIENCY_THRESHOLD = 0.3
SALIENCY_BLUR_SIGMA = 0.5


def discover_rois() -> list[tuple[str, str]]:
    rois: list[tuple[str, str]] = []
    for category in ("healthy", "severe"):
        col_dir = GT_COLORED / category
        for col_path in sorted(col_dir.glob("*.png")):
            stem = col_path.stem
            if (IMAGES_ORIGINAL / category / f"{stem}.tif").is_file():
                rois.append((category, stem))
    return rois


def _copy_dataset(dest: Path, *, symlink_images: bool) -> None:
    """Copy annotations; copy or symlink original TIFFs."""
    src_root = DATA
    dst_root = dest / "oral_epithelium_db"
    if dst_root.exists():
        print(f"Dataset dir exists: {dst_root}")
    else:
        shutil.copytree(
            src_root / "annotations",
            dst_root / "annotations",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        print(f"Copied annotations -> {dst_root / 'annotations'}")

    dst_img = dst_root / "images" / "original"
    dst_img.mkdir(parents=True, exist_ok=True)
    for category in ("healthy", "severe"):
        (dst_img / category).mkdir(parents=True, exist_ok=True)
        for tif in sorted((src_root / "images" / "original" / category).glob("*.tif")):
            dst = dst_img / category / tif.name
            if dst.exists():
                continue
            if symlink_images:
                dst.symlink_to(tif.resolve())
            else:
                shutil.copy2(tif, dst)
    print(f"Images -> {dst_img} ({'symlinks' if symlink_images else 'copies'})")


def _cellprob_to_u8(cellprob: np.ndarray) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-np.clip(cellprob.astype(np.float32), -50.0, 50.0)))
    return (np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)


def _overlay_cellprob_on_rgb(rgb: np.ndarray, cellprob_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    from PIL import Image

    h, w = min(rgb.shape[0], cellprob_u8.shape[0]), min(rgb.shape[1], cellprob_u8.shape[1])
    rgb = rgb[:h, :w].astype(np.float32)
    cp = cellprob_u8[:h, :w].astype(np.float32)
    # simple green-yellow heat on prob
    heat = np.zeros((h, w, 3), dtype=np.float32)
    heat[..., 1] = cp
    heat[..., 0] = cp * 0.35
    m = cp > 8
    out = rgb.copy()
    out[m] = (1.0 - alpha) * rgb[m] + alpha * heat[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def _export_cellpose_roi(category: str, stem: str, pack_root: Path) -> None:
    from PIL import Image

    sys.path.insert(0, str(PIPE))
    from percell_sicle_cellprob_pipeline import load_cellprob_masks

    case_src = CP_ROOT / category / stem
    cp_dir = case_src / "cp_flow"
    out_dir = pack_root / "cellpose_per_roi" / category / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_png = case_src / f"{stem}.png"
    if roi_png.is_file():
        shutil.copy2(roi_png, out_dir / "roi_rgb.png")
    else:
        from cellpose import io

        tif = IMAGES_ORIGINAL / category / f"{stem}.tif"
        img = io.imread(str(tif))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        Image.fromarray(img[..., :3].astype(np.uint8)).save(out_dir / "roi_rgb.png")

    cellprob, masks, dP = load_cellprob_masks(cp_dir)
    cp_u8 = _cellprob_to_u8(cellprob)
    Image.fromarray(cp_u8, mode="L").save(out_dir / "cellprob_activation_u8.png")

    rgb = np.asarray(Image.open(out_dir / "roi_rgb.png").convert("RGB"))
    Image.fromarray(_overlay_cellprob_on_rgb(rgb, cp_u8)).save(out_dir / "cellprob_overlay_rgb.png")

    np.save(out_dir / "cellprob_logits.npy", cellprob.astype(np.float32))
    np.save(out_dir / "cellpose_masks_int32.npy", masks.astype(np.int32))

    flow_png = cp_dir / "step03a_dP_flow_dx_to_circ.png"
    if flow_png.is_file():
        shutil.copy2(flow_png, out_dir / "cellpose_flow_hsv.png")

    masks_rgb = cp_dir / "step04_masks_rgb.png"
    if masks_rgb.is_file():
        shutil.copy2(masks_rgb, out_dir / "cellpose_masks_rgb.png")


def _export_per_cell_strict(category: str, stem: str, pack_root: Path) -> int:
    from PIL import Image

    sys.path.insert(0, str(PIPE))
    from percell_sicle_cellprob_pipeline import (
        apply_saliency_blur_u8,
        apply_saliency_threshold_u8,
        bbox_for_label,
        cellprob_crop_to_saliency_u8,
        load_cellprob_masks,
    )

    case_src = CP_ROOT / category / stem
    cp_dir = case_src / "cp_flow"
    out_roi = pack_root / "per_cell_strict" / category / stem
    out_roi.mkdir(parents=True, exist_ok=True)

    roi_png = pack_root / "cellpose_per_roi" / category / stem / "roi_rgb.png"
    if not roi_png.is_file():
        _export_cellpose_roi(category, stem, pack_root)
    rgb_full = np.asarray(Image.open(roi_png).convert("RGB"))

    cellprob, masks, _ = load_cellprob_masks(cp_dir)
    h, w = masks.shape
    n_cells = 0

    for lab in sorted(int(x) for x in np.unique(masks) if int(x) > 0):
        r0, r1, c0, c1 = bbox_for_label(masks, lab, MARGIN, h, w)
        crop_rgb = rgb_full[r0:r1, c0:c1]
        crop_cp = cellprob[r0:r1, c0:c1]
        crop_m = (masks[r0:r1, c0:c1] == lab).astype(np.uint8)

        # Strict: sigmoid only, zero outside this Cellpose instance
        sigmoid_u8 = cellprob_crop_to_saliency_u8(
            crop_cp, cell_mask=crop_m.astype(bool), linearize=False
        )
        strict_u8 = apply_saliency_threshold_u8(sigmoid_u8, SALIENCY_THRESHOLD)
        if SALIENCY_BLUR_SIGMA > 0.0:
            strict_u8 = apply_saliency_blur_u8(strict_u8, SALIENCY_BLUR_SIGMA)
        # Re-zero outside mask after blur (strict envelope)
        inside = crop_m.astype(bool)
        strict_u8 = strict_u8.copy()
        strict_u8[~inside] = 0

        cell_dir = out_roi / f"cell_{lab:05d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop_rgb.astype(np.uint8)).save(cell_dir / "crop_rgb.png")
        Image.fromarray((crop_m * 255).astype(np.uint8), mode="L").save(cell_dir / "cellpose_mask_u8.png")
        Image.fromarray(sigmoid_u8, mode="L").save(cell_dir / "cellprob_sigmoid_strict_u8.png")
        Image.fromarray(strict_u8, mode="L").save(cell_dir / "saliency_strict_thr_blur_u8.png")

        meta = {
            "label": lab,
            "bbox_r0": r0,
            "bbox_r1": r1,
            "bbox_c0": c0,
            "bbox_c1": c1,
            "margin_px": MARGIN,
            "area_px": int(crop_m.sum()),
            "saliency_threshold": SALIENCY_THRESHOLD,
            "saliency_blur_sigma": SALIENCY_BLUR_SIGMA,
            "saliency_mode": "sigmoid_inside_mask_only_no_otsu",
        }
        (cell_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        n_cells += 1

    (out_roi / "manifest.json").write_text(
        json.dumps(
            {
                "category": category,
                "roi": stem,
                "n_cells": n_cells,
                "margin_px": MARGIN,
                "cp_flow_source": str(cp_dir.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return n_cells


def _write_readme(pack_root: Path, n_rois: int, n_cells: int) -> None:
    text = f"""# Oral Epithelium — activation pack

Self-contained export for analysis and figures.

## Contents

| Path | Description |
|------|-------------|
| `oral_epithelium_db/` | Copy of dataset (annotations + original TIFFs) |
| `cellpose_per_roi/{{healthy,severe}}/{{roi}}/` | Full-ROI Cellpose outputs |
| `per_cell_strict/{{healthy,severe}}/{{roi}}/cell_XXXXX/` | Per-cell crop + strict saliency |

## Cellpose per ROI

- `roi_rgb.png` — H&E crop used in pipeline
- `cellprob_activation_u8.png` — sigmoid(cellprob logits), grayscale
- `cellprob_overlay_rgb.png` — heat overlay on RGB
- `cellprob_logits.npy` — raw logits from `step03_dP_cellprob.npz`
- `cellpose_masks_int32.npy` — instance labels (step04)
- `cellpose_flow_hsv.png` — flow visualization (if available)
- `cellpose_masks_rgb.png` — Cellpose color labels

## Per-cell strict saliency

Pipeline-aligned (**no Otsu**):

1. `cellprob_sigmoid_strict_u8.png` — sigmoid(cellprob) **only inside** the Cellpose mask for that cell (0 outside)
2. `saliency_strict_thr_blur_u8.png` — threshold {SALIENCY_THRESHOLD}, Gaussian blur σ={SALIENCY_BLUR_SIGMA}, re-masked to the cell

Also: `crop_rgb.png`, `cellpose_mask_u8.png`, `meta.json` (bbox, margin={MARGIN}px).

## Stats

- ROIs exported: {n_rois}
- Per-cell folders: {n_cells}

## Regenerate

```bash
cd new_pipeline
export PYTHONPATH="$(pwd):$(pwd)/pipeline:$(pwd)/oral"
python3 oral/export_oral_epithelium_pack.py
```
"""
    (pack_root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symlink-images", action="store_true", help="Symlink TIFFs instead of copying (~75MB)")
    p.add_argument("--skip-dataset", action="store_true", help="Only export cellpose + per-cell")
    p.add_argument("--roi", action="append", default=[], help="Limit to category/stem e.g. healthy/healthy-18-roi2")
    args = p.parse_args()

    pack_root = OUT_PACK
    pack_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_dataset:
        _copy_dataset(pack_root, symlink_images=args.symlink_images)

    rois = discover_rois()
    if args.roi:
        filt = set()
        for item in args.roi:
            cat, _, stem = item.partition("/")
            filt.add((cat, stem))
        rois = [r for r in rois if r in filt]

    total_cells = 0
    for i, (category, stem) in enumerate(rois, 1):
        print(f"[{i}/{len(rois)}] {category}/{stem}")
        try:
            _export_cellpose_roi(category, stem, pack_root)
            n = _export_per_cell_strict(category, stem, pack_root)
            total_cells += n
            print(f"    {n} cells")
        except Exception as e:
            print(f"    FAILED: {e}")

    _write_readme(pack_root, len(rois), total_cells)
    print(f"\nDone: {pack_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
