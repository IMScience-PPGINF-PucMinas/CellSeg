#!/usr/bin/env python3
"""
Per-cell iDISF on Cellpose crops (Oral Epithelium layout).

Uses the same **exclude-other-cells** idea as the SICLE per-cell pipeline:
  - **BG scribbles** = border band of the crop only (not other cells);
  - **other cells** = inconquerable (neutralized in the input image + forced to background
    in the label map + clip to conquest ROI), like SICLE ``--mask``.

Prerequisites under ``--from-dir`` (e.g. ``cp_flow`` from ``reproduce_cellpose_pipeline.py``)::

    step04_masks_uint16.npy

Usage::

    cd new_pipeline
    export PYTHONPATH="$(pwd):$(pwd)/pipeline:$(pwd)/cellpose:.."
    python pipeline/percell_idisf_cellpose_pipeline.py \\
        --from-dir outputs/runs/.../cp_flow -o ./percell_idisf_out \\
        --image path/to/roi.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_DOUTORADO = _REPO_ROOT.parent
_CELLPOSE_DIR = _REPO_ROOT / "cellpose"
_IDISF_PY = _DOUTORADO / "iDISF" / "python3"
for _d in (_PKG_DIR, _CELLPOSE_DIR, _DOUTORADO, _IDISF_PY):
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)

from cellpose_to_idisf_pipeline import (  # noqa: E402
    eroded_foreground_mask,
    mask_to_scribble_coords,
    run_idisf_on_crop,
)
from percell_conquest_viz import (  # noqa: E402
    background_scribble_mask,
    force_idisf_unconquerable_background,
    neutralize_unconquerable_rgb,
    unconquerable_mask,
    write_percell_marker_figure,
)
from percell_sicle_cellprob_pipeline import (  # noqa: E402
    bbox_for_label,
    conquest_roi_mask,
    load_cellprob_masks,
)


def _idisf_object_mask(
    label_img,
    full_mask_crop,
    cell_mask,
    label: int,
    *,
    exclude_other_cells: bool,
    and_cellpose: bool,
) -> "np.ndarray":
    import numpy as np

    obj = np.asarray(label_img, dtype=np.int32) == 1
    if exclude_other_cells:
        roi = conquest_roi_mask(full_mask_crop, label).astype(bool)
        obj &= roi
    if and_cellpose:
        obj &= np.asarray(cell_mask, dtype=bool)
    return obj


def main() -> int:
    import numpy as np
    from cellpose import plot
    from PIL import Image

    p = argparse.ArgumentParser(description="Per-cell iDISF with other-cell exclusion.")
    p.add_argument("--from-dir", type=str, required=True)
    p.add_argument("-o", "--out-dir", type=str, default="./percell_idisf_out")
    p.add_argument("--image", type=str, default=None, help="RGB for overlay (optional)")
    p.add_argument("--margin", type=int, default=4)
    p.add_argument("--min-cell-area", type=int, default=128)
    p.add_argument("--erosion-fg", type=int, default=1)
    p.add_argument("--erosion-bg", type=int, default=1)
    p.add_argument("--bg-margin", type=int, default=2)
    p.add_argument("--idisf-n0", type=int, default=1000)
    p.add_argument("--idisf-iterations", type=int, default=6)
    p.add_argument("--idisf-f", type=int, default=4)
    p.add_argument("--idisf-c1", type=float, default=0.7)
    p.add_argument("--idisf-c2", type=float, default=0.8)
    p.add_argument(
        "--no-exclude-other-cells",
        action="store_true",
        help="Only border-band background scribbles; no other-cell BG; no conquest ROI clip.",
    )
    p.add_argument(
        "--disable-and-merge",
        action="store_true",
        help="Paste raw iDISF object in bbox (still clipped to conquest ROI when exclusion is on).",
    )
    args = p.parse_args()

    from_dir = Path(args.from_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, masks, _ = load_cellprob_masks(from_dir)
    h, w = masks.shape
    exclude_other = not args.no_exclude_other_cells

    img_rgb = None
    if args.image:
        img = np.asarray(Image.open(args.image).convert("RGB"))
        if img.shape[0] != h or img.shape[1] != w:
            import cv2

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgb = img

    labels = sorted(int(x) for x in np.unique(masks) if int(x) > 0)
    merged = np.zeros((h, w), dtype=np.int32)
    meta: list[str] = []
    percell_dir = out_dir / "percell_cell_outputs"
    percell_dir.mkdir(parents=True, exist_ok=True)

    for lab in labels:
        r0, r1, c0, c1 = bbox_for_label(masks, lab, args.margin, h, w)
        full_crop = masks[r0:r1, c0:c1]
        crop_m = (full_crop == lab).astype(np.uint8)
        ch, cw = crop_m.shape
        area = int(crop_m.sum())
        name = f"cell_{lab:05d}"
        crop_input = None if img_rgb is None else img_rgb[r0:r1, c0:c1]

        if area < args.min_cell_area:
            merged[r0:r1, c0:c1][crop_m.astype(bool)] = lab
            meta.append(f"label {lab}: area={area} < min, kept Cellpose")
            continue

        fg_mask = eroded_foreground_mask(crop_m, args.erosion_fg)
        unconq = unconquerable_mask(full_crop, lab) if exclude_other else None
        bg_mask = background_scribble_mask(
            ch,
            cw,
            full_crop,
            lab,
            border_px=args.bg_margin,
            use_unconquerable=exclude_other,
            erosion_bg_pixels=args.erosion_bg,
        )
        if exclude_other and unconq is not None:
            import numpy as np

            u = np.asarray(unconq, dtype=bool)
            bg_mask = (np.asarray(bg_mask, dtype=bool) & ~u).astype(np.uint8)
        fg_coords = mask_to_scribble_coords(fg_mask)
        bg_coords = mask_to_scribble_coords(bg_mask)
        if not fg_coords:
            rows, cols = np.where(crop_m > 0)
            if rows.size:
                fg_coords = [(int(cols.mean()), int(rows.mean()))]
        if not bg_coords:
            bg_coords = [(0, 0), (cw - 1, 0), (0, ch - 1), (cw - 1, ch - 1)]

        if not fg_coords or not bg_coords:
            merged[r0:r1, c0:c1][crop_m.astype(bool)] = lab
            meta.append(f"label {lab}: missing scribbles, kept Cellpose")
            continue

        if crop_input is None:
            import cv2

            gray = np.zeros((ch, cw), dtype=np.uint8)
            crop_input = np.stack([gray, gray, gray], axis=-1)

        idisf_input = crop_input
        if exclude_other and unconq is not None:
            idisf_input = neutralize_unconquerable_rgb(
                crop_input, unconq, full_mask_crop=full_crop
            )

        try:
            label_img, _border = run_idisf_on_crop(
                idisf_input,
                fg_coords,
                bg_coords,
                n0=args.idisf_n0,
                iterations=args.idisf_iterations,
                f=args.idisf_f,
                c1=args.idisf_c1,
                c2=args.idisf_c2,
            )
            if exclude_other and unconq is not None:
                label_img = force_idisf_unconquerable_background(label_img, unconq)
        except Exception as e:
            merged[r0:r1, c0:c1][crop_m.astype(bool)] = lab
            meta.append(f"label {lab}: iDISF failed ({e}), kept Cellpose")
            cell_dir = percell_dir / name
            write_percell_marker_figure(
                cell_dir,
                crop_input,
                fg_mask=fg_mask,
                bg_mask=bg_mask,
                ignored_mask=unconq,
                fg_coords=fg_coords,
                bg_coords=bg_coords,
            )
            continue

        obj = _idisf_object_mask(
            label_img,
            full_crop,
            crop_m,
            lab,
            exclude_other_cells=exclude_other,
            and_cellpose=not args.disable_and_merge,
        )
        if not obj.any():
            obj = crop_m.astype(bool)
            meta.append(f"label {lab}: empty iDISF FG, kept Cellpose")
        else:
            meta.append(f"label {lab}: iDISF ok exclude_other={exclude_other} merge={'raw' if args.disable_and_merge else 'and'}")

        merged[r0:r1, c0:c1][obj] = lab

        cell_dir = percell_dir / name
        cell_dir.mkdir(parents=True, exist_ok=True)
        if crop_input is not None:
            Image.fromarray(np.asarray(crop_input, dtype=np.uint8)).save(cell_dir / "input_image.png")
        Image.fromarray((np.clip(label_img, 0, 2) * 127).astype(np.uint8), mode="L").save(
            cell_dir / "idisf_label_u8.png"
        )
        Image.fromarray((obj.astype(np.uint8) * 255), mode="L").save(cell_dir / "output_in_cell.png")
        write_percell_marker_figure(
            cell_dir,
            crop_input,
            fg_mask=fg_mask,
            bg_mask=bg_mask,
            ignored_mask=unconq,
            fg_coords=fg_coords,
            bg_coords=bg_coords,
        )
        if exclude_other:
            Image.fromarray(conquest_roi_mask(full_crop, lab) * 255, mode="L").save(
                cell_dir / "conquest_roi_mask.png"
            )

    np.save(out_dir / "merged_percell_idisf_masks_int32.npy", merged)
    Image.fromarray(plot.mask_rgb(merged)).save(out_dir / "merged_percell_idisf_masks_rgb.png")
    if img_rgb is not None:
        from cellpose import plot as cp_plot

        ov = cp_plot.mask_overlay(img_rgb.astype(np.float32), merged)
        Image.fromarray(ov).save(out_dir / "merged_percell_idisf_overlay.png")

    (out_dir / "percell_idisf_log.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'merged_percell_idisf_masks_int32.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
