#!/usr/bin/env python3
"""
Per-cell SICLE on **bounding boxes**, using **cropped cellprob** as saliency (same preprocessing as
the global SICLE step in ``reproduce_cellpose_pipeline.py``):

  sigmoid(cellprob) → uint8 → **Otsu on the crop** → two-piece [0,0.5]∪[0.5,1] normalization → uint8

Each Cellpose instance is processed in its bbox (+ margin); ``run_sicle_on_crop`` (k≈2 superpixels)
uses mask pixels as foreground seeds (optional erosion via ``--fg-erosion-pixels``). The **object**
superpixel is pasted into a full-image label map
only where ``SICLE_fg & (mask==cell_id)``.

Prerequisites (e.g. from ``reproduce_cellpose_pipeline.py``)::

    step03_dP_cellprob.npz   (``cellprob_slice0`` or ``cellprob``)
    step04_masks_uint16.npy

Usage::

    cd doutorado/new_pipeline
    PYTHONPATH=../cellpose python percell_sicle_cellprob_pipeline.py \\
        --from-dir ./cp_flow_out -o ./percell_sicle_out \\
        --image ../path/to/GR07-1.svs_slice1.tiff

``RunOvlayBorders`` (manual or ``--run-ovlay-borders``) needs **PNM** inputs: use the written
``.ppm``/``.pgm`` files, not TIFF/PNG. The **label** image must be integer masks (e.g. merged labels),
not ``merged_percell_sicle_overlay.png``.

With ``--image``, ``merged_percell_sicle_overlay.png`` keeps the **original pixels** and draws only
**instance borders** (default green; see ``--overlay-border-color`` / ``--overlay-border-thickness``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# This file lives in ``doutorado/new_pipeline/``; Cellpose and SICLE live under ``doutorado/``.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_CELLPOSE_DIR = _REPO_ROOT / "cellpose"
for _d in (_PKG_DIR, _CELLPOSE_DIR):
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)


def load_cellprob_masks(from_dir: Path):
    import numpy as np

    npz = from_dir / "step03_dP_cellprob.npz"
    mpath = from_dir / "step04_masks_uint16.npy"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    if not mpath.is_file():
        raise FileNotFoundError(mpath)
    z = np.load(npz)
    if "cellprob_slice0" in z.files:
        cellprob = np.asarray(z["cellprob_slice0"], dtype=np.float32)
    elif "cellprob" in z.files:
        cellprob = np.asarray(z["cellprob"], dtype=np.float32)
        if cellprob.ndim == 3 and cellprob.shape[0] == 1:
            cellprob = cellprob[0]
    else:
        raise SystemExit(f"{npz}: need cellprob_slice0 or cellprob")
    masks = np.load(mpath).astype(np.int32)
    return cellprob, masks


def cellprob_crop_to_saliency_u8(cellprob_crop: "np.ndarray") -> "np.ndarray":
    """Sigmoid + Otsu + two-piece map (same as ``reproduce_cellpose_pipeline`` step 5), on one crop."""
    import cv2
    import numpy as np

    cp = np.asarray(cellprob_crop, dtype=np.float32)
    cp_prob = 1.0 / (1.0 + np.exp(-np.clip(cp, -50.0, 50.0)))
    cp_u8 = (np.clip(cp_prob, 0.0, 1.0) * 255.0).astype(np.uint8)
    otsu_t, _ = cv2.threshold(cp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(otsu_t) / 255.0
    eps = 1e-8
    sal = np.empty_like(cp_prob, dtype=np.float32)
    lo = cp_prob <= t
    hi = ~lo
    sal[lo] = 0.5 * cp_prob[lo] / max(t, eps)
    sal[hi] = 0.5 + 0.5 * (cp_prob[hi] - t) / max(1.0 - t, eps)
    sal = np.clip(sal, 0.0, 1.0)
    return (sal * 255.0).astype(np.uint8)


def apply_saliency_threshold_u8(sal_u8: "np.ndarray", thr01: float) -> "np.ndarray":
    """Threshold saliency in [0,1]: values below threshold become 0."""
    import numpy as np

    s = np.asarray(sal_u8, dtype=np.uint8).copy()
    t = int(round(float(thr01) * 255.0))
    s[s < t] = 0
    return s


def bbox_for_label(
    masks: "np.ndarray", label: int, margin: int, h: int, w: int
) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.where(masks == label)
    if ys.size == 0:
        raise ValueError(f"empty mask for label {label}")
    r0, r1 = int(ys.min()) - margin, int(ys.max()) + 1 + margin
    c0, c1 = int(xs.min()) - margin, int(xs.max()) + 1 + margin
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(h, r1), min(w, c1)
    return r0, r1, c0, c1


def fg_scribble_coords(
    crop_bin: "np.ndarray",
    max_points: int = 400,
    erosion_pixels: int = 0,
) -> list[tuple[int, int]]:
    """(x, y) in crop coordinates for SICLE foreground seeds.

    If ``erosion_pixels`` is 0 (default), seeds come from the full binary mask. If > 0, the mask is
    eroded that many iterations (3×3); if erosion empties the mask, falls back to the uneroded mask.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion

    m = crop_bin.astype(bool)
    if erosion_pixels > 0:
        er = binary_erosion(m, structure=np.ones((3, 3), dtype=bool), iterations=int(erosion_pixels))
        if not er.any():
            er = m
    else:
        er = m
    ys, xs = np.where(er)
    if ys.size == 0:
        return []
    idx = np.linspace(0, ys.size - 1, num=min(max_points, ys.size), dtype=int)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def _outline_only_overlay(
    img: "np.ndarray",
    masks: "np.ndarray",
    *,
    border_color_rgb: tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 1,
) -> "np.ndarray":
    """RGB uint8 image unchanged except on instance borders (Cellpose ``masks_to_outlines``)."""
    import cv2
    import numpy as np
    from cellpose import utils

    base = np.asarray(img[..., :3], dtype=np.uint8).copy()
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError("_outline_only_overlay: expected RGB [H,W,3]")
    m = np.asarray(masks, dtype=np.int32)
    outlines = utils.masks_to_outlines(m).astype(bool)
    if border_thickness > 1:
        k = max(3, 2 * int(border_thickness) - 1)
        ker = np.ones((k, k), dtype=np.uint8)
        outlines = cv2.dilate(outlines.astype(np.uint8), ker, iterations=1).astype(bool)
    r, g, b = border_color_rgb
    base[outlines, 0] = r
    base[outlines, 1] = g
    base[outlines, 2] = b
    return base


def _find_run_ovlay_borders_binary(sicle_bin: Path) -> Path | None:
    p = Path(sicle_bin).resolve().parent / "RunOvlayBorders"
    return p if p.is_file() else None


def _run_run_ovlay_borders(
    ovlay_bin: Path,
    img_path: Path,
    labels_path: Path,
    out_path: Path,
) -> tuple[bool, str]:
    """``RunOvlayBorders`` expects PNM-style inputs (IFT); avoid TIFF/PNG for ``--img``/``--labels``."""
    cmd = [
        str(ovlay_bin),
        "--img",
        str(img_path.resolve()),
        "--labels",
        str(labels_path.resolve()),
        "--out",
        str(out_path.resolve()),
        "--rgb",
        "0,1,0",
        "--thick",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    msg = (proc.stderr or "") + (proc.stdout or "")
    return proc.returncode == 0 and out_path.exists(), msg


def _write_ovlay_labels_pgm(merged: "np.ndarray", path: Path) -> None:
    """8-bit PGM label image (RunOvlayBorders / ift). Requires max instance id ≤ 255."""
    import numpy as np
    from PIL import Image

    mx = int(np.max(merged))
    if mx > 255:
        raise ValueError(
            f"RunOvlayBorders (PGM): max label is {mx} (> 255). "
            "IFT expects 8-bit labels; skip --run-ovlay-borders or remap labels."
        )
    Image.fromarray(np.asarray(merged, dtype=np.uint8), mode="L").save(path)


def _write_percell_debug_outputs(
    cell_dir: Path,
    input_crop_rgb: "np.ndarray | None",
    sal_u8: "np.ndarray",
    output_in_cell: "np.ndarray",
) -> None:
    """Write per-cell debug assets: input image, saliency map, and output-in-cell mask."""
    import numpy as np
    from PIL import Image

    cell_dir.mkdir(parents=True, exist_ok=True)
    if input_crop_rgb is None:
        # Fallback when no --image is provided: visualize saliency as gray RGB.
        input_crop_rgb = np.stack([sal_u8, sal_u8, sal_u8], axis=-1)
    input_u8 = np.asarray(input_crop_rgb[..., :3], dtype=np.uint8)
    sal_u8 = np.asarray(sal_u8, dtype=np.uint8)
    out_u8 = (np.asarray(output_in_cell, dtype=bool).astype(np.uint8) * 255)

    Image.fromarray(input_u8).save(cell_dir / "input_image.png")
    Image.fromarray(sal_u8, mode="L").save(cell_dir / "saliency_map.png")
    Image.fromarray(out_u8, mode="L").save(cell_dir / "output_in_cell.png")


def main() -> int:
    import numpy as np
    from cellpose import plot
    from cellpose_to_idisf_pipeline import (
        SICLE_ADHR_DEFAULT,
        SICLE_ALPHA_DEFAULT,
        SICLE_IRREG_DEFAULT,
        SICLE_MAXITERS_DEFAULT,
        SICLE_N0_DEFAULT,
        SICLE_NF_DEFAULT,
        find_sicle_binary,
        resolve_sicle_path_cost,
        run_sicle_on_crop,
    )

    p = argparse.ArgumentParser(description="Per-bbox SICLE on cellprob saliency → merged label image.")
    p.add_argument("--from-dir", type=str, required=True, help="Folder with step03 npz + step04 masks npy")
    p.add_argument("-o", "--out-dir", type=str, default="./percell_sicle_out")
    p.add_argument("--margin", type=int, default=8, help="Pixels to pad each cell bbox")
    p.add_argument("--min-cell-area", type=int, default=64, help="Skip SICLE for tiny regions; keep Cellpose mask")
    p.add_argument(
        "--fg-erosion-pixels",
        type=int,
        default=0,
        metavar="N",
        help="Erode the cell mask before sampling FG scribbles for SICLE (0 = off, default; try 1 for interior-only seeds)",
    )
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional RGB path; writes merged_percell_sicle_overlay.png (original pixels + colored borders only)",
    )
    p.add_argument(
        "--overlay-border-thickness",
        type=int,
        default=1,
        help="Border line thickness in pixels for merged_percell_sicle_overlay.png (>=1)",
    )
    p.add_argument(
        "--overlay-border-color",
        type=str,
        default="0,255,0",
        help="Overlay border color as R,G,B in 0-255 (default: 0,255,0 green)",
    )
    p.add_argument("--sicle-preset", choices=("irregular", "compact"), default="irregular")
    p.add_argument("--sicle-conn-opt", type=str, default=None)
    p.add_argument("--sicle-crit-opt", type=str, default=None)
    p.add_argument("--sicle-n0", type=int, default=SICLE_N0_DEFAULT)
    p.add_argument("--sicle-nf", type=int, default=SICLE_NF_DEFAULT)
    p.add_argument("--sicle-alpha", type=float, default=SICLE_ALPHA_DEFAULT)
    p.add_argument("--sicle-max-iters", type=int, default=SICLE_MAXITERS_DEFAULT)
    p.add_argument("--sicle-irreg", type=float, default=SICLE_IRREG_DEFAULT)
    p.add_argument("--sicle-adhr", type=int, default=SICLE_ADHR_DEFAULT)
    p.add_argument(
        "--saliency-threshold",
        type=float,
        default=0.3,
        help=(
            "Threshold in [0,1] applied to per-cell saliency before SICLE "
            "(values below threshold are set to 0). Default: 0.3"
        ),
    )
    p.add_argument(
        "--run-ovlay-borders",
        action="store_true",
        help=(
            "After merge, run SICLE RunOvlayBorders on PPM/PGM files only (IFT often rejects TIFF/PNG). "
            "Requires max merged label ≤ 255. Writes merged_percell_sicle_ovlay_*.ppm/pgm."
        ),
    )
    args = p.parse_args()
    try:
        br, bg, bb = (int(x.strip()) for x in args.overlay_border_color.split(","))
        border_color = (max(0, min(255, br)), max(0, min(255, bg)), max(0, min(255, bb)))
    except ValueError:
        raise SystemExit("--overlay-border-color must be like 0,255,0") from None
    if args.overlay_border_thickness < 1:
        raise SystemExit("--overlay-border-thickness must be >= 1")
    if args.fg_erosion_pixels < 0:
        raise SystemExit("--fg-erosion-pixels must be >= 0")
    if not (0.0 <= args.saliency_threshold <= 1.0):
        raise SystemExit("--saliency-threshold must be in [0,1]")

    from_dir = Path(args.from_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    percell_dir = out_dir / "percell_cell_outputs"
    percell_dir.mkdir(parents=True, exist_ok=True)

    cellprob, masks = load_cellprob_masks(from_dir)
    h, w = cellprob.shape
    if masks.shape != (h, w):
        raise SystemExit(f"shape mismatch cellprob {cellprob.shape} vs masks {masks.shape}")

    img_rgb_resized = None
    if args.image:
        from cellpose import io

        img = io.imread(args.image)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[0] != h or img.shape[1] != w:
            import cv2

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgb_resized = np.asarray(img[..., :3], dtype=np.uint8)

    sicle_bin = find_sicle_binary()
    conn_opt, crit_opt = resolve_sicle_path_cost(
        args.sicle_preset, args.sicle_conn_opt, args.sicle_crit_opt
    )

    labels = sorted(int(x) for x in np.unique(masks) if int(x) > 0)
    merged = np.zeros((h, w), dtype=np.int32)
    meta: list[str] = []

    with tempfile.TemporaryDirectory(prefix="percell_sicle_") as tmp:
        tmp_path = Path(tmp)
        for lab in labels:
            r0, r1, c0, c1 = bbox_for_label(masks, lab, args.margin, h, w)
            crop_cp = cellprob[r0:r1, c0:c1]
            crop_m = (masks[r0:r1, c0:c1] == lab).astype(np.uint8)
            sal_u8 = cellprob_crop_to_saliency_u8(crop_cp)
            sal_u8 = apply_saliency_threshold_u8(sal_u8, args.saliency_threshold)
            crop_input = None if img_rgb_resized is None else img_rgb_resized[r0:r1, c0:c1]
            area = int(crop_m.sum())
            name = f"cell_{lab:05d}"
            output_in_cell = crop_m.astype(bool)
            if area < args.min_cell_area:
                merged[r0:r1, c0:c1][output_in_cell] = lab
                meta.append(f"label {lab}: area={area} < min_cell_area, kept Cellpose mask")
                _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)
                continue

            fg = fg_scribble_coords(crop_m.astype(bool), erosion_pixels=args.fg_erosion_pixels)
            if not fg:
                merged[r0:r1, c0:c1][output_in_cell] = lab
                meta.append(f"label {lab}: no fg seeds, kept Cellpose mask")
                _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)
                continue

            try:
                sicle_lbl = run_sicle_on_crop(
                    sal_u8,
                    fg,
                    tmp_path,
                    name,
                    sicle_bin,
                    n0=args.sicle_n0,
                    nf=args.sicle_nf,
                    alpha=args.sicle_alpha,
                    max_iters=args.sicle_max_iters,
                    irreg=args.sicle_irreg,
                    adhr=args.sicle_adhr,
                    conn_opt=conn_opt,
                    crit_opt=crit_opt,
                )
            except Exception as e:
                merged[r0:r1, c0:c1][output_in_cell] = lab
                meta.append(f"label {lab}: SICLE failed ({e}), kept Cellpose mask")
                _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)
                continue

            obj = sicle_lbl == 1
            output_in_cell = obj & crop_m.astype(bool)
            merged[r0:r1, c0:c1][output_in_cell] = lab
            meta.append(
                f"label {lab}: bbox=({r0},{r1},{c0},{c1}) placed_pixels={int(output_in_cell.sum())}"
            )
            _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)

    np.save(out_dir / "merged_percell_sicle_masks_int32.npy", merged)
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(plot.mask_rgb(merged)).save(out_dir / "merged_percell_sicle_masks_rgb.png")
    else:
        imwrite(out_dir / "merged_percell_sicle_masks_rgb.png", plot.mask_rgb(merged))

    (out_dir / "percell_sicle_log.txt").write_text(
        "\n".join(
            [
                f"from_dir: {from_dir.resolve()}",
                f"cells: {len(labels)}",
                f"margin: {args.margin}",
                f"fg_erosion_pixels: {args.fg_erosion_pixels}",
                f"saliency_threshold: {args.saliency_threshold}",
                f"sicle preset: {args.sicle_preset} conn={conn_opt} crit={crit_opt}",
                "",
                *meta,
            ]
        ),
        encoding="utf-8",
    )

    if args.image:
        from PIL import Image
        ov = _outline_only_overlay(
            img_rgb_resized,
            merged,
            border_color_rgb=border_color,
            border_thickness=args.overlay_border_thickness,
        )
        try:
            from imageio import imwrite
        except ImportError:
            Image.fromarray(ov).save(out_dir / "merged_percell_sicle_overlay.png")
        else:
            imwrite(out_dir / "merged_percell_sicle_overlay.png", ov)

    if args.run_ovlay_borders:
        from PIL import Image

        ov_bin = _find_run_ovlay_borders_binary(sicle_bin)
        if ov_bin is None:
            print("RunOvlayBorders not found next to RunSICLE; skipped.")
        else:
            labels_pgm = out_dir / "merged_percell_sicle_ovlay_labels.pgm"
            try:
                _write_ovlay_labels_pgm(merged, labels_pgm)
            except ValueError as e:
                print(e)
            else:
                base_ppm = out_dir / "merged_percell_sicle_ovlay_base.ppm"
                if img_rgb_resized is not None:
                    Image.fromarray(img_rgb_resized).save(base_ppm)
                else:
                    sal = cellprob_crop_to_saliency_u8(cellprob)
                    rgb = np.stack([sal, sal, sal], axis=-1)
                    Image.fromarray(rgb).save(base_ppm)
                out_ov = out_dir / "merged_percell_sicle_ovlay_borders.ppm"
                ok, msg = _run_run_ovlay_borders(ov_bin, base_ppm, labels_pgm, out_ov)
                if ok:
                    print(f"Wrote {out_ov} (RunOvlayBorders)")
                else:
                    print("RunOvlayBorders failed:\n", msg[:1200])

    print(f"Wrote {out_dir / 'merged_percell_sicle_masks_int32.npy'}")
    print(f"Wrote {out_dir / 'merged_percell_sicle_masks_rgb.png'}")
    print(f"Wrote {out_dir / 'percell_sicle_log.txt'}")
    print(f"Wrote per-cell outputs in {percell_dir}")
    if args.image:
        print(f"Wrote {out_dir / 'merged_percell_sicle_overlay.png'}")
    if args.run_ovlay_borders and (out_dir / "merged_percell_sicle_ovlay_borders.ppm").is_file():
        print(f"Wrote {out_dir / 'merged_percell_sicle_ovlay_borders.ppm'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
