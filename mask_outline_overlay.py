#!/usr/bin/env python3
"""
Draw **instance borders only** on top of an image (pixels unchanged elsewhere).

Useful for TIFF slices and **SVS** whole-slide images (via OpenSlide), paired with a label mask
(``.npy`` / ``.npz`` / single-channel TIFF) from Cellpose or ``percell_sicle_cellprob_pipeline``.

If image and mask shapes differ (e.g. mask from a downsampled run), the image is **resized** to
the mask size with bilinear interpolation.

Examples::

    cd doutorado/new_pipeline
    PYTHONPATH=../cellpose python mask_outline_overlay.py \\
        --image ../data/my_slice.tif \\
        --masks ./cp_flow_out/step04_masks_uint16.npy \\
        -o ./viz/step04_outline.png

    # Whole slide: pick pyramid level (0 = largest; higher = smaller)
    PYTHONPATH=../cellpose python mask_outline_overlay.py \\
        --image ../slides/GR07-1.svs --svs-level 2 \\
        --masks ./cp_flow_out/step04_masks_uint16.npy \\
        -o ./viz/svs_outline.png

    PYTHONPATH=../cellpose python mask_outline_overlay.py \\
        --masks remix.npz --masks-npz-key masks --image slice.tif -o out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_CELLPOSE_DIR = _REPO_ROOT / "cellpose"
for _d in (_PKG_DIR, _CELLPOSE_DIR):
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)


def _outline_only_overlay(
    img: "np.ndarray",
    masks: "np.ndarray",
    *,
    border_color_rgb: tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 1,
) -> "np.ndarray":
    import cv2
    import numpy as np
    from cellpose import utils

    base = np.asarray(img[..., :3], dtype=np.uint8).copy()
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError("expected RGB image [H,W,3]")
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


def _to_uint8_rgb(img: "np.ndarray") -> "np.ndarray":
    """Normalize to H×W×3 uint8 for saving."""
    import numpy as np

    x = np.asarray(img)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.ndim == 3 and x.shape[-1] > 3:
        x = x[..., :3]
    if x.dtype == np.uint8:
        return x
    v = x.astype(np.float32)
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return v.astype(np.uint8)


def _load_rgb_image(path: Path, svs_level: int) -> "np.ndarray":
    import numpy as np

    ext = path.suffix.lower()
    if ext == ".svs":
        try:
            import openslide
        except ImportError as e:
            raise SystemExit(
                "Reading .svs needs: pip install openslide-python\n"
                "and system libopenslide (e.g. Debian/Ubuntu: apt install python3-openslide openslide-tools)."
            ) from e
        slide = openslide.OpenSlide(str(path))
        level = max(0, min(int(svs_level), slide.level_count - 1))
        w, h = slide.level_dimensions[level]
        rgb = np.array(slide.read_region((0, 0), level, (w, h)).convert("RGB"))
        return rgb

    from cellpose import io

    img = io.imread_2D(str(path))
    if img is None:
        raise RuntimeError(f"cellpose could not read: {path}")
    return _to_uint8_rgb(img)


def _resize_rgb_to_mask(rgb: "np.ndarray", mask_hw: tuple[int, int]) -> "np.ndarray":
    import cv2

    mh, mw = mask_hw
    ih, iw = rgb.shape[0], rgb.shape[1]
    if (ih, iw) == (mh, mw):
        return rgb
    return cv2.resize(rgb, (mw, mh), interpolation=cv2.INTER_LINEAR)


def main() -> int:
    import numpy as np
    from compare_segmentation_masks_diff import load_label_array

    p = argparse.ArgumentParser(description="Border-only mask overlay on TIFF/SVS (and common formats).")
    p.add_argument("--image", type=str, required=True, help="Path to image (.tif, .png, .jpg, … or .svs)")
    p.add_argument("--masks", type=str, required=True, help="Instance label map: .npy, .npz, or grey label TIFF/PNG")
    p.add_argument(
        "--masks-npz-key",
        type=str,
        default=None,
        help="If --masks is .npz, array name (default: masks)",
    )
    p.add_argument("-o", "--out", type=str, required=True, help="Output PNG path")
    p.add_argument(
        "--svs-level",
        type=int,
        default=0,
        help="OpenSlide pyramid level for .svs (0 = full resolution of that level; increase to downsample)",
    )
    p.add_argument("--border-thickness", type=int, default=1, help="Outline thickness in pixels (>=1)")
    p.add_argument(
        "--border-color",
        type=str,
        default="0,255,0",
        help="R,G,B in 0–255 (default 0,255,0 green)",
    )
    args = p.parse_args()

    try:
        br, bg, bb = (int(x.strip()) for x in args.border_color.split(","))
        color = (max(0, min(255, br)), max(0, min(255, bg)), max(0, min(255, bb)))
    except ValueError:
        raise SystemExit("--border-color must be like 0,255,0") from None
    if args.border_thickness < 1:
        raise SystemExit("--border-thickness must be >= 1")

    img_path = Path(args.image)
    mask_path = Path(args.masks)
    if not img_path.is_file():
        raise SystemExit(f"image not found: {img_path}")
    if not mask_path.is_file():
        raise SystemExit(f"masks not found: {mask_path}")

    key = args.masks_npz_key or "masks"
    masks = load_label_array(mask_path, npz_key=key if mask_path.suffix.lower() == ".npz" else None)
    if masks.ndim != 2:
        raise SystemExit(f"masks must be 2D label image, got shape {masks.shape}")

    rgb = _load_rgb_image(img_path, args.svs_level)
    rgb = np.asarray(rgb[..., :3])
    if rgb.dtype != np.uint8:
        rgb = _to_uint8_rgb(rgb)
    rgb = _resize_rgb_to_mask(rgb, (masks.shape[0], masks.shape[1]))

    out = _outline_only_overlay(
        rgb,
        masks,
        border_color_rgb=color,
        border_thickness=args.border_thickness,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from imageio import imwrite

        imwrite(out_path, out)
    except ImportError:
        from PIL import Image

        Image.fromarray(out).save(out_path)

    print(f"Wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
