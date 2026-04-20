#!/usr/bin/env python3
"""
Compute Cellpose **cell probability** (last network channel; trained as logits) and save a heatmap PNG.

Usage::

    PYTHONPATH=/path/to/cellpose python cellprob_heatmap.py image.tif -o cellprob_heat.png --gpu

Optional **--sigmoid** maps logits to (0,1) before coloring; omit to color raw logits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLPOSE_DIR = ROOT / "cellpose"
if str(CELLPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(CELLPOSE_DIR))


def cellprob_to_heatmap_u8(
    cellprob: "np.ndarray",
    *,
    use_sigmoid: bool,
    colormap: str = "turbo",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> "np.ndarray":
    """Return uint8 RGB heatmap [Ly, Lx, 3]."""
    import numpy as np
    import cv2

    x = np.asarray(cellprob, dtype=np.float64)
    if use_sigmoid:
        x = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    lo, hi = np.percentile(x, p_low), np.percentile(x, p_high)
    if hi <= lo + 1e-12:
        hi = lo + 1.0
    norm = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)

    name = colormap.upper()
    cmap_id = getattr(cv2, f"COLORMAP_{name}", None)
    if cmap_id is None:
        cmap_id = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    bgr = cv2.applyColorMap(u8, cmap_id)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description="Cellpose cellprob → heatmap PNG")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("-o", "--output", default="cellprob_heatmap.png", help="Output PNG")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--sigmoid",
        action="store_true",
        help="Apply sigmoid (logits → probability-like [0,1]) before coloring",
    )
    parser.add_argument("--colormap", default="turbo", help="OpenCV colormap name (e.g. turbo, jet)")
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--pretrained-model", default="cpsam")
    parser.add_argument("--save-npy", type=str, default=None, help="Optional path to save cellprob as .npy")
    args = parser.parse_args()

    from cellpose import io
    from cellpose.models import CellposeModel

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"not found: {img_path}", file=sys.stderr)
        return 1

    img = io.imread(str(img_path))
    model = CellposeModel(gpu=args.gpu, pretrained_model=args.pretrained_model)

    # compute_masks=False is faster if you only need flows; we need cellprob
    _masks, flows, _styles = model.eval(
        img,
        diameter=args.diameter,
        compute_masks=False,
    )
    # Single image: flows = [circ, dP, cellprob]
    cellprob = flows[2]
    cellprob = np.asarray(cellprob).squeeze()

    heat_rgb = cellprob_to_heatmap_u8(
        cellprob,
        use_sigmoid=args.sigmoid,
        colormap=args.colormap,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(heat_rgb).save(out)
    else:
        imwrite(out, heat_rgb)

    if args.save_npy:
        np.save(args.save_npy, cellprob.astype(np.float32))
        print(f"saved cellprob array {cellprob.shape} → {args.save_npy}")

    print(f"saved heatmap → {out}")
    print(f"  cellprob shape {cellprob.shape}, dtype {cellprob.dtype}")
    print(f"  raw range [{cellprob.min():.4f}, {cellprob.max():.4f}]")
    if args.sigmoid:
        p = 1.0 / (1.0 + np.exp(-np.clip(cellprob, -50, 50)))
        print(f"  after sigmoid range [{p.min():.4f}, {p.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
