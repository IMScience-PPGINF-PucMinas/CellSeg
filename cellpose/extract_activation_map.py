#!/usr/bin/env python3
"""
Save a merged activation-map RGB preview using the same preprocessing as Cellpose ``eval``.

Example::

    PYTHONPATH=/path/to/cellpose python extract_activation_map.py image.png -o act.png --layer out

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract CP-SAM intermediate activations (neck or last conv) and save RGB preview."
    )
    parser.add_argument("image", type=str, help="Path to PNG/TIF/etc.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="activation_rgb.png",
        help="Output PNG path (RGB preview).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="out",
        choices=("neck", "out", "last_conv"),
        help="neck: encoder.neck (256 ch); out: last 1x1 conv before upsampling.",
    )
    parser.add_argument("--diameter", type=float, default=None, help="Same as Cellpose eval (optional).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to cpsam weights (optional).")
    parser.add_argument(
        "--heatmap-out",
        type=str,
        default=None,
        help="Path for false-color heatmap PNG (default: <output_stem>_heatmap.png).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        help="OpenCV colormap name, e.g. turbo, jet, viridis (if available).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Heatmap only (no blend with the original image).",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Heatmap blend weight when overlaying on the image (0–1).",
    )
    args = parser.parse_args()

    from cellpose import io
    from cellpose.activation_maps import activation_to_heatmap_rgb
    from cellpose.models import CellposeModel

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"file not found: {img_path}", file=sys.stderr)
        return 1

    img = io.imread(str(img_path))
    model = CellposeModel(gpu=args.gpu, pretrained_model=args.pretrained_model or "cpsam")
    act, act_rgb = model.extract_activation_maps(
        img, layer=args.layer, diameter=args.diameter, return_rgb=True
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _to_uint8_rgb(im: np.ndarray) -> np.ndarray:
        x = np.asarray(im, dtype=np.float32)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[-1] > 3:
            x = x[..., :3]
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        if hi > lo + 1e-8:
            x = (x - lo) / (hi - lo)
        else:
            x = np.zeros_like(x)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    overlay = None if args.no_overlay else _to_uint8_rgb(img)
    heatmap = activation_to_heatmap_rgb(
        act,
        colormap=args.colormap,
        overlay=overlay,
        overlay_alpha=args.overlay_alpha,
    )
    heat_path = Path(args.heatmap_out) if args.heatmap_out else out.with_name(f"{out.stem}_heatmap.png")

    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(act_rgb).save(out)
        Image.fromarray(heatmap).save(heat_path)
    else:
        imwrite(out, act_rgb)
        imwrite(heat_path, heatmap)

    np.save(out.with_suffix(".npy"), act)
    print(f"wrote {out}, {heat_path}, and {out.with_suffix('.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
