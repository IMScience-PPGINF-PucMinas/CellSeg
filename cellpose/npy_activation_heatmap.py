#!/usr/bin/env python3
"""Build a heatmap PNG from a saved activation ``.npy`` (``[Ly, Lx, C]`` or ``[N, Ly, Lx, C]``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="Colormap heatmap from activation .npy")
    p.add_argument("npy", type=str, help="Path to .npy (e.g. act_out.npy)")
    p.add_argument("-o", "--output", default=None, help="Output PNG (default: <npy_stem>_heatmap.png)")
    p.add_argument("--image", type=str, default=None, help="Optional image for underlay (same size or resized)")
    p.add_argument("--colormap", default="turbo")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--overlay-alpha", type=float, default=0.45)
    p.add_argument(
        "--greyscale-out",
        default=None,
        help="Optional path for uint8 greyscale PNG (scalar map, no colormap).",
    )
    p.add_argument(
        "--reduce-mode",
        default="l2",
        choices=("l2", "mean_abs", "max_abs"),
        help="How to collapse channels to one scalar (default: l2).",
    )
    args = p.parse_args()

    path = Path(args.npy)
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    from cellpose.activation_maps import activation_to_greyscale_u8, activation_to_heatmap_rgb

    act = np.load(path)
    overlay = None
    if not args.no_overlay and args.image:
        from cellpose import io

        im = io.imread(args.image)

        def _to_u8(a: np.ndarray) -> np.ndarray:
            x = np.asarray(a, dtype=np.float32)
            if x.ndim == 2:
                x = np.stack([x, x, x], -1)
            if x.shape[-1] > 3:
                x = x[..., :3]
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            if hi > lo + 1e-8:
                x = (x - lo) / (hi - lo)
            else:
                x = np.zeros_like(x)
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)

        overlay = _to_u8(im)

    heat = activation_to_heatmap_rgb(
        act,
        colormap=args.colormap,
        overlay=overlay,
        overlay_alpha=args.overlay_alpha,
        reduce_mode=args.reduce_mode,
    )
    out = Path(args.output) if args.output else path.with_name(f"{path.stem}_heatmap.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(heat).save(out)
    else:
        imwrite(out, heat)
    print(f"wrote {out}")

    if args.greyscale_out:
        gpath = Path(args.greyscale_out)
        grey = activation_to_greyscale_u8(act, reduce_mode=str(args.reduce_mode))
        gpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            from imageio import imwrite as imwrite_g
        except ImportError:
            from PIL import Image as PILImage

            PILImage.fromarray(grey).save(gpath)
        else:
            imwrite_g(gpath, grey)
        print(f"wrote {gpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
