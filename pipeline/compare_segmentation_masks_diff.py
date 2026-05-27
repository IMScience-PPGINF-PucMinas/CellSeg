#!/usr/bin/env python3
"""
Compare two **instance label maps** (same shape) and write **difference-only** views.

Typical use: **baseline Cellpose** (e.g. ``step04_masks_uint16.npy`` from
``reproduce_cellpose_pipeline.py``) vs **remixed masks** from
``cellpose_masks_modified_cellprob.py``, which saves ``remix_arrays.npz`` with array
key ``masks``.

Outputs (in ``-o`` directory)
-----------------------------
* ``diff_binary.png`` — white = any disagreement, black = identical labels
* ``diff_rgb_explained.png`` — color-coded reason (see legend in ``diff_stats.txt``)
* ``diff_stats.txt`` — pixel counts and optional per-label notes

Inputs must be **label images** (integer per pixel: 0 = background, 1…N = instances),
not RGB preview PNGs. Pass ``.npy``, ``.npz`` (use ``--mask-*-npz-key``, default ``masks``),
or 16-bit single-channel TIFF.

Examples::

    # Baseline vs cellprob-remix (remix_arrays.npz from cellpose_masks_modified_cellprob.py)
    python compare_segmentation_masks_diff.py \\
        --mask-a cp_flow_out/step04_masks_uint16.npy \\
        --mask-b cellpose_remix_out/remix_arrays.npz \\
        -o compare_out

    python compare_segmentation_masks_diff.py \\
        --mask-a masks_a.tif --mask-b masks_b.tif -o compare_out
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_label_array(path: Path, npz_key: str | None = None) -> "np.ndarray":
    import numpy as np

    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(path)
        key = npz_key if npz_key is not None else "masks"
        if key not in z.files:
            raise ValueError(
                f"{path}: no array {key!r}. Available: {list(z.files)}"
            )
        x = z[key]
    elif suf == ".npy":
        x = np.load(path)
    elif suf in (".tif", ".tiff"):
        try:
            import tifffile

            x = tifffile.imread(path)
        except ImportError:
            from PIL import Image

            x = np.array(Image.open(path))
    elif suf in (".png", ".pgm", ".bmp"):
        from PIL import Image

        im = Image.open(path)
        x = np.array(im)
        if x.ndim == 3:
            raise ValueError(
                f"{path}: expected single-channel label image; got RGB/RGBA. Use .npy or 16-bit grey TIFF."
            )
    else:
        raise ValueError(f"Unsupported extension: {path}")

    if x.dtype == np.float32 or x.dtype == np.float64:
        x = np.round(x).astype(np.int64)
    else:
        x = x.astype(np.int64)
    return x


def build_difference_maps(
    a: "np.ndarray", b: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray", dict]:
    """
    Returns
    -------
    diff_binary : uint8 [H,W]  255 where a!=b else 0
    diff_rgb    : uint8 [H,W,3] color-coded
    stats       : dict of counts
    """
    import numpy as np

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    eq = a == b
    both_bg = (a == 0) & (b == 0)
    only_a = (a > 0) & (b == 0)
    only_b = (a == 0) & (b > 0)
    both_fg_diff = (a > 0) & (b > 0) & (a != b)

    diff_any = ~eq
    diff_binary = (diff_any.astype(np.uint8)) * 255

    rgb = np.zeros((*a.shape, 3), dtype=np.uint8)
    # agreement: near-black (include both-bg as "match")
    rgb[eq] = (28, 28, 28)
    # only in A (present in baseline, absent in other)
    rgb[only_a] = (230, 60, 60)
    # only in B
    rgb[only_b] = (60, 90, 230)
    # both foreground but different instance id / boundary disagreement
    rgb[both_fg_diff] = (240, 220, 50)

    stats = {
        "pixels_total": int(a.size),
        "pixels_match": int(eq.sum()),
        "pixels_differ": int(diff_any.sum()),
        "both_background": int(both_bg.sum()),
        "only_in_a_foreground": int(only_a.sum()),
        "only_in_b_foreground": int(only_b.sum()),
        "both_foreground_different_label": int(both_fg_diff.sum()),
    }
    return diff_binary, rgb, stats


def main() -> int:
    import numpy as np

    p = argparse.ArgumentParser(
        description="Compare two label maps; save difference-only masks + stats."
    )
    p.add_argument(
        "--mask-a",
        type=str,
        required=True,
        help="Label map A (e.g. Cellpose / without SICLE)",
    )
    p.add_argument(
        "--mask-b",
        type=str,
        required=True,
        help="Label map B: often remix_arrays.npz from cellpose_masks_modified_cellprob.py",
    )
    p.add_argument(
        "--mask-a-npz-key",
        type=str,
        default=None,
        help="If --mask-a is .npz, array name (default: masks)",
    )
    p.add_argument(
        "--mask-b-npz-key",
        type=str,
        default=None,
        help="If --mask-b is .npz, array name (default: masks — matches remix_arrays.npz)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="./mask_compare_out",
        help="Output directory",
    )
    p.add_argument(
        "--also-save-diff-only-rgb",
        action="store_true",
        help="Also save diff_rgb_explained.png but with matching pixels set to black (pure difference view).",
    )
    args = p.parse_args()

    pa, pb = Path(args.mask_a), Path(args.mask_b)
    if not pa.is_file() or not pb.is_file():
        print("mask-a or mask-b not found", file=__import__("sys").stderr)
        return 1

    key_a = args.mask_a_npz_key or "masks"
    key_b = args.mask_b_npz_key or "masks"
    a = load_label_array(pa, npz_key=key_a if pa.suffix.lower() == ".npz" else None)
    b = load_label_array(pb, npz_key=key_b if pb.suffix.lower() == ".npz" else None)
    diff_bin, diff_rgb, stats = build_difference_maps(a, b)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(diff_bin).save(out / "diff_binary.png")
        Image.fromarray(diff_rgb).save(out / "diff_rgb_explained.png")
    else:
        imwrite(out / "diff_binary.png", diff_bin)
        imwrite(out / "diff_rgb_explained.png", diff_rgb)

    if args.also_save_diff_only_rgb:
        rgb2 = diff_rgb.copy()
        m = (a == b).astype(bool)
        rgb2[m] = 0
        try:
            from imageio import imwrite
        except ImportError:
            from PIL import Image

            Image.fromarray(rgb2).save(out / "diff_rgb_differences_only.png")
        else:
            imwrite(out / "diff_rgb_differences_only.png", rgb2)

    legend = """
    Legend (diff_rgb_explained.png)
    -------------------------------
    Dark gray   : same label in A and B (including both background)
    Red-ish     : foreground in A only (missing in B)
    Blue-ish    : foreground in B only (missing in A)
    Yellow      : both maps have a non-zero label but labels differ (split/merge/shift)
    """
    lines = [
        f"mask_a: {pa}",
        f"mask_b: {pb}",
        "",
        "Pixel counts",
        "------------",
    ]
    for k, v in stats.items():
        lines.append(f"  {k}: {v}")
    frac = 100.0 * stats["pixels_differ"] / max(1, stats["pixels_total"])
    lines.append(f"  fraction_differing_pixels: {frac:.4f}%")
    lines.append("")
    lines.append(legend.strip())

    (out / "diff_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'diff_binary.png'}")
    print(f"Wrote {out / 'diff_rgb_explained.png'}")
    print(f"Wrote {out / 'diff_stats.txt'}")
    if args.also_save_diff_only_rgb:
        print(f"Wrote {out / 'diff_rgb_differences_only.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
