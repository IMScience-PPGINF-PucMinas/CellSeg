#!/usr/bin/env python3
"""
Either **run Cellpose** on each image, or **load existing segmentations** from Cellpose ``*_seg.npy``
files (GUI “Save masks”, dict with ``masks``, ``outlines``, …) and continue with the same exports.

Writes per image:
  - ``*_cellpose_masks_rgb.png``, ``*_cellpose_overlay.png``, ``*_cellpose_instances.png``
  - ``*_nuclei_only.png`` (and copy under ``nucleos/``)

When ``--from-npy`` is set, looks for ``<npy-dir>/<stem><suffix>`` (default suffix ``_seg.npy``).

When ``--bg-dir`` points at FUNDO hand-marked images (default: ``dataset_gio/dataset_bg``),
each slide is paired by stem (e.g. ``12122 severa.tif`` ↔ ``12122 severaFundo.tif``,
``Mucosanormalnovo.tif`` ↔ ``Mucosanormalnovo2FUNDO...``). Pixels where |RGB_orig − RGB_FUNDO|
exceeds ``--diff-threshold`` (L1 sum) are treated as drawn exclusions: label ids are cleared
there after Cellpose so they no longer appear in overlays or ``*_instances.png``.

FUNDO and the source image should be **aligned**; JPEG FUNDO files can differ globally from
TIFF — then raise ``--diff-threshold`` or use a lossless FUNDO export.

Use ``--no-bg-exclude`` to skip FUNDO masking entirely (Cellpose / npy masks unchanged).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "dataset_gio"
DEFAULT_BG = ROOT / "dataset_gio" / "dataset_bg"

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _strip_trailing_extensions(filename: str) -> str:
    """Remove .jpeg/.jpg/.tif/... repeatedly (handles ``.jpg.jpeg``)."""
    n = filename
    while True:
        m = re.match(r"^(.+)\.(jpeg|jpg|tif|tiff|png|bmp)$", n, re.I)
        if not m:
            break
        n = m.group(1)
    return n


def _marker_stem_key(filename: str) -> str:
    """Stem key before FUNDO suffix, e.g. ``12122 severaFundo.tif`` → ``12122 severa``."""
    base = _strip_trailing_extensions(filename)
    base = re.sub(r"(FUNDO|Fundo|fundo)\d*\s*$", "", base, flags=re.I)
    return base.strip()


def _stem_keys_match(input_stem: str, marker_key: str) -> bool:
    """Match ``Mucosanormalnovo.tif`` to ``Mucosanormalnovo2FUNDO...``."""

    def compact(s: str) -> str:
        return re.sub(r"\s+", "", s).lower()

    a, b = compact(input_stem), compact(marker_key)
    if a == b:
        return True
    # Allow trailing digit variant on marker side (e.g. ...novo2)
    if a == b.rstrip("0123456789"):
        return True
    if b == a.rstrip("0123456789"):
        return True
    return False


def find_marker_image(input_path: Path, bg_dir: Path) -> Path | None:
    """Return FUNDO marker file for ``input_path``, or None."""
    if not bg_dir.is_dir():
        return None
    stem = input_path.stem
    candidates: list[Path] = []
    for p in bg_dir.iterdir():
        if not p.is_file():
            continue
        if "fundo" not in p.name.lower():
            continue
        key = _marker_stem_key(p.name)
        if _stem_keys_match(stem, key):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.name))
    return candidates[0]


def load_image_rgb_u8(path: Path) -> np.ndarray:
    im = np.array(Image.open(path))
    if im.ndim == 2:
        return np.stack([im, im, im], axis=-1)
    if im.shape[-1] > 3:
        return im[..., :3]
    return im


def exclusion_mask_diff(
    img_rgb: np.ndarray,
    marker_rgb: np.ndarray,
    diff_threshold: float,
) -> np.ndarray:
    """
    Pixels where the FUNDO image differs from the original (hand highlights).
    Shape (H, W), bool True = exclude from segmentation.
    """
    hi, wi = img_rgb.shape[:2]
    hm, wm = marker_rgb.shape[:2]
    if (hm, wm) != (hi, wi):
        marker_rgb = np.array(
            Image.fromarray(marker_rgb).resize((wi, hi), Image.Resampling.BILINEAR)
        )
    d = np.abs(img_rgb.astype(np.int16) - marker_rgb.astype(np.int16)).sum(axis=-1)
    return d > diff_threshold


def apply_exclusion_to_masks(
    masks: np.ndarray,
    exclude: np.ndarray,
) -> np.ndarray:
    """Set labels to 0 wherever ``exclude`` is True (same shape as masks)."""
    out = np.asarray(masks, dtype=np.int32).copy()
    if exclude.shape != out.shape[:2]:
        raise ValueError(f"exclude {exclude.shape} != masks spatial {out.shape[:2]}")
    out[exclude] = 0
    return out


def load_masks_from_cellpose_seg_npy(npy_path: Path) -> np.ndarray:
    """
    Load instance labels from Cellpose GUI ``*_seg.npy`` (0 = background, >0 = instance id).

    The file is typically a 0-dim object array wrapping a dict with key ``masks`` (H, W).
    Plain 2D arrays are also accepted.
    """
    raw = np.load(npy_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and getattr(raw, "shape", None) == ():
        raw = raw.item()
    if isinstance(raw, dict):
        if "masks" not in raw:
            raise ValueError(f"{npy_path} has no 'masks' key (keys: {list(raw.keys())})")
        m = np.asarray(raw["masks"])
    else:
        m = np.asarray(raw)
    if m.ndim != 2:
        raise ValueError(f"{npy_path}: expected 2D masks, got shape {m.shape}")
    return m.astype(np.int32)


def ensure_mask_shape(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize label map with nearest-neighbor if spatial size differs from the image."""
    if masks.shape[0] == height and masks.shape[1] == width:
        return masks.astype(np.int32)
    from scipy.ndimage import zoom

    zy = height / float(masks.shape[0])
    zx = width / float(masks.shape[1])
    out = zoom(masks.astype(np.float64), (zy, zx), order=0)
    return np.rint(out).astype(np.int32)


def run_cellpose_mask(
    img: np.ndarray,
    model_type: str,
    diameter: float | None,
    gpu: bool,
) -> np.ndarray:
    from cellpose import models

    if img.ndim == 2:
        img_3 = np.stack([img] * 3, axis=-1)
    else:
        img_3 = img
    model = models.CellposeModel(gpu=gpu, model_type=model_type)
    masks, *_ = model.eval(img_3, diameter=diameter, channel_axis=-1)
    if isinstance(masks, list):
        masks = masks[0]
    return np.asarray(masks, dtype=np.int32)


def nuclei_only_image(img: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Keep original values where mask > 0; set everything else to 0 (black)."""
    fg = masks > 0
    out = np.zeros_like(img)
    if img.ndim == 2:
        out[fg] = img[fg]
    else:
        for c in range(img.shape[-1]):
            ch = img[..., c]
            out[..., c] = np.where(fg, ch, 0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cellpose nuclei on dataset_gio → optional FUNDO exclusion → masked PNGs"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder of images (default: dataset_gio)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder (default: <input-dir>/cellpose_nuclei_out)",
    )
    p.add_argument(
        "--bg-dir",
        type=Path,
        default=DEFAULT_BG,
        help="Folder with FUNDO hand-marked images (default: dataset_gio/dataset_bg). "
        "Unused for masking when --no-bg-exclude is set.",
    )
    p.add_argument(
        "--no-bg-exclude",
        action="store_true",
        help="Do not apply FUNDO / background exclusion (ignore --bg-dir and --diff-threshold).",
    )
    p.add_argument(
        "--diff-threshold",
        type=float,
        default=20.0,
        help="L1 RGB sum threshold: pixels with |orig-marker| sum above this are excluded (default: 20). "
        "Ignored with --no-bg-exclude.",
    )
    p.add_argument(
        "--exclude-dilate",
        type=int,
        default=0,
        help="Optional binary dilation iterations on exclusion mask (default: 0)",
    )
    p.add_argument(
        "--save-exclude-debug",
        action="store_true",
        help="Write *_exclude_mask.png (white = excluded region)",
    )
    p.add_argument(
        "--model",
        default="nuclei",
        help="Cellpose model_type (default: nuclei)",
    )
    p.add_argument("--diameter", type=float, default=None, help="Optional Cellpose diameter")
    p.add_argument("--no-gpu", action="store_true", help="Run Cellpose on CPU")
    p.add_argument(
        "--from-npy",
        action="store_true",
        help="Load segmentation from Cellpose *_seg.npy next to each image (see --npy-dir / --npy-suffix)",
    )
    p.add_argument(
        "--npy-dir",
        type=Path,
        default=None,
        help="Folder containing *_seg.npy (default: same as --input-dir)",
    )
    p.add_argument(
        "--npy-suffix",
        default="_seg.npy",
        help="File name suffix after image stem (default: _seg.npy → e.g. 12122 severa_seg.npy)",
    )
    p.add_argument(
        "--skip-missing-npy",
        action="store_true",
        help="With --from-npy: skip images that have no matching .npy instead of exiting",
    )
    args = p.parse_args()

    inp = args.input_dir.resolve()
    npy_dir = (args.npy_dir.resolve() if args.npy_dir is not None else inp)
    bg_dir = args.bg_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else inp / "cellpose_nuclei_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        x for x in inp.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        print(f"No images found in {inp}", file=sys.stderr)
        sys.exit(1)

    from cellpose import plot

    gpu = not args.no_gpu
    use_bg = bg_dir.is_dir() and not args.no_bg_exclude
    if args.from_npy:
        print(f"[from-npy] npy-dir={npy_dir} suffix={args.npy_suffix!r} -> {out_dir}")
    else:
        print(f"[cellpose] model={args.model} gpu={gpu} -> {out_dir}")
    if args.no_bg_exclude:
        print("[exclude] off (--no-bg-exclude)")
    elif use_bg:
        print(f"[exclude] bg-dir={bg_dir} diff_threshold={args.diff_threshold}")
    else:
        print("[exclude] disabled (bg-dir missing or not a directory)")

    for img_path in files:
        print(f"  {img_path.name}")
        img = load_image_rgb_u8(img_path)
        hi, wi = img.shape[:2]

        if args.from_npy:
            npy_path = npy_dir / f"{img_path.stem}{args.npy_suffix}"
            if not npy_path.is_file():
                msg = f"no segmentation file: {npy_path}"
                if args.skip_missing_npy:
                    print(f"    skip ({msg})", file=sys.stderr)
                    continue
                print(f"ERROR: {msg}", file=sys.stderr)
                sys.exit(1)
            masks = load_masks_from_cellpose_seg_npy(npy_path)
            masks = ensure_mask_shape(masks, hi, wi)
            print(f"    masks from {npy_path.name} shape={masks.shape} n_labels={len(np.unique(masks)) - 1}")
        else:
            masks = run_cellpose_mask(
                img,
                model_type=args.model,
                diameter=args.diameter,
                gpu=gpu,
            )

        marker_path = find_marker_image(img_path, bg_dir) if use_bg else None
        if use_bg and marker_path is None:
            print(
                f"    warning: no FUNDO marker for stem {img_path.stem!r} in {bg_dir}",
                file=sys.stderr,
            )
        elif marker_path is not None:
            marker = load_image_rgb_u8(marker_path)
            exclude = exclusion_mask_diff(img, marker, args.diff_threshold)
            if args.exclude_dilate > 0:
                from scipy.ndimage import binary_dilation

                s = np.ones((3, 3), dtype=bool)
                for _ in range(args.exclude_dilate):
                    exclude = binary_dilation(exclude, structure=s)
            n_ex = int(np.count_nonzero(exclude))
            h, w = exclude.shape[:2]
            frac = n_ex / float(h * w)
            print(f"    exclude: {marker_path.name} ({n_ex} px, {100.0 * frac:.1f}% of image)")
            if frac > 0.25:
                print(
                    "    warning: large excluded fraction — if the FUNDO file is JPEG vs TIFF, "
                    "try a higher --diff-threshold (e.g. 60–100) or a lossless FUNDO export.",
                    file=sys.stderr,
                )
            masks = apply_exclusion_to_masks(masks, exclude)
            if args.save_exclude_debug:
                stem = img_path.stem
                dbg = (exclude.astype(np.uint8) * 255)
                Image.fromarray(dbg).save(out_dir / f"{stem}_exclude_mask.png")

        stem = img_path.stem
        Image.fromarray(plot.mask_rgb(masks)).save(
            out_dir / f"{stem}_cellpose_masks_rgb.png"
        )
        img_float = np.asarray(img, dtype=np.float32)
        if img_float.ndim == 2:
            img_float = img_float[..., np.newaxis]
        overlay = plot.mask_overlay(img_float, masks)
        Image.fromarray(overlay).save(out_dir / f"{stem}_cellpose_overlay.png")
        Image.fromarray(masks.astype(np.uint16)).save(
            out_dir / f"{stem}_cellpose_instances.png"
        )

        img_orig = np.array(Image.open(img_path))
        if img_orig.ndim == 3 and img_orig.shape[-1] > 3:
            img_orig = img_orig[..., :3]
        only = nuclei_only_image(img_orig, masks)
        if only.dtype != np.uint8:
            only = np.clip(only, 0, 255).astype(np.uint8)
        Image.fromarray(only).save(out_dir / f"{stem}_nuclei_only.png")
        nuclei_sub = out_dir / "nucleos"
        nuclei_sub.mkdir(parents=True, exist_ok=True)
        Image.fromarray(only).save(nuclei_sub / f"{stem}_nuclei_only.png")

    print("Done.")


if __name__ == "__main__":
    main()
