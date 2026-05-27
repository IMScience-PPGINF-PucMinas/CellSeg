"""
Batch overlay MoNuSeg XML polygon annotations on matching H&E TIFF images.

Default (this repo): all 51 training XMLs under monuseg/MoNuSeg2018/Annotations
matched to Tissue Images, outputs written under monuseg_runs/exp1/:
  - annotation_overlays/   *_overlay.png
  - annotation_labels/     *_labels.png  (unless --no-mask)

Examples:
  python overlay_monuseg_annotations.py
  python overlay_monuseg_annotations.py --run-dir monuseg_runs/exp2

Legacy explicit paths:
  python overlay_monuseg_annotations.py --tissue PATH --annotations PATH --overlays PATH

Also writes label masks (0 = background, 1..N = instance id) as PNG for convenience.
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TISSUE = PROJECT_ROOT / "monuseg" / "MoNuSeg2018" / "Tissue Images"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "monuseg" / "MoNuSeg2018" / "Annotations"
DEFAULT_RUN_DIR = PROJECT_ROOT / "monuseg_runs" / "exp1"


def parse_regions(xml_path: Path) -> list[np.ndarray]:
    """Return list of (K,2) float arrays of polygon vertices (x,y) per Region."""
    tree = ET.parse(xml_path)
    regions: list[np.ndarray] = []
    for region in tree.findall(".//Region"):
        verts = region.findall("Vertices/Vertex")
        if not verts:
            verts = region.findall("Vertex")
        if not verts:
            continue
        pts = np.zeros((len(verts), 2), dtype=np.float64)
        for i, v in enumerate(verts):
            pts[i, 0] = float(v.attrib["X"])
            pts[i, 1] = float(v.attrib["Y"])
        regions.append(pts)
    return regions


def rasterize_label_mask(
    height: int, width: int, regions: list[np.ndarray]
) -> np.ndarray:
    """Match MATLAB he_to_binary_mask labeling: non-overlapping instance ids."""
    mask = np.zeros((height, width), dtype=np.int32)
    for zz, xy in enumerate(regions, start=1):
        flat = [(float(x), float(y)) for x, y in xy]
        layer = Image.new("L", (width, height), 0)
        ld = ImageDraw.Draw(layer)
        ld.polygon(flat, fill=1)
        poly = np.asarray(layer, dtype=np.int32)
        free = (mask == 0) & (poly > 0)
        mask[free] = zz
    return mask


def colorize_mask(mask: np.ndarray, seed: int | None = 42) -> np.ndarray:
    """RGB visualization: random color per instance id (stable with seed)."""
    rng = random.Random(seed)
    max_id = int(mask.max())
    lut = np.zeros((max_id + 1, 3), dtype=np.float32)
    for i in range(1, max_id + 1):
        lut[i] = [rng.random(), rng.random(), rng.random()]
    rgb = lut[np.clip(mask, 0, max_id)]
    return (rgb * 255).astype(np.uint8)


def overlay_rgb(
    he_rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.35
) -> np.ndarray:
    """Alpha-blend color instance map on H&E (ignore background id 0)."""
    bg = color_mask.sum(axis=2) == 0
    # instance pixels use blended color
    a = np.zeros(he_rgb.shape[:2], dtype=np.float32)
    a[~bg] = alpha
    a = a[..., np.newaxis]
    out = he_rgb.astype(np.float32) * (1 - a) + color_mask.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def find_tiff(tissue_dir: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = tissue_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def process_one(
    xml_path: Path,
    tissue_dir: Path,
    overlay_dir: Path,
    label_dir: Path,
    save_mask: bool,
    alpha: float,
) -> bool:
    stem = xml_path.stem
    tif_path = find_tiff(tissue_dir, stem)
    if tif_path is None:
        print(f"[skip] no TIFF for {stem}")
        return False

    regions = parse_regions(xml_path)
    im = Image.open(tif_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    elif im.mode == "L":
        im = im.convert("RGB")
    elif im.mode not in ("RGB",):
        im = im.convert("RGB")

    w, h = im.size
    he = np.asarray(im)
    mask = rasterize_label_mask(h, w, regions)
    colored = colorize_mask(mask, seed=hash(stem) % (2**31))
    overlay = overlay_rgb(he, colored, alpha=alpha)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(overlay_dir / f"{stem}_overlay.png")

    if save_mask:
        label_dir.mkdir(parents=True, exist_ok=True)
        # 16-bit PNG if needed for >255 nuclei
        max_id = int(mask.max())
        if max_id <= 65535:
            Image.fromarray(mask.astype(np.uint16)).save(
                label_dir / f"{stem}_labels.png"
            )
    print(f"[ok] {stem}  regions={len(regions)}  size={w}x{h}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="MoNuSeg XML + TIFF overlay export")
    ap.add_argument(
        "--tissue",
        type=Path,
        default=None,
        help=f"Tissue .tif folder (default: {DEFAULT_TISSUE})",
    )
    ap.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=f"XML annotations folder (default: {DEFAULT_ANNOTATIONS})",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Experiment folder; writes annotation_overlays/ and annotation_labels/ inside "
            f"(default: {DEFAULT_RUN_DIR} when --overlays/--output omitted)"
        ),
    )
    ap.add_argument(
        "--overlays",
        type=Path,
        default=None,
        help="Folder for *_overlay.png (overrides --run-dir layout if set)",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Folder for *_labels.png (defaults next to overlays)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Legacy: single folder for both overlays and labels",
    )
    ap.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not save *_labels.png instance masks",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay blend strength (0-1)",
    )
    args = ap.parse_args()

    tissue_dir = args.tissue if args.tissue is not None else DEFAULT_TISSUE
    ann_dir = args.annotations if args.annotations is not None else DEFAULT_ANNOTATIONS
    if not tissue_dir.is_dir():
        ap.error(f"Tissue folder not found: {tissue_dir}")
    if not ann_dir.is_dir():
        ap.error(f"Annotations folder not found: {ann_dir}")

    if args.output is not None:
        overlay_dir = args.output
        label_dir = args.labels if args.labels is not None else args.output
    elif args.overlays is not None:
        overlay_dir = args.overlays
        label_dir = args.labels if args.labels is not None else args.overlays
    else:
        run_dir = args.run_dir if args.run_dir is not None else DEFAULT_RUN_DIR
        overlay_dir = run_dir / "annotation_overlays"
        label_dir = (
            args.labels
            if args.labels is not None
            else run_dir / "annotation_labels"
        )

    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        print(f"No XML files in {ann_dir}")
        return

    print(f"Tissue:      {tissue_dir}")
    print(f"Annotations: {ann_dir}  ({len(xmls)} XML)")
    print(f"Overlays ->  {overlay_dir}")
    if not args.no_mask:
        print(f"Labels ->    {label_dir}")

    n_ok = 0
    for xml_path in xmls:
        if process_one(
            xml_path,
            tissue_dir,
            overlay_dir,
            label_dir,
            save_mask=not args.no_mask,
            alpha=args.alpha,
        ):
            n_ok += 1
    print(f"Done. {n_ok}/{len(xmls)} overlays -> {overlay_dir}")
    if not args.no_mask:
        print(f"         labels -> {label_dir}")


if __name__ == "__main__":
    main()
