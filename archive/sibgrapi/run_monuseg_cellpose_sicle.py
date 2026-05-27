#!/usr/bin/env python3
"""
MoNuSeg experiment: Cellpose instance masks → same per-cell crops as ``cellpose_to_idisf_pipeline`` → SICLE.

Foreground/background scribbles for RunSICLE can come from:
  - ``cellpose`` (default): eroded Cellpose mask + border / other cells (same as iDISF pipeline).
  - ``activation``: greyscale scalar from the CP-SAM last conv (``layer=out``), percentile thresholds
    **inside each crop** at the same ``(r0,r1,c0,c1)`` as the pipeline; saved as ``*_activation_grey.png``.

Resolves ``RunSICLE`` via ``SICLE_BIN``, then ``<repo>/SICLE/bin/RunSICLE``, then other defaults in ``find_sicle_binary``.

Examples:
  # SICLE with Cellpose erosion scribbles (MoNuSeg TIFFs)
  python run_monuseg_cellpose_sicle.py --work-dir monuseg_runs/exp_sicle_cp

  # SICLE with activation-derived scribbles (extracts activations with cpsam; masks still from cyto3)
  python run_monuseg_cellpose_sicle.py --scribble-source activation --work-dir monuseg_runs/exp_sicle_act

  # One image, CPU
  python run_monuseg_cellpose_sicle.py --max-images 1 --no-gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "cellpose_to_idisf_pipeline.py"
if not PIPE.is_file():
    raise RuntimeError(f"Expected {PIPE}")

# Import pipeline after potential sys.path tweaks in that module
sys.path.insert(0, str(ROOT))
from cellpose_to_idisf_pipeline import (  # noqa: E402
    load_image,
    process_image,
    run_cellpose,
)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def main() -> int:
    p = argparse.ArgumentParser(description="MoNuSeg: Cellpose crops → SICLE (optional activation scribbles)")
    p.add_argument(
        "--tissue-dir",
        type=Path,
        default=ROOT / "monuseg" / "MoNuSeg2018" / "Tissue Images",
        help="MoNuSeg tissue images folder",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "monuseg_runs" / "exp_sicle",
        help="Output root; each image gets a subfolder <stem>_sicle/",
    )
    p.add_argument("--cellpose-model", default="cyto3", help="Model for **instance masks** and crops (default: cyto3)")
    p.add_argument("--diameter", type=float, default=None, help="Cellpose diameter (optional)")
    p.add_argument("--no-gpu", action="store_true", help="Cellpose CPU only")
    p.add_argument(
        "--scribble-source",
        choices=("cellpose", "activation"),
        default="cellpose",
        help="cellpose: eroded mask scribbles; activation: last-layer scalar map (uses --activation-model, default cpsam)",
    )
    p.add_argument(
        "--activation-model",
        default="cpsam",
        help="Cellpose model used only to extract [H,W,C] activations when --scribble-source activation",
    )
    p.add_argument(
        "--activation-layer",
        default="out",
        choices=("neck", "out", "last_conv"),
        help="Which layer to hook for activations (default: out = last conv before upsampling)",
    )
    p.add_argument("--margin", type=int, default=10, help="Crop margin around each cell (px)")
    p.add_argument("--erosion-fg", type=int, default=1, help="Foreground erosion depth (px)")
    p.add_argument("--erosion-bg", type=int, default=1, help="Other-cell background erosion (px)")
    p.add_argument("--bg-margin", type=int, default=2, help="Border band for BG scribbles (px)")
    p.add_argument("--no-bg-cells", action="store_true", help="Do not mark other nuclei in crop as BG")
    p.add_argument(
        "--sicle-bin",
        default=None,
        help="Path to RunSICLE (else SICLE_BIN or PIPELINE_UOIFT_SICLE/.../RunSICLE)",
    )
    p.add_argument("--sicle-preset", default="irregular", choices=("irregular", "compact"))
    p.add_argument("--sicle-nf", type=int, default=2)
    p.add_argument("--max-images", type=int, default=None, help="Process only first N images (debug)")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if <work-dir>/<stem>_sicle already exists and is non-empty",
    )
    args = p.parse_args()

    tissue_dir = args.tissue_dir.resolve()
    work = args.work_dir.resolve()
    if not tissue_dir.is_dir():
        print(f"Not a directory: {tissue_dir}", file=sys.stderr)
        return 1

    files = sorted(
        x for x in tissue_dir.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS
    )
    if args.max_images is not None:
        files = files[: max(0, args.max_images)]
    if not files:
        print(f"No images in {tissue_dir}", file=sys.stderr)
        return 1

    work.mkdir(parents=True, exist_ok=True)
    gpu = not args.no_gpu
    use_bg_cells = not args.no_bg_cells

    for i, img_path in enumerate(files):
        stem = img_path.stem
        out_dir = work / f"{stem}_sicle"
        if args.skip_existing and out_dir.is_dir() and any(out_dir.iterdir()):
            print(f"[{i+1}/{len(files)}] skip existing: {out_dir}")
            continue

        print(f"[{i+1}/{len(files)}] {img_path.name} → {out_dir}")
        img = load_image(img_path)
        masks = run_cellpose(
            img,
            model_type=args.cellpose_model,
            gpu=gpu,
            diameter=args.diameter,
        )

        process_image(
            img_path,
            out_dir,
            margin=args.margin,
            erosion_fg=args.erosion_fg,
            erosion_bg=args.erosion_bg,
            bg_margin=args.bg_margin,
            use_bg_cells=use_bg_cells,
            cellpose_model=args.cellpose_model,
            cellpose_diameter=args.diameter,
            run_idisf=True,
            segmenter="sicle",
            sicle_bin=args.sicle_bin,
            sicle_nf=args.sicle_nf,
            sicle_preset=args.sicle_preset,
            save_reunited_mosaic=True,
            gpu=gpu,
            masks_precomputed=masks,
            scribble_source=args.scribble_source,
            activation_layer=args.activation_layer,
            activation_model=args.activation_model,
        )

    print(f"Done. Outputs under {work} (one folder per image: <stem>_sicle).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
