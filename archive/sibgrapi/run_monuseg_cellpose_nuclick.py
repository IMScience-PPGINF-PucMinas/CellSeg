#!/usr/bin/env python3
"""
MoNuSeg pipeline:
  1) Run Cellpose on every image in monuseg/MoNuSeg2018/Tissue Images (or --tissue-dir).
  2) Export nucleus centroids as NuClick-style {stem}_dots.mat (key 'centers', rows [y, x]).
  3) Run NuClick batch inference (NuClick/run_monuseg_batch.py).

Example:
  python run_monuseg_cellpose_nuclick.py \\
    --tissue-dir "monuseg/MoNuSeg2018/Tissue Images" \\
    --work-dir monuseg_runs/exp1 \\
    --nuclick-weights NuClick/weights/weights-NuClick_Nucleus_MultiScaleResUnet_complexBCEweighted.h5

Download NuClick pretrained weights into NuClick/weights/ before step 3 (see NuClick/README.md).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat
from scipy.ndimage import center_of_mass

ROOT = Path(__file__).resolve().parent
NUCLICK_ROOT = ROOT / "NuClick"

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def ensure_cellpose():
    try:
        from cellpose import models  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Install cellpose in this environment (e.g. pip install cellpose)."
        ) from e


def run_cellpose_mask(img: np.ndarray, model_type: str, diameter: float | None, gpu: bool) -> np.ndarray:
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


def mask_to_centroids_mat(mask: np.ndarray) -> np.ndarray:
    """NuClick expects centers as Nx2, rows [row, col] i.e. [y, x] (see NuClick readImageAndCentroids)."""
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if labels.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    cents = center_of_mass(mask > 0, labels=mask, index=labels)
    return np.asarray(cents, dtype=np.float64)


def phase_cellpose(
    tissue_dir: Path,
    mats_dir: Path,
    cellpose_dir: Path,
    model_type: str,
    diameter: float | None,
    gpu: bool,
    skip_existing: bool,
) -> None:
    ensure_cellpose()
    from cellpose import plot

    mats_dir.mkdir(parents=True, exist_ok=True)
    cellpose_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in tissue_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise SystemExit(f"No images found in {tissue_dir}")

    print(f"[cellpose] Found {len(files)} images in {tissue_dir}")
    for img_path in files:
        stem = img_path.stem
        mat_name = f"{stem}_dots.mat"
        mat_path = mats_dir / mat_name
        if skip_existing and mat_path.exists():
            print(f"  skip (mat exists): {mat_name}")
            continue

        print(f"  {img_path.name}")
        img = np.array(Image.open(img_path))
        if img.ndim == 3 and img.shape[-1] > 3:
            img = img[..., :3]

        masks = run_cellpose_mask(img, model_type=model_type, diameter=diameter, gpu=gpu)
        centers = mask_to_centroids_mat(masks)
        savemat(str(mat_path), {"centers": centers})

        # Visual exports: raw instance IDs as uint16 PNG look "all black" (values 1..N vs 0..65535).
        rgb = plot.mask_rgb(masks)
        Image.fromarray(rgb).save(cellpose_dir / f"{stem}_cellpose_masks_rgb.png")

        img_float = np.asarray(img, dtype=np.float32)
        if img_float.ndim == 2:
            img_float = img_float[..., np.newaxis]
        overlay = plot.mask_overlay(img_float, masks)
        Image.fromarray(overlay).save(cellpose_dir / f"{stem}_cellpose_overlay.png")
        # Instance map for metrics (same as legacy *_cellpose_labels.png; uint16 OK for N<65536)
        Image.fromarray(masks.astype(np.uint16)).save(cellpose_dir / f"{stem}_cellpose_instances.png")


def phase_nuclick(images_dir: Path, mats_dir: Path, save_dir: Path, weights: Path, nuclick_gpu: bool = False) -> None:
    batch_script = NUCLICK_ROOT / "run_monuseg_batch.py"
    if not batch_script.is_file():
        raise FileNotFoundError(f"Missing {batch_script}")

    cmd = [
        sys.executable,
        str(batch_script),
        "--images",
        str(images_dir),
        "--mats",
        str(mats_dir),
        "--save",
        str(save_dir),
        "--weights",
        str(weights),
        "--application",
        "Nucleus",
    ]
    if nuclick_gpu:
        cmd.append("--gpu")
    print("[nuclick]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="MoNuSeg: Cellpose → centroid .mat → NuClick")
    p.add_argument(
        "--tissue-dir",
        type=str,
        default=str(ROOT / "monuseg" / "MoNuSeg2018" / "Tissue Images"),
        help="Folder with MoNuSeg tissue images (default: monuseg/MoNuSeg2018/Tissue Images)",
    )
    p.add_argument(
        "--work-dir",
        type=str,
        default=str(ROOT / "monuseg_runs" / "default"),
        help="Working directory for mats, cellpose debug, nuclick output",
    )
    p.add_argument("--cellpose-model", type=str, default="cyto3", help="Cellpose model type")
    p.add_argument("--diameter", type=float, default=None, help="Cellpose diameter (optional)")
    p.add_argument("--no-gpu", action="store_true", help="Cellpose CPU only")
    p.add_argument(
        "--skip-existing-mats",
        action="store_true",
        help="Skip Cellpose for images that already have *_dots.mat",
    )
    p.add_argument(
        "--nuclick-weights",
        type=str,
        default="",
        help="Path to NuClick .h5 weights file (required unless --cellpose-only)",
    )
    p.add_argument(
        "--cellpose-only",
        action="store_true",
        help="Only run Cellpose and write *_dots.mat; do not run NuClick",
    )
    p.add_argument(
        "--nuclick-only",
        action="store_true",
        help="Only run NuClick (expects mats already in work-dir/mats)",
    )
    p.add_argument(
        "--nuclick-gpu",
        action="store_true",
        help="Run NuClick on GPU (default: CPU; avoids CuDNN mismatch with TF if system CuDNN is older)",
    )
    args = p.parse_args()

    tissue_dir = Path(args.tissue_dir).resolve()
    if not tissue_dir.is_dir():
        raise SystemExit(f"Not a directory: {tissue_dir}")

    work = Path(args.work_dir).resolve()
    mats_dir = work / "mats"
    cellpose_dir = work / "cellpose"
    nuclick_out = work / "nuclick_instances"

    if not args.nuclick_only:
        phase_cellpose(
            tissue_dir=tissue_dir,
            mats_dir=mats_dir,
            cellpose_dir=cellpose_dir,
            model_type=args.cellpose_model,
            diameter=args.diameter,
            gpu=not args.no_gpu,
            skip_existing=args.skip_existing_mats,
        )

    if args.cellpose_only:
        print(f"Done (cellpose only). Mats: {mats_dir}")
        return 0

    wpath = Path(args.nuclick_weights).resolve() if args.nuclick_weights else None
    if not wpath or not wpath.exists():
        raise SystemExit(
            "Provide existing --nuclick-weights path to the .h5 file, or use --cellpose-only.\n"
            "Example after downloading weights:\n"
            "  --nuclick-weights NuClick/weights/weights-NuClick_Nucleus_MultiScaleResUnet_complexBCEweighted.h5"
        )

    phase_nuclick(
        images_dir=tissue_dir,
        mats_dir=mats_dir,
        save_dir=nuclick_out,
        weights=wpath,
        nuclick_gpu=args.nuclick_gpu,
    )
    print(f"Done. NuClick instance maps: {nuclick_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
