#!/usr/bin/env python3
"""
Download and convert nuclei delineation benchmarks into new_pipeline/data layout.

Target layout (same as IHC_TMA_dataset):
  data/<dataset>/
    images/<sample_id>.png
    masks/<sample_id>.npy   # int32 instance map (0=bg, 1..N=nucleus id)
    README.md

Datasets (all instance/delineation focused):
  - monuseg   : MoNuSeg 2018 (H&E, XML polygons → instance masks)
  - dsb2018   : Data Science Bowl 2018 / BBBC038 (multi-modality nuclei)
  - pannuke   : PanNuke (H&E patches, instance + type labels in mask dict)
  - consep    : CoNSeP (H&E tiles, .mat inst_map)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data"

# Existing local sources (repo root = doutorado/)
DOUTORADO = REPO.parent
MONUSEG_SRC = DOUTORADO / "monuseg" / "MoNuSeg2018"
DSB_NPZ_SRC = DOUTORADO / "self_adapt" / "data" / "DSB2018_n0"

DSB_TRAIN_ZIP_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
CONSEP_ZIP_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip"
PANNuke_HF_REPO = "RationAI/PanNuke"
PANNuke_FOLD_URLS = {
    1: "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
    2: "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
    3: "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip",
}


def _save_pair(out_root: Path, sample_id: str, image_rgb: np.ndarray, inst: np.ndarray) -> None:
    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb.astype(np.uint8)).save(img_dir / f"{sample_id}.png")
    np.save(mask_dir / f"{sample_id}.npy", inst.astype(np.int32))


def _tiff_to_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        if im.mode in ("I;16", "I"):
            arr = np.asarray(im, dtype=np.float64)
            lo, hi = float(arr.min()), float(arr.max())
            arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
            im = Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L").convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def _parse_monuseg_regions(xml_path: Path) -> list[np.ndarray]:
    tree = ET.parse(xml_path)
    regions: list[np.ndarray] = []
    for region in tree.findall(".//Region"):
        verts = region.findall("Vertices/Vertex") or region.findall("Vertex")
        if not verts:
            continue
        pts = np.array([(float(v.attrib["X"]), float(v.attrib["Y"])) for v in verts], dtype=np.float64)
        regions.append(pts)
    return regions


def _rasterize_regions(regions: list[np.ndarray], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.int32)
    for zz, xy in enumerate(regions, start=1):
        layer = Image.new("L", (width, height), 0)
        ImageDraw.Draw(layer).polygon([(float(x), float(y)) for x, y in xy], fill=1)
        poly = np.asarray(layer, dtype=np.int32)
        free = (mask == 0) & (poly > 0)
        mask[free] = zz
    return mask


def _remap_label(pred: np.ndarray) -> np.ndarray:
    ids = [int(x) for x in np.unique(pred) if x != 0]
    out = np.zeros(pred.shape, dtype=np.int32)
    for idx, inst_id in enumerate(ids, start=1):
        out[pred == inst_id] = idx
    return out


def prepare_monuseg(dst: Path, src: Path = MONUSEG_SRC) -> int:
    tissue = src / "Tissue Images"
    ann = src / "Annotations"
    if not tissue.is_dir() or not ann.is_dir():
        raise FileNotFoundError(f"MoNuSeg source missing: {tissue} / {ann}")

    count = 0
    for xml_path in sorted(ann.glob("*.xml")):
        stem = xml_path.stem
        tif = tissue / f"{stem}.tif"
        if not tif.is_file():
            tif = tissue / f"{stem}.tiff"
        if not tif.is_file():
            print(f"  skip {stem}: missing image")
            continue
        rgb = _tiff_to_rgb(tif)
        h, w = rgb.shape[:2]
        regions = _parse_monuseg_regions(xml_path)
        inst = _rasterize_regions(regions, h, w)
        _save_pair(dst, stem, rgb, inst)
        count += 1
    return count


def _merge_dsb_masks(mask_dir: Path, height: int, width: int) -> np.ndarray:
    inst = np.zeros((height, width), dtype=np.int32)
    masks = sorted(mask_dir.glob("*.png"))
    for i, mp in enumerate(masks, start=1):
        m = np.asarray(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        inst[m > 0] = i
    return inst


def prepare_dsb2018(dst: Path, *, use_npz_fallback: bool = True) -> int:
    cache = DATA_ROOT / "_cache" / "dsb2018"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "stage1_train.zip"

    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000:
        print(f"  downloading {DSB_TRAIN_ZIP_URL}")
        import urllib.request

        urllib.request.urlretrieve(DSB_TRAIN_ZIP_URL, zip_path)

    marker = cache / ".extracted"
    if not marker.is_file():
        print("  extracting stage1_train.zip …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache)
        marker.write_text("ok", encoding="utf-8")

    # BBBC layout: <ImageId>/images/<file>.png + masks/*.png (flat under extract root)
    case_dirs = sorted(
        p for p in cache.iterdir()
        if p.is_dir() and (p / "images").is_dir() and (p / "masks").is_dir()
    )

    count = 0
    for case_dir in case_dirs:
        img_dir = case_dir / "images"
        mask_dir = case_dir / "masks"
        imgs = sorted(img_dir.glob("*.png"))
        if not imgs:
            continue
        rgb = np.asarray(Image.open(imgs[0]).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        inst = _merge_dsb_masks(mask_dir, h, w)
        _save_pair(dst, case_dir.name, rgb, inst)
        count += 1

    if count == 0 and use_npz_fallback and DSB_NPZ_SRC.is_dir():
        print("  BBBC zip parse failed; falling back to local DSB2018_n0 npz")
        for split, tag in (("train", "train"), ("test", "test")):
            npz_path = DSB_NPZ_SRC / split / f"{split}_data.npz"
            if not npz_path.is_file():
                continue
            blob = np.load(npz_path)
            x_key = "X_train" if split == "train" else "X_test"
            y_key = "Y_train" if split == "train" else "Y_test"
            if x_key not in blob:
                x_key, y_key = "X_val", "Y_val"
            xs, ys = blob[x_key], blob[y_key]
            for i in range(len(xs)):
                img = xs[i]
                if img.ndim == 2:
                    rgb = np.stack([img, img, img], axis=-1)
                    if rgb.dtype != np.uint8:
                        rgb = (np.clip(rgb, 0, 255)).astype(np.uint8)
                else:
                    rgb = img.astype(np.uint8)
                inst = ys[i].astype(np.int32)
                _save_pair(dst, f"{tag}_{i:05d}", rgb, inst)
                count += 1
    return count


def _pannuke_inst_from_layers(mask_layers: list[np.ndarray]) -> np.ndarray:
    inst = np.zeros((256, 256), dtype=np.int32)
    off = 0
    for layer in mask_layers:
        layer_res = _remap_label(layer.astype(np.int32))
        inst = np.where(layer_res != 0, layer_res + off, inst)
        off += int(layer_res.max())
    return _remap_label(inst)


def _pannuke_from_npy_folds(cache: Path, dst: Path) -> int:
    import urllib.request

    count = 0
    for fold, url in PANNuke_FOLD_URLS.items():
        zip_path = cache / f"fold_{fold}.zip"
        if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000:
            print(f"  downloading fold {fold} …")
            urllib.request.urlretrieve(url, zip_path)

        fold_cache = cache / f"fold{fold}"
        marker = fold_cache / ".ready"
        if not marker.is_file():
            print(f"  extracting fold {fold} …")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(fold_cache)
            marker.write_text("ok", encoding="utf-8")

        # Official layout: foldN/images/foldN/images.npy + foldN/masks/foldN/masks.npy
        img_npy = next(fold_cache.rglob("images.npy"))
        mask_npy = next(fold_cache.rglob("masks.npy"))
        images = np.load(img_npy, mmap_mode="r")
        masks = np.load(mask_npy, mmap_mode="r")
        n = len(images)
        for i in range(n):
            rgb = np.asarray(images[i], dtype=np.uint8)
            mask = masks[i]
            layers = [mask[:, :, k].astype(np.int32) for k in range(mask.shape[-1])]
            inst = _pannuke_inst_from_layers(layers)
            _save_pair(dst, f"fold{fold}_{i:05d}", rgb, inst)
            count += 1
            if (i + 1) % 500 == 0:
                print(f"  fold{fold}: {i + 1}/{n}")
    return count


def prepare_pannuke(dst: Path) -> int:
    cache = DATA_ROOT / "_cache" / "pannuke"
    cache.mkdir(parents=True, exist_ok=True)
    return _pannuke_from_npy_folds(cache, dst)


def prepare_consep(dst: Path) -> int:
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise RuntimeError("scipy required for CoNSeP") from e

    cache = DATA_ROOT / "_cache" / "consep"
    cache.mkdir(parents=True, exist_ok=True)

    zip_candidates = [
        cache / "consep_dataset.zip",
        cache / "consep.zip",
        cache / "OpenDataLab___CoNSeP" / "raw" / "consep.zip",
    ]
    zip_path = next((p for p in zip_candidates if p.is_file() and p.stat().st_size > 1_000_000), None)
    if zip_path is None:
        raise RuntimeError(
            "CoNSeP zip not found. Download with:\n"
            "  openxlab login\n"
            "  openxlab dataset download --dataset-repo OpenDataLab/CoNSeP "
            f"--source-path /raw/consep.zip --target-path {cache}\n"
            "Or place consep_dataset.zip / consep.zip under data/_cache/consep/"
        )

    extract_dir = cache / "CoNSeP"
    marker = extract_dir / ".ready"
    if not marker.is_file():
        print(f"  extracting {zip_path.name} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache)
        if not extract_dir.is_dir():
            for p in cache.rglob("Train"):
                extract_dir = p.parent
                break
        marker.write_text("ok", encoding="utf-8")

    train_img = extract_dir / "Train" / "Images"
    train_lab = extract_dir / "Train" / "Labels"
    test_img = extract_dir / "Test" / "Images"
    test_lab = extract_dir / "Test" / "Labels"

    count = 0
    for split, img_dir, lab_dir in (
        ("train", train_img, train_lab),
        ("test", test_img, test_lab),
    ):
        if not img_dir.is_dir() or not lab_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            mat_path = lab_dir / f"{img_path.stem}.mat"
            if not mat_path.is_file():
                continue
            rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            mat = loadmat(mat_path)
            inst = mat["inst_map"].astype(np.int32)
            _save_pair(dst, f"{split}_{img_path.stem}", rgb, inst)
            count += 1
    return count


def write_readme(dst: Path, *, name: str, task: str, n_images: int, source: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(
        f"# {name}\n\n"
        f"- **Task:** {task}\n"
        f"- **Samples:** {n_images}\n"
        f"- **Layout:** `images/*.png`, `masks/*.npy` (int32 instance map)\n"
        f"- **Source:** {source}\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare benchmark delineation datasets")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["monuseg", "dsb2018", "pannuke", "consep"],
        choices=["monuseg", "dsb2018", "pannuke", "consep", "all"],
    )
    args = ap.parse_args()
    selected = {"monuseg", "dsb2018", "pannuke", "consep"} if "all" in args.datasets else set(args.datasets)

    meta = {
        "monuseg": (
            "MoNuSeg 2018",
            "Nuclei instance delineation (H&E, polygon annotations)",
            "monuseg.grand-challenge.org / local monuseg/MoNuSeg2018",
        ),
        "dsb2018": (
            "DSB 2018 / BBBC038",
            "Nuclei instance delineation (multi-modality fluorescence/histology)",
            "https://bbbc.broadinstitute.org/BBBC038",
        ),
        "pannuke": (
            "PanNuke",
            "Nuclei instance delineation + 5-class phenotype (H&E patches)",
            "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/ (via HuggingFace RationAI/PanNuke)",
        ),
        "consep": (
            "CoNSeP",
            "Nuclei instance delineation + 4-class phenotype (H&E colorectal)",
            "OpenDataLab/CoNSeP (https://opendatalab.com/OpenDataLab/CoNSeP)",
        ),
    }

    for key in ("monuseg", "dsb2018", "pannuke", "consep"):
        if key not in selected:
            continue
        name, task, source = meta[key]
        dst = DATA_ROOT / key
        print(f"\n=== {name} → {dst}")
        try:
            if key == "monuseg":
                n = prepare_monuseg(dst)
            elif key == "dsb2018":
                n = prepare_dsb2018(dst)
            elif key == "pannuke":
                n = prepare_pannuke(dst)
            else:
                n = prepare_consep(dst)
            write_readme(dst, name=name, task=task, n_images=n, source=source)
            print(f"  done: {n} samples")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
