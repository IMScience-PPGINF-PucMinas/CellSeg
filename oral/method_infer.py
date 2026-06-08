#!/usr/bin/env python3
"""Thin inference wrappers for multi-method benchmarks."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CELLVIT_ROOT = Path(__file__).resolve().parents[2] / "CellViT"
DEFAULT_CELLVIT_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "cellvit-256" / "model_best.pth"

# SERAPH PathoSAMAdapter defaults (patho_sam_adapter.py)
_PATHOSAM_TILE_SHAPE = (384, 384)
_PATHOSAM_HALO = (64, 64)
_PATHOSAM_CACHE: dict[tuple[str, bool], tuple[object, object]] = {}


def _cellvit_sys_path() -> list[str]:
    # CellViT repo root must precede cell_segmentation/ so ``utils.tools`` resolves.
    return [str(CELLVIT_ROOT), str(CELLVIT_ROOT / "cell_segmentation")]


def run_cellpose(image_path: Path, out_dir: Path, *, gpu: bool = False) -> Path:
    """Run reproduce_cellpose_pipeline; returns step04 masks path."""
    pipe = Path(__file__).resolve().parents[1] / "pipeline"
    cellpose_dir = Path(__file__).resolve().parents[1] / "cellpose"
    for d in (str(pipe), str(cellpose_dir)):
        if d not in sys.path:
            sys.path.insert(0, d)
    from reproduce_cellpose_pipeline import run_pipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(
        image_path,
        out_dir,
        gpu,
        None,
        0.0,
        0.4,
        "cpsam",
        False,
        False,
    )
    out = out_dir / "step04_masks_uint16.npy"
    if not out.is_file():
        raise FileNotFoundError(out)
    return out


def run_cellvit(
    image_path: Path,
    out_dir: Path,
    *,
    checkpoint: Path | None = None,
    gpu: int | None = None,
) -> Path:
    """Run CellViT-256 HV head; returns step04 masks path."""
    import torch
    import albumentations as A
    import cv2

    ckpt = Path(checkpoint or DEFAULT_CELLVIT_CKPT)
    if not ckpt.is_file():
        raise FileNotFoundError(f"CellViT checkpoint missing: {ckpt}")

    def _unflatten_dict(d: dict, sep: str = ".") -> dict:
        output_dict: dict = {}
        for key, value in d.items():
            keys = key.split(sep)
            cur = output_dict
            for k in keys[:-1]:
                cur = cur.setdefault(k, {})
            cur[keys[-1]] = value
        return output_dict

    for p in _cellvit_sys_path():
        if p not in sys.path:
            sys.path.insert(0, p)

    from models.segmentation.cell_segmentation.cellvit import CellViT256, CellViTSAM
    from models.segmentation.cell_segmentation.cellvit_shared import CellViT256Shared, CellViTSAMShared

    out_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() and (gpu is None or gpu >= 0)
    device = f"cuda:{gpu or 0}" if use_cuda else "cpu"

    ckpt_obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    run_conf = _unflatten_dict(ckpt_obj["config"], ".")
    arch = ckpt_obj.get("arch", run_conf.get("model", {}).get("arch", "CellViT256"))
    num_nc = int(run_conf["data"]["num_nuclei_classes"])
    num_tc = int(run_conf["data"]["num_tissue_classes"])

    if arch in ("CellViT256", "CellViT256Shared"):
        cls = CellViT256Shared if arch == "CellViT256Shared" else CellViT256
        model = cls(model256_path=None, num_nuclei_classes=num_nc, num_tissue_classes=num_tc,
                    regression_loss=run_conf["model"].get("regression_loss", False))
    elif arch in ("CellViTSAM", "CellViTSAMShared"):
        shared = bool(run_conf["model"].get("shared_skip_connections", True))
        cls = CellViTSAMShared if shared or arch == "CellViTSAMShared" else CellViTSAM
        model = cls(model_path=None, num_nuclei_classes=num_nc, num_tissue_classes=num_tc,
                    vit_structure=run_conf["model"]["backbone"],
                    regression_loss=run_conf["model"].get("regression_loss", False))
    else:
        raise ValueError(f"Unsupported CellViT arch: {arch}")

    model.load_state_dict(ckpt_obj["model_state_dict"])
    model.eval().to(device)

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h0, w0 = rgb.shape[:2]
    infer_size = 256
    rgb_small = cv2.resize(rgb, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    tf = A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    x = tf(image=rgb_small)["image"]
    tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.forward(tensor)
        else:
            pred = model.forward(tensor)
        inst_t, _types = model.calculate_instance_map(pred, magnification=40)

    inst = inst_t[0].detach().cpu().numpy().astype(np.int32)
    if inst.shape != (h0, w0):
        inst = cv2.resize(inst, (w0, h0), interpolation=cv2.INTER_NEAREST)

    out = out_dir / "step04_masks_uint16.npy"
    np.save(out, np.asarray(inst, dtype=np.uint16))
    return out


def _pathosam_models(device: str = "cpu", *, tiled: bool = True):
    key = (device, tiled)
    if key not in _PATHOSAM_CACHE:
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter

        _PATHOSAM_CACHE[key] = get_predictor_and_segmenter(
            model_type="vit_l_histopathology",
            checkpoint=None,
            device=device,
            segmentation_mode="ais",
            is_tiled=tiled,
        )
    return _PATHOSAM_CACHE[key]


def capture_pathosam_foreground_prob(segmenter) -> np.ndarray | None:
    """
    SERAPH-compatible foreground probability map from AIS state.

    Mirrors ``PathoSAMAdapter._capture_probability_map``:
    ``segmenter.get_state()["foreground"]``, clipped to [0, 1].
    """
    try:
        state = segmenter.get_state()
        foreground = state.get("foreground")
        if foreground is None:
            return None
        prob = np.asarray(foreground, dtype=np.float32)
        if prob.ndim != 2:
            return None
        return np.clip(prob, 0.0, 1.0)
    except Exception:
        return None


def _cleanup_pathosam(predictor, segmenter) -> None:
    """Release per-image micro_sam state (SERAPH ``cleanup_after_segment``)."""
    try:
        if segmenter is not None and hasattr(segmenter, "clear_state"):
            segmenter.clear_state()
    except Exception:
        pass
    try:
        if predictor is not None and hasattr(predictor, "reset_image"):
            predictor.reset_image()
    except Exception:
        pass
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _save_pathosam_saliency(prob: np.ndarray, out_dir: Path) -> Path:
    """Write float32 saliency + uint8 preview PNG."""
    npy_path = out_dir / "step03_pathosam_foreground_prob.npy"
    png_path = out_dir / "step03_pathosam_foreground_prob.png"
    np.save(npy_path, prob.astype(np.float32))
    Image.fromarray((np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)).save(png_path)
    return npy_path


def pathosam_saliency_path(out_dir: Path) -> Path:
    return out_dir / "step03_pathosam_foreground_prob.npy"


def run_pathosam(
    image_path: Path,
    out_dir: Path,
    *,
    device: str = "cpu",
    tiled: bool = True,
    save_saliency: bool = True,
    force: bool = False,
) -> Path:
    """Run PathoSAM (vit_l_histopathology via micro_sam AIS) + foreground saliency map."""
    import torch
    from micro_sam.automatic_segmentation import automatic_instance_segmentation

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "step04_masks_uint16.npy"
    sal_path = pathosam_saliency_path(out_dir)
    if not force and out.is_file() and out.stat().st_size > 0:
        arr = np.load(out)
        if arr.ndim == 2 and int(arr.max()) > 0 and (not save_saliency or sal_path.is_file()):
            return out

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    predictor, segmenter = _pathosam_models(device=device, tiled=tiled)
    ais_kwargs: dict = {
        "predictor": predictor,
        "segmenter": segmenter,
        "input_path": rgb,
        "ndim": 2,
        "verbose": False,
        "return_embeddings": False,
        "batch_size": 1,
    }
    if tiled:
        ais_kwargs["tile_shape"] = _PATHOSAM_TILE_SHAPE
        ais_kwargs["halo"] = _PATHOSAM_HALO

    try:
        instances = automatic_instance_segmentation(**ais_kwargs)
        instances = np.asarray(instances)
        if instances.ndim != 2:
            raise ValueError(f"PathoSAM expected 2D instance map, got shape {instances.shape}")
        np.save(out, instances.astype(np.uint16))

        if save_saliency:
            prob = capture_pathosam_foreground_prob(segmenter)
            if prob is None:
                raise RuntimeError("PathoSAM foreground probability map not available in segmenter state")
            if prob.shape != instances.shape:
                raise ValueError(
                    f"PathoSAM saliency shape {prob.shape} != instance map {instances.shape}"
                )
            _save_pathosam_saliency(prob, out_dir)
    finally:
        _cleanup_pathosam(predictor, segmenter)

    return out


def merge_cellpose_pathosam_seeds(
    cellpose_masks: np.ndarray,
    pathosam_masks: np.ndarray,
    *,
    iou_thresh: float = 0.5,
) -> np.ndarray:
    """
    Union of Cellpose + PathoSAM instance maps as iDISF seeds.

    Keeps all Cellpose labels; adds PathoSAM instances whose max IoU with any
    Cellpose instance is below ``iou_thresh``.
    """
    cp = np.asarray(cellpose_masks, dtype=np.int32)
    ps = np.asarray(pathosam_masks, dtype=np.int32)
    if cp.shape != ps.shape:
        raise ValueError(f"shape mismatch cellpose {cp.shape} vs pathosam {ps.shape}")
    merged = cp.copy()
    next_id = int(merged.max()) + 1
    cp_ids = [int(x) for x in np.unique(cp) if int(x) > 0]
    for pid in sorted(int(x) for x in np.unique(ps) if int(x) > 0):
        ps_m = ps == pid
        if not ps_m.any():
            continue
        max_iou = 0.0
        for cid in cp_ids:
            cp_m = cp == cid
            union = int((ps_m | cp_m).sum())
            if union <= 0:
                continue
            max_iou = max(max_iou, int((ps_m & cp_m).sum()) / union)
        if max_iou >= iou_thresh:
            continue
        merged[ps_m] = next_id
        next_id += 1
    return merged


def prepare_merged_seed_dir(
    cp_flow_dir: Path,
    pathosam_masks_path: Path,
    seed_dir: Path,
    *,
    iou_thresh: float = 0.5,
) -> Path:
    """Write ``step04_masks_uint16.npy`` (CP+PS union) + link ``step03_dP_cellprob.npz``."""
    npz = cp_flow_dir / "step03_dP_cellprob.npz"
    cp_masks = cp_flow_dir / "step04_masks_uint16.npy"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    if not cp_masks.is_file():
        raise FileNotFoundError(cp_masks)
    seed_dir.mkdir(parents=True, exist_ok=True)
    dst_npz = seed_dir / "step03_dP_cellprob.npz"
    if not dst_npz.exists():
        try:
            dst_npz.symlink_to(npz.resolve())
        except OSError:
            import shutil
            shutil.copy2(npz, dst_npz)
    cp = np.load(cp_masks)
    ps = np.load(pathosam_masks_path)
    merged = merge_cellpose_pathosam_seeds(cp, ps, iou_thresh=iou_thresh)
    np.save(seed_dir / "step04_masks_uint16.npy", merged.astype(np.uint16))
    return seed_dir


def run_idisf_merged_cp_pathosam_seeds(
    image_path: Path,
    cp_flow_dir: Path,
    pathosam_masks_path: Path,
    out_dir: Path,
    *,
    pipe_dir: Path,
    repo_dir: Path,
    env: dict[str, str] | None = None,
    iou_thresh: float = 0.5,
) -> Path:
    """Per-cell iDISF with Cellpose+PathoSAM union as seeds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = out_dir / "merged_percell_idisf_masks_int32.npy"
    if pr.is_file() and pr.stat().st_size > 0:
        return pr
    seed_dir = out_dir / "merged_seed_cp_ps"
    prepare_merged_seed_dir(cp_flow_dir, pathosam_masks_path, seed_dir, iou_thresh=iou_thresh)
    idisf_args = [
        "--margin", "4",
        "--min-cell-area", "128",
        "--erosion-fg", "1",
        "--erosion-bg", "1",
        "--bg-margin", "2",
        "--disable-and-merge",
    ]
    subprocess.run(
        [
            sys.executable,
            str(pipe_dir / "percell_idisf_cellpose_pipeline.py"),
            "--from-dir", str(seed_dir),
            "-o", str(out_dir),
            "--image", str(image_path),
            *idisf_args,
        ],
        cwd=str(repo_dir),
        env=env,
        check=True,
    )
    return pr


def run_sicle_percell(
    image_path: Path,
    cp_dir: Path,
    out_dir: Path,
    *,
    pipe_dir: Path,
    repo_dir: Path,
    env: dict[str, str] | None = None,
) -> Path:
    """Per-cell SICLE on Cellpose seeds (Nf=2, raw merge)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = out_dir / "merged_percell_sicle_masks_int32.npy"
    if pr.is_file() and pr.stat().st_size > 0:
        return pr
    sicle_args = [
        "--no-saliency-linearize",
        "--sicle-conn-opt", "gradvmaxmul",
        "--sicle-crit-opt", "minsc",
        "--sicle-alpha", "2.0",
        "--saliency-threshold", "0.3",
        "--saliency-blur-sigma", "0.5",
        "--margin", "4",
        "--min-cell-area", "128",
        "--sicle-irreg", "0",
        "--sicle-adhr", "1",
        "--sicle-max-iters", "7",
        "--disable-and-merge",
        "--closing-radius", "0",
        "--sicle-nf", "2",
        "--sicle-n0", "200",
    ]
    subprocess.run(
        [
            sys.executable,
            str(pipe_dir / "percell_sicle_cellprob_pipeline.py"),
            "--from-dir", str(cp_dir),
            "-o", str(out_dir),
            "--image", str(image_path),
            *sicle_args,
        ],
        cwd=str(repo_dir),
        env=env,
        check=True,
    )
    return pr


def run_idisf_percell(
    image_path: Path,
    cp_dir: Path,
    out_dir: Path,
    *,
    pipe_dir: Path,
    repo_dir: Path,
    env: dict[str, str] | None = None,
) -> Path:
    """Per-cell iDISF on Cellpose seeds (unconquerable neighbors)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = out_dir / "merged_percell_idisf_masks_int32.npy"
    if pr.is_file() and pr.stat().st_size > 0:
        return pr
    idisf_args = [
        "--margin", "4",
        "--min-cell-area", "128",
        "--erosion-fg", "1",
        "--erosion-bg", "1",
        "--bg-margin", "2",
        "--disable-and-merge",
    ]
    subprocess.run(
        [
            sys.executable,
            str(pipe_dir / "percell_idisf_cellpose_pipeline.py"),
            "--from-dir", str(cp_dir),
            "-o", str(out_dir),
            "--image", str(image_path),
            *idisf_args,
        ],
        cwd=str(repo_dir),
        env=env,
        check=True,
    )
    return pr


def ihc_mask_to_instances(mask3: np.ndarray) -> np.ndarray:
    """Merge IHC TMA ch0/ch1 per-nucleus labels into one int32 instance map."""
    gt = np.zeros(mask3.shape[1:], dtype=np.int32)
    off = 0
    for ch in (0, 1):
        lab = mask3[ch]
        for i in range(1, int(lab.max()) + 1):
            off += 1
            gt[lab == i] = off
    return gt
