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
    saliency_prob: np.ndarray | None = None,
    saliency_min: float = 0.0,
    min_area: int = 0,
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
    sal = None if saliency_prob is None else np.asarray(saliency_prob, dtype=np.float32)
    if sal is not None and sal.shape != cp.shape:
        h = min(cp.shape[0], sal.shape[0])
        w = min(cp.shape[1], sal.shape[1])
        cp, ps, merged, sal = cp[:h, :w], ps[:h, :w], merged[:h, :w], sal[:h, :w]
    for pid in sorted(int(x) for x in np.unique(ps) if int(x) > 0):
        ps_m = ps == pid
        if not ps_m.any() or int(ps_m.sum()) < min_area:
            continue
        if sal is not None and float(sal[ps_m].mean()) < saliency_min:
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


def pathosam_novel_seeds_only(
    cellpose_masks: np.ndarray,
    pathosam_masks: np.ndarray,
    *,
    iou_thresh: float = 0.5,
    saliency_prob: np.ndarray | None = None,
    saliency_min: float = 0.0,
    min_area: int = 128,
) -> np.ndarray:
    """PathoSAM instances not overlapping Cellpose (IoU < thresh), relabeled 1..N."""
    union = merge_cellpose_pathosam_seeds(
        cellpose_masks,
        pathosam_masks,
        iou_thresh=iou_thresh,
        saliency_prob=saliency_prob,
        saliency_min=saliency_min,
        min_area=min_area,
    )
    cp = np.asarray(cellpose_masks, dtype=np.int32)
    if union.shape != cp.shape:
        h = min(union.shape[0], cp.shape[0])
        w = min(union.shape[1], cp.shape[1])
        union, cp = union[:h, :w], cp[:h, :w]
    novel = np.zeros_like(union, dtype=np.int32)
    next_id = 1
    for lab in sorted(int(x) for x in np.unique(union) if int(x) > 0):
        m = union == lab
        if not (m & (cp > 0)).any():
            novel[m] = next_id
            next_id += 1
    return novel


def merge_idisf_split_outputs(
    cp_out: np.ndarray,
    ps_out: np.ndarray,
) -> np.ndarray:
    """
    Keep Cellpose+iDISF cells unchanged; add PathoSAM-only iDISF where CP left background.
    """
    merged = np.asarray(cp_out, dtype=np.int32).copy()
    ps_out = np.asarray(ps_out, dtype=np.int32)
    if ps_out.shape != merged.shape:
        h = min(merged.shape[0], ps_out.shape[0])
        w = min(merged.shape[1], ps_out.shape[1])
        merged, ps_out = merged[:h, :w], ps_out[:h, :w]
    next_id = int(merged.max()) + 1
    for pid in sorted(int(x) for x in np.unique(ps_out) if int(x) > 0):
        place = (ps_out == pid) & (merged == 0)
        if not place.any():
            continue
        merged[place] = next_id
        next_id += 1
    return merged


def _write_seed_masks(seed_dir: Path, masks: np.ndarray, cp_flow_dir: Path) -> Path:
    """Write step04 + symlink cellprob npz from cp_flow."""
    npz = cp_flow_dir / "step03_dP_cellprob.npz"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    seed_dir.mkdir(parents=True, exist_ok=True)
    dst_npz = seed_dir / "step03_dP_cellprob.npz"
    if not dst_npz.exists():
        try:
            dst_npz.symlink_to(npz.resolve())
        except OSError:
            import shutil
            shutil.copy2(npz, dst_npz)
    np.save(seed_dir / "step04_masks_uint16.npy", np.asarray(masks, dtype=np.uint16))
    return seed_dir


def prepare_merged_seed_dir(
    cp_flow_dir: Path,
    pathosam_masks_path: Path,
    seed_dir: Path,
    *,
    iou_thresh: float = 0.5,
    saliency_path: Path | None = None,
    saliency_min: float = 0.0,
    min_area: int = 0,
) -> Path:
    """Write ``step04_masks_uint16.npy`` (CP+PS union) + link ``step03_dP_cellprob.npz``."""
    cp_masks = cp_flow_dir / "step04_masks_uint16.npy"
    if not cp_masks.is_file():
        raise FileNotFoundError(cp_masks)
    cp = np.load(cp_masks)
    ps = np.load(pathosam_masks_path)
    sal = np.load(saliency_path).astype(np.float32) if saliency_path and saliency_path.is_file() else None
    merged = merge_cellpose_pathosam_seeds(
        cp, ps, iou_thresh=iou_thresh, saliency_prob=sal, saliency_min=saliency_min, min_area=min_area,
    )
    return _write_seed_masks(seed_dir, merged, cp_flow_dir)


def _idisf_percell_args() -> list[str]:
    return [
        "--margin", "4",
        "--min-cell-area", "128",
        "--erosion-fg", "1",
        "--erosion-bg", "1",
        "--bg-margin", "2",
        "--disable-and-merge",
    ]


def _run_idisf_percell(
    image_path: Path,
    seed_dir: Path,
    out_dir: Path,
    *,
    pipe_dir: Path,
    repo_dir: Path,
    env: dict[str, str] | None,
) -> Path:
    subprocess.run(
        [
            sys.executable,
            str(pipe_dir / "percell_idisf_cellpose_pipeline.py"),
            "--from-dir", str(seed_dir),
            "-o", str(out_dir),
            "--image", str(image_path),
            *_idisf_percell_args(),
        ],
        cwd=str(repo_dir),
        env=env,
        check=True,
    )
    pr = out_dir / "merged_percell_idisf_masks_int32.npy"
    if not pr.is_file() or pr.stat().st_size == 0:
        raise RuntimeError(f"missing iDISF output: {pr}")
    return pr


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
    strategy: str = "union",
    saliency_min: float = 0.0,
    min_area: int = 128,
) -> Path:
    """
    Per-cell iDISF with Cellpose + PathoSAM seeds.

    Strategies:
      - ``union``: single iDISF on merged seed map (legacy; PS seeds can affect CP neighbors).
      - ``split``: iDISF on CP seeds, then iDISF on PS-novel only; merge without touching CP cells.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = out_dir / "merged_percell_idisf_masks_int32.npy"
    if pr.is_file() and pr.stat().st_size > 0:
        return pr

    ps_dir = pathosam_masks_path.parent
    sal_path = pathosam_saliency_path(ps_dir)
    sal = np.load(sal_path).astype(np.float32) if sal_path.is_file() else None

    if strategy == "union":
        seed_dir = out_dir / "merged_seed_cp_ps"
        prepare_merged_seed_dir(
            cp_flow_dir, pathosam_masks_path, seed_dir,
            iou_thresh=iou_thresh, saliency_path=sal_path if sal_path.is_file() else None,
            saliency_min=saliency_min, min_area=min_area,
        )
        _run_idisf_percell(image_path, seed_dir, out_dir, pipe_dir=pipe_dir, repo_dir=repo_dir, env=env)
        return pr

    if strategy != "split":
        raise ValueError(f"unknown strategy: {strategy}")

    cp = np.load(cp_flow_dir / "step04_masks_uint16.npy")
    ps = np.load(pathosam_masks_path)
    novel = pathosam_novel_seeds_only(
        cp, ps, iou_thresh=iou_thresh, saliency_prob=sal, saliency_min=saliency_min, min_area=min_area,
    )
    cp_only_dir = out_dir / "_split_cp_seeds"
    cp_work = out_dir / "_split_cp_work"
    _write_seed_masks(cp_only_dir, cp, cp_flow_dir)
    _run_idisf_percell(image_path, cp_only_dir, cp_work, pipe_dir=pipe_dir, repo_dir=repo_dir, env=env)
    cp_out = np.load(cp_work / "merged_percell_idisf_masks_int32.npy").astype(np.int32)

    if not (novel > 0).any():
        np.save(pr, cp_out)
        return pr

    ps_only_dir = out_dir / "_split_ps_seeds"
    ps_work = out_dir / "_split_ps_work"
    _write_seed_masks(ps_only_dir, novel, cp_flow_dir)
    _run_idisf_percell(image_path, ps_only_dir, ps_work, pipe_dir=pipe_dir, repo_dir=repo_dir, env=env)
    ps_out = np.load(ps_work / "merged_percell_idisf_masks_int32.npy").astype(np.int32)
    merged = merge_idisf_split_outputs(cp_out, ps_out)
    np.save(pr, merged)
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


_STARDIST_CACHE: dict[str, object] = {}


def stardist_model_name_for_dataset(dataset: str) -> str:
    """Pick a pretrained StarDist2D model by modality."""
    if dataset in ("dsb2018", "ihc_tma"):
        return "2D_versatile_fluo"
    return "2D_versatile_he"


def _stardist_model(model_name: str):
    if model_name not in _STARDIST_CACHE:
        from stardist.models import StarDist2D

        _STARDIST_CACHE[model_name] = StarDist2D.from_pretrained(model_name)
    return _STARDIST_CACHE[model_name]


def _stardist_input_array(image_path: Path, model_name: str) -> np.ndarray:
    """Build YXC array matching the pretrained model's ``n_channel_in``."""
    from skimage import color

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    if model_name == "2D_versatile_fluo":
        gray = color.rgb2gray(rgb)
        return gray[..., np.newaxis]
    return rgb


def run_stardist(
    image_path: Path,
    out_dir: Path,
    *,
    model_name: str = "2D_versatile_he",
    gpu: bool = False,
) -> Path:
    """Run pretrained StarDist2D; returns step04 masks path."""
    import os

    from csbdeep.utils import normalize

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "step04_masks_uint16.npy"
    if out.is_file() and out.stat().st_size > 0:
        arr = np.load(out)
        if arr.ndim == 2 and int(arr.max()) > 0:
            return out

    # StarDist uses TensorFlow; force CPU when gpu=False or when CuDNN is unavailable.
    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        model = _stardist_model(model_name)
        img = _stardist_input_array(image_path, model_name)
        img_norm = normalize(img, 1, 99.8, axis=(0, 1))
        labels, _ = model.predict_instances(img_norm)
    except Exception:
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            model = _stardist_model(model_name)
            img = _stardist_input_array(image_path, model_name)
            img_norm = normalize(img, 1, 99.8, axis=(0, 1))
            labels, _ = model.predict_instances(img_norm)
        else:
            raise
    finally:
        if not gpu:
            if prev_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda

    np.save(out, np.asarray(labels, dtype=np.uint16))
    return out


def run_watershed(
    image_path: Path,
    out_dir: Path,
    *,
    sigma: float = 1.0,
    min_distance_frac: float = 0.015,
    min_object_size: int = 64,
) -> Path:
    """
    Classical marker-controlled watershed on an Otsu foreground mask.

    Markers are local maxima of the Euclidean distance transform; watershed
    runs on the Sobel gradient inside the foreground mask.
    """
    from scipy import ndimage as ndi
    from skimage import color, filters, morphology, segmentation
    from skimage.feature import peak_local_max
    from skimage.measure import label

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "step04_masks_uint16.npy"
    if out.is_file() and out.stat().st_size > 0:
        arr = np.load(out)
        if arr.ndim == 2 and int(arr.max()) > 0:
            return out

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    gray = color.rgb2gray(rgb)
    smoothed = filters.gaussian(gray, sigma=sigma, preserve_range=True).astype(np.float32)
    binary = smoothed > filters.threshold_otsu(smoothed)
    binary = morphology.opening(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, max_size=min_object_size - 1)

    distance = ndi.distance_transform_edt(binary)
    min_dist = max(4, int(min(rgb.shape[:2]) * min_distance_frac))
    coords = peak_local_max(
        distance,
        min_distance=min_dist,
        labels=label(binary),
    )
    markers = np.zeros(distance.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    if markers.max() == 0:
        np.save(out, np.zeros(rgb.shape[:2], dtype=np.uint16))
        return out

    gradient = filters.sobel(smoothed)
    labels = segmentation.watershed(gradient, markers, mask=binary)
    np.save(out, np.asarray(labels, dtype=np.uint16))
    return out


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
