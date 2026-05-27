#!/usr/bin/env python3
"""
Pipeline: Cellpose → per-cell crops (10px margin) → scribble annotation → run scribble-based segmenter.
Segmenter: iDISF (default), PyIFT (pyift), UOIFT (uoift), SICLE (RunSICLE), or fusion (majority vote of iDISF+PyIFT+SICLE).

Scribbles: default ``scribble_source=cellpose`` (eroded mask + border / other cells). Optional
``scribble_source=activation`` uses a greyscale map from the CP-SAM last conv (``extract_activation_maps``,
``layer=out``) at the **same crop** as Cellpose; thresholds are percentiles **inside the cell mask**.

SICLE path costs: ``--sicle-preset irregular|compact`` or ``--sicle-conn-opt`` / ``--sicle-crit-opt``.
Optionally builds a reunited mosaic.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, binary_opening, generate_binary_structure
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CELLPOSE_DIR = ROOT / "cellpose"
IDISF_PY_DIR = ROOT / "iDISF" / "python3"
UNSUPSEG_DIR = ROOT / "unsupseg"
if str(CELLPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(CELLPOSE_DIR))
if str(IDISF_PY_DIR) not in sys.path:
    sys.path.insert(0, str(IDISF_PY_DIR))

try:
    from cellpose import models, utils, plot
    from cellpose.transforms import convert_image
except ImportError as e:
    raise ImportError("Cellpose not found. Add cellpose to PYTHONPATH or run from doutorado with ./cellpose present.") from e

try:
    from idisf import iDISF_scribbles
except ImportError as e:
    raise ImportError(
        "iDISF Python module not found. Build: cd iDISF && make lib && cd python3 && python setup.py build_ext --inplace"
    ) from e

_PYIFT_AVAILABLE = False
try:
    import pyift.pyift as ift
    _PYIFT_AVAILABLE = True
except ImportError:
    ift = None


MARGIN_DEFAULT = 10
EROSION_DEFAULT = 1
EROSION_MAX = 16  # max erosion depth (px): result of erosion = strip at most this many px from border
EROSION_BG_DEFAULT = 1  # default erosion for "other cells" in background (same cap EROSION_MAX)
BG_MARGIN_DEFAULT = 2  # background scribbles: band width (px) from crop borders only
IDISF_N0_DEFAULT = 1000
IDISF_ITERATIONS_DEFAULT = 6
IDISF_F_DEFAULT = 4
IDISF_C1_DEFAULT = 0.7
IDISF_C2_DEFAULT = 0.8
UOIFT_POLARITY_DEFAULT = 0.5
UOIFT_SPSIZE_DEFAULT = 100
SICLE_N0_DEFAULT = 500
SICLE_NF_DEFAULT = 2  # target 2 superpixels
SICLE_ALPHA_DEFAULT = 0.9
SICLE_MAXITERS_DEFAULT = 22
SICLE_IRREG_DEFAULT = 0.12
SICLE_ADHR_DEFAULT = 16
# IFT path costs (RunSICLE --conn-opt / --crit-opt): match RunSICLEIRREG.c vs RunSICLECOMP.c and
# Domain-Specific Seed Removal SICLE (nuclear / domain presets).
SICLE_PRESET_CONN_CRIT: dict[str, tuple[str, str]] = {
    "irregular": ("fmax", "minsc"),  # same as RunSICLE defaults / RunSICLEIRREG
    "compact": ("fsum", "maxsc"),  # RunSICLECOMP-optimized pair
}
SICLE_PEN_OPT_DEFAULT = "none"


def resolve_sicle_path_cost(
    preset: str,
    conn_opt: str | None,
    crit_opt: str | None,
) -> tuple[str, str]:
    """Return (conn_opt, crit_opt) for RunSICLE. Override preset when both conn and crit are set."""
    if conn_opt is not None or crit_opt is not None:
        if conn_opt is None or crit_opt is None:
            raise ValueError(
                "Override SICLE path costs with both --sicle-conn-opt and --sicle-crit-opt, or use neither (preset applies)."
            )
        return conn_opt, crit_opt
    return SICLE_PRESET_CONN_CRIT[preset]


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    img = np.array(Image.open(path))
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
    return img


def find_sicle_binary() -> Path:
    """Locate SICLE RunSICLE binary (Linux path, not via wsl)."""
    env_path = os.environ.get("SICLE_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"SICLE_BIN={env_path} does not exist")
    candidates: list[Path] = []
    _up = ROOT.parent
    if (_up / "cellpose").is_dir():
        candidates.extend(
            [
                _up / "SICLE" / "bin" / "RunSICLE",
                _up / "PIPELINE_UOIFT_SICLE" / "uoift_sicle" / "SICLE" / "bin" / "RunSICLE",
            ]
        )
    candidates.extend(
        [
            ROOT / "SICLE" / "bin" / "RunSICLE",
            ROOT / "PIPELINE_UOIFT_SICLE" / "uoift_sicle" / "SICLE" / "bin" / "RunSICLE",
            Path.home() / "SICLE" / "bin" / "RunSICLE",
            Path("/usr/local/bin/RunSICLE"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "SICLE binary not found. Options:\n"
        "  1. set SICLE_BIN=/path/to/RunSICLE\n"
        "  2. place binary at <repo>/SICLE/bin/RunSICLE\n"
        "  3. place binary at PIPELINE_UOIFT_SICLE/uoift_sicle/SICLE/bin/RunSICLE\n"
        "  4. place binary at ~/SICLE/bin/RunSICLE or /usr/local/bin/RunSICLE"
    )


def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img


def run_cellpose(img: np.ndarray, model_type: str = "cyto3", gpu: bool = True, diameter: float | None = None) -> np.ndarray:
    img_3ch = ensure_3ch(img)
    model = models.CellposeModel(gpu=gpu, model_type=model_type)
    masks, *_ = model.eval(img_3ch, diameter=diameter, channel_axis=-1)
    if isinstance(masks, list):
        masks = masks[0]
    return np.asarray(masks, dtype=np.int32)


def bbox_of_mask(mask: np.ndarray, label: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask == label)
    if rows.size == 0 or cols.size == 0:
        return 0, 0, 0, 0
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def crop_with_margin(img: np.ndarray, mask: np.ndarray, label: int, margin: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    H, W = mask.shape
    rmin, rmax, cmin, cmax = bbox_of_mask(mask, label)
    r0 = max(0, rmin - margin)
    r1 = min(H, rmax + margin)
    c0 = max(0, cmin - margin)
    c1 = min(W, cmax + margin)
    img_crop = img[r0:r1, c0:c1]
    mask_crop = (mask[r0:r1, c0:c1] == label).astype(np.uint8)
    return img_crop, mask_crop, (r0, r1, c0, c1)


def eroded_foreground_mask(mask_crop: np.ndarray, erosion_pixels: int = 1, max_erosion: int | None = EROSION_MAX) -> np.ndarray:
    """Erode foreground mask. erosion_pixels = depth (px removed from border); capped at max_erosion so result is up to that many px."""
    if erosion_pixels <= 0:
        return mask_crop.copy()
    if max_erosion is not None and erosion_pixels > max_erosion:
        erosion_pixels = max_erosion
    if erosion_pixels == 1:
        se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    else:
        y, x = np.ogrid[-erosion_pixels : erosion_pixels + 1, -erosion_pixels : erosion_pixels + 1]
        se = (x * x + y * y <= erosion_pixels * erosion_pixels)
    return binary_erosion(mask_crop, structure=se).astype(np.uint8)


def margin_mask(h: int, w: int, margin: int) -> np.ndarray:
    """Binary mask: 1 in a band of width `margin` along crop borders."""
    band = np.zeros((h, w), dtype=np.uint8)
    if margin <= 0:
        return band
    band[:margin, :] = 1
    band[-margin:, :] = 1
    band[:, :margin] = 1
    band[:, -margin:] = 1
    return band


def background_mask_for_crop(
    h: int,
    w: int,
    full_mask_crop: np.ndarray,
    current_cell_id: int,
    border_px: int = BG_MARGIN_DEFAULT,
    erosion_bg_pixels: int = 0,
    use_other_cells: bool = True,
    max_erosion: int | None = EROSION_MAX,
) -> np.ndarray:
    """Background = border band (+ optionally other-cell pixels in crop).

    - Always includes a border band of width `border_px` around the crop.
    - If `use_other_cells` is True, also includes pixels from *other* cells in the crop,
      optionally eroded by `erosion_bg_pixels` (capped at `max_erosion`).
    """
    border = margin_mask(h, w, border_px)
    if use_other_cells:
        other_cells = ((full_mask_crop != 0) & (full_mask_crop != current_cell_id)).astype(np.uint8)
        if erosion_bg_pixels > 0:
            other_cells = eroded_foreground_mask(other_cells, erosion_bg_pixels, max_erosion=max_erosion)
    else:
        other_cells = np.zeros((h, w), dtype=np.uint8)
    return np.clip(border.astype(np.uint8) + other_cells, 0, 1).astype(np.uint8)


def _activation_extract_array(extracted: np.ndarray | tuple) -> np.ndarray:
    """``CellposeModel.extract_activation_maps`` returns ``(act, act_rgb)``; normalize to ``[H,W,C]``."""
    if isinstance(extracted, tuple):
        extracted = extracted[0]
    a = np.asarray(extracted)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    return a


def scribbles_from_activation(
    act_crop: np.ndarray,
    mask_crop: np.ndarray,
    full_mask_crop: np.ndarray,
    cell_id: int,
    h: int,
    w: int,
    erosion_fg: int,
    erosion_bg: int,
    bg_margin: int,
    use_bg_cells: bool,
    fg_percentile: float = 66.0,
    bg_low_percentile: float = 35.0,
    reduce_mode: str = "l2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Foreground / background seed masks from last-layer activation (greyscale scalar), at the
    same crop as the Cellpose-based pipeline (``crop_with_margin``).

    High response inside the current cell mask → foreground scribbles; low response inside the
    cell (and border / other cells) → background, matching the role of erosion + border in the
    Cellpose scribble recipe.

    Also returns per-crop visualization:
    - ``grey_u8``: normalized activation 0–255 (whole crop).
    - ``bw_u8``: thresholded **before** erosion — white (255) = object (activation ≥ FG percentile
      inside the Cellpose mask), black (0) = background. Same rule as the binary seed region
      ``fg_bin`` (not the eroded ``fg_mask``).
    """
    from cellpose.activation_maps import reduce_activation_to_scalar

    scalar = reduce_activation_to_scalar(act_crop, mode=reduce_mode)
    lo = float(np.percentile(scalar, 1.0))
    hi = float(np.percentile(scalar, 99.0))
    if hi <= lo + 1e-12:
        hi = lo + 1.0
    norm = np.clip((scalar - lo) / (hi - lo), 0.0, 1.0)
    grey_u8 = (norm * 255.0).astype(np.uint8)

    inside = mask_crop > 0
    if not inside.any():
        empty_fg = np.zeros((h, w), dtype=np.uint8)
        bw_u8 = np.zeros((h, w), dtype=np.uint8)
        return empty_fg, empty_fg, grey_u8, bw_u8

    vals = norm[inside]
    t_fg = float(np.percentile(vals, fg_percentile))
    t_bg = float(np.percentile(vals, bg_low_percentile))

    fg_bin = (norm >= t_fg) & inside
    fg_mask = eroded_foreground_mask(fg_bin.astype(np.uint8), erosion_fg)

    bg_mask = background_mask_for_crop(
        h,
        w,
        full_mask_crop,
        cell_id,
        border_px=bg_margin,
        erosion_bg_pixels=erosion_bg,
        use_other_cells=use_bg_cells,
    )
    low_in_cell = (norm <= t_bg) & inside
    bg_mask = np.clip(bg_mask.astype(np.float32) + low_in_cell.astype(np.float32), 0, 1).astype(np.uint8)

    # White = object (high-activation nucleus band); black = rest of crop (threshold matches fg_bin)
    bw_u8 = np.where(fg_bin, 255, 0).astype(np.uint8)
    return fg_mask, bg_mask, grey_u8, bw_u8


def mask_to_scribble_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(mask > 0)
    coords = list(zip(cols.tolist(), rows.tolist()))
    if len(coords) > 50000:
        step = len(coords) // 50000
        coords = coords[:: max(1, step)]
    return coords


def write_idisf_annotation(path: str | Path, fg_coords: list[tuple[int, int]], bg_coords: list[tuple[int, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("2\n")
        f.write(f"{len(fg_coords)}\n")
        for (x, y) in fg_coords:
            f.write(f"{x};{y}\n")
        f.write(f"{len(bg_coords)}\n")
        for i, (x, y) in enumerate(bg_coords):
            sep = "\n" if i < len(bg_coords) - 1 else ""
            f.write(f"{x};{y}{sep}")


def run_idisf_on_crop(
    img_crop: np.ndarray,
    fg_coords: list[tuple[int, int]],
    bg_coords: list[tuple[int, int]],
    n0: int = IDISF_N0_DEFAULT,
    iterations: int = IDISF_ITERATIONS_DEFAULT,
    f: int = IDISF_F_DEFAULT,
    c1: float = IDISF_C1_DEFAULT,
    c2: float = IDISF_C2_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    num_pixels = int(img_crop.shape[0]) * int(img_crop.shape[1])
    n0_capped = min(n0, max(1, num_pixels // 2))

    if img_crop.ndim == 2:
        img_3ch = np.stack([img_crop, img_crop, img_crop], axis=-1).astype(np.int32)
    else:
        img_3ch = np.asarray(img_crop, dtype=np.int32)
    if img_3ch.shape[-1] != 3:
        img_3ch = np.stack([img_3ch[..., 0], img_3ch[..., 0], img_3ch[..., 0]], axis=-1)

    coords = np.array(fg_coords + bg_coords, dtype=np.int32)
    size_scribbles = np.array([len(fg_coords), len(bg_coords)], dtype=np.int32)
    num_obj = 1
    segm_method = 1
    all_borders = 1

    label_img, border_img = iDISF_scribbles(
        img_3ch, n0_capped, iterations, coords, size_scribbles, num_obj, f, c1, c2, segm_method, all_borders,
    )
    return label_img, border_img


def run_pyift_dyntree_on_crop(
    img_crop: np.ndarray,
    fg_coords: list[tuple[int, int]],
    bg_coords: list[tuple[int, int]],
    temp_dir: Path,
    crop_name: str,
    delta: int = 1,
    gamma: float = 0.0,
    closest_root: bool = False,
) -> np.ndarray:
    """Segment crop with PyIFT dynamic tree from scribbles. Returns label_img (1=object, 2=background). Requires pyift."""
    if not _PYIFT_AVAILABLE or ift is None:
        raise ImportError("PyIFT not available. Install pyift to use --segmenter pyift.")
    h, w = img_crop.shape[0], img_crop.shape[1]
    # Marker image: 0 unlabeled. LibIFT uses 1=background, 2=object; we put fg (cell) as 2, bg as 1 so "object" = inside.
    marker = np.zeros((h, w), dtype=np.uint8)
    for (x, y) in fg_coords:
        if 0 <= y < h and 0 <= x < w:
            marker[y, x] = 2   # object (cell / inside)
    for (x, y) in bg_coords:
        if 0 <= y < h and 0 <= x < w:
            marker[y, x] = 1   # background (outside)
    crop_path = temp_dir / f"{crop_name}_pyift_input.png"
    marker_path = temp_dir / f"{crop_name}_pyift_markers.pgm"
    segm_path = temp_dir / f"{crop_name}_pyift_segm.pgm"
    img_u8 = np.asarray(img_crop, dtype=np.uint8)
    if img_u8.ndim == 2:
        Image.fromarray(img_u8, mode="L").save(crop_path)
    else:
        Image.fromarray(img_u8[..., :3]).save(crop_path)
    Image.fromarray(marker, mode="L").save(marker_path)

    orig = ift.ReadImageByExt(str(crop_path))
    mrk = ift.ReadImageByExt(str(marker_path))
    seeds = ift.LabeledSetFromSeedImage(mrk, True)
    mimg = ift.ImageToMImage(orig, ift.LABNorm_CSPACE)
    A = ift.Circular(1.0)
    if closest_root:
        segm = ift.DynTreeClosestRoot(mimg, A, seeds, delta, gamma, None, 0.0)
    else:
        segm = ift.DynTreeRoot(mimg, A, seeds, delta, gamma, None, 0.0)
    ift.WriteImageByExt(segm, str(segm_path))
    label_img = np.array(Image.open(segm_path), dtype=np.int32)
    if label_img.ndim > 2:
        label_img = label_img.squeeze()
    if label_img.ndim > 2:
        label_img = label_img[..., 0]
    # Map to 1=object, 2=background: which label covers more fg_coords?
    uniq = np.unique(label_img)
    uniq = uniq[uniq > 0]
    if len(uniq) < 2:
        out = np.ones_like(label_img, dtype=np.uint8)
        out[label_img != uniq[0] if len(uniq) == 1 else True] = 2
        return out
    cnt = {}
    for u in uniq:
        cnt[u] = sum(1 for (x, y) in fg_coords if 0 <= y < h and 0 <= x < w and label_img[y, x] == u)
    obj_label = max(uniq, key=lambda u: cnt.get(u, 0))
    bg_label = next(u for u in uniq if u != obj_label)
    out = np.zeros_like(label_img, dtype=np.uint8)
    out[label_img == obj_label] = 1
    out[label_img == bg_label] = 2
    return out


def fuse_label_maps(label_list: list[np.ndarray], object_label: int = 1, background_label: int = 2) -> np.ndarray:
    """Fuse multiple label images (1=object, 2=background) by per-pixel majority vote. Returns single label map (1 or 2)."""
    if not label_list:
        raise ValueError("fuse_label_maps requires at least one label image")
    label_list = [np.asarray(L).squeeze().astype(np.int32) for L in label_list]
    h, w = label_list[0].shape[0], label_list[0].shape[1]
    for L in label_list:
        if L.shape[0] != h or L.shape[1] != w:
            raise ValueError("All label images must have the same shape")
    votes_obj = np.zeros((h, w), dtype=np.int32)
    votes_bg = np.zeros((h, w), dtype=np.int32)
    for L in label_list:
        votes_obj += (L == object_label).astype(np.int32)
        votes_bg += (L == background_label).astype(np.int32)
    out = np.where(votes_obj >= votes_bg, object_label, background_label).astype(np.uint8)
    return out


def run_sicle_on_crop(
    img_crop: np.ndarray,
    fg_coords: list[tuple[int, int]],
    temp_dir: Path,
    crop_name: str,
    sicle_bin: Path,
    n0: int = SICLE_N0_DEFAULT,
    nf: int = SICLE_NF_DEFAULT,
    alpha: float = SICLE_ALPHA_DEFAULT,
    max_iters: int = SICLE_MAXITERS_DEFAULT,
    irreg: float = SICLE_IRREG_DEFAULT,
    adhr: int = SICLE_ADHR_DEFAULT,
    conn_opt: str = "fmax",
    crit_opt: str = "minsc",
    pen_opt: str = SICLE_PEN_OPT_DEFAULT,
) -> np.ndarray:
    """Segment crop with SICLE into ~nf superpixels (default 2) and map the one overlapping FG scribbles to object (1), others to background (2).

    Path costs follow RunSICLE: ``--conn-opt`` (fmax, fsum, gradvmax, gradvmaxmul, custom) and ``--crit-opt`` (size, minsc, maxsc, spread, custom).
    Presets ``irregular`` (fmax+minsc) and ``compact`` (fsum+maxsc) mirror RunSICLEIRREG / RunSICLECOMP.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    img_u8 = np.asarray(img_crop, dtype=np.uint8)
    if img_u8.ndim == 2:
        img_u8_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    else:
        img_u8_rgb = img_u8[..., :3]
    img_path = temp_dir / f"{crop_name}_sicle_input.ppm"
    out_path = temp_dir / f"{crop_name}_sicle_output.pgm"
    Image.fromarray(img_u8_rgb).save(img_path)

    objsm_path = temp_dir / f"{crop_name}_sicle_objsm.pgm"
    if conn_opt in ("gradvmax", "gradvmaxmul"):
        r = img_u8_rgb[:, :, 0].astype(np.float32)
        g = img_u8_rgb[:, :, 1].astype(np.float32)
        b = img_u8_rgb[:, :, 2].astype(np.float32)
        if np.allclose(r, g) and np.allclose(r, b):
            sal_gray = img_u8_rgb[:, :, 0]
        else:
            sal_gray = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(sal_gray, mode="L").save(objsm_path)

    # Helper to actually run SICLE once with a given n0
    def _run_once(current_n0: int) -> tuple[bool, str]:
        cmd = [
            str(sicle_bin),
            "--img", str(img_path),
            "--out", str(out_path),
            "--n0", str(current_n0),
            "--nf", str(nf),
            "--alpha", str(alpha),
            "--max-iters", str(max_iters),
            "--conn-opt", conn_opt,
            "--crit-opt", crit_opt,
            "--pen-opt", pen_opt,
            "--sampl-opt", "grid",
            "--irreg", str(irreg),
            "--adhr", str(adhr),
        ]
        if conn_opt in ("gradvmax", "gradvmaxmul"):
            cmd += ["--objsm", str(objsm_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        ok = proc.returncode == 0 and out_path.exists()
        return ok, proc.stderr

    # First attempt with requested n0; on "Invalid N0" error, adapt to SICLE's suggested upper bound and retry once
    ok, stderr = _run_once(n0)
    if not ok and "Invalid N0 value of" in stderr and "It must be within ]2," in stderr:
        # Parse upper bound from message: "It must be within ]2,1645["
        import re
        m = re.search(r"It must be within ]2,(\d+)\[", stderr)
        if m:
            max_allowed = int(m.group(1)) - 1
            if max_allowed > 2:
                n0_adapted = max(3, max_allowed)
                print(f"Warning: SICLE N0={n0} too large for {crop_name}. Retrying with N0={n0_adapted}.")
                ok, stderr = _run_once(n0_adapted)
    if not ok:
        raise RuntimeError(f"SICLE failed for {crop_name}: {stderr}")
    label_raw = np.array(Image.open(out_path), dtype=np.int32)
    if label_raw.ndim > 2:
        label_raw = label_raw.squeeze()
    if label_raw.ndim > 2:
        label_raw = label_raw[..., 0]
    h, w = label_raw.shape[:2]
    uniq = np.unique(label_raw)
    uniq = uniq[uniq > 0]
    if uniq.size == 0:
        return np.zeros_like(label_raw, dtype=np.uint8)
    # Choose object label as the one that covers more FG scribbles
    cnt: dict[int, int] = {}
    for u in uniq:
        cnt[u] = sum(
            1
            for (x, y) in fg_coords
            if 0 <= y < h and 0 <= x < w and label_raw[y, x] == u
        )
    obj_label = max(uniq, key=lambda u: cnt.get(u, 0))
    out = np.full_like(label_raw, 2, dtype=np.uint8)
    out[label_raw == obj_label] = 1
    return out


def run_uoift_unsupseg_on_crop(
    img_crop: np.ndarray,
    fg_coords: list[tuple[int, int]],
    run_dir: Path,
    unsupseg_bin: Path,
    polarity: float = UOIFT_POLARITY_DEFAULT,
    spsize: int = UOIFT_SPSIZE_DEFAULT,
    force_grayscale: bool = False,
) -> np.ndarray:
    """Segment crop with unsupseg (UOIFT: divisive clustering by OIFT, k=2). Returns label_img (1=object, 2=background)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_crop.shape[0], img_crop.shape[1]
    img_u8 = np.asarray(img_crop, dtype=np.uint8)
    if img_u8.ndim == 3 and img_u8.shape[-1] >= 3:
        gray = np.round(0.299 * img_u8[..., 0] + 0.587 * img_u8[..., 1] + 0.114 * img_u8[..., 2]).astype(np.uint8)
    else:
        gray = img_u8 if img_u8.ndim == 2 else img_u8[:, :, 0]
    use_color = not force_grayscale and (img_crop.ndim == 3 and img_crop.shape[-1] >= 3)
    img_type = 1 if use_color else 0
    input_path = run_dir / "input.ppm" if img_type == 1 else run_dir / "input.pgm"
    input_name = "input.ppm" if img_type == 1 else "input.pgm"
    if img_type == 1:
        Image.fromarray(img_u8[..., :3]).save(input_path)
    else:
        Image.fromarray(gray, mode="L").save(input_path)
    # Pass only filename so unsupseg opens it in cwd (run_dir)
    cmd = [
        str(unsupseg_bin),
        str(img_type),
        input_name,
        "0",
        "2",
        str(polarity),
        str(spsize),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    label_path = run_dir / "label.pgm"
    if proc.returncode != 0 and not force_grayscale and (img_crop.ndim == 3 and img_crop.shape[-1] >= 3):
        # Retry with grayscale (avoids double-free in some GFT PPM paths)
        return run_uoift_unsupseg_on_crop(
            img_crop, fg_coords, run_dir, unsupseg_bin,
            polarity=polarity, spsize=spsize, force_grayscale=True,
        )
    # GFT/unsupseg often crashes in destructors (double free) after writing label.pgm; use result if present
    if proc.returncode != 0 and not label_path.exists():
        err = (proc.stderr or proc.stdout or "").strip()
        if "double free" in err or proc.returncode in (-6, 134):
            err = f"{err} (unsupseg/GFT memory bug; try --uoift-grayscale or use --segmenter idisf/pyift)"
        raise RuntimeError(f"unsupseg failed (exit {proc.returncode}): {err}")
    if not label_path.exists():
        raise FileNotFoundError(f"unsupseg did not produce {label_path}")
    label_img = np.array(Image.open(label_path), dtype=np.int32)
    if label_img.ndim > 2:
        label_img = label_img.squeeze()
    if label_img.shape[0] != h or label_img.shape[1] != w:
        raise ValueError(f"unsupseg label size {label_img.shape} != crop size ({h},{w})")
    uniq = np.unique(label_img)
    uniq = uniq[uniq >= 0]
    if len(uniq) < 2:
        out = np.ones((h, w), dtype=np.uint8)
        if len(uniq) == 1:
            out[label_img != uniq[0]] = 2
        return out
    cnt = {}
    for u in uniq:
        cnt[u] = sum(1 for (x, y) in fg_coords if 0 <= y < h and 0 <= x < w and label_img[y, x] == u)
    obj_label = max(uniq, key=lambda u: cnt.get(u, 0))
    bg_label = next(u for u in uniq if u != obj_label)
    out = np.zeros((h, w), dtype=np.uint8)
    out[label_img == obj_label] = 1
    out[label_img == bg_label] = 2
    return out


def plot_markers(
    img: np.ndarray,
    fg_coords: list[tuple[int, int]],
    bg_coords: list[tuple[int, int]],
    fg_color: tuple[int, int, int] = (0, 255, 0),
    bg_color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 2,
) -> np.ndarray:
    """Draw foreground (object) and background marker coords on the image. Coords are (x, y). Returns RGB uint8."""
    img = np.asarray(img)
    if img.ndim == 2:
        out = np.stack([img, img, img], axis=-1).astype(np.uint8)
    else:
        out = np.asarray(img, dtype=np.uint8)[..., :3].copy()
    if out.max() <= 1:
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    H, W = out.shape[0], out.shape[1]
    for (x, y) in fg_coords:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    r, c = y + dy, x + dx
                    if 0 <= r < H and 0 <= c < W:
                        out[r, c] = fg_color
    for (x, y) in bg_coords:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    r, c = y + dy, x + dx
                    if 0 <= r < H and 0 <= c < W:
                        out[r, c] = bg_color
    return out


def overlay_borders(
    img: np.ndarray,
    border_mask: np.ndarray,
    border_color_rgb: tuple[float, float, float] = (0.0, 1.0, 1.0),
    norm_val: int = 255,
) -> np.ndarray:
    """Overlay border pixels on the image (like iDISF overlayBorders). border_mask: 0 = no border, non-zero = border. Returns RGB uint8."""
    border_mask = np.asarray(border_mask)
    if border_mask.ndim > 2:
        border_mask = border_mask.squeeze()
    is_border = border_mask.astype(bool) if border_mask.dtype != bool else border_mask

    img = np.asarray(img)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img[..., :3].copy()
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, norm_val).astype(np.uint8)

    out = img_rgb.copy()
    r, g, b = int(border_color_rgb[0] * norm_val), int(border_color_rgb[1] * norm_val), int(border_color_rgb[2] * norm_val)
    out[is_border, 0] = r
    out[is_border, 1] = g
    out[is_border, 2] = b
    return out


def mask_inside_border(img: np.ndarray, label_img: np.ndarray, object_label: int = 1) -> np.ndarray:
    """Keep pixel values inside the object (label == object_label); set all other pixels to black. Returns same shape as img, uint8."""
    L = np.asarray(label_img).squeeze()
    inside = L == object_label
    img = np.asarray(img)
    if img.max() <= 1 and img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        out = np.where(inside, img, 0)
    else:
        out = np.where(inside[..., np.newaxis], img, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def label_opening(
    label_img: np.ndarray,
    opening_px: int,
    object_label: int = 1,
    background_label: int = 2,
) -> np.ndarray:
    """Apply morphological opening to the object (object_label) region. opening_px=0 returns a copy unchanged; >=1 uses that many iterations with a 3x3 cross."""
    L = np.asarray(label_img).squeeze().astype(np.uint8)
    if opening_px <= 0:
        return L.copy()
    obj = (L == object_label)
    structure = generate_binary_structure(2, 1)  # 3x3 cross
    opened = binary_opening(obj, structure=structure, iterations=opening_px)
    out = np.where(opened, object_label, background_label).astype(np.uint8)
    return out


def label_borders(label_img: np.ndarray, one_pixel: bool = True) -> np.ndarray:
    """Pixels at label boundaries. If one_pixel=True, mark only object-side (label 1) boundary for 1 px thickness."""
    L = np.asarray(label_img).squeeze().astype(np.int32)
    if one_pixel:
        # Object-side only: (L==1) and any 4-neighbor is not 1 -> 1 px line (no double thickness)
        Lpad = np.pad(L, 1, mode="constant", constant_values=0)
        has_neighbor_not1 = (
            (Lpad[1:-1, 2:] != 1) | (Lpad[1:-1, :-2] != 1) |
            (Lpad[2:, 1:-1] != 1) | (Lpad[:-2, 1:-1] != 1)
        )
        out = (L == 1) & has_neighbor_not1
    else:
        out = np.zeros(L.shape, dtype=bool)
        out[:-1, :] |= L[:-1, :] != L[1:, :]
        out[1:, :] |= L[:-1, :] != L[1:, :]
        out[:, :-1] |= L[:, :-1] != L[:, 1:]
        out[:, 1:] |= L[:, :-1] != L[:, 1:]
    return out


def build_mosaic(
    images: list[np.ndarray],
    ncols: int | None = None,
    pad: int = 2,
    fill: int = 0,
    labels: list[str] | list[int] | None = None,
    subtitle_height: int = 22,
) -> np.ndarray:
    """Arrange crop images in a grid. If labels is provided, draw a subtitle under each image (e.g. 'Cell N')."""
    if not images:
        return np.array([], dtype=np.uint8).reshape(0, 0)
    n = len(images)
    ncols = ncols or min(n, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = (n + ncols - 1) // ncols
    max_h = max(im.shape[0] for im in images)
    max_w = max(im.shape[1] for im in images)
    ndim = max(im.ndim for im in images)
    nch = 3
    for im in images:
        if im.ndim == 3:
            nch = im.shape[2]
            break
    use_labels = labels is not None and len(labels) >= n
    cell_h = max_h + pad + (subtitle_height if use_labels else 0)
    cell_w = max_w + pad
    if ndim == 3:
        mosaic = np.full((nrows * cell_h + pad, ncols * cell_w + pad, nch), fill, dtype=np.uint8)
    else:
        mosaic = np.full((nrows * cell_h + pad, ncols * cell_w + pad), fill, dtype=np.uint8)
    for idx, im in enumerate(images):
        im = np.asarray(im, dtype=np.uint8)
        if im.ndim == 2 and ndim == 3:
            im = np.stack([im] * nch, axis=-1)
        r, c = idx // ncols, idx % ncols
        y0 = pad + r * cell_h
        x0 = pad + c * cell_w
        h, w = im.shape[0], im.shape[1]
        if ndim == 3:
            mosaic[y0 : y0 + h, x0 : x0 + w, :] = im
        else:
            mosaic[y0 : y0 + h, x0 : x0 + w] = im
    if use_labels:
        from PIL import ImageDraw, ImageFont
        pil_mosaic = Image.fromarray(mosaic)
        draw = ImageDraw.Draw(pil_mosaic)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, subtitle_height - 6))
        except (OSError, Exception):
            font = ImageFont.load_default()
        for idx in range(n):
            r, c = idx // ncols, idx % ncols
            y0 = pad + r * cell_h + max_h + pad
            x0 = pad + c * cell_w
            text = f"Cell {labels[idx]}" if not isinstance(labels[idx], str) else str(labels[idx])
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(text, font=font)
            tx = x0 + (cell_w - tw) // 2
            ty = y0 + (subtitle_height - th) // 2
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
        mosaic = np.array(pil_mosaic)
    return mosaic


def process_image(
    image_path: str | Path,
    out_dir: str | Path,
    margin: int = MARGIN_DEFAULT,
    erosion_fg: int = EROSION_DEFAULT,
    erosion_bg: int = EROSION_BG_DEFAULT,
    bg_margin: int = BG_MARGIN_DEFAULT,
    use_bg_cells: bool = True,
    cellpose_model: str = "cyto3",
    cellpose_diameter: float | None = None,
    run_idisf: bool = True,
    segmenter: str = "idisf",
    idisf_n0: int = IDISF_N0_DEFAULT,
    idisf_iterations: int = IDISF_ITERATIONS_DEFAULT,
    idisf_f: int = IDISF_F_DEFAULT,
    idisf_c1: float = IDISF_C1_DEFAULT,
    idisf_c2: float = IDISF_C2_DEFAULT,
    pyift_delta: int = 1,
    pyift_gamma: float = 0.0,
    pyift_closest_root: bool = False,
    uoift_bin: str | Path | None = None,
    uoift_polarity: float = UOIFT_POLARITY_DEFAULT,
    uoift_spsize: int = UOIFT_SPSIZE_DEFAULT,
    uoift_grayscale: bool = False,
    uoift_fallback_idisf: bool = True,
    sicle_bin: str | Path | None = None,
    sicle_nf: int = SICLE_NF_DEFAULT,
    sicle_preset: str = "irregular",
    sicle_conn_opt: str | None = None,
    sicle_crit_opt: str | None = None,
    sicle_pen_opt: str = SICLE_PEN_OPT_DEFAULT,
    fusion_opening: int = 1,
    save_crops: bool = True,
    save_annotations: bool = True,
    save_idisf_masks: bool = True,
    save_reunited_mosaic: bool = True,
    mosaic_cols: int | None = None,
    gpu: bool = True,
    superres_image: str | Path | None = None,
    masks_precomputed: np.ndarray | None = None,
    scribble_source: str = "cellpose",
    activation_volume: np.ndarray | None = None,
    activation_layer: str = "out",
    activation_model: str = "cpsam",
    activation_fg_percentile: float = 66.0,
    activation_bg_low_percentile: float = 35.0,
    activation_reduce_mode: str = "l2",
) -> list[dict]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Optional: use a super-resolved image (e.g. produced by BiaPy super-resolution workflow)
    if superres_image is not None:
        sr_path = Path(superres_image)
        if not sr_path.exists():
            raise FileNotFoundError(f"Super-resolution image not found: {sr_path}")
        print(f"Using super-resolution image instead of original: {sr_path}")
        img = load_image(sr_path)
    else:
        img = load_image(image_path)
    sicle_conn, sicle_crit = resolve_sicle_path_cost(sicle_preset, sicle_conn_opt, sicle_crit_opt)

    if scribble_source not in ("cellpose", "activation"):
        raise ValueError("scribble_source must be 'cellpose' or 'activation'")

    act_vol = activation_volume
    if scribble_source == "activation" and act_vol is None:
        from cellpose.models import CellposeModel

        am = CellposeModel(gpu=gpu, model_type=activation_model)
        act_vol = _activation_extract_array(
            am.extract_activation_maps(
                img,
                layer=activation_layer,
                return_rgb=False,
                diameter=cellpose_diameter,
            )
        )
        try:
            from cellpose.activation_maps import activation_to_greyscale_u8

            grey_full = activation_to_greyscale_u8(act_vol, reduce_mode=activation_reduce_mode)
            Image.fromarray(grey_full).save(out_dir / f"{stem}_activation_grey.png")
        except Exception:
            pass

    if masks_precomputed is not None:
        masks = np.asarray(masks_precomputed, dtype=np.int32)
    else:
        masks = run_cellpose(img, model_type=cellpose_model, gpu=gpu, diameter=cellpose_diameter)
    labels = sorted(set(np.unique(masks)) - {0})
    if not labels:
        print("No cells detected by Cellpose.")
        return []

    # Save full-image Cellpose original result: colorized masks and overlay on image
    cellpose_masks_rgb = plot.mask_rgb(masks)
    cellpose_masks_path = out_dir / f"{stem}_cellpose_masks.png"
    Image.fromarray(cellpose_masks_rgb).save(cellpose_masks_path)
    img_for_overlay = np.asarray(img, dtype=np.float32)
    if img_for_overlay.ndim == 2:
        img_for_overlay = img_for_overlay[..., np.newaxis]
    cellpose_overlay_rgb = plot.mask_overlay(img_for_overlay, masks)
    cellpose_overlay_path = out_dir / f"{stem}_cellpose_overlay.png"
    Image.fromarray(cellpose_overlay_rgb).save(cellpose_overlay_path)
    # Full-image RGB border overlay for Cellpose masks (like fused RGB overlays)
    cellpose_outline_full = utils.masks_to_outlines(masks.astype(np.int32))
    cellpose_overlay_borders = overlay_borders(img, cellpose_outline_full, border_color_rgb=(0.0, 1.0, 1.0))
    cellpose_overlay_borders_path = out_dir / f"{stem}_overlay_cellpose_borders.png"
    Image.fromarray(cellpose_overlay_borders).save(cellpose_overlay_borders_path)

    results = []
    crop_list: list[tuple[int, np.ndarray]] = []  # (cell_id, crop) to keep mosaic labels in sync with files
    # Accumulate per-cell segmenter borders back into full-image (for final rebuild overlay) to avoid merged regions
    full_border_segmenter = None
    full_border_fused_opened = None
    full_border_sicle = None  # SICLE-alone full-image borders (used in fusion to build overlay_sicle.png)
    # Keep approximate cell centers (global coords) for labelling final overlay with cell ids
    cell_centers_segmenter: list[tuple[int, int, int]] = []
    if save_idisf_masks and run_idisf:
        H, W = masks.shape[:2]
        full_border_segmenter = np.zeros((H, W), dtype=bool)
        if segmenter == "fusion":
            full_border_sicle = np.zeros((H, W), dtype=bool)
            if fusion_opening > 0:
                full_border_fused_opened = np.zeros((H, W), dtype=bool)
    for cell_id in labels:
        img_crop, mask_crop, (r0, r1, c0, c1) = crop_with_margin(img, masks, cell_id, margin)
        h, w = mask_crop.shape[0], mask_crop.shape[1]
        full_mask_crop = masks[r0:r1, c0:c1]  # all labels in crop (for other-cells-as-background)
        if scribble_source == "activation":
            if act_vol is None:
                raise RuntimeError("activation volume missing (scribble_source=activation)")
            act_crop = act_vol[r0:r1, c0:c1, :]
            if act_crop.shape[0] != h or act_crop.shape[1] != w:
                raise ValueError(
                    f"Activation crop {act_crop.shape[:2]} does not match crop size {(h, w)}"
                )
            fg_mask, bg_mask, act_grey_crop, act_bw_crop = scribbles_from_activation(
                act_crop,
                mask_crop,
                full_mask_crop,
                cell_id,
                h,
                w,
                erosion_fg,
                erosion_bg,
                bg_margin,
                use_bg_cells,
                fg_percentile=activation_fg_percentile,
                bg_low_percentile=activation_bg_low_percentile,
                reduce_mode=activation_reduce_mode,
            )
        else:
            fg_mask = eroded_foreground_mask(mask_crop, erosion_fg)  # foreground erosion depth capped at EROSION_MAX
            bg_mask = background_mask_for_crop(
                h,
                w,
                full_mask_crop,
                cell_id,
                border_px=bg_margin,
                erosion_bg_pixels=erosion_bg,
                use_other_cells=use_bg_cells,
            )

        cellpose_outline = utils.masks_to_outlines(mask_crop.astype(np.int32))

        fg_coords = mask_to_scribble_coords(fg_mask)
        bg_coords = mask_to_scribble_coords(bg_mask)
        if not fg_coords:
            rows, cols = np.where(mask_crop > 0)
            if rows.size:
                cy, cx = int(rows.mean()), int(cols.mean())
                fg_coords = [(cx, cy)]
        if not bg_coords:
            h, w = mask_crop.shape
            bg_coords = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]

        crop_name = f"{stem}_cell{cell_id}"
        crop_path = out_dir / f"{crop_name}_crop.png"
        anno_path = out_dir / f"{crop_name}_anno.txt"

        if save_crops:
            crop_u8 = np.asarray(img_crop, dtype=np.uint8)
            Image.fromarray(crop_u8).save(crop_path)
        crop_list.append((cell_id, np.asarray(img_crop, dtype=np.uint8)))

        if save_annotations:
            write_idisf_annotation(anno_path, fg_coords, bg_coords)

        rec = {"cell_id": cell_id, "crop_path": crop_path, "anno_path": anno_path}

        if save_idisf_masks:
            if scribble_source == "activation":
                act_grey_path = out_dir / f"{crop_name}_activation_grey.png"
                act_bw_path = out_dir / f"{crop_name}_activation_bw.png"
                Image.fromarray(act_grey_crop).save(act_grey_path)
                Image.fromarray(act_bw_crop).save(act_bw_path)
                rec["activation_grey_crop_path"] = act_grey_path
                rec["activation_bw_path"] = act_bw_path
            # Plot of markers (foreground = green, background = red) on crop
            markers_plot = plot_markers(img_crop, fg_coords, bg_coords, fg_color=(0, 255, 0), bg_color=(255, 0, 0), radius=2)
            markers_path = out_dir / f"{crop_name}_markers.png"
            Image.fromarray(markers_plot).save(markers_path)
            rec["markers_path"] = markers_path
            cellpose_border_path = out_dir / f"{crop_name}_cellpose_border.png"
            Image.fromarray((cellpose_outline.astype(np.uint8) * 255)).save(cellpose_border_path)
            rec["cellpose_border_path"] = cellpose_border_path
            # Overlay Cellpose borders on crop (like iDISF overlayBorders; cyan)
            ovlay_cellpose = overlay_borders(img_crop, cellpose_outline, border_color_rgb=(0.0, 1.0, 1.0))
            ovlay_cellpose_path = out_dir / f"{crop_name}_overlay_cellpose.png"
            Image.fromarray(ovlay_cellpose).save(ovlay_cellpose_path)
            rec["overlay_cellpose_path"] = ovlay_cellpose_path
            # Image with inside Cellpose border kept, rest set to black
            cellpose_inside = mask_inside_border(img_crop, mask_crop, object_label=1)
            cellpose_inside_path = out_dir / f"{crop_name}_cellpose_inside.png"
            Image.fromarray(cellpose_inside).save(cellpose_inside_path)
            rec["cellpose_inside_path"] = cellpose_inside_path

        if run_idisf and fg_coords and bg_coords:
            seg_prefix = segmenter
            if segmenter == "idisf":
                label_img, border_img = run_idisf_on_crop(
                    img_crop, fg_coords, bg_coords,
                    n0=idisf_n0, iterations=idisf_iterations,
                    f=idisf_f, c1=idisf_c1, c2=idisf_c2,
                )
            elif segmenter == "pyift":
                label_img = run_pyift_dyntree_on_crop(
                    img_crop, fg_coords, bg_coords,
                    temp_dir=out_dir, crop_name=crop_name,
                    delta=pyift_delta, gamma=pyift_gamma, closest_root=pyift_closest_root,
                )
                border_img = None
            elif segmenter == "uoift":
                if uoift_bin is None:
                    uoift_bin = os.environ.get("UOIFT_BIN") or str(UNSUPSEG_DIR / "unsupseg")
                run_dir = out_dir / f"{crop_name}_uoift_run"
                try:
                    label_img = run_uoift_unsupseg_on_crop(
                        img_crop, fg_coords, run_dir=run_dir,
                        unsupseg_bin=Path(uoift_bin),
                        polarity=uoift_polarity, spsize=uoift_spsize,
                        force_grayscale=uoift_grayscale,
                    )
                    border_img = None
                except RuntimeError as e:
                    err = str(e)
                    if uoift_fallback_idisf and ("double free" in err or "exit -6" in err or "exit 134" in err):
                        print(f"Warning: UOIFT crashed for {crop_name} (known unsupseg/GFT bug). Using iDISF for this crop.")
                        label_img, border_img = run_idisf_on_crop(
                            img_crop, fg_coords, bg_coords,
                            n0=idisf_n0, iterations=idisf_iterations,
                            f=idisf_f, c1=idisf_c1, c2=idisf_c2,
                        )
                        seg_prefix = "idisf"
                    else:
                        raise
            elif segmenter == "sicle":
                # SICLE as standalone segmenter: unsupervised 2-SP segmentation, mapped to FG via scribbles
                if sicle_bin is None:
                    sicle_bin = os.environ.get("SICLE_BIN") or str(find_sicle_binary())
                label_img = run_sicle_on_crop(
                    img_crop, fg_coords, temp_dir=out_dir, crop_name=crop_name,
                    sicle_bin=Path(sicle_bin), nf=sicle_nf,
                    conn_opt=sicle_conn, crit_opt=sicle_crit, pen_opt=sicle_pen_opt,
                )
                border_img = None
            elif segmenter == "fusion":
                # (method_name, label_array) for each successful method
                partials: list[tuple[str, np.ndarray]] = []
                # iDISF
                L_idisf, _ = run_idisf_on_crop(
                    img_crop, fg_coords, bg_coords,
                    n0=idisf_n0, iterations=idisf_iterations,
                    f=idisf_f, c1=idisf_c1, c2=idisf_c2,
                )
                partials.append(("idisf", L_idisf))
                # PyIFT
                try:
                    L_pyift = run_pyift_dyntree_on_crop(
                        img_crop, fg_coords, bg_coords,
                        temp_dir=out_dir, crop_name=crop_name,
                        delta=pyift_delta, gamma=pyift_gamma, closest_root=pyift_closest_root,
                    )
                    partials.append(("pyift", L_pyift))
                except Exception as e:
                    print(f"Warning: PyIFT failed for {crop_name}: {e}. Fusing without PyIFT.")
                # SICLE (third method instead of UOIFT): unsupervised 2-SP segmentation, mapped to FG via scribbles
                try:
                    if sicle_bin is None:
                        sicle_bin = os.environ.get("SICLE_BIN") or str(find_sicle_binary())
                    L_sicle = run_sicle_on_crop(
                        img_crop, fg_coords, temp_dir=out_dir, crop_name=crop_name,
                        sicle_bin=Path(sicle_bin), nf=sicle_nf,
                        conn_opt=sicle_conn, crit_opt=sicle_crit, pen_opt=sicle_pen_opt,
                    )
                    partials.append(("sicle", L_sicle))
                except Exception as e:
                    print(f"Warning: SICLE failed for {crop_name}: {e}. Fusing without SICLE.")
                label_imgs = [L for _, L in partials]
                label_img = fuse_label_maps(label_imgs)
                border_img = None
                seg_prefix = "fused"
                # Save each partial (idisf, pyift, uoift) for comparison
                if save_idisf_masks:
                    for name, L in partials:
                        label_path = out_dir / f"{crop_name}_{name}_label.png"
                        Image.fromarray((np.clip(L, 0, 2) * 127).astype(np.uint8)).save(label_path)
                        rec[f"{name}_label_path"] = label_path
                        seg_border = label_borders(L)
                        ovlay_path = out_dir / f"{crop_name}_overlay_{name}.png"
                        Image.fromarray(overlay_borders(img_crop, seg_border, border_color_rgb=(0.0, 1.0, 0.0))).save(ovlay_path)
                        rec[f"overlay_{name}_path"] = ovlay_path
                        inside_path = out_dir / f"{crop_name}_{name}_inside.png"
                        Image.fromarray(mask_inside_border(img_crop, L, object_label=1)).save(inside_path)
                        rec[f"{name}_inside_path"] = inside_path
                        # Accumulate SICLE borders for full-image overlay_sicle.png (fusion only)
                        if name == "sicle" and full_border_sicle is not None:
                            full_border_sicle[r0:r1, c0:c1] |= seg_border
            else:
                raise ValueError(f"Unknown segmenter: {segmenter}")
            if save_idisf_masks:
                label_path = out_dir / f"{crop_name}_{seg_prefix}_label.png"
                Image.fromarray((np.clip(label_img, 0, 2) * 127).astype(np.uint8)).save(label_path)
                rec[f"{seg_prefix}_label_path"] = label_path
                if border_img is not None:
                    border_path = out_dir / f"{crop_name}_{seg_prefix}_border.png"
                    border_u8 = np.clip(np.asarray(border_img), 0, 255).astype(np.uint8)
                    Image.fromarray(border_u8, mode="L").save(border_path)
                    rec[f"{seg_prefix}_border_path"] = border_path
                seg_border = label_borders(label_img)
                ovlay_path = out_dir / f"{crop_name}_overlay_{seg_prefix}.png"
                ovlay = overlay_borders(img_crop, seg_border, border_color_rgb=(0.0, 1.0, 0.0))
                Image.fromarray(ovlay).save(ovlay_path)
                rec[f"overlay_{seg_prefix}_path"] = ovlay_path
                inside_only = mask_inside_border(img_crop, label_img, object_label=1)
                inside_path = out_dir / f"{crop_name}_{seg_prefix}_inside.png"
                Image.fromarray(inside_only).save(inside_path)
                rec[f"{seg_prefix}_inside_path"] = inside_path
                # Accumulate segmenter borders back into full image (for final rebuild overlay without merging regions)
                if full_border_segmenter is not None:
                    full_border_segmenter[r0:r1, c0:c1] |= seg_border
                    # Also store an approximate center for this cell (object region centroid) for id labelling
                    obj_mask = (label_img == 1)
                    if obj_mask.any():
                        ys, xs = np.where(obj_mask)
                        cy = int(ys.mean())
                        cx = int(xs.mean())
                        cell_centers_segmenter.append((cell_id, r0 + cy, c0 + cx))
                # Fusion: optionally apply opening and save before/after (before = above; after = opened)
                if segmenter == "fusion" and fusion_opening > 0:
                    label_opened = label_opening(label_img, fusion_opening, object_label=1, background_label=2)
                    label_opened_path = out_dir / f"{crop_name}_{seg_prefix}_opened_label.png"
                    Image.fromarray((np.clip(label_opened, 0, 2) * 127).astype(np.uint8)).save(label_opened_path)
                    rec[f"{seg_prefix}_opened_label_path"] = label_opened_path
                    seg_border_opened = label_borders(label_opened)
                    ovlay_opened_path = out_dir / f"{crop_name}_overlay_{seg_prefix}_opened.png"
                    Image.fromarray(overlay_borders(img_crop, seg_border_opened, border_color_rgb=(0.0, 1.0, 0.0))).save(ovlay_opened_path)
                    rec[f"overlay_{seg_prefix}_opened_path"] = ovlay_opened_path
                    inside_opened = mask_inside_border(img_crop, label_opened, object_label=1)
                    inside_opened_path = out_dir / f"{crop_name}_{seg_prefix}_opened_inside.png"
                    Image.fromarray(inside_opened).save(inside_opened_path)
                    rec[f"{seg_prefix}_opened_inside_path"] = inside_opened_path
                    # Accumulate opened fused borders back into full image
                    if full_border_fused_opened is not None:
                        full_border_fused_opened[r0:r1, c0:c1] |= seg_border_opened
        results.append(rec)

    # After processing all crops: build full-image rebuild overlay for the active segmenter (all cases)
    if save_idisf_masks and full_border_segmenter is not None:
        seg_name = "fused" if segmenter == "fusion" else segmenter
        full_overlay = overlay_borders(img, full_border_segmenter, border_color_rgb=(0.0, 1.0, 0.0))
        full_overlay_path = out_dir / f"{stem}_overlay_{seg_name}.png"
        Image.fromarray(full_overlay).save(full_overlay_path)
        # Also create a numbered variant to show from which crop each border comes (cell id)
        try:
            base_img = Image.fromarray(full_overlay.copy())
        except Exception:
            base_img = Image.fromarray(np.asarray(full_overlay, dtype=np.uint8))
        draw = ImageDraw.Draw(base_img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for cid, gy, gx in cell_centers_segmenter:
            text = str(cid)
            # small offset so text is not exactly on the border pixel
            tx = max(0, min(base_img.width - 1, gx))
            ty = max(0, min(base_img.height - 1, gy))
            draw.text((tx, ty), text, fill=(255, 255, 0), font=font)
        full_overlay_cells_path = out_dir / f"{stem}_overlay_{seg_name}_cells.png"
        base_img.save(full_overlay_cells_path)

        # Fusion: also save opened variant when enabled (plus numbered variant)
        if segmenter == "fusion" and fusion_opening > 0 and full_border_fused_opened is not None:
            full_overlay_opened = overlay_borders(img, full_border_fused_opened, border_color_rgb=(0.0, 1.0, 0.0))
            full_overlay_opened_path = out_dir / f"{stem}_overlay_fused_opened.png"
            Image.fromarray(full_overlay_opened).save(full_overlay_opened_path)
            try:
                base_img_opened = Image.fromarray(full_overlay_opened.copy())
            except Exception:
                base_img_opened = Image.fromarray(np.asarray(full_overlay_opened, dtype=np.uint8))
            draw_opened = ImageDraw.Draw(base_img_opened)
            for cid, gy, gx in cell_centers_segmenter:
                text = str(cid)
                tx = max(0, min(base_img_opened.width - 1, gx))
                ty = max(0, min(base_img_opened.height - 1, gy))
                draw_opened.text((tx, ty), text, fill=(255, 255, 0), font=font)
            full_overlay_opened_cells_path = out_dir / f"{stem}_overlay_fused_opened_cells.png"
            base_img_opened.save(full_overlay_opened_cells_path)

        # Fusion: also build full-image overlay for SICLE alone (overlay_sicle.png + _cells.png)
        if segmenter == "fusion" and full_border_sicle is not None:
            full_overlay_sicle = overlay_borders(img, full_border_sicle, border_color_rgb=(0.0, 1.0, 0.0))
            full_overlay_sicle_path = out_dir / f"{stem}_overlay_sicle.png"
            Image.fromarray(full_overlay_sicle).save(full_overlay_sicle_path)
            try:
                base_sicle = Image.fromarray(full_overlay_sicle.copy())
            except Exception:
                base_sicle = Image.fromarray(np.asarray(full_overlay_sicle, dtype=np.uint8))
            draw_sicle = ImageDraw.Draw(base_sicle)
            for cid, gy, gx in cell_centers_segmenter:
                text = str(cid)
                tx = max(0, min(base_sicle.width - 1, gx))
                ty = max(0, min(base_sicle.height - 1, gy))
                draw_sicle.text((tx, ty), text, fill=(255, 255, 0), font=font)
            full_overlay_sicle_cells_path = out_dir / f"{stem}_overlay_sicle_cells.png"
            base_sicle.save(full_overlay_sicle_cells_path)

    if save_reunited_mosaic and crop_list:
        crop_images = [c[1] for c in crop_list]
        cell_ids = [c[0] for c in crop_list]  # same order as files: cell_id from each crop
        mosaic = build_mosaic(crop_images, ncols=mosaic_cols, pad=4, fill=0, labels=cell_ids)
        mosaic_path = out_dir / f"{stem}_reunited_mosaic.png"
        Image.fromarray(mosaic).save(mosaic_path)
        print(f"Reunited mosaic saved: {mosaic_path}")

    # Optional: copy key artifacts into category folders to ease analysis
    if save_idisf_masks and results:
        answers_dir = out_dir / "answers"
        markers_dir = out_dir / "markers"
        overlays_dir = out_dir / "overlays"
        inside_dir = out_dir / "inside"
        answers_dir.mkdir(exist_ok=True)
        markers_dir.mkdir(exist_ok=True)
        overlays_dir.mkdir(exist_ok=True)
        inside_dir.mkdir(exist_ok=True)

        for rec in results:
            for key, value in rec.items():
                if not isinstance(value, (str, Path)):
                    continue
                src = Path(value)
                if not src.exists():
                    continue
                # Markers
                if key.endswith("markers_path"):
                    shutil.copy2(src, markers_dir / src.name)
                # All iDISF/PyIFT/UOIFT/fusion label maps (answers)
                elif key.endswith("label_path"):
                    shutil.copy2(src, answers_dir / src.name)
                # All crop overlays (Cellpose + segmenters)
                elif key.startswith("overlay_") and key.endswith("_path"):
                    shutil.copy2(src, overlays_dir / src.name)
                # Inside-only images
                elif key.endswith("inside_path"):
                    shutil.copy2(src, inside_dir / src.name)

    return results


def main():
    parser = argparse.ArgumentParser(description="Cellpose → crop (10px margin) → erosion → iDISF annotation → run iDISF")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--out_dir", "-o", default="./cellpose_idisf_out", help="Output directory")
    parser.add_argument("--margin", "-m", type=int, default=MARGIN_DEFAULT, help="Margin (px) around each cell crop")
    parser.add_argument("--erosion-fg", "-e", type=int, default=EROSION_DEFAULT, help="Foreground erosion depth (px from border), max %d" % EROSION_MAX)
    parser.add_argument("--erosion-bg", type=int, default=EROSION_BG_DEFAULT, help="Background (other-cells) erosion depth (px), max %d" % EROSION_MAX)
    parser.add_argument(
        "--no-erosion-bg",
        action="store_true",
        help="Disable erosion on other-cells (use full other-cell pixels as background; still use them as background)",
    )
    parser.add_argument(
        "--no-bg-cells",
        action="store_true",
        help="Do not use other cells inside the crop as background (only use border band).",
    )
    parser.add_argument("--bg-margin", type=int, default=BG_MARGIN_DEFAULT, help="Background scribble band width from crop border (px)")
    parser.add_argument("--cellpose_model", default="cyto3", help="Cellpose model type")
    parser.add_argument("--diameter", type=float, default=None, help="Cellpose cell diameter (optional)")
    parser.add_argument("--no_idisf", action="store_true", help="Only generate crops and annotations; do not run scribble segmentation")
    parser.add_argument(
        "--segmenter",
        choices=("idisf", "pyift", "uoift", "sicle", "fusion"),
        default="idisf",
        help="Segmenter: idisf, pyift, uoift, sicle (SICLE via RunSICLE), or fusion (majority vote of idisf+pyift+sicle)",
    )
    parser.add_argument("--idisf_n0", type=int, default=IDISF_N0_DEFAULT, help="iDISF n0 (GRID seeds); only when segmenter=idisf")
    parser.add_argument("--idisf_iterations", type=int, default=IDISF_ITERATIONS_DEFAULT, help="iDISF iterations; only when segmenter=idisf")
    parser.add_argument("--idisf-f", type=int, default=IDISF_F_DEFAULT, help="iDISF path-cost function ID (1–5). Higher values increase regularization terms.")
    parser.add_argument("--idisf-c1", type=float, default=IDISF_C1_DEFAULT, help="iDISF internal gradient scaling c1 (0<c1<=1).")
    parser.add_argument("--idisf-c2", type=float, default=IDISF_C2_DEFAULT, help="iDISF regularization/compactness weight c2 (0<c2<=1).")
    parser.add_argument("--pyift-delta", type=int, default=1, help="PyIFT DynTree delta; only when segmenter=pyift")
    parser.add_argument("--pyift-gamma", type=float, default=0.0, help="PyIFT DynTree gamma (neighbor dist); only when segmenter=pyift")
    parser.add_argument("--pyift-closest-root", action="store_true", help="Use DynTreeClosestRoot instead of DynTreeRoot; only when segmenter=pyift")
    parser.add_argument("--uoift-bin", default=None, help="Path to unsupseg binary (default: $UOIFT_BIN or unsupseg/unsupseg); only when segmenter=uoift")
    parser.add_argument("--uoift-polarity", type=float, default=UOIFT_POLARITY_DEFAULT, help="Boundary polarity in [-1,1]; only when segmenter=uoift")
    parser.add_argument("--uoift-spsize", type=int, default=UOIFT_SPSIZE_DEFAULT, help="Superpixel size (px); only when segmenter=uoift")
    parser.add_argument("--uoift-grayscale", action="store_true", help="Force grayscale input for unsupseg (avoids PPM double-free in some builds)")
    parser.add_argument("--no-uoift-fallback", action="store_true", help="Do not fall back to iDISF when UOIFT crashes (default: fall back)")
    parser.add_argument(
        "--sicle-bin",
        default=None,
        help="Path to RunSICLE (default: $SICLE_BIN, then <repo>/SICLE/bin/RunSICLE, then PIPELINE_UOIFT_SICLE/...)",
    )
    parser.add_argument("--sicle-nf", type=int, default=SICLE_NF_DEFAULT, help="Target number of SICLE superpixels in fusion (default: 2)")
    parser.add_argument(
        "--sicle-preset",
        choices=tuple(SICLE_PRESET_CONN_CRIT.keys()),
        default="irregular",
        help="SICLE IFT path-cost pair: irregular=fmax+minsc (RunSICLEIRREG-style); compact=fsum+maxsc (RunSICLECOMP-style). Ignored if --sicle-conn-opt/--sicle-crit-opt are set.",
    )
    parser.add_argument(
        "--sicle-conn-opt",
        default=None,
        choices=("fmax", "fsum", "gradvmax", "gradvmaxmul", "custom"),
        help="Override --sicle-preset: RunSICLE --conn-opt (use with --sicle-crit-opt). gradvmax/gradvmaxmul need saliency (--objsm).",
    )
    parser.add_argument(
        "--sicle-crit-opt",
        default=None,
        choices=("size", "minsc", "maxsc", "spread", "custom"),
        help="Override --sicle-preset: RunSICLE --crit-opt (use with --sicle-conn-opt).",
    )
    parser.add_argument(
        "--sicle-pen-opt",
        default=SICLE_PEN_OPT_DEFAULT,
        choices=("none", "obj", "bord", "osb", "bobs", "custom"),
        help="RunSICLE --pen-opt (seed relevance penalization; default: none).",
    )
    parser.add_argument("--fusion-opening", type=int, default=1, metavar="N", help="After fusion: morphological opening size (iterations). 0=disabled, 1=minimal (default). Saves both before and after when >0.")
    parser.add_argument("--no_reunited", action="store_true", help="Do not save reunited mosaic of crops")
    parser.add_argument("--mosaic_cols", type=int, default=None, help="Number of columns in reunited mosaic (default: auto)")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU for Cellpose")
    parser.add_argument(
        "--scribble-source",
        choices=("cellpose", "activation"),
        default="cellpose",
        help="How to place FG/BG seeds: cellpose (eroded mask + border) or activation "
        "(greyscale from last conv, same crops). Activation uses CP-SAM (see --activation-model).",
    )
    parser.add_argument(
        "--activation-npy",
        default=None,
        help="Optional [H,W,C] activation .npy aligned to the image; skips on-the-fly extraction.",
    )
    parser.add_argument(
        "--activation-model",
        default="cpsam",
        help="Cellpose model for extract_activation_maps when --scribble-source activation and no --activation-npy (default: cpsam).",
    )
    parser.add_argument("--activation-layer", default="out", choices=("neck", "out", "last_conv"), help="Hook layer for activations")
    parser.add_argument(
        "--activation-fg-percentile",
        type=float,
        default=66.0,
        help="Inside-cell FG seeds: pixels above this percentile of normalized activation (default: 66).",
    )
    parser.add_argument(
        "--activation-bg-low-percentile",
        type=float,
        default=35.0,
        help="Inside-cell extra BG seeds: pixels below this percentile (default: 35).",
    )
    parser.add_argument(
        "--activation-reduce-mode",
        default="l2",
        choices=("l2", "mean_abs", "max_abs"),
        help="Channel reduction for activation greyscale / scribbles (default: l2).",
    )
    args = parser.parse_args()
    erosion_bg = 0 if args.no_erosion_bg else args.erosion_bg
    use_bg_cells = not args.no_bg_cells

    activation_volume = None
    if args.activation_npy:
        a = _activation_extract_array(np.load(args.activation_npy))
        if a.ndim != 3:
            raise SystemExit(f"--activation-npy must be [H,W,C] or [1,H,W,C], got {a.shape}")
        activation_volume = a

    results = process_image(
        args.image,
        args.out_dir,
        margin=args.margin,
        erosion_fg=args.erosion_fg,
        erosion_bg=erosion_bg,
        bg_margin=args.bg_margin,
        use_bg_cells=use_bg_cells,
        cellpose_model=args.cellpose_model,
        cellpose_diameter=args.diameter,
        run_idisf=not args.no_idisf,
        segmenter=args.segmenter,
        idisf_n0=args.idisf_n0,
        idisf_iterations=args.idisf_iterations,
        idisf_f=args.idisf_f,
        idisf_c1=args.idisf_c1,
        idisf_c2=args.idisf_c2,
        pyift_delta=args.pyift_delta,
        pyift_gamma=args.pyift_gamma,
        pyift_closest_root=args.pyift_closest_root,
        uoift_bin=args.uoift_bin,
        uoift_polarity=args.uoift_polarity,
        uoift_spsize=args.uoift_spsize,
        uoift_grayscale=args.uoift_grayscale,
        uoift_fallback_idisf=not args.no_uoift_fallback,
        sicle_bin=args.sicle_bin,
        sicle_nf=args.sicle_nf,
        sicle_preset=args.sicle_preset,
        sicle_conn_opt=args.sicle_conn_opt,
        sicle_crit_opt=args.sicle_crit_opt,
        sicle_pen_opt=args.sicle_pen_opt,
        fusion_opening=args.fusion_opening,
        save_reunited_mosaic=not args.no_reunited,
        mosaic_cols=args.mosaic_cols,
        gpu=not args.no_gpu,
        scribble_source=args.scribble_source,
        activation_volume=activation_volume,
        activation_layer=args.activation_layer,
        activation_model=args.activation_model,
        activation_fg_percentile=args.activation_fg_percentile,
        activation_bg_low_percentile=args.activation_bg_low_percentile,
        activation_reduce_mode=args.activation_reduce_mode,
    )
    print(f"Processed {len(results)} cells. Outputs in {args.out_dir}")
    for r in results:
        print(f"  cell {r['cell_id']}: {r['crop_path'].name}, {r['anno_path'].name}", end="")
        seg_key = "fused_label_path" if args.segmenter == "fusion" else f"{args.segmenter}_label_path"
        if seg_key in r:
            print(f", {r[seg_key].name}")
        else:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
