#!/usr/bin/env python3
"""
Per-cell SICLE on **bounding boxes**, using **cropped cellprob** as saliency (same preprocessing as
the global SICLE step in ``reproduce_cellpose_pipeline.py``):

  sigmoid(cellprob) → uint8 → **Otsu on the crop (inside the cell mask only)** → two-piece map → uint8;
  pixels outside the current Cellpose instance are **zeroed** so the saliency map is single-cell only.

Each Cellpose instance is processed in its bbox (+ margin); ``run_sicle_on_crop`` (k≈2 superpixels)
uses mask pixels as foreground seeds (optional erosion via ``--fg-erosion-pixels``).

Merge behavior is configurable:
- default (conservative): only paste where ``SICLE_fg & (mask==cell_id)``
- optional: paste raw ``SICLE_fg`` in the bbox (no AND) via ``--disable-and-merge``
- optional: ``--disable-and-merge`` + ``--and-unless-round`` — paste raw only if the SICLE
  foreground looks round enough (circularity + solidity on largest CC); otherwise AND with Cellpose.

Prerequisites (e.g. from ``reproduce_cellpose_pipeline.py``)::

    step03_dP_cellprob.npz   (``cellprob_slice0`` or ``cellprob``)
    step04_masks_uint16.npy

Usage::

    cd doutorado/new_pipeline
    PYTHONPATH=../cellpose python percell_sicle_cellprob_pipeline.py \\
        --from-dir ./cp_flow_out -o ./percell_sicle_out \\
        --image ../path/to/GR07-1.svs_slice1.tiff

``RunOvlayBorders`` (manual or ``--run-ovlay-borders``) needs **PNM** inputs: use the written
``.ppm``/``.pgm`` files, not TIFF/PNG. The **label** image must be integer masks (e.g. merged labels),
not ``merged_percell_sicle_overlay.png``.

With ``--image``, ``merged_percell_sicle_overlay.png`` keeps the **original pixels** and draws
instance borders per ``--overlay-border-source``: **sicle** (merge only), **cellpose** (step04 only),
or **both** (Cellpose color under, SICLE on top; see ``--overlay-border-color`` /
``--overlay-cellpose-border-color`` / ``--overlay-border-thickness``).

Use ``--write-compare-vs-step04`` to also write ``compare_final_vs_step04/`` (``compare_segmentation_masks_diff``:
mask A = ``step04_masks_uint16.npy``, mask B = merged ``.npy``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# This file lives in ``doutorado/new_pipeline/``; Cellpose and SICLE live under ``doutorado/``.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_CELLPOSE_DIR = _REPO_ROOT / "cellpose"
for _d in (_PKG_DIR, _CELLPOSE_DIR):
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)


def load_cellprob_masks(from_dir: Path, cellprob_dir: Path | None = None):
    import numpy as np

    prob_root = Path(cellprob_dir) if cellprob_dir is not None else Path(from_dir)
    npz = prob_root / "step03_dP_cellprob.npz"
    mpath = from_dir / "step04_masks_uint16.npy"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    if not mpath.is_file():
        raise FileNotFoundError(mpath)
    z = np.load(npz)
    if "cellprob_slice0" in z.files:
        cellprob = np.asarray(z["cellprob_slice0"], dtype=np.float32)
    elif "cellprob" in z.files:
        cellprob = np.asarray(z["cellprob"], dtype=np.float32)
        if cellprob.ndim == 3 and cellprob.shape[0] == 1:
            cellprob = cellprob[0]
    else:
        raise SystemExit(f"{npz}: need cellprob_slice0 or cellprob")
    masks = np.load(mpath).astype(np.int32)
    dP: "np.ndarray | None" = None
    if "dP_slice01" in z.files:
        dP = np.asarray(z["dP_slice01"], dtype=np.float32)
    elif "dP_slice0" in z.files:
        dP = np.asarray(z["dP_slice0"], dtype=np.float32)
    elif "dP_full" in z.files:
        dP = np.asarray(z["dP_full"], dtype=np.float32)
        if dP.ndim == 4:
            dP = dP[:, 0]
    return cellprob, masks, dP


def cellprob_crop_to_saliency_u8(
    cellprob_crop: "np.ndarray",
    cell_mask: "np.ndarray | None" = None,
    *,
    linearize: bool = True,
) -> "np.ndarray":
    """Build per-cell saliency uint8 from a cellprob crop.

    Default (``linearize=True``): sigmoid → Otsu on masked pixels → two-piece linear map
    to [0, 0.5] below threshold and [0.5, 1] above (legacy behavior).

    ``linearize=False``: sigmoid probability only (no Otsu piecewise map); still zeros
    pixels outside ``cell_mask`` when provided.

    If ``cell_mask`` is given (bool or 0/1, same shape as ``cellprob_crop``), Otsu uses only
    pixels inside that mask and saliency is **0 outside** the instance — avoids lighting
    neighboring cells in the same bbox.
    """
    import cv2
    import numpy as np

    cp = np.asarray(cellprob_crop, dtype=np.float32)
    cp_prob = 1.0 / (1.0 + np.exp(-np.clip(cp, -50.0, 50.0)))

    inside: np.ndarray | None = None
    if cell_mask is not None:
        inside = np.asarray(cell_mask, dtype=bool)
        if inside.shape != cp_prob.shape:
            raise ValueError(
                f"cell_mask shape {inside.shape} != cellprob_crop shape {cp_prob.shape}"
            )

    if not linearize:
        sal = np.clip(cp_prob, 0.0, 1.0)
        if inside is not None:
            sal = sal.copy()
            sal[~inside] = 0.0
        return (sal * 255.0).astype(np.uint8)

    cp_u8 = (np.clip(cp_prob, 0.0, 1.0) * 255.0).astype(np.uint8)

    if inside is not None and inside.any():
        otsu_t, _ = cv2.threshold(
            cp_u8[inside].reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        otsu_t, _ = cv2.threshold(cp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    t = float(otsu_t) / 255.0
    eps = 1e-8
    sal = np.empty_like(cp_prob, dtype=np.float32)
    lo = cp_prob <= t
    hi = ~lo
    sal[lo] = 0.5 * cp_prob[lo] / max(t, eps)
    sal[hi] = 0.5 + 0.5 * (cp_prob[hi] - t) / max(1.0 - t, eps)
    sal = np.clip(sal, 0.0, 1.0)
    if inside is not None:
        sal[~inside] = 0.0
    return (sal * 255.0).astype(np.uint8)


def apply_saliency_threshold_u8(sal_u8: "np.ndarray", thr01: float) -> "np.ndarray":
    """Threshold saliency in [0,1]: values below threshold become 0."""
    import numpy as np

    s = np.asarray(sal_u8, dtype=np.uint8).copy()
    t = int(round(float(thr01) * 255.0))
    s[s < t] = 0
    return s


def apply_saliency_blur_u8(sal_u8: "np.ndarray", sigma: float) -> "np.ndarray":
    """Gaussian blur on saliency (in u8) to soften near-binary cellprob.

    Useful when feeding ``fmax + w_root^alpha`` (Eq. 2 of JMIV 2023): a smooth
    saliency with non-trivial gradients on the cell border is required to make
    ``alpha`` actually amplify arc costs.  ``sigma <= 0`` returns input unchanged.
    """
    import numpy as np

    if float(sigma) <= 0.0:
        return np.asarray(sal_u8, dtype=np.uint8)
    try:
        import cv2

        s = np.asarray(sal_u8, dtype=np.uint8)
        k = max(3, 2 * int(round(2.5 * float(sigma))) + 1)
        blurred = cv2.GaussianBlur(s, (k, k), float(sigma))
        return np.asarray(blurred, dtype=np.uint8)
    except ImportError:
        from scipy.ndimage import gaussian_filter

        s = np.asarray(sal_u8, dtype=np.float32)
        return np.clip(gaussian_filter(s, float(sigma)), 0.0, 255.0).astype(np.uint8)


def _normalize_to_u8(arr: "np.ndarray", mask: "np.ndarray | None" = None) -> "np.ndarray":
    """Linear normalize float array to uint8 using mask (or full array) min/max."""
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        vals = a[m] if m.any() else a.ravel()
    else:
        vals = a.ravel()
    vmin = float(vals.min()) if vals.size else 0.0
    vmax = float(vals.max()) if vals.size else 1.0
    if vmax <= vmin + 1e-8:
        out = np.zeros_like(a, dtype=np.uint8)
    else:
        out = np.clip((a - vmin) / (vmax - vmin) * 255.0, 0.0, 255.0).astype(np.uint8)
    if mask is not None:
        out[~np.asarray(mask, dtype=bool)] = 0
    return out


def image_grad_l_u8(crop_rgb: "np.ndarray") -> "np.ndarray":
    """|∇L| on LAB L-channel (edge prior, SEEDS-style)."""
    import cv2
    import numpy as np
    from skimage.color import rgb2lab

    rgb = np.asarray(crop_rgb[..., :3], dtype=np.float32) / 255.0
    lab = rgb2lab(rgb)
    l_ch = lab[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_ch, cv2.CV_32F, 0, 1, ksize=3)
    return _normalize_to_u8(np.hypot(gx, gy))


def flow_mag_u8(dP_crop: "np.ndarray") -> "np.ndarray":
    """Cellpose flow magnitude |dP| normalized to uint8."""
    import numpy as np

    dP = np.asarray(dP_crop, dtype=np.float32)
    if dP.ndim == 3 and dP.shape[0] == 2:
        mag = np.hypot(dP[0], dP[1])
    elif dP.ndim == 3 and dP.shape[-1] == 2:
        mag = np.hypot(dP[..., 0], dP[..., 1])
    else:
        raise ValueError(f"flow_mag_u8: expected dP (2,H,W) or (H,W,2), got {dP.shape}")
    return _normalize_to_u8(mag)


def enhance_saliency_u8(
    sal_u8: "np.ndarray",
    *,
    mode: str,
    cell_mask: "np.ndarray | None" = None,
    crop_rgb: "np.ndarray | None" = None,
    dP_crop: "np.ndarray | None" = None,
    mix_weight: float = 0.35,
    flow_gamma: float = 0.5,
) -> "np.ndarray":
    """Post-process base cellprob saliency with image gradient or flow prior.

    Modes:
      cellprob   — unchanged
      grad_l_mix — (1-w)*cellprob + w*|∇L|, renormalized inside mask
      flow_mul   — cellprob * (1 + γ * norm(|dP|))
    """
    import numpy as np

    base = np.asarray(sal_u8, dtype=np.float32)
    mask = np.asarray(cell_mask, dtype=bool) if cell_mask is not None else None

    if mode == "cellprob":
        return np.asarray(sal_u8, dtype=np.uint8)

    if mode == "grad_l_mix":
        if crop_rgb is None:
            raise ValueError("grad_l_mix requires RGB crop (--image)")
        w = float(np.clip(mix_weight, 0.0, 1.0))
        grad = image_grad_l_u8(crop_rgb).astype(np.float32)
        if mask is not None:
            grad[~mask] = 0.0
            base = base.copy()
            base[~mask] = 0.0
        mixed = (1.0 - w) * base + w * grad
        return _normalize_to_u8(mixed, mask=mask)

    if mode == "flow_mul":
        if dP_crop is None:
            raise ValueError("flow_mul requires dP from step03 npz")
        g = float(max(0.0, flow_gamma))
        flow = flow_mag_u8(dP_crop).astype(np.float32) / 255.0
        enhanced = base * (1.0 + g * flow)
        if mask is not None:
            enhanced[~mask] = 0.0
        return np.clip(enhanced, 0.0, 255.0).astype(np.uint8)

    raise ValueError(f"Unknown saliency mode: {mode}")


def bbox_for_label(
    masks: "np.ndarray", label: int, margin: int, h: int, w: int
) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.where(masks == label)
    if ys.size == 0:
        raise ValueError(f"empty mask for label {label}")
    r0, r1 = int(ys.min()) - margin, int(ys.max()) + 1 + margin
    c0, c1 = int(xs.min()) - margin, int(xs.max()) + 1 + margin
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(h, r1), min(w, c1)
    return r0, r1, c0, c1


def fg_scribble_coords(
    crop_bin: "np.ndarray",
    max_points: int = 400,
    erosion_pixels: int = 0,
) -> list[tuple[int, int]]:
    """(x, y) in crop coordinates for SICLE foreground seeds.

    If ``erosion_pixels`` is 0 (default), seeds come from the full binary mask. If > 0, the mask is
    eroded that many iterations (3×3); if erosion empties the mask, falls back to the uneroded mask.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion

    m = crop_bin.astype(bool)
    if erosion_pixels > 0:
        er = binary_erosion(m, structure=np.ones((3, 3), dtype=bool), iterations=int(erosion_pixels))
        if not er.any():
            er = m
    else:
        er = m
    ys, xs = np.where(er)
    if ys.size == 0:
        return []
    idx = np.linspace(0, ys.size - 1, num=min(max_points, ys.size), dtype=int)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def _outline_only_overlay(
    img: "np.ndarray",
    masks: "np.ndarray",
    *,
    border_color_rgb: tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 1,
) -> "np.ndarray":
    """RGB uint8 image unchanged except on instance borders (Cellpose ``masks_to_outlines``)."""
    import cv2
    import numpy as np
    from cellpose import utils

    base = np.asarray(img[..., :3], dtype=np.uint8).copy()
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError("_outline_only_overlay: expected RGB [H,W,3]")
    m = np.asarray(masks, dtype=np.int32)
    outlines = utils.masks_to_outlines(m).astype(bool)
    if border_thickness > 1:
        k = max(3, 2 * int(border_thickness) - 1)
        ker = np.ones((k, k), dtype=np.uint8)
        outlines = cv2.dilate(outlines.astype(np.uint8), ker, iterations=1).astype(bool)
    r, g, b = border_color_rgb
    base[outlines, 0] = r
    base[outlines, 1] = g
    base[outlines, 2] = b
    return base


def _selective_border_overlay(
    img: "np.ndarray",
    *,
    merged_sicle: "np.ndarray",
    cellpose_masks: "np.ndarray | None",
    source: str,
    border_sicle_rgb: tuple[int, int, int],
    border_cellpose_rgb: tuple[int, int, int],
    border_thickness: int,
) -> "np.ndarray":
    """Contornos sobre RGB: ``sicle`` (merge final), ``cellpose`` (step04), ou ``both`` (Cellpose primeiro, SICLE por cima)."""
    import cv2
    import numpy as np
    from cellpose import utils

    base = np.asarray(img[..., :3], dtype=np.uint8).copy()
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError("_selective_border_overlay: expected RGB [H,W,3]")

    def outlines_of(m: "np.ndarray") -> "np.ndarray":
        o = utils.masks_to_outlines(np.asarray(m, dtype=np.int32)).astype(bool)
        if border_thickness > 1:
            k = max(3, 2 * int(border_thickness) - 1)
            ker = np.ones((k, k), dtype=np.uint8)
            o = cv2.dilate(o.astype(np.uint8), ker, iterations=1).astype(bool)
        return o

    src = source.strip().lower()
    if src == "cellpose":
        if cellpose_masks is None:
            raise ValueError("cellpose_masks required for overlay-border-source=cellpose")
        o = outlines_of(cellpose_masks)
        r, g, b = border_cellpose_rgb
        base[o, 0], base[o, 1], base[o, 2] = r, g, b
    elif src == "both":
        if cellpose_masks is None:
            raise ValueError("cellpose_masks required for overlay-border-source=both")
        oc = outlines_of(cellpose_masks)
        r, g, b = border_cellpose_rgb
        base[oc, 0], base[oc, 1], base[oc, 2] = r, g, b
        osicle = outlines_of(merged_sicle)
        r, g, b = border_sicle_rgb
        base[osicle, 0], base[osicle, 1], base[osicle, 2] = r, g, b
    elif src == "sicle":
        o = outlines_of(merged_sicle)
        r, g, b = border_sicle_rgb
        base[o, 0], base[o, 1], base[o, 2] = r, g, b
    else:
        raise ValueError(f"unknown overlay-border-source: {source!r}")
    return base


def _translucent_mask_overlay(
    rgb: "np.ndarray",
    masks: "np.ndarray",
    *,
    color_rgb: tuple[int, int, int],
    alpha: float = 0.45,
) -> "np.ndarray":
    """Pinta toda a área de instâncias (label > 0) com uma única cor ``color_rgb`` e mistura com ``rgb``.

    Pixels de fundo (label 0) permanecem inalterados; ``alpha`` é a opacidade (0 = invisível, 1 = sólida).
    """
    import numpy as np

    base = np.asarray(rgb[..., :3], dtype=np.uint8).copy()
    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError("_translucent_mask_overlay: expected RGB [H,W,3]")
    m = np.asarray(masks, dtype=np.int32)
    if m.shape[:2] != base.shape[:2]:
        raise ValueError(
            f"mask shape {m.shape} != image shape {base.shape[:2]}"
        )

    fg = m > 0
    if not fg.any():
        return base
    a = float(max(0.0, min(1.0, alpha)))
    r, g, b = (int(c) for c in color_rgb)
    color = np.array([r, g, b], dtype=np.float32)
    blended = base.astype(np.float32)
    blended[fg] = (1.0 - a) * blended[fg] + a * color
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _cell_label_centroids(
    masks: "np.ndarray",
    labels: list[int],
) -> list[tuple[int, int, int]]:
    """(label_id, cy, cx) in full-image coordinates from the Cellpose instance mask."""
    import numpy as np

    centers: list[tuple[int, int, int]] = []
    m = np.asarray(masks)
    for lab in labels:
        ys, xs = np.where(m == lab)
        if ys.size:
            centers.append((lab, int(ys.mean()), int(xs.mean())))
    return centers


def _overlay_with_cell_label_ids(
    rgb: "np.ndarray",
    centers: list[tuple[int, int, int]],
    *,
    text_rgb: tuple[int, int, int] = (255, 255, 0),
) -> "np.ndarray":
    """Draw Cellpose label IDs at instance centroids (same idea as ``cellpose_to_idisf_pipeline``)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    base = Image.fromarray(np.asarray(rgb[..., :3], dtype=np.uint8).copy())
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, Exception):
        font = ImageFont.load_default()
    r, g, b = text_rgb
    fill = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    for cid, cy, cx in centers:
        text = str(cid)
        tx = max(0, min(base.width - 1, cx))
        ty = max(0, min(base.height - 1, cy))
        draw.text((tx, ty), text, fill=fill, font=font)
    return np.asarray(base)


def _find_run_ovlay_borders_binary(sicle_bin: Path) -> Path | None:
    p = Path(sicle_bin).resolve().parent / "RunOvlayBorders"
    return p if p.is_file() else None


def _run_run_ovlay_borders(
    ovlay_bin: Path,
    img_path: Path,
    labels_path: Path,
    out_path: Path,
) -> tuple[bool, str]:
    """``RunOvlayBorders`` expects PNM-style inputs (IFT); avoid TIFF/PNG for ``--img``/``--labels``."""
    cmd = [
        str(ovlay_bin),
        "--img",
        str(img_path.resolve()),
        "--labels",
        str(labels_path.resolve()),
        "--out",
        str(out_path.resolve()),
        "--rgb",
        "0,1,0",
        "--thick",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    msg = (proc.stderr or "") + (proc.stdout or "")
    return proc.returncode == 0 and out_path.exists(), msg


def _write_ovlay_labels_pgm(merged: "np.ndarray", path: Path) -> None:
    """8-bit PGM label image (RunOvlayBorders / ift). Requires max instance id ≤ 255."""
    import numpy as np
    from PIL import Image

    mx = int(np.max(merged))
    if mx > 255:
        raise ValueError(
            f"RunOvlayBorders (PGM): max label is {mx} (> 255). "
            "IFT expects 8-bit labels; skip --run-ovlay-borders or remap labels."
        )
    Image.fromarray(np.asarray(merged, dtype=np.uint8), mode="L").save(path)


def _clean_sicle_fg_mask(
    obj: "np.ndarray",
    *,
    closing_radius: int = 0,
    fill_holes: bool = False,
    keep_largest_cc: bool = False,
) -> tuple["np.ndarray", dict]:
    """Topological cleanup on the SICLE FG mask of a single cell crop.

    Steps (applied only if requested, in order):
      1. ``closing_radius`` ≥ 1: binary closing with a ``(2r+1)`` square structuring element.
      2. ``fill_holes``: fill any 4-/8-connected background hole **interior** to the FG.
      3. ``keep_largest_cc``: keep only the largest 8-connected FG component.

    Returns the cleaned bool mask and a dict with diagnostic counters.
    """
    import numpy as np
    from scipy.ndimage import binary_closing, binary_fill_holes, label

    diag = {"closing_radius": int(closing_radius), "holes_filled_px": 0, "cc_dropped_px": 0}
    m = np.asarray(obj, dtype=bool).copy()
    if not m.any():
        return m, diag

    if closing_radius >= 1:
        k = max(3, 2 * int(closing_radius) + 1)
        m = binary_closing(m, structure=np.ones((k, k), dtype=bool), iterations=1)

    if fill_holes:
        filled = binary_fill_holes(m)
        diag["holes_filled_px"] = int(filled.sum() - m.sum())
        m = filled

    if keep_largest_cc:
        lab, n = label(m, structure=np.ones((3, 3), dtype=bool))
        if n > 1:
            areas = np.bincount(lab.ravel())
            areas[0] = 0
            keep = int(areas.argmax())
            dropped = int(m.sum() - areas[keep])
            m = lab == keep
            diag["cc_dropped_px"] = dropped

    return m, diag


def _write_percell_debug_outputs(
    cell_dir: Path,
    input_crop_rgb: "np.ndarray | None",
    sal_u8: "np.ndarray",
    output_in_cell: "np.ndarray",
) -> None:
    """Write per-cell debug assets: input image, saliency map, and output-in-cell mask."""
    import numpy as np
    from PIL import Image

    cell_dir.mkdir(parents=True, exist_ok=True)
    if input_crop_rgb is None:
        # Fallback when no --image is provided: visualize saliency as gray RGB.
        input_crop_rgb = np.stack([sal_u8, sal_u8, sal_u8], axis=-1)
    input_u8 = np.asarray(input_crop_rgb[..., :3], dtype=np.uint8)
    sal_u8 = np.asarray(sal_u8, dtype=np.uint8)
    out_u8 = (np.asarray(output_in_cell, dtype=bool).astype(np.uint8) * 255)

    Image.fromarray(input_u8).save(cell_dir / "input_image.png")
    Image.fromarray(sal_u8, mode="L").save(cell_dir / "saliency_map.png")
    Image.fromarray(out_u8, mode="L").save(cell_dir / "output_in_cell.png")


def sicle_fg_shape_circularity_solidity(fg_bool: "np.ndarray") -> tuple[float, float]:
    """Métricas no maior componente conexo do foreground SICLE (crop).

    * **circularity** — ``4π·área/perímetro²`` (1 = círculo; quadrado ≈ 0,79).
    * **solidity** — ``área / área_do_convex_hull`` (1 = convexo).

    Valores em [0, 1]. Máscara vazia → (0, 0).
    """
    import cv2
    import numpy as np

    m = (np.asarray(fg_bool, dtype=np.uint8) * 255).astype(np.uint8)
    if m.size == 0 or int(m.max()) == 0:
        return 0.0, 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 1.0:
        return 0.0, 0.0
    perim = float(cv2.arcLength(c, True))
    if perim <= 1e-6:
        return 0.0, 0.0
    circ = (4.0 * np.pi * area) / (perim * perim)
    circ = float(np.clip(circ, 0.0, 1.0))
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    sol = float(area / hull_area) if hull_area > 1e-6 else 0.0
    sol = float(np.clip(sol, 0.0, 1.0))
    return circ, sol


def _fg_label_from_sicle_output(
    label_raw: "np.ndarray", fg_coords: list[tuple[int, int]]
) -> int | None:
    """Pick the superpixel label that covers the most FG seed pixels."""
    import numpy as np

    h, w = label_raw.shape[:2]
    uniq = np.unique(label_raw)
    uniq = uniq[uniq > 0]
    if uniq.size == 0:
        return None
    cnt: dict[int, int] = {}
    for u in uniq:
        cnt[int(u)] = sum(
            1 for (x, y) in fg_coords if 0 <= y < h and 0 <= x < w and label_raw[y, x] == u
        )
    if not cnt or max(cnt.values()) == 0:
        return None
    return max(cnt, key=lambda u: cnt[u])


def _label_image_to_fg_uint8(label_raw: "np.ndarray", fg_coords: list[tuple[int, int]]) -> "np.ndarray":
    """Convert a SICLE label image to uint8 FG mask (1=object, 2=background)."""
    import numpy as np

    obj_label = _fg_label_from_sicle_output(label_raw, fg_coords)
    if obj_label is None:
        return np.zeros_like(label_raw, dtype=np.uint8)
    out = np.full_like(label_raw, 2, dtype=np.uint8)
    out[label_raw == obj_label] = 1
    return out


def compute_adaptive_sicle_seeds(
    area_px: int,
    crop_h: int,
    crop_w: int,
    n0: int,
    nf: int,
    *,
    ref_area: float = 1500.0,
) -> tuple[int, int]:
    """Scale N0 with sqrt(area); bump Nf for large cells (Belém multiscale schedule)."""
    import math

    max_n0 = max(4, crop_h * crop_w - 1)
    scale = math.sqrt(max(int(area_px), 64) / max(ref_area, 1.0))
    n0_adapt = int(round(float(n0) * scale))
    n0_adapt = max(30, min(n0_adapt, max_n0 - 1))

    if area_px < 600:
        nf_adapt = 2
    elif area_px < 2500:
        nf_adapt = int(nf)
    else:
        nf_adapt = max(int(nf), 3)
    nf_adapt = max(2, min(nf_adapt, n0_adapt - 1))
    return n0_adapt, nf_adapt


def _mask_iou(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np

    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    return float(inter / union) if union > 0 else 0.0


def select_multiscale_fg_mask(
    fg_masks: list["np.ndarray"],
    cell_mask: "np.ndarray",
    mode: str,
    *,
    min_solidity: float = 0.0,
) -> tuple["np.ndarray", int, float]:
    """Pick one FG bool mask from multiscale SICLE outputs (Veta-style selection).

    Returns (mask, scale_index, score). ``last`` uses the finest scale.
    """
    import numpy as np

    if not fg_masks:
        raise ValueError("select_multiscale_fg_mask: empty candidate list")
    cell_bool = np.asarray(cell_mask, dtype=bool)

    if mode == "last":
        idx = len(fg_masks) - 1
        return fg_masks[idx], idx, 0.0

    best_idx = len(fg_masks) - 1
    best_score = -1.0
    for i, raw in enumerate(fg_masks):
        fg = np.asarray(raw, dtype=bool)
        if not fg.any():
            continue
        _circ, sol = sicle_fg_shape_circularity_solidity(fg)
        if min_solidity > 0.0 and sol < min_solidity:
            continue
        iou = _mask_iou(fg, cell_bool)
        if mode == "veta_solidity":
            score = sol + 1e-4 * iou
        elif mode == "veta_composite":
            score = sol * (0.35 + 0.65 * iou)
        else:
            raise ValueError(f"Unknown multiscale selection mode: {mode}")
        if score > best_score:
            best_score = score
            best_idx = i

    return fg_masks[best_idx], best_idx, best_score


SICLE_N0_DEFAULT = 500
SICLE_NF_DEFAULT = 2
SICLE_ALPHA_DEFAULT = 0.9
SICLE_MAXITERS_DEFAULT = 22
SICLE_IRREG_DEFAULT = 0.12
SICLE_ADHR_DEFAULT = 16
SICLE_PEN_OPT_DEFAULT = "none"


def _run_compare_final_vs_step04(from_dir: Path, out_dir: Path) -> None:
    """``compare_segmentation_masks_diff``: A=step04, B=merged → ``compare_final_vs_step04/``."""
    import subprocess
    import sys

    step04 = from_dir / "step04_masks_uint16.npy"
    merged = out_dir / "merged_percell_sicle_masks_int32.npy"
    if not step04.is_file():
        print(f"Warning: skip compare vs step04 (missing {step04})")
        return
    if not merged.is_file():
        return
    script = Path(__file__).resolve().parent / "compare_segmentation_masks_diff.py"
    cmp_dir = out_dir / "compare_final_vs_step04"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mask-a",
            str(step04.resolve()),
            "--mask-b",
            str(merged.resolve()),
            "-o",
            str(cmp_dir.resolve()),
            "--also-save-diff-only-rgb",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        msg = ((r.stderr or "") + (r.stdout or ""))[:1200]
        print(f"Warning: compare vs step04 failed (exit {r.returncode}): {msg}")
    else:
        print(f"Wrote {cmp_dir}/ (mask A=step04, mask B=merged)")


def find_sicle_binary() -> Path:
    env_path = os.environ.get("SICLE_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"SICLE_BIN={env_path} does not exist")
    candidates = [
        _REPO_ROOT / "SICLE" / "bin" / "RunSICLE",
        _REPO_ROOT / "PIPELINE_UOIFT_SICLE" / "uoift_sicle" / "SICLE" / "bin" / "RunSICLE",
        Path.home() / "SICLE" / "bin" / "RunSICLE",
        Path("/usr/local/bin/RunSICLE"),
    ]
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


def resolve_sicle_path_cost(
    preset: str,
    conn_opt: str | None,
    crit_opt: str | None,
) -> tuple[str, str]:
    preset_map = {
        "irregular": ("fmax", "minsc"),
        "compact": ("fsum", "maxsc"),
    }
    if conn_opt is not None or crit_opt is not None:
        if conn_opt is None or crit_opt is None:
            raise ValueError(
                "Override SICLE path costs with both --sicle-conn-opt and --sicle-crit-opt, or use neither (preset applies)."
            )
        return conn_opt, crit_opt
    return preset_map[preset]


def run_sicle_on_crop(
    img_crop: "np.ndarray",
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
    saliency_u8: "np.ndarray | None" = None,
    multiscale: bool = False,
    scale_select: str = "last",
    scale_min_solidity: float = 0.0,
    cell_mask: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Run SICLE on a per-cell crop.

    ``img_crop`` is the input image (typically the **original RGB**).
    ``saliency_u8`` is an OPTIONAL grayscale saliency map (e.g. cellprob-derived).
    When provided it is always passed as ``--objsm`` to the binary, so canonical
    saliency-aware path costs (e.g. ``fmax`` with ``w_root^(1+alpha*|sal_diff|)``,
    Eq.2 of Belém et al., JMIV 2023) are properly enabled and ``alpha`` actually
    has an effect — not only for ``gradvmax``/``gradvmaxmul``.

    With ``multiscale=True``, RunSICLE writes ``out_1.pgm`` … ``out_K.pgm``; the FG
    superpixel is chosen per scale and the best scale is picked via ``scale_select``.
    """
    import numpy as np
    from PIL import Image

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
    has_saliency = False
    if saliency_u8 is not None:
        sal_gray = np.asarray(saliency_u8, dtype=np.uint8)
        if sal_gray.ndim == 3:
            sal_gray = sal_gray[..., 0]
        Image.fromarray(sal_gray, mode="L").save(objsm_path)
        has_saliency = True
    elif conn_opt in ("gradvmax", "gradvmaxmul"):
        r = img_u8_rgb[:, :, 0].astype(np.float32)
        g = img_u8_rgb[:, :, 1].astype(np.float32)
        b = img_u8_rgb[:, :, 2].astype(np.float32)
        if np.allclose(r, g) and np.allclose(r, b):
            sal_gray = img_u8_rgb[:, :, 0]
        else:
            sal_gray = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(sal_gray, mode="L").save(objsm_path)
        has_saliency = True

    def _load_label(path: Path) -> "np.ndarray":
        label_raw = np.array(Image.open(path), dtype=np.int32)
        if label_raw.ndim > 2:
            label_raw = label_raw.squeeze()
        if label_raw.ndim > 2:
            label_raw = label_raw[..., 0]
        return label_raw

    def _run_once(current_n0: int) -> tuple[bool, str, list[Path]]:
        for old in temp_dir.glob(f"{crop_name}_sicle_output*.pgm"):
            old.unlink(missing_ok=True)
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
        if has_saliency:
            cmd += ["--objsm", str(objsm_path)]
        if multiscale:
            cmd += ["--multiscale"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if multiscale:
            scale_paths = sorted(
                temp_dir.glob(f"{crop_name}_sicle_output_*.pgm"),
                key=lambda p: int(re.search(r"_(\d+)\.", p.name).group(1)),
            )
            ok = proc.returncode == 0 and len(scale_paths) > 0
            return ok, proc.stderr, scale_paths
        ok = proc.returncode == 0 and out_path.exists()
        return ok, proc.stderr, [out_path] if ok else []

    ok, stderr, label_paths = _run_once(n0)
    if not ok and "Invalid N0 value of" in stderr and "It must be within ]2," in stderr:
        m = re.search(r"It must be within ]2,(\d+)\[", stderr)
        if m:
            max_allowed = int(m.group(1)) - 1
            if max_allowed > 2:
                n0_adapted = max(3, max_allowed)
                print(f"Warning: SICLE N0={n0} too large for {crop_name}. Retrying with N0={n0_adapted}.")
                ok, stderr, label_paths = _run_once(n0_adapted)
    if not ok:
        raise RuntimeError(f"SICLE failed for {crop_name}: {stderr}")

    if not multiscale:
        label_raw = _load_label(label_paths[-1])
        return _label_image_to_fg_uint8(label_raw, fg_coords)

    if scale_select == "last":
        label_raw = _load_label(label_paths[-1])
        return _label_image_to_fg_uint8(label_raw, fg_coords)

    if cell_mask is None:
        raise ValueError("multiscale scale selection requires cell_mask")

    fg_candidates: list[np.ndarray] = []
    for lp in label_paths:
        label_raw = _load_label(lp)
        fg_u8 = _label_image_to_fg_uint8(label_raw, fg_coords)
        fg_candidates.append(fg_u8 == 1)

    chosen, _scale_idx, _score = select_multiscale_fg_mask(
        fg_candidates,
        cell_mask,
        scale_select,
        min_solidity=scale_min_solidity,
    )
    out = np.full(chosen.shape, 2, dtype=np.uint8)
    out[chosen] = 1
    return out


def main() -> int:
    import numpy as np
    from cellpose import plot

    p = argparse.ArgumentParser(description="Per-bbox SICLE on cellprob saliency → merged label image.")
    p.add_argument("--from-dir", type=str, required=True, help="Folder with step04 masks npy (detector, e.g. CellViT)")
    p.add_argument(
        "--cellprob-from-dir",
        type=str,
        default=None,
        help="Optional folder with step03_dP_cellprob.npz (e.g. Cellpose cp_flow). "
        "Masks still come from --from-dir.",
    )
    p.add_argument("-o", "--out-dir", type=str, default="./percell_sicle_out")
    p.add_argument("--margin", type=int, default=8, help="Pixels to pad each cell bbox")
    p.add_argument("--min-cell-area", type=int, default=64, help="Skip SICLE for tiny regions; keep Cellpose mask")
    p.add_argument(
        "--fg-erosion-pixels",
        type=int,
        default=0,
        metavar="N",
        help="Erode the cell mask before sampling FG scribbles for SICLE (0 = off, default; try 1 for interior-only seeds)",
    )
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional RGB path; writes merged_percell_sicle_overlay.png (original pixels + colored borders only)",
    )
    p.add_argument(
        "--overlay-border-thickness",
        type=int,
        default=1,
        help="Border line thickness in pixels for merged_percell_sicle_overlay.png (>=1)",
    )
    p.add_argument(
        "--overlay-border-color",
        type=str,
        default="0,255,0",
        help="Overlay: R,G,B for SICLE borders (default 0,255,0); in both mode, SICLE is drawn on top.",
    )
    p.add_argument(
        "--overlay-border-source",
        choices=("sicle", "cellpose", "both"),
        default="sicle",
        help=(
            "Which labels to outline on merged_percell_sicle_overlay.png: "
            "sicle=merged result only; cellpose=step04 only; both=Cellpose then SICLE (see colors)."
        ),
    )
    p.add_argument(
        "--overlay-cellpose-border-color",
        type=str,
        default="255,0,0",
        help="R,G,B for Cellpose (step04) outlines when overlay-border-source is cellpose or both.",
    )
    p.add_argument(
        "--write-compare-vs-step04",
        action="store_true",
        help=(
            "After merge, run compare_segmentation_masks_diff.py: "
            "mask A = from-dir/step04_masks_uint16.npy, mask B = merged_percell_sicle_masks_int32.npy; "
            "output directory compare_final_vs_step04/ (diff_rgb_explained.png, diff_stats.txt, …)."
        ),
    )
    p.add_argument(
        "--overlay-number-labels",
        action="store_true",
        help=(
            "With --image, also write merged_percell_sicle_overlay_numbered.png: same border overlay "
            "plus each Cellpose label ID at the instance centroid (as in cellpose_to_idisf_pipeline "
            "*_overlay_*_cells.png)."
        ),
    )
    p.add_argument(
        "--overlay-label-color",
        type=str,
        default="255,255,0",
        help="R,G,B for label IDs on the numbered overlay (default 255,255,0).",
    )
    p.add_argument(
        "--translucent-mask-overlay",
        action="store_true",
        help=(
            "With --image, also write two semi-transparent fill overlays per instance: "
            "merged_percell_sicle_translucent_sicle.png (merged result) and "
            "merged_percell_sicle_translucent_cellpose.png (step04 mask). "
            "Each label gets a unique pseudo-random color."
        ),
    )
    p.add_argument(
        "--translucent-alpha",
        type=float,
        default=0.45,
        help="Opacity (0..1) for --translucent-mask-overlay (default 0.45).",
    )
    p.add_argument("--sicle-preset", choices=("irregular", "compact"), default="irregular")
    p.add_argument("--sicle-conn-opt", type=str, default=None)
    p.add_argument("--sicle-crit-opt", type=str, default=None)
    p.add_argument(
        "--sicle-pen-opt",
        type=str,
        default=SICLE_PEN_OPT_DEFAULT,
        choices=("none", "obj", "bord", "osb", "bobs"),
        help=(
            "Seed-relevance penalization (Belém et al. JMIV 2023). "
            "obj=concentrate seeds inside object; bord=focus on saliency borders; "
            "osb=spread inside object; bobs=spread inside, contrast outside. "
            "Default: none."
        ),
    )
    p.add_argument(
        "--sicle-use-rgb-image",
        action="store_true",
        help=(
            "Feed SICLE with the original RGB crop and the cellprob saliency separately "
            "(via --objsm). When NOT set (default), the saliency map is fed as both image "
            "and saliency, which is the legacy behavior."
        ),
    )
    p.add_argument(
        "--sicle-min-solidity",
        type=float,
        default=0.0,
        help=(
            "Veta-style shape filter (Belém et al., UOIFT-SICLE 2024): if SICLE output has "
            "solidity < this threshold, fall back to the Cellpose mask. Default: 0.0 (off). "
            "Recommended: 0.80."
        ),
    )
    p.add_argument("--sicle-n0", type=int, default=SICLE_N0_DEFAULT)
    p.add_argument("--sicle-nf", type=int, default=SICLE_NF_DEFAULT)
    p.add_argument(
        "--sicle-adaptive-seeds",
        action="store_true",
        help=(
            "Adapt N0/Nf per cell from bbox area: N0 scales with sqrt(area); "
            "Nf=3 for large cells (>2500 px). Uses --sicle-n0/--sicle-nf as bases."
        ),
    )
    p.add_argument(
        "--sicle-adaptive-ref-area",
        type=float,
        default=1500.0,
        help="Reference cell area (px) for adaptive N0 scaling (default 1500).",
    )
    p.add_argument(
        "--sicle-multiscale",
        action="store_true",
        help="Run RunSICLE with --multiscale and pick one scale per cell.",
    )
    p.add_argument(
        "--sicle-scale-select",
        type=str,
        default="last",
        choices=("last", "veta_solidity", "veta_composite"),
        help=(
            "Multiscale selection: last (finest), veta_solidity (max solidity), "
            "veta_composite (solidity × IoU with Cellpose)."
        ),
    )
    p.add_argument(
        "--sicle-scale-min-solidity",
        type=float,
        default=0.0,
        help="Min solidity for veta_* scale selection (0 = no filter).",
    )
    p.add_argument("--sicle-alpha", type=float, default=SICLE_ALPHA_DEFAULT)
    p.add_argument("--sicle-max-iters", type=int, default=SICLE_MAXITERS_DEFAULT)
    p.add_argument("--sicle-irreg", type=float, default=SICLE_IRREG_DEFAULT)
    p.add_argument("--sicle-adhr", type=int, default=SICLE_ADHR_DEFAULT)
    p.add_argument(
        "--no-saliency-linearize",
        action="store_true",
        help=(
            "Skip Otsu + two-piece linear map on cellprob saliency; use sigmoid "
            "probability only (still masked to the Cellpose instance). Default: linearize."
        ),
    )
    p.add_argument(
        "--saliency-threshold",
        type=float,
        default=0.3,
        help=(
            "Threshold in [0,1] applied to per-cell saliency before SICLE "
            "(values below threshold are set to 0). Default: 0.3"
        ),
    )
    p.add_argument(
        "--saliency-blur-sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian blur (sigma in pixels) on the saliency AFTER thresholding. "
            "Softens near-binary cellprob so that fmax+w_root^alpha (Eq.2 JMIV 2023) "
            "and gradvmaxmul gradient magnitudes have smooth transitions on the cell "
            "border. 0 disables. Recommended: 0.5."
        ),
    )
    p.add_argument(
        "--saliency-mode",
        type=str,
        default="cellprob",
        choices=("cellprob", "grad_l_mix", "flow_mul"),
        help=(
            "Saliency enhancement after blur: cellprob (default), "
            "grad_l_mix (blend with |∇L| from RGB), flow_mul (multiply by 1+γ|dP|)."
        ),
    )
    p.add_argument(
        "--saliency-mix-weight",
        type=float,
        default=0.35,
        help="For grad_l_mix: weight of |∇L| in [0,1] (default 0.35).",
    )
    p.add_argument(
        "--saliency-flow-gamma",
        type=float,
        default=0.5,
        help="For flow_mul: scale γ in sal *= (1 + γ·norm(|dP|)) (default 0.5).",
    )
    p.add_argument(
        "--run-ovlay-borders",
        action="store_true",
        help=(
            "After merge, run SICLE RunOvlayBorders on PPM/PGM files only (IFT often rejects TIFF/PNG). "
            "Requires max merged label ≤ 255. Writes merged_percell_sicle_ovlay_*.ppm/pgm."
        ),
    )
    p.add_argument(
        "--disable-and-merge",
        action="store_true",
        help=(
            "Do not apply the conservative AND with Cellpose mask when pasting each cell. "
            "When enabled, each label uses raw SICLE foreground inside its bbox."
        ),
    )
    p.add_argument(
        "--and-unless-round",
        action="store_true",
        help=(
            "Only with --disable-and-merge: paste raw SICLE foreground if it looks sufficiently "
            "round (circularity + solidity on largest CC); otherwise use AND with the Cellpose "
            "mask in the bbox. Tune with --min-fg-circularity / --min-fg-solidity."
        ),
    )
    p.add_argument(
        "--min-fg-circularity",
        type=float,
        default=0.70,
        help="With --and-unless-round: minimum 4πA/P² (1=circle; square≈0.79). Default: 0.70",
    )
    p.add_argument(
        "--min-fg-solidity",
        type=float,
        default=0.88,
        help="With --and-unless-round: minimum area/convex_hull_area. Default: 0.88",
    )
    p.add_argument(
        "--fill-holes",
        action="store_true",
        help=(
            "After SICLE, fill background holes interior to the FG region of each cell. "
            "Mitigates speckle/hole artifacts that lower Dice/AJI/PQ."
        ),
    )
    p.add_argument(
        "--keep-largest-cc",
        action="store_true",
        help=(
            "After SICLE, keep only the largest 8-connected FG component per cell crop. "
            "Removes spurious small islands that hurt instance metrics."
        ),
    )
    p.add_argument(
        "--closing-radius",
        type=int,
        default=0,
        help=(
            "After SICLE, apply morphological closing with disk radius R before fill/CC steps "
            "(0 = off). Helps glue tiny gaps along the boundary."
        ),
    )
    args = p.parse_args()
    try:
        br, bg, bb = (int(x.strip()) for x in args.overlay_border_color.split(","))
        border_color = (max(0, min(255, br)), max(0, min(255, bg)), max(0, min(255, bb)))
    except ValueError:
        raise SystemExit("--overlay-border-color must be like 0,255,0") from None
    try:
        cr, cg, cb = (int(x.strip()) for x in args.overlay_cellpose_border_color.split(","))
        border_color_cellpose = (max(0, min(255, cr)), max(0, min(255, cg)), max(0, min(255, cb)))
    except ValueError:
        raise SystemExit("--overlay-cellpose-border-color must be like 255,0,0") from None
    try:
        lr, lg, lb = (int(x.strip()) for x in args.overlay_label_color.split(","))
        overlay_label_color = (max(0, min(255, lr)), max(0, min(255, lg)), max(0, min(255, lb)))
    except ValueError:
        raise SystemExit("--overlay-label-color must be like 255,255,0") from None
    if args.overlay_border_thickness < 1:
        raise SystemExit("--overlay-border-thickness must be >= 1")
    if args.overlay_border_source in ("cellpose", "both") and not args.image:
        raise SystemExit("--overlay-border-source cellpose|both requires --image (RGB base for outlines).")
    if args.overlay_number_labels and not args.image:
        raise SystemExit("--overlay-number-labels requires --image.")
    if args.translucent_mask_overlay and not args.image:
        raise SystemExit("--translucent-mask-overlay requires --image.")
    if not (0.0 <= args.translucent_alpha <= 1.0):
        raise SystemExit("--translucent-alpha must be in [0,1]")
    if args.fg_erosion_pixels < 0:
        raise SystemExit("--fg-erosion-pixels must be >= 0")
    if not (0.0 <= args.saliency_threshold <= 1.0):
        raise SystemExit("--saliency-threshold must be in [0,1]")
    if args.and_unless_round and not args.disable_and_merge:
        raise SystemExit("--and-unless-round requires --disable-and-merge (raw paste when round, else AND).")
    if args.and_unless_round and not (0.0 <= args.min_fg_circularity <= 1.0):
        raise SystemExit("--min-fg-circularity must be in [0,1]")
    if args.and_unless_round and not (0.0 <= args.min_fg_solidity <= 1.0):
        raise SystemExit("--min-fg-solidity must be in [0,1]")
    if args.closing_radius < 0:
        raise SystemExit("--closing-radius must be >= 0")
    if args.saliency_mode == "grad_l_mix" and not args.image:
        raise SystemExit("--saliency-mode grad_l_mix requires --image (RGB for |∇L|)")
    if not (0.0 <= args.saliency_mix_weight <= 1.0):
        raise SystemExit("--saliency-mix-weight must be in [0,1]")
    if args.saliency_flow_gamma < 0.0:
        raise SystemExit("--saliency-flow-gamma must be >= 0")
    if args.sicle_multiscale and args.sicle_scale_select != "last" and args.sicle_scale_min_solidity < 0.0:
        raise SystemExit("--sicle-scale-min-solidity must be >= 0")
    if args.sicle_adaptive_ref_area <= 0.0:
        raise SystemExit("--sicle-adaptive-ref-area must be > 0")

    from_dir = Path(args.from_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    percell_dir = out_dir / "percell_cell_outputs"
    percell_dir.mkdir(parents=True, exist_ok=True)

    cellprob_dir = Path(args.cellprob_from_dir) if args.cellprob_from_dir else None
    cellprob, masks, dP = load_cellprob_masks(from_dir, cellprob_dir=cellprob_dir)
    h, w = cellprob.shape
    if masks.shape != (h, w):
        raise SystemExit(f"shape mismatch cellprob {cellprob.shape} vs masks {masks.shape}")
    if args.saliency_mode == "flow_mul" and dP is None:
        raise SystemExit("--saliency-mode flow_mul requires dP in step03 npz")
    if dP is not None:
        if dP.ndim == 3 and dP.shape[0] == 2:
            if dP.shape[1:] != (h, w):
                raise SystemExit(f"dP shape {dP.shape} vs cellprob {(h, w)}")
        elif dP.ndim == 3 and dP.shape[-1] == 2:
            if dP.shape[:2] != (h, w):
                raise SystemExit(f"dP shape {dP.shape} vs cellprob {(h, w)}")
        else:
            raise SystemExit(f"unsupported dP shape {dP.shape}")

    img_rgb_resized = None
    if args.image:
        from cellpose import io

        img = io.imread(args.image)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[0] != h or img.shape[1] != w:
            import cv2

            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgb_resized = np.asarray(img[..., :3], dtype=np.uint8)

    sicle_bin = find_sicle_binary()
    conn_opt, crit_opt = resolve_sicle_path_cost(
        args.sicle_preset, args.sicle_conn_opt, args.sicle_crit_opt
    )

    labels = sorted(int(x) for x in np.unique(masks) if int(x) > 0)
    merged = np.zeros((h, w), dtype=np.int32)
    meta: list[str] = []

    with tempfile.TemporaryDirectory(prefix="percell_sicle_") as tmp:
        tmp_path = Path(tmp)
        for lab in labels:
            r0, r1, c0, c1 = bbox_for_label(masks, lab, args.margin, h, w)
            crop_cp = cellprob[r0:r1, c0:c1]
            crop_m = (masks[r0:r1, c0:c1] == lab).astype(np.uint8)
            crop_input = None if img_rgb_resized is None else img_rgb_resized[r0:r1, c0:c1]
            crop_dP = None
            if dP is not None:
                if dP.ndim == 3 and dP.shape[0] == 2:
                    crop_dP = dP[:, r0:r1, c0:c1]
                else:
                    crop_dP = dP[r0:r1, c0:c1, :]
            sal_u8 = cellprob_crop_to_saliency_u8(
                crop_cp,
                cell_mask=crop_m.astype(bool),
                linearize=not args.no_saliency_linearize,
            )
            sal_u8 = apply_saliency_threshold_u8(sal_u8, args.saliency_threshold)
            if args.saliency_blur_sigma > 0.0:
                sal_u8 = apply_saliency_blur_u8(sal_u8, args.saliency_blur_sigma)
            if args.saliency_mode != "cellprob":
                sal_u8 = enhance_saliency_u8(
                    sal_u8,
                    mode=args.saliency_mode,
                    cell_mask=crop_m.astype(bool),
                    crop_rgb=crop_input,
                    dP_crop=crop_dP,
                    mix_weight=args.saliency_mix_weight,
                    flow_gamma=args.saliency_flow_gamma,
                )
            area = int(crop_m.sum())
            name = f"cell_{lab:05d}"
            output_in_cell = crop_m.astype(bool)
            if area < args.min_cell_area:
                merged[r0:r1, c0:c1][output_in_cell] = lab
                meta.append(f"label {lab}: area={area} < min_cell_area, kept Cellpose mask")
                _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)
                continue

            fg = fg_scribble_coords(crop_m.astype(bool), erosion_pixels=args.fg_erosion_pixels)
            if not fg:
                merged[r0:r1, c0:c1][output_in_cell] = lab
                meta.append(f"label {lab}: no fg seeds, kept Cellpose mask")
                _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)
                continue

            if args.sicle_use_rgb_image and crop_input is not None:
                sicle_input_img = np.asarray(crop_input, dtype=np.uint8)
            else:
                sicle_input_img = sal_u8
            crop_h, crop_w = crop_m.shape
            n0_run, nf_run = args.sicle_n0, args.sicle_nf
            if args.sicle_adaptive_seeds:
                n0_run, nf_run = compute_adaptive_sicle_seeds(
                    area,
                    crop_h,
                    crop_w,
                    args.sicle_n0,
                    args.sicle_nf,
                    ref_area=args.sicle_adaptive_ref_area,
                )
            try:
                sicle_lbl = run_sicle_on_crop(
                    sicle_input_img,
                    fg,
                    tmp_path,
                    name,
                    sicle_bin,
                    n0=n0_run,
                    nf=nf_run,
                    alpha=args.sicle_alpha,
                    max_iters=args.sicle_max_iters,
                    irreg=args.sicle_irreg,
                    adhr=args.sicle_adhr,
                    conn_opt=conn_opt,
                    crit_opt=crit_opt,
                    pen_opt=args.sicle_pen_opt,
                    saliency_u8=sal_u8,
                    multiscale=args.sicle_multiscale,
                    scale_select=args.sicle_scale_select,
                    scale_min_solidity=args.sicle_scale_min_solidity,
                    cell_mask=crop_m.astype(bool),
                )
            except Exception as e:
                merged[r0:r1, c0:c1][output_in_cell] = lab
                meta.append(f"label {lab}: SICLE failed ({e}), kept Cellpose mask")
                _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)
                continue

            obj = sicle_lbl == 1
            obj, clean_diag = _clean_sicle_fg_mask(
                obj,
                closing_radius=args.closing_radius,
                fill_holes=args.fill_holes,
                keep_largest_cc=args.keep_largest_cc,
            )
            if args.sicle_min_solidity > 0.0:
                _circ_v, _sol_v = sicle_fg_shape_circularity_solidity(obj)
                if _sol_v < args.sicle_min_solidity:
                    merged[r0:r1, c0:c1][crop_m.astype(bool)] = lab
                    meta.append(
                        f"label {lab}: solidity={_sol_v:.3f} < {args.sicle_min_solidity:.2f}, "
                        f"reverted to Cellpose mask (Veta filter)"
                    )
                    _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, crop_m.astype(bool))
                    continue
            if args.disable_and_merge:
                if args.and_unless_round:
                    circ, sol = sicle_fg_shape_circularity_solidity(obj)
                    round_ok = circ >= args.min_fg_circularity and sol >= args.min_fg_solidity
                    if round_ok:
                        output_in_cell = obj
                        merge_note = f"raw_round circ={circ:.3f} sol={sol:.3f}"
                    else:
                        output_in_cell = obj & crop_m.astype(bool)
                        merge_note = f"and_non_round circ={circ:.3f} sol={sol:.3f}"
                else:
                    output_in_cell = obj
                    merge_note = "raw"
            else:
                output_in_cell = obj & crop_m.astype(bool)
                merge_note = "and"
            merged[r0:r1, c0:c1][output_in_cell] = lab
            cleanup_tag = (
                f" closing={clean_diag['closing_radius']}"
                f" holes_filled={clean_diag['holes_filled_px']}"
                f" cc_dropped={clean_diag['cc_dropped_px']}"
            )
            meta.append(
                f"label {lab}: bbox=({r0},{r1},{c0},{c1}) placed_pixels={int(output_in_cell.sum())} merge={merge_note}{cleanup_tag}"
            )
            _write_percell_debug_outputs(percell_dir / name, crop_input, sal_u8, output_in_cell)

    np.save(out_dir / "merged_percell_sicle_masks_int32.npy", merged)
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(plot.mask_rgb(merged)).save(out_dir / "merged_percell_sicle_masks_rgb.png")
    else:
        imwrite(out_dir / "merged_percell_sicle_masks_rgb.png", plot.mask_rgb(merged))

    (out_dir / "percell_sicle_log.txt").write_text(
        "\n".join(
            [
                f"from_dir: {from_dir.resolve()}",
                f"cells: {len(labels)}",
                f"margin: {args.margin}",
                f"fg_erosion_pixels: {args.fg_erosion_pixels}",
                f"saliency_linearize: {not args.no_saliency_linearize}",
                f"saliency_threshold: {args.saliency_threshold}",
                f"saliency_blur_sigma: {args.saliency_blur_sigma}",
                f"saliency_mode: {args.saliency_mode}",
                f"saliency_mix_weight: {args.saliency_mix_weight}",
                f"saliency_flow_gamma: {args.saliency_flow_gamma}",
                f"disable_and_merge: {args.disable_and_merge}",
                f"and_unless_round: {args.and_unless_round}",
                f"min_fg_circularity: {args.min_fg_circularity}",
                f"min_fg_solidity: {args.min_fg_solidity}",
                f"fill_holes: {args.fill_holes}",
                f"keep_largest_cc: {args.keep_largest_cc}",
                f"closing_radius: {args.closing_radius}",
                f"overlay_border_source: {args.overlay_border_source}",
                f"overlay_cellpose_border_color: {args.overlay_cellpose_border_color}",
                f"write_compare_vs_step04: {args.write_compare_vs_step04}",
                f"sicle preset: {args.sicle_preset} conn={conn_opt} crit={crit_opt}",
                f"sicle_n0: {args.sicle_n0} sicle_nf: {args.sicle_nf}",
                f"sicle_adaptive_seeds: {args.sicle_adaptive_seeds} ref_area: {args.sicle_adaptive_ref_area}",
                f"sicle_multiscale: {args.sicle_multiscale} scale_select: {args.sicle_scale_select}",
                f"sicle_scale_min_solidity: {args.sicle_scale_min_solidity}",
                "",
                *meta,
            ]
        ),
        encoding="utf-8",
    )

    if args.image:
        from PIL import Image

        if args.overlay_border_source == "sicle":
            ov = _outline_only_overlay(
                img_rgb_resized,
                merged,
                border_color_rgb=border_color,
                border_thickness=args.overlay_border_thickness,
            )
        else:
            ov = _selective_border_overlay(
                img_rgb_resized,
                merged_sicle=merged,
                cellpose_masks=masks,
                source=args.overlay_border_source,
                border_sicle_rgb=border_color,
                border_cellpose_rgb=border_color_cellpose,
                border_thickness=args.overlay_border_thickness,
            )
        overlay_path = out_dir / "merged_percell_sicle_overlay.png"
        try:
            from imageio import imwrite
        except ImportError:
            Image.fromarray(ov).save(overlay_path)
        else:
            imwrite(overlay_path, ov)

        if args.overlay_number_labels:
            cell_centers = _cell_label_centroids(masks, labels)
            ov_num = _overlay_with_cell_label_ids(ov, cell_centers, text_rgb=overlay_label_color)
            numbered_path = out_dir / "merged_percell_sicle_overlay_numbered.png"
            try:
                from imageio import imwrite
            except ImportError:
                Image.fromarray(ov_num).save(numbered_path)
            else:
                imwrite(numbered_path, ov_num)

        if args.translucent_mask_overlay:
            try:
                from imageio import imwrite as _imw
            except ImportError:
                _imw = None
            tr_sicle = _translucent_mask_overlay(
                img_rgb_resized,
                merged,
                color_rgb=border_color,
                alpha=args.translucent_alpha,
            )
            tr_cp = _translucent_mask_overlay(
                img_rgb_resized,
                masks,
                color_rgb=border_color_cellpose,
                alpha=args.translucent_alpha,
            )
            sicle_path = out_dir / "merged_percell_sicle_translucent_sicle.png"
            cp_path = out_dir / "merged_percell_sicle_translucent_cellpose.png"
            if _imw is None:
                Image.fromarray(tr_sicle).save(sicle_path)
                Image.fromarray(tr_cp).save(cp_path)
            else:
                _imw(sicle_path, tr_sicle)
                _imw(cp_path, tr_cp)

    if args.write_compare_vs_step04:
        _run_compare_final_vs_step04(from_dir, out_dir)

    if args.run_ovlay_borders:
        from PIL import Image

        ov_bin = _find_run_ovlay_borders_binary(sicle_bin)
        if ov_bin is None:
            print("RunOvlayBorders not found next to RunSICLE; skipped.")
        else:
            labels_pgm = out_dir / "merged_percell_sicle_ovlay_labels.pgm"
            try:
                _write_ovlay_labels_pgm(merged, labels_pgm)
            except ValueError as e:
                print(e)
            else:
                base_ppm = out_dir / "merged_percell_sicle_ovlay_base.ppm"
                if img_rgb_resized is not None:
                    Image.fromarray(img_rgb_resized).save(base_ppm)
                else:
                    sal = cellprob_crop_to_saliency_u8(cellprob)
                    rgb = np.stack([sal, sal, sal], axis=-1)
                    Image.fromarray(rgb).save(base_ppm)
                out_ov = out_dir / "merged_percell_sicle_ovlay_borders.ppm"
                ok, msg = _run_run_ovlay_borders(ov_bin, base_ppm, labels_pgm, out_ov)
                if ok:
                    print(f"Wrote {out_ov} (RunOvlayBorders)")
                else:
                    print("RunOvlayBorders failed:\n", msg[:1200])

    print(f"Wrote {out_dir / 'merged_percell_sicle_masks_int32.npy'}")
    print(f"Wrote {out_dir / 'merged_percell_sicle_masks_rgb.png'}")
    print(f"Wrote {out_dir / 'percell_sicle_log.txt'}")
    print(f"Wrote per-cell outputs in {percell_dir}")
    if args.image:
        print(f"Wrote {out_dir / 'merged_percell_sicle_overlay.png'} (overlay_border_source={args.overlay_border_source})")
    if args.image and args.overlay_number_labels:
        print(f"Wrote {out_dir / 'merged_percell_sicle_overlay_numbered.png'}")
    if args.image and args.translucent_mask_overlay:
        print(f"Wrote {out_dir / 'merged_percell_sicle_translucent_sicle.png'}")
        print(f"Wrote {out_dir / 'merged_percell_sicle_translucent_cellpose.png'}")
    if args.write_compare_vs_step04:
        print(f"Compare step04 vs merged: {out_dir / 'compare_final_vs_step04'}")
    if args.run_ovlay_borders and (out_dir / "merged_percell_sicle_ovlay_borders.ppm").is_file():
        print(f"Wrote {out_dir / 'merged_percell_sicle_ovlay_borders.ppm'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
