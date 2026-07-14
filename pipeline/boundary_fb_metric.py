"""
Boundary F-measure (Fb) for instance segmentation — Arbeláez / Martin / BSDS style.

Treats each cell as a binary foreground mask, extracts 1-pixel-wide boundaries
(``seg2bmap``), and matches pred/GT boundaries with tolerance

    d = ceil(bound_th * ||(H, W)||),  default bound_th = 0.0075.

Fb = 2 * P * R / (P + R), with P/R from dilated boundary overlap (DAVIS ``f_boundary``).

Per-cell macro (aligned with ``mean_br_macro``): for each GT instance, isolate the
best-overlap prediction in the bbox (+margin) and compute Fb on the binary masks.

Boundary recall (BR) uses a **shared** tolerance ``DEFAULT_BR_TOLERANCE`` (2 px,
Euclidean EDT) and thick label boundaries for both aggregation modes:
  - **macro** — mean per GT instance after strict instance matching;
  - **pooled** — single recall over all GT boundary pixels in the tile (BR Area).
"""
from __future__ import annotations

import numpy as np

DEFAULT_BOUND_TH = 0.0075
DEFAULT_BR_TOLERANCE = 2


def seg2bmap(seg: "np.ndarray") -> "np.ndarray":
    """1-pixel-wide boundary of a binary foreground mask (Martin / DAVIS seg2bmap)."""
    m = np.asarray(seg, dtype=bool)
    if m.ndim > 2:
        m = m[..., 0]
    e = np.zeros_like(m)
    s = np.zeros_like(m)
    se = np.zeros_like(m)
    e[:, :-1] = m[:, 1:]
    s[:-1, :] = m[1:, :]
    se[:-1, :-1] = m[1:, 1:]
    b = m ^ e | m ^ s | m ^ se
    b[-1, :] = m[-1, :] ^ e[-1, :]
    b[:, -1] = m[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def tolerance_pixels(h: int, w: int, bound_th: float = DEFAULT_BOUND_TH) -> int:
    if bound_th >= 1.0:
        return int(bound_th)
    return max(1, int(np.ceil(bound_th * float(np.linalg.norm((h, w))))))


def boundary_fb_binary(
    pred_fg: "np.ndarray",
    gt_fg: "np.ndarray",
    *,
    bound_th: float = DEFAULT_BOUND_TH,
) -> tuple[float, float, float]:
    """
    Fb, precision, recall for two binary foreground masks (same shape).

    Reference: Arbeláez et al. boundary precision-recall; DAVIS ``db_eval_boundary``.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    pred = np.asarray(pred_fg, dtype=bool)
    gt = np.asarray(gt_fg, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")

    h, w = pred.shape
    r = tolerance_pixels(h, w, bound_th)
    try:
        from skimage.morphology import disk as _disk

        se = _disk(r)
    except ImportError:
        se = generate_binary_structure(2, 1)
        for _ in range(max(0, r - 1)):
            se = binary_dilation(se, structure=generate_binary_structure(2, 1))

    fg_b = seg2bmap(pred)
    gt_b = seg2bmap(gt)
    fg_dil = binary_dilation(fg_b, structure=se)
    gt_dil = binary_dilation(gt_b, structure=se)

    gt_match = gt_b & fg_dil
    fg_match = fg_b & gt_dil

    n_fg = int(fg_b.sum())
    n_gt = int(gt_b.sum())

    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = float(fg_match.sum()) / float(n_fg)
        recall = float(gt_match.sum()) / float(n_gt)

    if precision + recall <= 0.0:
        fb = 0.0
    else:
        fb = 2.0 * precision * recall / (precision + recall)
    return fb, precision, recall


def mean_fb_strict(
    gt: "np.ndarray",
    pr: "np.ndarray",
    margin: int = 8,
    *,
    bound_th: float = DEFAULT_BOUND_TH,
) -> float:
    """Macro mean Fb per GT cell (strict: best-matching pred instance in bbox)."""
    from percell_boundary_recall import bbox_of_mask, isolate_pred_for_gt

    vals: list[float] = []
    h, w = gt.shape
    for gid in np.unique(gt):
        gid = int(gid)
        if gid <= 0:
            continue
        m = gt == gid
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
        r1, c1 = min(h, r1 + margin), min(w, c1 + margin)
        gt_crop = gt[r0:r1, c0:c1]
        pr_crop = pr[r0:r1, c0:c1]
        gt_bin = gt_crop == gid
        pr_iso, _ = isolate_pred_for_gt(pr_crop, gt_crop, gid)
        pr_bin = pr_iso > 0
        fb, _, _ = boundary_fb_binary(pr_bin, gt_bin, bound_th=bound_th)
        vals.append(fb)
    return float(np.mean(vals)) if vals else float("nan")


def f_area_binary(gt_fg: "np.ndarray", pr_fg: "np.ndarray") -> tuple[float, float, float]:
    """
    F-measure por **região** (área): GT e pred como máscaras binárias 0/1.

    Por pixel p:
      TP (acertou) = A(p)=1 e B(p)=1
      FN (miss)    = A(p)=1 e B(p)=0
      FP (errou)   = A(p)=0 e B(p)=1

    Precisão = TP/(TP+FP), Revocação = TP/(TP+FN), F1 = 2PR/(P+R).
    Equivale ao Dice quando ambas as máscaras são binárias.
    """
    gt = np.asarray(gt_fg, dtype=bool)
    pr = np.asarray(pr_fg, dtype=bool)
    if gt.shape != pr.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {pr.shape}")

    tp = int(np.logical_and(gt, pr).sum())
    fp = int(np.logical_and(np.logical_not(gt), pr).sum())
    fn = int(np.logical_and(gt, np.logical_not(pr)).sum())

    if tp + fp == 0:
        precision = 1.0 if fn == 0 else 0.0
    else:
        precision = float(tp) / float(tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = float(tp) / float(tp + fn)
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return f1, precision, recall


def mean_f_area_strict(
    gt: "np.ndarray",
    pr: "np.ndarray",
    margin: int = 8,
) -> float:
    """Macro mean F-measure por **área** per GT cell (strict: best-matching pred in bbox)."""
    from percell_boundary_recall import bbox_of_mask, isolate_pred_for_gt

    vals: list[float] = []
    h, w = gt.shape
    for gid in np.unique(gt):
        gid = int(gid)
        if gid <= 0:
            continue
        m = gt == gid
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
        r1, c1 = min(h, r1 + margin), min(w, c1 + margin)
        gt_crop = gt[r0:r1, c0:c1]
        pr_crop = pr[r0:r1, c0:c1]
        gt_bin = gt_crop == gid
        pr_iso, _ = isolate_pred_for_gt(pr_crop, gt_crop, gid)
        pr_bin = pr_iso > 0
        f1, _, _ = f_area_binary(gt_bin, pr_bin)
        vals.append(f1)
    return float(np.mean(vals)) if vals else float("nan")


def _thick_boundaries(labels: "np.ndarray") -> "np.ndarray":
    from skimage.segmentation import find_boundaries

    return find_boundaries(np.asarray(labels, dtype=np.int32), mode="thick")


def boundary_recall_binary(
    gt_fg: "np.ndarray",
    pred_fg: "np.ndarray",
    *,
    boundary_tolerance: int = DEFAULT_BR_TOLERANCE,
) -> float:
    """Recall on thick boundaries of two binary masks (EDT tolerance, pooled over GT px)."""
    from scipy.ndimage import distance_transform_edt

    gt = np.asarray(gt_fg, dtype=bool)
    pred = np.asarray(pred_fg, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {pred.shape}")

    gt_b = _thick_boundaries(gt.astype(np.int32))
    pred_b = _thick_boundaries(pred.astype(np.int32))
    if not gt_b.any():
        return float("nan")
    if not pred_b.any():
        return 0.0
    d_to_pred = distance_transform_edt(~pred_b)
    return float(np.mean(d_to_pred[gt_b] <= boundary_tolerance))


def mean_br_macro(
    gt: "np.ndarray",
    pr: "np.ndarray",
    margin: int = 8,
    *,
    boundary_tolerance: int = DEFAULT_BR_TOLERANCE,
) -> float:
    """
    Macro mean boundary recall per GT instance (strict matching).

    For each GT cell: isolate the best-matching prediction, compute pooled recall
    on that cell's thick boundary at ``boundary_tolerance`` (same definition as
    tile-level BR Area, but averaged over instances).
    """
    from percell_boundary_recall import bbox_of_mask, isolate_pred_for_gt

    vals: list[float] = []
    h, w = gt.shape
    for gid in np.unique(gt):
        gid = int(gid)
        if gid <= 0:
            continue
        m = gt == gid
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
        r1, c1 = min(h, r1 + margin), min(w, c1 + margin)
        gt_crop = gt[r0:r1, c0:c1]
        pr_crop = pr[r0:r1, c0:c1]
        gt_bin = gt_crop == gid
        pr_iso, _ = isolate_pred_for_gt(pr_crop, gt_crop, gid)
        pr_bin = pr_iso > 0
        br_i = boundary_recall_binary(
            gt_bin, pr_bin, boundary_tolerance=boundary_tolerance
        )
        if np.isfinite(br_i):
            vals.append(br_i)
    return float(np.mean(vals)) if vals else float("nan")


def boundary_recall_pooled(
    gt: "np.ndarray",
    pred: "np.ndarray",
    *,
    boundary_tolerance: int = DEFAULT_BR_TOLERANCE,
) -> tuple[float, float, float]:
    """
    Tile-level boundary recall (pooled GT pixels, no per-instance averaging).

    Returns (recall, precision, F-measure) on fused thick boundaries.
    """
    from scipy.ndimage import distance_transform_edt

    gt = np.asarray(gt, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.int32)
    if gt.shape != pred.shape:
        h = min(gt.shape[0], pred.shape[0])
        w = min(gt.shape[1], pred.shape[1])
        gt, pred = gt[:h, :w], pred[:h, :w]

    gt_b = _thick_boundaries(gt)
    pred_b = _thick_boundaries(pred)
    if not gt_b.any() or not pred_b.any():
        return float("nan"), float("nan"), float("nan")

    d_to_pred = distance_transform_edt(~pred_b)
    d_to_gt = distance_transform_edt(~gt_b)
    br = float(np.mean(d_to_pred[gt_b] <= boundary_tolerance))
    bp = float(np.mean(d_to_gt[pred_b] <= boundary_tolerance))
    bf = (2.0 * br * bp / (br + bp)) if (br + bp) > 0 else 0.0
    return br, bp, bf


def boundary_metrics_seraph(
    gt: "np.ndarray",
    pred: "np.ndarray",
    *,
    boundary_tolerance: int = DEFAULT_BR_TOLERANCE,
) -> tuple[float, float, float]:
    """Alias for :func:`boundary_recall_pooled` (tile-level BR Area)."""
    return boundary_recall_pooled(
        gt, pred, boundary_tolerance=boundary_tolerance
    )
