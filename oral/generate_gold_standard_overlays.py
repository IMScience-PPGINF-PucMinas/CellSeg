#!/usr/bin/env python3
"""Overlay instance-colored GT on original ROI images → outputs/reviews/."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from _paths import GT_COLORED, IMAGES_ORIGINAL, REPO, REVIEWS

COL_DIR = GT_COLORED
ORIG_DIR = IMAGES_ORIGINAL
OUT_DIR = REVIEWS / "gold_standard_overlays"
ALPHA = 0.45
BG_THRESH = 8


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _foreground_mask(col: np.ndarray) -> np.ndarray:
    return col.max(axis=2) > BG_THRESH


def _instance_borders(col: np.ndarray) -> np.ndarray:
    import cv2

    fg = _foreground_mask(col).astype(np.uint8)
    if fg.max() == 0:
        return np.zeros(fg.shape, dtype=bool)
    flat = col.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    lab = np.zeros(fg.shape, dtype=np.int32)
    lid = 1
    for c in uniq:
        if int(c.max()) <= BG_THRESH:
            continue
        m = np.all(col == c, axis=2)
        lab[m] = lid
        lid += 1
    border = np.zeros(fg.shape, dtype=bool)
    for i in range(1, lid):
        m = (lab == i).astype(np.uint8)
        er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        border |= (m > 0) & (er == 0)
    return border


def _translucent_overlay(orig: np.ndarray, col: np.ndarray, alpha: float) -> np.ndarray:
    fg = _foreground_mask(col)
    out = orig.astype(np.float32).copy()
    out[fg] = (1.0 - alpha) * orig[fg] + alpha * col[fg]
    return np.clip(out, 0, 255).astype(np.uint8)


def _border_overlay(orig: np.ndarray, col: np.ndarray) -> np.ndarray:
    border = _instance_borders(col)
    out = orig.copy()
    out[border] = np.array([0, 255, 0], dtype=np.uint8)
    return out


def _panel(orig: np.ndarray, col: np.ndarray, trans: np.ndarray, border: np.ndarray) -> np.ndarray:
    h, w = orig.shape[:2]
    gap = 4
    canvas = np.full((h, 4 * w + 3 * gap, 3), 255, dtype=np.uint8)
    x = 0
    for img in (orig, trans, border, col):
        canvas[:, x : x + w] = img
        x += w + gap
    return canvas


def _count_instances(col: np.ndarray) -> int:
    flat = col.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    return sum(1 for c in uniq if int(c.max()) > BG_THRESH)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> None:
    from PIL import Image

    rows: list[dict] = []
    n_ok = n_skip = 0

    for category in ("healthy", "severe"):
        col_cat = COL_DIR / category
        orig_cat = ORIG_DIR / category
        out_cat = OUT_DIR / category
        out_cat.mkdir(parents=True, exist_ok=True)

        for col_path in sorted(col_cat.glob("*.png")):
            stem = col_path.stem
            orig_path = orig_cat / f"{stem}.tif"
            if not orig_path.is_file():
                print(f"[skip] no original for {category}/{stem}")
                n_skip += 1
                continue

            col = _load_rgb(col_path)
            orig = _load_rgb(orig_path)
            if col.shape[:2] != orig.shape[:2]:
                h, w = min(col.shape[0], orig.shape[0]), min(col.shape[1], orig.shape[1])
                col, orig = col[:h, :w], orig[:h, :w]

            trans = _translucent_overlay(orig, col, ALPHA)
            border = _border_overlay(orig, col)
            panel = _panel(orig, col, trans, border)

            p_trans = out_cat / f"{stem}_overlay_translucent.png"
            p_border = out_cat / f"{stem}_overlay_borders.png"
            p_panel = out_cat / f"{stem}_panel_orig_trans_border_mask.png"

            Image.fromarray(trans).save(p_trans)
            Image.fromarray(border).save(p_border)
            Image.fromarray(panel).save(p_panel)

            n_inst = _count_instances(col)
            rows.append(
                {
                    "category": category,
                    "stem": stem,
                    "n_instances_colored": n_inst,
                    "width": col.shape[1],
                    "height": col.shape[0],
                    "colored_mask": _rel(col_path),
                    "original_image": _rel(orig_path),
                    "overlay_translucent": _rel(p_trans),
                    "overlay_borders": _rel(p_border),
                    "panel_4up": _rel(p_panel),
                }
            )
            n_ok += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "overlay_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    readme = OUT_DIR / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Gold Standard instance segmentation (colored) vs Original ROI images.",
                "",
                f"Generated {n_ok} pairs ({n_skip} skipped — missing .tif).",
                "",
                "Sources:",
                "  data/oral_epithelium/annotations/instance_colored/<cat>/*.png",
                "  data/oral_epithelium/images/original/<cat>/*.tif",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Done: {n_ok} overlays -> {OUT_DIR}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
