#!/usr/bin/env python3
"""
Extract ground-truth instance masks from a Slidescape ``slices.lab`` file.

For each tile in ``slices.lab``:
1. Match its ``rects`` against the per-slice ``*_tile.xml`` files in ``--data-dir`` to find
   which ``<stem>.png`` it corresponds to.
2. For each segmentation layer, rasterize its polygons (global WSI coords → local 512×512)
   into an int32 instance label mask (1 polygon = 1 instance ID).
3. Save under ``--out-root/<stem>/gt/``:
     - ``<layer>_masks_int32.npy``
     - ``<layer>_masks_rgb.png``  (random per-instance colors)
     - ``<layer>_overlay.png``    (instance borders over the original PNG, if present)
     - ``<layer>_translucent.png`` (semi-transparent fill for visual QC)

Usage::

    cd doutorado/new_pipeline
    python3 extract_slices_lab_gt.py \\
        --slices-lab ./slices.lab \\
        --data-dir   ./data_sibgrapi2026/data_sibgrapi2026 \\
        --out-root   ./out_sibgrapi2026
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_tile_xml_rect(xml_path: Path) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) from a slidescape tile XML, or None on failure."""
    try:
        txt = xml_path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(
        r'<bounds\s+x1="(\d+)"\s+y1="(\d+)"\s+x2="(\d+)"\s+y2="(\d+)"',
        txt,
    )
    if not m:
        return None
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def safe_layer_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "layer"


def rasterize_polygons(
    polygons: list[list[list[int]]],
    rect: tuple[int, int, int, int],
) -> "np.ndarray":
    """Each polygon → instance ID (1..N). Vertices in WSI coords; rect (x1,y1,x2,y2)."""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = rect
    h, w = y2 - y1, x2 - x1
    out = np.zeros((h, w), dtype=np.int32)
    for i, poly in enumerate(polygons, start=1):
        if not poly:
            continue
        pts = np.asarray(poly, dtype=np.float64)
        pts[:, 0] -= x1
        pts[:, 1] -= y1
        ip = np.round(pts).astype(np.int32)
        cv2.fillPoly(out, [ip], int(i))
    return out


def render_rgb_random(labels: "np.ndarray", seed: int = 7) -> "np.ndarray":
    import numpy as np

    max_id = int(labels.max())
    if max_id <= 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    palette = rng.integers(48, 256, size=(max_id + 1, 3), dtype=np.int32)
    palette[0] = (0, 0, 0)
    return palette[labels].astype(np.uint8)


def outline_overlay(rgb: "np.ndarray", labels: "np.ndarray", color=(0, 255, 0)) -> "np.ndarray":
    import numpy as np
    import cv2

    L = labels.astype(np.int32)
    Lpad = np.pad(L, 1, mode="constant", constant_values=0)
    border = (L > 0) & (
        (Lpad[1:-1, :-2] != L)
        | (Lpad[1:-1, 2:] != L)
        | (Lpad[:-2, 1:-1] != L)
        | (Lpad[2:, 1:-1] != L)
    )
    out = np.asarray(rgb[..., :3], dtype=np.uint8).copy()
    r, g, b = color
    out[border, 0] = r
    out[border, 1] = g
    out[border, 2] = b
    return out


def translucent_overlay(
    rgb: "np.ndarray", labels: "np.ndarray", color=(0, 255, 0), alpha: float = 0.45
) -> "np.ndarray":
    import numpy as np

    fg = labels > 0
    out = rgb.astype(np.float32).copy()
    r, g, b = (int(c) for c in color)
    out[fg] = (1.0 - alpha) * out[fg] + alpha * np.array([r, g, b], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    import numpy as np
    from PIL import Image

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slices-lab", type=str, default=str(here / "slices.lab"))
    p.add_argument("--data-dir", type=str, default=str(here / "data_sibgrapi2026/data_sibgrapi2026"))
    p.add_argument("--out-root", type=str, default=str(here / "out_sibgrapi2026"))
    p.add_argument("--alpha", type=float, default=0.45, help="Opacity for translucent PNG")
    p.add_argument("--color", type=str, default="0,255,0", help="RGB for borders/fill")
    args = p.parse_args()

    color = tuple(int(x) for x in args.color.split(","))
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)

    rect_to_stem: dict[tuple[int, int, int, int], str] = {}
    for xml in sorted(data_dir.glob("*_tile.xml")):
        stem = xml.name.removesuffix("_tile.xml")
        rect = parse_tile_xml_rect(xml)
        if rect is not None:
            rect_to_stem[rect] = stem
    print(f"Loaded {len(rect_to_stem)} tile XMLs from {data_dir}")

    data = json.loads(Path(args.slices_lab).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("slices.lab: expected non-empty list at top level")

    n_done = 0
    for wsi_entry in data:
        wsi_path = wsi_entry.get("path", "")
        tiles = wsi_entry.get("tiles", [])
        print(f"WSI {wsi_path}: {len(tiles)} tiles")
        for tile in tiles:
            rects = tile.get("rects", [])
            if not rects:
                continue
            rect = tuple(rects[0])
            if len(rect) != 4:
                continue
            stem = rect_to_stem.get(rect)
            if stem is None:
                print(f"  rect {rect}: no matching tile xml; skip")
                continue
            case_dir = out_root / stem
            gt_dir = case_dir / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)

            png_path = data_dir / f"{stem}.png"
            base_rgb = None
            if png_path.is_file():
                try:
                    base_rgb = np.asarray(Image.open(png_path).convert("RGB"))
                except OSError:
                    base_rgb = None

            layer_lnames: list[str] = []
            layer_label_imgs: list["np.ndarray"] = []
            for layer in tile.get("segmentation_layers", []):
                lname = safe_layer_name(layer.get("name", "layer"))
                polys = layer.get("polygons", []) or []
                labels = rasterize_polygons(polys, rect)
                n_inst = int(labels.max())
                npy_path = gt_dir / f"{lname}_masks_int32.npy"
                rgb_path = gt_dir / f"{lname}_masks_rgb.png"
                np.save(npy_path, labels.astype(np.int32))
                Image.fromarray(render_rgb_random(labels)).save(rgb_path)
                if base_rgb is not None:
                    Image.fromarray(outline_overlay(base_rgb, labels, color)).save(
                        gt_dir / f"{lname}_overlay.png"
                    )
                    Image.fromarray(translucent_overlay(base_rgb, labels, color, args.alpha)).save(
                        gt_dir / f"{lname}_translucent.png"
                    )
                print(f"  {stem} / {lname}: {n_inst:4d} instances -> {npy_path.name}")
                layer_lnames.append(lname)
                layer_label_imgs.append(labels)

            if layer_label_imgs:
                from scipy.ndimage import label as cc_label

                union_fg = np.zeros_like(layer_label_imgs[0], dtype=bool)
                for L in layer_label_imgs:
                    union_fg |= L > 0
                relabeled, n_cc = cc_label(union_fg, structure=np.ones((3, 3), dtype=bool))
                union_labels = relabeled.astype(np.int32)
                np.save(gt_dir / "union_masks_int32.npy", union_labels)
                Image.fromarray(render_rgb_random(union_labels)).save(
                    gt_dir / "union_masks_rgb.png"
                )
                if base_rgb is not None:
                    Image.fromarray(outline_overlay(base_rgb, union_labels, color)).save(
                        gt_dir / "union_overlay.png"
                    )
                    Image.fromarray(translucent_overlay(base_rgb, union_labels, color, args.alpha)).save(
                        gt_dir / "union_translucent.png"
                    )
                print(
                    f"  {stem} / UNION({', '.join(layer_lnames)}): "
                    f"{int(n_cc):4d} CCs -> union_masks_int32.npy"
                )
            n_done += 1

    print(f"\nDone: extracted GT for {n_done} tiles into {out_root}/<stem>/gt/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
