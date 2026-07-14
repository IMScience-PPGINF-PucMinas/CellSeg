#!/usr/bin/env python3
"""Build 3-panel iDISF process figure for the SIBGRAPI paper (Fig. idisf)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _paths import RUNS

DEFAULT_CASE = RUNS / "percell_idisf_full" / "healthy" / "healthy-18-roi2" / "idisf_exclude_other"
PAPER_FIGS = Path(__file__).resolve().parents[2] / "sibgrapi2026_draft" / "template-sibgrapi-2026" / "figs" / "paper"
REFINED_GREEN = (0, 220, 0)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _cell_centrality(obj_mask: np.ndarray) -> float:
    h, w = obj_mask.shape
    ys, xs = np.where(obj_mask)
    if len(xs) == 0:
        return 1.0
    cy, cx = ys.mean(), xs.mean()
    return float(np.hypot(cy - h / 2, cx - w / 2) / max(np.hypot(h / 2, w / 2), 1.0))


def _pick_cell(cells_root: Path, *, prefer_ignored: bool = True) -> Path:
    best: tuple[float, float, float, Path] | None = None
    for d in sorted(cells_root.glob("cell_*")):
        inp_p = d / "input_image.png"
        out_p = d / "output_in_cell.png"
        obj_p = d / "marker_object_mask.png"
        if not all(p.is_file() for p in (inp_p, out_p, obj_p)):
            continue
        obj = np.asarray(Image.open(obj_p)) > 0
        if obj.sum() < 200:
            continue
        out = np.asarray(Image.open(out_p))
        refined = out > 0 if out.ndim == 2 else (out[..., 0] > 0)
        diff = float((refined != obj).sum())
        if diff < 50:
            continue
        ign_p = d / "marker_ignored_mask.png"
        ign = float((np.asarray(Image.open(ign_p)) > 0).sum()) if ign_p.is_file() else 0.0
        centrality = _cell_centrality(obj)
        score = centrality - (0.02 if prefer_ignored and ign > 0 else 0.0)
        cand = (score, -diff, -obj.sum(), d)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise FileNotFoundError(f"No suitable cell under {cells_root}")
    return best[3]


def _partition_panel(rgb: np.ndarray, partition: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from percell_boundary_recall import draw_contours

    labels = np.zeros(partition.shape[:2], dtype=np.int32)
    labels[partition > 0] = 1
    return draw_contours(rgb, labels, REFINED_GREEN, thickness=2)


def _markers_panel(cell_dir: Path, rgb: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from percell_conquest_viz import plot_object_bg_ignored

    fg = np.asarray(Image.open(cell_dir / "marker_object_mask.png")) > 0
    bg = np.asarray(Image.open(cell_dir / "marker_background_mask.png")) > 0
    ign_p = cell_dir / "marker_ignored_mask.png"
    ign = np.asarray(Image.open(ign_p)) > 0 if ign_p.is_file() else None
    return plot_object_bg_ignored(rgb, fg_mask=fg, bg_mask=bg, ignored_mask=ign)


def _legend_font(size: int = 11) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()



def _draw_legend(width: int, *, font_size: int = 11, bar_h: int = 26) -> Image.Image:
    font = _legend_font(font_size)
    bar = Image.new("RGB", (width, bar_h), (255, 255, 255))
    draw = ImageDraw.Draw(bar)
    items = [
        ((0, 170, 0), "FG (eroded seed)"),
        ((220, 0, 0), "BG (crop border)"),
        ((170, 0, 200), "Unconquerable neighbor"),
        (REFINED_GREEN, "Refined contour"),
    ]
    swatch = max(7, font_size - 3)
    y0 = (bar_h - swatch) // 2
    x = 10
    for color, text in items:
        tw = int(draw.textlength(text, font=font)) if hasattr(draw, "textlength") else 6 * len(text)
        item_w = swatch + 5 + tw + 12
        if x + item_w > width - 6 and x > 10:
            x = 10
            y0 += swatch + 4
        draw.rectangle((x, y0, x + swatch, y0 + swatch), fill=color, outline=(80, 80, 80))
        draw.text((x + swatch + 4, y0 - 1), text, fill=(40, 40, 40), font=font)
        x += item_w
    return bar


def build_panel(cell_dir: Path, *, scale: float = 8.0) -> Image.Image:
    rgb = _load_rgb(cell_dir / "input_image.png")
    part = np.asarray(Image.open(cell_dir / "output_in_cell.png"))
    if part.ndim == 3:
        part = part[..., 0]

    panels = [
        ("(a)", rgb),
        ("(b)", _markers_panel(cell_dir, rgb)),
        ("(c)", _partition_panel(rgb, part)),
    ]

    h, w = rgb.shape[:2]
    gap = max(4, int(6 * scale / 8))
    panel_w = 3 * w + 2 * gap
    panel_row = Image.new("RGB", (panel_w, h), (255, 255, 255))

    for i, (_tag, img) in enumerate(panels):
        tile = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((w, h), Image.NEAREST)
        x0 = i * (w + gap)
        panel_row.paste(tile, (x0, 0))

    if scale != 1.0:
        panel_row = panel_row.resize(
            (int(panel_row.width * scale), int(panel_row.height * scale)),
            Image.Resampling.NEAREST,
        )

    legend_h = max(22, int(panel_row.height * 0.055))
    font_size = max(9, int(10 * scale / 8))
    legend = _draw_legend(panel_row.width, font_size=font_size, bar_h=legend_h)
    out = Image.new("RGB", (panel_row.width, panel_row.height + legend.height), (255, 255, 255))
    out.paste(panel_row, (0, 0))
    out.paste(legend, (0, panel_row.height))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE, help="iDISF per-cell run root")
    p.add_argument("--cell", default="", help="cell folder name, e.g. cell_00007")
    p.add_argument("--out", type=Path, default=PAPER_FIGS / "idisf_process_panel.png")
    p.add_argument("--scale", type=float, default=8.0)
    args = p.parse_args()

    cells_root = args.case / "percell_cell_outputs"
    if not cells_root.is_dir():
        raise SystemExit(f"Missing {cells_root}")

    if args.cell:
        cell_dir = cells_root / args.cell
        if not cell_dir.is_dir():
            raise SystemExit(f"Missing {cell_dir}")
    else:
        cell_dir = _pick_cell(cells_root)

    out = build_panel(cell_dir, scale=args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"Wrote {args.out} from {cell_dir.name}")


if __name__ == "__main__":
    main()
