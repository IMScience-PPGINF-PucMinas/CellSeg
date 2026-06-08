#!/usr/bin/env python3
"""
Painel comparativo com rotulo SOBRE cada imagem (qual resultado ela representa).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, RUNS

OUT_ROOT = RUNS / "path_cost_benchmark"
PANELS_OUT = OUT_ROOT / "panels" / "exemplar_comparison_labeled.png"

GT_CYAN = (0, 255, 255)
CP_YELLOW = (255, 255, 0)
CONN_COLORS = {
    "fmax_minsc": (255, 140, 0),
    "fsum_maxsc": (0, 180, 255),
    "gradvmaxmul_minsc": (0, 200, 0),
}
CRIT_COLORS = {
    "minsc": (0, 200, 0),
    "maxsc": (255, 140, 0),
    "size": (255, 0, 255),
    "spread": (0, 160, 220),
}
GAP = 8
TITLE_H = 22
ROW_LABEL_W = 130
BLOCK_GAP = 12

_FALLBACK_BR: dict[tuple[str, str, str], float] = {
    ("healthy", "healthy-18-roi2", "fmax_minsc"): 0.348,
    ("healthy", "healthy-18-roi2", "fsum_maxsc"): 0.331,
    ("healthy", "healthy-18-roi2", "gradvmaxmul_minsc"): 0.456,
}


def _load_font(size: int = 13, bold: bool = True):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def _br_val(conn_m: dict, crit_m: dict, cat: str, roi: str, config_id: str) -> float | None:
    m = crit_m if config_id.startswith("gradvmaxmul_") else conn_m
    v = m.get(config_id, {}).get((cat, roi))
    if v is None:
        v = _FALLBACK_BR.get((cat, roi, config_id))
    return v


def _load_metrics() -> tuple[dict, dict]:
    conn: dict[str, dict[tuple[str, str], float]] = {}
    crit: dict[str, dict[tuple[str, str], float]] = {}
    for path, store in ((OUT_ROOT / "metrics_by_roi.csv", conn), (OUT_ROOT / "metrics_criteria.csv", crit)):
        if path.is_file():
            for row in csv.DictReader(path.open(encoding="utf-8")):
                store[row["config_id"]] = store.get(row["config_id"], {})
                store[row["config_id"]][(row["category"], row["roi"])] = float(row["br_mean_strict"])
    return conn, crit


def _stamp_on_image(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Faixa semitransparente no rodape da imagem com o nome do resultado."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(13)
    font_sm = _load_font(10, bold=False)
    h, w = pil.size[1], pil.size[0]
    line_h = 15
    bar_h = 6 + len(lines) * line_h
    draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0, 175))
    pil = Image.alpha_composite(pil, overlay)
    draw = ImageDraw.Draw(pil)
    y = h - bar_h + 3
    for i, text in enumerate(lines):
        draw.text((8, y + i * line_h), text, fill=(255, 255, 255), font=font if i == 0 else font_sm)
    return np.asarray(pil.convert("RGB"))


def _overlay_pred(orig: np.ndarray, gt: np.ndarray, pr: np.ndarray, pred_color: tuple[int, int, int]) -> np.ndarray:
    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    img = draw_contours(orig, gt, GT_CYAN, thickness=1)
    return draw_contours(img, pr, pred_color, thickness=1)


def _build_panel(
    category: str,
    roi: str,
    spec: list[tuple[list[str], str | None, tuple[int, int, int] | None]],
) -> list[tuple[str, np.ndarray]]:
    """spec: (rotulos na imagem, mask_key, cor_predicao)."""
    from PIL import Image

    case = OUT_ROOT / category / roi
    orig = np.asarray(Image.open(case / f"{roi}.png").convert("RGB"))
    gt = np.load(case / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)

    panels: list[tuple[str, np.ndarray]] = []
    for labels, key, color in spec:
        if key is None:
            img = orig.copy()
        elif key == "gt":
            sys.path.insert(0, str(PIPE))
            from percell_boundary_recall import draw_contours

            img = draw_contours(orig, gt, GT_CYAN, thickness=1)
        elif key == "cp":
            cp = np.load(case / "cp_flow" / "step04_masks_uint16.npy").astype(np.int32)
            img = _overlay_pred(orig, gt, cp, CP_YELLOW)
        else:
            pr = np.load(case / key / "merged_percell_sicle_masks_int32.npy").astype(np.int32)
            img = _overlay_pred(orig, gt, pr, color)  # type: ignore[arg-type]
        panels.append((labels[0], _stamp_on_image(img, labels)))
    return panels


def _assemble_block(block_title: str, row_label: str, panels: list[tuple[str, np.ndarray]], font) -> np.ndarray:
    from PIL import Image, ImageDraw

    h, w = panels[0][1].shape[:2]
    n = len(panels)
    strip_w = n * w + (n - 1) * GAP
    strip_h = h + TITLE_H
    strip = np.full((strip_h, strip_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in panels:
        strip[TITLE_H:, x : x + w] = img
        x += w + GAP

    pil = Image.fromarray(strip)
    draw = ImageDraw.Draw(pil)
    font_sm = _load_font(10)
    x = 0
    for short, _ in panels:
        draw.text((x + 4, 4), short, fill=(0, 0, 0), font=font_sm)
        x += w + GAP

    row_canvas = np.full((strip_h, ROW_LABEL_W + strip_w, 3), 248, dtype=np.uint8)
    row_canvas[:, ROW_LABEL_W:] = strip
    pil_row = Image.fromarray(row_canvas)
    draw_row = ImageDraw.Draw(pil_row)
    draw_row.text((6, 6), row_label, fill=(0, 0, 0), font=font)
    draw_row.text((6, 22), block_title, fill=(90, 90, 90), font=_load_font(10))
    return np.asarray(pil_row)


def _conn_spec(cat: str, roi: str, conn_m: dict, crit_m: dict) -> list:
    def br_line(cid: str) -> str:
        v = _br_val(conn_m, crit_m, cat, roi, cid)
        return f"BR = {v:.3f}" if v is not None else ""

    return [
        (["Original"], None, None),
        (["GT", "Gold Standard (referencia)", "contorno ciano"], "gt", None),
        (["Cellpose", "segmentacao inicial", "contorno amarelo"], "cp", None),
        (
            ["fmax + minsc", "resultado SICLE-IRREG", br_line("fmax_minsc")],
            "fmax_minsc",
            CONN_COLORS["fmax_minsc"],
        ),
        (
            ["fsum + maxsc", "resultado SICLE-COMP", br_line("fsum_maxsc")],
            "fsum_maxsc",
            CONN_COLORS["fsum_maxsc"],
        ),
        (
            ["gradvmaxmul + minsc", "resultado escolhido", br_line("gradvmaxmul_minsc")],
            "gradvmaxmul_minsc",
            CONN_COLORS["gradvmaxmul_minsc"],
        ),
    ]


def _crit_spec(
    cat: str,
    roi: str,
    conn_m: dict,
    crit_m: dict,
    *,
    highlight_minsc: bool = False,
    highlight_maxsc: bool = False,
) -> list:
    def lines(crit: str, cid: str) -> list[str]:
        v = _br_val(conn_m, crit_m, cat, roi, cid)
        out = [crit, "resultado SICLE (predicao)"]
        if highlight_minsc and crit == "minsc":
            out.insert(1, "criterio escolhido")
        if highlight_maxsc and crit == "maxsc":
            out.insert(1, "melhor BR nesta ROI")
        if v is not None:
            out.append(f"BR = {v:.3f}")
        return out

    return [
        (["Original"], None, None),
        (["GT", "Gold Standard (referencia)", "contorno ciano"], "gt", None),
        (lines("minsc", "gradvmaxmul_minsc"), "gradvmaxmul_minsc", CRIT_COLORS["minsc"]),
        (lines("maxsc", "gradvmaxmul_maxsc"), "gradvmaxmul_maxsc", CRIT_COLORS["maxsc"]),
        (lines("size", "gradvmaxmul_size"), "gradvmaxmul_size", CRIT_COLORS["size"]),
        (lines("spread", "gradvmaxmul_spread"), "gradvmaxmul_spread", CRIT_COLORS["spread"]),
    ]


def main() -> int:
    conn_m, crit_m = _load_metrics()
    font = _load_font(12)
    blocks: list[np.ndarray] = []

    cat1, roi1 = "healthy", "healthy-18-roi2"
    blocks.append(_assemble_block("Conectividade", roi1, _build_panel(cat1, roi1, _conn_spec(cat1, roi1, conn_m, crit_m)), font))
    blocks.append(
        _assemble_block(
            "Criterio (gradvmaxmul)",
            roi1,
            _build_panel(cat1, roi1, _crit_spec(cat1, roi1, conn_m, crit_m, highlight_minsc=True)),
            font,
        )
    )

    cat3, roi3 = "severe", "severe-03-roi2"
    blocks.append(_assemble_block("Conectividade severe", roi3, _build_panel(cat3, roi3, _conn_spec(cat3, roi3, conn_m, crit_m)), font))

    cat4, roi4 = "healthy", "healthy-19-roi2"
    blocks.append(
        _assemble_block(
            "Criterio — campo denso",
            roi4,
            _build_panel(
                cat4,
                roi4,
                _crit_spec(cat4, roi4, conn_m, crit_m, highlight_maxsc=True),
            ),
            font,
        )
    )

    from PIL import Image, ImageDraw

    leg_w = blocks[0].shape[1]
    header_h = 36
    header = np.full((header_h, leg_w, 3), 255, dtype=np.uint8)
    ImageDraw.Draw(Image.fromarray(header)).text(
        (8, 8),
        "Oral Epithelium — cada painel mostra o resultado indicado na faixa inferior",
        fill=(0, 0, 0),
        font=font,
    )

    sep = np.full((BLOCK_GAP, leg_w, 3), 255, dtype=np.uint8)
    stacked = [header]
    for b in blocks:
        stacked.extend([sep, b])
    canvas = np.vstack(stacked)

    PANELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(PANELS_OUT)
    print(f"Wrote {PANELS_OUT} ({canvas.shape[1]}x{canvas.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
