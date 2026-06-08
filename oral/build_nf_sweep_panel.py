#!/usr/bin/env python3
"""Panel: GT vs SICLE raw at Nf=2, 10, 50 (why BR collapses with more superpixels)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from _paths import PIPE, REPO, RUNS

NF_ROOT = RUNS / "nf_sweep_full"
BENCH = RUNS / "postprocess_ablation_full"
GT_CYAN = (0, 255, 255)
COLORS = {
    2: (0, 220, 0),
    10: (255, 200, 0),
    50: (255, 80, 80),
    500: (255, 0, 255),
}


def _br_from_csv(category: str, roi: str, nf: int) -> float | None:
    csv_path = NF_ROOT / "metrics_nf_sweep.csv"
    if not csv_path.is_file():
        return None
    with csv_path.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row["category"] == category and row["roi"] == roi and int(row["nf"]) == nf:
                return float(row["br_mean_strict"])
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="healthy")
    p.add_argument("--roi", default="healthy-24-roi1")
    p.add_argument("--nfs", default="2,10,50", help="comma-separated Nf values to show")
    args = p.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    nfs = [int(x) for x in args.nfs.split(",")]
    cat, stem = args.category, args.roi
    case_bench = BENCH / cat / stem
    case = NF_ROOT / cat / stem
    orig = np.asarray(Image.open(case_bench / f"{stem}.png").convert("RGB"))
    gt = np.load(case_bench / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    h, w = orig.shape[:2]

    def load_pr(nf: int) -> np.ndarray:
        n0 = max(200, nf + 20)
        sub = case / f"nf{nf}_n0{n0}_raw"
        alt = list(case.glob(f"nf{nf}_n0*_raw"))
        path = sub / "merged_percell_sicle_masks_int32.npy"
        if not path.is_file() and alt:
            path = alt[0] / "merged_percell_sicle_masks_int32.npy"
        return np.load(path).astype(np.int32)

    panels: list[tuple[str, np.ndarray]] = [("Original", orig)]
    base_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)
    panels.append(("GT (ciano)", base_gt))

    for nf in nfs:
        pr = load_pr(nf)
        br = _br_from_csv(cat, stem, nf)
        br_s = f"{br:.3f}" if br is not None else "?"
        col = COLORS.get(nf, (200, 200, 200))
        img = draw_contours(base_gt.copy(), pr, col, thickness=1)
        panels.append((f"Nf={nf}  BR={br_s}", img))

    gap, title_h, foot_h = 6, 30, 48
    total_w = len(panels) * w + (len(panels) - 1) * gap
    canvas = np.full((h + title_h + foot_h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for _, img in panels:
        canvas[title_h : title_h + h, x : x + w] = img
        x += w + gap

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except OSError:
        font = font_sm = ImageFont.load_default()

    x = 0
    for title, _ in panels:
        draw.text((x + 4, 6), title, fill=(0, 0, 0), font=font)
        x += w + gap

    foot_lines = [
        f"{cat}/{stem} — SICLE cru (sem pos-processo). Mais Nf = mais remocao de seeds (minsc) na hierarquia.",
        "BR cai porque a mascara encolhe / fragmenta: contorno predito deixa de coincidir com GT.",
        "Verde=Nf2 | amarelo=Nf10 | vermelho=Nf50",
    ]
    y = h + title_h + 6
    for line in foot_lines:
        draw.text((4, y), line, fill=(30, 30, 30), font=font_sm)
        y += 14

    out = NF_ROOT / "panels" / f"{cat}_{stem}_nf_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    print(f"Wrote {out}")
    return 0


def export_nf_sweep_3x3(
    category: str,
    stem: str,
    dest: Path,
    nfs: tuple[int, ...] = (2, 5, 10, 25, 50, 100),
) -> None:
    """Nine tiles for Beamer 3x3: orig, GT, six Nf levels, BR legend."""
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(PIPE))
    from percell_boundary_recall import draw_contours

    cat, stem = category, stem
    case_bench = BENCH / cat / stem
    case = NF_ROOT / cat / stem
    orig = np.asarray(Image.open(case_bench / f"{stem}.png").convert("RGB"))
    gt = np.load(case_bench / "gt" / "gold_standard_masks_int32.npy").astype(np.int32)
    base_gt = draw_contours(orig, gt, GT_CYAN, thickness=1)

    def load_pr(nf: int) -> np.ndarray:
        n0 = max(200, nf + 20)
        sub = case / f"nf{nf}_n0{n0}_raw"
        alt = sorted(case.glob(f"nf{nf}_n0*_raw"))
        path = sub / "merged_percell_sicle_masks_int32.npy"
        if not path.is_file() and alt:
            path = alt[0] / "merged_percell_sicle_masks_int32.npy"
        return np.load(path).astype(np.int32)

    tiles: list[tuple[str, np.ndarray]] = [("Original", orig), ("GT", base_gt)]
    legend_lines = [f"{cat}/{stem}", "strict BR:"]
    for nf in nfs:
        pr = load_pr(nf)
        br = _br_from_csv(cat, stem, nf)
        br_s = f"{br:.3f}" if br is not None else "?"
        col = COLORS.get(nf, (200, 200, 200))
        tiles.append((f"Nf={nf}", draw_contours(base_gt.copy(), pr, col, thickness=1)))
        legend_lines.append(f"  Nf={nf}: BR={br_s}")

    h0, w0 = orig.shape[:2]
    leg = np.full((h0, w0, 3), 255, dtype=np.uint8)
    pil_leg = Image.fromarray(leg)
    draw = ImageDraw.Draw(pil_leg)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except OSError:
        font = font_b = ImageFont.load_default()
    y = 8
    draw.text((8, y), legend_lines[0], fill=(0, 0, 0), font=font_b)
    y += 16
    for line in legend_lines[1:]:
        draw.text((8, y), line, fill=(30, 30, 30), font=font)
        y += 14
    tiles.append(("BR summary", np.asarray(pil_leg)))

    names = (
        "nf33_orig",
        "nf33_gt",
        "nf33_nf02",
        "nf33_nf05",
        "nf33_nf10",
        "nf33_nf25",
        "nf33_nf50",
        "nf33_nf100",
        "nf33_legend",
    )
    dest.mkdir(parents=True, exist_ok=True)
    for (title, img), name in zip(tiles, names):
        out = dest / f"{name}.png"
        Image.fromarray(img).save(out)
        print(f"  {out.name}  ({title})")


if __name__ == "__main__":
    raise SystemExit(main())
