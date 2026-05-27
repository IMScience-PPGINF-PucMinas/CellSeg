#!/usr/bin/env python3
"""BR slice analysis: charts + markdown report for ref_sicle_cp_blur05."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
BR_ROOT = _HERE / "out_sibgrapi2026_blur05" / "br_analysis"
OUT_DIR = _HERE / "reports" / "sicle_br_analysis"
TIE_EPS = 0.02

SLICE_ORDER = [f"12121_40x_slice{i}" for i in list(range(1, 10)) + [10, 11, 12]]
STRONG_SICLE = {f"12121_40x_slice{i}" for i in (1, 2, 5, 6, 7, 8)}
STRONG_CP = {f"12121_40x_slice{i}" for i in (9, 10, 11, 12)}


def _load_summary() -> list[dict]:
    p = BR_ROOT / "per_cell_br_summary.csv"
    rows = []
    with p.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row["slice"] != "ALL":
                rows.append(row)
    order = {s: i for i, s in enumerate(SLICE_ORDER)}
    rows.sort(key=lambda r: order.get(r["slice"], 99))
    return rows


def _load_cells() -> list[dict]:
    p = BR_ROOT / "per_cell_br_all.csv"
    with p.open(encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _plot_slice_bars(summary: list[dict]) -> Path:
    slices = [r["slice"].replace("12121_40x_", "") for r in summary]
    br_s = [float(r["br_sicle_mean"]) for r in summary]
    br_c = [float(r["br_cellpose_mean"]) for r in summary]
    x = np.arange(len(slices))
    w = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, br_s, w, label="SICLE (ref blur05)", color="#2d8a4e")
    ax.bar(x + w / 2, br_c, w, label="Cellpose", color="#c9a227")
    for i, r in enumerate(summary):
        if r["slice"] in STRONG_SICLE:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="green")
        elif r["slice"] in STRONG_CP:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(slices, rotation=45, ha="right")
    ax.set_ylabel("BR médio por célula GT")
    ax.set_ylim(0, 1.0)
    ax.set_title("Boundary Recall por slice — ref SICLE vs Cellpose (ε=0.02)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "Fundo verde: slices 1,2,5–8 (SICLE competitivo) · laranja: 9–12 (Cellpose domina)",
        ha="center",
        fontsize=9,
        color="#555",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "br_mean_by_slice.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_wins_stacked(summary: list[dict]) -> Path:
    slices = [r["slice"].replace("12121_40x_", "") for r in summary]
    s_w = [int(r["sicle_wins"]) for r in summary]
    c_w = [int(r["cellpose_wins"]) for r in summary]
    t_w = [int(r["ties"]) for r in summary]
    x = np.arange(len(slices))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, s_w, label="Vitórias SICLE", color="#2d8a4e")
    ax.bar(x, t_w, bottom=s_w, label="Empates", color="#9aa0a6")
    ax.bar(x, c_w, bottom=[a + b for a, b in zip(s_w, t_w)], label="Vitórias Cellpose", color="#c9a227")
    ax.set_xticks(x)
    ax.set_xticklabels(slices, rotation=45, ha="right")
    ax.set_ylabel("Nº células GT")
    ax.set_title("Vitórias BR por célula (stacked) — ref SICLE vs Cellpose")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "br_wins_stacked_by_slice.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _top_examples(cells: list[dict], n: int = 8) -> tuple[list[dict], list[dict]]:
    for row in cells:
        row["br_diff"] = float(row["br_diff_sicle_minus_cp"])
        row["br_s"] = float(row["br_sicle"])
        row["br_c"] = float(row["br_cellpose"])
        row["gt_area"] = int(row["gt_area"])

    sicle = [r for r in cells if r["winner"] == "sicle"]
    cp = [r for r in cells if r["winner"] == "cellpose"]
    sicle.sort(key=lambda r: -r["br_diff"])
    cp.sort(key=lambda r: r["br_diff"])
    return sicle[:n], cp[:n]


def _write_report(
    summary: list[dict],
    sicle_ex: list[dict],
    cp_ex: list[dict],
    chart1: Path,
    chart2: Path,
) -> Path:
    md = OUT_DIR / "sicle_br_slice_analysis.md"
    lines = [
        "# Análise BR por slice — ref `out_sibgrapi2026_blur05`",
        "",
        "Fonte: `br_analysis/per_cell_br_summary.csv` e `per_cell_br_all.csv`.",
        f"Métrica: BR estrito por célula (melhor instância predita), empate |ΔBR| ≤ {TIE_EPS}.",
        "",
        "## Gráficos",
        "",
        f"![BR médio por slice]({chart1.name})",
        "",
        f"![Vitórias BR empilhadas]({chart2.name})",
        "",
        "## Resumo por slice",
        "",
        "| Slice | n_gt | SICLE vit. | CP vit. | Empates | BR SICLE | BR CP | Regime |",
        "|-------|------|------------|---------|---------|----------|-------|--------|",
    ]
    for r in summary:
        sl = r["slice"]
        if sl in STRONG_SICLE:
            regime = "SICLE forte"
        elif sl in STRONG_CP:
            regime = "Cellpose forte"
        else:
            regime = "misto"
        lines.append(
            f"| {sl.replace('12121_40x_', '')} | {r['n_gt']} | {r['sicle_wins']} | "
            f"{r['cellpose_wins']} | {r['ties']} | {float(r['br_sicle_mean']):.3f} | "
            f"{float(r['br_cellpose_mean']):.3f} | {regime} |"
        )
    lines += [
        "",
        "## Exemplos — SICLE ganha (maior ΔBR)",
        "",
        "Composites em `br_analysis/<slice>/sicle_wins/` (GT ciano, SICLE verde, CP amarelo).",
        "",
        "| Slice | gt_id | área | BR SICLE | BR CP | ΔBR | PNG |",
        "|-------|-------|------|----------|-------|-----|-----|",
    ]
    for r in sicle_ex:
        stem = r["slice"]
        gid = int(r["gt_id"])
        png = BR_ROOT / stem / "sicle_wins" / f"cell_{gid:05d}_brS{r['br_s']:.3f}_brC{r['br_c']:.3f}.png"
        rel = f"../../out_sibgrapi2026_blur05/br_analysis/{stem}/sicle_wins/{png.name}"
        if not png.is_file():
            rel = "—"
        lines.append(
            f"| {stem.replace('12121_40x_', '')} | {gid} | {r['gt_area']} | "
            f"{r['br_s']:.3f} | {r['br_c']:.3f} | {r['br_diff']:+.3f} | `{png.name}` |"
        )
    lines += [
        "",
        "## Exemplos — Cellpose ganha (maior |ΔBR|)",
        "",
        "Composites em `br_analysis/<slice>/cellpose_wins/`.",
        "",
        "| Slice | gt_id | área | BR SICLE | BR CP | ΔBR | PNG |",
        "|-------|-------|------|----------|-------|-----|-----|",
    ]
    for r in cp_ex:
        stem = r["slice"]
        gid = int(r["gt_id"])
        png = BR_ROOT / stem / "cellpose_wins" / f"cell_{gid:05d}_brS{r['br_s']:.3f}_brC{r['br_c']:.3f}.png"
        lines.append(
            f"| {stem.replace('12121_40x_', '')} | {gid} | {r['gt_area']} | "
            f"{r['br_s']:.3f} | {r['br_c']:.3f} | {r['br_diff']:+.3f} | `{png.name}` |"
        )
    lines += [
        "",
        "## Padrões",
        "",
        "- **Slices 1, 2, 5–8** (`macro_nuclick` GT): SICLE vence BR em ~45–60% das células; BR médio próximo ou acima do Cellpose.",
        "- **Slices 9–12** (`union` GT): Cellpose domina (60–81% vitórias); BR médio do CP muito maior.",
        "- **Empates** (~23% global): contornos equivalentes dentro de ε — refinamento SICLE não muda BR.",
        "- **Sem blur na saliência**: BR médio cai para ~0,53 (ver `sicle_nolin_noblur`); blur é pré-requisito para `gradvmaxmul`.",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()
    cells = _load_cells()
    c1 = _plot_slice_bars(summary)
    c2 = _plot_wins_stacked(summary)
    s_ex, c_ex = _top_examples(cells, n=10)
    md = _write_report(summary, s_ex, c_ex, c1, c2)
    print(f"Wrote {c1}")
    print(f"Wrote {c2}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
