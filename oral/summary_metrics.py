#!/usr/bin/env python3
"""Build summary.md with macro means and win counts (by mean vs by ROI count) for BR and Fb."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

TIE_DELTA = 0.01


def _method_key(row: dict) -> str:
    return row.get("method") or row.get("variant_id") or ""


def _roi_rows(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    """{(category, roi): {method: row}}."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        k = (r["category"], r["roi"])
        m = _method_key(r)
        if m:
            out[k][m] = r
    return out


def macro_means(rows: list[dict], metric: str) -> dict[str, float]:
    by_m: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        m = _method_key(r)
        v = r.get(metric)
        if m and v not in (None, ""):
            by_m[m].append(float(v))
    return {m: float(np.mean(v)) for m, v in by_m.items() if v}


def macro_winner(means: dict[str, float]) -> tuple[str, float]:
    if not means:
        return "—", float("nan")
    best = max(means, key=means.get)
    return best, means[best]


def pairwise_roi_wins(
    by_roi: dict[tuple[str, str], dict[str, dict]],
    method_a: str,
    method_b: str,
    metric: str,
    *,
    tol: float = TIE_DELTA,
) -> dict[str, int]:
    """Head-to-head per ROI: a wins / b wins / tie."""
    counts = {method_a: 0, method_b: 0, "tie": 0}
    for cells in by_roi.values():
        if method_a not in cells or method_b not in cells:
            continue
        va = float(cells[method_a][metric])
        vb = float(cells[method_b][metric])
        d = va - vb
        if d > tol:
            counts[method_a] += 1
        elif d < -tol:
            counts[method_b] += 1
        else:
            counts["tie"] += 1
    return counts


def best_on_roi_wins(
    by_roi: dict[tuple[str, str], dict[str, dict]],
    methods: list[str],
    metric: str,
    *,
    tol: float = TIE_DELTA,
) -> dict[str, int]:
    """Per ROI: method with highest metric (ties split or counted as tie)."""
    counts: dict[str, int] = {m: 0 for m in methods}
    counts["tie"] = 0
    for cells in by_roi.values():
        vals = [
            (m, float(cells[m][metric]))
            for m in methods
            if m in cells and cells[m].get(metric) not in (None, "")
        ]
        if len(vals) < 2:
            continue
        best_v = max(v for _, v in vals)
        near = [m for m, v in vals if abs(v - best_v) <= tol]
        if len(near) == 1:
            counts[near[0]] += 1
        else:
            counts["tie"] += 1
    return counts


def _format_wins_block(
    metric_label: str,
    metric_col: str,
    means: dict[str, float],
    methods: list[str],
    by_roi: dict[tuple[str, str], dict[str, dict]],
    *,
    reference: str,
    primary: str | None = None,
    second_ref: str | None = None,
    include_best_on_roi: bool = True,
) -> list[str]:
    lines: list[str] = []
    winner_m, winner_v = macro_winner(means)
    lines.append(f"### {metric_label}")
    lines.append("")
    lines.append("| Método | média macro |")
    lines.append("|--------|------------:|")
    for m in methods:
        if m in means:
            mark = " **← vitória média**" if m == winner_m else ""
            lines.append(f"| `{m}` | {means[m]:.4f}{mark} |")
    lines.append("")
    lines.append(
        f"**Vitória pela média macro:** `{winner_m}` ({winner_v:.4f}) "
        f"— maior {metric_label} médio nas {len(by_roi)} ROIs."
    )
    lines.append("")

    if primary and reference:
        pw = pairwise_roi_wins(by_roi, primary, reference, metric_col)
        n = pw[primary] + pw[reference] + pw["tie"]
        lines.append(
            f"**Vitórias por contagem de ROIs** (par a par `{primary}` vs `{reference}`, "
            f"Δ>{TIE_DELTA}):"
        )
        lines.append("")
        lines.append(f"- `{primary}`: **{pw[primary]}** ROIs")
        lines.append(f"- `{reference}`: **{pw[reference]}** ROIs")
        lines.append(f"- empate: **{pw['tie']}** ROIs")
        lines.append("")
        if pw[primary] > pw[reference]:
            lines.append(
                f"→ Na **contagem**, `{primary}` vence `{reference}` em {metric_label} "
                f"({pw[primary]} vs {pw[reference]} ROIs)."
            )
        elif pw[reference] > pw[primary]:
            lines.append(
                f"→ Na **contagem**, `{reference}` vence `{primary}` em {metric_label} "
                f"({pw[reference]} vs {pw[primary]} ROIs)."
            )
        else:
            lines.append(f"→ Na **contagem** (par a par), empate técnico em {metric_label}.")
        lines.append("")

    if primary and second_ref and second_ref != reference:
        pw2 = pairwise_roi_wins(by_roi, primary, second_ref, metric_col)
        lines.append(
            f"**Vitórias por contagem** (`{primary}` vs `{second_ref}`, Δ>{TIE_DELTA}): "
            f"{primary} **{pw2[primary]}** | {second_ref} **{pw2[second_ref]}** | "
            f"empate **{pw2['tie']}**"
        )
        lines.append("")

    if include_best_on_roi and len(methods) >= 2:
        bw = best_on_roi_wins(by_roi, methods, metric_col)
        lines.append(
            f"**Vitórias por contagem** (maior {metric_label} no ROI entre "
            f"{' / '.join(f'`{m}`' for m in methods)}):"
        )
        lines.append("")
        for m in methods:
            lines.append(f"- `{m}`: **{bw.get(m, 0)}** ROIs")
        lines.append(f"- empate (vários no topo): **{bw.get('tie', 0)}** ROIs")
        lines.append("")

    return lines


def write_summary_md(
    out_path: Path,
    *,
    title: str,
    intro_lines: list[str],
    csv_rel: str,
    rows: list[dict],
    methods: list[str],
    reference: str = "cellpose",
    primary: str | None = None,
    second_ref: str | None = None,
    extra_metric_cols: list[tuple[str, str]] | None = None,
) -> None:
    """Write unified summary with BR and Fb win breakdown."""
    by_roi = _roi_rows(rows)
    n_rois = len(by_roi)

    br_means = macro_means(rows, "br_mean_strict")
    fb_means = macro_means(rows, "fb_mean_strict")

    lines = [f"# {title}", ""]
    lines.extend(intro_lines)
    lines.append("")
    lines.append(f"CSV: `{csv_rel}`")
    lines.append("")
    lines.append(f"ROIs avaliadas: **{n_rois}**. Empate par a par: |Δ| ≤ {TIE_DELTA}.")
    lines.append("")
    lines.append("## Métricas de borda")
    lines.append("")
    lines.append(
        "- **BR** (Boundary Recall, Stutz/iftEvalBR): recall de pixels de borda GT "
        "cobertos pela predição (tolerância relativa à diagonal do crop)."
    )
    lines.append(
        "- **Fb** (Boundary F-measure, Arbeláez/BSDS): F1 entre contornos 1 px com "
        "tolerância 0.0075×diagonal (per-cell strict)."
    )
    lines.append("")
    lines.append(
        "Para cada métrica reportamos: (1) **vitória pela média macro** — quem tem o "
        "maior valor médio nas 100 ROIs; (2) **vitória por contagem** — quantas ROIs "
        "cada método ganha em comparações par a par ou como melhor no ROI."
    )
    lines.append("")

    if extra_metric_cols:
        lines.append("## Outras métricas (média macro)")
        lines.append("")
        lines.append("| Método | " + " | ".join(c[1] for c in extra_metric_cols) + " |")
        lines.append("|--------|" + "|".join("--------:" for _ in extra_metric_cols) + "|")
        for m in methods:
            cells = []
            for col, _ in extra_metric_cols:
                sub = [float(r[col]) for r in rows if _method_key(r) == m and r.get(col)]
                cells.append(f"{np.mean(sub):.4f}" if sub else "—")
            lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        _format_wins_block(
            "BR (Boundary Recall)",
            "br_mean_strict",
            br_means,
            methods,
            by_roi,
            reference=reference,
            primary=primary,
            second_ref=second_ref,
        )
    )
    lines.extend(
        _format_wins_block(
            "Fb (Boundary F-measure)",
            "fb_mean_strict",
            fb_means,
            methods,
            by_roi,
            reference=reference,
            primary=primary,
            second_ref=second_ref,
        )
    )

    # Resumo executivo: média vs contagem concordam?
    lines.append("## Resumo: média macro vs contagem de ROIs")
    lines.append("")
    br_w, _ = macro_winner(br_means)
    fb_w, _ = macro_winner(fb_means)
    if primary:
        br_pw = pairwise_roi_wins(by_roi, primary, reference, "br_mean_strict")
        fb_pw = pairwise_roi_wins(by_roi, primary, reference, "fb_mean_strict")
        br_count_winner = primary if br_pw[primary] > br_pw[reference] else (
            reference if br_pw[reference] > br_pw[primary] else "empate"
        )
        fb_count_winner = primary if fb_pw[primary] > fb_pw[reference] else (
            reference if fb_pw[reference] > fb_pw[primary] else "empate"
        )
        lines.append("| Métrica | Vitória **média macro** | Vitória **contagem** (vs `" + reference + "`) | Concordam? |")
        lines.append("|---------|-------------------------|---------------------------------------------|------------|")
        for label, mw, cw in (
            ("BR", br_w, br_count_winner),
            ("Fb", fb_w, fb_count_winner),
        ):
            ok = "sim" if mw == cw else "não"
            lines.append(f"| {label} | `{mw}` | `{cw}` | {ok} |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
