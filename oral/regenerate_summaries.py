#!/usr/bin/env python3
"""Regenerate summary.md with BR/Fb macro wins and ROI win counts."""
from __future__ import annotations

import csv
from pathlib import Path

from _paths import REPO, RUNS
from summary_metrics import write_summary_md


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def main() -> int:
    # iDISF 3-way
    p = RUNS / "percell_idisf_full" / "metrics_percell_idisf.csv"
    write_summary_md(
        RUNS / "percell_idisf_full" / "summary.md",
        title="Per-cell iDISF — outras células inconquistáveis",
        intro_lines=[
            "Pipeline: BG só na borda do crop; outras instâncias = inconquistáveis (estilo SICLE `--mask`).",
            "Merge: `--disable-and-merge` (SICLE cru equivalente).",
        ],
        csv_rel=str(p.relative_to(REPO)),
        rows=_read_csv(p),
        methods=["cellpose", "sicle_raw_legacy", "idisf_exclude_other"],
        reference="cellpose",
        primary="idisf_exclude_other",
        second_ref="sicle_raw_legacy",
        extra_metric_cols=[("pixel_dice", "Dice"), ("aji", "AJI")],
    )
    print(f"wrote {RUNS / 'percell_idisf_full' / 'summary.md'}")

    # Cellpose vs SICLE
    p = RUNS / "cellpose_vs_sicle" / "metrics_cellpose_vs_sicle.csv"
    write_summary_md(
        RUNS / "cellpose_vs_sicle" / "summary.md",
        title="Cellpose vs SICLE raw (Nf=2)",
        intro_lines=["Comparação direta: Cellpose step04 vs SICLE cru Nf=2 (gradvmaxmul+minsc)."],
        csv_rel=str(p.relative_to(REPO)),
        rows=_read_csv(p),
        methods=["cellpose", "sicle_nf2_raw"],
        reference="cellpose",
        primary="sicle_nf2_raw",
    )
    print(f"wrote {RUNS / 'cellpose_vs_sicle' / 'summary.md'}")

    # SICLE exclude other 3-way
    p = RUNS / "conquest_exclude_other_full" / "metrics_conquest_exclude_other.csv"
    write_summary_md(
        RUNS / "conquest_exclude_other_full" / "summary.md",
        title="SICLE — exclusão de outras células na conquista",
        intro_lines=[
            "Pipeline: gradvmaxmul+minsc, Nf=2, SICLE cru; `--mask` ROI + saliência zerada em vizinhos.",
        ],
        csv_rel=str(p.relative_to(REPO)),
        rows=_read_csv(p),
        methods=["cellpose", "sicle_raw_legacy", "sicle_raw_exclude_other"],
        reference="cellpose",
        primary="sicle_raw_exclude_other",
        second_ref="sicle_raw_legacy",
        extra_metric_cols=[("pixel_dice", "Dice")],
    )
    print(f"wrote {RUNS / 'conquest_exclude_other_full' / 'summary.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
