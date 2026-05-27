#!/usr/bin/env python3
"""Macro comparison: Boundary Recall (BR) wins per GT cell + secondary overlap metrics."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
TIE_EPS = 0.02
BR_MARGIN = 8
SLICES_EVAL = [f"12121_40x_slice{i}" for i in range(1, 9)]  # ignore 9–12 (union GT)
N_GT = 945  # cells in slices 1–8, macro_nuclick

METHODS: list[tuple[str, str, str]] = [
    # label, out_root (relative to _HERE), mask relpath
    (
        "ref_sicle_cp_blur05",
        "out_sibgrapi2026_blur05",
        "sicle/merged_percell_sicle_masks_int32.npy",
    ),
    (
        "sicle_nolin_blur05",
        "out_sibgrapi2026_nolin",
        "sicle/merged_percell_sicle_masks_int32.npy",
    ),
    (
        "sicle_nolin_noblur",
        "out_sibgrapi2026_nolin_noblur",
        "sicle/merged_percell_sicle_masks_int32.npy",
    ),
    ("cellpose", "out_cellvit_br", "cp_flow/step04_masks_uint16.npy"),
    ("cellvit", "out_cellvit_br", "cellvit_flow/step04_masks_uint16.npy"),
    ("sicle_cellvit", "out_cellvit_br", "sicle/merged_percell_sicle_masks_int32.npy"),
    ("br_pick_cp_vs_sicle", "out_cellvit_br", "br_merge/merged_br_pick_masks_int32.npy"),
]


def _gt_path(case: Path) -> Path | None:
    for name in (
        "macro_nuclick_masks_int32.npy",
        "union_masks_int32.npy",
        "nuclick_masks_int32.npy",
    ):
        p = case / "gt" / name
        if p.is_file():
            return p
    return None


def _br_per_gt(gt: np.ndarray, pr: np.ndarray, margin: int = BR_MARGIN) -> dict[int, float]:
    """Strict per-cell BR: best-matching pred instance only (iftEvalBR)."""
    from percell_boundary_recall import (
        bbox_of_mask,
        compute_boundary_recall,
        isolate_pred_for_gt,
    )

    gt = np.asarray(gt, dtype=np.int32)
    pr = np.asarray(pr, dtype=np.int32)
    if pr.shape != gt.shape:
        h, w = min(gt.shape[0], pr.shape[0]), min(gt.shape[1], pr.shape[1])
        gt, pr = gt[:h, :w], pr[:h, :w]
    h, w = gt.shape

    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    out: dict[int, float] = {}
    for gid in gt_ids:
        m = gt == int(gid)
        if not m.any():
            continue
        r0, r1, c0, c1 = bbox_of_mask(m)
        r0 = max(0, r0 - margin)
        c0 = max(0, c0 - margin)
        r1 = min(h, r1 + margin)
        c1 = min(w, c1 + margin)
        gt_crop = gt[r0:r1, c0:c1]
        pr_crop = pr[r0:r1, c0:c1]
        gt_iso = np.where(gt_crop == int(gid), gt_crop, 0)
        pr_iso, _ = isolate_pred_for_gt(pr_crop, gt_crop, int(gid))
        br, _, _ = compute_boundary_recall(pr_iso, gt_iso)
        out[int(gid)] = float(br)
    return out


def _macro_eval(root: Path, rel: str) -> dict[str, float]:
    from evaluate_sibgrapi2026 import evaluate_pair

    rows = []
    for case in sorted(root.iterdir()):
        if not case.is_dir():
            continue
        gt = _gt_path(case)
        pr = case / rel
        if gt is None or not pr.is_file():
            continue
        r = evaluate_pair(gt, pr)
        if "error" not in r:
            rows.append(r)
    if not rows:
        return {}
    return {
        "dice": float(np.mean([r["pixel_dice"] for r in rows])),
        "aji": float(np.mean([r["aji"] for r in rows])),
        "pq": float(np.mean([r["pq"] for r in rows])),
        "f1": float(np.mean([r["f1_0.5"] for r in rows])),
        "map": float(np.mean([r["map_dsb"] for r in rows])),
    }


def _ref_blur05_br_pairwise() -> tuple[int, int, int, float, float] | None:
    """SICLE vs Cellpose BR wins from out_sibgrapi2026_blur05/br_analysis/per_cell_br_summary.csv."""
    p = _HERE / "out_sibgrapi2026_blur05" / "br_analysis" / "per_cell_br_summary.csv"
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row.get("slice") == "ALL":
                return (
                    int(row["sicle_wins"]),
                    int(row["cellpose_wins"]),
                    int(row["ties"]),
                    float(row["br_sicle_mean"]),
                    float(row["br_cellpose_mean"]),
                )
    return None


def _br_pick_wins(root: Path) -> tuple[int, int, int]:
    """Per-GT BR pick: sicle vs cellpose (from br_merge CSVs)."""
    s, c, t = 0, 0, 0
    for case in sorted(root.iterdir()):
        if not case.is_dir():
            continue
        csv_path = case / "br_merge" / "per_cell_br_pick.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                w = row.get("winner", "")
                if w == "sicle":
                    s += 1
                elif w == "cellpose":
                    c += 1
                else:
                    t += 1
    return s, c, t


def main() -> None:
    labels = [m[0] for m in METHODS]
    wins = {lb: 0 for lb in labels}
    ties_multi = 0
    missed = 0
    br_mean_sum = {lb: 0.0 for lb in labels}
    br_mean_n = {lb: 0 for lb in labels}

    ref_root = _HERE / "out_cellvit_br"
    slices = [s for s in SLICES_EVAL if (ref_root / s / "gt").exists()]

    br_tables: dict[str, dict[tuple[str, int], float]] = {lb: {} for lb in labels}

    for stem in slices:
        case = ref_root / stem
        gt_path = _gt_path(case)
        if gt_path is None:
            continue
        gt = np.load(gt_path).astype(np.int32)
        for label, root_rel, rel in METHODS:
            root = _HERE / root_rel
            pr_path = root / stem / rel
            if not pr_path.is_file():
                continue
            pr = np.load(pr_path).astype(np.int32)
            if pr.shape != gt.shape:
                h, w = min(gt.shape[0], pr.shape[0]), min(gt.shape[1], pr.shape[1])
                gt_c, pr_c = gt[:h, :w], pr[:h, :w]
            else:
                gt_c, pr_c = gt, pr
            for gid, br in _br_per_gt(gt_c, pr_c).items():
                br_tables[label][(stem, gid)] = br
                br_mean_sum[label] += br
                br_mean_n[label] += 1

    all_keys = set()
    for t in br_tables.values():
        all_keys |= set(t.keys())

    for key in sorted(all_keys):
        scores = {lb: br_tables[lb].get(key, 0.0) for lb in labels}
        if max(scores.values()) <= 0.0 and all(v <= 0.0 for v in scores.values()):
            missed += 1
            continue
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_br = ranked[0][1]
        second_br = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_br - second_br > TIE_EPS:
            wins[ranked[0][0]] += 1
        else:
            ties_multi += 1

    metrics = {lb: _macro_eval(_HERE / root_rel, rel) for lb, root_rel, rel in METHODS}
    br_macro_mean = {
        lb: (br_mean_sum[lb] / br_mean_n[lb] if br_mean_n[lb] else float("nan"))
        for lb in labels
    }
    pick_s, pick_c, pick_t = _br_pick_wins(ref_root)
    ref_hist = _ref_blur05_br_pairwise()

    cp_br = br_tables["cellpose"]
    vs_cp: dict[str, dict[str, int]] = {
        lb: {"wins": 0, "losses": 0, "ties": 0} for lb in labels if lb != "cellpose"
    }
    for key in all_keys:
        br_cp = cp_br.get(key, 0.0)
        for lb in labels:
            if lb == "cellpose":
                continue
            br_m = br_tables[lb].get(key, 0.0)
            diff = br_m - br_cp
            if br_m <= 0.0 and br_cp <= 0.0:
                continue
            if diff > TIE_EPS:
                vs_cp[lb]["wins"] += 1
            elif -diff > TIE_EPS:
                vs_cp[lb]["losses"] += 1
            else:
                vs_cp[lb]["ties"] += 1

    out_csv = _HERE / "macro_methods_comparison.csv"
    out_md = _HERE / "macro_methods_comparison.md"

    rows_out = []
    for label, root_rel, rel in METHODS:
        m = metrics.get(label, {})
        w = wins.get(label, 0)
        row = {
            "method": label,
            "br_mean": br_macro_mean.get(label, float("nan")),
            "br_wins_n": w,
            "br_wins_pct": 100.0 * w / N_GT,
            "dice": m.get("dice", float("nan")),
            "aji": m.get("aji", float("nan")),
            "pq": m.get("pq", float("nan")),
            "f1": m.get("f1", float("nan")),
            "map": m.get("map", float("nan")),
        }
        if label != "cellpose" and label in vs_cp:
            row["vs_cellpose_wins"] = vs_cp[label]["wins"]
            row["vs_cellpose_pct"] = 100.0 * vs_cp[label]["wins"] / N_GT
            row["vs_cellpose_ties"] = vs_cp[label]["ties"]
        if label == "br_pick_cp_vs_sicle":
            row["br_picks_sicle"] = pick_s
            row["br_picks_cellpose"] = pick_c
        rows_out.append(row)

    fieldnames = [
        "method",
        "br_mean",
        "br_wins_n",
        "br_wins_pct",
        "vs_cellpose_wins",
        "vs_cellpose_pct",
        "vs_cellpose_ties",
        "dice",
        "aji",
        "pq",
        "f1",
        "map",
        "br_picks_sicle",
        "br_picks_cellpose",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    lines = [
        f"# Comparação macro (slices 1–8, GT macro_nuclick, n_gt={N_GT})",
        "",
        "**Métrica padrão: Boundary Recall (BR) estrito por célula** — só a instância predita "
        f"com maior overlap no GT; vitórias se |ΔBR|>{TIE_EPS}; bbox+margin={BR_MARGIN}px (iftEvalBR).",
        "Dice/AJI/PQ/F1/mAP ficam como métricas secundárias de sobreposição.",
        "",
        "| Método | BR médio | Vitórias BR* (n) | Vitórias BR (%) | vs Cellpose (n) | vs Cellpose (%) | Empates vs CP | Dice | AJI |",
        "|--------|----------|------------------|-----------------|-----------------|-----------------|---------------|------|-----|",
    ]
    for r in rows_out:
        if r["method"] == "cellpose":
            ref_row = next(x for x in rows_out if x["method"] == "ref_sicle_cp_blur05")
            cp_w = vs_cp["ref_sicle_cp_blur05"]["losses"]
            vcp = (
                f"{cp_w} ({100*cp_w/N_GT:.1f}%) | — | "
                f"{ref_row.get('vs_cellpose_ties', '—')}"
            )
        else:
            vcp = (
                f"{r.get('vs_cellpose_wins', '—')} | "
                f"{r.get('vs_cellpose_pct', 0):.1f}% | "
                f"{r.get('vs_cellpose_ties', '—')}"
            )
        lines.append(
            f"| {r['method']} | {r['br_mean']:.4f} | {r['br_wins_n']} | {r['br_wins_pct']:.1f}% | "
            f"{vcp} | {r['dice']:.4f} | {r['aji']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"* Vitórias BR = maior BR entre os {len(labels)} métodos "
        f"(empate multi-método: **{ties_multi}** células, {100*ties_multi/N_GT:.1f}%)."
    )
    lines.append(f"† Apenas slices 1–8, GT `macro_nuclick`, n_gt={N_GT}.")
    lines += [
        "",
        "### BR pick oracle (SICLE vs Cellpose por célula, `br_merge`)",
        "| Escolhe | Células (n) | % |",
        "|---------|-------------|---|",
        f"| SICLE | {pick_s} | {100*pick_s/N_GT:.1f}% |",
        f"| Cellpose | {pick_c} | {100*pick_c/N_GT:.1f}% |",
        f"| Empate BR | {pick_t} | {100*pick_t/N_GT:.1f}% |",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_md.read_text(encoding="utf-8"))
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
