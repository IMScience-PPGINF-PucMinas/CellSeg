#!/usr/bin/env python3
"""
Try several **RunSICLE** path-cost pairs ``(--conn-opt, --crit-opt)`` on the same per-cell crops
as ``percell_sicle_cellprob_pipeline.py`` (cellprob saliency + FG scribbles), and tabulate timing +
Dice overlap of the SICLE foreground with the Cellpose mask in that bbox.

Use this to compare presets (irregular vs compact) and a few cross-combinations before committing
to one setting in the full pipeline.

Examples::

    cd new_pipeline
    PYTHONPATH=../cellpose python test_sicle_path_costs.py \\
        --from-dir ./cp_flow_out -o ./sicle_path_cost_try --max-cells 15

    # Only the two literature-style presets
    PYTHONPATH=../cellpose python test_sicle_path_costs.py \\
        --from-dir ./cp_flow_out -o ./out --pair-mode presets_only --max-cells 30

    # Add custom pairs (semicolon-separated ``conn,crit``)
    PYTHONPATH=../cellpose python test_sicle_path_costs.py \\
        --from-dir ./cp_flow_out -o ./out --extra-pairs "fmax,size;fsum,spread"
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
_CELLPOSE_DIR = _REPO_ROOT / "cellpose"
for _d in (_PKG_DIR, _CELLPOSE_DIR):
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)


def _path_cost_table(pair_mode: str) -> list[tuple[str, str, str]]:
    """Return (label, conn_opt, crit_opt) rows to test."""
    presets = [
        ("irregular_fmax_minsc", "fmax", "minsc"),
        ("compact_fsum_maxsc", "fsum", "maxsc"),
    ]
    cross = [
        ("cross_fmax_maxsc", "fmax", "maxsc"),
        ("cross_fsum_minsc", "fsum", "minsc"),
    ]
    grid = [
        ("fmax_minsc", "fmax", "minsc"),
        ("fmax_maxsc", "fmax", "maxsc"),
        ("fsum_minsc", "fsum", "minsc"),
        ("fsum_maxsc", "fsum", "maxsc"),
    ]
    if pair_mode == "presets_only":
        return presets
    if pair_mode == "presets_and_cross":
        return presets + cross
    if pair_mode == "grid_2x2":
        return grid
    if pair_mode == "full_small_grid":
        # conn × crit (common RunSICLE options; some pairs may fail on certain crops)
        conns = ("fmax", "fsum")
        crits = ("minsc", "maxsc", "size", "spread")
        out: list[tuple[str, str, str]] = []
        for c in conns:
            for t in crits:
                out.append((f"{c}_{t}", c, t))
        return out
    raise ValueError(f"unknown pair_mode: {pair_mode}")


def _dice(pred: "np.ndarray", gt: "np.ndarray") -> float:
    import numpy as np

    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


SICLE_N0_DEFAULT = 500
SICLE_NF_DEFAULT = 2
SICLE_ALPHA_DEFAULT = 0.9
SICLE_MAXITERS_DEFAULT = 22
SICLE_IRREG_DEFAULT = 0.12
SICLE_ADHR_DEFAULT = 16
SICLE_PEN_OPT_DEFAULT = "none"


def _find_sicle_binary() -> Path:
    env_path = os.environ.get("SICLE_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"SICLE_BIN={env_path} does not exist")
    candidates = [
        _REPO_ROOT / "SICLE" / "bin" / "RunSICLE",
        _REPO_ROOT / "PIPELINE_UOIFT_SICLE" / "uoift_sicle" / "SICLE" / "bin" / "RunSICLE",
        Path.home() / "SICLE" / "bin" / "RunSICLE",
        Path("/usr/local/bin/RunSICLE"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("RunSICLE binary not found. Set SICLE_BIN or place it under SICLE/bin.")


def _run_sicle_on_crop(
    img_crop: "np.ndarray",
    fg_coords: list[tuple[int, int]],
    temp_dir: Path,
    crop_name: str,
    sicle_bin: Path,
    *,
    n0: int,
    nf: int,
    alpha: float,
    max_iters: int,
    irreg: float,
    adhr: int,
    conn_opt: str,
    crit_opt: str,
    pen_opt: str = SICLE_PEN_OPT_DEFAULT,
) -> "np.ndarray":
    import numpy as np
    from PIL import Image

    temp_dir.mkdir(parents=True, exist_ok=True)
    img_u8 = np.asarray(img_crop, dtype=np.uint8)
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8, img_u8, img_u8], axis=-1)
    else:
        img_u8 = img_u8[..., :3]

    img_path = temp_dir / f"{crop_name}_sicle_input.ppm"
    out_path = temp_dir / f"{crop_name}_sicle_output.pgm"
    Image.fromarray(img_u8).save(img_path)

    objsm_path = temp_dir / f"{crop_name}_sicle_objsm.pgm"
    if conn_opt in ("gradvmax", "gradvmaxmul"):
        r = img_u8[:, :, 0].astype(np.float32)
        g = img_u8[:, :, 1].astype(np.float32)
        b = img_u8[:, :, 2].astype(np.float32)
        if np.allclose(r, g) and np.allclose(r, b):
            sal_gray = img_u8[:, :, 0]
        else:
            sal_gray = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(sal_gray, mode="L").save(objsm_path)

    def _run_once(current_n0: int) -> tuple[bool, str]:
        cmd = [
            str(sicle_bin),
            "--img", str(img_path),
            "--out", str(out_path),
            "--n0", str(current_n0),
            "--nf", str(nf),
            "--alpha", str(alpha),
            "--max-iters", str(max_iters),
            "--conn-opt", conn_opt,
            "--crit-opt", crit_opt,
            "--pen-opt", pen_opt,
            "--sampl-opt", "grid",
            "--irreg", str(irreg),
            "--adhr", str(adhr),
        ]
        if conn_opt in ("gradvmax", "gradvmaxmul"):
            cmd += ["--objsm", str(objsm_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0 and out_path.exists(), proc.stderr

    ok, stderr = _run_once(n0)
    if not ok and "Invalid N0 value of" in stderr and "It must be within ]2," in stderr:
        m = re.search(r"It must be within ]2,(\d+)\[", stderr)
        if m:
            max_allowed = int(m.group(1)) - 1
            if max_allowed > 2:
                ok, stderr = _run_once(max(3, max_allowed))
    if not ok:
        raise RuntimeError(f"SICLE failed for {crop_name}: {stderr}")

    label_raw = np.array(Image.open(out_path), dtype=np.int32)
    if label_raw.ndim > 2:
        label_raw = label_raw.squeeze()
    if label_raw.ndim > 2:
        label_raw = label_raw[..., 0]
    h, w = label_raw.shape[:2]
    uniq = np.unique(label_raw)
    uniq = uniq[uniq > 0]
    if uniq.size == 0:
        return np.zeros_like(label_raw, dtype=np.uint8)

    cnt: dict[int, int] = {}
    for u in uniq:
        cnt[u] = sum(1 for (x, y) in fg_coords if 0 <= y < h and 0 <= x < w and label_raw[y, x] == u)
    obj_label = max(uniq, key=lambda u: cnt.get(u, 0))
    out = np.full_like(label_raw, 2, dtype=np.uint8)
    out[label_raw == obj_label] = 1
    return out


def main() -> int:
    import numpy as np
    from percell_sicle_cellprob_pipeline import (
        bbox_for_label,
        cellprob_crop_to_saliency_u8,
        fg_scribble_coords,
        load_cellprob_masks,
    )

    p = argparse.ArgumentParser(description="Benchmark RunSICLE conn/crit pairs on per-cell crops.")
    p.add_argument("--from-dir", type=str, required=True, help="Folder with step03 npz + step04 masks npy")
    p.add_argument("-o", "--out-dir", type=str, required=True, help="Output folder for CSV + summary")
    p.add_argument("--max-cells", type=int, default=20, help="Max number of Cellpose instances to test (sorted by label id)")
    p.add_argument(
        "--pair-mode",
        choices=("presets_only", "presets_and_cross", "grid_2x2", "full_small_grid"),
        default="presets_and_cross",
        help="Which (conn,crit) combinations to try (see script docstring)",
    )
    p.add_argument(
        "--extra-pairs",
        type=str,
        default="",
        help='Extra pairs: semicolon-separated ``conn,crit`` e.g. ``fmax,size;fsum,spread``',
    )
    p.add_argument("--margin", type=int, default=8)
    p.add_argument("--fg-erosion-pixels", type=int, default=0)
    p.add_argument("--min-cell-area", type=int, default=64)
    p.add_argument("--sicle-n0", type=int, default=SICLE_N0_DEFAULT)
    p.add_argument("--sicle-nf", type=int, default=SICLE_NF_DEFAULT)
    p.add_argument("--sicle-alpha", type=float, default=SICLE_ALPHA_DEFAULT)
    p.add_argument("--sicle-max-iters", type=int, default=SICLE_MAXITERS_DEFAULT)
    p.add_argument("--sicle-irreg", type=float, default=SICLE_IRREG_DEFAULT)
    p.add_argument("--sicle-adhr", type=int, default=SICLE_ADHR_DEFAULT)
    args = p.parse_args()

    pairs = _path_cost_table(args.pair_mode)
    if args.extra_pairs.strip():
        for chunk in args.extra_pairs.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [x.strip() for x in chunk.split(",")]
            if len(parts) != 2:
                raise SystemExit(f"bad --extra-pairs segment {chunk!r}; use conn,crit")
            conn, crit = parts[0], parts[1]
            pairs.append((f"extra_{conn}_{crit}", conn, crit))

    from_dir = Path(args.from_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cellprob, masks, _dP = load_cellprob_masks(from_dir)
    h, w = cellprob.shape
    if masks.shape != (h, w):
        raise SystemExit(f"shape mismatch {cellprob.shape} vs {masks.shape}")

    sicle_bin = _find_sicle_binary()
    labels = sorted(int(x) for x in np.unique(masks) if int(x) > 0)[: max(0, args.max_cells)]

    rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="sicle_path_cost_") as tmp:
        tmp_path = Path(tmp)
        for lab in labels:
            r0, r1, c0, c1 = bbox_for_label(masks, lab, args.margin, h, w)
            crop_cp = cellprob[r0:r1, c0:c1]
            crop_m = (masks[r0:r1, c0:c1] == lab).astype(np.uint8)
            area = int(crop_m.sum())
            if area < args.min_cell_area:
                continue
            sal_u8 = cellprob_crop_to_saliency_u8(crop_cp, cell_mask=crop_m.astype(bool))
            fg = fg_scribble_coords(crop_m.astype(bool), erosion_pixels=args.fg_erosion_pixels)
            if not fg:
                continue

            for pair_label, conn_opt, crit_opt in pairs:
                name = f"{pair_label}_c{lab:05d}"
                t0 = time.perf_counter()
                ok = True
                err = ""
                dice_v = float("nan")
                placed = 0
                try:
                    sicle_lbl = _run_sicle_on_crop(
                        sal_u8,
                        fg,
                        tmp_path,
                        name,
                        sicle_bin,
                        n0=args.sicle_n0,
                        nf=args.sicle_nf,
                        alpha=args.sicle_alpha,
                        max_iters=args.sicle_max_iters,
                        irreg=args.sicle_irreg,
                        adhr=args.sicle_adhr,
                        conn_opt=conn_opt,
                        crit_opt=crit_opt,
                    )
                    obj = sicle_lbl == 1
                    place = obj & crop_m.astype(bool)
                    placed = int(place.sum())
                    dice_v = _dice(place, crop_m.astype(bool))
                except Exception as e:
                    ok = False
                    err = str(e).replace("\n", " ")[:500]
                elapsed = time.perf_counter() - t0
                rows.append(
                    {
                        "pair_label": pair_label,
                        "conn_opt": conn_opt,
                        "crit_opt": crit_opt,
                        "cell_id": lab,
                        "bbox_h": r1 - r0,
                        "bbox_w": c1 - c0,
                        "cell_area": area,
                        "ok": ok,
                        "seconds": round(elapsed, 4),
                        "dice_vs_cellpose": (f"{dice_v:.6f}" if ok and not math.isnan(dice_v) else ""),
                        "placed_pixels": placed if ok else "",
                        "error": err if not ok else "",
                    }
                )

    csv_path = out_dir / "sicle_path_cost_results.csv"
    if not rows:
        (out_dir / "summary.txt").write_text("No rows (check min-cell-area, fg seeds, max-cells).\n", encoding="utf-8")
        print("No successful test rows; wrote", out_dir / "summary.txt")
        return 1

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Aggregate mean Dice per pair (successful rows only)
    from collections import defaultdict

    dice_sum: dict[str, list[float]] = defaultdict(list)
    time_sum: dict[str, list[float]] = defaultdict(list)
    fail_ct: dict[str, int] = defaultdict(int)
    for r in rows:
        key = f"{r['pair_label']} ({r['conn_opt']},{r['crit_opt']})"
        if r["ok"]:
            if r["dice_vs_cellpose"] != "":
                dice_sum[key].append(float(r["dice_vs_cellpose"]))
            time_sum[key].append(float(r["seconds"]))
        else:
            fail_ct[key] += 1

    lines = [
        f"from_dir: {from_dir.resolve()}",
        f"cells_sampled: {len(labels)}",
        f"pair_mode: {args.pair_mode}",
        f"extra_pairs: {args.extra_pairs or '(none)'}",
        f"sicle_n0: {args.sicle_n0}",
        "",
        "Mean Dice (SICLE fg ∩ Cellpose mask vs Cellpose mask) — 1.0 = identical inside bbox",
        "----------------------------------------------------------------------",
    ]
    for key in sorted(set(dice_sum.keys()) | set(fail_ct.keys())):
        vals = dice_sum.get(key, [])
        mean_d = sum(vals) / len(vals) if vals else float("nan")
        dice_part = "mean_dice=nan" if math.isnan(mean_d) else f"mean_dice={mean_d:.4f}"
        lines.append(f"  {key}: {dice_part}  n_dice={len(vals)}  fails={fail_ct.get(key, 0)}")
    lines.append("")
    lines.append("Mean runtime per (pair, cell) / s")
    lines.append("------------------------------------")
    for key in sorted(time_sum.keys()):
        vals = time_sum[key]
        lines.append(f"  {key}: mean={sum(vals)/len(vals):.4f}  n={len(vals)}")
    lines.append("")
    lines.append(f"Full table: {csv_path.resolve()}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
