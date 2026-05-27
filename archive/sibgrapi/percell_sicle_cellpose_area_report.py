#!/usr/bin/env python3
"""
Replicável: cruza ``percell_sicle_log.txt`` com ``step04_masks_uint16.npy`` e
``merged_percell_sicle_masks_int32.npy``, gera CSV com áreas e razões
(útil para diagnosticar expansão/encolhimento com ``--disable-and-merge``).

Quando ``log_merged_agree`` é ``no``, o ``placed_pixels_log`` (por célula) pode
diferir de ``merged_area_actual`` porque **vários labels competem pelo mesmo
pixel** no merge por bbox (sobrescrita).

Exemplo::

    cd new_pipeline
    python percell_sicle_cellpose_area_report.py \\
        --from-dir ./cp_flow_out \\
        --out-dir ./percell_sicle_out_gradvmax_no_and \\
        -o ./percell_sicle_out_gradvmax_no_and/sicle_vs_cellpose_areas.csv

Opcional: escrever máscara merge **recortada** por dilatação da instância Cellpose
(mitiga vazamento para fora da célula mantendo uma faixa de tolerância na borda)::

    python percell_sicle_cellpose_area_report.py \\
        --from-dir ./cp_flow_out \\
        --out-dir ./percell_sicle_out_gradvmax_no_and \\
        -o ./percell_sicle_out_gradvmax_no_and/sicle_vs_cellpose_areas.csv \\
        --write-merged-clipped-npy merged_percell_sicle_clipped_int32.npy \\
        --clip-dilate-pixels 2
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from merge_postprocess import clip_merged_to_cellpose_dilated


def load_masks(from_dir: Path) -> "tuple":
    import numpy as np

    cp = np.load(from_dir / "step04_masks_uint16.npy").astype(np.int32, copy=False)
    return cp


def parse_log(log_path: Path) -> dict[int, dict]:
    """Parse percell_sicle_log.txt lines into label -> metadata."""
    rx_bbox = re.compile(
        r"^label\s+(\d+):\s+bbox=\((\d+),(\d+),(\d+),(\d+)\)\s+placed_pixels=(\d+)(?:\s+merge=.*)?\s*$"
    )
    rx_min_area = re.compile(
        r"^label\s+(\d+):\s+area=(\d+)\s+<\s+min_cell_area,\s+kept Cellpose mask\s*$"
    )
    rx_no_fg = re.compile(r"^label\s+(\d+):\s+no fg seeds,\s+kept Cellpose mask\s*$")
    rx_sicle_fail = re.compile(r"^label\s+(\d+):\s+SICLE failed\s+\((.+)\),\s+kept Cellpose mask\s*$")

    out: dict[int, dict] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = rx_bbox.match(line)
        if m:
            lab = int(m.group(1))
            r0, r1, c0, c1 = map(int, m.groups()[1:5])
            placed = int(m.group(6))
            out[lab] = {
                "status": "sicle_pasted",
                "bbox": (r0, r1, c0, c1),
                "placed_pixels_log": placed,
                "min_area_fallback": None,
            }
            continue
        m = rx_min_area.match(line)
        if m:
            lab = int(m.group(1))
            area = int(m.group(2))
            out[lab] = {
                "status": "min_cell_area_fallback",
                "bbox": None,
                # ``area`` is Cellpose pixels in bbox (pipeline keeps full Cellpose mask).
                "placed_pixels_log": area,
                "min_area_fallback": area,
            }
            continue
        m = rx_no_fg.match(line)
        if m:
            lab = int(m.group(1))
            out[lab] = {
                "status": "no_fg_seeds_fallback",
                "bbox": None,
                "placed_pixels_log": None,
                "min_area_fallback": None,
            }
            continue
        m = rx_sicle_fail.match(line)
        if m:
            lab = int(m.group(1))
            err = m.group(2).strip().replace("\n", " ")[:500]
            out[lab] = {
                "status": "sicle_failed_fallback",
                "bbox": None,
                "placed_pixels_log": None,
                "min_area_fallback": None,
                "error": err,
            }
            continue
    return out


def cellpose_area_in_bbox(masks: "np.ndarray", lab: int, bbox: tuple[int, int, int, int]) -> int:
    import numpy as np

    r0, r1, c0, c1 = bbox
    crop = masks[r0:r1, c0:c1]
    return int(np.sum(crop == lab))


def main() -> int:
    import numpy as np

    p = argparse.ArgumentParser(
        description="CSV: Cellpose vs per-cell SICLE areas from log + npy masks."
    )
    p.add_argument(
        "--from-dir",
        type=str,
        required=True,
        help="Pasta com step04_masks_uint16.npy (ex.: ./cp_flow_out)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Pasta do run percell (merged_percell_sicle_masks_int32.npy + percell_sicle_log.txt)",
    )
    p.add_argument(
        "-o",
        "--csv-out",
        type=str,
        required=True,
        help="Caminho do CSV de saída",
    )
    p.add_argument(
        "--masks-merged",
        type=str,
        default=None,
        help="Override: merged int32 npy (default: <out-dir>/merged_percell_sicle_masks_int32.npy)",
    )
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help="Override: percell_sicle_log.txt (default: <out-dir>/percell_sicle_log.txt)",
    )
    p.add_argument(
        "--write-merged-clipped-npy",
        type=str,
        default=None,
        metavar="REL_OR_PATH",
        help=(
            "Se definido, grava máscara merge recortada por dilatação da instância Cellpose. "
            "Caminho relativo ao --out-dir ou absoluto."
        ),
    )
    p.add_argument(
        "--clip-dilate-pixels",
        type=int,
        default=0,
        help="Iterações 3x3 de dilatação na máscara Cellpose antes do AND com SICLE (default: 0)",
    )
    args = p.parse_args()

    from_dir = Path(args.from_dir)
    out_dir = Path(args.out_dir)
    log_path = Path(args.log) if args.log else out_dir / "percell_sicle_log.txt"
    merged_path = (
        Path(args.masks_merged)
        if args.masks_merged
        else out_dir / "merged_percell_sicle_masks_int32.npy"
    )

    if not log_path.is_file():
        raise SystemExit(f"missing log: {log_path}")
    if not merged_path.is_file():
        raise SystemExit(f"missing merged masks: {merged_path}")

    cellpose = load_masks(from_dir)
    merged = np.load(merged_path).astype(np.int32, copy=False)
    if merged.shape != cellpose.shape:
        raise SystemExit(f"shape mismatch merged {merged.shape} vs cellpose {cellpose.shape}")

    parsed = parse_log(log_path)
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    labels_sorted = sorted(parsed.keys())
    fieldnames = [
        "label",
        "status",
        "bbox_r0",
        "bbox_r1",
        "bbox_c0",
        "bbox_c1",
        "cellpose_area_full",
        "cellpose_area_in_bbox",
        "bbox_area",
        "placed_pixels_log",
        "merged_area_actual",
        "ratio_placed_vs_cp_bbox",
        "ratio_merged_vs_cp_full",
        "ratio_placed_vs_cp_full",
        "log_merged_agree",
        "notes",
    ]

    rows_out: list[dict] = []
    for lab in labels_sorted:
        meta = parsed[lab]
        status = meta["status"]
        cp_full = int(np.sum(cellpose == lab))
        bbox = meta.get("bbox")
        if bbox is not None:
            r0, r1, c0, c1 = bbox
            cp_bbox = cellpose_area_in_bbox(cellpose, lab, bbox)
            bbox_area = (r1 - r0) * (c1 - c0)
            br0, br1, bc0, bc1 = r0, r1, c0, c1
        else:
            cp_bbox = ""
            bbox_area = ""
            br0 = br1 = bc0 = bc1 = ""

        placed_log = meta.get("placed_pixels_log")
        merged_area = int(np.sum(merged == lab))

        def ratio(a: float, b: float) -> str:
            if b == 0:
                return ""
            return f"{a / b:.6f}"

        if placed_log is None:
            r_pb = r_pf = r_mf = ""
            agree = ""
        else:
            r_pb = ratio(float(placed_log), float(cp_bbox)) if bbox else ""
            r_pf = ratio(float(placed_log), float(cp_full)) if cp_full else ""
            r_mf = ratio(float(merged_area), float(cp_full)) if cp_full else ""
            agree = "yes" if placed_log == merged_area else "no"

        notes_parts: list[str] = []
        if status == "sicle_failed_fallback" and meta.get("error"):
            notes_parts.append(meta["error"][:200])
        if placed_log is not None and placed_log != merged_area:
            notes_parts.append(f"placed_log!=merged_area ({placed_log}!={merged_area})")

        row = {
            "label": lab,
            "status": status,
            "bbox_r0": br0,
            "bbox_r1": br1,
            "bbox_c0": bc0,
            "bbox_c1": bc1,
            "cellpose_area_full": cp_full,
            "cellpose_area_in_bbox": cp_bbox,
            "bbox_area": bbox_area,
            "placed_pixels_log": placed_log if placed_log is not None else "",
            "merged_area_actual": merged_area,
            "ratio_placed_vs_cp_bbox": r_pb,
            "ratio_merged_vs_cp_full": r_mf,
            "ratio_placed_vs_cp_full": r_pf,
            "log_merged_agree": agree,
            "notes": "; ".join(notes_parts),
        }
        rows_out.append(row)

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Wrote {csv_out} ({len(rows_out)} rows)")

    if args.write_merged_clipped_npy:
        import numpy as np

        rel = Path(args.write_merged_clipped_npy)
        clip_path = rel if rel.is_absolute() else out_dir / rel
        clipped = clip_merged_to_cellpose_dilated(
            merged, cellpose, dilate_pixels=args.clip_dilate_pixels
        )
        np.save(clip_path, clipped)
        print(f"Wrote clipped merged: {clip_path} (dilate={args.clip_dilate_pixels})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
