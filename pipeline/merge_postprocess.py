#!/usr/bin/env python3
"""
Pós-processamento de ``merged_percell_sicle_masks_int32.npy`` frente a Cellpose.

* **clip** — Alternativa A: manter SICLE só dentro de ``dilate(cellpose==L)``.
* **components** — Alternativa C: por label, manter só a componente 8-vizinhada
  de ``merged==L`` com maior ``|componente ∩ cellpose_L|`` (empate: maior área).

Exemplo::

    python merge_postprocess.py clip \\
        --cellpose ./cp_flow_out/step04_masks_uint16.npy \\
        --merged ./out/raw_no_and/merged_percell_sicle_masks_int32.npy \\
        --out ./out/alt_a/merged_percell_sicle_masks_int32.npy --dilate 2

    python merge_postprocess.py components \\
        --cellpose ./cp_flow_out/step04_masks_uint16.npy \\
        --merged ./out/raw_no_and/merged_percell_sicle_masks_int32.npy \\
        --out ./out/alt_c/merged_percell_sicle_masks_int32.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path


def clip_merged_to_cellpose_dilated(
    merged: "np.ndarray",
    cellpose: "np.ndarray",
    dilate_pixels: int = 0,
) -> "np.ndarray":
    """Por label L: foreground = (merged==L) ∧ dilatação^N(cellpose==L)."""
    import numpy as np
    from scipy.ndimage import binary_dilation

    struct = np.ones((3, 3), dtype=bool)
    fixed = np.zeros_like(merged, dtype=np.int32)
    labels = np.unique(merged)
    labels = labels[labels > 0]
    for lab in labels:
        cp = cellpose == lab
        if not np.any(cp):
            continue
        allow = cp.astype(bool).copy()
        for _ in range(max(0, int(dilate_pixels))):
            allow = binary_dilation(allow, structure=struct)
        fg = (merged == lab) & allow
        fixed[fg] = int(lab)
    return fixed


def filter_merged_largest_overlapping_component(
    merged: "np.ndarray",
    cellpose: "np.ndarray",
) -> "np.ndarray":
    """Por label: uma única componente 8-conexa de maior sobreposição com Cellpose L."""
    import numpy as np
    from scipy import ndimage

    struct = np.ones((3, 3), dtype=int)
    out = np.zeros_like(merged, dtype=np.int32)
    labs = sorted((set(np.unique(merged).tolist()) | set(np.unique(cellpose).tolist())) - {0})
    for L in labs:
        m = merged == L
        if not np.any(m):
            continue
        lbl, n = ndimage.label(m, structure=struct)
        cp = cellpose == L
        best_k = 0
        best_key = (-1, -1)
        for k in range(1, n + 1):
            comp = lbl == k
            inter = int(np.sum(comp & cp))
            sz = int(np.sum(comp))
            key = (inter, sz)
            if key > best_key:
                best_key = key
                best_k = k
        if best_k > 0:
            out[lbl == best_k] = int(L)
    return out


def _load_pair(cellpose_path: Path, merged_path: Path) -> tuple:
    import numpy as np

    cp = np.load(cellpose_path).astype(np.int32, copy=False)
    mg = np.load(merged_path).astype(np.int32, copy=False)
    if mg.shape != cp.shape:
        raise SystemExit(f"shape mismatch merged {mg.shape} vs cellpose {cp.shape}")
    return cp, mg


def main() -> int:
    import numpy as np

    p = argparse.ArgumentParser(description="Pós-processamento merge percell vs Cellpose")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_io(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--cellpose", type=str, required=True, help="step04_masks_uint16.npy")
        sp.add_argument("--merged", type=str, required=True, help="merged_percell_sicle_masks_int32.npy")
        sp.add_argument("--out", type=str, required=True, help="NPY de saída (mesmo dtype int32)")

    pc = sub.add_parser("clip", help="Recorte por dilatação da instância Cellpose (alternativa A)")
    add_io(pc)
    pc.add_argument("--dilate", type=int, default=0, help="Iterações 3x3 de dilatação (default 0)")

    ps = sub.add_parser("components", help="Uma componente por label (alternativa C)")
    add_io(ps)

    args = p.parse_args()
    cp_path = Path(args.cellpose)
    merged_path = Path(args.merged)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cellpose, merged = _load_pair(cp_path, merged_path)

    if args.cmd == "clip":
        out = clip_merged_to_cellpose_dilated(merged, cellpose, dilate_pixels=args.dilate)
    else:
        out = filter_merged_largest_overlapping_component(merged, cellpose)

    np.save(out_path, out.astype(np.int32, copy=False))
    print(f"Wrote {out_path} ({args.cmd})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
