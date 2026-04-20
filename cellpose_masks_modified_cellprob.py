#!/usr/bin/env python3
"""
Remix Cellpose instance masks by **keeping ``dP`` (flows) fixed** and **editing ``cellprob``**
before ``resize_and_compute_masks`` — option (1): modified probability → same CP dynamics engine.

Typical inputs are arrays saved by ``reproduce_cellpose_pipeline.py`` (``step03_dP_cellprob.npz``).

Examples::

    # Stronger foreground: sigmoid then run dynamics (threshold on logits still applies)
    PYTHONPATH=./cellpose python cellpose_masks_modified_cellprob.py \\
        --npz cp_flow_out/step03_dP_cellprob.npz -o remix_out --sigmoid --cellprob-threshold 0.25

    # Scale logits, then segment
    PYTHONPATH=./cellpose python cellpose_masks_modified_cellprob.py \\
        --npz cp_flow_out/step03_dP_cellprob.npz -o remix_out --scale 1.5

    # Multiply by an external weight map (same H×W as cellprob)
    PYTHONPATH=./cellpose python cellpose_masks_modified_cellprob.py \\
        --npz cp_flow_out/step03_dP_cellprob.npz -o remix_out --multiply weights.npy

Compare to baseline masks: ``compare_segmentation_masks_diff.py --mask-a cp_flow_out/step04_masks_uint16.npy --mask-b remix_out/remix_arrays.npz``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLPOSE_DIR = ROOT / "cellpose"
if str(CELLPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(CELLPOSE_DIR))


def load_dP_cellprob(npz_path: Path):
    import numpy as np

    z = np.load(npz_path)
    names = set(z.files)
    if "dP_slice01" in names and "cellprob_slice0" in names:
        dP = z["dP_slice01"]
        cellprob = z["cellprob_slice0"]
    elif "dP" in names and "cellprob" in names:
        dP = z["dP"]
        cellprob = z["cellprob"]
        if dP.ndim == 4 and dP.shape[1] == 1:
            dP = dP[:, 0]
        if cellprob.ndim == 3 and cellprob.shape[0] == 1:
            cellprob = cellprob[0]
    else:
        raise SystemExit(
            f"{npz_path}: need dP_slice01+cellprob_slice0 or dP+cellprob arrays"
        )
    return (
        np.asarray(dP, dtype=np.float32),
        np.asarray(cellprob, dtype=np.float32),
    )


def modify_cellprob(
    cellprob: "np.ndarray",
    *,
    sigmoid: bool,
    scale: float,
    offset: float,
    multiply_path: Path | None,
) -> "np.ndarray":
    import numpy as np

    m = cellprob.astype(np.float64)
    if sigmoid:
        m = 1.0 / (1.0 + np.exp(-np.clip(m, -50, 50)))
    m = scale * m + offset
    if multiply_path is not None:
        w = np.load(multiply_path)
        w = np.asarray(w, dtype=np.float64)
        if w.shape != m.shape:
            raise SystemExit(f"--multiply shape {w.shape} != cellprob {m.shape}")
        m = m * w
    return m.astype(np.float32)


def main() -> int:
    import numpy as np

    from cellpose import dynamics, plot
    from cellpose.core import assign_device

    p = argparse.ArgumentParser(
        description="Cellpose masks with edited cellprob (dP unchanged)."
    )
    p.add_argument(
        "--npz",
        type=str,
        required=True,
        help="step03_dP_cellprob.npz (or compatible) from reproduce_cellpose_pipeline.py",
    )
    p.add_argument("-o", "--out-dir", type=str, default="./cellpose_remix_out")
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--sigmoid",
        action="store_true",
        help="Apply sigmoid to cellprob before scale/offset/multiply",
    )
    p.add_argument("--scale", type=float, default=1.0, help="m = scale*m + offset")
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument(
        "--multiply",
        type=str,
        default=None,
        help="Optional .npy same shape as cellprob, multiplied after scale/offset",
    )
    p.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Passed to dynamics (pixels with modified cellprob > this join flows)",
    )
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--niter", type=int, default=200)
    args = p.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        print(f"not found: {npz_path}", file=sys.stderr)
        return 1

    dP, cellprob = load_dP_cellprob(npz_path)
    if dP.shape[0] != 2:
        print(f"expected dP shape [2,Ly,Lx], got {dP.shape}", file=sys.stderr)
        return 1

    mult = Path(args.multiply) if args.multiply else None
    cellprob_mod = modify_cellprob(
        cellprob,
        sigmoid=args.sigmoid,
        scale=args.scale,
        offset=args.offset,
        multiply_path=mult,
    )

    device, _gpu = assign_device(gpu=args.gpu)
    masks = dynamics.resize_and_compute_masks(
        dP,
        cellprob_mod,
        niter=args.niter,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        device=device,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "remix_arrays.npz",
        dP=dP,
        cellprob_original=cellprob,
        cellprob_modified=cellprob_mod,
        masks=masks.astype(np.uint16),
    )
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(plot.mask_rgb(masks)).save(out / "masks_remixed_rgb.png")
    else:
        imwrite(out / "masks_remixed_rgb.png", plot.mask_rgb(masks))

    meta = out / "remix_log.txt"
    meta.write_text(
        "\n".join(
            [
                f"source_npz: {npz_path}",
                f"dP shape: {dP.shape}",
                f"cellprob original range: [{cellprob.min():.5f}, {cellprob.max():.5f}]",
                f"cellprob modified range: [{cellprob_mod.min():.5f}, {cellprob_mod.max():.5f}]",
                f"sigmoid={args.sigmoid} scale={args.scale} offset={args.offset}",
                f"multiply={args.multiply}",
                f"cellprob_threshold={args.cellprob_threshold} flow_threshold={args.flow_threshold}",
                f"niter={args.niter}",
                f"mask max label: {int(masks.max())}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out / 'masks_remixed_rgb.png'}, {out / 'remix_arrays.npz'}")
    print(meta.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
