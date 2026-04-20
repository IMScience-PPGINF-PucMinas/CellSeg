#!/usr/bin/env python3
"""
Reproduce the Cellpose (CP-SAM) inference flow step-by-step **outside** the ``cellpose/`` package.

Writes an output folder with numbered artifacts, NumPy arrays, heatmaps, and ``manifest.txt``.

Steps (high level)
------------------
0. Load image, record shape
1. Preprocess like ``CellposeModel.eval`` (convert, optional diameter resize, normalize)
2. **Native** intermediate maps on **tile 0** (low resolution, hook geometry):
   ``encoder.neck`` [256,h,w] and ``out`` [nout·ps²,h,w] — these are what ``W2`` expects
3. Full-network outputs on the **whole** preprocessed stack: ``dP`` [2,Ly,Lx], ``cellprob`` [Ly,Lx]
4. **Dynamics** → instance masks (same as Cellpose)

Optional: merged **full-image** activation heatmaps (bilinear-upsampled per tile, like
``extract_activation_maps``) — slower on large WSI crops.

Usage::

    cd doutorado
    PYTHONPATH=./cellpose python reproduce_cellpose_pipeline.py image.tif -o ./cp_flow_out --gpu

"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLPOSE_DIR = ROOT / "cellpose"
if str(CELLPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(CELLPOSE_DIR))


def _print_step(n: int, title: str, detail: str = "") -> None:
    bar = "=" * 72
    print(f"\n{bar}\nSTEP {n}: {title}\n{bar}")
    if detail:
        print(textwrap.dedent(detail).strip() + "\n")


def _save_u8(path: Path, arr) -> None:
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(arr).save(path)
    else:
        imwrite(path, arr)


def run_pipeline(
    image_path: Path,
    out_dir: Path,
    gpu: bool,
    diameter: float | None,
    cellprob_threshold: float,
    flow_threshold: float,
    pretrained_model: str,
    full_merged_activations: bool,
    cellprob_sigmoid_heatmap: bool,
) -> None:
    import numpy as np

    from cellpose import dynamics, io, plot, transforms
    from cellpose.activation_maps import (
        activation_to_heatmap_rgb,
        capture_native_activation_first_tile,
        run_net_activation,
    )
    from cellpose.models import CellposeModel, normalize_default

    # Import heatmap helper from sibling script (same directory)
    sys.path.insert(0, str(ROOT))
    try:
        from cellprob_heatmap import cellprob_to_heatmap_u8
    except ImportError:
        cellprob_to_heatmap_u8 = None

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []

    def log(msg: str) -> None:
        manifest_lines.append(msg)
        print(msg)

    # --- STEP 0 ---
    _print_step(0, "Load image")
    img = io.imread(str(image_path))
    log(f"Loaded: {image_path.name}")
    log(f"  ndarray shape: {img.shape}, dtype: {img.dtype}")
    manifest_lines.append("")

    # --- STEP 1: preprocess (same as eval / extract_activation_maps) ---
    _print_step(
        1,
        "Preprocess (convert_image → optional diameter → normalize_img)",
        """
        Layout after convert: [nimg, Ly, Lx, 3]. This is what ``run_net`` consumes.
        """,
    )
    model = CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    x = transforms.convert_image(img, do_3D=False)
    if x.ndim < 4:
        x = x[np.newaxis, ...]
    if x.shape[0] != 1:
        raise SystemExit(
            "This pipeline currently supports a single 2D image (batch size 1)."
        )
    Ly_0, Lx_0 = x.shape[1], x.shape[2]
    log(f"x after convert_image: {x.shape} (nimg, Ly, Lx, ch)")

    image_scaling = None
    if diameter is not None:
        image_scaling = 30.0 / diameter
        x = transforms.resize_image(
            x,
            Ly=int(x.shape[1] * image_scaling),
            Lx=int(x.shape[2] * image_scaling),
        )
        log(
            f"x after diameter resize (30/diam={image_scaling:.4f}): {x.shape}; "
            f"original (Ly0,Lx0)=({Ly_0},{Lx_0})"
        )

    np_params = {**normalize_default, "normalize": True, "invert": False}
    if x.shape[0] > 1:
        np_params["norm3D"] = False
    x = transforms.normalize_img(x, **np_params)
    log(f"x after normalize_img: {x.shape}, float32 in ~[0,1]")

    # Preview (first channel grey, stretch)
    prev = x[0, :, :, 0]
    p1, p99 = np.percentile(prev, 1), np.percentile(prev, 99)
    if p99 > p1 + 1e-8:
        prev_u8 = np.clip((prev - p1) / (p99 - p1), 0, 1) * 255
    else:
        prev_u8 = np.zeros_like(prev)
    prev_u8 = prev_u8.astype(np.uint8)
    prev_rgb = np.stack([prev_u8, prev_u8, prev_u8], axis=-1)
    _save_u8(out_dir / "step01_preprocessed_grey_rgb.png", prev_rgb)
    log(f"  saved step01_preprocessed_grey_rgb.png")
    np.save(out_dir / "step01_preprocessed_x.npy", x.astype(np.float32))
    log(f"  saved step01_preprocessed_x.npy")

    # --- STEP 2: native intermediate (tile 0) ---
    _print_step(
        2,
        "Intermediate activations — native resolution, FIRST TILE ONLY",
        """
        These tensors match the hook outputs *before* ``conv_transpose2d(W2)``.
        Spatial size is ~ image_size/8 per side on the tile (e.g. 32×32 for 256×256 tile).

        Full-image merged activations (optional later) are bilinear-upsampled for viz only.
        """,
    )
    native_neck, _y_skip = capture_native_activation_first_tile(
        model.net, x, layer="neck", bsize=256, tile_index=0
    )
    native_out, y_tile0 = capture_native_activation_first_tile(
        model.net, x, layer="out", bsize=256, tile_index=0
    )
    log(f"native neck [C,h,w]: {native_neck.shape}  (C=256 for ViT-L neck)")
    log(f"native out   [C,h,w]: {native_out.shape}  (C=nout·ps² = {native_out.shape[0]})")
    log(f"full forward on tile0 [3,ly,lx]: {y_tile0.shape}  (Y-flow, X-flow, cellprob logit)")
    np.savez_compressed(
        out_dir / "step02_native_tile0.npz",
        neck_chw=native_neck,
        out_chw=native_out,
        y_three_ch_ly_lx=y_tile0,
    )
    log("  saved step02_native_tile0.npz")

    # --- STEP 3: full image network outputs ---
    _print_step(
        3,
        "Full image: ``_run_net`` → dP + cellprob",
        """
        ``dP``: [2, Ly, Lx] — axis 0 = Y flow, axis 1 = X flow (vector field for dynamics).
        ``cellprob``: [Ly, Lx] — logits for foreground (BCE-with-logits training; not sigmoid in net).
        """,
    )
    dP, cellprob, _styles = model._run_net(
        x,
        augment=False,
        batch_size=8,
        tile_overlap=0.1,
        bsize=256,
        do_3D=False,
    )
    log(f"dP after _run_net (before resample): {dP.shape}  [2, nimg, Ly, Lx]")
    log(f"cellprob after _run_net: {cellprob.shape}  [nimg, Ly, Lx]")

    # Match Cellpose eval: upsample flows/cellprob to original size when diameter was used
    if diameter is not None:
        dP = transforms.resize_image(
            dP.transpose(1, 2, 3, 0),
            Ly=Ly_0,
            Lx=Lx_0,
            no_channels=False,
        ).transpose(3, 0, 1, 2)
        cellprob = transforms.resize_image(
            cellprob, Ly=Ly_0, Lx=Lx_0, no_channels=True
        )
        log(f"after resample to ({Ly_0},{Lx_0}): dP {dP.shape}, cellprob {cellprob.shape}")

    dP_2d = dP[:, 0]
    cellprob_2d = cellprob[0]
    log(f"dP for dynamics [2, Ly, Lx]: {dP_2d.shape}")
    log(f"cellprob for dynamics [Ly, Lx]: {cellprob_2d.shape}")
    log(
        f"cellprob range (logits): [{float(cellprob_2d.min()):.4f}, {float(cellprob_2d.max()):.4f}]"
    )

    np.savez_compressed(
        out_dir / "step03_dP_cellprob.npz",
        dP_full=dP.astype(np.float32),
        cellprob_full=cellprob.astype(np.float32),
        dP_slice01=dP_2d.astype(np.float32),
        cellprob_slice0=cellprob_2d.astype(np.float32),
    )
    log("  saved step03_dP_cellprob.npz (includes dP_slice01 / cellprob_slice0 for dynamics)")

    flow_rgb = plot.dx_to_circ(dP_2d)
    _save_u8(out_dir / "step03a_dP_flow_dx_to_circ.png", flow_rgb)
    log("  saved step03a_dP_flow_dx_to_circ.png (HSV flow visualization)")

    if cellprob_to_heatmap_u8 is not None:
        heat_cp = cellprob_to_heatmap_u8(
            cellprob_2d,
            use_sigmoid=cellprob_sigmoid_heatmap,
            colormap="turbo",
        )
        _save_u8(out_dir / "step03b_cellprob_heatmap.png", heat_cp)
        log(
            f"  saved step03b_cellprob_heatmap.png (sigmoid={cellprob_sigmoid_heatmap})"
        )
    else:
        log("  (skipped cellprob heatmap: could not import cellprob_heatmap)")

    # --- STEP 4: dynamics → masks ---
    _print_step(
        4,
        "Dynamics → instance masks",
        """
        ``resize_and_compute_masks``: threshold cellprob → follow_flows(dP) → label pixels.
        Needs BOTH dP and cellprob — cellprob alone is not enough for instances.
        """,
    )
    niter_scale = 1.0 if image_scaling is None else image_scaling
    niter = max(1, int(200 / niter_scale))

    masks = dynamics.resize_and_compute_masks(
        dP_2d,
        cellprob_2d,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        device=model.device,
    )
    masks = np.squeeze(masks)
    log(f"masks shape: {masks.shape}, max label: {int(masks.max())}")
    _save_u8(out_dir / "step04_masks_rgb.png", plot.mask_rgb(masks))
    np.save(out_dir / "step04_masks_uint16.npy", masks.astype(np.uint16))
    log("  saved step04_masks_rgb.png, step04_masks_uint16.npy")

    # --- STEP 5 optional: full merged activation heatmaps ---
    if full_merged_activations:
        _print_step(
            5,
            "Optional: full-image merged activation heatmaps (slow)",
            """
            Same merge as ``extract_activation_maps``: per-tile hooks, bilinear upsample,
            tile average. For visualization — not valid input to W2 at full resolution.
            """,
        )
        for layer in ("neck", "out"):
            act_vol = run_net_activation(
                model.net,
                x,
                layer=layer,
                batch_size=8,
                augment=False,
                tile_overlap=0.1,
                bsize=256,
            )
            av = act_vol[0] if act_vol.ndim == 4 else act_vol
            heat = activation_to_heatmap_rgb(av[np.newaxis, ...], reduce_mode="l2")
            if heat.ndim == 4:
                heat = heat[0]
            _save_u8(
                out_dir / f"step05_merged_activation_{layer}_heatmap.png",
                heat,
            )
            np.save(
                out_dir / f"step05_merged_activation_{layer}.npy",
                av.astype(np.float32),
            )
            log(f"  saved merged {layer}: shape {av.shape}")

    # --- Manifest ---
    manifest_lines.insert(
        0,
        textwrap.dedent(
            f"""
            Cellpose reproduction pipeline
            ==============================
            Input: {image_path}
            Output directory: {out_dir}
            diameter: {diameter}
            cellprob_threshold: {cellprob_threshold}, flow_threshold: {flow_threshold}
            full_merged_activations: {full_merged_activations}

            File guide
            ----------
            step01_preprocessed_x.npy     — input to the network [nimg,Ly,Lx,3]
            step02_native_tile0.npz       — neck/out native + y on tile 0 only
            step03_dP_cellprob.npz        — dP_full, cellprob_full, dP_slice01, cellprob_slice0
            step03a_dP_flow_dx_to_circ.png — flow field color wheel
            step03b_cellprob_heatmap.png — cellprob heatmap (if available)
            step04_masks_uint16.npy       — instance labels

            Pipeline order
            --------------
            image → preprocess → [encoder → neck → out → W2] → (dP, cellprob) → dynamics → masks
            Hooks on neck/out capture tensors *before* W2; dP/cellprob are *after* W2 on full image.
            """
        ).strip(),
    )
    (out_dir / "manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"\nWrote manifest: {out_dir / 'manifest.txt'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step-by-step Cellpose inference artifact dump (outside cellpose package)."
    )
    parser.add_argument("image", type=str)
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="./reproduce_cellpose_out",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--pretrained-model", default="cpsam")
    parser.add_argument(
        "--full-merged-activations",
        action="store_true",
        help="Also run merged full-image activation heatmaps for neck+out (slower).",
    )
    parser.add_argument(
        "--cellprob-heatmap-sigmoid",
        action="store_true",
        help="Use sigmoid on cellprob before heatmap coloring.",
    )
    args = parser.parse_args()

    run_pipeline(
        Path(args.image),
        Path(args.out_dir),
        gpu=args.gpu,
        diameter=args.diameter,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        pretrained_model=args.pretrained_model,
        full_merged_activations=args.full_merged_activations,
        cellprob_sigmoid_heatmap=args.cellprob_heatmap_sigmoid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
