#!/usr/bin/env python3
# ruff: noqa: E501
"""
================================================================================
HOW CELLPOSE TURNS A NETWORK OUTPUT INTO INSTANCE SEGMENTATION (CP-SAM / v4)
================================================================================

This script lives **outside** the ``cellpose/`` tree. It only adds ``cellpose`` to
``sys.path`` and imports it like your other projects.

Important distinction — **activation maps ≠ segmentation input**
----------------------------------------------------------------
The maps you extracted by hooking ``encoder.neck`` or ``out`` (before upsampling)
are **intermediate features**. Cellpose does **not** threshold or cluster those
to build masks **unless** you apply the **remaining layers** (see below).

**Completing the forward from an intermediate map**
---------------------------------------------------
For CP-SAM, the readout after your hook is fixed:

  * From **neck** (256 ch, low resolution): ``neck → Conv2d out → conv_transpose2d(W2)`` → 3 channels.
  * From **out** (``nout·ps²`` ch, low resolution): ``conv_transpose2d(W2)`` → 3 channels.

Only tensors at this **native** resolution (the same shape the hook sees) can be
used. The **full-image activation .npy** from ``extract_activation_map.py`` that was
**bilinear-upsampled** to full image size **cannot** be fed into ``W2`` correctly.

This script can:

  * Load a **native** map (e.g. from ``--export-native`` here) and reconstruct
    flows + cellprob, then run dynamics → masks.
  * Or load an array that is **already** the 3-channel output ``[3, Ly, Lx]`` and
    run dynamics only.

**Segmentation uses the final tensor** after ``conv_transpose2d``: Y-flow, X-flow,
cellprob logit → ``follow_flows`` → instance labels.

References: ``cellpose/vit_sam.py``, ``cellpose/dynamics.py``.

================================================================================
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLPOSE_DIR = ROOT / "cellpose"
if str(CELLPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(CELLPOSE_DIR))


def print_explanation() -> None:
    """Print the module docstring (the narrative above)."""
    print(inspect.getdoc(sys.modules[__name__]) or "")


def _load_array_file(path: Path):
    import numpy as np

    if path.suffix.lower() in (".npz",):
        z = np.load(path)
        keys = list(z.keys())
        if len(keys) == 1:
            return z[keys[0]], None
        if "neck" in keys:
            return z["neck"], "neck"
        if "out" in keys:
            return z["out"], "out"
        raise ValueError(f"NPZ {path} must contain 'neck', 'out', or a single array.")
    return np.load(path), None


def _np_to_chw3(arr):
    """Return [3, Ly, Lx] float32 or None if not 3-channel final output."""
    import numpy as np

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        return None
    if a.shape[0] == 3:
        return a
    if a.shape[-1] == 3:
        return np.transpose(a, (2, 0, 1))
    return None


def run_demo(
    image_path: Path,
    out_dir: Path,
    gpu: bool,
    diameter: float | None,
    cellprob_threshold: float,
    flow_threshold: float,
) -> None:
    """Run Cellpose once, save flows + recomputed masks to illustrate the pipeline."""
    import numpy as np

    from cellpose import dynamics, io, plot
    from cellpose.models import CellposeModel

    out_dir.mkdir(parents=True, exist_ok=True)
    img = io.imread(str(image_path))
    model = CellposeModel(gpu=gpu, pretrained_model="cpsam")

    masks, flows, _styles = model.eval(
        img,
        diameter=diameter,
        compute_masks=True,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )

    _circ, dP, cellprob = flows[0], flows[1], flows[2]
    mask = masks.squeeze()

    np.savez_compressed(
        out_dir / "pipeline_arrays.npz",
        dP=dP,
        cellprob=cellprob,
        mask=mask,
        note="dP shape [2,Ly,Lx] = Y-flow, X-flow; cellprob [Ly,Lx]; mask labelled uint",
    )

    circ = _circ
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(circ).save(out_dir / "flow_dx_to_circ.png")
        Image.fromarray((cellprob / (cellprob.max() + 1e-12) * 255).astype(np.uint8)).save(
            out_dir / "cellprob_gray.png"
        )
        Image.fromarray(plot.mask_rgb(mask)).save(out_dir / "masks_rgb.png")
    else:
        imwrite(out_dir / "flow_dx_to_circ.png", circ)
        imwrite(
            out_dir / "cellprob_gray.png",
            (cellprob / (cellprob.max() + 1e-12) * 255).astype(np.uint8),
        )
        imwrite(out_dir / "masks_rgb.png", plot.mask_rgb(mask))

    masks2 = dynamics.resize_and_compute_masks(
        dP,
        cellprob,
        niter=200,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        device=model.device,
    )
    same = np.array_equal(mask, masks2)
    with open(out_dir / "pipeline_log.txt", "w", encoding="utf-8") as f:
        f.write("Cellpose segmentation pipeline demo\n")
        f.write(f"dP shape: {dP.shape}  (axis 0 = Y flow, axis 1 = X flow)\n")
        f.write(f"cellprob shape: {cellprob.shape}\n")
        f.write(f"mask shape: {mask.shape}, max label: {mask.max()}\n")
        f.write(
            f"masks from eval == masks from dynamics.resize_and_compute_masks: {same}\n"
        )
        if not same:
            f.write(
                "(Small differences can happen if eval used different niter/post steps; "
                "usually identical.)\n"
            )

    print(f"Wrote outputs under {out_dir}")
    print(f"  dP {dP.shape}, cellprob {cellprob.shape}, mask max label {mask.max()}")
    print(f"  eval vs dynamics-only masks identical: {same}")


def run_export_native(
    image_path: Path,
    out_npz: Path,
    layer: str,
    gpu: bool,
    diameter: float | None,
) -> None:
    """Save native [C,h,w] activation + full [3,ly,lx] y for the first tile."""
    import numpy as np

    from cellpose import io
    from cellpose.activation_maps import capture_native_activation_first_tile
    from cellpose.models import CellposeModel, normalize_default

    img = io.imread(str(image_path))
    model = CellposeModel(gpu=gpu, pretrained_model="cpsam")
    from cellpose import transforms

    x = transforms.convert_image(img, do_3D=False)
    if x.ndim < 4:
        x = x[np.newaxis, ...]
    if diameter is not None:
        sc = 30.0 / diameter
        x = transforms.resize_image(x, Ly=int(x.shape[1] * sc), Lx=int(x.shape[2] * sc))
    np_params = {**normalize_default, "normalize": True, "invert": False}
    x = transforms.normalize_img(x, **np_params)

    native, y = capture_native_activation_first_tile(
        model.net, x, layer=layer, bsize=256, tile_index=0
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    key = "neck" if layer == "neck" else "out"
    np.savez_compressed(
        out_npz,
        **{key: native},
        y_three_channel=y,
        layer=layer,
        note="native: hook output [C,h,w]; y_three_channel: full tile output [3,ly,lx]",
    )
    print(f"Wrote {out_npz}")
    print(f"  native {native.shape} ({key}), y {y.shape}")


def run_from_intermediate_file(
    activation_path: Path,
    from_layer: str,
    out_dir: Path,
    gpu: bool,
    cellprob_threshold: float,
    flow_threshold: float,
    pretrained_model: str,
) -> None:
    """Load native activation OR 3-channel output, then masks."""
    import numpy as np

    from cellpose import dynamics, plot
    from cellpose.activation_maps import intermediate_to_final_output
    from cellpose.models import CellposeModel

    out_dir.mkdir(parents=True, exist_ok=True)
    model = CellposeModel(gpu=gpu, pretrained_model=pretrained_model)

    arr, layer_hint = _load_array_file(activation_path)
    if layer_hint:
        from_layer = layer_hint

    y3 = _np_to_chw3(arr)
    if y3 is not None:
        print("Detected 3-channel final output [3, Ly, Lx] (or HWC); running dynamics only.")
    else:
        if from_layer not in ("neck", "out", "last_conv"):
            raise SystemExit(
                "For multi-channel intermediate maps, pass --from-layer neck or --from-layer out"
            )
        print(
            f"Completing readout from {from_layer} activation {arr.shape} → 3 channels..."
        )
        if arr.ndim == 3 and arr.shape[0] not in (256, 192) and arr.shape[-1] in (256, 192):
            arr = np.transpose(arr, (2, 0, 1))
        y3 = intermediate_to_final_output(model.net, arr, from_layer=from_layer)

    dP, cellprob = y3[:2], y3[2]
    masks = dynamics.resize_and_compute_masks(
        dP,
        cellprob,
        niter=200,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        device=model.device,
    )

    np.savez_compressed(
        out_dir / "reconstructed_pipeline.npz",
        dP=dP,
        cellprob=cellprob,
        mask=masks,
        y_three_channel=y3,
    )
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(plot.mask_rgb(masks)).save(out_dir / "masks_from_activation.png")
    else:
        imwrite(out_dir / "masks_from_activation.png", plot.mask_rgb(masks))

    print(f"Wrote {out_dir / 'masks_from_activation.png'} and reconstructed_pipeline.npz")
    print(f"  dP {dP.shape}, cellprob {cellprob.shape}, max label {masks.max()}")


def run_native_tile_pipeline(
    image_path: Path,
    out_dir: Path,
    layer: str,
    gpu: bool,
    diameter: float | None,
    cellprob_threshold: float,
    flow_threshold: float,
) -> None:
    """First tile: native activation → complete to y (check) → dynamics → masks."""
    import numpy as np

    from cellpose import dynamics, io, plot
    from cellpose.activation_maps import (
        capture_native_activation_first_tile,
        intermediate_to_final_output,
    )
    from cellpose.models import CellposeModel, normalize_default

    from cellpose import transforms

    out_dir.mkdir(parents=True, exist_ok=True)
    img = io.imread(str(image_path))
    model = CellposeModel(gpu=gpu, pretrained_model="cpsam")

    x = transforms.convert_image(img, do_3D=False)
    if x.ndim < 4:
        x = x[np.newaxis, ...]
    if diameter is not None:
        sc = 30.0 / diameter
        x = transforms.resize_image(x, Ly=int(x.shape[1] * sc), Lx=int(x.shape[2] * sc))
    np_params = {**normalize_default, "normalize": True, "invert": False}
    x = transforms.normalize_img(x, **np_params)

    native, y_direct = capture_native_activation_first_tile(
        model.net, x, layer=layer, bsize=256, tile_index=0
    )
    y_from_act = intermediate_to_final_output(model.net, native, from_layer=layer)
    err = float(np.abs(y_direct - y_from_act).max())
    print(f"Native activation {native.shape}; max |y_direct - reconstruct(y)| = {err:.6f}")

    dP, cellprob = y_direct[:2], y_direct[2]
    masks = dynamics.resize_and_compute_masks(
        dP,
        cellprob,
        niter=200,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        device=model.device,
    )

    np.savez_compressed(
        out_dir / "native_tile_pipeline.npz",
        native_activation=native,
        y_three_channel=y_direct,
        dP=dP,
        cellprob=cellprob,
        mask=masks,
    )
    try:
        from imageio import imwrite
    except ImportError:
        from PIL import Image

        Image.fromarray(plot.mask_rgb(masks)).save(out_dir / "masks_native_tile.png")
    else:
        imwrite(out_dir / "masks_native_tile.png", plot.mask_rgb(masks))

    print(f"Wrote {out_dir / 'masks_native_tile.png'} (first tile only)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explain Cellpose pipeline; optional demos: eval, native activation → segmentation."
    )
    parser.add_argument(
        "--explain-only",
        action="store_true",
        help="Print the pipeline explanation and exit.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image for --demo-eval or --native-tile-pipeline or --export-native.",
    )
    parser.add_argument(
        "--demo-eval",
        action="store_true",
        help="Run full Cellpose eval (original demo) and save flows/masks.",
    )
    parser.add_argument(
        "--export-native",
        type=str,
        default=None,
        metavar="OUT.npz",
        help="With --image: save native [C,h,w] + y [3,ly,lx] for first tile (use --native-layer).",
    )
    parser.add_argument(
        "--native-layer",
        type=str,
        default="out",
        choices=("neck", "out", "last_conv"),
        help="Layer to hook for --export-native / --native-tile-pipeline.",
    )
    parser.add_argument(
        "--native-tile-pipeline",
        action="store_true",
        help="With --image: first tile only — capture native activation, reconstruct 3ch, segment.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        metavar="FILE.npy|.npz",
        help="Native intermediate [C,h,w] or NPZ with 'neck'/'out', OR already [3,Ly,Lx] flows+cellprob.",
    )
    parser.add_argument(
        "--from-layer",
        type=str,
        default="out",
        choices=("neck", "out", "last_conv"),
        help="Which intermediate map (required for multi-channel inputs that are not 3-channel).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="./cellpose_pipeline_demo_out",
        help="Output directory.",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="cpsam",
        help="Cellpose pretrained model name or path.",
    )
    args = parser.parse_args()

    if args.explain_only:
        print_explanation()
        return 0

    if args.export_native:
        if not args.image:
            print("--export-native requires --image", file=sys.stderr)
            return 1
        print_explanation()
        run_export_native(
            Path(args.image),
            Path(args.export_native),
            layer=args.native_layer,
            gpu=args.gpu,
            diameter=args.diameter,
        )
        return 0

    if args.native_tile_pipeline:
        if not args.image:
            print("--native-tile-pipeline requires --image", file=sys.stderr)
            return 1
        print_explanation()
        run_native_tile_pipeline(
            Path(args.image),
            Path(args.out_dir),
            layer=args.native_layer,
            gpu=args.gpu,
            diameter=args.diameter,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
        )
        return 0

    if args.activation:
        print_explanation()
        run_from_intermediate_file(
            Path(args.activation),
            from_layer=args.from_layer,
            out_dir=Path(args.out_dir),
            gpu=args.gpu,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            pretrained_model=args.pretrained_model,
        )
        return 0

    if args.demo_eval or args.image:
        if not args.image:
            print("--image required for --demo-eval", file=sys.stderr)
            return 1
        print_explanation()
        print("\n--- Full eval demo ---\n")
        run_demo(
            Path(args.image),
            Path(args.out_dir),
            gpu=args.gpu,
            diameter=args.diameter,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
        )
        return 0

    print_explanation()
    print(
        "\nModes:\n"
        "  --explain-only\n"
        "  --image IMG --demo-eval  (full Cellpose, save flows/masks)\n"
        "  --image IMG --export-native OUT.npz  (save native activation + 3ch for tile 0)\n"
        "  --image IMG --native-tile-pipeline  (activation → 3ch → masks, first tile)\n"
        "  --activation FILE.npy|.npz [--from-layer neck|out]  (intermediate or 3ch → masks)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
