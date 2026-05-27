#!/usr/bin/env python3
"""
Gera ``merged_percell_sicle_overlay.png`` (bordos das instâncias sobre a imagem)
a partir de ``merged_percell_sicle_masks_int32.npy`` — mesmo critério do
``percell_sicle_cellprob_pipeline.py`` com ``--image``.

Útil para pastas produzidas só por ``merge_postprocess.py`` (alternativas A/C)
ou para repetir o overlay após trocar o NPY merge.

Exemplo::

    python write_merged_percell_overlay.py \\
        --image /path/to/GR07-1.svs_slice1.tiff \\
        --masks ./percell_three_alts_.../01_alt_a_clip_d2/merged_percell_sicle_masks_int32.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from percell_sicle_cellprob_pipeline import _outline_only_overlay, _selective_border_overlay


def load_rgb_for_shape(image_path: Path, h: int, w: int):
    import cv2
    import numpy as np
    from cellpose import io

    img = io.imread(str(image_path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = np.asarray(img[..., :3], dtype=np.uint8)
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def main() -> int:
    import numpy as np
    from PIL import Image

    p = argparse.ArgumentParser(description="Overlay de bordos do merge percell sobre RGB.")
    p.add_argument("--image", type=str, required=True, help="Imagem RGB (ex. TIFF da lâmina)")
    p.add_argument(
        "--masks",
        type=str,
        required=True,
        help="merged_percell_sicle_masks_int32.npy",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="PNG de saída (default: mesmo dir que --masks / merged_percell_sicle_overlay.png)",
    )
    p.add_argument("--border-color", type=str, default="0,255,0", help="R,G,B for SICLE outline (sicle|both)")
    p.add_argument("--border-thickness", type=int, default=1, help="Espessura em pixels (>=1)")
    p.add_argument(
        "--overlay-source",
        choices=("sicle", "cellpose", "both"),
        default="sicle",
        help="Contornos: só merge; só Cellpose (requer --cellpose); ambos (Cellpose por baixo).",
    )
    p.add_argument(
        "--cellpose",
        type=str,
        default=None,
        help="step04_masks_uint16.npy (obrigatório se overlay-source for cellpose ou both)",
    )
    p.add_argument(
        "--cellpose-border-color",
        type=str,
        default="255,0,0",
        help="R,G,B para contornos Cellpose (cellpose|both)",
    )
    p.add_argument(
        "--also-masks-rgb",
        action="store_true",
        help="Também grava merged_percell_sicle_masks_rgb.png (cellpose.plot.mask_rgb)",
    )
    args = p.parse_args()

    try:
        br, bg, bb = (int(x.strip()) for x in args.border_color.split(","))
        border = (max(0, min(255, br)), max(0, min(255, bg)), max(0, min(255, bb)))
    except ValueError:
        raise SystemExit("--border-color must be like 0,255,0") from None
    try:
        cr, cg, cb = (int(x.strip()) for x in args.cellpose_border_color.split(","))
        border_cp = (max(0, min(255, cr)), max(0, min(255, cg)), max(0, min(255, cb)))
    except ValueError:
        raise SystemExit("--cellpose-border-color must be like 255,0,0") from None
    if args.border_thickness < 1:
        raise SystemExit("--border-thickness must be >= 1")
    if args.overlay_source in ("cellpose", "both") and not args.cellpose:
        raise SystemExit("--cellpose path required when --overlay-source is cellpose or both")

    masks_path = Path(args.masks)
    merged = np.load(masks_path).astype(np.int32, copy=False)
    h, w = merged.shape[:2]

    img_rgb = load_rgb_for_shape(Path(args.image), h, w)
    if args.overlay_source == "sicle":
        ov = _outline_only_overlay(
            img_rgb,
            merged,
            border_color_rgb=border,
            border_thickness=args.border_thickness,
        )
    else:
        cp = np.load(Path(args.cellpose)).astype(np.int32, copy=False)
        if cp.shape != merged.shape:
            raise SystemExit(f"cellpose shape {cp.shape} != merged {merged.shape}")
        ov = _selective_border_overlay(
            img_rgb,
            merged_sicle=merged,
            cellpose_masks=cp,
            source=args.overlay_source,
            border_sicle_rgb=border,
            border_cellpose_rgb=border_cp,
            border_thickness=args.border_thickness,
        )

    out_path = Path(args.out) if args.out else masks_path.parent / "merged_percell_sicle_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ov).save(out_path)
    print(f"Wrote {out_path}")

    if args.also_masks_rgb:
        from cellpose import plot

        rgb_path = masks_path.parent / "merged_percell_sicle_masks_rgb.png"
        try:
            from imageio import imwrite

            imwrite(rgb_path, plot.mask_rgb(merged))
        except ImportError:
            Image.fromarray(plot.mask_rgb(merged)).save(rgb_path)
        print(f"Wrote {rgb_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
