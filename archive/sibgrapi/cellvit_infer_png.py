#!/usr/bin/env python3
"""Run CellViT (HV head) on one PNG and write cp_flow-compatible artifacts.

Outputs in ``out_dir``:
  - step04_masks_uint16.npy   (instance labels, int)
  - step03_dP_cellprob.npz    (cellprob_slice0 from nuclei foreground prob; dP omitted)

Requires a PanNuke-trained CellViT checkpoint (e.g. CellViT-256).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
_CELLVIT = _REPO / "CellViT"
# CellViT repo root must precede cell_segmentation/ so ``utils.tools`` resolves correctly.
for _p in (_CELLVIT / "cell_segmentation", _CELLVIT):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)


def _load_model(checkpoint: Path, device: str):
    import torch
    from models.segmentation.cell_segmentation.cellvit import CellViT256, CellViTSAM
    from models.segmentation.cell_segmentation.cellvit_shared import (
        CellViT256Shared,
        CellViTSAMShared,
    )
    from utils.tools import unflatten_dict

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    run_conf = unflatten_dict(ckpt["config"], ".")
    arch = ckpt.get("arch", run_conf.get("model", {}).get("arch", "CellViT256"))
    num_nc = int(run_conf["data"]["num_nuclei_classes"])
    num_tc = int(run_conf["data"]["num_tissue_classes"])
    shared = bool(run_conf["model"].get("shared_skip_connections", True))

    if arch in ("CellViT256", "CellViT256Shared"):
        # Checkpoint ``arch`` wins; ``shared_skip_connections`` alone is not enough.
        cls = CellViT256Shared if arch == "CellViT256Shared" else CellViT256
        model = cls(
            model256_path=None,
            num_nuclei_classes=num_nc,
            num_tissue_classes=num_tc,
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    elif arch in ("CellViTSAM", "CellViTSAMShared"):
        cls = CellViTSAMShared if shared or arch == "CellViTSAMShared" else CellViTSAM
        model = cls(
            model_path=None,
            num_nuclei_classes=num_nc,
            num_tissue_classes=num_tc,
            vit_structure=run_conf["model"]["backbone"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    else:
        raise SystemExit(f"Unsupported CellViT arch in checkpoint: {arch}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, run_conf


def infer_png(
    image_path: Path,
    checkpoint: Path,
    out_dir: Path,
    *,
    gpu: int = 0,
    magnification: int = 40,
    infer_size: int = 256,
) -> dict:
    import numpy as np
    import torch
    import albumentations as A
    import cv2
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    h0, w0 = rgb.shape[:2]

    # Resize for the network (PanNuke tiles are 256px).
    rgb_small = cv2.resize(rgb, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    tf = A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    x = tf(image=rgb_small)["image"]
    tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    model, _run_conf = _load_model(checkpoint, device)
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.forward(tensor)
        else:
            pred = model.forward(tensor)
        inst_t, _types = model.calculate_instance_map(pred, magnification=magnification)

    inst = inst_t[0].detach().cpu().numpy().astype(np.int32)
    nb = pred["nuclei_binary_map"]
    cellprob = torch.softmax(nb, dim=1)[0, 1].detach().cpu().numpy().astype(np.float32)

    if inst.shape != (h0, w0):
        inst = cv2.resize(inst, (w0, h0), interpolation=cv2.INTER_NEAREST)
        cellprob = cv2.resize(cellprob, (w0, h0), interpolation=cv2.INTER_LINEAR)

    masks_u16 = np.asarray(inst, dtype=np.uint16)
    np.save(out_dir / "step04_masks_uint16.npy", masks_u16)
    np.savez_compressed(out_dir / "step03_dP_cellprob.npz", cellprob_slice0=cellprob)

    n_inst = len(np.unique(inst)) - (1 if (inst > 0).any() else 0)
    return {
        "image": str(image_path),
        "shape": [int(h0), int(w0)],
        "n_instances": int(n_inst),
        "device": device,
        "checkpoint": str(checkpoint),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=str, help="Input PNG/TIFF path")
    p.add_argument("-o", "--out-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="CellViT model_best.pth")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--magnification", type=int, default=40, choices=(20, 40))
    p.add_argument("--infer-size", type=int, default=256)
    args = p.parse_args()
    info = infer_png(
        Path(args.image),
        Path(args.checkpoint),
        Path(args.out_dir),
        gpu=args.gpu,
        magnification=args.magnification,
        infer_size=args.infer_size,
    )
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
