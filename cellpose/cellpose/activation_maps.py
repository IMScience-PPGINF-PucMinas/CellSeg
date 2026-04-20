"""
Capture intermediate activations from the Cellpose network (CP-SAM / Transformer).

Hooks the SAM image encoder neck (256-channel map) or the final 1x1 conv (``out``)
before the patch upsampling step, then merges tiles the same way as ``core.run_net``.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from . import transforms, utils
from .core import _from_device, _to_device

activation_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(activation_logger, level=logging.INFO)

LayerName = Literal["neck", "out", "last_conv"]


def _unaugment_tiles_spatial_only(y: np.ndarray) -> np.ndarray:
    """Reverse tile flips from ``make_tiles(..., augment=True)`` without flow sign flips."""
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, :]
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = y[j, i, :, :, ::-1]
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, ::-1]
    return y


def resolve_activation_module(net: nn.Module, layer: LayerName) -> nn.Module:
    """
    Return the submodule to hook for the CP-SAM ``Transformer`` model.

    Parameters
    ----------
    net : nn.Module
        ``CellposeModel.net`` (``cellpose.vit_sam.Transformer``).
    layer : str
        - ``\"neck\"``: output of ``encoder.neck`` (256 channels, low resolution).
        - ``\"out\"`` / ``\"last_conv\"``: output of the 1x1 conv before ``conv_transpose2d``.

    Returns
    -------
    nn.Module
    """
    if layer in ("out", "last_conv"):
        if not hasattr(net, "out"):
            raise AttributeError(
                "Model has no .out layer; activation maps are only implemented for CP-SAM Transformer."
            )
        return net.out
    if layer == "neck":
        if not hasattr(net, "encoder") or not hasattr(net.encoder, "neck"):
            raise AttributeError(
                "Model has no encoder.neck; activation maps are only implemented for CP-SAM Transformer."
            )
        return net.encoder.neck
    raise ValueError(f"Unknown layer={layer!r}; use 'neck', 'out', or 'last_conv'.")


def _forward_capture(
    net: nn.Module, x: np.ndarray, activation_module: nn.Module
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``net`` on a batch of tiles and return outputs plus hooked activations."""
    X = _to_device(x, device=net.device, dtype=net.dtype)
    net.eval()
    captured: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        captured.append(out.detach())

    handle = activation_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            y, style = net(X)[:2]
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Forward hook did not capture any activation.")
    act = captured[0].float().cpu().numpy()
    y = _from_device(y)
    style = _from_device(style)
    return y, style, act


def _upsample_activation_batch(act: np.ndarray, ly: int, lx: int) -> np.ndarray:
    """Upsample [B, C, h, w] activations to tile size (ly, lx)."""
    t = torch.from_numpy(act.astype(np.float32))
    t = F.interpolate(t, size=(ly, lx), mode="bilinear", align_corners=False)
    return t.numpy()


def reduce_channels_to_rgb(
    act: np.ndarray, mode: str = "norm"
) -> np.ndarray:
    """
    Turn a channel-first map ``[C, Ly, Lx]`` into RGB ``[Ly, Lx, 3]`` for saving.

    Parameters
    ----------
    act : ndarray
        Shape ``[C, Ly, Lx]``.
    mode : str
        ``\"norm\"`` — take first three channels, per-channel min–max to 0–1.
        ``\"max\"`` — repeat max over channels into three bands.
    """
    if act.ndim != 3:
        raise ValueError(f"Expected [C, Ly, Lx], got shape {act.shape}")
    c, ly, lx = act.shape
    if mode == "max":
        m = np.max(act, axis=0)
        m = m - m.min()
        if m.max() > 1e-8:
            m = m / m.max()
        return np.stack([m, m, m], axis=-1)
    # norm: first three channels
    out = np.zeros((ly, lx, 3), dtype=np.float32)
    for k in range(min(3, c)):
        ch = act[k]
        lo, hi = ch.min(), ch.max()
        if hi - lo > 1e-8:
            out[..., k] = (ch - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def run_net_activation(
    net: nn.Module,
    imgi: np.ndarray,
    layer: LayerName = "out",
    batch_size: int = 8,
    augment: bool = False,
    tile_overlap: float = 0.1,
    bsize: int = 256,
    rsz: Optional[Union[float, list, np.ndarray]] = None,
) -> np.ndarray:
    """
    Run the network like ``core.run_net`` and merge activation maps across tiles.

    Parameters
    ----------
    net : nn.Module
        Cellpose network (``model.net``).
    imgi : ndarray
        Stack ``[Lz x Ly x Lx x nchan]`` (same layout as ``run_net``).
    layer : str
        ``\"neck\"`` or ``\"out\"`` (``\"last_conv\"`` is an alias for ``\"out\"``).

    Returns
    -------
    ndarray
        ``[Lz x Ly x Lx x C_up]`` where ``C_up`` is the number of channels after the
        hooked layer, upsampled to full tile resolution and merged like flows.
    """
    activation_module = resolve_activation_module(net, layer)

    Lz, Ly0, Lx0, nchan = imgi.shape
    if rsz is not None:
        if not isinstance(rsz, (list, np.ndarray)):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0

    ly, lx = bsize, bsize
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(bsize, bsize))
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])

    if augment:
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
    else:
        ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))

    ntiles = ny * nx
    nimgs = max(1, batch_size // ntiles)
    niter = int(np.ceil(Lz / nimgs))
    ziterator = trange(niter, file=tqdm_out, mininterval=30) if niter > 10 or Lz > 1 else range(niter)

    n_act_ch: Optional[int] = None

    for k in ziterator:
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32")
        for i, b in enumerate(inds):
            imgb = transforms.resize_image(imgi[b], rsz=rsz) if rsz is not None else imgi[b].copy()
            imgb = np.pad(imgb.transpose(2, 0, 1), pads, mode="constant")
            IMG, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                imgb, bsize=bsize, augment=augment, tile_overlap=tile_overlap
            )
            IMGa[i * ntiles : (i + 1) * ntiles] = np.reshape(IMG, (ny * nx, nchan, ly, lx))

        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            _, _, act0 = _forward_capture(net, IMGa[bslc], activation_module)
            act0 = _upsample_activation_batch(act0, ly, lx)
            if j == 0:
                n_act_ch = act0.shape[1]
                aa = np.zeros((IMGa.shape[0], n_act_ch, ly, lx), np.float32)
            aa[bslc] = act0

        for i, b in enumerate(inds):
            if i == 0 and k == 0:
                actf = np.zeros((Lz, n_act_ch, Ly, Lx), np.float32)
            y = aa[i * ntiles : (i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, n_act_ch, ly, lx))
                y = _unaugment_tiles_spatial_only(y)
                y = np.reshape(y, (-1, n_act_ch, ly, lx))
            yfi = transforms.average_tiles(y, ysub, xsub, Lyt, Lxt)
            actf[b] = yfi[:, : imgb.shape[-2], : imgb.shape[-1]]

    assert n_act_ch is not None
    actf = actf[:, :, ypad1 : Ly - ypad2, xpad1 : Lx - xpad2]
    return actf.transpose(0, 2, 3, 1)


def activation_volume_to_rgb(
    act_vol: np.ndarray, mode: str = "norm"
) -> np.ndarray:
    """Convert ``[Lz, Ly, Lx, C]`` to ``[Lz, Ly, Lx, 3]`` uint8 image."""
    lz, ly, lx, _c = act_vol.shape
    rgb = np.zeros((lz, ly, lx, 3), dtype=np.float32)
    for z in range(lz):
        rgb[z] = reduce_channels_to_rgb(act_vol[z].transpose(2, 0, 1), mode=mode)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def reduce_activation_to_scalar(
    act: np.ndarray, mode: str = "l2"
) -> np.ndarray:
    """
    Collapse channel dimension to a single 2D map for heatmap visualization.

    Parameters
    ----------
    act : ndarray
        ``[Ly, Lx, C]`` (or ``[Lz, Ly, Lx, C]`` — first plane is used if Z>1).
    mode : str
        ``l2`` — Euclidean norm across channels; ``mean_abs`` — mean of |x|; ``max_abs`` — max |x|.
    """
    if act.ndim == 4:
        act = act[0]
    if act.ndim != 3:
        raise ValueError(f"Expected [Ly, Lx, C], got {act.shape}")
    if mode == "l2":
        return np.sqrt(np.sum(act.astype(np.float64) ** 2, axis=-1)).astype(np.float32)
    if mode == "mean_abs":
        return np.mean(np.abs(act), axis=-1).astype(np.float32)
    if mode == "max_abs":
        return np.max(np.abs(act), axis=-1).astype(np.float32)
    raise ValueError(f"Unknown mode={mode!r}")


def activation_to_greyscale_u8(
    act: np.ndarray,
    *,
    reduce_mode: str = "l2",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Single-channel uint8 greyscale map (same geometry as ``act``), for labels / scribbles.

    Collapses channels with :func:`reduce_activation_to_scalar`, then percentile-normalizes
    to 0–255 (same intensity mapping as :func:`activation_to_heatmap_rgb` before colormap).
    """
    scalar = reduce_activation_to_scalar(act, mode=reduce_mode)
    lo = float(np.percentile(scalar, p_low))
    hi = float(np.percentile(scalar, p_high))
    if hi <= lo + 1e-12:
        hi = lo + 1.0
    norm = (scalar - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def activation_to_heatmap_rgb(
    act: np.ndarray,
    *,
    reduce_mode: str = "l2",
    colormap: str = "turbo",
    p_low: float = 1.0,
    p_high: float = 99.0,
    overlay: Optional[np.ndarray] = None,
    overlay_alpha: float = 0.45,
) -> np.ndarray:
    """
    False-color heatmap (uint8 RGB) from a multi-channel activation tensor.

    Uses OpenCV ``applyColorMap`` (``turbo`` if available, else ``jet``).

    Parameters
    ----------
    overlay : ndarray, optional
        Grayscale or RGB ``[Ly, Lx]`` or ``[Ly, Lx, 3]`` image (0–255) aligned with ``act``.
    overlay_alpha : float
        Blend weight for the heatmap; ``1 - alpha`` goes to the overlay.
    """
    import cv2

    scalar = reduce_activation_to_scalar(act, mode=reduce_mode)
    lo = float(np.percentile(scalar, p_low))
    hi = float(np.percentile(scalar, p_high))
    if hi <= lo + 1e-12:
        hi = lo + 1.0
    norm = (scalar - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)

    name = colormap.upper()
    cmap_id = getattr(cv2, f"COLORMAP_{name}", None)
    if cmap_id is None:
        cmap_id = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heat_bgr = cv2.applyColorMap(u8, cmap_id)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    if overlay is None:
        return heat

    ov = np.asarray(overlay)
    if ov.ndim == 2:
        ov = np.stack([ov, ov, ov], axis=-1)
    if ov.dtype != np.uint8:
        ov = np.clip(ov, 0, 255).astype(np.uint8)
    if ov.shape[:2] != heat.shape[:2]:
        ov = cv2.resize(ov, (heat.shape[1], heat.shape[0]), interpolation=cv2.INTER_LINEAR)

    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    blend = (a * heat.astype(np.float32) + (1.0 - a) * ov.astype(np.float32)).round()
    return np.clip(blend, 0, 255).astype(np.uint8)


def intermediate_to_final_output(
    net: nn.Module,
    act: np.ndarray,
    from_layer: LayerName,
) -> np.ndarray:
    """
    Apply the remaining CP-SAM readout so an intermediate map becomes the 3-plane output.

    ``neck`` → ``net.out`` → ``conv_transpose2d(W2)``; ``out`` / ``last_conv`` → only ``W2``.

    Parameters
    ----------
    act : ndarray
        Native-resolution map ``[C, h, w]`` or ``[1, C, h, w]`` (same tensor the hook sees,
        **not** the bilinear-upsampled full-res export used for heatmaps).

    Returns
    -------
    ndarray
        ``[3, Ly, Lx]`` — Y-flow, X-flow, cellprob logit (same layout as ``net`` forward).
    """
    if act.ndim == 4:
        act = act[0]
    if act.ndim != 3:
        raise ValueError(f"Expected [C, h, w], got {act.shape}")
    t = torch.from_numpy(act.astype(np.float32)).to(device=net.device, dtype=net.dtype)
    t = t.unsqueeze(0)
    if from_layer == "neck":
        t = net.out(t)
    elif from_layer not in ("out", "last_conv"):
        raise ValueError("from_layer must be 'neck', 'out', or 'last_conv'")
    y = F.conv_transpose2d(t, net.W2, stride=net.ps, padding=0)
    return y.float().cpu().numpy()[0]


def capture_native_activation_first_tile(
    net: nn.Module,
    imgi: np.ndarray,
    layer: LayerName = "out",
    bsize: int = 256,
    tile_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run **one** tile (same tiling as ``run_net``) and return native hook tensor + full ``y``.

    Parameters
    ----------
    imgi : ndarray
        ``[1, Ly, Lx, nchan]`` preprocessed stack (Cellpose layout).
    tile_index : int
        Which tile to use (0 = top-left), for multi-tile images.

    Returns
    -------
    native : ndarray
        ``[C, h, w]`` activation at hook (before upsampling).
    y : ndarray
        ``[3, ly, lx]`` full network output for that tile (flows + cellprob).
    """
    Lz, Ly0, Lx0, nchan = imgi.shape
    if Lz != 1:
        raise ValueError("capture_native_activation_first_tile expects Lz==1.")
    ly, lx = bsize, bsize
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Ly0, Lx0, min_size=(bsize, bsize))
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])
    imgb = np.pad(imgi[0].transpose(2, 0, 1), pads, mode="constant")
    IMG, _ysub, _xsub, _Lyt, _Lxt = transforms.make_tiles(
        imgb, bsize=bsize, augment=False, tile_overlap=0.1
    )
    ny, nx = IMG.shape[0], IMG.shape[1]
    IMGa = np.reshape(IMG, (ny * nx, nchan, ly, lx))
    if tile_index < 0 or tile_index >= IMGa.shape[0]:
        raise ValueError(f"tile_index {tile_index} out of range for {ny * nx} tiles.")
    batch = IMGa[tile_index : tile_index + 1]
    mod = resolve_activation_module(net, layer)
    y, _style, act = _forward_capture(net, batch, mod)
    return act[0], y[0]
