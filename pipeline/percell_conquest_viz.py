"""Per-cell figures and conquest ROI helpers (SICLE-style unconquerable regions)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def unconquerable_mask(full_mask_crop: "np.ndarray", label: int) -> "np.ndarray":
    """Other Cellpose instances in the bbox: outside conquest ROI (not BG scribbles)."""
    m = np.asarray(full_mask_crop, dtype=np.int32)
    lab = int(label)
    return ((m != 0) & (m != lab)).astype(np.uint8)


def ignored_region_mask(full_mask_crop: "np.ndarray", label: int) -> "np.ndarray":
    """Alias for :func:`unconquerable_mask` (legacy name)."""
    return unconquerable_mask(full_mask_crop, label)


def background_scribble_mask(
    h: int,
    w: int,
    full_mask_crop: "np.ndarray",
    label: int,
    *,
    border_px: int = 2,
    use_unconquerable: bool,
    erosion_bg_pixels: int = 0,
) -> "np.ndarray":
    """BG scribbles: border band only when ``use_unconquerable`` (other cells are not BG)."""
    import sys

    _dout = Path(__file__).resolve().parent.parent.parent
    if str(_dout) not in sys.path:
        sys.path.insert(0, str(_dout))
    from cellpose_to_idisf_pipeline import background_mask_for_crop

    return background_mask_for_crop(
        h,
        w,
        full_mask_crop,
        label,
        border_px=border_px,
        erosion_bg_pixels=erosion_bg_pixels if not use_unconquerable else 0,
        use_other_cells=not use_unconquerable,
    )


def neutralize_unconquerable_rgb(
    img_crop: "np.ndarray",
    unconquerable: "np.ndarray",
    *,
    full_mask_crop: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Replace unconquerable pixels with flat color (mean of crop background) before iDISF."""
    img = np.asarray(img_crop, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = img[..., :3].copy()
    if img.max() <= 1.0:
        img = img * 255.0
    u = np.asarray(unconquerable, dtype=bool)
    if not u.any():
        return np.clip(img, 0, 255).astype(np.uint8)

    if full_mask_crop is not None:
        ref = np.asarray(full_mask_crop, dtype=np.int32) == 0
        if ref.any():
            fill = img[ref].mean(axis=0)
        else:
            fill = img[~u].mean(axis=0) if (~u).any() else np.array([128.0, 128.0, 128.0])
    else:
        fill = img[~u].mean(axis=0) if (~u).any() else np.array([128.0, 128.0, 128.0])

    out = img.copy()
    out[u] = fill
    return np.clip(out, 0, 255).astype(np.uint8)


def force_idisf_unconquerable_background(
    label_img: "np.ndarray",
    unconquerable: "np.ndarray",
    *,
    background_label: int = 2,
) -> "np.ndarray":
    """Post-iDISF: pixels on other cells cannot remain object (like SICLE outside ``--mask``)."""
    L = np.asarray(label_img, dtype=np.int32).copy()
    L[np.asarray(unconquerable, dtype=bool)] = background_label
    return L


def plot_object_bg_ignored(
    img_crop: "np.ndarray",
    *,
    fg_mask: "np.ndarray | None" = None,
    bg_mask: "np.ndarray | None" = None,
    ignored_mask: "np.ndarray | None" = None,
    fg_coords: list[tuple[int, int]] | None = None,
    bg_coords: list[tuple[int, int]] | None = None,
    alpha_ignored: float = 0.50,
    alpha_bg: float = 0.40,
    alpha_fg: float = 0.40,
    dot_radius: int = 2,
) -> "np.ndarray":
    """
    RGB overlay on the crop:
      - magenta/purple tint = inconquistável (outras células, fora da ROI)
      - red tint = fundo (scribbles BG, só borda do crop)
      - green tint = objeto (scribbles FG)
    Scribble coords (if given) are drawn as bright dots on top.
    """
    img = np.asarray(img_crop)
    if img.ndim == 2:
        base = np.stack([img, img, img], axis=-1).astype(np.float32)
    else:
        base = np.asarray(img[..., :3], dtype=np.float32).copy()
    if base.max() <= 1.0:
        base = base * 255.0
    out = np.clip(base * 0.55, 0, 255).astype(np.float32)
    h, w = out.shape[0], out.shape[1]

    def _blend(mask: "np.ndarray", color: tuple[float, float, float], alpha: float) -> None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (h, w):
            return
        if not m.any():
            return
        for c, v in enumerate(color):
            out[..., c] = np.where(m, out[..., c] * (1.0 - alpha) + v * alpha, out[..., c])

    if ignored_mask is not None:
        ign = np.asarray(ignored_mask, dtype=bool)
        # draw ignored first so BG/FG markers stay visible on top
        _blend(ign, (200.0, 80.0, 255.0), alpha_ignored)
    if bg_mask is not None:
        bg = np.asarray(bg_mask, dtype=bool)
        if ignored_mask is not None:
            bg = bg & ~np.asarray(ignored_mask, dtype=bool)
        _blend(bg, (255.0, 60.0, 60.0), alpha_bg)
    if fg_mask is not None:
        fg = np.asarray(fg_mask, dtype=bool)
        if ignored_mask is not None:
            fg = fg & ~np.asarray(ignored_mask, dtype=bool)
        _blend(fg, (60.0, 255.0, 80.0), alpha_fg)

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    def _dots(coords: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
        for (x, y) in coords:
            for dy in range(-dot_radius, dot_radius + 1):
                for dx in range(-dot_radius, dot_radius + 1):
                    if dx * dx + dy * dy <= dot_radius * dot_radius:
                        r, c = y + dy, x + dx
                        if 0 <= r < h and 0 <= c < w:
                            out_u8[r, c] = color

    if fg_coords:
        _dots(fg_coords, (0, 255, 0))
    if bg_coords:
        _dots(bg_coords, (255, 0, 0))

    return out_u8


def write_marker_legend_strip(width: int = 400, height: int = 88) -> "np.ndarray":
    """Small RGB legend: objeto / fundo (borda) / inconquistável."""
    from PIL import Image, ImageDraw, ImageFont

    strip = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    items = [
        ((60, 255, 80), "Objeto (FG)"),
        ((255, 60, 60), "Fundo (BG, borda crop)"),
        ((200, 80, 255), "Inconquistavel (outras cel.)"),
    ]
    y = 12
    for color, text in items:
        draw.rectangle([12, y, 28, y + 14], fill=color)
        draw.text((36, y), text, fill=(240, 240, 240), font=font)
        y += 20
    return np.asarray(strip, dtype=np.uint8)


def write_percell_marker_figure(
    cell_dir: Path,
    img_crop: "np.ndarray",
    *,
    fg_mask: "np.ndarray",
    bg_mask: "np.ndarray",
    ignored_mask: "np.ndarray | None",
    fg_coords: list[tuple[int, int]] | None = None,
    bg_coords: list[tuple[int, int]] | None = None,
    basename: str = "markers_object_bg_ignored",
) -> Path:
    """Save combined marker figure + mask channels + legend."""
    from PIL import Image

    cell_dir = Path(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    panel = plot_object_bg_ignored(
        img_crop,
        fg_mask=fg_mask,
        bg_mask=bg_mask,
        ignored_mask=ignored_mask,
        fg_coords=fg_coords,
        bg_coords=bg_coords,
    )
    legend = write_marker_legend_strip()
    h = max(panel.shape[0], legend.shape[0])
    w = panel.shape[1] + legend.shape[1] + 8
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: panel.shape[0], : panel.shape[1]] = panel
    y0 = (h - legend.shape[0]) // 2
    canvas[y0 : y0 + legend.shape[0], panel.shape[1] + 8 : panel.shape[1] + 8 + legend.shape[1]] = legend

    out_path = cell_dir / f"{basename}.png"
    Image.fromarray(canvas).save(out_path)

    Image.fromarray((np.asarray(fg_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255, mode="L").save(
        cell_dir / "marker_object_mask.png"
    )
    Image.fromarray((np.asarray(bg_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255, mode="L").save(
        cell_dir / "marker_background_mask.png"
    )
    if ignored_mask is not None:
        u8 = (np.asarray(ignored_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        Image.fromarray(u8, mode="L").save(cell_dir / "marker_unconquerable_mask.png")
        Image.fromarray(u8, mode="L").save(cell_dir / "marker_ignored_mask.png")
    return out_path
