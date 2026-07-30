from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _normalize_image(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((img - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = np.repeat(_normalize_image(image)[..., None], 3, axis=2).astype(np.float32)
    overlay = base.copy()
    overlay[mask > 0] = color
    blended = (base * (1.0 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def save_comparison_panel(
    path: str | Path,
    image: np.ndarray,
    gt: np.ndarray,
    initial: np.ndarray,
    corrected: np.ndarray,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("image", Image.fromarray(_normalize_image(image)).convert("RGB")),
        ("gt", overlay_mask(image, gt, (40, 220, 90))),
        ("initial", overlay_mask(image, initial, (240, 80, 70))),
        ("corrected", overlay_mask(image, corrected, (60, 140, 255))),
    ]
    w, h = panels[0][1].size
    canvas = Image.new("RGB", (w * len(panels), h + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    for i, (label, panel) in enumerate(panels):
        x = i * w
        canvas.paste(panel, (x, 28))
        draw.text((x + 8, 30), label, fill=(255, 255, 255))
    canvas.save(path)


def save_candidate_grid(
    path: str | Path,
    image: np.ndarray,
    candidates: list[np.ndarray],
    names: list[str],
    max_cols: int = 4,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        return
    panels = [overlay_mask(image, c, (255, 180, 40)) for c in candidates]
    w, h = panels[0].size
    cols = min(max_cols, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    canvas = Image.new("RGB", (cols * w, rows * (h + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, panel in enumerate(panels):
        row, col = divmod(i, cols)
        x = col * w
        y = row * (h + 20)
        canvas.paste(panel, (x, y + 20))
        draw.text((x + 6, y + 4), names[i][:36], fill=(0, 0, 0))
    canvas.save(path)


