from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def mask_to_box(mask: np.ndarray, expand_ratio: float = 0.1) -> list[int] | None:
    """Return an expanded foreground bounding box as [x1, y1, x2, y2]."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    w = max(1.0, x2 - x1 + 1.0)
    h = max(1.0, y2 - y1 + 1.0)
    x1 -= expand_ratio * w
    x2 += expand_ratio * w
    y1 -= expand_ratio * h
    y2 += expand_ratio * h
    height, width = mask.shape
    return [
        int(np.clip(round(x1), 0, width - 1)),
        int(np.clip(round(y1), 0, height - 1)),
        int(np.clip(round(x2), 0, width - 1)),
        int(np.clip(round(y2), 0, height - 1)),
    ]


def mask_to_core_point(mask: np.ndarray) -> list[int] | None:
    foreground = mask > 0
    if not np.any(foreground):
        return None
    dist = distance_transform_edt(foreground)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [int(x), int(y)]


def interpolate_mask(prev_mask: np.ndarray, next_mask: np.ndarray) -> np.ndarray:
    """Intersect neighboring masks to keep regions supported by both slices."""
    prior = ((prev_mask.astype(np.float32) + next_mask.astype(np.float32)) / 2.0) > 0.5
    return prior.astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component."""
    mask = np.asarray(mask) > 0
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    for y0, x0 in zip(*np.where(mask & ~visited)):
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    out = np.zeros(mask.shape, dtype=np.uint8)
    if best:
        yy, xx = zip(*best)
        out[np.asarray(yy), np.asarray(xx)] = 1
    return out


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    h, w = mask.shape
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)
    if src_y1 < src_y2 and src_x1 < src_x2:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return out


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    for _ in range(iterations):
        padded = np.pad(out, 1)
        out = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        ).astype(np.uint8)
    return out


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    for _ in range(iterations):
        padded = np.pad(out, 1)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        ).astype(np.uint8)
    return out

