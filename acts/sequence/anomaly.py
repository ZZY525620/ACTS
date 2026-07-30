from __future__ import annotations

import numpy as np

from acts.evaluation.metrics import dice


def _norm(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    mn = float(values.min())
    mx = float(values.max())
    if mx - mn < 1e-6:
        return np.zeros_like(values)
    return (values - mn) / (mx - mn)


def hist_distance(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    ha, _ = np.histogram(a.ravel(), bins=bins, range=(0, 255), density=True)
    hb, _ = np.histogram(b.ravel(), bins=bins, range=(0, 255), density=True)
    return float(np.abs(ha - hb).mean())


def area_change(a: np.ndarray, b: np.ndarray) -> float:
    aa = float((a > 0).sum())
    bb = float((b > 0).sum())
    return abs(aa - bb) / max(aa, bb, 1.0)


def centroid_shift(a: np.ndarray, b: np.ndarray) -> float:
    def center(mask):
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        return np.asarray([ys.mean(), xs.mean()], dtype=np.float32)

    ca = center(a)
    cb = center(b)
    if ca is None and cb is None:
        return 0.0
    if ca is None or cb is None:
        return 1.0
    diag = float(np.hypot(*a.shape))
    return float(np.linalg.norm(ca - cb) / diag)


def compute_anomaly_scores(images_uint8: np.ndarray, masks: np.ndarray) -> np.ndarray:
    depth = masks.shape[2]
    image_change = np.zeros(depth, dtype=np.float32)
    mask_change = np.zeros(depth, dtype=np.float32)
    for t in range(1, depth - 1):
        image_change[t] = hist_distance(images_uint8[:, :, t], images_uint8[:, :, t - 1]) + hist_distance(
            images_uint8[:, :, t], images_uint8[:, :, t + 1]
        )
        cur = masks[:, :, t]
        prev = masks[:, :, t - 1]
        nxt = masks[:, :, t + 1]
        mask_change[t] = (
            (1.0 - dice(cur, prev))
            + (1.0 - dice(cur, nxt))
            + area_change(cur, prev)
            + area_change(cur, nxt)
            + centroid_shift(cur, prev)
            + centroid_shift(cur, nxt)
        )
    scores = np.abs(_norm(mask_change) - _norm(image_change))
    scores[0] = -1.0
    scores[-1] = -1.0
    return scores


def select_topk_slices(scores: np.ndarray, k: int, foreground_mask: np.ndarray | None = None) -> list[int]:
    adjusted = scores.copy()
    if foreground_mask is not None:
        fg = foreground_mask.sum(axis=(0, 1)) > 0
        adjusted[~fg] = -1.0
    k = max(1, min(int(k), len(scores)))
    order = np.argsort(adjusted)[::-1]
    return [int(x) for x in order[:k] if adjusted[x] >= 0]


