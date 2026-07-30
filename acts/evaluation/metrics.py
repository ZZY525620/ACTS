from __future__ import annotations

import numpy as np


def dice(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-6) -> float:
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    denom = float(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(a, b).sum() + eps) / (denom + eps))


def iou(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-6) -> float:
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    union = float(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float((np.logical_and(a, b).sum() + eps) / (union + eps))


def slice_wise_dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.asarray([dice(pred[:, :, z], gt[:, :, z]) for z in range(pred.shape[2])], dtype=np.float32)


def evaluate_volume(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    sw = slice_wise_dice(pred, gt)
    fg = np.where((pred.sum(axis=(0, 1)) + gt.sum(axis=(0, 1))) > 0)[0]
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "slice_dice_mean_fg": float(sw[fg].mean()) if fg.size else 1.0,
        "slice_dice_min_fg": float(sw[fg].min()) if fg.size else 1.0,
    }


