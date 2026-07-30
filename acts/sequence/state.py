from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acts.evaluation.metrics import dice


@dataclass(frozen=True)
class SequenceState:
    name: str
    area_ratio_to_ref: float
    area_ratio_to_neighbors: float
    centroid_shift_to_ref: float
    image_change: float
    continuity_to_neighbors: float
    compactness: float
    step_from_ref: int = 0


def estimate_sequence_state(
    image: np.ndarray,
    prev_image: np.ndarray,
    ref_mask: np.ndarray,
    current_mask: np.ndarray,
    prev_mask: np.ndarray | None = None,
    next_mask: np.ndarray | None = None,
    step_from_ref: int = 0,
    direction: int = 0,
) -> SequenceState:
    """Estimate a generic CT sequence state without organ-specific rules."""
    current = (current_mask > 0).astype(np.uint8)
    ref = (ref_mask > 0).astype(np.uint8)
    neighbor = _neighbor_prior(prev_mask, next_mask, fallback=current)

    current_area = float(current.sum())
    ref_area = max(float(ref.sum()), 1.0)
    neighbor_area = max(float(neighbor.sum()), 1.0)
    area_ratio_to_ref = current_area / ref_area
    area_ratio_to_neighbors = current_area / neighbor_area
    centroid_shift = _centroid_shift(current, ref)
    image_change = _hist_distance(image, prev_image)
    continuity = dice(current, neighbor)
    compactness = _compactness(current)

    far_from_reference = step_from_ref >= 20
    weak_tail_evidence = (
        far_from_reference
        and area_ratio_to_ref < 0.62
        and centroid_shift > 0.12
        and area_ratio_to_neighbors < 1.08
    )

    if current_area == 0 and neighbor_area < 0.05 * ref_area:
        name = "disappearing"
    elif current_area == 0:
        name = "expanding"
    elif area_ratio_to_ref < 0.08 and neighbor_area < 0.10 * ref_area:
        name = "disappearing"
    elif weak_tail_evidence:
        name = "disappearing"
    elif area_ratio_to_ref > 2.20 or compactness < 0.0005:
        name = "unreliable"
    elif centroid_shift > 0.35:
        name = "drift"
    elif area_ratio_to_neighbors < 0.72:
        name = "expanding"
    elif area_ratio_to_neighbors > 1.28:
        name = "shrinking"
    elif area_ratio_to_ref < 0.50:
        name = "shrinking"
    elif area_ratio_to_ref > 1.35:
        name = "expanding"
    else:
        name = "stable"

    return SequenceState(
        name=name,
        area_ratio_to_ref=float(area_ratio_to_ref),
        area_ratio_to_neighbors=float(area_ratio_to_neighbors),
        centroid_shift_to_ref=float(centroid_shift),
        image_change=float(image_change),
        continuity_to_neighbors=float(continuity),
        compactness=float(compactness),
        step_from_ref=int(step_from_ref),
    )


def _neighbor_prior(
    prev_mask: np.ndarray | None,
    next_mask: np.ndarray | None,
    fallback: np.ndarray,
) -> np.ndarray:
    masks = []
    if prev_mask is not None:
        masks.append((prev_mask > 0).astype(np.float32))
    if next_mask is not None:
        masks.append((next_mask > 0).astype(np.float32))
    if not masks:
        return (fallback > 0).astype(np.uint8)
    return (np.sum(masks, axis=0) > 0).astype(np.uint8)


def _centroid_shift(mask: np.ndarray, ref_mask: np.ndarray) -> float:
    mask_centroid = _centroid(mask)
    ref_centroid = _centroid(ref_mask)
    if mask_centroid is None or ref_centroid is None:
        return 1.0
    y, x = mask_centroid
    ref_y, ref_x = ref_centroid
    height, width = mask.shape
    norm = max(float(np.hypot(height, width)), 1.0)
    return float(np.hypot(y - ref_y, x - ref_x) / norm)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _hist_distance(image: np.ndarray, prev_image: np.ndarray) -> float:
    hist, _ = np.histogram(image.astype(np.float32), bins=32, range=(0, 255), density=True)
    prev_hist, _ = np.histogram(prev_image.astype(np.float32), bins=32, range=(0, 255), density=True)
    return float(0.5 * np.abs(hist - prev_hist).sum())


def _compactness(mask: np.ndarray) -> float:
    area = float((mask > 0).sum())
    if area == 0:
        return 0.0
    boundary = float(_boundary(mask).sum())
    return float(min((4.0 * np.pi * area) / max(boundary * boundary, 1.0), 1.0))


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask > 0
    padded = np.pad(m.astype(np.uint8), 1)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return np.logical_and(m, ~eroded)

