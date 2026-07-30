from __future__ import annotations

import numpy as np

from acts.evaluation.metrics import dice
from acts.sequence.state import SequenceState


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


def contrast_score(candidate: np.ndarray, image: np.ndarray) -> float:
    mask = candidate > 0
    if mask.sum() == 0 or (~mask).sum() == 0:
        return 0.0
    inside = float(image[mask].mean())
    outside = float(image[~mask].mean())
    return min(abs(inside - outside) / 80.0, 1.0)


def boundary_gradient_score(candidate: np.ndarray, image: np.ndarray) -> float:
    boundary = _boundary(candidate)
    if boundary.sum() == 0:
        return 0.0
    gy = np.zeros_like(image, dtype=np.float32)
    gx = np.zeros_like(image, dtype=np.float32)
    gy[1:-1, :] = np.abs(image[2:, :].astype(np.float32) - image[:-2, :].astype(np.float32))
    gx[:, 1:-1] = np.abs(image[:, 2:].astype(np.float32) - image[:, :-2].astype(np.float32))
    grad = np.hypot(gx, gy)
    return float(min(grad[boundary].mean() / 80.0, 1.0))


def shape_score(candidate: np.ndarray) -> float:
    area = float((candidate > 0).sum())
    if area == 0:
        return 0.0
    boundary = float(_boundary(candidate).sum())
    compactness = min((4.0 * np.pi * area) / max(boundary * boundary, 1.0), 1.0)
    return float(compactness)


def no_gt_score(candidate: np.ndarray, image: np.ndarray, prev_mask: np.ndarray, next_mask: np.ndarray, sam_score: float) -> float:
    s_seq = 0.5 * dice(candidate, prev_mask) + 0.5 * dice(candidate, next_mask)
    s_img = 0.5 * contrast_score(candidate, image) + 0.5 * boundary_gradient_score(candidate, image)
    s_shape = shape_score(candidate)
    return float(0.4 * s_seq + 0.2 * s_img + 0.2 * s_shape + 0.2 * sam_score)


def state_aware_no_gt_score(
    candidate: np.ndarray,
    image: np.ndarray,
    prev_mask: np.ndarray,
    next_mask: np.ndarray,
    sam_score: float,
    state: SequenceState,
    ref_mask: np.ndarray,
) -> float:
    mask = (candidate > 0).astype(np.uint8)
    prev = (prev_mask > 0).astype(np.uint8)
    next_m = (next_mask > 0).astype(np.uint8)
    ref = (ref_mask > 0).astype(np.uint8)

    area = float(mask.sum())
    prev_area = max(float(prev.sum()), 1.0)
    next_area = max(float(next_m.sum()), 1.0)
    ref_area = max(float(ref.sum()), 1.0)
    neighbor_area = max((prev_area + next_area) / 2.0, 1.0)

    if area == 0:
        if state.name == "disappearing":
            return 0.86
        if state.name == "shrinking":
            return float(0.60 + max(0.0, 0.58 - state.area_ratio_to_ref) * 0.80)
        if state.name == "stable" and state.area_ratio_to_ref < 0.80 and state.centroid_shift_to_ref > 0.10:
            return 0.78
        if (
            state.name == "stable"
            and state.step_from_ref >= 18
            and state.area_ratio_to_ref < 0.90
            and state.centroid_shift_to_ref > 0.15
        ):
            return 0.82
        if state.name == "unreliable":
            return 0.50
        return 0.05

    s_seq = 0.5 * dice(mask, prev) + 0.5 * dice(mask, next_m)
    s_img = 0.5 * contrast_score(mask, image) + 0.5 * boundary_gradient_score(mask, image)
    s_shape = shape_score(mask)
    area_similarity = min(area, neighbor_area) / max(area, neighbor_area, 1.0)

    growth = max(0.0, area / (prev_area * 1.12) - 1.0)
    shrink = max(0.0, prev_area / max(area * 1.12, 1.0) - 1.0)
    ref_over = max(0.0, area / (ref_area * 1.45) - 1.0)
    ref_under = max(0.0, (ref_area * 0.10) / area - 1.0)

    if state.name == "stable":
        return float(
            0.34 * s_seq
            + 0.18 * s_img
            + 0.16 * s_shape
            + 0.18 * sam_score
            + 0.14 * area_similarity
            - 0.35 * abs(area / prev_area - 1.0)
            - 0.35 * ref_over
        )
    if state.name == "expanding":
        return float(
            0.24 * s_seq
            + 0.24 * s_img
            + 0.14 * s_shape
            + 0.20 * sam_score
            + 0.10 * min(growth, 1.0)
            + 0.08 * area_similarity
            - 0.30 * shrink
            - 0.25 * ref_over
        )
    if state.name == "shrinking":
        return float(
            0.26 * s_seq
            + 0.18 * s_img
            + 0.16 * s_shape
            + 0.16 * sam_score
            + 0.12 * min(shrink, 1.0)
            + 0.12 * area_similarity
            - 0.75 * growth
            - 0.35 * ref_over
            - 0.20 * ref_under
        )
    if state.name == "disappearing":
        large_nonempty_penalty = max(0.0, area / (prev_area * 0.70) - 1.0)
        existence_penalty = max(0.0, area / ref_area - 0.12)
        return float(
            0.15 * s_seq
            + 0.18 * s_img
            + 0.12 * s_shape
            + 0.10 * sam_score
            - 0.75 * large_nonempty_penalty
            - 0.45 * ref_over
            - 0.95 * existence_penalty
        )
    if state.name == "drift":
        return float(
            0.18 * s_seq
            + 0.30 * s_img
            + 0.14 * s_shape
            + 0.24 * sam_score
            + 0.10 * area_similarity
            - 0.30 * ref_over
        )
    return float(
        0.22 * s_seq
        + 0.24 * s_img
        + 0.18 * s_shape
        + 0.10 * sam_score
        + 0.10 * area_similarity
        - 0.65 * growth
        - 0.45 * ref_over
    )

def select_by_no_gt_score(
    candidates: list[np.ndarray],
    image: np.ndarray,
    prev_mask: np.ndarray,
    next_mask: np.ndarray,
    sam_scores: list[float],
) -> tuple[np.ndarray, int, list[float]]:
    scores = [no_gt_score(c, image, prev_mask, next_mask, s) for c, s in zip(candidates, sam_scores)]
    best_idx = int(np.argmax(scores))
    return candidates[best_idx].astype(np.uint8), best_idx, scores


def select_by_state_aware_score(
    candidates: list[np.ndarray],
    image: np.ndarray,
    prev_mask: np.ndarray,
    next_mask: np.ndarray,
    sam_scores: list[float],
    state: SequenceState,
    ref_mask: np.ndarray,
) -> tuple[np.ndarray, int, list[float]]:
    scores = [
        state_aware_no_gt_score(c, image, prev_mask, next_mask, s, state, ref_mask)
        for c, s in zip(candidates, sam_scores)
    ]
    best_idx = int(np.argmax(scores))
    return candidates[best_idx].astype(np.uint8), best_idx, scores

