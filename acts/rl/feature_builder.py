from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from acts.evaluation.metrics import dice
from acts.sequence.score import boundary_gradient_score, contrast_score
from acts.sequence.state import SequenceState


STATE_NAMES = ["stable", "expanding", "shrinking", "disappearing", "drift", "unreliable"]


@dataclass(frozen=True)
class FeatureVector:
    names: list[str]
    values: list[float]


def build_feature_vector(
    *,
    slice_index: int,
    depth: int,
    ref_index: int,
    direction: int,
    step_id: int,
    current_mask: np.ndarray,
    prev_mask: np.ndarray,
    next_mask: np.ndarray,
    ref_mask: np.ndarray,
    image: np.ndarray,
    prev_image: np.ndarray,
    next_image: np.ndarray,
    state: SequenceState,
    candidate_no_gt_scores: list[float],
    candidate_sam_scores: list[float],
    empty_score: float,
    best_non_empty_score: float,
    best_sam_score: float,
    best_non_sam_score: float,
) -> FeatureVector:
    current = (current_mask > 0).astype(np.uint8)
    prev = (prev_mask > 0).astype(np.uint8)
    next_m = (next_mask > 0).astype(np.uint8)
    ref = (ref_mask > 0).astype(np.uint8)

    current_area = float(current.sum())
    prev_area = float(prev.sum())
    next_area = float(next_m.sum())
    ref_area = max(float(ref.sum()), 1.0)

    names: list[str] = []
    values: list[float] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(float(value))

    add("slice_index_norm", slice_index / max(depth - 1, 1))
    add("distance_to_ref_norm", abs(slice_index - ref_index) / max(depth - 1, 1))
    add("direction", float(direction))
    add("step_id_norm", step_id / 3.0)

    add("current_area_norm", current_area / max(float(current.size), 1.0))
    add("area_ratio_to_ref", current_area / ref_area)
    add("dice_to_prev", dice(current, prev))
    add("dice_to_next", dice(current, next_m))
    add("area_change_to_prev", _relative_change(current_area, prev_area))
    add("area_change_to_next", _relative_change(current_area, next_area))
    add("centroid_shift_to_ref", _centroid_shift(current, ref))
    add("component_count_norm", min(_component_count(current) / 8.0, 1.0))
    add("is_empty", 1.0 if current_area == 0 else 0.0)

    add("hist_diff_prev", _hist_distance(image, prev_image))
    add("hist_diff_next", _hist_distance(image, next_image))
    add("mask_inside_mean", _inside_mean(image, current))
    add("mask_ring_mean", _ring_mean(image, current))
    add("mask_inside_ring_diff", abs(_inside_mean(image, current) - _ring_mean(image, current)))
    add("boundary_gradient", boundary_gradient_score(current, image))
    add("contrast_score", contrast_score(current, image))

    for state_name in STATE_NAMES:
        add(f"state_{state_name}", 1.0 if state.name == state_name else 0.0)

    add("num_candidates_norm", min(len(candidate_no_gt_scores) / 128.0, 1.0))
    add("current_no_gt_score", _safe_current(candidate_no_gt_scores))
    add("best_candidate_no_gt_score", max(candidate_no_gt_scores) if candidate_no_gt_scores else 0.0)
    add("empty_mask_score", empty_score)
    add("best_non_empty_score", best_non_empty_score)
    add("best_sam_candidate_score", best_sam_score)
    add("best_non_sam_candidate_score", best_non_sam_score)
    add("sam_score_mean", float(np.mean(candidate_sam_scores)) if candidate_sam_scores else 0.0)
    add("sam_score_max", max(candidate_sam_scores) if candidate_sam_scores else 0.0)

    return FeatureVector(names=names, values=values)


def _relative_change(a: float, b: float) -> float:
    return abs(a - b) / max(a, b, 1.0)


def _component_count(mask: np.ndarray) -> int:
    _, num = ndimage.label(mask > 0)
    return int(num)


def _centroid_shift(mask: np.ndarray, ref_mask: np.ndarray) -> float:
    c1 = _centroid(mask)
    c2 = _centroid(ref_mask)
    if c1 is None and c2 is None:
        return 0.0
    if c1 is None or c2 is None:
        return 1.0
    height, width = mask.shape
    norm = max(float(np.hypot(height, width)), 1.0)
    return float(np.linalg.norm(np.asarray(c1) - np.asarray(c2)) / norm)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    hist_a, _ = np.histogram(a.astype(np.float32), bins=32, range=(0, 255), density=True)
    hist_b, _ = np.histogram(b.astype(np.float32), bins=32, range=(0, 255), density=True)
    return float(np.abs(hist_a - hist_b).mean())


def _inside_mean(image: np.ndarray, mask: np.ndarray) -> float:
    fg = mask > 0
    if not fg.any():
        return 0.0
    return float(image[fg].mean() / 255.0)


def _ring_mean(image: np.ndarray, mask: np.ndarray) -> float:
    fg = mask > 0
    if not fg.any():
        return 0.0
    dilated = ndimage.binary_dilation(fg, iterations=4)
    ring = np.logical_and(dilated, ~fg)
    if not ring.any():
        return 0.0
    return float(image[ring].mean() / 255.0)


def _safe_current(scores: list[float]) -> float:
    return float(scores[0]) if scores else 0.0

