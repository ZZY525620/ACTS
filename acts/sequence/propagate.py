from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acts.evaluation.metrics import dice
from acts.prompts.prompt_from_mask import dilate, erode, mask_to_box, mask_to_core_point
from acts.sequence.state import SequenceState, estimate_sequence_state


@dataclass
class MaskCandidate:
    name: str
    mask: np.ndarray
    sam_score: float
    score: float = 0.0


@dataclass
class PropagationStep:
    slice: int
    direction: int
    source_slice: int
    state: str
    selected: str
    score: float
    reliable: bool
    area: int
    source_area: int
    stop_triggered: bool = False


def propagate_sequence(
    images: np.ndarray,
    ref_index: int,
    ref_mask: np.ndarray,
    sam_tool,
    use_reliable_masks: bool = True,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[PropagationStep]]:
    """Dynamic prompt-pool propagation with an optional reliable-mask list.

    CT Sequence Agent.pdf experiment 3 says unreliable masks should not be
    used as the next prompt source. When use_reliable_masks=True, prompt
    generation falls back to the nearest reliable mask instead of blindly
    using the previous slice.
    """
    height, width, depth = ref_mask.shape[0], ref_mask.shape[1], images.shape[2]
    masks = np.zeros((height, width, depth), dtype=np.uint8)
    masks[:, :, ref_index] = ref_mask.astype(np.uint8)
    ref_area = max(float(ref_mask.sum()), 1.0)
    report: list[PropagationStep] = []

    _propagate_direction(
        images,
        masks,
        sam_tool,
        ref_index,
        ref_area,
        direction=1,
        use_reliable_masks=use_reliable_masks,
        report=report,
    )
    _propagate_direction(
        images,
        masks,
        sam_tool,
        ref_index,
        ref_area,
        direction=-1,
        use_reliable_masks=use_reliable_masks,
        report=report,
    )

    if return_report:
        return masks, report
    return masks


def _propagate_direction(
    images: np.ndarray,
    masks: np.ndarray,
    sam_tool,
    ref_index: int,
    ref_area: float,
    direction: int,
    use_reliable_masks: bool,
    report: list[PropagationStep],
) -> None:
    depth = images.shape[2]
    last_reliable = ref_index
    if direction > 0:
        iterator = range(ref_index + 1, depth)
    else:
        iterator = range(ref_index - 1, -1, -1)

    empty_count = 0
    gray_images = images[..., 0] if images.ndim == 4 else images
    for t in iterator:
        source_index = last_reliable if use_reliable_masks else t - direction
        source_mask = masks[:, :, source_index]
        if source_mask.sum() == 0:
            break

        sam_tool.set_slice_index(t)
        sam_tool.set_image(images[:, :, t])
        previous_index = max(0, min(depth - 1, t - direction))
        state = estimate_sequence_state(
            image=gray_images[:, :, t],
            prev_image=gray_images[:, :, previous_index],
            ref_mask=masks[:, :, ref_index],
            current_mask=source_mask,
            prev_mask=masks[:, :, previous_index],
            next_mask=None,
            step_from_ref=abs(t - ref_index),
            direction=direction,
        )
        candidates = _generate_dynamic_candidates(source_mask, sam_tool, state)
        best = _select_candidate(candidates, source_mask, ref_area, state, step_from_ref=abs(t - ref_index))
        masks[:, :, t] = best.mask

        reliable = _is_reliable_mask(best.mask, source_mask, ref_area, best.score, state, step_from_ref=abs(t - ref_index))
        if reliable:
            last_reliable = t
        if best.mask.sum() == 0:
            empty_count += 1
        else:
            empty_count = 0
        stop_triggered = _should_stop_propagation(state, best.mask, best.score, empty_count)

        report.append(
            PropagationStep(
                slice=int(t),
                direction=int(direction),
                source_slice=int(source_index),
                state=state.name,
                selected=best.name,
                score=float(best.score),
                reliable=bool(reliable),
                area=int(best.mask.sum()),
                source_area=int(source_mask.sum()),
                stop_triggered=bool(stop_triggered),
            )
        )
        if stop_triggered:
            break


def _generate_dynamic_candidates(prev_mask: np.ndarray, sam_tool, state: SequenceState) -> list[MaskCandidate]:
    candidates: list[MaskCandidate] = []
    prev = (prev_mask > 0).astype(np.uint8)
    point = mask_to_core_point(prev)
    base_box = mask_to_box(prev, expand_ratio=0.0)

    if base_box is not None:
        for name, box in _box_variants(base_box, prev.shape):
            if not _prompt_allowed_for_state(name, state):
                continue
            candidates.extend(_predict_candidates(sam_tool, name, box=box))
            if point is not None and name in {"tight_box", "expand_box_5", "shrink_box_10"}:
                candidates.extend(_predict_candidates(sam_tool, f"{name}_core_point", box=box, points=[point], point_labels=[1]))

    if point is not None and state.name not in {"disappearing", "unreliable"}:
        candidates.extend(_predict_candidates(sam_tool, "core_positive_point", points=[point], point_labels=[1]))

    if prev.sum() > 0:
        if state.name in {"stable", "drift"}:
            candidates.extend(_predict_candidates(sam_tool, "prev_mask_prompt", mask_input=prev))
        if state.name in {"shrinking", "disappearing", "unreliable", "stable"}:
            candidates.extend(_predict_candidates(sam_tool, "eroded_mask_prompt", mask_input=erode(prev, 2)))
        if state.name in {"expanding", "stable"}:
            candidates.extend(_predict_candidates(sam_tool, "dilated_mask_prompt", mask_input=dilate(prev, 2)))

        candidates.append(MaskCandidate("keep_prev_mask", prev, 0.0))
        candidates.append(MaskCandidate("eroded_prev_mask", erode(prev, 2), 0.0))
        if state.name in {"expanding", "stable"}:
            candidates.append(MaskCandidate("dilated_prev_mask", dilate(prev, 1), 0.0))

    candidates.append(MaskCandidate("empty_mask", np.zeros_like(prev, dtype=np.uint8), 0.0))
    return candidates


def _predict_candidates(sam_tool, name: str, **kwargs) -> list[MaskCandidate]:
    pred = sam_tool.predict(**kwargs)
    return [
        MaskCandidate(f"{name}:{i}", mask.astype(np.uint8), float(score))
        for i, (mask, score) in enumerate(zip(pred.masks, pred.scores))
    ]


def _box_variants(box: list[int], shape: tuple[int, int]) -> list[tuple[str, list[int]]]:
    variants = [
        ("tight_box", _scale_box(box, 1.0, shape)),
        ("expand_box_5", _scale_box(box, 1.05, shape)),
        ("expand_box_10", _scale_box(box, 1.10, shape)),
        ("shrink_box_10", _scale_box(box, 0.90, shape)),
        ("shrink_box_20", _scale_box(box, 0.80, shape)),
    ]
    x1, y1, x2, y2 = box
    shift = max(4, int(round(max(x2 - x1 + 1, y2 - y1 + 1) * 0.08)))
    variants.extend(
        [
            ("shifted_box_left", _shift_box(box, -shift, 0, shape)),
            ("shifted_box_right", _shift_box(box, shift, 0, shape)),
            ("shifted_box_up", _shift_box(box, 0, -shift, shape)),
            ("shifted_box_down", _shift_box(box, 0, shift, shape)),
        ]
    )
    return variants


def _prompt_allowed_for_state(name: str, state: SequenceState) -> bool:
    if state.name == "stable":
        return name in {"tight_box", "expand_box_5", "shrink_box_10"}
    if state.name == "expanding":
        return name in {"tight_box", "expand_box_5", "expand_box_10", "shifted_box_left", "shifted_box_right", "shifted_box_up", "shifted_box_down"}
    if state.name == "shrinking":
        return name in {"tight_box", "shrink_box_10", "shrink_box_20", "shifted_box_left", "shifted_box_right", "shifted_box_up", "shifted_box_down"}
    if state.name == "disappearing":
        return name in {"shrink_box_10", "shrink_box_20"}
    if state.name == "drift":
        return name.startswith("shifted_box") or name in {"tight_box", "expand_box_5"}
    if state.name == "unreliable":
        return name in {"tight_box", "shrink_box_10", "shrink_box_20"}
    return True


def _select_candidate(candidates: list[MaskCandidate], prev_mask: np.ndarray, ref_area: float, state: SequenceState, step_from_ref: int) -> MaskCandidate:
    if not candidates:
        return MaskCandidate("empty_mask", np.zeros_like(prev_mask, dtype=np.uint8), 0.0, 0.0)
    for candidate in candidates:
        candidate.score = _candidate_score(candidate, prev_mask, ref_area, state, step_from_ref)
    return max(candidates, key=lambda item: item.score)


def _candidate_score(candidate: MaskCandidate, prev_mask: np.ndarray, ref_area: float, state: SequenceState, step_from_ref: int) -> float:
    mask = (candidate.mask > 0).astype(np.uint8)
    prev = (prev_mask > 0).astype(np.uint8)
    area = float(mask.sum())
    prev_area = max(float(prev.sum()), 1.0)

    if area == 0:
        empty_bonus = 0.15 + min(0.45, step_from_ref / 80.0)
        if state.name == "disappearing":
            empty_bonus += 0.50
        elif state.name == "shrinking":
            empty_bonus += 0.15
        elif state.name == "unreliable":
            empty_bonus += 0.25
        if step_from_ref > 45 and prev_area < 0.50 * ref_area:
            empty_bonus += 0.20
        if prev_area < 0.12 * ref_area:
            empty_bonus += 0.35
        return empty_bonus

    seq_score = dice(mask, prev)
    area_similarity = min(area, prev_area) / max(area, prev_area, 1.0)
    growth_excess = max(0.0, area / (prev_area * 1.12) - 1.0)
    ref_excess = max(0.0, area / (ref_area * 1.35) - 1.0)
    tiny_penalty = 0.20 if area < 0.01 * ref_area and prev_area > 0.10 * ref_area else 0.0

    conservative_bonus = 0.0
    if any(key in candidate.name for key in ("shrink", "eroded", "keep_prev")):
        conservative_bonus = 0.04
    if "dilated" in candidate.name or "expand_box_10" in candidate.name:
        conservative_bonus -= 0.04
    if state.name == "shrinking":
        if any(key in candidate.name for key in ("shrink", "eroded")):
            conservative_bonus += 0.08
        if any(key in candidate.name for key in ("dilated", "expand_box")):
            conservative_bonus -= 0.12
    elif state.name == "expanding":
        if any(key in candidate.name for key in ("expand_box", "dilated")):
            conservative_bonus += 0.06
        if any(key in candidate.name for key in ("shrink", "eroded")):
            conservative_bonus -= 0.06
    elif state.name == "disappearing":
        if any(key in candidate.name for key in ("shrink", "eroded")):
            conservative_bonus += 0.04
        if any(key in candidate.name for key in ("expand", "dilated", "core_positive")):
            conservative_bonus -= 0.20
    elif state.name == "drift":
        if "shifted_box" in candidate.name:
            conservative_bonus += 0.08
    elif state.name == "unreliable":
        if any(key in candidate.name for key in ("keep_prev", "dilated", "expand")):
            conservative_bonus -= 0.15

    disappearance_penalty = 0.0
    if state.name == "disappearing":
        disappearance_penalty = max(0.0, area / max(ref_area, 1.0) - 0.12) * 0.85

    return (
        0.28 * float(candidate.sam_score)
        + 0.34 * seq_score
        + 0.24 * area_similarity
        + conservative_bonus
        - 0.65 * growth_excess
        - 0.90 * ref_excess
        - tiny_penalty
        - disappearance_penalty
    )


def _is_reliable_mask(mask: np.ndarray, source_mask: np.ndarray, ref_area: float, score: float, state: SequenceState, step_from_ref: int) -> bool:
    area = float((mask > 0).sum())
    if area == 0:
        return False
    source_area = max(float((source_mask > 0).sum()), 1.0)
    area_ratio_to_ref = area / max(ref_area, 1.0)
    area_ratio_to_source = area / source_area
    continuity = dice(mask, source_mask)

    if score < 0.35:
        return False
    if state.name in {"disappearing", "unreliable"}:
        return False
    if area_ratio_to_ref > 1.65:
        return False
    if area_ratio_to_ref < 0.015 and step_from_ref < 20:
        return False
    if area_ratio_to_source > 1.45 or area_ratio_to_source < 0.35:
        return False
    if continuity < 0.12 and step_from_ref < 30:
        return False
    return True


def _should_stop_propagation(state: SequenceState, best_mask: np.ndarray, best_score: float, empty_count: int) -> bool:
    if empty_count >= 2:
        return True
    if state.name == "disappearing" and best_mask.sum() == 0 and best_score >= 0.60:
        return True
    return False


def _scale_box(box: list[int], scale: float, shape: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, (x2 - x1 + 1.0) * scale)
    h = max(1.0, (y2 - y1 + 1.0) * scale)
    return _clip_box([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], shape)


def _shift_box(box: list[int], dx: int, dy: int, shape: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    return _clip_box([x1 + dx, y1 + dy, x2 + dx, y2 + dy], shape)


def _clip_box(box: list[float], shape: tuple[int, int]) -> list[int]:
    height, width = shape
    x1, y1, x2, y2 = box
    return [
        int(np.clip(round(x1), 0, width - 1)),
        int(np.clip(round(y1), 0, height - 1)),
        int(np.clip(round(x2), 0, width - 1)),
        int(np.clip(round(y2), 0, height - 1)),
    ]

