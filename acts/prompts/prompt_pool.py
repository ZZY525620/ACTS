from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prompt_from_mask import dilate, erode, interpolate_mask, mask_to_box, mask_to_core_point


@dataclass(frozen=True)
class Prompt:
    name: str
    box: list[int] | None = None
    points: list[list[int]] | None = None
    point_labels: list[int] | None = None
    mask_input: np.ndarray | None = None


def generate_prompt_pool(masks: np.ndarray, t: int, state=None) -> list[Prompt]:
    if t <= 0 or t >= masks.shape[2] - 1:
        return []
    prev_mask = masks[:, :, t - 1]
    cur_mask = masks[:, :, t]
    next_mask = masks[:, :, t + 1]
    interp = interpolate_mask(prev_mask, next_mask)
    prompts: list[Prompt] = []

    state_name = getattr(state, "name", "stable")

    for name, mask in (("prev", prev_mask), ("next", next_mask), ("interp", interp)):
        box = mask_to_box(mask, expand_ratio=0.0)
        if box is not None:
            point = mask_to_core_point(mask)
            for box_name, variant_box in _box_variants_for_state(box, mask.shape, state_name):
                prompts.append(Prompt(name=f"{name}_{box_name}", box=variant_box))
                if point is not None and state_name not in {"disappearing", "unreliable"}:
                    prompts.append(Prompt(name=f"{name}_{box_name}_point", box=variant_box, points=[point], point_labels=[1]))

        if mask.sum() > 0:
            if state_name in {"stable", "drift"}:
                prompts.append(Prompt(name=f"{name}_mask_prompt", mask_input=mask))
            if state_name in {"shrinking", "disappearing", "unreliable", "stable"}:
                prompts.append(Prompt(name=f"{name}_eroded_mask_prompt", mask_input=erode(mask, 2)))
            if state_name in {"expanding", "stable"}:
                prompts.append(Prompt(name=f"{name}_dilated_mask_prompt", mask_input=dilate(mask, 2)))

    # Potential false negative region: supported by neighbors but absent in the current mask.
    miss = np.logical_and(interp > 0, cur_mask == 0).astype(np.uint8)
    miss_point = mask_to_core_point(miss)
    if miss_point is not None:
        prompts.append(Prompt(name="positive_miss_point", points=[miss_point], point_labels=[1]))

    # Potential false positive region: present only in the current mask.
    fp = np.logical_and(cur_mask > 0, interp == 0).astype(np.uint8)
    fp_point = mask_to_core_point(fp)
    if fp_point is not None:
        prompts.append(Prompt(name="negative_fp_point", points=[fp_point], point_labels=[0], mask_input=interp))

    return prompts


def _box_variants_for_state(box: list[int], shape: tuple[int, int], state_name: str) -> list[tuple[str, list[int]]]:
    if state_name == "stable":
        specs = [("tight_box", 1.0), ("expand_box_5", 1.05), ("shrink_box_10", 0.90)]
    elif state_name == "expanding":
        specs = [("tight_box", 1.0), ("expand_box_5", 1.05), ("expand_box_10", 1.10)]
    elif state_name in {"shrinking", "disappearing", "unreliable"}:
        specs = [("tight_box", 1.0), ("shrink_box_10", 0.90), ("shrink_box_20", 0.80)]
    elif state_name == "drift":
        specs = [("tight_box", 1.0), ("expand_box_5", 1.05)]
    else:
        specs = [("tight_box", 1.0), ("expand_box_5", 1.05), ("shrink_box_10", 0.90)]

    variants = [(name, _scale_box(box, scale, shape)) for name, scale in specs]
    if state_name in {"drift", "shrinking", "disappearing"}:
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

