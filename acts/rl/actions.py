from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    KEEP_CURRENT = 0
    EMPTY_MASK = 1
    STOP_DIRECTION = 2
    SELECT_TIGHT_BOX = 3
    SELECT_EXPAND_BOX = 4
    SELECT_SHRINK_BOX = 5
    SELECT_SHIFTED_BOX = 6
    SELECT_PREV_MASK_PROMPT = 7
    SELECT_ERODED_MASK_PROMPT = 8
    SELECT_DILATED_MASK_PROMPT = 9
    SELECT_INTERPOLATED_MASK_PROMPT = 10
    SELECT_POSITIVE_POINT = 11
    SELECT_NEGATIVE_POINT = 12
    SELECT_BEST_SAM = 13
    SELECT_BEST_NON_SAM = 14


ACTION_NAMES = {action.value: action.name.lower() for action in Action}


def action_ids_for_candidate(candidate_name: str) -> list[int]:
    """Map a cached candidate name to one or more discrete RL actions."""
    name = candidate_name.lower()
    actions: list[Action] = []

    is_sam_candidate = ":" in name
    if is_sam_candidate:
        actions.append(Action.SELECT_BEST_SAM)
    else:
        actions.append(Action.SELECT_BEST_NON_SAM)

    if name == "keep_current_mask":
        actions.append(Action.KEEP_CURRENT)
    if name == "empty_mask":
        actions.append(Action.EMPTY_MASK)
    if name == "stop_direction":
        actions.append(Action.STOP_DIRECTION)
    if "tight_box" in name:
        actions.append(Action.SELECT_TIGHT_BOX)
    if "expand_box" in name:
        actions.append(Action.SELECT_EXPAND_BOX)
    if "shrink_box" in name:
        actions.append(Action.SELECT_SHRINK_BOX)
    if "shifted_box" in name:
        actions.append(Action.SELECT_SHIFTED_BOX)
    if "mask_prompt" in name:
        if "eroded" in name:
            actions.append(Action.SELECT_ERODED_MASK_PROMPT)
        elif "dilated" in name:
            actions.append(Action.SELECT_DILATED_MASK_PROMPT)
        elif "interp" in name or "interpolated" in name:
            actions.append(Action.SELECT_INTERPOLATED_MASK_PROMPT)
        else:
            actions.append(Action.SELECT_PREV_MASK_PROMPT)
    if "positive" in name or "miss_point" in name:
        actions.append(Action.SELECT_POSITIVE_POINT)
    if "negative" in name or "fp_point" in name:
        actions.append(Action.SELECT_NEGATIVE_POINT)
    if name == "interpolated_mask":
        actions.append(Action.SELECT_INTERPOLATED_MASK_PROMPT)
    if name == "eroded_current_mask":
        actions.append(Action.SELECT_ERODED_MASK_PROMPT)
    if name == "dilated_current_mask":
        actions.append(Action.SELECT_DILATED_MASK_PROMPT)

    unique: list[int] = []
    for action in actions:
        value = int(action.value)
        if value not in unique:
            unique.append(value)
    return unique


