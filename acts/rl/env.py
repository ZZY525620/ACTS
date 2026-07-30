from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from acts.rl.actions import ACTION_NAMES, Action


@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class CTSliceAgentEnv:
    """Cached slice-level environment for the first DQN MVP.

    The environment reads precomputed SAM/non-SAM candidates from
    candidate_cache.py. It does not call SAM online. One episode is one
    abnormal slice with up to max_steps candidate-selection actions.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_steps: int = 3,
        alpha_exist: float = 0.5,
        beta_seq: float = 0.2,
        gamma_reliable: float = 0.1,
        action_cost_weight: float = 0.02,
        invalid_action_weight: float = 0.5,
        accept_only_dice_improvement: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "cache_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.samples: list[dict[str, Any]] = list(self.metadata["samples"])
        if not self.samples:
            raise ValueError("Candidate cache has no samples.")

        self.max_steps = int(max_steps)
        self.alpha_exist = float(alpha_exist)
        self.beta_seq = float(beta_seq)
        self.gamma_reliable = float(gamma_reliable)
        self.action_cost_weight = float(action_cost_weight)
        self.invalid_action_weight = float(invalid_action_weight)
        self.accept_only_dice_improvement = bool(accept_only_dice_improvement)

        self.sample_index = 0
        self.sample: dict[str, Any] | None = None
        self.candidate_masks: np.ndarray | None = None
        self.current_dice = 0.0
        self.current_no_gt_score = 0.0
        self.current_empty = False
        self.step_id = 0
        self.done = False

    @property
    def action_dim(self) -> int:
        return len(ACTION_NAMES)

    @property
    def state_dim(self) -> int:
        return len(self.samples[0]["state_vector"])

    def reset(self, sample_index: int | None = None) -> np.ndarray:
        if sample_index is None:
            sample_index = self.sample_index
            self.sample_index = (self.sample_index + 1) % len(self.samples)
        if not 0 <= sample_index < len(self.samples):
            raise IndexError(f"sample_index out of range: {sample_index}")

        self.sample = self.samples[sample_index]
        mask_path = Path(self.sample["mask_file"])
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing cached candidate masks: {mask_path}")
        self.candidate_masks = np.load(mask_path)["masks"].astype(np.uint8)
        self.current_dice = float(self.sample["current_dice"])
        self.current_no_gt_score = float(self.sample["current_no_gt_score"])
        self.current_empty = bool(self.sample["current_empty"])
        self.step_id = 0
        self.done = False
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.sample is None or self.candidate_masks is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            return self._state(), 0.0, True, {"already_done": True}

        action = int(action)
        action_name = ACTION_NAMES.get(action, f"unknown_{action}")
        old_dice = self.current_dice
        old_empty = self.current_empty
        candidate_record = self._candidate_for_action(action)
        invalid_action = candidate_record is None

        if invalid_action:
            reward = -self.invalid_action_weight
            self.step_id += 1
            self.done = self.step_id >= self.max_steps
            info = {
                "action": action,
                "action_name": action_name,
                "invalid_action": True,
                "accepted": False,
                "old_dice": old_dice,
                "new_dice": old_dice,
                "reward_terms": {"invalid": -self.invalid_action_weight},
            }
            return self._state(), float(reward), self.done, info

        new_dice = float(candidate_record["dice_to_gt"])
        new_empty = int(candidate_record["area"]) == 0
        action_cost = 1.0 if bool(candidate_record["is_sam_candidate"]) else 0.0
        stop_direction = action == int(Action.STOP_DIRECTION)

        reward, reward_terms = self._compute_reward(
            old_dice=old_dice,
            new_dice=new_dice,
            old_empty=old_empty,
            new_empty=new_empty,
            action_cost=action_cost,
        )
        accepted = self._accept(old_dice, new_dice)
        if accepted:
            self.current_dice = new_dice
            self.current_no_gt_score = float(candidate_record["no_gt_score"])
            self.current_empty = new_empty

        self.step_id += 1
        terminal_action = action in {
            int(Action.KEEP_CURRENT),
            int(Action.EMPTY_MASK),
            int(Action.STOP_DIRECTION),
        }
        self.done = terminal_action or stop_direction or self.step_id >= self.max_steps

        info = {
            "action": action,
            "action_name": action_name,
            "candidate_index": int(candidate_record["index"]),
            "candidate_name": candidate_record["name"],
            "invalid_action": False,
            "accepted": bool(accepted),
            "stop_direction": bool(stop_direction),
            "old_dice": float(old_dice),
            "new_dice": float(new_dice),
            "current_dice": float(self.current_dice),
            "old_empty": bool(old_empty),
            "new_empty": bool(new_empty),
            "reward_terms": reward_terms,
        }
        return self._state(), float(reward), self.done, info

    def _candidate_for_action(self, action: int) -> dict[str, Any] | None:
        assert self.sample is not None
        if action == int(Action.KEEP_CURRENT):
            return {
                "index": -1,
                "name": "keep_current",
                "dice_to_gt": self.current_dice,
                "no_gt_score": self.current_no_gt_score,
                "area": 0 if self.current_empty else 1,
                "is_sam_candidate": False,
            }
        by_action = self.sample["best_candidate_by_action"]
        compact_record = by_action.get(str(action))
        if compact_record is None:
            return None
        candidate_index = int(compact_record["candidate_index"])
        if candidate_index < 0:
            return None
        return self.sample["candidates"][candidate_index]

    def _compute_reward(
        self,
        *,
        old_dice: float,
        new_dice: float,
        old_empty: bool,
        new_empty: bool,
        action_cost: float,
    ) -> tuple[float, dict[str, float]]:
        assert self.sample is not None
        gt_empty = bool(self.sample["gt_empty"])

        r_dice = 2.0 * (new_dice - old_dice)
        if gt_empty and new_empty:
            r_exist = 1.0
        elif gt_empty and not new_empty:
            r_exist = -1.0
        elif (not gt_empty) and new_empty:
            r_exist = -1.0
        else:
            r_exist = 0.2

        # Cached MVP uses Dice improvement as the main sequence-local signal.
        # Full image/mask sequence consistency can be added after multi-case cache.
        r_seq = new_dice - old_dice
        r_reliable = 0.1 if (not new_empty and new_dice >= old_dice) else -0.1 if new_empty and not gt_empty else 0.0
        r_cost = action_cost

        total = (
            r_dice
            + self.alpha_exist * r_exist
            + self.beta_seq * r_seq
            + self.gamma_reliable * r_reliable
            - self.action_cost_weight * r_cost
        )
        return float(total), {
            "dice": float(r_dice),
            "exist": float(self.alpha_exist * r_exist),
            "seq": float(self.beta_seq * r_seq),
            "reliable": float(self.gamma_reliable * r_reliable),
            "cost": float(-self.action_cost_weight * r_cost),
        }

    def _accept(self, old_dice: float, new_dice: float) -> bool:
        if self.accept_only_dice_improvement:
            return new_dice > old_dice
        return True

    def _state(self) -> np.ndarray:
        assert self.sample is not None
        state = np.asarray(self.sample["state_vector"], dtype=np.float32).copy()
        feature_names = self.sample["feature_names"]
        updates = {
            "step_id_norm": self.step_id / max(float(self.max_steps), 1.0),
            "current_no_gt_score": self.current_no_gt_score,
            "is_empty": 1.0 if self.current_empty else 0.0,
        }
        for name, value in updates.items():
            if name in feature_names:
                state[feature_names.index(name)] = float(value)
        return state

