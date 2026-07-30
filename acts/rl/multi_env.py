from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from acts.rl.env import CTSliceAgentEnv


class CTMultiCacheSliceAgentEnv(CTSliceAgentEnv):
    """Slice-level environment backed by multiple candidate-cache folders."""

    def __init__(
        self,
        cache_dirs: list[str | Path],
        max_steps: int = 3,
        alpha_exist: float = 0.5,
        beta_seq: float = 0.2,
        gamma_reliable: float = 0.1,
        action_cost_weight: float = 0.02,
        invalid_action_weight: float = 0.5,
        accept_only_dice_improvement: bool = True,
    ) -> None:
        if not cache_dirs:
            raise ValueError("At least one cache directory is required.")

        self.cache_dirs = [Path(p) for p in cache_dirs]
        self.case_metadata: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        for cache_dir in self.cache_dirs:
            metadata_path = cache_dir / "cache_metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.case_metadata.append(metadata)
            case_id = str(metadata.get("case_id", cache_dir.name))
            for sample in metadata["samples"]:
                sample = dict(sample)
                sample["source_cache_dir"] = str(cache_dir)
                sample["case_id"] = str(sample.get("case_id", case_id))
                samples.append(sample)

        if not samples:
            raise ValueError("Candidate caches have no samples.")

        self.cache_dir = self.cache_dirs[0]
        self.metadata = {
            "cache_dirs": [str(p) for p in self.cache_dirs],
            "case_ids": [str(m.get("case_id", "")) for m in self.case_metadata],
            "num_samples": len(samples),
        }
        self.samples = samples

        self.max_steps = int(max_steps)
        self.alpha_exist = float(alpha_exist)
        self.beta_seq = float(beta_seq)
        self.gamma_reliable = float(gamma_reliable)
        self.action_cost_weight = float(action_cost_weight)
        self.invalid_action_weight = float(invalid_action_weight)
        self.accept_only_dice_improvement = bool(accept_only_dice_improvement)

        self.sample_index = 0
        self.sample = None
        self.candidate_masks = None
        self.current_dice = 0.0
        self.current_no_gt_score = 0.0
        self.current_empty = False
        self.step_id = 0
        self.done = False

