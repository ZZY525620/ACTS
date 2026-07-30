from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acts.rl.actions import Action
from acts.rl.env import CTSliceAgentEnv


def run_smoke_test(cache_dir: str, sample_index: int) -> dict:
    env = CTSliceAgentEnv(cache_dir)
    state = env.reset(sample_index)
    sample = env.sample
    assert sample is not None

    action_sequence = [
        int(Action.SELECT_BEST_SAM),
        int(Action.EMPTY_MASK),
        int(Action.KEEP_CURRENT),
    ]
    steps = []
    for action in action_sequence:
        next_state, reward, done, info = env.step(action)
        steps.append(
            {
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "state_dim": int(next_state.shape[0]),
            }
        )
        if done:
            break

    return {
        "cache_dir": cache_dir,
        "sample_index": sample_index,
        "slice": sample["slice"],
        "state": sample["state"],
        "gt_empty": sample["gt_empty"],
        "initial_state_dim": int(state.shape[0]),
        "action_dim": env.action_dim,
        "steps": steps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the cached RL environment.")
    parser.add_argument("--cache-dir", default=r".\outputs\rl_cache_case0001_liver")
    parser.add_argument("--sample-index", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(run_smoke_test(args.cache_dir, args.sample_index), ensure_ascii=False, indent=2))

