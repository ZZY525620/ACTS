from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acts.rl.evaluate_dqn import evaluate_dqn


DEFAULT_CASES = {
    "0001": {
        "cache_dir": r".\outputs\rl_cache_case0001_liver",
        "baseline_dir": r".\outputs\ablation_case0001_liver\selected_runs\09_pdf_complete_guarded_final",
    },
    "0002": {
        "cache_dir": r".\outputs\rl_cache_case0002_liver",
        "baseline_dir": r".\outputs\rule_baseline_case0002_liver",
    },
    "0003": {
        "cache_dir": r".\outputs\rl_cache_case0003_liver",
        "baseline_dir": r".\outputs\rule_baseline_case0003_liver",
    },
}


def evaluate_multicase(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_results = []
    for case_id in args.case_ids:
        if case_id not in DEFAULT_CASES:
            raise ValueError(f"No default cache/baseline mapping for case {case_id}.")
        spec = DEFAULT_CASES[case_id]
        case_output_dir = output_dir / f"case_{case_id}"
        case_args = argparse.Namespace(
            data_dir=args.data_dir,
            case_id=case_id,
            liver_label=args.liver_label,
            cache_dir=spec["cache_dir"],
            policy_path=args.policy_path,
            baseline_dir=spec["baseline_dir"],
            output_dir=str(case_output_dir),
            max_steps=args.max_steps,
            hidden_dim=args.hidden_dim,
            window_min=args.window_min,
            window_max=args.window_max,
            device=args.device,
        )
        result = evaluate_dqn(case_args)
        case_results.append(_compact_case_result(result))

    aggregate = {
        "case_ids": args.case_ids,
        "policy_path": args.policy_path,
        "output_dir": str(output_dir),
        "case_results": case_results,
        "note": "Full-volume multi-case evaluation of a DQN policy network. Each RL volume starts from Initial SAM and replaces cached abnormal slices with accepted DQN-selected candidates.",
    }
    (output_dir / "multicase_eval_metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "multicase_eval_table.csv", case_results)
    _write_readme(output_dir / "README_MULTICASE_RL_EVAL.md", aggregate)
    return aggregate


def _compact_case_result(result: dict[str, Any]) -> dict[str, Any]:
    full = result["full_volume_metrics"]
    cached = result["cached_slice_metrics"]
    return {
        "case_id": result["case_id"],
        "cache_dir": result["cache_dir"],
        "baseline_dir": result["baseline_dir"],
        "case_output_dir": result["output_dir"],
        "full_initial_dice": full["initial"]["dice"],
        "full_rule_dice": full["rule_selection"]["dice"],
        "full_rl_dice": full["rl_agent"]["dice"],
        "full_oracle_dice": full["oracle_candidate"]["dice"],
        "cached_current_dice": cached["avg_current_dice"],
        "cached_rule_dice": cached["avg_rule_candidate_dice"],
        "cached_rl_dice": cached["avg_rl_dice"],
        "cached_oracle_dice": cached["avg_oracle_candidate_dice"],
        "num_cached_slices": cached["num_cached_slices"],
        "invalid_action_ratio": result["invalid_action_ratio"],
        "accepted_action_ratio": result["accepted_action_ratio"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "num_cached_slices",
        "full_initial_dice",
        "full_rule_dice",
        "full_rl_dice",
        "full_oracle_dice",
        "cached_current_dice",
        "cached_rule_dice",
        "cached_rl_dice",
        "cached_oracle_dice",
        "invalid_action_ratio",
        "accepted_action_ratio",
        "case_output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_readme(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Multi-case RL DQN Evaluation",
        "",
        f"Policy: {aggregate['policy_path']}",
        "",
        "## Full-volume Dice",
        "",
        "| Case | Initial | Rule | DQN Agent | Oracle |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate["case_results"]:
        lines.append(
            f"| {row['case_id']} | "
            f"{row['full_initial_dice']:.6f} | "
            f"{row['full_rule_dice']:.6f} | "
            f"{row['full_rl_dice']:.6f} | "
            f"{row['full_oracle_dice']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Cached Abnormal-slice Average Dice",
            "",
            "| Case | Current | Rule | DQN Agent | Oracle |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate["case_results"]:
        lines.append(
            f"| {row['case_id']} | "
            f"{row['cached_current_dice']:.6f} | "
            f"{row['cached_rule_dice']:.6f} | "
            f"{row['cached_rl_dice']:.6f} | "
            f"{row['cached_oracle_dice']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `multicase_eval_table.csv`: compact table for reporting.",
            "- `multicase_eval_metrics.json`: full aggregate metrics.",
            "- `case_0001/`, `case_0002/`, `case_0003/`: per-case RL mask, action report, metrics, and visualizations.",
            "",
            "This is an offline cached-candidate DQN agent evaluation. It is not yet an online SAM-interaction agent.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a DQN policy on multiple FLARE liver cases.")
    parser.add_argument("--data-dir", default=r"data")
    parser.add_argument("--case-ids", nargs="+", default=["0001", "0002", "0003"])
    parser.add_argument("--liver-label", type=int, default=1)
    parser.add_argument("--policy-path", default=r".\outputs\rl_dqn_liver_multicase_v1\dqn_policy.pt")
    parser.add_argument("--output-dir", default=r".\outputs\rl_dqn_liver_multicase_v1_eval")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--window-min", type=float, default=-160.0)
    parser.add_argument("--window-max", type=float, default=240.0)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    output = evaluate_multicase(build_parser().parse_args())
    print(json.dumps({"output_dir": output["output_dir"], "case_results": output["case_results"]}, ensure_ascii=False, indent=2))

