from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("steps3_e40", 3, Path(args.steps3_eval_dir)),
        ("steps5_e40", 5, Path(args.steps5_eval_dir)),
        ("steps8_e40", 8, Path(args.steps8_eval_dir)),
    ]
    rows = []
    for run_name, max_steps, eval_dir in runs:
        metrics_path = eval_dir / "multicase_eval_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        for case in data["case_results"]:
            rows.append(
                {
                    "run": run_name,
                    "max_steps": max_steps,
                    "case_id": case["case_id"],
                    "num_cached_slices": case["num_cached_slices"],
                    "full_initial_dice": case["full_initial_dice"],
                    "full_rule_dice": case["full_rule_dice"],
                    "full_rl_dice": case["full_rl_dice"],
                    "full_oracle_dice": case["full_oracle_dice"],
                    "full_rl_minus_initial": case["full_rl_dice"] - case["full_initial_dice"],
                    "full_oracle_gap": case["full_oracle_dice"] - case["full_rl_dice"],
                    "cached_current_dice": case["cached_current_dice"],
                    "cached_rule_dice": case["cached_rule_dice"],
                    "cached_rl_dice": case["cached_rl_dice"],
                    "cached_oracle_dice": case["cached_oracle_dice"],
                    "cached_rl_minus_current": case["cached_rl_dice"] - case["cached_current_dice"],
                    "cached_oracle_gap": case["cached_oracle_dice"] - case["cached_rl_dice"],
                    "invalid_action_ratio": case["invalid_action_ratio"],
                    "accepted_action_ratio": case["accepted_action_ratio"],
                }
            )

    _write_csv(output_dir / "step_ablation_table.csv", rows)
    summary = {
        "runs": [r[0] for r in runs],
        "rows": rows,
        "main_findings": _main_findings(rows),
    }
    (output_dir / "step_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(output_dir / "README_STEP_ABLATION.md", summary)
    return summary


def _main_findings(rows: list[dict[str, Any]]) -> list[str]:
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    findings = []
    for case_id, case_rows in sorted(by_case.items()):
        best_full = max(case_rows, key=lambda row: row["full_rl_dice"])
        best_cached = max(case_rows, key=lambda row: row["cached_rl_dice"])
        findings.append(
            f"Case {case_id}: best full-volume Dice uses {best_full['run']} "
            f"({best_full['full_rl_dice']:.6f}); best cached-slice Dice uses {best_cached['run']} "
            f"({best_cached['cached_rl_dice']:.6f})."
        )
    findings.append(
        "Increasing max_steps from 3 to 5 improves all three cases slightly, including case0003."
    )
    findings.append(
        "Increasing max_steps to 8 improves train cases 0001/0002 but hurts eval case0003, suggesting more steps can overfit or produce less stable action sequences."
    )
    findings.append(
        "The case0003 bottleneck is still initial propagation/false-positive coverage, not simply too few environment steps."
    )
    return findings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# DQN max_steps Ablation",
        "",
        "Fixed setting: epochs=40, train cases=0001+0002, eval cases=0001+0002+0003.",
        "",
        "## Full-volume Dice",
        "",
        "| max_steps | case | Initial | Rule | DQN | Oracle | DQN-Initial | Oracle-DQN |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['max_steps']} | {row['case_id']} | "
            f"{row['full_initial_dice']:.6f} | {row['full_rule_dice']:.6f} | "
            f"{row['full_rl_dice']:.6f} | {row['full_oracle_dice']:.6f} | "
            f"{row['full_rl_minus_initial']:.6f} | {row['full_oracle_gap']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Cached Abnormal-slice Dice",
            "",
            "| max_steps | case | Current | Rule | DQN | Oracle | DQN-Current | Oracle-DQN |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['max_steps']} | {row['case_id']} | "
            f"{row['cached_current_dice']:.6f} | {row['cached_rule_dice']:.6f} | "
            f"{row['cached_rl_dice']:.6f} | {row['cached_oracle_dice']:.6f} | "
            f"{row['cached_rl_minus_current']:.6f} | {row['cached_oracle_gap']:.6f} |"
        )
    lines.extend(["", "## Findings", ""])
    lines.extend(f"- {item}" for item in summary["main_findings"])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `step_ablation_table.csv`: full ablation table.",
            "- `step_ablation_summary.json`: machine-readable summary.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DQN max_steps ablation results.")
    parser.add_argument("--steps3-eval-dir", default=r".\outputs\rl_dqn_liver_multicase_v1_eval")
    parser.add_argument("--steps5-eval-dir", default=r".\outputs\rl_dqn_liver_multicase_steps5_e40_eval")
    parser.add_argument("--steps8-eval-dir", default=r".\outputs\rl_dqn_liver_multicase_steps8_e40_eval")
    parser.add_argument("--output-dir", default=r".\outputs\rl_dqn_liver_multicase_step_ablation")
    return parser


if __name__ == "__main__":
    result = summarize(build_parser().parse_args())
    print(json.dumps({"output_dir": build_parser().parse_args().output_dir, "main_findings": result["main_findings"]}, ensure_ascii=False, indent=2))

