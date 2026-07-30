from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def audit_false_positives(args: argparse.Namespace) -> dict[str, Any]:
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_rows = _read_csv(analysis_dir / "case_0003_slice_error_table.csv")
    action_reports = json.loads(Path(args.action_report).read_text(encoding="utf-8"))
    action_by_slice = {int(item["slice"]): item for item in action_reports}
    cache = json.loads(Path(args.cache_metadata).read_text(encoding="utf-8"))
    cache_by_slice = {int(item["slice"]): item for item in cache["samples"]}

    audit_rows = []
    for row in slice_rows:
        z = int(row["slice"])
        gt_area = int(row["gt_area"])
        initial_area = int(row["initial_area"])
        if gt_area != 0 or initial_area <= args.min_initial_area:
            continue

        cache_sample = cache_by_slice.get(z)
        action = action_by_slice.get(z)
        candidates = cache_sample.get("candidates", []) if cache_sample else []
        empty_candidates = [c for c in candidates if c.get("name") == "empty_mask" or int(c.get("area", -1)) == 0]
        oracle_candidate = None
        if cache_sample is not None:
            oracle_candidate = candidates[int(cache_sample["oracle_candidate_index"])]

        dqn_selected = action.get("rl_candidate_name", "") if action else ""
        oracle_name = oracle_candidate.get("name", "") if oracle_candidate else ""
        empty_best_dice = max((float(c.get("dice_to_gt", 0.0)) for c in empty_candidates), default="")
        empty_best_score = max((float(c.get("no_gt_score", 0.0)) for c in empty_candidates), default="")

        audit_rows.append(
            {
                "slice": z,
                "gt_area": gt_area,
                "initial_area": initial_area,
                "rl_area": int(row["rl_area"]),
                "oracle_area": int(row["oracle_area"]),
                "initial_dice": float(row["initial_dice"]),
                "rl_dice": float(row["rl_dice"]),
                "oracle_dice": float(row["oracle_dice"]),
                "in_cache": cache_sample is not None,
                "has_empty_candidate": bool(empty_candidates),
                "num_empty_or_zero_area_candidates": len(empty_candidates),
                "empty_best_dice": empty_best_dice,
                "empty_best_no_gt_score": empty_best_score,
                "dqn_selected": dqn_selected,
                "dqn_cleared_to_empty": int(row["rl_area"]) == 0,
                "oracle_selected": oracle_name,
                "oracle_cleared_to_empty": int(row["oracle_area"]) == 0,
                "oracle_candidate_dice_cache": "" if oracle_candidate is None else float(oracle_candidate["dice_to_gt"]),
                "dqn_selected_empty_like": _is_empty_like_name(dqn_selected),
                "oracle_selected_empty_like": _is_empty_like_name(oracle_name),
                "last_action": action.get("steps", [{}])[-1].get("action_name", "") if action and action.get("steps") else "",
                "accepted_steps": sum(1 for step in action.get("steps", []) if step.get("info", {}).get("accepted")) if action else 0,
            }
        )

    summary = _summarize(audit_rows)
    _write_csv(output_dir / "case_0003_false_positive_audit.csv", audit_rows)
    (output_dir / "case_0003_false_positive_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(output_dir / "README_FALSE_POSITIVE_AUDIT.md", summary, audit_rows[: args.top_n])
    return summary


def _is_empty_like_name(name: str) -> bool:
    name = str(name).lower()
    return name == "empty_mask" or "negative" in name or "fp_point" in name or "eroded" in name


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cached = [r for r in rows if r["in_cache"]]
    not_cached = [r for r in rows if not r["in_cache"]]
    has_empty = [r for r in cached if r["has_empty_candidate"]]
    dqn_cleared = [r for r in rows if r["dqn_cleared_to_empty"]]
    oracle_cleared = [r for r in rows if r["oracle_cleared_to_empty"]]
    dqn_kept_fp_with_empty = [r for r in has_empty if not r["dqn_cleared_to_empty"]]
    oracle_empty_dqn_not = [r for r in rows if r["oracle_cleared_to_empty"] and not r["dqn_cleared_to_empty"]]
    dqn_kept_count = len(dqn_kept_fp_with_empty)
    not_cached_count = len(not_cached)
    findings = [
        "Many empty-GT slices have large Initial false-positive masks.",
    ]
    if not_cached_count > 0:
        findings.append(
            "Some empty-GT false-positive slices are still outside the candidate cache, so abnormal-slice coverage remains a bottleneck."
        )
    else:
        findings.append("All empty-GT false-positive slices are covered by the candidate cache.")
    if dqn_kept_count > 0:
        findings.append(
            "For cached false-positive slices, empty/zero-area candidates exist but DQN still keeps some false-positive masks."
        )
    else:
        findings.append("For cached false-positive slices with empty candidates, DQN clears all of them to empty masks.")
    findings.append(
        "Next fixes should focus on covering the remaining uncached false-positive slices and then validating the policy on more cases."
    )

    return {
        "num_empty_gt_false_positive_slices": len(rows),
        "num_in_cache": len(cached),
        "num_not_in_cache": len(not_cached),
        "num_cached_with_empty_candidate": len(has_empty),
        "num_dqn_cleared_to_empty": len(dqn_cleared),
        "num_oracle_cleared_to_empty": len(oracle_cleared),
        "num_cached_has_empty_but_dqn_kept_fp": len(dqn_kept_fp_with_empty),
        "num_oracle_empty_but_dqn_not_empty": len(oracle_empty_dqn_not),
        "total_initial_fp_area": int(sum(r["initial_area"] for r in rows)),
        "total_rl_fp_area": int(sum(r["rl_area"] for r in rows)),
        "total_oracle_fp_area": int(sum(r["oracle_area"] for r in rows)),
        "top_initial_fp_slices": [
            {
                "slice": r["slice"],
                "initial_area": r["initial_area"],
                "rl_area": r["rl_area"],
                "oracle_area": r["oracle_area"],
                "in_cache": r["in_cache"],
                "has_empty_candidate": r["has_empty_candidate"],
                "dqn_selected": r["dqn_selected"],
                "oracle_selected": r["oracle_selected"],
            }
            for r in sorted(rows, key=lambda item: item["initial_area"], reverse=True)[:10]
        ],
        "main_findings": findings,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, summary: dict[str, Any], preview_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Case 0003 False-positive Audit",
        "",
        "This audit focuses on slices where GT has no liver but Initial predicts liver.",
        "",
        "## Summary",
        "",
        f"- Empty-GT false-positive slices: {summary['num_empty_gt_false_positive_slices']}",
        f"- In candidate cache: {summary['num_in_cache']}",
        f"- Not in candidate cache: {summary['num_not_in_cache']}",
        f"- Cached with empty/zero-area candidate: {summary['num_cached_with_empty_candidate']}",
        f"- DQN cleared to empty: {summary['num_dqn_cleared_to_empty']}",
        f"- Oracle cleared to empty: {summary['num_oracle_cleared_to_empty']}",
        f"- Has empty candidate but DQN kept false positive: {summary['num_cached_has_empty_but_dqn_kept_fp']}",
        f"- Oracle empty but DQN not empty: {summary['num_oracle_empty_but_dqn_not_empty']}",
        "",
        "## False-positive Area",
        "",
        f"- Initial total FP area on these slices: {summary['total_initial_fp_area']}",
        f"- DQN total FP area on these slices: {summary['total_rl_fp_area']}",
        f"- Oracle total FP area on these slices: {summary['total_oracle_fp_area']}",
        "",
        "## Top Initial FP Slices",
        "",
        "| Slice | Initial area | DQN area | Oracle area | In cache | Empty cand | DQN selected | Oracle selected |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["top_initial_fp_slices"]:
        lines.append(
            f"| {row['slice']} | {row['initial_area']} | {row['rl_area']} | {row['oracle_area']} | "
            f"{row['in_cache']} | {row['has_empty_candidate']} | {row['dqn_selected']} | {row['oracle_selected']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["main_findings"])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `case_0003_false_positive_audit.csv`: per-slice audit table.",
            "- `case_0003_false_positive_audit_summary.json`: machine-readable summary.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit empty-GT false-positive slices for case0003.")
    parser.add_argument("--analysis-dir", default=r".\outputs\analysis_case0003_failure")
    parser.add_argument("--action-report", default=r".\outputs\rl_dqn_liver_multicase_v1_eval\case_0003\rl_action_report.json")
    parser.add_argument("--cache-metadata", default=r".\outputs\rl_cache_case0003_liver\cache_metadata.json")
    parser.add_argument("--output-dir", default=r".\outputs\analysis_case0003_failure\false_positive_audit")
    parser.add_argument("--min-initial-area", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=10)
    return parser


if __name__ == "__main__":
    result = audit_false_positives(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))

