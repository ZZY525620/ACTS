from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def load_cache(cache_dir: Path) -> dict[int, list[dict[str, Any]]]:
    metadata_path = cache_dir / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_slice: dict[int, list[dict[str, Any]]] = {}
    for sample in metadata["samples"]:
        by_slice[int(sample["slice"])] = list(sample["candidates"])
    return by_slice


def candidate_values(candidates: list[dict[str, Any]], mode: str) -> list[float]:
    values: list[float] = []
    for cand in candidates:
        name = str(cand.get("name", ""))
        area = int(cand.get("area", 0))
        is_sam = bool(cand.get("is_sam_candidate", False))
        if mode == "non_empty" and (area <= 0 or name in {"empty_mask", "stop_direction"}):
            continue
        if mode == "sam_only" and not is_sam:
            continue
        values.append(float(cand["dice_to_gt"]))
    return values


def slice_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "num_candidates": 0,
            "mean": 0.0,
            "median": 0.0,
            "oracle": 0.0,
            "top3_mean": 0.0,
            "good_ratio_08": 0.0,
            "excellent_ratio_09": 0.0,
            "bad_ratio_05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    sorted_values = sorted(values, reverse=True)
    arr = np.asarray(values, dtype=np.float32)
    return {
        "num_candidates": int(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "oracle": float(max(values)),
        "top3_mean": float(mean(sorted_values[: min(3, len(sorted_values))])),
        "good_ratio_08": float(np.mean(arr >= 0.8)),
        "excellent_ratio_09": float(np.mean(arr >= 0.9)),
        "bad_ratio_05": float(np.mean(arr < 0.5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def summarize_slices(cache: dict[int, list[dict[str, Any]]], slices: list[int], mode: str) -> dict[str, float]:
    rows = [slice_stats(candidate_values(cache[s], mode)) for s in slices]
    if not rows:
        return empty_summary()
    keys = rows[0].keys()
    return {key: float(mean(float(row[key]) for row in rows)) for key in keys}


def empty_summary() -> dict[str, float]:
    return {
        "num_candidates": 0.0,
        "mean": 0.0,
        "median": 0.0,
        "oracle": 0.0,
        "top3_mean": 0.0,
        "good_ratio_08": 0.0,
        "excellent_ratio_09": 0.0,
        "bad_ratio_05": 0.0,
        "p25": 0.0,
        "p75": 0.0,
    }


def cache_dir(root: Path, case_id: str, pattern: str) -> Path:
    return root / pattern.format(case_id=case_id)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    modes = ["all", "non_empty", "sam_only"]
    for case_id in args.case_ids:
        frozen = load_cache(cache_dir(Path(args.frozen_cache_root), case_id, args.frozen_pattern))
        tuned = load_cache(cache_dir(Path(args.tuned_cache_root), case_id, args.tuned_pattern))
        slice_sets = {
            "native_frozen": sorted(frozen.keys()),
            "native_tuned": sorted(tuned.keys()),
            "common": sorted(set(frozen.keys()) & set(tuned.keys())),
        }
        for mode in modes:
            for split_name, slices in slice_sets.items():
                if split_name == "native_frozen":
                    summary = summarize_slices(frozen, slices, mode)
                    model = "frozen"
                elif split_name == "native_tuned":
                    summary = summarize_slices(tuned, slices, mode)
                    model = "tuned"
                else:
                    frozen_summary = summarize_slices(frozen, slices, mode)
                    tuned_summary = summarize_slices(tuned, slices, mode)
                    rows.append(row(case_id, "frozen", split_name, mode, len(slices), frozen_summary))
                    rows.append(row(case_id, "tuned", split_name, mode, len(slices), tuned_summary))
                    continue
                rows.append(row(case_id, model, split_name, mode, len(slices), summary))

    mean_rows = []
    for model in ["frozen", "tuned"]:
        for split_name in ["native_frozen", "native_tuned", "common"]:
            for mode in modes:
                selected = [r for r in rows if r["model"] == model and r["slice_set"] == split_name and r["candidate_mode"] == mode]
                if not selected:
                    continue
                mean_rows.append(mean_row(model, split_name, mode, selected))
    all_rows = rows + mean_rows
    write_csv(output_dir / "candidate_pool_quality_by_case.csv", all_rows)

    compact = [
        r
        for r in all_rows
        if r["case_id"] == "mean" and r["candidate_mode"] == "non_empty" and r["slice_set"] in {"native_frozen", "native_tuned", "common"}
    ]
    write_csv(output_dir / "candidate_pool_quality_summary.csv", compact)
    write_readme(output_dir / "README_CANDIDATE_POOL_QUALITY.md", compact, args)
    result = {"output_dir": str(output_dir), "summary": compact}
    (output_dir / "candidate_pool_quality_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def row(case_id: str, model: str, slice_set: str, mode: str, num_slices: int, summary: dict[str, float]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "model": model,
        "slice_set": slice_set,
        "candidate_mode": mode,
        "num_slices": int(num_slices),
        **summary,
    }


def mean_row(model: str, slice_set: str, mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "num_slices",
        "num_candidates",
        "mean",
        "median",
        "oracle",
        "top3_mean",
        "good_ratio_08",
        "excellent_ratio_09",
        "bad_ratio_05",
        "p25",
        "p75",
    ]
    return {
        "case_id": "mean",
        "model": model,
        "slice_set": slice_set,
        "candidate_mode": mode,
        **{key: float(mean(float(r[key]) for r in rows)) for key in numeric_keys},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Candidate Pool Quality Analysis",
        "",
        "This analysis compares candidate-mask Dice distributions before and after SAM fine-tuning.",
        "",
        f"Frozen cache root: `{args.frozen_cache_root}`",
        f"Tuned cache root: `{args.tuned_cache_root}`",
        "",
        "Main columns:",
        "",
        "- `oracle`: best candidate Dice per slice.",
        "- `mean` / `median`: average candidate-pool quality per slice.",
        "- `top3_mean`: average Dice of the best three candidates per slice.",
        "- `good_ratio_08`: ratio of candidates with Dice >= 0.8.",
        "- `bad_ratio_05`: ratio of candidates with Dice < 0.5.",
        "",
        "Important: `common` compares only slices cached by both runs. This is the fairest slice-level comparison.",
        "",
        "## Mean Summary, Non-empty Candidates",
        "",
        "| Model | Slice set | Slices | Mean | Median | Oracle | Top3 | Dice>=0.8 | Dice<0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['slice_set']} | {float(r['num_slices']):.1f} | "
            f"{float(r['mean']):.4f} | {float(r['median']):.4f} | {float(r['oracle']):.4f} | "
            f"{float(r['top3_mean']):.4f} | {float(r['good_ratio_08']):.4f} | {float(r['bad_ratio_05']):.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare frozen vs fine-tuned SAM candidate-pool quality.")
    parser.add_argument("--case-ids", nargs="+", default=[f"{i:04d}" for i in range(11, 16)])
    parser.add_argument(
        "--frozen-cache-root",
        default=r".\outputs\experiments\liver_dqn_softwin_20260708\01_caches_fpaware_m20",
    )
    parser.add_argument(
        "--tuned-cache-root",
        default=r".\outputs\experiments\liver_dqn_softwin_20260708\11_sam_finetune_aug_weighted_pseudo_e4\03_finetuned_sam_caches_fpaware",
    )
    parser.add_argument("--frozen-pattern", default="rl_cache_case{case_id}_liver_fpaware_m20_softwin")
    parser.add_argument("--tuned-pattern", default="rl_cache_case{case_id}_liver_fpaware_samft_augweight_e4")
    parser.add_argument(
        "--output-dir",
        default=r".\outputs\experiments\liver_dqn_softwin_20260708\11_sam_finetune_aug_weighted_pseudo_e4\07_candidate_pool_quality",
    )
    return parser


if __name__ == "__main__":
    print(json.dumps(analyze(build_parser().parse_args()), ensure_ascii=False, indent=2))

