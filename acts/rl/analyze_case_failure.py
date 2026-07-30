from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acts.data.dataset import load_flare_case, slice_range
from acts.data.nifti import load_nii
from acts.data.preprocess import to_uint8, window_ct
from acts.evaluation.metrics import dice
from acts.evaluation.visualize import overlay_mask


def analyze_case(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    panels_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    case = load_flare_case(args.data_dir, args.case_id, liver_label=args.liver_label)
    image_uint8 = to_uint8(window_ct(case.image.array.astype(np.float32), args.window_min, args.window_max))
    gt = case.liver_mask.astype(np.uint8)

    baseline_dir = Path(args.baseline_dir)
    rl_dir = Path(args.rl_dir)
    initial = _load_mask(baseline_dir / f"case_{args.case_id}_initial_liver_mask.nii.gz")
    rule = _load_mask(baseline_dir / f"case_{args.case_id}_corrected_liver_mask.nii.gz")
    rl_oracle_path = rl_dir / f"case_{args.case_id}_oracle_liver_mask.nii.gz"
    oracle_path = rl_oracle_path if rl_oracle_path.exists() else baseline_dir / f"case_{args.case_id}_oracle_liver_mask.nii.gz"
    oracle = _load_mask(oracle_path)
    rl = _load_mask(rl_dir / f"case_{args.case_id}_rl_liver_mask.nii.gz")

    action_reports = json.loads((rl_dir / "rl_action_report.json").read_text(encoding="utf-8"))
    action_by_slice = {int(item["slice"]): item for item in action_reports}
    cached_slices = sorted(action_by_slice)

    rows = []
    for z in range(gt.shape[2]):
        action = action_by_slice.get(z)
        row = {
            "slice": z,
            "cached": action is not None,
            "gt_area": int(gt[:, :, z].sum()),
            "initial_area": int(initial[:, :, z].sum()),
            "rule_area": int(rule[:, :, z].sum()),
            "rl_area": int(rl[:, :, z].sum()),
            "oracle_area": int(oracle[:, :, z].sum()),
            "initial_dice": dice(initial[:, :, z], gt[:, :, z]),
            "rule_dice": dice(rule[:, :, z], gt[:, :, z]),
            "rl_dice": dice(rl[:, :, z], gt[:, :, z]),
            "oracle_dice": dice(oracle[:, :, z], gt[:, :, z]),
            "rl_minus_initial": dice(rl[:, :, z], gt[:, :, z]) - dice(initial[:, :, z], gt[:, :, z]),
            "oracle_minus_rl": dice(oracle[:, :, z], gt[:, :, z]) - dice(rl[:, :, z], gt[:, :, z]),
            "rl_candidate_name": "",
            "oracle_candidate_name": "",
            "rl_dice_cache_space": "",
            "oracle_dice_cache_space": "",
            "accepted_steps": 0,
            "last_action": "",
        }
        if action is not None:
            row.update(
                {
                    "rl_candidate_name": action.get("rl_candidate_name", ""),
                    "oracle_candidate_name": action.get("oracle_candidate_name", ""),
                    "rl_dice_cache_space": action.get("rl_dice_cache_space", ""),
                    "oracle_dice_cache_space": action.get("oracle_candidate_dice", ""),
                    "accepted_steps": sum(1 for step in action.get("steps", []) if step.get("info", {}).get("accepted")),
                    "last_action": action.get("steps", [{}])[-1].get("action_name", "") if action.get("steps") else "",
                }
            )
        rows.append(row)

    _write_csv(output_dir / "case_0003_slice_error_table.csv", rows)
    _save_curves(output_dir / "case_0003_dice_curves.png", rows)
    _save_area_curves(output_dir / "case_0003_area_curves.png", rows)

    selected = _select_key_slices(rows, cached_slices)
    for group_name, slices in selected.items():
        for z in slices:
            _save_panel(
                panels_dir / f"{group_name}_slice_{z:03d}.png",
                z,
                image_uint8[:, :, z],
                gt[:, :, z],
                initial[:, :, z],
                rule[:, :, z],
                rl[:, :, z],
                oracle[:, :, z],
                rows[z],
            )

    summary = _build_summary(args.case_id, gt, initial, rule, rl, oracle, rows, selected)
    (output_dir / "case_0003_failure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(output_dir / "README_CASE0003_FAILURE_ANALYSIS.md", summary)
    return summary


def _load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return (load_nii(path).array > 0).astype(np.uint8)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_key_slices(rows: list[dict[str, Any]], cached_slices: list[int]) -> dict[str, list[int]]:
    gt_rows = [r for r in rows if r["gt_area"] > 0]
    empty_fp_rows = [r for r in rows if r["gt_area"] == 0 and r["initial_area"] > 0]
    cached_rows = [rows[z] for z in cached_slices]

    return {
        "worst_initial_gt": [r["slice"] for r in sorted(gt_rows, key=lambda r: r["initial_dice"])[:6]],
        "largest_initial_fp_empty_gt": [r["slice"] for r in sorted(empty_fp_rows, key=lambda r: r["initial_area"], reverse=True)[:6]],
        "best_dqn_improvement": [r["slice"] for r in sorted(cached_rows, key=lambda r: r["rl_minus_initial"], reverse=True)[:6]],
        "largest_oracle_gap": [r["slice"] for r in sorted(cached_rows, key=lambda r: r["oracle_minus_rl"], reverse=True)[:6]],
    }


def _build_summary(
    case_id: str,
    gt: np.ndarray,
    initial: np.ndarray,
    rule: np.ndarray,
    rl: np.ndarray,
    oracle: np.ndarray,
    rows: list[dict[str, Any]],
    selected: dict[str, list[int]],
) -> dict[str, Any]:
    gt_start, gt_end = slice_range(gt)
    pred_ranges = {
        "initial": _nonzero_range(initial),
        "rule": _nonzero_range(rule),
        "rl": _nonzero_range(rl),
        "oracle": _nonzero_range(oracle),
    }
    empty_gt_fp_slices = [r for r in rows if r["gt_area"] == 0 and r["initial_area"] > 0]
    gt_slices = [r for r in rows if r["gt_area"] > 0]
    cached_rows = [r for r in rows if r["cached"]]
    improved_cached = [r for r in cached_rows if r["rl_minus_initial"] > 1e-6]
    worsened_cached = [r for r in cached_rows if r["rl_minus_initial"] < -1e-6]
    unchanged_cached = [r for r in cached_rows if abs(r["rl_minus_initial"]) <= 1e-6]

    return {
        "case_id": case_id,
        "gt_liver_slice_range": [gt_start, gt_end],
        "prediction_slice_ranges": pred_ranges,
        "full_volume_dice": {
            "initial": dice(initial, gt),
            "rule": dice(rule, gt),
            "rl": dice(rl, gt),
            "oracle": dice(oracle, gt),
        },
        "global_voxel_counts": {
            "gt_voxels": int(gt.sum()),
            "initial_voxels": int(initial.sum()),
            "rule_voxels": int(rule.sum()),
            "rl_voxels": int(rl.sum()),
            "oracle_voxels": int(oracle.sum()),
            "initial_false_positive_voxels": int(np.logical_and(initial > 0, gt == 0).sum()),
            "rl_false_positive_voxels": int(np.logical_and(rl > 0, gt == 0).sum()),
            "initial_false_negative_voxels": int(np.logical_and(initial == 0, gt > 0).sum()),
            "rl_false_negative_voxels": int(np.logical_and(rl == 0, gt > 0).sum()),
        },
        "slice_error_counts": {
            "num_slices": len(rows),
            "num_gt_liver_slices": len(gt_slices),
            "num_initial_fp_empty_gt_slices": len(empty_gt_fp_slices),
            "num_cached_slices": len(cached_rows),
            "num_cached_improved_by_dqn": len(improved_cached),
            "num_cached_worsened_by_dqn": len(worsened_cached),
            "num_cached_unchanged_by_dqn": len(unchanged_cached),
        },
        "mean_slice_dice_on_gt_slices": {
            "initial": _mean([r["initial_dice"] for r in gt_slices]),
            "rule": _mean([r["rule_dice"] for r in gt_slices]),
            "rl": _mean([r["rl_dice"] for r in gt_slices]),
            "oracle": _mean([r["oracle_dice"] for r in gt_slices]),
        },
        "mean_slice_dice_on_cached_slices": {
            "initial": _mean([r["initial_dice"] for r in cached_rows]),
            "rule": _mean([r["rule_dice"] for r in cached_rows]),
            "rl": _mean([r["rl_dice"] for r in cached_rows]),
            "oracle": _mean([r["oracle_dice"] for r in cached_rows]),
        },
        "selected_slices_for_visual_review": selected,
        "main_diagnosis": [
            "Initial prediction extends far outside the GT liver z-range, creating many false-positive empty-GT slices.",
            "DQN improves part of the cached abnormal slices, but most cached slices remain unchanged or still far from Oracle.",
            "Oracle is much higher on cached slices, so candidate generation has useful masks; the current policy does not reliably choose them.",
        ],
    }


def _nonzero_range(mask: np.ndarray) -> list[int]:
    areas = mask.sum(axis=(0, 1))
    idx = np.where(areas > 0)[0]
    if idx.size == 0:
        return [-1, -1]
    return [int(idx[0]), int(idx[-1])]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _save_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    w, h = 1200, 420
    margin = 48
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "case0003 slice-wise Dice", fill=(0, 0, 0))
    _draw_axes(draw, w, h, margin, max_x=len(rows) - 1, max_y=1.0)
    series = [
        ("initial", (220, 70, 70)),
        ("rule", (60, 140, 255)),
        ("rl", (255, 160, 40)),
        ("oracle", (160, 80, 210)),
    ]
    for name, color in series:
        points = []
        for r in rows:
            x = margin + r["slice"] / max(len(rows) - 1, 1) * (w - 2 * margin)
            y = h - margin - float(r[f"{name}_dice"]) * (h - 2 * margin)
            points.append((x, y))
        draw.line(points, fill=color, width=3)
        lx = margin + 12 + series.index((name, color)) * 170
        draw.line((lx, h - 24, lx + 32, h - 24), fill=color, width=4)
        draw.text((lx + 40, h - 32), name, fill=(0, 0, 0))
    canvas.save(path)


def _save_area_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    w, h = 1200, 420
    margin = 48
    max_area = max(max(r["gt_area"], r["initial_area"], r["rl_area"], r["oracle_area"]) for r in rows)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "case0003 slice area curves", fill=(0, 0, 0))
    _draw_axes(draw, w, h, margin, max_x=len(rows) - 1, max_y=max_area)
    series = [
        ("gt", (40, 200, 90)),
        ("initial", (220, 70, 70)),
        ("rl", (255, 160, 40)),
        ("oracle", (160, 80, 210)),
    ]
    for name, color in series:
        points = []
        for r in rows:
            x = margin + r["slice"] / max(len(rows) - 1, 1) * (w - 2 * margin)
            y = h - margin - float(r[f"{name}_area"]) / max(max_area, 1) * (h - 2 * margin)
            points.append((x, y))
        draw.line(points, fill=color, width=3)
        lx = margin + 12 + series.index((name, color)) * 170
        draw.line((lx, h - 24, lx + 32, h - 24), fill=color, width=4)
        draw.text((lx + 40, h - 32), name, fill=(0, 0, 0))
    canvas.save(path)


def _draw_axes(draw: ImageDraw.ImageDraw, w: int, h: int, margin: int, max_x: int, max_y: float) -> None:
    draw.line((margin, h - margin, w - margin, h - margin), fill=(0, 0, 0), width=1)
    draw.line((margin, margin, margin, h - margin), fill=(0, 0, 0), width=1)
    draw.text((margin, h - margin + 8), "0", fill=(0, 0, 0))
    draw.text((w - margin - 24, h - margin + 8), str(max_x), fill=(0, 0, 0))
    draw.text((8, margin - 8), f"{max_y:.2f}", fill=(0, 0, 0))


def _save_panel(
    path: Path,
    z: int,
    image: np.ndarray,
    gt: np.ndarray,
    initial: np.ndarray,
    rule: np.ndarray,
    rl: np.ndarray,
    oracle: np.ndarray,
    row: dict[str, Any],
) -> None:
    panels = [
        ("image", Image.fromarray(image).convert("RGB")),
        (f"gt a={row['gt_area']}", overlay_mask(image, gt, (40, 220, 90))),
        (f"initial d={row['initial_dice']:.2f}", overlay_mask(image, initial, (240, 80, 70))),
        (f"rule d={row['rule_dice']:.2f}", overlay_mask(image, rule, (60, 140, 255))),
        (f"dqn d={row['rl_dice']:.2f}", overlay_mask(image, rl, (255, 180, 40))),
        (f"oracle d={row['oracle_dice']:.2f}", overlay_mask(image, oracle, (180, 80, 230))),
    ]
    w, h = panels[0][1].size
    canvas = Image.new("RGB", (w * len(panels), h + 52), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"case0003 slice {z} | cached={row['cached']} | "
        f"DQN={row['rl_candidate_name']} | Oracle={row['oracle_candidate_name']}"
    )
    draw.text((8, 6), title[:180], fill=(0, 0, 0))
    for i, (label, panel) in enumerate(panels):
        x = i * w
        canvas.paste(panel, (x, 52))
        draw.text((x + 8, 32), label, fill=(0, 0, 0))
    canvas.save(path)


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    full = summary["full_volume_dice"]
    counts = summary["slice_error_counts"]
    vox = summary["global_voxel_counts"]
    lines = [
        "# Case 0003 Failure Analysis",
        "",
        f"GT liver slice range: {summary['gt_liver_slice_range']}",
        f"Initial predicted slice range: {summary['prediction_slice_ranges']['initial']}",
        f"DQN predicted slice range: {summary['prediction_slice_ranges']['rl']}",
        "",
        "## Full-volume Dice",
        "",
        f"- Initial: {full['initial']:.6f}",
        f"- Rule: {full['rule']:.6f}",
        f"- DQN: {full['rl']:.6f}",
        f"- Oracle: {full['oracle']:.6f}",
        "",
        "## Key Counts",
        "",
        f"- GT liver slices: {counts['num_gt_liver_slices']}",
        f"- Empty-GT slices with Initial false positives: {counts['num_initial_fp_empty_gt_slices']}",
        f"- Cached abnormal slices: {counts['num_cached_slices']}",
        f"- Cached slices improved by DQN: {counts['num_cached_improved_by_dqn']}",
        f"- Cached slices worsened by DQN: {counts['num_cached_worsened_by_dqn']}",
        f"- Cached slices unchanged by DQN: {counts['num_cached_unchanged_by_dqn']}",
        "",
        "## Voxel Error",
        "",
        f"- Initial false-positive voxels: {vox['initial_false_positive_voxels']}",
        f"- DQN false-positive voxels: {vox['rl_false_positive_voxels']}",
        f"- Initial false-negative voxels: {vox['initial_false_negative_voxels']}",
        f"- DQN false-negative voxels: {vox['rl_false_negative_voxels']}",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["main_diagnosis"])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `case_0003_slice_error_table.csv`: per-slice dice/area/action table.",
            "- `case_0003_dice_curves.png`: slice-wise Dice curves.",
            "- `case_0003_area_curves.png`: GT/prediction area curves.",
            "- `panels/`: selected slices for visual review.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze failure modes for one FLARE liver case.")
    parser.add_argument("--data-dir", default=r"data")
    parser.add_argument("--case-id", default="0003")
    parser.add_argument("--liver-label", type=int, default=1)
    parser.add_argument("--baseline-dir", default=r".\outputs\rule_baseline_case0003_liver")
    parser.add_argument("--rl-dir", default=r".\outputs\rl_dqn_liver_multicase_v1_eval\case_0003")
    parser.add_argument("--output-dir", default=r".\outputs\analysis_case0003_failure")
    parser.add_argument("--window-min", type=float, default=-160.0)
    parser.add_argument("--window-max", type=float, default=240.0)
    return parser


if __name__ == "__main__":
    result = analyze_case(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))

