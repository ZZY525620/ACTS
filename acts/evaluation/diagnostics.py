from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_sequence_diagnostics(
    output_dir: str | Path,
    gt: np.ndarray,
    initial: np.ndarray,
    corrected: np.ndarray,
    oracle: np.ndarray,
    metrics: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_area = _area_curve(gt)
    initial_area = _area_curve(initial)
    corrected_area = _area_curve(corrected)
    oracle_area = _area_curve(oracle)
    liver_slices = np.where(gt_area > 0)[0]
    liver_range = (int(liver_slices[0]), int(liver_slices[-1])) if liver_slices.size else (None, None)

    _save_area_curve(
        output_dir / "diagnostic_area_curves.png",
        gt_area,
        initial_area,
        corrected_area,
        oracle_area,
        liver_range,
        int(metrics["ref_index"]),
    )
    _save_state_source_curve(
        output_dir / "diagnostic_state_source_curves.png",
        metrics.get("candidate_report", []),
        metrics.get("propagation_report", []),
        depth=gt.shape[2],
        ref_index=int(metrics["ref_index"]),
    )

    summary = {
        "liver_slice_range": list(liver_range),
        "num_no_liver_slices": int((gt_area == 0).sum()),
        "initial_false_positive": _false_positive_summary(gt_area, initial_area),
        "corrected_false_positive": _false_positive_summary(gt_area, corrected_area),
        "oracle_false_positive": _false_positive_summary(gt_area, oracle_area),
        "region_evaluation": _region_evaluation(gt, initial, corrected, oracle),
        "area_curve_files": [
            "diagnostic_area_curves.png",
            "diagnostic_state_source_curves.png",
        ],
    }
    with (output_dir / "diagnostic_false_positive_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _area_curve(mask_volume: np.ndarray) -> np.ndarray:
    return (mask_volume > 0).sum(axis=(0, 1)).astype(np.float32)


def _false_positive_summary(gt_area: np.ndarray, pred_area: np.ndarray) -> dict:
    no_liver = gt_area == 0
    fp_slices = np.where(no_liver & (pred_area > 0))[0]
    fp_area = pred_area[fp_slices]
    return {
        "num_non_empty_no_liver_slices": int(fp_slices.size),
        "false_positive_area_sum": int(fp_area.sum()) if fp_area.size else 0,
        "false_positive_area_mean": float(fp_area.mean()) if fp_area.size else 0.0,
        "false_positive_area_max": int(fp_area.max()) if fp_area.size else 0,
        "false_positive_slices": [int(x) for x in fp_slices.tolist()],
    }


def _region_evaluation(gt: np.ndarray, initial: np.ndarray, corrected: np.ndarray, oracle: np.ndarray) -> dict:
    gt_area = _area_curve(gt)
    fg_slices = np.where(gt_area > 0)[0]
    no_target_slices = np.where(gt_area == 0)[0]
    if fg_slices.size == 0:
        groups = {"target_body": [], "target_edge": [], "target_transition": [], "no_target": no_target_slices.tolist()}
    else:
        fg_area = gt_area[fg_slices]
        body_thr = float(np.quantile(fg_area, 0.60))
        edge_thr = float(np.quantile(fg_area, 0.25))
        body = fg_slices[fg_area >= body_thr]
        edge = fg_slices[fg_area <= edge_thr]
        transition = np.asarray([x for x in fg_slices.tolist() if x not in set(body.tolist()) and x not in set(edge.tolist())])
        groups = {
            "target_body": body.tolist(),
            "target_edge": edge.tolist(),
            "target_transition": transition.tolist(),
            "no_target": no_target_slices.tolist(),
        }

    return {
        name: {
            "num_slices": len(indices),
            "initial_dice": _dice_on_slices(initial, gt, indices),
            "corrected_dice": _dice_on_slices(corrected, gt, indices),
            "oracle_dice": _dice_on_slices(oracle, gt, indices),
        }
        for name, indices in groups.items()
    }


def _dice_on_slices(pred: np.ndarray, gt: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    pred_region = pred[:, :, indices] > 0
    gt_region = gt[:, :, indices] > 0
    intersection = float(np.logical_and(pred_region, gt_region).sum())
    denom = float(pred_region.sum() + gt_region.sum())
    if denom == 0:
        return 1.0
    return float((2.0 * intersection + 1e-6) / (denom + 1e-6))


def _save_area_curve(
    path: Path,
    gt_area: np.ndarray,
    initial_area: np.ndarray,
    corrected_area: np.ndarray,
    oracle_area: np.ndarray,
    liver_range: tuple[int | None, int | None],
    ref_index: int,
) -> None:
    x = np.arange(gt_area.shape[0])
    plt.figure(figsize=(12, 5))
    plt.plot(x, gt_area, label="GT liver area", linewidth=2.2, color="#2ca02c")
    plt.plot(x, initial_area, label="initial area", linewidth=1.6, color="#d62728")
    plt.plot(x, corrected_area, label="corrected area", linewidth=1.6, color="#1f77b4")
    plt.plot(x, oracle_area, label="oracle candidate area", linewidth=1.4, color="#9467bd", linestyle="--")
    if liver_range[0] is not None:
        plt.axvspan(liver_range[0], liver_range[1], color="#2ca02c", alpha=0.08, label="GT liver range")
    plt.axvline(ref_index, color="#111111", linestyle=":", linewidth=1.5, label="reference slice")
    plt.xlabel("slice index")
    plt.ylabel("foreground area")
    plt.title("Sequence Area Curves")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_state_source_curve(
    path: Path,
    candidate_report: list[dict],
    propagation_report: list[dict],
    depth: int,
    ref_index: int,
) -> None:
    state_to_id = {
        "disappearing": 0,
        "shrinking": 1,
        "stable": 2,
        "expanding": 3,
        "drift": 4,
        "unreliable": 5,
    }
    id_to_state = {value: key for key, value in state_to_id.items()}

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    if candidate_report:
        xs = [int(item["slice"]) for item in candidate_report]
        ys = [state_to_id.get(str(item.get("state", "unreliable")), 5) for item in candidate_report]
        accepted = [bool(item.get("accepted", False)) for item in candidate_report]
        colors = ["#1f77b4" if ok else "#d62728" for ok in accepted]
        ax1.scatter(xs, ys, c=colors, s=48)
    ax1.axvline(ref_index, color="#111111", linestyle=":", linewidth=1.2)
    ax1.set_xlim(0, depth - 1)
    ax1.set_yticks(sorted(id_to_state))
    ax1.set_yticklabels([id_to_state[i] for i in sorted(id_to_state)])
    ax1.set_ylabel("state")
    ax1.set_title("State Decisions on Corrected Slices")

    ax2 = plt.subplot(2, 1, 2)
    if propagation_report:
        xs = [int(item["slice"]) for item in propagation_report]
        sources = [int(item["source_slice"]) for item in propagation_report]
        reliable = [bool(item["reliable"]) for item in propagation_report]
        colors = ["#2ca02c" if ok else "#ff7f0e" for ok in reliable]
        ax2.scatter(xs, sources, c=colors, s=18)
    ax2.axvline(ref_index, color="#111111", linestyle=":", linewidth=1.2)
    ax2.set_xlim(0, depth - 1)
    ax2.set_xlabel("slice index")
    ax2.set_ylabel("prompt source slice")
    ax2.set_title("Propagation Source Slice and Reliability")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

