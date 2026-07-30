from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acts.data.dataset import choose_reference_slice, load_flare_case, slice_range
from acts.data.nifti import save_nii_like
from acts.data.preprocess import preprocess_ct_for_sam, resize_volume_xy, restore_mask_volume_xy
from acts.evaluation.diagnostics import save_sequence_diagnostics
from acts.evaluation.metrics import dice, evaluate_volume
from acts.evaluation.visualize import save_candidate_grid, save_comparison_panel
from acts.prompts.prompt_from_mask import dilate, erode, interpolate_mask, mask_to_core_point
from acts.prompts.prompt_pool import generate_prompt_pool
from acts.sam.sam_tool import SegmentAnythingSAMTool
from acts.sequence.anomaly import compute_anomaly_scores, select_topk_slices
from acts.sequence.propagate import propagate_sequence
from acts.sequence.score import (
    no_gt_score,
    select_by_no_gt_score,
    select_by_state_aware_score,
    state_aware_no_gt_score,
)
from acts.sequence.state import estimate_sequence_state


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case = load_flare_case(args.data_dir, args.case_id, liver_label=args.liver_label)
    image = case.image.array.astype(np.float32)
    gt = case.liver_mask.astype(np.uint8)
    preprocessed = preprocess_ct_for_sam(image, args.window_min, args.window_max, args.sam_size)
    image_uint8 = preprocessed.image_uint8
    sam_images_rgb = preprocessed.sam_images_rgb
    sam_images_gray = sam_images_rgb[..., 0]
    gt_sam = resize_volume_xy(gt, size=args.sam_size, nearest=True).astype(np.uint8)

    ref_index = args.ref_index if args.ref_index is not None else choose_reference_slice(gt_sam)
    liver_start, liver_end = slice_range(gt_sam)

    if args.model_path is None:
        raise ValueError("--model-path is required for real SAM inference.")
    sam_tool = SegmentAnythingSAMTool(args.model_path, model_type=args.model_type, device=args.device)
    sam_tool.set_slice_index(ref_index)
    sam_tool.set_image(sam_images_rgb[:, :, ref_index, :])
    ref_box = sam_tool.reference_box(gt_sam[:, :, ref_index])
    ref_point = mask_to_core_point(gt_sam[:, :, ref_index])
    ref_mask, ref_prompt_report = select_reference_seed_mask(
        sam_tool,
        gt_sam[:, :, ref_index],
        ref_box,
        ref_point,
        selection_mode=args.reference_seed_selection,
    )

    initial_masks, propagation_report = propagate_sequence(
        sam_images_rgb,
        ref_index,
        ref_mask,
        sam_tool,
        use_reliable_masks=args.use_reliable_masks,
        return_report=True,
    )
    corrected_masks = initial_masks.copy()
    anomaly_scores = compute_anomaly_scores(sam_images_gray, initial_masks)
    k = args.topk if args.topk is not None else max(1, int(round(args.topk_ratio * initial_masks.shape[2])))
    anomaly_foreground = initial_masks if args.anomaly_foreground == "prediction" else gt_sam
    abnormal_slices = select_topk_slices(anomaly_scores, k=k, foreground_mask=anomaly_foreground)
    n_slices = initial_masks.shape[2]
    abnormal_slices = [t for t in abnormal_slices if 0 < t < n_slices - 1]

    candidate_report = []
    oracle_masks = initial_masks.copy()
    for t in abnormal_slices:
        state = estimate_sequence_state(
            image=sam_images_gray[:, :, t],
            prev_image=sam_images_gray[:, :, t - 1],
            ref_mask=ref_mask,
            current_mask=corrected_masks[:, :, t],
            prev_mask=corrected_masks[:, :, t - 1],
            next_mask=corrected_masks[:, :, t + 1],
            step_from_ref=abs(t - ref_index),
        )
        prompts = generate_prompt_pool(corrected_masks, t, state=state)
        candidates: list[np.ndarray] = []
        candidate_names: list[str] = []
        sam_scores: list[float] = []
        for prompt in prompts:
            sam_tool.set_slice_index(t)
            sam_tool.set_image(sam_images_rgb[:, :, t, :])
            pred = sam_tool.predict(
                box=prompt.box,
                points=prompt.points,
                point_labels=prompt.point_labels,
                mask_input=prompt.mask_input,
            )
            for i, (mask, score) in enumerate(zip(pred.masks, pred.scores)):
                candidates.append(mask)
                sam_scores.append(float(score))
                candidate_names.append(f"{prompt.name}:{i}")

        add_non_sam_candidates(corrected_masks, t, candidates, candidate_names, sam_scores)
        if not candidates:
            continue

        if args.score_mode == "state":
            best_mask, best_idx, no_gt_scores = select_by_state_aware_score(
                candidates,
                image=sam_images_gray[:, :, t],
                prev_mask=corrected_masks[:, :, t - 1],
                next_mask=corrected_masks[:, :, t + 1],
                sam_scores=sam_scores,
                state=state,
                ref_mask=ref_mask,
            )
            current_score = state_aware_no_gt_score(
                corrected_masks[:, :, t],
                sam_images_gray[:, :, t],
                corrected_masks[:, :, t - 1],
                corrected_masks[:, :, t + 1],
                sam_score=args.current_mask_score,
                state=state,
                ref_mask=ref_mask,
            )
        else:
            best_mask, best_idx, no_gt_scores = select_by_no_gt_score(
                candidates,
                image=sam_images_gray[:, :, t],
                prev_mask=corrected_masks[:, :, t - 1],
                next_mask=corrected_masks[:, :, t + 1],
                sam_scores=sam_scores,
            )
            current_score = no_gt_score(
                corrected_masks[:, :, t],
                sam_images_gray[:, :, t],
                corrected_masks[:, :, t - 1],
                corrected_masks[:, :, t + 1],
                sam_score=args.current_mask_score,
            )
        oracle_idx = int(np.argmax([dice(c, gt_sam[:, :, t]) for c in candidates]))
        current_area_before = int((corrected_masks[:, :, t] > 0).sum())
        selected_area = int((best_mask > 0).sum())
        accepted, acceptance_reason, required_margin = should_accept_candidate(
            current_mask=corrected_masks[:, :, t],
            candidate=best_mask,
            current_score=current_score,
            candidate_score=no_gt_scores[best_idx],
            state=state,
            ref_mask=ref_mask,
            base_margin=args.accept_margin,
        )
        if accepted:
            corrected_masks[:, :, t] = best_mask
        oracle_masks[:, :, t] = candidates[oracle_idx]

        candidate_report.append(
            {
                "slice": int(t),
                "anomaly_score": float(anomaly_scores[t]),
                "num_candidates": len(candidates),
                "selected": candidate_names[best_idx],
                "selected_no_gt_score": float(no_gt_scores[best_idx]),
                "current_no_gt_score": float(current_score),
                "score_gain": float(no_gt_scores[best_idx] - current_score),
                "required_margin": float(required_margin),
                "acceptance_reason": acceptance_reason,
                "accepted": bool(accepted),
                "current_area": current_area_before,
                "selected_area": selected_area,
                "state": state.name,
                "state_features": state.__dict__,
                "initial_dice": dice(initial_masks[:, :, t], gt_sam[:, :, t]),
                "selected_dice": dice(best_mask, gt_sam[:, :, t]),
                "final_dice": dice(corrected_masks[:, :, t], gt_sam[:, :, t]),
                "oracle": candidate_names[oracle_idx],
                "oracle_dice": dice(candidates[oracle_idx], gt_sam[:, :, t]),
            }
        )

        if len(candidate_report) <= args.visualize_top:
            save_candidate_grid(
                output_dir / f"case_{args.case_id}_slice_{t:03d}_candidates.png",
                sam_images_gray[:, :, t],
                candidates,
                candidate_names,
            )

    initial_masks_original = restore_mask_volume_xy(initial_masks, preprocessed.original_size)
    corrected_masks_original = restore_mask_volume_xy(corrected_masks, preprocessed.original_size)
    oracle_masks_original = restore_mask_volume_xy(oracle_masks, preprocessed.original_size)

    metrics = {
        "case_id": args.case_id,
        "liver_label": args.liver_label,
        "shape_xyz": list(image.shape),
        "original_size_hw": list(preprocessed.original_size),
        "sam_input_size_hw": list(preprocessed.sam_input_size),
        "liver_slice_range": [liver_start, liver_end],
        "ref_index": int(ref_index),
        "ref_box": ref_box,
        "ref_point": ref_point,
        "ref_prompt_report": ref_prompt_report,
        "reference_seed_selection": args.reference_seed_selection,
        "ref_box_space": "sam_input",
        "sam_backend": "sam",
        "model_type": args.model_type,
        "model_path": args.model_path,
        "device": args.device,
        "score_mode": args.score_mode,
        "accept_margin": args.accept_margin,
        "current_mask_score": args.current_mask_score,
        "use_reliable_masks": args.use_reliable_masks,
        "anomaly_foreground": args.anomaly_foreground,
        "topk": k,
        "topk_ratio": args.topk_ratio,
        "abnormal_slices": abnormal_slices,
        "propagation_report": [step.__dict__ for step in propagation_report],
        "initial": evaluate_volume(initial_masks_original, gt),
        "corrected": evaluate_volume(corrected_masks_original, gt),
        "oracle_candidate": evaluate_volume(oracle_masks_original, gt),
        "candidate_report": candidate_report,
        "note": "Real SAM backend result.",
    }

    save_nii_like(output_dir / f"case_{args.case_id}_initial_liver_mask.nii.gz", initial_masks_original, case.label)
    save_nii_like(output_dir / f"case_{args.case_id}_corrected_liver_mask.nii.gz", corrected_masks_original, case.label)
    save_nii_like(output_dir / f"case_{args.case_id}_oracle_liver_mask.nii.gz", oracle_masks_original, case.label)
    with (output_dir / f"case_{args.case_id}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_sequence_diagnostics(
        output_dir,
        gt,
        initial_masks_original,
        corrected_masks_original,
        oracle_masks_original,
        metrics,
    )

    key_slices = [ref_index] + abnormal_slices[: args.visualize_top]
    seen = set()
    for t in key_slices:
        if t in seen:
            continue
        seen.add(t)
        save_comparison_panel(
            output_dir / f"case_{args.case_id}_slice_{t:03d}_comparison.png",
            image_uint8[:, :, t],
            gt[:, :, t],
            initial_masks_original[:, :, t],
            corrected_masks_original[:, :, t],
            title=f"case {args.case_id} slice {t}",
        )

    return metrics


def add_non_sam_candidates(
    masks: np.ndarray,
    t: int,
    candidates: list[np.ndarray],
    candidate_names: list[str],
    sam_scores: list[float],
) -> None:
    current = masks[:, :, t].astype(np.uint8)
    prev_mask = masks[:, :, t - 1].astype(np.uint8)
    next_mask = masks[:, :, t + 1].astype(np.uint8)
    interp = interpolate_mask(prev_mask, next_mask)
    non_sam = [
        ("keep_current_mask", current),
        ("empty_mask", np.zeros_like(current, dtype=np.uint8)),
        ("eroded_current_mask", erode(current, 2)),
        ("dilated_current_mask", dilate(current, 1)),
        ("interpolated_mask", interp),
    ]
    for name, mask in non_sam:
        candidates.append(mask.astype(np.uint8))
        candidate_names.append(name)
        sam_scores.append(0.0)


def should_accept_candidate(
    current_mask: np.ndarray,
    candidate: np.ndarray,
    current_score: float,
    candidate_score: float,
    state,
    ref_mask: np.ndarray,
    base_margin: float,
) -> tuple[bool, str, float]:
    current_area = float((current_mask > 0).sum())
    candidate_area = float((candidate > 0).sum())
    ref_area = max(float((ref_mask > 0).sum()), 1.0)
    required_margin = float(base_margin)

    if state.name == "stable" and state.continuity_to_neighbors > 0.90:
        required_margin += 0.03

    clearing_current = current_area > 0.20 * ref_area and candidate_area < 0.02 * ref_area
    no_clear_evidence = state.name != "disappearing" and state.centroid_shift_to_ref < 0.10 and state.area_ratio_to_ref > 0.25
    if clearing_current and no_clear_evidence:
        return False, "reject_clear_without_disappearing_evidence", required_margin

    edge_stable_growth = (
        state.name == "stable"
        and current_area > 0
        and candidate_area / max(current_area, 1.0) > 1.06
        and state.area_ratio_to_ref < 0.95
        and state.centroid_shift_to_ref > 0.10
    )
    if edge_stable_growth:
        return False, "reject_edge_growth_without_evidence", required_margin

    gain = float(candidate_score - current_score)
    if gain > required_margin:
        return True, "score_gain", required_margin
    return False, "insufficient_score_gain", required_margin


def select_reference_seed_mask(
    sam_tool: SegmentAnythingSAMTool,
    reference_gt_mask: np.ndarray,
    ref_box: list[int],
    ref_point: list[int] | None,
    selection_mode: str = "heuristic",
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    prompt_specs = [("ref_box", {"box": ref_box})]
    if ref_point is not None:
        prompt_specs.extend(
            [
                ("ref_box_point", {"box": ref_box, "points": [ref_point], "point_labels": [1]}),
                ("ref_core_point", {"points": [ref_point], "point_labels": [1]}),
            ]
        )

    candidates: list[tuple[str, int, np.ndarray, float, float]] = []
    for prompt_name, kwargs in prompt_specs:
        pred = sam_tool.predict(**kwargs)
        for i, (mask, score) in enumerate(zip(pred.masks, pred.scores)):
            mask = mask.astype(np.uint8)
            ref_score = reference_candidate_score(mask, float(score), ref_box, ref_point, prompt_name)
            candidates.append((prompt_name, i, mask, float(score), ref_score))

    if not candidates:
        raise RuntimeError("No reference seed candidates were generated.")

    if selection_mode == "heuristic":
        best_name, best_index, best_mask, _, _ = max(candidates, key=lambda item: item[4])
    elif selection_mode == "reference_gt_dice":
        best_name, best_index, best_mask, _, _ = max(candidates, key=lambda item: dice(item[2], reference_gt_mask))
    else:
        raise ValueError(f"Unknown reference seed selection mode: {selection_mode}")
    x1, y1, x2, y2 = ref_box
    box_area = max(float((x2 - x1 + 1) * (y2 - y1 + 1)), 1.0)
    report = [
        {
            "prompt": name,
            "candidate_index": int(index),
            "sam_score": float(score),
            "reference_selection_score": float(ref_score),
            "reference_seed_selection": selection_mode,
            "area": int(mask.sum()),
            "box_fill_ratio": float(mask[y1 : y2 + 1, x1 : x2 + 1].sum() / box_area),
            "point_inside_mask": bool(ref_point is not None and mask[ref_point[1], ref_point[0]] > 0),
            "dice_to_reference_gt": dice(mask, reference_gt_mask),
            "selected": name == best_name and index == best_index,
        }
        for name, index, mask, score, ref_score in candidates
    ]
    return best_mask, report


def reference_candidate_score(
    mask: np.ndarray,
    sam_score: float,
    ref_box: list[int],
    ref_point: list[int] | None,
    prompt_name: str,
) -> float:
    x1, y1, x2, y2 = ref_box
    box_area = max(float((x2 - x1 + 1) * (y2 - y1 + 1)), 1.0)
    box_fill = float(mask[y1 : y2 + 1, x1 : x2 + 1].sum() / box_area)

    score = float(sam_score)
    score -= 0.80 * abs(box_fill - 0.55)
    if box_fill > 0.75:
        score -= 0.35 * (box_fill - 0.75) / 0.25
    if box_fill < 0.05:
        score -= 0.50
    if ref_point is not None and mask[ref_point[1], ref_point[0]] == 0:
        score -= 0.35
    if prompt_name == "ref_core_point":
        score -= 0.08
    return score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-sequence liver MVP.")
    parser.add_argument("--data-dir", default=r".")
    parser.add_argument("--case-id", default="0001")
    parser.add_argument("--liver-label", type=int, default=1)
    parser.add_argument("--ref-index", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--topk-ratio", type=float, default=0.5)
    parser.add_argument("--anomaly-foreground", choices=["prediction", "gt"], default="prediction")
    parser.add_argument("--output-dir", default="outputs/liver_mvp_0001_real_sam")
    parser.add_argument("--window-min", type=float, default=-160.0)
    parser.add_argument("--window-max", type=float, default=240.0)
    parser.add_argument("--sam-size", type=int, default=256)
    parser.add_argument("--model-path", default=r".\sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualize-top", type=int, default=5)
    parser.add_argument("--score-mode", choices=["uniform", "state"], default="state")
    parser.add_argument("--accept-margin", type=float, default=0.08)
    parser.add_argument("--current-mask-score", type=float, default=0.5)
    parser.add_argument("--use-reliable-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reference-seed-selection",
        choices=["heuristic", "reference_gt_dice"],
        default="heuristic",
        help="How to choose the initial SAM seed mask on the single annotated reference slice.",
    )
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))

