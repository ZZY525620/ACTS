from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acts.data.dataset import choose_reference_slice, load_flare_case
from acts.data.preprocess import preprocess_ct_for_sam, resize_volume_xy
from acts.evaluation.metrics import dice
from acts.main_liver_mvp import add_non_sam_candidates, select_reference_seed_mask
from acts.prompts.prompt_from_mask import mask_to_core_point
from acts.prompts.prompt_pool import generate_prompt_pool
from acts.rl.actions import ACTION_NAMES, Action, action_ids_for_candidate
from acts.rl.feature_builder import build_feature_vector
from acts.sam.sam_tool import SegmentAnythingSAMTool
from acts.sequence.anomaly import compute_anomaly_scores, select_topk_slices
from acts.sequence.propagate import propagate_sequence
from acts.sequence.score import state_aware_no_gt_score
from acts.sequence.state import estimate_sequence_state


def build_candidate_cache(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    case = load_flare_case(args.data_dir, args.case_id, liver_label=args.liver_label)
    image = case.image.array.astype(np.float32)
    gt = case.liver_mask.astype(np.uint8)
    preprocessed = preprocess_ct_for_sam(image, args.window_min, args.window_max, args.sam_size)
    sam_images_rgb = preprocessed.sam_images_rgb
    sam_images_gray = sam_images_rgb[..., 0]
    gt_sam = resize_volume_xy(gt, size=args.sam_size, nearest=True).astype(np.uint8)

    ref_index = args.ref_index if args.ref_index is not None else choose_reference_slice(gt_sam)
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
        use_reliable_masks=True,
        return_report=True,
    )
    anomaly_scores = compute_anomaly_scores(sam_images_gray, initial_masks)
    topk = args.topk if args.topk is not None else max(1, int(round(args.topk_ratio * initial_masks.shape[2])))
    topk_slices = select_topk_slices(anomaly_scores, k=topk, foreground_mask=initial_masks)
    fp_aware_slices, fp_aware_report = select_fp_aware_slices(
        initial_masks=initial_masks,
        ref_index=ref_index,
        ref_mask=ref_mask,
        enabled=args.fp_aware,
        tail_margin=args.fp_tail_margin,
        min_area_ratio=args.fp_min_area_ratio,
        max_extra_slices=args.fp_max_extra_slices,
    )
    abnormal_slices = _merge_slices(topk_slices, fp_aware_slices)
    if args.max_slices is not None:
        abnormal_slices = abnormal_slices[: args.max_slices]

    samples = []
    for t in abnormal_slices:
        if t <= 0 or t >= initial_masks.shape[2] - 1:
            continue
        sample = _cache_slice(
            t=t,
            case_id=args.case_id,
            masks_dir=masks_dir,
            sam_tool=sam_tool,
            sam_images_rgb=sam_images_rgb,
            sam_images_gray=sam_images_gray,
            current_masks=initial_masks,
            gt_sam=gt_sam,
            ref_mask=ref_mask,
            ref_index=ref_index,
            anomaly_score=float(anomaly_scores[t]),
        )
        samples.append(sample)

    metadata = {
        "case_id": args.case_id,
        "liver_label": args.liver_label,
        "sam_input_size_hw": list(preprocessed.sam_input_size),
        "ref_index": int(ref_index),
        "ref_box": ref_box,
        "ref_point": ref_point,
        "ref_prompt_report": ref_prompt_report,
        "reference_seed_selection": args.reference_seed_selection,
        "action_names": ACTION_NAMES,
        "topk": int(topk),
        "slice_selection": {
            "topk_slices": [int(x) for x in topk_slices],
            "fp_aware_enabled": bool(args.fp_aware),
            "fp_aware_slices": [int(x) for x in fp_aware_slices],
            "fp_aware_report": fp_aware_report,
        },
        "cached_slices": [int(s["slice"]) for s in samples],
        "num_samples": len(samples),
        "propagation_report": [step.__dict__ for step in propagation_report],
        "samples": samples,
        "note": "Cached candidates for slice-level DQN. SAM is frozen; GT metrics are for training/reward analysis.",
    }
    (output_dir / "cache_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def select_fp_aware_slices(
    *,
    initial_masks: np.ndarray,
    ref_index: int,
    ref_mask: np.ndarray,
    enabled: bool,
    tail_margin: int,
    min_area_ratio: float,
    max_extra_slices: int | None,
) -> tuple[list[int], dict]:
    """Select suspicious foreground tails without looking at GT labels.

    The target failure mode is a propagated mask that continues far beyond the
    plausible liver range. We use only the initial prediction area curve and the
    reference slice, so this can be used at inference time.
    """
    areas = initial_masks.sum(axis=(0, 1)).astype(np.float32)
    ref_area = max(float(ref_mask.sum()), 1.0)
    min_area = float(min_area_ratio) * ref_area
    fg = areas >= min_area
    fg_indices = np.where(fg)[0]
    report = {
        "enabled": bool(enabled),
        "ref_index": int(ref_index),
        "ref_area": float(ref_area),
        "min_area_ratio": float(min_area_ratio),
        "min_area": float(min_area),
        "tail_margin": int(tail_margin),
        "predicted_foreground_range": [-1, -1]
        if fg_indices.size == 0
        else [int(fg_indices[0]), int(fg_indices[-1])],
        "max_extra_slices": None if max_extra_slices is None else int(max_extra_slices),
    }
    if not enabled or fg_indices.size == 0:
        return [], report

    lower_cutoff = int(ref_index) - int(tail_margin)
    upper_cutoff = int(ref_index) + int(tail_margin)
    candidates = [
        int(z)
        for z in fg_indices
        if 0 < int(z) < initial_masks.shape[2] - 1 and (int(z) < lower_cutoff or int(z) > upper_cutoff)
    ]
    candidates = sorted(candidates, key=lambda z: (abs(z - ref_index), float(areas[z])), reverse=True)
    if max_extra_slices is not None:
        candidates = candidates[: int(max_extra_slices)]
    candidates = sorted(candidates)
    report["lower_cutoff"] = lower_cutoff
    report["upper_cutoff"] = upper_cutoff
    report["num_selected"] = len(candidates)
    return candidates, report


def _merge_slices(primary: list[int], extra: list[int]) -> list[int]:
    seen = set()
    merged = []
    for value in list(primary) + list(extra):
        value = int(value)
        if value not in seen:
            seen.add(value)
            merged.append(value)
    return merged


def _cache_slice(
    *,
    t: int,
    case_id: str,
    masks_dir: Path,
    sam_tool: SegmentAnythingSAMTool,
    sam_images_rgb: np.ndarray,
    sam_images_gray: np.ndarray,
    current_masks: np.ndarray,
    gt_sam: np.ndarray,
    ref_mask: np.ndarray,
    ref_index: int,
    anomaly_score: float,
) -> dict:
    state = estimate_sequence_state(
        image=sam_images_gray[:, :, t],
        prev_image=sam_images_gray[:, :, t - 1],
        ref_mask=ref_mask,
        current_mask=current_masks[:, :, t],
        prev_mask=current_masks[:, :, t - 1],
        next_mask=current_masks[:, :, t + 1],
        step_from_ref=abs(t - ref_index),
    )
    prompts = generate_prompt_pool(current_masks, t, state=state)
    candidates: list[np.ndarray] = []
    candidate_names: list[str] = []
    sam_scores: list[float] = []

    sam_tool.set_slice_index(t)
    sam_tool.set_image(sam_images_rgb[:, :, t, :])
    for prompt in prompts:
        pred = sam_tool.predict(
            box=prompt.box,
            points=prompt.points,
            point_labels=prompt.point_labels,
            mask_input=prompt.mask_input,
        )
        for i, (mask, score) in enumerate(zip(pred.masks, pred.scores)):
            candidates.append(mask.astype(np.uint8))
            sam_scores.append(float(score))
            candidate_names.append(f"{prompt.name}:{i}")

    add_non_sam_candidates(current_masks, t, candidates, candidate_names, sam_scores)
    candidates.append(np.zeros_like(current_masks[:, :, t], dtype=np.uint8))
    candidate_names.append("stop_direction")
    sam_scores.append(0.0)

    current_mask = current_masks[:, :, t].astype(np.uint8)
    prev_mask = current_masks[:, :, t - 1].astype(np.uint8)
    next_mask = current_masks[:, :, t + 1].astype(np.uint8)
    image = sam_images_gray[:, :, t]
    gt_slice = gt_sam[:, :, t].astype(np.uint8)
    current_dice = dice(current_mask, gt_slice)
    current_no_gt_score = state_aware_no_gt_score(
        current_mask,
        image,
        prev_mask,
        next_mask,
        sam_score=0.5,
        state=state,
        ref_mask=ref_mask,
    )

    no_gt_scores: list[float] = []
    candidate_records = []
    action_to_candidate: dict[int, dict] = {}
    empty_score = 0.0
    best_non_empty_score = 0.0
    best_sam_candidate_score = 0.0
    best_non_sam_candidate_score = 0.0

    for idx, (name, mask, sam_score) in enumerate(zip(candidate_names, candidates, sam_scores)):
        score = state_aware_no_gt_score(
            mask,
            image,
            prev_mask,
            next_mask,
            sam_score=sam_score,
            state=state,
            ref_mask=ref_mask,
        )
        no_gt_scores.append(float(score))
        candidate_dice = dice(mask, gt_slice)
        action_ids = action_ids_for_candidate(name)
        is_sam = ":" in name
        if name == "empty_mask":
            empty_score = float(score)
        if mask.sum() > 0:
            best_non_empty_score = max(best_non_empty_score, float(score))
        if is_sam:
            best_sam_candidate_score = max(best_sam_candidate_score, float(score))
        else:
            best_non_sam_candidate_score = max(best_non_sam_candidate_score, float(score))

        record = {
            "index": int(idx),
            "name": name,
            "action_ids": action_ids,
            "sam_score": float(sam_score),
            "no_gt_score": float(score),
            "dice_to_gt": float(candidate_dice),
            "dice_improvement": float(candidate_dice - current_dice),
            "area": int(mask.sum()),
            "is_sam_candidate": bool(is_sam),
        }
        candidate_records.append(record)
        for action_id in action_ids:
            current_best = action_to_candidate.get(action_id)
            if current_best is None or record["no_gt_score"] > current_best["no_gt_score"]:
                action_to_candidate[action_id] = record

    feature_vector = build_feature_vector(
        slice_index=t,
        depth=current_masks.shape[2],
        ref_index=ref_index,
        direction=1 if t >= ref_index else -1,
        step_id=0,
        current_mask=current_mask,
        prev_mask=prev_mask,
        next_mask=next_mask,
        ref_mask=ref_mask,
        image=sam_images_gray[:, :, t],
        prev_image=sam_images_gray[:, :, t - 1],
        next_image=sam_images_gray[:, :, t + 1],
        state=state,
        candidate_no_gt_scores=[current_no_gt_score] + no_gt_scores,
        candidate_sam_scores=sam_scores,
        empty_score=empty_score,
        best_non_empty_score=best_non_empty_score,
        best_sam_score=best_sam_candidate_score,
        best_non_sam_score=best_non_sam_candidate_score,
    )

    masks = np.stack(candidates, axis=0).astype(np.uint8)
    mask_file = masks_dir / f"case_{case_id}_slice_{t:03d}_candidates.npz"
    np.savez_compressed(mask_file, masks=masks)

    return {
        "case_id": case_id,
        "slice": int(t),
        "mask_file": str(mask_file),
        "anomaly_score": float(anomaly_score),
        "state": state.name,
        "state_features": state.__dict__,
        "feature_names": feature_vector.names,
        "state_vector": feature_vector.values,
        "current_dice": float(current_dice),
        "current_no_gt_score": float(current_no_gt_score),
        "gt_empty": bool(gt_slice.sum() == 0),
        "current_empty": bool(current_mask.sum() == 0),
        "num_candidates": len(candidate_records),
        "candidates": candidate_records,
        "best_candidate_by_action": {
            str(action_id): {
                "action_name": ACTION_NAMES[int(action_id)],
                "candidate_index": int(record["index"]),
                "candidate_name": record["name"],
                "no_gt_score": float(record["no_gt_score"]),
                "dice_to_gt": float(record["dice_to_gt"]),
                "dice_improvement": float(record["dice_improvement"]),
            }
            for action_id, record in sorted(action_to_candidate.items())
        },
        "oracle_candidate_index": int(np.argmax([r["dice_to_gt"] for r in candidate_records])),
        "rule_candidate_index": int(np.argmax(no_gt_scores)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cached candidates for slice-level DQN training.")
    parser.add_argument("--data-dir", default=r".")
    parser.add_argument("--case-id", default="0001")
    parser.add_argument("--liver-label", type=int, default=1)
    parser.add_argument("--ref-index", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--topk-ratio", type=float, default=0.5)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--fp-aware", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fp-tail-margin", type=int, default=25)
    parser.add_argument("--fp-min-area-ratio", type=float, default=0.05)
    parser.add_argument("--fp-max-extra-slices", type=int, default=None)
    parser.add_argument(
        "--reference-seed-selection",
        choices=["heuristic", "reference_gt_dice"],
        default="heuristic",
        help="How to choose the initial SAM seed mask on the single annotated reference slice.",
    )
    parser.add_argument("--output-dir", default=r".\outputs\rl_cache_case0001_liver")
    parser.add_argument("--window-min", type=float, default=-160.0)
    parser.add_argument("--window-max", type=float, default=240.0)
    parser.add_argument("--sam-size", type=int, default=256)
    parser.add_argument("--model-path", default=r".\sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    result = build_candidate_cache(build_parser().parse_args())
    print(json.dumps({"output": "candidate_cache", "num_samples": result["num_samples"]}, ensure_ascii=False, indent=2))

