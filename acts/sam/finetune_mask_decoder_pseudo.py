from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acts.data.dataset import choose_reference_slice, load_flare_case
from acts.data.nifti import load_nii
from acts.data.preprocess import preprocess_ct_for_sam, resize_volume_xy
from acts.prompts.prompt_from_mask import mask_to_box, mask_to_core_point


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = torch.sum(prob * target, dim=dims)
    union = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    return 1.0 - torch.mean((2.0 * inter + eps) / (union + eps))


def mask_to_sam_lowres_logits(mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.float32)
    if mask.shape != (256, 256):
        image = Image.fromarray((mask * 255).astype(np.uint8))
        image = image.resize((256, 256), Image.Resampling.NEAREST)
        mask = (np.asarray(image) > 0).astype(np.float32)
    return np.where(mask > 0, 20.0, -20.0).astype(np.float32)


def pseudo_loss_weight(record: dict[str, Any], args: argparse.Namespace) -> float:
    if args.pseudo_weight_mode == "constant":
        return float(args.pseudo_loss_weight)
    reward = float(record.get("total_reward", 0.0))
    raw = float(args.pseudo_loss_weight) * (1.0 + float(args.reward_weight_scale) * max(0.0, reward - args.min_total_reward))
    return float(np.clip(raw, args.min_pseudo_weight, args.max_pseudo_weight))


def _resolve_covid_path(data_dir: str, case_id: str) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    if case_id.startswith("coronacases_"):
        return image_dir / f"{case_id}.nii.gz", label_dir / f"{case_id}.nii.gz"
    if case_id.startswith("radiopaedia_"):
        key = case_id.removeprefix("radiopaedia_")
        return image_dir / f"radiopaedia_org_covid-19-pneumonia-{key}-dcm.nii.gz", label_dir / f"radiopaedia_{key}.nii.gz"
    raise ValueError(f"Unknown covid case id: {case_id}")


def _load_covid_case(data_dir: str, case_id: str):
    from acts.data.dataset import CTCase

    image_path, label_path = _resolve_covid_path(data_dir, case_id)
    image = load_nii(image_path)
    label = load_nii(label_path)
    lesion_mask = (label.array > 0).astype(np.uint8)
    return CTCase(case_id=case_id, image=image, label=label, liver_mask=lesion_mask)


def build_pseudo_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    pseudo_root = Path(args.pseudo_root)
    for case_id in args.case_ids:
        if getattr(args, "covid", False):
            case = _load_covid_case(args.data_dir, case_id)
        else:
            case = load_flare_case(args.data_dir, case_id, liver_label=args.liver_label)
        case_pseudo_dir = pseudo_root / f"case_{case_id}"
        pseudo_path = pseudo_root / f"case_{case_id}" / f"case_{case_id}_rl_liver_mask.nii.gz"
        if not pseudo_path.exists():
            pseudo_path = pseudo_root / f"case_{case_id}_rl_liver_mask.nii.gz"
        if not pseudo_path.exists():
            raise FileNotFoundError(f"Missing pseudo label for case {case_id}: {pseudo_path}")

        pseudo_volume = (load_nii(pseudo_path).array > 0).astype(np.uint8)
        if pseudo_volume.shape != case.image.array.shape:
            raise ValueError(
                f"Pseudo label shape mismatch for case {case_id}: "
                f"{pseudo_volume.shape} vs {case.image.array.shape}"
            )

        preprocessed = preprocess_ct_for_sam(
            case.image.array.astype(np.float32),
            window_min=args.window_min,
            window_max=args.window_max,
            sam_size=args.sam_size,
        )
        gt_sam = resize_volume_xy(case.liver_mask.astype(np.uint8), size=args.sam_size, nearest=True).astype(np.uint8)
        if args.include_gt_anchor:
            ref_index = choose_reference_slice(gt_sam)
            ref_mask = gt_sam[:, :, ref_index].astype(np.uint8)
            ref_box = mask_to_box(ref_mask, expand_ratio=args.box_expand_ratio)
            ref_point = mask_to_core_point(ref_mask)
            if ref_box is not None and ref_point is not None:
                samples.append(
                    {
                        "case_id": case_id,
                        "slice": int(ref_index),
                        "area": int(ref_mask.sum()),
                        "image": preprocessed.sam_images_rgb[:, :, ref_index, :].astype(np.uint8),
                        "target_mask": ref_mask,
                        "box": ref_box,
                        "point": ref_point,
                        "pseudo_path": "GT",
                        "source": "gt_anchor",
                        "confidence": 1.0,
                        "rl_candidate_name": "GT",
                        "total_reward": 0.0,
                        "accepted_steps": 0,
                        "loss_weight": float(args.gt_anchor_weight),
                    }
                )

        pseudo_sam = resize_volume_xy(pseudo_volume, size=args.sam_size, nearest=True).astype(np.uint8)
        areas = pseudo_sam.sum(axis=(0, 1))
        slice_records = select_pseudo_slice_records(case_pseudo_dir, areas, args)

        for record in slice_records:
            z = int(record["slice"])
            mask = pseudo_sam[:, :, z].astype(np.uint8)
            box = mask_to_box(mask, expand_ratio=args.box_expand_ratio)
            point = mask_to_core_point(mask)
            if box is None or point is None:
                continue
            samples.append(
                {
                    "case_id": case_id,
                    "slice": int(z),
                    "area": int(mask.sum()),
                    "image": preprocessed.sam_images_rgb[:, :, z, :].astype(np.uint8),
                    "target_mask": mask,
                    "box": box,
                    "point": point,
                    "pseudo_path": str(pseudo_path),
                    "source": record["source"],
                    "confidence": float(record["confidence"]),
                    "rl_candidate_name": record.get("rl_candidate_name", ""),
                    "total_reward": float(record.get("total_reward", 0.0)),
                    "accepted_steps": int(record.get("accepted_steps", 0)),
                    "loss_weight": pseudo_loss_weight(record, args),
                }
            )
    return samples


def select_pseudo_slice_records(case_pseudo_dir: Path, areas: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pseudo_slice_source == "all_non_empty":
        records = [
            {
                "slice": int(z),
                "area": int(areas[z]),
                "source": "all_non_empty",
                "confidence": float(areas[z]),
            }
            for z in np.where(areas >= args.min_area)[0].tolist()
        ]
    else:
        report_path = case_pseudo_dir / "rl_action_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing DQN action report for high-confidence pseudo labels: {report_path}")
        reports = json.loads(report_path.read_text(encoding="utf-8"))
        records = []
        excluded = set(args.exclude_candidate_names)
        for report in reports:
            z = int(report["slice"])
            if z < 0 or z >= areas.shape[0] or int(areas[z]) < args.min_area:
                continue
            candidate_name = str(report.get("rl_candidate_name", ""))
            if candidate_name in excluded:
                continue
            accepted_steps = sum(1 for step in report.get("steps", []) if bool(step.get("info", {}).get("accepted", False)))
            if args.require_accepted and accepted_steps <= 0:
                continue
            total_reward = float(report.get("total_reward", 0.0))
            if total_reward < args.min_total_reward:
                continue
            # This is a no-GT confidence proxy: DQN accepted a candidate and
            # received a positive episode reward. GT Dice fields in the report
            # are deliberately not used for pseudo-label filtering.
            confidence = total_reward + 0.02 * accepted_steps
            records.append(
                {
                    "slice": z,
                    "area": int(areas[z]),
                    "source": "dqn_action_report",
                    "confidence": float(confidence),
                    "rl_candidate_name": candidate_name,
                    "total_reward": total_reward,
                    "accepted_steps": int(accepted_steps),
                }
            )

    records = sorted(records, key=lambda item: (float(item["confidence"]), int(item["area"])), reverse=True)
    if args.max_slices_per_case is not None:
        records = records[: args.max_slices_per_case]
    return sorted(records, key=lambda item: int(item["slice"]))


def prepare_image(image_rgb: np.ndarray, transform, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    input_image = transform.apply_image(image_rgb)
    input_size = tuple(input_image.shape[:2])
    image_tensor = torch.as_tensor(input_image, device=device)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :].float()
    return image_tensor, input_size


def augment_box(box: list[int], mask_shape: tuple[int, int], jitter_ratio: float) -> list[int]:
    if jitter_ratio <= 0:
        return box
    x1, y1, x2, y2 = [int(v) for v in box]
    height, width = mask_shape
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    max_dx = max(1, int(round(bw * jitter_ratio)))
    max_dy = max(1, int(round(bh * jitter_ratio)))
    nx1 = x1 + random.randint(-max_dx, max_dx)
    nx2 = x2 + random.randint(-max_dx, max_dx)
    ny1 = y1 + random.randint(-max_dy, max_dy)
    ny2 = y2 + random.randint(-max_dy, max_dy)
    if nx1 > nx2:
        nx1, nx2 = nx2, nx1
    if ny1 > ny2:
        ny1, ny2 = ny2, ny1
    return [
        int(np.clip(nx1, 0, width - 1)),
        int(np.clip(ny1, 0, height - 1)),
        int(np.clip(nx2, 0, width - 1)),
        int(np.clip(ny2, 0, height - 1)),
    ]


def sample_point(mask: np.ndarray, fallback_point: list[int], point_mode: str) -> list[int]:
    if point_mode == "core":
        return fallback_point
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return fallback_point
    idx = random.randrange(xs.size)
    return [int(xs[idx]), int(ys[idx])]


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    samples = build_pseudo_samples(args)
    if not samples:
        raise RuntimeError("No pseudo-label training slices were selected.")

    sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
    sam.to(device=device)

    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()
    sam.mask_decoder.train()

    transform = ResizeLongestSide(sam.image_encoder.img_size)
    optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    manifest_rows = [
        {
            "case_id": sample["case_id"],
            "slice": sample["slice"],
            "area": sample["area"],
            "box": json.dumps(sample["box"]),
            "point": json.dumps(sample["point"]),
            "pseudo_path": sample["pseudo_path"],
            "source": sample["source"],
            "confidence": sample["confidence"],
            "rl_candidate_name": sample["rl_candidate_name"],
            "total_reward": sample["total_reward"],
            "accepted_steps": sample["accepted_steps"],
            "loss_weight": sample["loss_weight"],
        }
        for sample in samples
    ]
    with (output_dir / "pseudo_label_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "slice",
                "area",
                "box",
                "point",
                "pseudo_path",
                "source",
                "confidence",
                "rl_candidate_name",
                "total_reward",
                "accepted_steps",
                "loss_weight",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logs: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        random.shuffle(samples)
        epoch_losses: list[float] = []
        epoch_bce: list[float] = []
        epoch_dice: list[float] = []

        for sample in samples:
            image_tensor, input_size = prepare_image(sample["image"], transform, device)
            original_size = sample["image"].shape[:2]
            target_mask_np = sample["target_mask"]
            target_mask = torch.as_tensor(target_mask_np[None, None, :, :], dtype=torch.float32, device=device)

            with torch.no_grad():
                image_embeddings = sam.image_encoder(sam.preprocess(image_tensor))

            for _ in range(args.augmentations_per_sample):
                aug_box = augment_box(sample["box"], target_mask_np.shape, args.box_jitter_ratio)
                aug_point = sample_point(target_mask_np, sample["point"], args.point_mode)

                box_np = np.asarray(aug_box, dtype=np.float32)[None, :]
                box_np = transform.apply_boxes(box_np, original_size)
                box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=device)

                point_np = np.asarray([aug_point], dtype=np.float32)
                point_np = transform.apply_coords(point_np, original_size)
                point_coords = torch.as_tensor(point_np[None, :, :], dtype=torch.float32, device=device)
                point_labels = torch.ones((1, 1), dtype=torch.int64, device=device)

                mask_input = None
                if args.prompt_mode == "box_point_mask":
                    mask_logits_np = mask_to_sam_lowres_logits(target_mask_np)
                    mask_input = torch.as_tensor(mask_logits_np[None, None, :, :], dtype=torch.float32, device=device)

                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=(point_coords, point_labels),
                        boxes=box_torch,
                        masks=mask_input,
                    )

                with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                    low_res_masks, iou_predictions = sam.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_logits = sam.postprocess_masks(low_res_masks, input_size=input_size, original_size=original_size)
                    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
                    dloss = dice_loss_from_logits(pred_logits, target_mask)
                    loss = args.bce_weight * bce + args.dice_weight * dloss
                    if args.iou_weight > 0:
                        with torch.no_grad():
                            pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
                            inter = torch.sum(pred_bin * target_mask)
                            union = torch.sum(pred_bin) + torch.sum(target_mask) - inter
                            pseudo_iou = inter / torch.clamp(union, min=1.0)
                        loss = loss + args.iou_weight * F.mse_loss(iou_predictions[:, 0], pseudo_iou[None])
                    loss = loss * float(sample["loss_weight"])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(float(loss.detach().cpu()))
                epoch_bce.append(float(bce.detach().cpu()))
                epoch_dice.append(float(dloss.detach().cpu()))

        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(epoch_losses)),
            "bce_loss": float(np.mean(epoch_bce)),
            "dice_loss": float(np.mean(epoch_dice)),
            "num_slices": len(samples),
            "num_augmented_steps": len(epoch_losses),
        }
        logs.append(row)
        print(json.dumps(row, ensure_ascii=False))

    full_path = output_dir / "sam_vit_b_liver_pseudo_maskdecoder_finetuned_full.pth"
    decoder_path = output_dir / "sam_vit_b_liver_pseudo_maskdecoder_only.pth"
    torch.save(sam.state_dict(), full_path)
    torch.save(sam.mask_decoder.state_dict(), decoder_path)

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "model_type": args.model_type,
        "base_model_path": args.model_path,
        "full_finetuned_sam_path": str(full_path),
        "mask_decoder_only_path": str(decoder_path),
        "case_ids": args.case_ids,
        "num_training_slices": len(samples),
        "num_gt_anchor_slices": sum(1 for sample in samples if sample["source"] == "gt_anchor"),
        "num_pseudo_slices": sum(1 for sample in samples if sample["source"] != "gt_anchor"),
        "pseudo_slice_source": args.pseudo_slice_source,
        "prompt_mode": args.prompt_mode,
        "augmentations_per_sample": args.augmentations_per_sample,
        "box_jitter_ratio": args.box_jitter_ratio,
        "point_mode": args.point_mode,
        "pseudo_weight_mode": args.pseudo_weight_mode,
        "reward_weight_scale": args.reward_weight_scale,
        "min_pseudo_weight": args.min_pseudo_weight,
        "max_pseudo_weight": args.max_pseudo_weight,
        "include_gt_anchor": args.include_gt_anchor,
        "gt_anchor_weight": args.gt_anchor_weight,
        "pseudo_loss_weight": args.pseudo_loss_weight,
        "require_accepted": args.require_accepted,
        "min_total_reward": args.min_total_reward,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "trainable_part": "sam.mask_decoder only",
        "frozen_parts": ["sam.image_encoder", "sam.prompt_encoder"],
        "logs": logs,
        "note": "SAM mask decoder fine-tuned from DQN-generated high-confidence pseudo labels. GT Dice fields are not used for pseudo-label filtering.",
    }
    (output_dir / "sam_finetune_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir / "README_SAM_FINETUNE.md", summary)
    return summary


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# SAM Mask Decoder Pseudo-label Fine-tuning",
        "",
        "This experiment follows the second step suggested by the senior student:",
        "",
        "1. Freeze the original SAM image encoder and prompt encoder.",
        "2. Train only SAM's mask decoder.",
        "3. Use high-confidence DQN-generated liver masks as pseudo labels.",
        "",
        "## Saved Weights",
        "",
        f"- Full fine-tuned SAM: `{summary['full_finetuned_sam_path']}`",
        f"- Mask decoder only: `{summary['mask_decoder_only_path']}`",
        "",
        "## Training Data",
        "",
        f"- Cases: {', '.join(summary['case_ids'])}",
        f"- Pseudo-label slices: {summary['num_training_slices']}",
        f"- GT anchor slices: {summary['num_gt_anchor_slices']}",
        f"- DQN pseudo slices: {summary['num_pseudo_slices']}",
        f"- Pseudo-label source: `{summary['pseudo_slice_source']}`",
        f"- Prompt mode: `{summary['prompt_mode']}`",
        f"- Augmentations per sample: `{summary['augmentations_per_sample']}`",
        f"- Box jitter ratio: `{summary['box_jitter_ratio']}`",
        f"- Point mode: `{summary['point_mode']}`",
        f"- Pseudo weight mode: `{summary['pseudo_weight_mode']}`",
        f"- GT anchor enabled: `{summary['include_gt_anchor']}`",
        "- Manifest: `pseudo_label_manifest.csv`",
        "",
        "Important: these are pseudo labels produced by the current DQN pipeline, not manual/GT labels.",
        "GT Dice values in action reports are not used to select pseudo-label slices.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune only SAM mask decoder using DQN pseudo labels.")
    parser.add_argument("--data-dir", default=r"data")
    parser.add_argument("--case-ids", nargs="+", default=[f"{i:04d}" for i in range(1, 11)])
    parser.add_argument("--pseudo-root", default=r".\outputs\experiments\liver_dqn_softwin_20260708\07_sam_finetune_pseudo\01_train_pseudo_labels")
    parser.add_argument("--output-dir", default=r".\outputs\experiments\liver_dqn_softwin_20260708\07_sam_finetune_pseudo\02_sam_mask_decoder_finetune")
    parser.add_argument("--model-path", default=r".\sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--liver-label", type=int, default=1)
    parser.add_argument("--window-min", type=float, default=-160.0)
    parser.add_argument("--window-max", type=float, default=240.0)
    parser.add_argument("--sam-size", type=int, default=256)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--max-slices-per-case", type=int, default=64)
    parser.add_argument("--pseudo-slice-source", choices=["action_report", "all_non_empty"], default="action_report")
    parser.add_argument("--min-total-reward", type=float, default=0.12)
    parser.add_argument("--require-accepted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exclude-candidate-names",
        nargs="*",
        default=["initial_current_mask", "empty_mask", "stop_direction"],
    )
    parser.add_argument("--prompt-mode", choices=["box_point", "box_point_mask"], default="box_point")
    parser.add_argument("--augmentations-per-sample", type=int, default=1)
    parser.add_argument("--box-jitter-ratio", type=float, default=0.0)
    parser.add_argument("--point-mode", choices=["core", "random"], default="core")
    parser.add_argument("--pseudo-weight-mode", choices=["constant", "reward"], default="constant")
    parser.add_argument("--reward-weight-scale", type=float, default=1.0)
    parser.add_argument("--min-pseudo-weight", type=float, default=0.5)
    parser.add_argument("--max-pseudo-weight", type=float, default=2.0)
    parser.add_argument("--include-gt-anchor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gt-anchor-weight", type=float, default=3.0)
    parser.add_argument("--pseudo-loss-weight", type=float, default=1.0)
    parser.add_argument("--box-expand-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--covid", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))

