"""Fixed AMOS pipeline.

Key fixes compared with the quick AMOS script:
1. Pseudo labels and final evaluation start from the full initial SAM volume,
   then replace only cached abnormal slices.
2. Fine-tuned caches are rebuilt organ by organ with that organ's SAM weights.
3. DQN policies are trained per organ instead of mixing all organs together.
4. SAM fine-tuning prompts are scaled from 256 mask space to 1024 SAM space.

Default is liver only for a fast deadline check.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import torchvision.extension as _torchvision_ext

    _torchvision_ext._check_cuda_version = lambda: None
except Exception:
    pass


ROOT = Path(r".")
sys.path.insert(0, str(ROOT))

from acts.data.dataset import choose_reference_slice
from acts.data.nifti import load_nii, save_nii_like
from acts.data.preprocess import (
    preprocess_ct_for_sam,
    resize_mask_to_original,
    resize_volume_xy,
    restore_mask_volume_xy,
)
from acts.evaluation.metrics import evaluate_volume
from acts.main_liver_mvp import select_reference_seed_mask
from acts.prompts.prompt_from_mask import mask_to_box, mask_to_core_point
from acts.rl.candidate_cache import _cache_slice, _merge_slices, select_fp_aware_slices
from acts.rl.dqn import DQNAgent, ReplayBuffer, select_action, train_dqn_one_step
from acts.rl.env import CTSliceAgentEnv
from acts.sam.sam_tool import SegmentAnythingSAMTool
from acts.sequence.anomaly import compute_anomaly_scores, select_topk_slices
from acts.sequence.propagate import propagate_sequence


DATA = ROOT / "Data" / "amos"
SAM_CKPT = ROOT / "sam_vit_b_01ec64.pth"

TRAIN_CASES = ["amos_0001", "amos_0004", "amos_0005", "amos_0006", "amos_0007", "amos_0009", "amos_0010"]
TEST_CASES = ["amos_0011", "amos_0014", "amos_0015"]
ORGANS = {
    "liver": 6,
    "spleen": 1,
    "right_kidney": 2,
    "left_kidney": 3,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_case(case_id: str, organ_label: int) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    img_nii = load_nii(DATA / "images" / f"{case_id}.nii.gz")
    lbl_nii = load_nii(DATA / "label" / f"{case_id}.nii.gz")
    image = img_nii.array.astype(np.float32)
    gt = (lbl_nii.array == organ_label).astype(np.uint8)
    if gt.sum() == 0:
        raise ValueError(f"No foreground for {case_id}, label={organ_label}")
    return img_nii, lbl_nii, image, gt


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def build_cache_for_organ(
    *,
    case_list: list[str],
    organ_name: str,
    organ_label: int,
    sam_model_path: Path,
    cache_root: Path,
    sam_size: int,
    topk_ratio: float,
    device: str,
) -> None:
    """Build cache and save full initial/rule/oracle masks for one organ."""
    organ_cache = cache_root / organ_name
    organ_cache.mkdir(parents=True, exist_ok=True)
    sam_tool = SegmentAnythingSAMTool(str(sam_model_path), model_type="vit_b", device=device)

    for case_id in case_list:
        cache_dir = organ_cache / case_id
        meta_path = cache_dir / "cache_metadata.json"
        initial_path = cache_dir / "initial_sam.nii.gz"
        rule_path = cache_dir / "rule_sam.nii.gz"
        oracle_path = cache_dir / "oracle_sam.nii.gz"
        if meta_path.exists() and initial_path.exists() and rule_path.exists() and oracle_path.exists():
            print(f"  {case_id} {organ_name}: skip cache (exists)")
            continue

        _img_nii, lbl_nii, image, gt = load_case(case_id, organ_label)
        pp = preprocess_ct_for_sam(image, sam_size=sam_size)
        sam_rgb = pp.sam_images_rgb
        sam_gray = sam_rgb[..., 0]
        gt_sam = resize_volume_xy(gt, size=sam_size, nearest=True).astype(np.uint8)
        ref_z = choose_reference_slice(gt_sam)

        sam_tool.set_slice_index(ref_z)
        sam_tool.set_image(sam_rgb[:, :, ref_z, :])
        ref_box = sam_tool.reference_box(gt_sam[:, :, ref_z])
        ref_pt = mask_to_core_point(gt_sam[:, :, ref_z])
        ref_mask, _ = select_reference_seed_mask(
            sam_tool,
            gt_sam[:, :, ref_z],
            ref_box,
            ref_pt,
            selection_mode="heuristic",
        )
        initial_masks, _ = propagate_sequence(
            sam_rgb,
            ref_z,
            ref_mask,
            sam_tool,
            use_reliable_masks=True,
            return_report=True,
        )

        anomaly = compute_anomaly_scores(sam_gray, initial_masks)
        topk = max(1, int(round(topk_ratio * initial_masks.shape[2])))
        topk_slices = select_topk_slices(anomaly, k=topk, foreground_mask=initial_masks)
        fp_slices, _ = select_fp_aware_slices(
            initial_masks=initial_masks,
            ref_index=ref_z,
            ref_mask=ref_mask,
            enabled=True,
            tail_margin=25,
            min_area_ratio=0.05,
            max_extra_slices=None,
        )
        abnormal = [z for z in _merge_slices(topk_slices, fp_slices) if 0 < z < initial_masks.shape[2] - 1]

        masks_dir = cache_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for t in abnormal:
            sample = _cache_slice(
                t=int(t),
                case_id=case_id,
                masks_dir=masks_dir,
                sam_tool=sam_tool,
                sam_images_rgb=sam_rgb,
                sam_images_gray=sam_gray,
                current_masks=initial_masks,
                gt_sam=gt_sam,
                ref_mask=ref_mask,
                ref_index=ref_z,
                anomaly_score=float(anomaly[t]),
            )
            sample["organ"] = organ_name
            samples.append(sample)

        metadata = {
            "case_id": case_id,
            "organ": organ_name,
            "organ_label": organ_label,
            "sam_model_path": str(sam_model_path),
            "ref_index": int(ref_z),
            "num_samples": len(samples),
            "samples": samples,
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        rule_masks = initial_masks.copy()
        oracle_masks = initial_masks.copy()
        for sample in samples:
            t = int(sample["slice"])
            masks = np.load(sample["mask_file"], allow_pickle=False)["masks"].astype(np.uint8)
            rule_masks[:, :, t] = masks[int(sample["rule_candidate_index"])]
            oracle_masks[:, :, t] = masks[int(sample["oracle_candidate_index"])]

        save_nii_like(initial_path, restore_mask_volume_xy(initial_masks, pp.original_size), lbl_nii)
        save_nii_like(rule_path, restore_mask_volume_xy(rule_masks, pp.original_size), lbl_nii)
        save_nii_like(oracle_path, restore_mask_volume_xy(oracle_masks, pp.original_size), lbl_nii)
        print(f"  {case_id} {organ_name}: cached {len(samples)} slices")


def train_dqn_for_organ(
    *,
    cache_root: Path,
    organ_name: str,
    case_list: list[str],
    out_dir: Path,
    max_steps: int,
    hidden_dim: int,
    epochs: int,
    device: str,
) -> Path | None:
    samples = []
    for case_id in case_list:
        meta_path = cache_root / organ_name / case_id / "cache_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for sample in meta["samples"]:
            sample = dict(sample)
            sample["case_id"] = case_id
            sample["organ"] = organ_name
            samples.append(sample)

    if not samples:
        print(f"  {organ_name}: no DQN samples")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_cache = out_dir / "_temp_cache"
    temp_cache.mkdir(parents=True, exist_ok=True)
    (temp_cache / "cache_metadata.json").write_text(
        json.dumps({"case_id": "multi_train", "organ": organ_name, "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    td = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = CTSliceAgentEnv(temp_cache, max_steps=max_steps)
    policy = DQNAgent(env.state_dim, env.action_dim, hidden_dim).to(td)
    target = DQNAgent(env.state_dim, env.action_dim, hidden_dim).to(td)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    buf = ReplayBuffer(20000)
    epsilon = 1.0
    logs = []
    global_step = 0

    print(f"  {organ_name}: train DQN on {len(samples)} cached slices")
    for epoch in range(epochs):
        indices = list(range(len(env.samples)))
        random.shuffle(indices)
        rewards = []
        losses = []
        for idx in indices:
            state = env.reset(idx)
            for _ in range(max_steps):
                action = select_action(policy, state, epsilon, env.action_dim, td)
                next_state, reward, done, _info = env.step(action)
                buf.push(state, action, reward, next_state, done)
                state = next_state
                rewards.append(float(reward))
                global_step += 1
                if len(buf) >= 64:
                    losses.append(float(train_dqn_one_step(policy, target, opt, buf, 64, 0.5, td)))
                if global_step % 100 == 0:
                    target.load_state_dict(policy.state_dict())
                if done:
                    break
        epsilon = max(0.08, epsilon * 0.92)
        row = {
            "epoch": epoch + 1,
            "epsilon": float(epsilon),
            "loss": mean(losses),
            "reward": mean(rewards),
        }
        logs.append(row)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch + 1:03d}: loss={row['loss']:.4f} reward={row['reward']:.4f}")

    policy_path = out_dir / f"{organ_name}_dqn_policy.pt"
    torch.save(policy.state_dict(), policy_path)
    (out_dir / f"{organ_name}_training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.rmtree(temp_cache, ignore_errors=True)
    return policy_path


def apply_dqn_to_cache(
    *,
    case_id: str,
    organ_name: str,
    organ_label: int,
    cache_root: Path,
    policy_path: Path,
    out_dir: Path,
    max_steps: int,
    hidden_dim: int,
    device: str,
    save_prediction: bool,
) -> dict[str, Any]:
    cache_dir = cache_root / organ_name / case_id
    env = CTSliceAgentEnv(cache_dir, max_steps=max_steps)
    td = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = DQNAgent(env.state_dim, env.action_dim, hidden_dim).to(td)
    policy.load_state_dict(torch.load(policy_path, map_location=td))
    policy.eval()

    lbl_nii = load_nii(DATA / "label" / f"{case_id}.nii.gz")
    gt = (lbl_nii.array == organ_label).astype(np.uint8)
    initial = (load_nii(cache_dir / "initial_sam.nii.gz").array > 0).astype(np.uint8)
    rule = (load_nii(cache_dir / "rule_sam.nii.gz").array > 0).astype(np.uint8)
    oracle = (load_nii(cache_dir / "oracle_sam.nii.gz").array > 0).astype(np.uint8)
    pred = initial.copy()

    action_report = []
    for sample_index in range(len(env.samples)):
        state = env.reset(sample_index)
        sample = env.samples[sample_index]
        t = int(sample["slice"])
        candidate_masks = env.candidate_masks
        final_idx = None
        steps = []
        for _ in range(max_steps):
            action = select_action(policy, state, epsilon=0.0, action_dim=env.action_dim, device=td)
            next_state, reward, done, info = env.step(action)
            if bool(info.get("accepted", False)) and int(info.get("candidate_index", -1)) >= 0:
                final_idx = int(info["candidate_index"])
            steps.append(
                {
                    "action": int(action),
                    "action_name": str(info.get("action_name", "")),
                    "candidate_index": int(info.get("candidate_index", -1)),
                    "accepted": bool(info.get("accepted", False)),
                    "reward": float(reward),
                    "current_dice": float(info.get("current_dice", info.get("new_dice", 0.0))),
                }
            )
            state = next_state
            if done:
                break
        if final_idx is not None:
            pred[:, :, t] = resize_mask_to_original(candidate_masks[final_idx].astype(np.uint8), initial.shape[:2])
        action_report.append({"slice": t, "final_candidate_index": final_idx, "steps": steps})

    metrics = {
        "case_id": case_id,
        "organ": organ_name,
        "initial_dice": float(evaluate_volume(initial, gt)["dice"]),
        "rule_dice": float(evaluate_volume(rule, gt)["dice"]),
        "dqn_dice": float(evaluate_volume(pred, gt)["dice"]),
        "oracle_dice": float(evaluate_volume(oracle, gt)["dice"]),
        "num_cached_slices": len(env.samples),
    }

    if save_prediction:
        case_out = out_dir / organ_name / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        save_nii_like(case_out / f"{case_id}_{organ_name}_initial.nii.gz", initial, lbl_nii)
        save_nii_like(case_out / f"{case_id}_{organ_name}_rule.nii.gz", rule, lbl_nii)
        save_nii_like(case_out / f"{case_id}_{organ_name}_dqn.nii.gz", pred, lbl_nii)
        save_nii_like(case_out / f"{case_id}_{organ_name}_oracle.nii.gz", oracle, lbl_nii)
        (case_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (case_out / "rl_action_report.json").write_text(json.dumps(action_report, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics


def generate_pseudo_labels(
    *,
    cache_root: Path,
    policy_paths: dict[str, Path],
    organs: dict[str, int],
    out_dir: Path,
    max_steps: int,
    hidden_dim: int,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for organ_name, organ_label in organs.items():
        if organ_name not in policy_paths:
            continue
        organ_out = out_dir / organ_name
        organ_out.mkdir(parents=True, exist_ok=True)
        for case_id in TRAIN_CASES:
            cache_dir = cache_root / organ_name / case_id
            if not (cache_dir / "cache_metadata.json").exists():
                continue
            metrics = apply_dqn_to_cache(
                case_id=case_id,
                organ_name=organ_name,
                organ_label=organ_label,
                cache_root=cache_root,
                policy_path=policy_paths[organ_name],
                out_dir=out_dir / "_train_pseudo_debug",
                max_steps=max_steps,
                hidden_dim=hidden_dim,
                device=device,
                save_prediction=True,
            )
            pred_path = out_dir / "_train_pseudo_debug" / organ_name / case_id / f"{case_id}_{organ_name}_dqn.nii.gz"
            target_path = organ_out / f"{case_id}.nii.gz"
            shutil.copyfile(pred_path, target_path)
            print(f"  {case_id} {organ_name}: pseudo saved, dice={metrics['dqn_dice']:.4f}")


def finetune_sam_for_organ(
    *,
    organ_name: str,
    organ_label: int,
    pseudo_root: Path,
    out_dir: Path,
    sam_size: int,
    epochs: int,
    device: str,
) -> Path | None:
    from PIL import Image
    from segment_anything import sam_model_registry
    import torch.nn.functional as F

    ft_model = out_dir / f"sam_vit_b_{organ_name}_amos_maskdecoder_fixed.pth"
    if ft_model.exists():
        print(f"  {organ_name}: skip SAM FT (exists)")
        return ft_model

    samples = []
    pseudo_dir = pseudo_root / organ_name
    for case_id in TRAIN_CASES:
        pseudo_path = pseudo_dir / f"{case_id}.nii.gz"
        if not pseudo_path.exists():
            continue
        _img_nii, _lbl_nii, image, gt = load_case(case_id, organ_label)
        pseudo = (load_nii(pseudo_path).array > 0).astype(np.uint8)
        ref_z = choose_reference_slice(gt)
        if pseudo[:, :, ref_z].sum() == 0:
            ref_z = choose_reference_slice(pseudo)
        if pseudo[:, :, ref_z].sum() == 0:
            continue

        img_slice = np.clip(image[:, :, ref_z].astype(np.float32), -160.0, 240.0)
        img_slice = ((img_slice + 160.0) / 400.0 * 255.0).clip(0, 255).astype(np.uint8)
        img_rgb = np.array(Image.fromarray(img_slice).convert("RGB").resize((1024, 1024), Image.BILINEAR))

        pseudo_sam = resize_volume_xy(pseudo, size=sam_size, nearest=True).astype(np.uint8)
        target = pseudo_sam[:, :, ref_z].astype(np.uint8)
        if target.sum() < 500:
            continue
        box256 = mask_to_box(target, expand_ratio=0.1)
        point256 = mask_to_core_point(target)
        if box256 is None or point256 is None:
            continue
        scale = 1024.0 / float(sam_size)
        box1024 = [float(v) * scale for v in box256]
        point1024 = [float(point256[0]) * scale, float(point256[1]) * scale]
        samples.append({"image": img_rgb, "target": target.astype(np.float32), "box": box1024, "point": point1024})

    if not samples:
        print(f"  {organ_name}: no SAM FT samples")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    td = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT)).to(td)
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    opt = torch.optim.Adam(sam.mask_decoder.parameters(), lr=3e-6)

    print(f"  {organ_name}: SAM FT on {len(samples)} samples")
    logs = []
    for epoch in range(epochs):
        random.shuffle(samples)
        losses = []
        for sample in samples:
            img_t = torch.as_tensor(sample["image"], dtype=torch.float32, device=td).permute(2, 0, 1).unsqueeze(0)
            img_t = sam.preprocess(img_t)
            tgt_t = torch.as_tensor(sample["target"], dtype=torch.float32, device=td).unsqueeze(0).unsqueeze(0)
            box_t = torch.as_tensor(np.array(sample["box"]), dtype=torch.float32, device=td).unsqueeze(0)
            point_t = torch.as_tensor([[sample["point"]]], dtype=torch.float32, device=td)
            point_l = torch.ones((1, 1), dtype=torch.int64, device=td)
            with torch.no_grad():
                img_emb = sam.image_encoder(img_t)
            sparse_emb, dense_emb = sam.prompt_encoder(points=(point_t, point_l), boxes=box_t, masks=None)
            logits, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            target_low = F.interpolate(tgt_t, size=logits.shape[-2:], mode="nearest")
            prob = torch.sigmoid(logits)
            inter = (prob * target_low).sum()
            union = (prob + target_low).sum()
            dice_loss = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
            loss = F.binary_cross_entropy_with_logits(logits, target_low) + dice_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch + 1, "loss": mean(losses)}
        logs.append(row)
        print(f"    epoch {epoch + 1}: loss={row['loss']:.4f}")

    torch.save(sam.state_dict(), ft_model)
    (out_dir / f"{organ_name}_sam_finetune_log.json").write_text(
        json.dumps({"samples": len(samples), "logs": logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ft_model


def write_eval_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "organ",
        "case_id",
        "initial_dice",
        "rule_dice",
        "dqn_dice",
        "oracle_dice",
        "num_cached_slices",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed AMOS SAM-CT pipeline.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "amos_pipeline_fixed")
    parser.add_argument("--organs", nargs="+", default=["liver"], choices=sorted(ORGANS))
    parser.add_argument("--sam-size", type=int, default=256)
    parser.add_argument("--topk-ratio", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dqn-epochs", type=int, default=40)
    parser.add_argument("--ft-epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-sam-ft", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    selected_organs = {name: ORGANS[name] for name in args.organs}

    config = {
        "train_cases": TRAIN_CASES,
        "test_cases": TEST_CASES,
        "organs": selected_organs,
        "sam_checkpoint": str(SAM_CKPT),
        "sam_size": args.sam_size,
        "topk_ratio": args.topk_ratio,
        "max_steps": args.max_steps,
        "dqn_epochs": args.dqn_epochs,
        "ft_epochs": args.ft_epochs,
        "device": args.device,
        "seed": args.seed,
        "note": "Fixed AMOS pipeline: full initial volume is preserved for pseudo-labels and evaluation.",
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("STEP 1: Build frozen SAM caches")
    print("=" * 60)
    frozen_cache = out / "caches_frozen"
    for organ_name, organ_label in selected_organs.items():
        build_cache_for_organ(
            case_list=TRAIN_CASES + TEST_CASES,
            organ_name=organ_name,
            organ_label=organ_label,
            sam_model_path=SAM_CKPT,
            cache_root=frozen_cache,
            sam_size=args.sam_size,
            topk_ratio=args.topk_ratio,
            device=args.device,
        )

    print("\n" + "=" * 60)
    print("STEP 2: Train frozen-cache DQN per organ")
    print("=" * 60)
    frozen_policy_paths: dict[str, Path] = {}
    for organ_name in selected_organs:
        policy_path = train_dqn_for_organ(
            cache_root=frozen_cache,
            organ_name=organ_name,
            case_list=TRAIN_CASES,
            out_dir=out / "dqn_frozen" / organ_name,
            max_steps=args.max_steps,
            hidden_dim=args.hidden_dim,
            epochs=args.dqn_epochs,
            device=args.device,
        )
        if policy_path is not None:
            frozen_policy_paths[organ_name] = policy_path

    print("\n" + "=" * 60)
    print("STEP 3: Evaluate frozen DQN")
    print("=" * 60)
    eval_rows: list[dict[str, Any]] = []
    for organ_name, organ_label in selected_organs.items():
        if organ_name not in frozen_policy_paths:
            continue
        for case_id in TEST_CASES:
            row = apply_dqn_to_cache(
                case_id=case_id,
                organ_name=organ_name,
                organ_label=organ_label,
                cache_root=frozen_cache,
                policy_path=frozen_policy_paths[organ_name],
                out_dir=out / "predictions_frozen",
                max_steps=args.max_steps,
                hidden_dim=args.hidden_dim,
                device=args.device,
                save_prediction=True,
            )
            row["stage"] = "frozen"
            eval_rows.append(row)
            print(
                f"  frozen {case_id} {organ_name}: "
                f"init={row['initial_dice']:.4f} rule={row['rule_dice']:.4f} "
                f"dqn={row['dqn_dice']:.4f} oracle={row['oracle_dice']:.4f}"
            )

    if args.skip_sam_ft:
        write_eval_table(out / "eval_table.csv", eval_rows)
        print(f"\nDone without SAM FT. Outputs: {out}")
        return

    print("\n" + "=" * 60)
    print("STEP 4: Generate pseudo labels from frozen DQN")
    print("=" * 60)
    pseudo_root = out / "pseudo_labels"
    generate_pseudo_labels(
        cache_root=frozen_cache,
        policy_paths=frozen_policy_paths,
        organs=selected_organs,
        out_dir=pseudo_root,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("STEP 5: Fine-tune SAM mask decoder per organ")
    print("=" * 60)
    ft_model_paths: dict[str, Path] = {}
    for organ_name, organ_label in selected_organs.items():
        ft_model = finetune_sam_for_organ(
            organ_name=organ_name,
            organ_label=organ_label,
            pseudo_root=pseudo_root,
            out_dir=out / "sam_finetune",
            sam_size=args.sam_size,
            epochs=args.ft_epochs,
            device=args.device,
        )
        if ft_model is not None:
            ft_model_paths[organ_name] = ft_model

    print("\n" + "=" * 60)
    print("STEP 6: Rebuild caches with organ-specific fine-tuned SAM")
    print("=" * 60)
    ft_cache = out / "caches_finetuned"
    for organ_name, organ_label in selected_organs.items():
        if organ_name not in ft_model_paths:
            print(f"  {organ_name}: skip FT cache, missing SAM FT model")
            continue
        build_cache_for_organ(
            case_list=TRAIN_CASES + TEST_CASES,
            organ_name=organ_name,
            organ_label=organ_label,
            sam_model_path=ft_model_paths[organ_name],
            cache_root=ft_cache,
            sam_size=args.sam_size,
            topk_ratio=args.topk_ratio,
            device=args.device,
        )

    print("\n" + "=" * 60)
    print("STEP 7: Train fine-tuned-cache DQN per organ")
    print("=" * 60)
    ft_policy_paths: dict[str, Path] = {}
    for organ_name in selected_organs:
        if organ_name not in ft_model_paths:
            continue
        policy_path = train_dqn_for_organ(
            cache_root=ft_cache,
            organ_name=organ_name,
            case_list=TRAIN_CASES,
            out_dir=out / "dqn_finetuned" / organ_name,
            max_steps=args.max_steps,
            hidden_dim=args.hidden_dim,
            epochs=args.dqn_epochs,
            device=args.device,
        )
        if policy_path is not None:
            ft_policy_paths[organ_name] = policy_path

    print("\n" + "=" * 60)
    print("STEP 8: Evaluate fine-tuned DQN")
    print("=" * 60)
    for organ_name, organ_label in selected_organs.items():
        if organ_name not in ft_policy_paths:
            continue
        for case_id in TEST_CASES:
            row = apply_dqn_to_cache(
                case_id=case_id,
                organ_name=organ_name,
                organ_label=organ_label,
                cache_root=ft_cache,
                policy_path=ft_policy_paths[organ_name],
                out_dir=out / "predictions_finetuned",
                max_steps=args.max_steps,
                hidden_dim=args.hidden_dim,
                device=args.device,
                save_prediction=True,
            )
            row["stage"] = "finetuned"
            eval_rows.append(row)
            print(
                f"  finetuned {case_id} {organ_name}: "
                f"init={row['initial_dice']:.4f} rule={row['rule_dice']:.4f} "
                f"dqn={row['dqn_dice']:.4f} oracle={row['oracle_dice']:.4f}"
            )

    write_eval_table(out / "eval_table.csv", eval_rows)
    (out / "DONE.txt").write_text("done", encoding="utf-8")
    print(f"\nDone. Outputs: {out}")
    print(f"Eval table: {out / 'eval_table.csv'}")


if __name__ == "__main__":
    main()

