"""Run the ACTS full pipeline on FLARE22 large organs.

This script mirrors the fixed AMOS pipeline while adapting only dataset-specific
items: FLARE case naming, FLARE organ labels, and FLARE train/test splits.
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

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

import run_amos_full_pipeline_fixed as base_pipeline
from acts.data.dataset import load_flare_case
from acts.data.nifti import save_nii_like
from acts.data.preprocess import resize_mask_to_original
from acts.evaluation.metrics import evaluate_volume
from acts.rl.actions import ACTION_NAMES
from acts.rl.dqn import DQNAgent, select_action
from acts.rl.env import CTSliceAgentEnv


DEFAULT_DATA = ROOT / "Data" / "FLARE22"
DEFAULT_SAM_CKPT = ROOT / "sam_vit_b_01ec64.pth"

TRAIN_CASES = [f"{i:04d}" for i in range(1, 41)]
TEST_CASES = [f"{i:04d}" for i in range(41, 51)]
ORGANS = {
    "liver": 1,
    "right_kidney": 2,
    "spleen": 3,
    "left_kidney": 13,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_case(case_id: str, organ_label: int) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    case = load_flare_case(base_pipeline.DATA, case_id, liver_label=organ_label)
    image = case.image.array.astype(np.float32)
    gt = case.liver_mask.astype(np.uint8)
    if gt.sum() == 0:
        raise ValueError(f"No foreground for FLARE case {case_id}, label={organ_label}")
    return case.image, case.label, image, gt


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
    try:
        state_dict = torch.load(policy_path, map_location=td, weights_only=True)
    except TypeError:
        state_dict = torch.load(policy_path, map_location=td)
    policy.load_state_dict(state_dict)
    policy.eval()

    case = load_flare_case(base_pipeline.DATA, case_id, liver_label=organ_label)
    lbl_nii = case.label
    gt = case.liver_mask.astype(np.uint8)
    initial = (base_pipeline.load_nii(cache_dir / "initial_sam.nii.gz").array > 0).astype(np.uint8)
    rule = (base_pipeline.load_nii(cache_dir / "rule_sam.nii.gz").array > 0).astype(np.uint8)
    oracle = (base_pipeline.load_nii(cache_dir / "oracle_sam.nii.gz").array > 0).astype(np.uint8)
    pred = initial.copy()

    action_report = []
    for sample_index in range(len(env.samples)):
        state = env.reset(sample_index)
        sample = env.samples[sample_index]
        t = int(sample["slice"])
        candidate_masks = env.candidate_masks
        if candidate_masks is None:
            raise RuntimeError("Environment did not load candidate masks.")
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
                    "action_name": ACTION_NAMES.get(int(action), f"unknown_{action}"),
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


def patch_base_pipeline(data_dir: Path, sam_checkpoint: Path) -> None:
    base_pipeline.DATA = data_dir
    base_pipeline.SAM_CKPT = sam_checkpoint
    base_pipeline.MODEL_TAG = "flare"
    base_pipeline.TRAIN_CASES = TRAIN_CASES
    base_pipeline.TEST_CASES = TEST_CASES
    base_pipeline.ORGANS = ORGANS
    base_pipeline.load_case = load_case
    base_pipeline.apply_dqn_to_cache = apply_dqn_to_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACTS full pipeline on FLARE22 large organs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CKPT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "flare_pipeline")
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
    patch_base_pipeline(args.data_dir, args.sam_checkpoint)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    selected_organs = {name: ORGANS[name] for name in args.organs}

    config = {
        "dataset": "FLARE22",
        "train_cases": TRAIN_CASES,
        "test_cases": TEST_CASES,
        "organs": selected_organs,
        "sam_checkpoint": str(args.sam_checkpoint),
        "sam_size": args.sam_size,
        "topk_ratio": args.topk_ratio,
        "max_steps": args.max_steps,
        "dqn_epochs": args.dqn_epochs,
        "ft_epochs": args.ft_epochs,
        "device": args.device,
        "seed": args.seed,
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("STEP 1: Build frozen SAM caches")
    print("=" * 60)
    frozen_cache = out / "caches_frozen"
    for organ_name, organ_label in selected_organs.items():
        base_pipeline.build_cache_for_organ(
            case_list=TRAIN_CASES + TEST_CASES,
            organ_name=organ_name,
            organ_label=organ_label,
            sam_model_path=args.sam_checkpoint,
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
        policy_path = base_pipeline.train_dqn_for_organ(
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
    base_pipeline.generate_pseudo_labels(
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
        ft_model = base_pipeline.finetune_sam_for_organ(
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
        base_pipeline.build_cache_for_organ(
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
        policy_path = base_pipeline.train_dqn_for_organ(
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
    shutil.rmtree(out / "_tmp", ignore_errors=True)
    (out / "DONE.txt").write_text("done", encoding="utf-8")
    print(f"\nDone. Outputs: {out}")
    print(f"Eval table: {out / 'eval_table.csv'}")


if __name__ == "__main__":
    main()
