from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acts.data.dataset import load_flare_case
from acts.data.nifti import load_nii, save_nii_like
from acts.data.preprocess import resize_mask_to_original, window_ct, to_uint8
from acts.evaluation.metrics import dice, evaluate_volume
from acts.evaluation.visualize import overlay_mask
from acts.rl.actions import ACTION_NAMES
from acts.rl.dqn import DQNAgent, select_action
from acts.rl.env import CTSliceAgentEnv


def evaluate_dqn(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = CTSliceAgentEnv(args.cache_dir, max_steps=args.max_steps)
    policy_net = DQNAgent(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    try:
        state_dict = torch.load(args.policy_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.policy_path, map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    case = load_flare_case(args.data_dir, args.case_id, liver_label=args.liver_label)
    gt = case.liver_mask.astype(np.uint8)
    image_uint8 = to_uint8(window_ct(case.image.array.astype(np.float32), args.window_min, args.window_max))

    baseline_dir = Path(args.baseline_dir)
    initial_volume = _load_binary_volume(baseline_dir / f"case_{args.case_id}_initial_liver_mask.nii.gz")
    rule_volume = _load_binary_volume(baseline_dir / f"case_{args.case_id}_corrected_liver_mask.nii.gz")
    rl_volume = initial_volume.copy()
    oracle_volume = initial_volume.copy()

    slice_reports: list[dict[str, Any]] = []
    cached_current_dice: list[float] = []
    cached_rule_dice: list[float] = []
    cached_rl_dice: list[float] = []
    cached_oracle_dice: list[float] = []
    action_counts = {str(i): 0 for i in range(env.action_dim)}
    invalid_count = 0
    accepted_count = 0

    for sample_index, sample in enumerate(env.samples):
        state = env.reset(sample_index)
        t = int(sample["slice"])
        candidate_masks = env.candidate_masks
        if candidate_masks is None:
            raise RuntimeError("Environment did not load candidate masks.")

        steps = []
        final_candidate_index: int | None = None
        final_candidate_name = "initial_current_mask"
        total_reward = 0.0

        for step_id in range(args.max_steps):
            action = select_action(policy_net, state, epsilon=0.0, action_dim=env.action_dim, device=device)
            next_state, reward, done, info = env.step(action)
            action_counts[str(action)] += 1
            invalid_count += int(bool(info.get("invalid_action", False)))
            accepted_count += int(bool(info.get("accepted", False)))
            total_reward += float(reward)

            if bool(info.get("accepted", False)) and int(info.get("candidate_index", -1)) >= 0:
                final_candidate_index = int(info["candidate_index"])
                final_candidate_name = str(info["candidate_name"])

            steps.append(
                {
                    "step": int(step_id),
                    "action": int(action),
                    "action_name": ACTION_NAMES.get(int(action), f"unknown_{action}"),
                    "reward": float(reward),
                    "done": bool(done),
                    "info": _jsonable(info),
                }
            )
            state = next_state
            if done:
                break

        if final_candidate_index is not None:
            selected_sam_mask = candidate_masks[final_candidate_index].astype(np.uint8)
            selected_original_mask = resize_mask_to_original(selected_sam_mask, original_size=gt.shape[:2])
            rl_volume[:, :, t] = selected_original_mask

        rl_slice_dice_cache = float(env.current_dice)
        rl_slice_dice_original = dice(rl_volume[:, :, t], gt[:, :, t])
        rule_candidate = sample["candidates"][int(sample["rule_candidate_index"])]
        oracle_candidate = sample["candidates"][int(sample["oracle_candidate_index"])]
        oracle_sam_mask = candidate_masks[int(sample["oracle_candidate_index"])].astype(np.uint8)
        oracle_original_mask = resize_mask_to_original(oracle_sam_mask, original_size=gt.shape[:2])
        oracle_volume[:, :, t] = oracle_original_mask
        cached_current_dice.append(float(sample["current_dice"]))
        cached_rule_dice.append(float(rule_candidate["dice_to_gt"]))
        cached_rl_dice.append(float(rl_slice_dice_cache))
        cached_oracle_dice.append(float(oracle_candidate["dice_to_gt"]))

        slice_reports.append(
            {
                "sample_index": int(sample_index),
                "slice": int(t),
                "gt_empty": bool(sample["gt_empty"]),
                "initial_dice": float(sample["current_dice"]),
                "rule_candidate_index": int(sample["rule_candidate_index"]),
                "rule_candidate_name": str(rule_candidate["name"]),
                "rule_candidate_dice": float(rule_candidate["dice_to_gt"]),
                "rl_candidate_index": None if final_candidate_index is None else int(final_candidate_index),
                "rl_candidate_name": final_candidate_name,
                "rl_dice_cache_space": float(rl_slice_dice_cache),
                "rl_dice_original_space": float(rl_slice_dice_original),
                "oracle_candidate_index": int(sample["oracle_candidate_index"]),
                "oracle_candidate_name": str(oracle_candidate["name"]),
                "oracle_candidate_dice": float(oracle_candidate["dice_to_gt"]),
                "total_reward": float(total_reward),
                "steps": steps,
            }
        )

    full_metrics = {
        "initial": evaluate_volume(initial_volume, gt),
        "rule_selection": evaluate_volume(rule_volume, gt),
        "rl_agent": evaluate_volume(rl_volume, gt),
        "oracle_candidate": evaluate_volume(oracle_volume, gt),
    }
    cached_metrics = {
        "avg_current_dice": _mean(cached_current_dice),
        "avg_rule_candidate_dice": _mean(cached_rule_dice),
        "avg_rl_dice": _mean(cached_rl_dice),
        "avg_oracle_candidate_dice": _mean(cached_oracle_dice),
        "num_cached_slices": len(cached_rl_dice),
    }
    result = {
        "case_id": args.case_id,
        "liver_label": args.liver_label,
        "cache_dir": str(args.cache_dir),
        "policy_path": str(args.policy_path),
        "baseline_dir": str(args.baseline_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "action_names": ACTION_NAMES,
        "full_volume_metrics": full_metrics,
        "cached_slice_metrics": cached_metrics,
        "invalid_action_ratio": float(invalid_count / max(sum(action_counts.values()), 1)),
        "accepted_action_ratio": float(accepted_count / max(sum(action_counts.values()), 1)),
        "action_counts": action_counts,
        "slice_reports": slice_reports,
        "note": "Offline DQN evaluation on cached slice candidates. Full RL volume starts from Initial SAM and replaces cached abnormal slices with accepted DQN-selected candidates.",
    }

    save_nii_like(output_dir / f"case_{args.case_id}_rl_liver_mask.nii.gz", rl_volume, case.label)
    save_nii_like(output_dir / f"case_{args.case_id}_oracle_liver_mask.nii.gz", oracle_volume, case.label)
    (output_dir / "rl_eval_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "rl_action_report.json").write_text(json.dumps(slice_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(output_dir / "README_RL_EVAL.md", result)
    _save_key_visualizations(output_dir, args.case_id, image_uint8, gt, initial_volume, rule_volume, rl_volume, oracle_volume, slice_reports)
    return result


def _load_binary_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline volume: {path}")
    return (load_nii(path).array > 0).astype(np.uint8)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_readme(path: Path, result: dict[str, Any]) -> None:
    full = result["full_volume_metrics"]
    cached = result["cached_slice_metrics"]
    lines = [
        "# RL DQN Evaluation",
        "",
        f"Case: {result['case_id']}",
        f"Cache: {result['cache_dir']}",
        f"Policy: {result['policy_path']}",
        "",
        "## Full-volume Dice",
        "",
        f"- Initial SAM: {full['initial']['dice']:.6f}",
        f"- Rule Selection: {full['rule_selection']['dice']:.6f}",
        f"- RL Agent: {full['rl_agent']['dice']:.6f}",
        f"- Oracle Candidate: {full['oracle_candidate']['dice']:.6f}",
        "",
        "## Cached-slice Average Dice",
        "",
        f"- Current: {cached['avg_current_dice']:.6f}",
        f"- Rule Candidate: {cached['avg_rule_candidate_dice']:.6f}",
        f"- RL Agent: {cached['avg_rl_dice']:.6f}",
        f"- Oracle Candidate: {cached['avg_oracle_candidate_dice']:.6f}",
        "",
        "## Outputs",
        "",
        f"- `case_{result['case_id']}_rl_liver_mask.nii.gz`: RL agent 3D liver mask.",
        "- `rl_eval_metrics.json`: metrics and action statistics.",
        "- `rl_action_report.json`: per-slice DQN actions and selected candidates.",
        "- `visualizations/`: compact comparison panels.",
        "",
        "Note: this is still a single-case offline cached-candidate RL evaluation, not a validated multi-case model.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_key_visualizations(
    output_dir: Path,
    case_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    initial: np.ndarray,
    rule: np.ndarray,
    rl: np.ndarray,
    oracle: np.ndarray,
    slice_reports: list[dict[str, Any]],
) -> None:
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for old_png in vis_dir.glob("*.png"):
        old_png.unlink()
    ranked = sorted(
        slice_reports,
        key=lambda r: abs(float(r["rl_dice_cache_space"]) - float(r["rule_candidate_dice"])),
        reverse=True,
    )
    selected = []
    for report in ranked[:6]:
        selected.append(int(report["slice"]))
    for t in selected:
        _save_six_panel(
            vis_dir / f"case_{case_id}_slice_{t:03d}_initial_rule_rl_oracle.png",
            image[:, :, t],
            gt[:, :, t],
            initial[:, :, t],
            rule[:, :, t],
            rl[:, :, t],
            oracle[:, :, t],
            title=f"case {case_id} slice {t}",
        )


def _save_six_panel(
    path: Path,
    image: np.ndarray,
    gt: np.ndarray,
    initial: np.ndarray,
    rule: np.ndarray,
    rl: np.ndarray,
    oracle: np.ndarray,
    title: str,
) -> None:
    panels = [
        ("image", overlay_mask(image, np.zeros_like(gt, dtype=np.uint8), (0, 0, 0), alpha=0.0)),
        ("gt", overlay_mask(image, gt, (40, 220, 90))),
        ("initial", overlay_mask(image, initial, (240, 80, 70))),
        ("rule", overlay_mask(image, rule, (60, 140, 255))),
        ("rl", overlay_mask(image, rl, (255, 180, 40))),
        ("oracle", overlay_mask(image, oracle, (180, 80, 230))),
    ]
    w, h = panels[0][1].size
    canvas = Image.new("RGB", (w * len(panels), h + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    for i, (label, panel) in enumerate(panels):
        x = i * w
        canvas.paste(panel, (x, 28))
        draw.text((x + 8, 30), label, fill=(255, 255, 255))
    canvas.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN policy on cached CT-SAM candidates.")
    parser.add_argument("--data-dir", default=r".")
    parser.add_argument("--case-id", default="0001")
    parser.add_argument("--liver-label", type=int, default=1)
    parser.add_argument("--cache-dir", default=r".\outputs\rl_cache_case0001_liver")
    parser.add_argument("--policy-path", default=r".\outputs\rl_dqn_case0001_liver_e30\dqn_policy.pt")
    parser.add_argument("--baseline-dir", default=r".\outputs\ablation_case0001_liver\selected_runs\09_pdf_complete_guarded_final")
    parser.add_argument("--output-dir", default=r".\outputs\rl_dqn_case0001_liver_e30_eval")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--window-min", type=float, default=-160.0)
    parser.add_argument("--window-max", type=float, default=240.0)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    output = evaluate_dqn(build_parser().parse_args())
    print(
        json.dumps(
            {
                "output_dir": output["output_dir"],
                "full_volume_metrics": output["full_volume_metrics"],
                "cached_slice_metrics": output["cached_slice_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

