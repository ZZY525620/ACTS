from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acts.rl.actions import ACTION_NAMES
from acts.rl.dqn import DQNAgent, ReplayBuffer, select_action, train_dqn_one_step
from acts.rl.multi_env import CTMultiCacheSliceAgentEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cache_dirs = [Path(p) for p in args.train_cache_dirs]
    eval_cache_dirs = [Path(p) for p in (args.eval_cache_dirs or args.train_cache_dirs)]
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_env = CTMultiCacheSliceAgentEnv(train_cache_dirs, max_steps=args.max_steps)
    policy_net = DQNAgent(train_env.state_dim, train_env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = DQNAgent(train_env.state_dim, train_env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    baseline = baseline_summary(train_env)
    logs = []
    global_step = 0
    epsilon = args.epsilon_start

    for epoch in range(args.epochs):
        indices = list(range(len(train_env.samples)))
        random.shuffle(indices)
        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []
        action_counts = {str(i): 0 for i in range(train_env.action_dim)}
        invalid_count = 0
        accepted_count = 0
        train_dices: list[float] = []

        for sample_index in indices:
            state = train_env.reset(sample_index)
            total_reward = 0.0
            for _ in range(args.max_steps):
                action = select_action(policy_net, state, epsilon, train_env.action_dim, device)
                next_state, reward, done, info = train_env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                action_counts[str(action)] += 1
                invalid_count += int(bool(info.get("invalid_action", False)))
                accepted_count += int(bool(info.get("accepted", False)))
                total_reward += float(reward)
                state = next_state
                global_step += 1

                if len(replay_buffer) >= args.batch_size:
                    loss = train_dqn_one_step(
                        policy_net,
                        target_net,
                        optimizer,
                        replay_buffer,
                        args.batch_size,
                        args.gamma,
                        device,
                    )
                    epoch_losses.append(loss)

                if global_step % args.target_update_interval == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                if done:
                    break

            epoch_rewards.append(total_reward)
            train_dices.append(float(train_env.current_dice))

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        eval_summary = evaluate_policy(train_env, policy_net, device)
        logs.append(
            {
                "epoch": epoch + 1,
                "epsilon": float(epsilon),
                "avg_reward": _mean(epoch_rewards),
                "avg_loss": _mean(epoch_losses),
                "avg_train_episode_dice": _mean(train_dices),
                "invalid_action_ratio": float(invalid_count / max(sum(action_counts.values()), 1)),
                "accepted_action_ratio": float(accepted_count / max(sum(action_counts.values()), 1)),
                "action_counts": action_counts,
                "train_eval": eval_summary,
            }
        )

    final_train_eval = evaluate_policy(train_env, policy_net, device)
    per_case_eval = []
    for cache_dir in eval_cache_dirs:
        env = CTMultiCacheSliceAgentEnv([cache_dir], max_steps=args.max_steps)
        row = {
            "case_id": _case_id_from_env(env),
            "cache_dir": str(cache_dir),
            "split": "train" if cache_dir in train_cache_dirs else "eval",
            **baseline_summary(env),
            **_prefix_keys(evaluate_policy(env, policy_net, device), "rl_"),
        }
        per_case_eval.append(row)

    result = {
        "train_cache_dirs": [str(p) for p in train_cache_dirs],
        "eval_cache_dirs": [str(p) for p in eval_cache_dirs],
        "output_dir": str(output_dir),
        "device": str(device),
        "state_dim": train_env.state_dim,
        "action_dim": train_env.action_dim,
        "action_names": ACTION_NAMES,
        "train_baseline": baseline,
        "final_train_eval": final_train_eval,
        "per_case_eval": per_case_eval,
        "logs": logs,
        "note": "Multi-case offline DQN policy network trained on cached SAM/non-SAM liver candidates.",
    }

    (output_dir / "training_log.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(policy_net.state_dict(), output_dir / "dqn_policy.pt")
    torch.save(target_net.state_dict(), output_dir / "dqn_target.pt")
    write_eval_csv(output_dir / "eval_table.csv", per_case_eval)
    write_summary(output_dir / "README_DQN_MULTICASE.md", result)
    return result


def evaluate_policy(env: CTMultiCacheSliceAgentEnv, policy_net: DQNAgent, device: torch.device) -> dict[str, Any]:
    rewards: list[float] = []
    dices: list[float] = []
    action_counts = {str(i): 0 for i in range(env.action_dim)}
    invalid_count = 0
    accepted_count = 0

    for sample_index in range(len(env.samples)):
        state = env.reset(sample_index)
        total_reward = 0.0
        for _ in range(env.max_steps):
            action = select_action(policy_net, state, epsilon=0.0, action_dim=env.action_dim, device=device)
            next_state, reward, done, info = env.step(action)
            action_counts[str(action)] += 1
            invalid_count += int(bool(info.get("invalid_action", False)))
            accepted_count += int(bool(info.get("accepted", False)))
            total_reward += float(reward)
            state = next_state
            if done:
                break
        rewards.append(total_reward)
        dices.append(float(env.current_dice))

    return {
        "avg_reward": _mean(rewards),
        "avg_dice": _mean(dices),
        "invalid_action_ratio": float(invalid_count / max(sum(action_counts.values()), 1)),
        "accepted_action_ratio": float(accepted_count / max(sum(action_counts.values()), 1)),
        "action_counts": action_counts,
    }


def baseline_summary(env: CTMultiCacheSliceAgentEnv) -> dict[str, float]:
    current = []
    rule = []
    oracle = []
    for sample in env.samples:
        current.append(float(sample["current_dice"]))
        rule_record = sample["candidates"][int(sample["rule_candidate_index"])]
        oracle_record = sample["candidates"][int(sample["oracle_candidate_index"])]
        rule.append(float(rule_record["dice_to_gt"]))
        oracle.append(float(oracle_record["dice_to_gt"]))
    return {
        "avg_current_dice": _mean(current),
        "avg_rule_candidate_dice": _mean(rule),
        "avg_oracle_candidate_dice": _mean(oracle),
        "num_samples": len(env.samples),
    }


def write_eval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "split",
        "num_samples",
        "avg_current_dice",
        "avg_rule_candidate_dice",
        "rl_avg_dice",
        "avg_oracle_candidate_dice",
        "rl_invalid_action_ratio",
        "rl_accepted_action_ratio",
        "cache_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Multi-case DQN Policy Network",
        "",
        f"Device: {result['device']}",
        f"State dim: {result['state_dim']}",
        f"Action dim: {result['action_dim']}",
        "",
        "## Train baseline",
        "",
        f"- Current avg Dice: {result['train_baseline']['avg_current_dice']:.6f}",
        f"- Rule avg Dice: {result['train_baseline']['avg_rule_candidate_dice']:.6f}",
        f"- Oracle avg Dice: {result['train_baseline']['avg_oracle_candidate_dice']:.6f}",
        "",
        "## Final train eval",
        "",
        f"- DQN avg Dice: {result['final_train_eval']['avg_dice']:.6f}",
        f"- Invalid action ratio: {result['final_train_eval']['invalid_action_ratio']:.6f}",
        f"- Accepted action ratio: {result['final_train_eval']['accepted_action_ratio']:.6f}",
        "",
        "## Per-case eval",
        "",
    ]
    for row in result["per_case_eval"]:
        lines.append(
            "- "
            f"{row['case_id']} ({row['split']}): "
            f"Current {row['avg_current_dice']:.6f}, "
            f"Rule {row['avg_rule_candidate_dice']:.6f}, "
            f"DQN {row['rl_avg_dice']:.6f}, "
            f"Oracle {row['avg_oracle_candidate_dice']:.6f}"
        )
    lines.extend(
        [
            "",
            "Outputs:",
            "",
            "- `dqn_policy.pt`: trained DQN policy network.",
            "- `training_log.json`: full training/evaluation log.",
            "- `eval_table.csv`: compact per-case result table.",
            "",
            "This is an offline cached-candidate DQN agent. PPO/online SAM interaction can be compared later.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _case_id_from_env(env: CTMultiCacheSliceAgentEnv) -> str:
    if env.case_metadata:
        return str(env.case_metadata[0].get("case_id", "unknown"))
    return "unknown"


def _prefix_keys(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multi-case DQN policy on cached CT-SAM liver candidates.")
    parser.add_argument(
        "--train-cache-dirs",
        nargs="+",
        default=[
            r".\outputs\rl_cache_case0001_liver",
            r".\outputs\rl_cache_case0002_liver",
        ],
    )
    parser.add_argument(
        "--eval-cache-dirs",
        nargs="*",
        default=[
            r".\outputs\rl_cache_case0001_liver",
            r".\outputs\rl_cache_case0002_liver",
            r".\outputs\rl_cache_case0003_liver",
        ],
    )
    parser.add_argument("--output-dir", default=r".\outputs\rl_dqn_liver_multicase_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-buffer-size", type=int, default=20000)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.08)
    parser.add_argument("--epsilon-decay", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    output = train(build_parser().parse_args())
    print(json.dumps({"output_dir": output["output_dir"], "per_case_eval": output["per_case_eval"]}, ensure_ascii=False, indent=2))

