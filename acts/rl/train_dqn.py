from __future__ import annotations

import argparse
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
from acts.rl.env import CTSliceAgentEnv


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

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = CTSliceAgentEnv(args.cache_dir, max_steps=args.max_steps)
    policy_net = DQNAgent(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net = DQNAgent(env.state_dim, env.action_dim, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.replay_buffer_size)

    baseline = _baseline_summary(env)
    logs = []
    global_step = 0
    epsilon = args.epsilon_start

    for epoch in range(args.epochs):
        indices = list(range(len(env.samples)))
        random.shuffle(indices)
        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []
        action_counts = {str(i): 0 for i in range(env.action_dim)}
        invalid_count = 0
        accepted_count = 0
        rl_dices: list[float] = []

        for sample_index in indices:
            state = env.reset(sample_index)
            total_reward = 0.0
            for _ in range(args.max_steps):
                action = select_action(policy_net, state, epsilon, env.action_dim, device)
                next_state, reward, done, info = env.step(action)
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
            rl_dices.append(float(env.current_dice))

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        eval_summary = evaluate_policy(env, policy_net, device)
        log = {
            "epoch": epoch + 1,
            "epsilon": float(epsilon),
            "avg_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "avg_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "avg_train_episode_dice": float(np.mean(rl_dices)) if rl_dices else 0.0,
            "invalid_action_ratio": float(invalid_count / max(sum(action_counts.values()), 1)),
            "accepted_action_ratio": float(accepted_count / max(sum(action_counts.values()), 1)),
            "action_counts": action_counts,
            "eval": eval_summary,
        }
        logs.append(log)

    final_eval = evaluate_policy(env, policy_net, device)
    result = {
        "cache_dir": str(args.cache_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "action_names": ACTION_NAMES,
        "baseline": baseline,
        "final_eval": final_eval,
        "logs": logs,
    }
    (output_dir / "training_log.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(policy_net.state_dict(), output_dir / "dqn_policy.pt")
    torch.save(target_net.state_dict(), output_dir / "dqn_target.pt")
    _write_summary(output_dir / "README_DQN_TRAINING.md", result)
    return result


def evaluate_policy(env: CTSliceAgentEnv, policy_net: DQNAgent, device: torch.device) -> dict[str, Any]:
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
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_dice": float(np.mean(dices)) if dices else 0.0,
        "invalid_action_ratio": float(invalid_count / max(sum(action_counts.values()), 1)),
        "accepted_action_ratio": float(accepted_count / max(sum(action_counts.values()), 1)),
        "action_counts": action_counts,
    }


def _baseline_summary(env: CTSliceAgentEnv) -> dict[str, float]:
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
        "avg_current_dice": float(np.mean(current)),
        "avg_rule_candidate_dice": float(np.mean(rule)),
        "avg_oracle_candidate_dice": float(np.mean(oracle)),
    }


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    final_eval = result["final_eval"]
    lines = [
        "# DQN Training Summary",
        "",
        f"Cache: {result['cache_dir']}",
        f"Device: {result['device']}",
        f"State dim: {result['state_dim']}",
        f"Action dim: {result['action_dim']}",
        "",
        "## Baseline",
        "",
        f"- Current avg Dice: {baseline['avg_current_dice']:.6f}",
        f"- Rule avg Dice: {baseline['avg_rule_candidate_dice']:.6f}",
        f"- Oracle avg Dice: {baseline['avg_oracle_candidate_dice']:.6f}",
        "",
        "## Final Eval",
        "",
        f"- RL avg Dice: {final_eval['avg_dice']:.6f}",
        f"- Avg reward: {final_eval['avg_reward']:.6f}",
        f"- Invalid action ratio: {final_eval['invalid_action_ratio']:.6f}",
        f"- Accepted action ratio: {final_eval['accepted_action_ratio']:.6f}",
        "",
        "Goal from PDF: Initial SAM < Rule Selection < RL Agent < Oracle Candidate.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal DQN on cached CT-SAM slice candidates.")
    parser.add_argument("--cache-dir", default=r".\outputs\rl_cache_case0001_liver")
    parser.add_argument("--output-dir", default=r".\outputs\rl_dqn_case0001_liver_smoke")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-buffer-size", type=int, default=10000)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    output = train(build_parser().parse_args())
    print(json.dumps({"output_dir": output["output_dir"], "final_eval": output["final_eval"]}, ensure_ascii=False, indent=2))

