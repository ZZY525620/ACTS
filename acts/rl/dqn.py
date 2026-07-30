from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


class DQNAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=int(capacity))

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, int(batch_size))

    def __len__(self) -> int:
        return len(self.buffer)


def select_action(policy_net: DQNAgent, state: np.ndarray, epsilon: float, action_dim: int, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def train_dqn_one_step(
    policy_net: DQNAgent,
    target_net: DQNAgent,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    transitions = replay_buffer.sample(batch_size)
    states = torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32, device=device)
    actions = torch.as_tensor([t.action for t in transitions], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=device)
    dones = torch.as_tensor([t.done for t in transitions], dtype=torch.float32, device=device)

    q_current = policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        q_next = target_net(next_states).max(dim=1).values
        q_target = rewards + float(gamma) * q_next * (1.0 - dones)

    loss = nn.functional.mse_loss(q_current, q_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())

