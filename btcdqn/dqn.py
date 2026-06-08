"""Double DQN agent for the BTC Up/Down env.

Trains on the shared EpisodeEnv so it is directly comparable to the fair-value
baseline. Uses:
  - a small MLP Q-network (state -> Q per action),
  - experience replay + a target network,
  - Double DQN targets (policy net selects the action, target net evaluates it),
  - action masking (only valid actions given the current position).
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import env as E


# ── action validity (mirrors env semantics) ────────────────────────────────
def valid_mask(position: int) -> np.ndarray:
    """Boolean mask over [HOLD, BUY_UP, BUY_DOWN, CLOSE] for a given position."""
    m = np.zeros(E.N_ACTIONS, dtype=bool)
    m[E.HOLD] = True
    if position == E.FLAT:
        m[E.BUY_UP] = True
        m[E.BUY_DOWN] = True
    else:
        m[E.CLOSE] = True
    return m


def default_device() -> str:
    """Prefer Apple MPS, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QNet(nn.Module):
    def __init__(self, state_size: int, n_actions: int, hidden: int = C.DQN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = C.DQN_BUFFER):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done, mask2):
        self.buf.append((s, a, r, s2, done, mask2))

    def sample(self, batch: int):
        rows = random.sample(self.buf, batch)
        s, a, r, s2, d, m2 = zip(*rows)
        return (np.array(s, np.float32), np.array(a, np.int64), np.array(r, np.float32),
                np.array(s2, np.float32), np.array(d, np.float32), np.array(m2, bool))

    def __len__(self):
        return len(self.buf)


class DoubleDQNAgent:
    def __init__(self, state_size=E.STATE_SIZE, n_actions=E.N_ACTIONS, seed=C.SEED, device=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = device or default_device()
        self.policy = QNet(state_size, n_actions).to(self.device)
        self.target = QNet(state_size, n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=C.DQN_LR)
        self.buffer = ReplayBuffer()
        self.n_actions = n_actions
        self.learn_steps = 0

    # ── acting ──────────────────────────────────────────────────────────────
    def act(self, obs: E.Obs, epsilon: float) -> int:
        mask = valid_mask(obs.position)
        valid = np.flatnonzero(mask)
        if random.random() < epsilon:
            return int(random.choice(valid))
        with torch.no_grad():
            q = self.policy(torch.as_tensor(obs.vec, device=self.device)).cpu().numpy()
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    # ── learning (Double DQN) ────────────────────────────────────────────────
    def learn(self):
        if len(self.buffer) < max(C.DQN_BATCH, C.DQN_WARMUP):
            return None
        s, a, r, s2, d, m2 = self.buffer.sample(C.DQN_BATCH)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)
        m2 = torch.as_tensor(m2, device=self.device)

        q = self.policy(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            # policy net picks next action (masked), target net evaluates it
            q_next_policy = self.policy(s2).masked_fill(~m2, -float("inf"))
            next_a = q_next_policy.argmax(1, keepdim=True)
            q_next = self.target(s2).gather(1, next_a).squeeze(1)
            target = r + C.DQN_GAMMA * q_next * (1.0 - d)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % C.DQN_TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())


class GreedyPolicy:
    """Wrap a trained agent as a deterministic policy for the metrics backtest."""
    def __init__(self, agent: DoubleDQNAgent):
        self.agent = agent

    def reset(self):
        pass

    def __call__(self, obs: E.Obs) -> int:
        return self.agent.act(obs, epsilon=0.0)
