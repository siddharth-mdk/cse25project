"""DQN agent for the BTC Up/Down env (standard / vanilla DQN).

Trains on the shared EpisodeEnv so it is directly comparable to the fair-value
baseline. Uses a small MLP Q-network, experience replay, a target network, and
action masking. The agent is parameterized (learning rate, width, gamma, ...) so
the notebook can train and compare multiple hyperparameter configurations.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config as C
from . import env as E
from . import metrics as M


def default_device() -> str:
    """Prefer Apple MPS, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


class DQNAgent:
    """Standard DQN with replay, a target network, and action masking."""

    def __init__(self, state_size=E.STATE_SIZE, n_actions=E.N_ACTIONS, *,
                 hidden=C.DQN_HIDDEN, lr=C.DQN_LR, gamma=C.DQN_GAMMA, batch=C.DQN_BATCH,
                 buffer=C.DQN_BUFFER, target_update=C.DQN_TARGET_UPDATE, warmup=C.DQN_WARMUP,
                 device=None, seed=C.SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = device or default_device()
        self.gamma, self.batch = gamma, batch
        self.target_update, self.warmup = target_update, warmup
        self.policy = QNet(state_size, n_actions, hidden).to(self.device)
        self.target = QNet(state_size, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer)
        self.n_actions = n_actions
        self.learn_steps = 0

    # ── acting ────────────────────────────────────────────────────────────--
    def act(self, obs: E.Obs, epsilon: float) -> int:
        mask = valid_mask(obs.position)
        valid = np.flatnonzero(mask)
        if random.random() < epsilon:
            return int(random.choice(valid))
        with torch.no_grad():
            q = self.policy(torch.as_tensor(obs.vec, device=self.device)).cpu().numpy()
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    # ── learning (standard DQN target) ───────────────────────────────────────
    def learn(self):
        if len(self.buffer) < max(self.batch, self.warmup):
            return None
        s, a, r, s2, d, m2 = self.buffer.sample(self.batch)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)
        m2 = torch.as_tensor(m2, device=self.device)

        q = self.policy(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            # standard DQN: target net both selects and evaluates (max over valid actions)
            q_next = self.target(s2).masked_fill(~m2, -float("inf")).max(1)[0]
            target = r + self.gamma * q_next * (1.0 - d)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
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
    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def reset(self):
        pass

    def __call__(self, obs: E.Obs) -> int:
        return self.agent.act(obs, epsilon=0.0)


# ── training ────────────────────────────────────────────────────────────────
def _episode_index(eps, manifest, split):
    mids = list(manifest[manifest.split == split].market_id)
    labels = dict(zip(manifest.market_id, manifest.label_up))
    sub = eps[eps.market_id.isin(set(mids))]
    by_mkt = {mid: df for mid, df in sub.groupby("market_id")}
    return mids, by_mkt, labels


def _run_episode_train(agent, env, epsilon):
    obs = env.reset()
    total = 0.0
    while not env.done:
        a = agent.act(obs, epsilon)
        s = obs.vec
        next_obs, r, done, _ = env.step(a)
        agent.buffer.push(s, a, r, next_obs.vec, float(done), valid_mask(next_obs.position))
        agent.learn()
        obs = next_obs
        total += r
    return total


def train(eps, manifest, *, epochs=C.DQN_EPOCHS, lr=C.DQN_LR, hidden=C.DQN_HIDDEN,
          gamma=C.DQN_GAMMA, eps_start=C.DQN_EPS_START, eps_end=C.DQN_EPS_END,
          eps_decay_frac=C.DQN_EPS_DECAY_FRAC, device="cpu", seed=C.SEED,
          label="dqn", save_path=None, verbose=True):
    """Train a DQN; return {agent, history(DataFrame), label}.

    Keeps the best-on-validation weights. `device='cpu'` is the default because
    the network is tiny (MPS is slower here); pass device=None to auto-select.
    """
    tr_mids, tr_by, tr_lab = _episode_index(eps, manifest, "train")
    agent = DQNAgent(hidden=hidden, lr=lr, gamma=gamma, device=device, seed=seed)

    total_eps = epochs * len(tr_mids)
    decay_eps = max(1, int(eps_decay_frac * total_eps))
    rng = random.Random(seed)

    def eps_at(i):
        return max(eps_end, eps_start - (eps_start - eps_end) * i / decay_eps)

    gi, best, best_sd, history = 0, -1e9, None, []
    for epoch in range(1, epochs + 1):
        order = tr_mids[:]
        rng.shuffle(order)
        rewards = []
        for mid in order:
            env = E.EpisodeEnv(tr_by[mid], tr_lab[mid], fee_rate=0.0)
            rewards.append(_run_episode_train(agent, env, eps_at(gi)))
            gi += 1
        val = M.run_backtest(GreedyPolicy(agent), eps, manifest, "val", fee_rate=0.0)
        history.append({"epoch": epoch, "eps": round(eps_at(gi), 4),
                        "train_meanR": float(np.mean(rewards)),
                        "val_mean_pnl": val["mean_pnl"], "val_sharpe": val["sharpe"],
                        "val_win_rate": val["win_rate"]})
        if val["mean_pnl"] > best:
            best = val["mean_pnl"]
            best_sd = {k: v.detach().clone() for k, v in agent.policy.state_dict().items()}
        if verbose:
            print(f"[{label}] epoch {epoch:2d} | eps={eps_at(gi):.3f} "
                  f"| train_meanR={np.mean(rewards):+.4f} "
                  f"| val_meanPnL={val['mean_pnl']:+.4f} sharpe={val['sharpe']:+.3f}")

    if best_sd is not None:  # restore best-on-val weights
        agent.policy.load_state_dict(best_sd)
        agent.target.load_state_dict(best_sd)
    if save_path:
        agent.save(save_path)
    return {"agent": agent, "history": pd.DataFrame(history), "label": label}
