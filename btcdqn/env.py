"""Shared episode trading engine for BTC Up/Down markets.

This single environment defines the trading dynamics used by BOTH the
fair-value baseline and the DQN, so their comparison is apples-to-apples.

One env instance = one hourly market (one episode). Gym-style reset()/step().

Contract mechanics (binary prediction market):
  - Buy "Up" share at price p_up  -> pays 1 if market resolves Up else 0.
  - Buy "Down" share at price p_dn -> pays 1 if market resolves Down else 0.
  - Closing early sells the held side at its current price.
  - Any open position at the final step resolves at the terminal outcome.

Actions: 0=hold, 1=buy Up, 2=buy Down, 3=close.
Positions: 0=flat, 1=long Up, 2=long Down.
Reward: realized P&L at close / resolution (0 otherwise), minus fees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

# action ids
HOLD, BUY_UP, BUY_DOWN, CLOSE = 0, 1, 2, 3
N_ACTIONS = 4
# position ids
FLAT, LONG_UP, LONG_DOWN = 0, 1, 2

Z_COLS = [c + "_z" for c in C.FEATURE_COLS]
# state = z-features + [is_up, is_down, unrealized_pnl]
STATE_SIZE = len(Z_COLS) + 3


@dataclass
class Obs:
    """Observation passed to a policy: numeric vector for the DQN, raw row for rules."""
    vec: np.ndarray          # (STATE_SIZE,) float32 — DQN input
    row: pd.Series           # raw episode row — fair-value rules read spot/up_price/tau
    position: int
    entry_price: float
    step: int


def fee(price: float, fee_rate: float) -> float:
    """Polymarket-style symmetric fee on one share (0 when fee_rate=0)."""
    if fee_rate <= 0:
        return 0.0
    return fee_rate * min(price, 1.0 - price)


class EpisodeEnv:
    def __init__(self, episode_df: pd.DataFrame, label_up: int, fee_rate: float = 0.0):
        self.df = episode_df.sort_values("step").reset_index(drop=True)
        self.label_up = int(label_up)
        self.fee_rate = float(fee_rate)
        self.n = len(self.df)
        self._z = self.df[Z_COLS].to_numpy(dtype=np.float32)
        self._up = self.df["up_price_raw"].to_numpy(dtype=float)
        self._dn = self.df["down_price"].to_numpy(dtype=float)
        self._tau = self.df["tau"].to_numpy(dtype=float)
        self.reset()

    # ── helpers ────────────────────────────────────────────────────────────
    def _price(self, side: int, i: int) -> float:
        return self._up[i] if side == LONG_UP else self._dn[i]

    def _unrealized(self, i: int) -> float:
        if self.position == FLAT:
            return 0.0
        return self._price(self.position, i) - self.entry_price

    def _obs(self) -> Obs:
        i = self.i
        extra = np.array(
            [self.position == LONG_UP, self.position == LONG_DOWN, self._unrealized(i)],
            dtype=np.float32,
        )
        vec = np.concatenate([self._z[i], extra]).astype(np.float32)
        return Obs(vec=vec, row=self.df.iloc[i], position=self.position,
                   entry_price=self.entry_price, step=i)

    # ── gym API ──────────────────────────────────────────────────────────--
    def reset(self) -> Obs:
        self.i = 0
        self.position = FLAT
        self.entry_price = 0.0
        self.done = False
        self.trades: list[dict] = []
        return self._obs()

    def step(self, action: int):
        """Apply action at current step, advance time, return (obs, reward, done, info)."""
        assert not self.done, "step() called on a finished episode"
        i = self.i
        reward = 0.0

        # open
        if action == BUY_UP and self.position == FLAT:
            self.position, self.entry_price = LONG_UP, self._up[i]
            reward -= fee(self.entry_price, self.fee_rate)
            self.trades.append({"step": i, "act": "buy_up", "price": self.entry_price})
        elif action == BUY_DOWN and self.position == FLAT:
            self.position, self.entry_price = LONG_DOWN, self._dn[i]
            reward -= fee(self.entry_price, self.fee_rate)
            self.trades.append({"step": i, "act": "buy_down", "price": self.entry_price})
        # close early
        elif action == CLOSE and self.position != FLAT:
            exit_price = self._price(self.position, i)
            pnl = exit_price - self.entry_price - fee(exit_price, self.fee_rate)
            reward += pnl
            self.trades.append({"step": i, "act": "close", "price": exit_price, "pnl": pnl})
            self.position, self.entry_price = FLAT, 0.0

        # advance / terminal resolution
        if i >= self.n - 1:
            self.done = True
            if self.position != FLAT:
                won = (self.position == LONG_UP and self.label_up == 1) or (
                    self.position == LONG_DOWN and self.label_up == 0
                )
                payoff = 1.0 if won else 0.0
                pnl = payoff - self.entry_price  # no fee on resolution payout
                reward += pnl
                self.trades.append({"step": i, "act": "resolve", "payoff": payoff, "pnl": pnl})
                self.position, self.entry_price = FLAT, 0.0
            return self._obs(), reward, True, {"trades": self.trades}

        self.i += 1
        return self._obs(), reward, False, {}


def run_policy(env: EpisodeEnv, policy) -> dict:
    """Roll a policy through one episode. policy(obs)->action; optional policy.reset()."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset()
    total = 0.0
    while not env.done:
        action = policy(obs)
        obs, r, done, info = env.step(action)
        total += r
    closed = [t for t in env.trades if "pnl" in t]
    return {
        "pnl": total,
        "n_trades": len(closed),
        "trades": env.trades,
        "traded": len(closed) > 0,
        "wins": sum(t["pnl"] > 0 for t in closed),
    }
