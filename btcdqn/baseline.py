"""Baseline trading policies for the BTC Up/Down env.

A policy is a callable obs -> action (see env.Obs). The fair-value policies
compute a spot-implied probability that the market resolves Up, then trade the
gap against the live contract price.

Tier 0 (trivial):  FV_up = 1[spot > ref]            -- "follow spot direction"
Tier 1 (real):     FV_up = Phi( ln(spot/ref) / (sigma * sqrt(tau_hours)) )
                   i.e. P(spot ends above the hour's open) under a driftless
                   Gaussian log-return with per-hour vol sigma*sqrt(remaining).
"""
from __future__ import annotations

import math
import random

import numpy as np

from . import config as C
from . import env as E


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ── trivial baselines ──────────────────────────────────────────────────────
class RandomPolicy:
    """Uniformly random over valid-ish actions; open positions resolve at end."""
    def __init__(self, seed: int = C.SEED):
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def __call__(self, obs: E.Obs) -> int:
        if obs.position == E.FLAT:
            return self.rng.choice([E.HOLD, E.BUY_UP, E.BUY_DOWN])
        return self.rng.choice([E.HOLD, E.CLOSE])


class AlwaysSidePolicy:
    """Buy a fixed side on the first step and hold to resolution (buy-and-hold)."""
    def __init__(self, side: int):
        self.side = side  # E.BUY_UP or E.BUY_DOWN

    def reset(self):
        pass

    def __call__(self, obs: E.Obs) -> int:
        return self.side if obs.position == E.FLAT and obs.step == 0 else E.HOLD


# ── fair-value baseline ────────────────────────────────────────────────────
class FairValuePolicy:
    """Trade the gap between a spot-implied fair value and the contract price.

    Args:
      tier: 0 (sign of spot move) or 1 (Gaussian probability).
      threshold: only trade when |contract - FV| exceeds this edge.
      sigma_per_min: per-minute BTC log-return vol (Tier 1 only; from train).
      exit_on_converge: if True, close once the gap shrinks below threshold/2;
                        if False (default), hold to resolution.
    """
    def __init__(self, tier: int = 1, threshold: float = 0.05,
                 sigma_per_min: float = 0.0008, exit_on_converge: bool = False):
        self.tier = tier
        self.threshold = threshold
        self.sigma_per_min = sigma_per_min
        self.exit_on_converge = exit_on_converge

    def reset(self):
        pass

    def fair_value_up(self, row) -> float:
        ln_ratio = math.log1p(float(row["spot_vs_ref"]))  # ln(spot/ref)
        if self.tier == 0:
            return 1.0 if ln_ratio > 0 else 0.0
        tau_hours = max(float(row["tau"]), 1e-6)
        remaining_min = tau_hours * 60.0
        sigma_rem = self.sigma_per_min * math.sqrt(remaining_min)
        if sigma_rem <= 1e-12:
            return 1.0 if ln_ratio > 0 else 0.0
        return _phi(ln_ratio / sigma_rem)

    def __call__(self, obs: E.Obs) -> int:
        row = obs.row
        fv_up = self.fair_value_up(row)
        up_price = float(row["up_price_raw"])

        if obs.position == E.FLAT:
            if up_price < fv_up - self.threshold:      # Up underpriced
                return E.BUY_UP
            if up_price > fv_up + self.threshold:      # Up overpriced -> Down underpriced
                return E.BUY_DOWN
            return E.HOLD

        # holding
        if self.exit_on_converge and abs(up_price - fv_up) < self.threshold / 2:
            return E.CLOSE
        return E.HOLD


def estimate_sigma_per_min(eps_train) -> float:
    """Per-minute BTC log-return vol from the training episodes (for Tier 1)."""
    g = eps_train.sort_values(["market_id", "step"]).groupby("market_id")
    rets = g.spot.apply(lambda s: np.log(s).diff()).to_numpy()
    rets = rets[np.isfinite(rets)]
    return float(np.std(rets))
