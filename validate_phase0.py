"""Sanity checks for the Phase 0 episode dataset.

Run after build_data.py. Verifies structural integrity, label consistency,
absence of leakage, and that the data reflects the validated resolution rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btcdqn import config as C
from btcdqn import data

eps, manifest = data.load()
ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


print("=== Phase 0 validation ===")
print(f"episodes={manifest.market_id.nunique():,}  rows={len(eps):,}")

# 1. structure: every episode has exactly N_STEPS steps
sizes = eps.groupby("market_id").size()
check("every episode has N_STEPS rows", (sizes == C.N_STEPS).all(),
      f"sizes in [{sizes.min()},{sizes.max()}], expected {C.N_STEPS}")

# 2. no NaNs / infs in modelling columns
zcols = [c for c in eps.columns if c.endswith("_z")]
modcols = zcols + ["up_price_raw", "up_price", "down_price", "tau"]
finite = np.isfinite(eps[modcols].to_numpy()).all()
check("no NaN/inf in modelling columns", bool(finite))

# 3. label consistency: terminal contract price matches label
term = eps.sort_values("step").groupby("market_id").tail(1)
agree = ((term.up_price_raw > 0.5).astype(int) == term.label_up).mean()
check("terminal up_price matches label", agree > 0.99, f"{agree:.1%} agree")

# 4. validated resolution rule: spot(end) > spot(start) <=> label_up
first = eps.sort_values("step").groupby("market_id").first()
last = eps.sort_values("step").groupby("market_id").last()
spot_up = (last.spot.values > first.spot.values).astype(int)
rule = (spot_up == manifest.set_index("market_id").loc[first.index, "label_up"].values).mean()
check("spot move agrees with label (rule)", rule > 0.85, f"{rule:.1%} (in-window spot)")

# 5. no temporal leakage: split time ranges are ordered & disjoint
ranges = manifest.groupby("split", observed=True).H.agg(["min", "max"])
tr, va, te = ranges.loc["train"], ranges.loc["val"], ranges.loc["test"]
check("train < val < test in time", tr["max"] <= va["min"] and va["max"] <= te["min"])

# 6. normalization: train z-features ~ mean 0 / std 1
tr_eps = eps[eps.split == "train"]
mu = tr_eps[zcols].mean().abs().max()
sd = tr_eps[zcols].std()
check("train z-features standardized", mu < 1e-6 and (sd.sub(1).abs() < 1e-3).all(),
      f"|mean|max={mu:.1e}, std range [{sd.min():.3f},{sd.max():.3f}]")

# 7. learnable signal: contract price separates by outcome and sharpens near tau->0
early = eps[eps.step < 5].groupby("label_up").up_price_raw.mean()
latep = eps[eps.step >= C.N_STEPS - 5].groupby("label_up").up_price_raw.mean()
sep_e = early.get(1, np.nan) - early.get(0, np.nan)
sep_l = latep.get(1, np.nan) - latep.get(0, np.nan)
check("contract price sharpens toward resolution", sep_l > sep_e,
      f"up-vs-down price gap: early={sep_e:.2f} -> late={sep_l:.2f}")

print("\nlabel balance by split:")
print(manifest.groupby("split", observed=True).label_up.mean().round(3).to_string())
print("\n=== {} ===".format("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
