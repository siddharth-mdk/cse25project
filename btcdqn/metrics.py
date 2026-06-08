"""Shared backtest + metric suite, used identically by baseline and DQN."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import env as E


def run_backtest(policy, eps: pd.DataFrame, manifest: pd.DataFrame,
                 split: str, fee_rate: float = 0.0) -> dict:
    """Run a policy over every episode in `split`; return per-episode + aggregate stats."""
    mids = manifest[manifest.split == split]
    labels = dict(zip(manifest.market_id, manifest.label_up))
    eps_split = eps[eps.market_id.isin(set(mids.market_id))]
    by_mkt = dict(tuple(eps_split.groupby("market_id")))

    pnls, traded, n_tr, wins = [], 0, 0, 0
    for mid in mids.market_id:
        ep = by_mkt.get(mid)
        if ep is None:
            continue
        env = E.EpisodeEnv(ep, labels[mid], fee_rate=fee_rate)
        res = E.run_policy(env, policy)
        pnls.append(res["pnl"])
        traded += int(res["traded"])
        n_tr += res["n_trades"]
        wins += res["wins"]

    pnls = np.array(pnls, dtype=float)
    n = len(pnls)
    return {
        "split": split,
        "episodes": n,
        "total_pnl": float(pnls.sum()),
        "mean_pnl": float(pnls.mean()) if n else 0.0,
        "pnl_std": float(pnls.std()) if n else 0.0,
        "sharpe": float(pnls.mean() / pnls.std()) if n and pnls.std() > 0 else 0.0,
        "n_trades": n_tr,
        "trade_rate": traded / n if n else 0.0,
        "pnl_per_trade": float(pnls.sum() / n_tr) if n_tr else 0.0,
        "win_rate": wins / n_tr if n_tr else float("nan"),
        "_pnls": pnls,
    }


def metrics_table(results: list[dict]) -> str:
    cols = ["episodes", "total_pnl", "mean_pnl", "sharpe", "n_trades",
            "trade_rate", "pnl_per_trade", "win_rate"]
    rows = []
    for r in results:
        rows.append([r["name"], r["episodes"], f"{r['total_pnl']:+.2f}",
                     f"{r['mean_pnl']:+.4f}", f"{r['sharpe']:+.3f}", r["n_trades"],
                     f"{r['trade_rate']:.0%}", f"{r['pnl_per_trade']:+.4f}",
                     f"{r['win_rate']:.1%}" if not np.isnan(r["win_rate"]) else "—"])
    df = pd.DataFrame(rows, columns=["policy"] + cols)
    return df.to_string(index=False)


def fair_value_calibration(policy, eps: pd.DataFrame, manifest: pd.DataFrame,
                           split: str, n_bins: int = 10) -> pd.DataFrame:
    """Reliability of the fair-value estimate vs realized outcomes (Tier 1)."""
    mids = set(manifest[manifest.split == split].market_id)
    labels = dict(zip(manifest.market_id, manifest.label_up))
    sub = eps[eps.market_id.isin(mids)]
    fv = sub.apply(lambda r: policy.fair_value_up(r), axis=1).to_numpy()
    y = sub.market_id.map(labels).to_numpy().astype(float)
    bins = np.clip((fv * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.sum():
            out.append({"bin": f"{b/n_bins:.1f}-{(b+1)/n_bins:.1f}",
                        "n": int(m.sum()), "pred_fv": round(fv[m].mean(), 3),
                        "actual_up": round(y[m].mean(), 3)})
    return pd.DataFrame(out)
