"""Phase 0 dataset builder.

Turns the raw HuggingFace BTC 1-hour Up/Down markets + Coinbase spot into a
clean, fixed-length, per-episode dataset ready for the env / baseline / DQN.

Pipeline:
  1. download raw HF parquet (prices, markets) -> data/raw/
  2. select cleanly-resolved ~1h markets, derive labels & resolution hour H
  3. for each market, build the resolution-hour episode on a 1-min grid:
       contract up/down price, spot, spot/contract features, tau, label
  4. chronological split by H; z-score features on TRAIN stats only
  5. cache episodes.parquet + manifest.parquet + norm_stats.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import requests

from . import config as C
from . import spot as spot_mod

EPISODES_PATH = os.path.join(C.PROC_DIR, "episodes.parquet")
MANIFEST_PATH = os.path.join(C.PROC_DIR, "manifest.parquet")
NORM_PATH = os.path.join(C.PROC_DIR, "norm_stats.json")


# ── 1. raw download ────────────────────────────────────────────────────────
def download_raw(refresh: bool = False) -> dict[str, str]:
    paths = {}
    for name, rel in C.HF_FILES.items():
        dst = os.path.join(C.RAW_DIR, f"{name}.parquet")
        if not os.path.exists(dst) or refresh:
            r = requests.get(C.HF_BASE + rel, timeout=600)
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)
            print(f"[raw] downloaded {name} ({len(r.content)/1e6:.1f} MB)")
        paths[name] = dst
    return paths


# ── 2. market selection + labels ───────────────────────────────────────────
def select_markets() -> pd.DataFrame:
    paths = download_raw()
    px = pd.read_parquet(paths["prices_btc_1h"]).sort_values(["market_id", "timestamp"])
    g = px.groupby("market_id")
    summ = pd.DataFrame(
        {
            "term": g.up_price.last(),
            "first_ts": g.timestamp.first(),
            "last_ts": g.timestamp.last(),
            "npts": g.size(),
        }
    )
    summ["dur_h"] = (summ.last_ts - summ.first_ts) / 3600.0

    # cleanly resolved (terminal price collapsed to ~0/1)
    converged = (summ.term < C.CONVERGE_LO) | (summ.term > C.CONVERGE_HI)
    # genuine ~1h markets
    good_dur = summ.dur_h.between(C.MIN_DURATION_H, C.MAX_DURATION_H)
    sel = summ[converged & good_dur].copy()

    sel["label_up"] = (sel.term > 0.5).astype(int)
    # The contract trades during [last_ts - 1h, last_ts]; last_ts is the
    # resolution time. (Diagnosed: last_ts sits ~59 min past the floored hour,
    # so the tradeable hour is [last_ts-1h, last_ts], NOT [floor-1h, floor].)
    sel["H"] = sel.last_ts                       # resolution timestamp (path end)
    sel["ref_ts"] = sel.H - C.EPISODE_SECONDS    # window start (= hour open, fair-value ref)
    sel = sel.reset_index().rename(columns={"index": "market_id"})
    if "market_id" not in sel.columns:
        sel = sel.rename(columns={sel.columns[0]: "market_id"})
    print(
        f"[select] {len(summ)} markets -> {converged.sum()} converged -> "
        f"{len(sel)} after duration filter"
    )
    return px, sel


# ── 3. per-episode grid build ──────────────────────────────────────────────
def _align_at_or_before(path_ts: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """For each grid time, take the last value at-or-before it (step / ffill)."""
    idx = np.searchsorted(path_ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(path_ts) - 1)
    return values[idx]


def build_episode(mkt_px: pd.DataFrame, row, spot_at) -> pd.DataFrame | None:
    H, ref_ts = int(row.H), int(row.ref_ts)
    grid = ref_ts + np.arange(C.N_STEPS) * C.GRID_STEP_S  # [H-1h, H)

    # contract points actually inside the resolution hour (data-quality gate)
    in_win = mkt_px[(mkt_px.timestamp >= ref_ts) & (mkt_px.timestamp <= H)]
    if len(in_win) < C.MIN_POINTS_IN_WINDOW:
        return None

    path_ts = mkt_px.timestamp.to_numpy()
    up = _align_at_or_before(path_ts, mkt_px.up_price.to_numpy(), grid)
    down = _align_at_or_before(path_ts, mkt_px.down_price.to_numpy(), grid)

    # Spot features use real pre-window history (we fetched padded spot), so the
    # first steps of the episode aren't degenerate. Compute on an extended grid
    # [ref_ts - lookback, H) then slice to the episode window.
    lookback = max(C.SPOT_RET_WINDOWS + (C.SPOT_VOL_WINDOW,))
    ext_grid = (ref_ts - lookback * C.GRID_STEP_S) + np.arange(C.N_STEPS + lookback) * C.GRID_STEP_S
    ext_spot = np.array([spot_at(int(t)) for t in ext_grid], dtype=float)
    es = pd.Series(ext_spot)
    spot_ret_5 = es.pct_change(5).to_numpy()[lookback:]
    spot_ret_15 = es.pct_change(15).to_numpy()[lookback:]
    spot_vol = es.pct_change().rolling(C.SPOT_VOL_WINDOW).std().to_numpy()[lookback:]

    spot = ext_spot[lookback:]
    ref_spot = spot_at(ref_ts)
    spot_vs_ref = (spot / ref_spot) - 1.0
    spot_ret_5 = np.nan_to_num(spot_ret_5)
    spot_ret_15 = np.nan_to_num(spot_ret_15)
    spot_vol = np.nan_to_num(spot_vol)

    # contract features
    u = pd.Series(up)
    up_mom_5 = (u - u.shift(C.CONTRACT_MOM_WINDOW)).fillna(0.0)

    tau = (H - grid) / C.EPISODE_SECONDS  # time-to-resolution, ~1 -> ~0

    return pd.DataFrame(
        {
            "market_id": row.market_id,
            "step": np.arange(C.N_STEPS),
            "t": grid,
            "tau": tau,
            "up_price": up,
            "down_price": down,
            "spot": spot,
            "spot_ret_5": spot_ret_5,
            "spot_ret_15": spot_ret_15,
            "spot_vol": spot_vol,
            "spot_vs_ref": spot_vs_ref,
            "up_mom_5": up_mom_5.to_numpy(),
            "label_up": int(row.label_up),
        }
    )


# ── 4 & 5. split, normalize, save ──────────────────────────────────────────
def _assign_split(manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest.sort_values("H").reset_index(drop=True)
    n = len(m)
    i1 = int(n * C.TRAIN_FRAC)
    i2 = int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
    m["split"] = "test"
    m.loc[: i1 - 1, "split"] = "train"
    m.loc[i1:i2 - 1, "split"] = "val"
    return m


def build(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(EPISODES_PATH) and not refresh:
        return pd.read_parquet(EPISODES_PATH), pd.read_parquet(MANIFEST_PATH)

    px, sel = select_markets()
    spot_df = spot_mod.fetch_spot(int(sel.ref_ts.min()), int(sel.H.max()))
    spot_at = spot_mod.spot_lookup(spot_df)

    episodes = []
    kept = []
    px_by_mkt = dict(tuple(px.groupby("market_id")))
    for row in sel.itertuples(index=False):
        mkt_px = px_by_mkt.get(row.market_id)
        if mkt_px is None:
            continue
        ep = build_episode(mkt_px, row, spot_at)
        if ep is not None:
            episodes.append(ep)
            kept.append(row.market_id)

    eps = pd.concat(episodes, ignore_index=True)
    manifest = sel[sel.market_id.isin(kept)][
        ["market_id", "H", "ref_ts", "label_up", "npts", "dur_h"]
    ].copy()
    manifest = _assign_split(manifest)
    print(f"[build] kept {len(manifest)} episodes "
          f"({(manifest.split=='train').sum()}/{(manifest.split=='val').sum()}/"
          f"{(manifest.split=='test').sum()} train/val/test)")

    # attach split to episode rows
    eps = eps.merge(manifest[["market_id", "split"]], on="market_id", how="left")

    # z-score features on TRAIN stats only
    eps["up_price_raw"] = eps["up_price"]  # keep raw for env P&L
    train_mask = eps.split == "train"
    norm = {}
    for col in C.FEATURE_COLS:
        mu = float(eps.loc[train_mask, col].mean())
        sd = float(eps.loc[train_mask, col].std()) or 1.0
        eps[col + "_z"] = (eps[col] - mu) / sd
        norm[col] = {"mean": mu, "std": sd}

    eps.to_parquet(EPISODES_PATH, index=False)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    with open(NORM_PATH, "w") as f:
        json.dump(norm, f, indent=2)
    print(f"[build] saved -> {EPISODES_PATH}\n               {MANIFEST_PATH}\n               {NORM_PATH}")
    return eps, manifest


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience loader for the notebook / later phases."""
    return pd.read_parquet(EPISODES_PATH), pd.read_parquet(MANIFEST_PATH)
