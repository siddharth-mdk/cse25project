# BTC Up/Down Prediction-Market DQN

Train a DQN to trade Polymarket **BTC "Up or Down" 1-hour** contracts, and
compare it against a principled **fair-value** baseline. Code lives in Python
modules (`btcdqn/`); the final DQN + ablations/evals live in one notebook.

## Approach
Each hourly market is an **episode**. The contract price (`up_price` ∈ [0,1])
is the tradeable asset; it resolves to 0/1 at the end of the hour based on
whether BTC spot rose over that hour. The agent trades **two-sided** (buy
Up/Down, close) and is rewarded by realized P&L + terminal resolution.

## Data pipeline (Phase 0 — done)
Source: HuggingFace `BrockMisner/polymarket-btc-updown` (contract prices) +
**Coinbase** 1-min spot (Binance is geo-blocked; yfinance 1-min only spans ~7d).

Key validated facts:
- **Resolution rule:** market resolves **Up iff spot(last_ts) > spot(last_ts − 1h)**
  (100% match vs derived labels using Coinbase spot; 92.6% cross-check with
  yfinance hourly).
- **Labels** taken from the contract's terminal price (converged to ~0/1).
- **Episode window** = `[last_ts − 1h, last_ts]` (the tradeable hour), on a fixed
  1-min grid (`N_STEPS = 60`).
- **1,609** clean episodes, balanced (~50% Up), chronological split
  **1126 / 241 / 242** (train Dec'25–Feb'26, val Feb–Mar, test Mar).

### Build / verify
```bash
python build_data.py            # build (caches raw + spot)
python build_data.py --refresh  # force re-download + rebuild
python validate_phase0.py       # 7 integrity / leakage / label checks
```

### Outputs (`data/processed/`)
- `episodes.parquet` — long format: `market_id, step, t, tau, up_price,
  down_price, spot, <features>, <features>_z, up_price_raw, label_up, split`
- `manifest.parquet` — per-episode: `market_id, H, ref_ts, label_up, split, …`
- `norm_stats.json` — train-only z-score mean/std per feature

Features (z-scored on **train only**): `spot_ret_5, spot_ret_15, spot_vol,
spot_vs_ref, up_mom_5, up_price, tau`. `up_price_raw` is kept unnormalized for
env P&L.

## Roadmap
- **Phase 0 — Data pipeline** ✅ `btcdqn/{config,spot,data}.py`, `build_data.py`
- **Phase 1 — Fair-value baseline** ✅ `btcdqn/{env,baseline,metrics}.py`,
  `run_baselines.py`. Tier 0 (`FV = 1[spot>ref]`, degenerate ≈ always-Down) and
  Tier 1 (`FV = Φ(ln(spot/ref)/(σ√τ))`). Tier 1 is clearly profitable and well
  calibrated; **Tier 1 + early-exit** reaches mean P&L ≈ +0.43/episode,
  Sharpe ≈ 0.96, win-rate ≈ 75% on test (fee = 0). This is the DQN's bar.
- **Phase 2 — Env** ✅ (built early as the shared engine) `btcdqn/env.py` —
  gym-style, two-sided actions, state includes position + unrealized P&L + `tau`;
  baseline and DQN use identical dynamics.
- **Phase 3 — DQN** ✅ `btcdqn/dqn.py`, `train_dqn.py` — Double DQN, replay +
  target net, action masking, MPS/CPU. Stable val Sharpe ~2.1 across epochs;
  **test: mean P&L +0.98, Sharpe 2.20, win 79.5%** (beats fairval_t1+exit's
  +0.43 / 0.96). Best-on-val checkpoint → `data/processed/dqn_best.pt`,
  per-epoch history → `dqn_history.json`. Device auto-selects MPS; `--device cpu`
  is ~5x faster for this tiny MLP.
- **Phase 4 — Notebook** ✅ `dqn_experiments.ipynb` (built by `build_notebook.py`).
  Loads the checkpoint; test comparison, cumulative-P&L, calibration, DQN
  behavior, example episode, and **σ + threshold sweep** ablations. Rebuild:
  `python build_notebook.py`; execute with the `cse25` kernel.

All P&L is at **fee = 0** (upper bound on exploitable signal).

Shared metrics: total P&L, P&L/trade, #trades, win rate, Sharpe (+ calibration
for fair value).
