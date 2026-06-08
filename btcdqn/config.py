"""Central configuration for the BTC Up/Down DQN project.

Everything path- or constant-like lives here so the pipeline, the baseline,
and the final notebook all agree on the same numbers.
"""
from __future__ import annotations

import os

# ── Paths ──────────────────────────────────────────────────────────────────
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
for _d in (RAW_DIR, PROC_DIR):
    os.makedirs(_d, exist_ok=True)

# ── HuggingFace source (BrockMisner/polymarket-btc-updown) ─────────────────
HF_BASE = "https://huggingface.co/datasets/BrockMisner/polymarket-btc-updown/resolve/main/"
HF_FILES = {
    "prices_btc_1h": "data/prices/crypto=BTC/timeframe=1-hour/part-0.parquet",
    "markets": "data/markets.parquet",
}

# ── Market selection / labelling ───────────────────────────────────────────
CRYPTO = "BTC"
TIMEFRAME = "1-hour"
# A market is "cleanly resolved" if its terminal contract price collapsed to ~0/1.
CONVERGE_LO = 0.05
CONVERGE_HI = 0.95
# Keep only genuine ~1-hour markets (a degenerate early batch ran ~49h).
MIN_DURATION_H = 0.5
MAX_DURATION_H = 2.0
# Require at least this many raw contract-price points inside the resolution hour.
MIN_POINTS_IN_WINDOW = 10

# ── Episode grid ───────────────────────────────────────────────────────────
# Each episode is the resolution hour [H - 1h, H], sampled on a fixed 1-min grid.
GRID_STEP_S = 60
EPISODE_SECONDS = 3600
N_STEPS = EPISODE_SECONDS // GRID_STEP_S  # 60 decision steps per episode

# ── Spot (Coinbase 1-min BTC-USD) ──────────────────────────────────────────
COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
COINBASE_GRAN = 60          # 1-minute candles
COINBASE_MAX_PER_REQ = 300  # API cap
COINBASE_PAUSE_S = 0.15     # politeness between requests
# Pad the spot fetch around the market range so lookback features have history.
SPOT_PAD_DAYS = 2

# ── Features computed on the grid ──────────────────────────────────────────
# Spot lookback windows (in 1-min steps) and contract-momentum window.
SPOT_RET_WINDOWS = (5, 15)
SPOT_VOL_WINDOW = 10
CONTRACT_MOM_WINDOW = 5

# Columns that get z-scored (stats fit on TRAIN only). Raw up_price is kept
# unnormalised because the env needs it for P&L.
FEATURE_COLS = [
    "spot_ret_5",
    "spot_ret_15",
    "spot_vol",
    "spot_vs_ref",
    "up_mom_5",
    "up_price",   # also used as a feature (normalised copy: up_price_z)
    "tau",
]

# ── Chronological split (by resolution time H) ─────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder -> test

# ── DQN (Phase 3) ──────────────────────────────────────────────────────────
DQN_HIDDEN = 64
DQN_EPOCHS = 10            # passes over the shuffled training markets
DQN_LR = 1e-3
DQN_GAMMA = 0.99
DQN_BATCH = 64
DQN_BUFFER = 50_000
DQN_EPS_START = 1.0
DQN_EPS_END = 0.05
DQN_EPS_DECAY_FRAC = 0.6   # decay eps over this fraction of total training episodes
DQN_TARGET_UPDATE = 1000   # hard target-net sync every N learn steps
DQN_WARMUP = 1000          # steps of pure exploration before learning starts
DQN_LEARN_EVERY = 1        # learn every N env steps
DQN_CKPT = os.path.join(PROC_DIR, "dqn_best.pt")

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
