"""BTC spot price fetching (Coinbase 1-minute candles).

Binance is geo-blocked from US locations, and yfinance only serves 1-minute
data for ~7 days. Coinbase's public Exchange API serves 1-minute candles going
back years with no auth, which covers the full market range. We fetch once and
cache to parquet.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

from . import config as C

_SPOT_CACHE = os.path.join(C.RAW_DIR, "spot_coinbase_1m.parquet")


def _fetch_window(start_s: int, end_s: int) -> list[list[float]]:
    """One Coinbase request: up to 300 1-min candles in [start_s, end_s]."""
    r = requests.get(
        C.COINBASE_URL,
        params={
            "granularity": C.COINBASE_GRAN,
            "start": pd.Timestamp(start_s, unit="s", tz="UTC").isoformat(),
            "end": pd.Timestamp(end_s, unit="s", tz="UTC").isoformat(),
        },
        headers={"User-Agent": "cse25-research"},
        timeout=30,
    )
    if r.status_code == 429:  # rate limited -> back off and retry once
        time.sleep(1.0)
        r = requests.get(
            C.COINBASE_URL,
            params={
                "granularity": C.COINBASE_GRAN,
                "start": pd.Timestamp(start_s, unit="s", tz="UTC").isoformat(),
                "end": pd.Timestamp(end_s, unit="s", tz="UTC").isoformat(),
            },
            headers={"User-Agent": "cse25-research"},
            timeout=30,
        )
    r.raise_for_status()
    return r.json()  # rows: [time, low, high, open, close, volume]


def fetch_spot(start_s: int, end_s: int, refresh: bool = False) -> pd.DataFrame:
    """Return 1-min BTC-USD spot over [start_s, end_s] as DataFrame[ts, close].

    Cached to parquet; pass refresh=True to force a re-download.
    """
    if os.path.exists(_SPOT_CACHE) and not refresh:
        df = pd.read_parquet(_SPOT_CACHE)
        if df.ts.min() <= start_s and df.ts.max() >= end_s:
            return df

    pad = C.SPOT_PAD_DAYS * 86400
    lo, hi = start_s - pad, end_s + pad
    span = C.COINBASE_MAX_PER_REQ * C.COINBASE_GRAN  # seconds per request
    rows: list[list[float]] = []
    cursor = lo
    n_req = 0
    while cursor < hi:
        chunk = _fetch_window(cursor, min(cursor + span, hi))
        rows.extend(chunk)
        cursor += span
        n_req += 1
        time.sleep(C.COINBASE_PAUSE_S)

    arr = np.array(rows, dtype=float)
    df = (
        pd.DataFrame({"ts": arr[:, 0].astype("int64"), "close": arr[:, 4]})
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    df.to_parquet(_SPOT_CACHE, index=False)
    print(f"[spot] fetched {len(df):,} 1-min candles in {n_req} requests -> {_SPOT_CACHE}")
    return df


def spot_lookup(spot: pd.DataFrame):
    """Return a fast 'price at-or-before timestamp t' closure (step-interpolated)."""
    ts = spot.ts.to_numpy()
    close = spot.close.to_numpy()

    def at(t: int) -> float:
        i = np.searchsorted(ts, t, side="right") - 1  # last candle at-or-before t
        i = int(np.clip(i, 0, len(ts) - 1))
        return float(close[i])

    return at
