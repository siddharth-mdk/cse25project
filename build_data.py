"""Phase 0 entry point: build and cache the BTC 1-hour episode dataset.

Usage:
    python build_data.py            # build (uses caches where present)
    python build_data.py --refresh  # force re-download + rebuild
"""
from __future__ import annotations

import argparse

import pandas as pd

from btcdqn import data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="force re-download + rebuild")
    args = ap.parse_args()

    eps, manifest = data.build(refresh=args.refresh)

    print("\n=== Phase 0 summary ===")
    print(f"episodes (markets): {manifest.market_id.nunique():,}")
    print(f"rows (market x step): {len(eps):,}")
    print("\nsplit sizes (markets):")
    print(manifest.split.value_counts().reindex(["train", "val", "test"]).to_string())
    print("\nlabel balance by split:")
    print(manifest.groupby("split", observed=True).label_up.mean().round(3).to_string())
    print("\ndate range by split (resolution hour H, UTC):")
    for sp in ["train", "val", "test"]:
        s = manifest[manifest.split == sp]
        if len(s):
            print(f"  {sp:5s}: {pd.to_datetime(s.H.min(), unit='s')} -> "
                  f"{pd.to_datetime(s.H.max(), unit='s')}")
    print("\nfeature columns (z-scored):", [c for c in eps.columns if c.endswith("_z")])
    print("sample episode rows:")
    print(eps.head(3).to_string())


if __name__ == "__main__":
    main()
