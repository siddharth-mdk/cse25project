"""Phase 1 entry point: run the fair-value baseline + trivial baselines.

Usage:
    python run_baselines.py            # fee = 0 (default)
    python run_baselines.py --fee 0.1  # symmetric fee_rate (e.g. 1000 bps)
"""
from __future__ import annotations

import argparse

from btcdqn import baseline as B
from btcdqn import data
from btcdqn import env as E
from btcdqn import metrics as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=0.0, help="symmetric fee rate (0 = none)")
    ap.add_argument("--threshold", type=float, default=0.05, help="fair-value edge threshold")
    args = ap.parse_args()

    eps, manifest = data.load()
    sigma = B.estimate_sigma_per_min(eps[eps.split == "train"])
    print(f"estimated sigma_per_min (train) = {sigma:.5f}  | fee_rate = {args.fee}\n")

    policies = {
        "random": B.RandomPolicy(),
        "always_up": B.AlwaysSidePolicy(E.BUY_UP),
        "always_down": B.AlwaysSidePolicy(E.BUY_DOWN),
        f"fairval_t0 (thr={args.threshold})": B.FairValuePolicy(tier=0, threshold=args.threshold),
        f"fairval_t1 (thr={args.threshold})": B.FairValuePolicy(
            tier=1, threshold=args.threshold, sigma_per_min=sigma),
        f"fairval_t1+exit": B.FairValuePolicy(
            tier=1, threshold=args.threshold, sigma_per_min=sigma, exit_on_converge=True),
    }

    for split in ("val", "test"):
        results = []
        for name, pol in policies.items():
            r = M.run_backtest(pol, eps, manifest, split, fee_rate=args.fee)
            r["name"] = name
            results.append(r)
        print(f"===== {split.upper()} =====")
        print(M.metrics_table(results))
        print()

    # fair-value calibration on test (does FV predict outcomes?)
    fv = B.FairValuePolicy(tier=1, threshold=args.threshold, sigma_per_min=sigma)
    print("===== Fair-value (Tier 1) calibration — TEST =====")
    print(M.fair_value_calibration(fv, eps, manifest, "test").to_string(index=False))


if __name__ == "__main__":
    main()
