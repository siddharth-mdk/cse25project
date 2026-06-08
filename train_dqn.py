"""Optional CLI to train a single DQN and compare it to the baselines on test.

The notebook (dqn_experiments.ipynb) is the primary place training happens; this
script is a convenience for a one-off run.

Usage:
    python train_dqn.py                  # config defaults
    python train_dqn.py --epochs 8 --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import os

from btcdqn import baseline as B
from btcdqn import config as C
from btcdqn import data
from btcdqn import dqn as D
from btcdqn import env as E
from btcdqn import metrics as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DQN_EPOCHS)
    ap.add_argument("--lr", type=float, default=C.DQN_LR)
    ap.add_argument("--hidden", type=int, default=C.DQN_HIDDEN)
    ap.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    args = ap.parse_args()

    eps, manifest = data.load()
    out = D.train(eps, manifest, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                  device=args.device, save_path=C.DQN_CKPT, label="dqn")
    out["history"].to_json(os.path.join(C.PROC_DIR, "dqn_history.json"), orient="records")

    sigma = B.estimate_sigma_per_min(eps[eps.split == "train"])
    contenders = {
        "random": B.RandomPolicy(),
        "always_down": B.AlwaysSidePolicy(E.BUY_DOWN),
        "fairval_t1": B.FairValuePolicy(tier=1, threshold=0.05, sigma_per_min=sigma),
        "fairval_t1+exit": B.FairValuePolicy(tier=1, threshold=0.05, sigma_per_min=sigma,
                                             exit_on_converge=True),
        "DQN": D.GreedyPolicy(out["agent"]),
    }
    print("\n===== TEST: DQN vs baselines =====")
    results = []
    for name, pol in contenders.items():
        r = M.run_backtest(pol, eps, manifest, "test", fee_rate=0.0)
        r["name"] = name
        results.append(r)
    print(M.metrics_table(results))


if __name__ == "__main__":
    main()
