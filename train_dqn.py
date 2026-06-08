"""Phase 3 entry point: train the Double DQN and compare to the baseline.

Usage:
    python train_dqn.py                 # train with config defaults
    python train_dqn.py --epochs 15     # override epochs
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from btcdqn import baseline as B
from btcdqn import config as C
from btcdqn import data
from btcdqn import dqn as D
from btcdqn import env as E
from btcdqn import metrics as M


def build_episode_index(eps, manifest, split):
    mids = list(manifest[manifest.split == split].market_id)
    labels = dict(zip(manifest.market_id, manifest.label_up))
    sub = eps[eps.market_id.isin(set(mids))]
    by_mkt = {mid: df for mid, df in sub.groupby("market_id")}
    return mids, by_mkt, labels


def run_episode_train(agent, env, epsilon):
    obs = env.reset()
    total = 0.0
    while not env.done:
        a = agent.act(obs, epsilon)
        s = obs.vec
        next_obs, r, done, _ = env.step(a)
        agent.buffer.push(s, a, r, next_obs.vec, float(done), D.valid_mask(next_obs.position))
        agent.learn()
        obs = next_obs
        total += r
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DQN_EPOCHS)
    ap.add_argument("--device", type=str, default=None,
                    help="mps | cpu | cuda (default: auto, prefers mps)")
    ap.add_argument("--variant", choices=["double", "vanilla"], default="double",
                    help="double = Double DQN (default); vanilla = standard DQN (ablation)")
    args = ap.parse_args()

    is_double = args.variant == "double"
    ckpt_path = C.DQN_CKPT if is_double else C.DQN_CKPT_VANILLA
    hist_path = os.path.join(C.PROC_DIR,
                             "dqn_history.json" if is_double else "dqn_history_vanilla.json")

    eps, manifest = data.load()
    tr_mids, tr_by, tr_lab = build_episode_index(eps, manifest, "train")

    agent = D.DoubleDQNAgent(device=args.device, double=is_double)
    print(f"variant: {args.variant} | device: {agent.device}")
    total_eps = args.epochs * len(tr_mids)
    decay_eps = max(1, int(C.DQN_EPS_DECAY_FRAC * total_eps))
    rng = random.Random(C.SEED)

    def epsilon_at(i):
        return max(C.DQN_EPS_END,
                   C.DQN_EPS_START - (C.DQN_EPS_START - C.DQN_EPS_END) * i / decay_eps)

    print(f"training: {args.epochs} epochs x {len(tr_mids)} markets = {total_eps} episodes\n")
    gi = 0
    best_val = -1e9
    history = []
    for epoch in range(1, args.epochs + 1):
        order = tr_mids[:]
        rng.shuffle(order)
        ep_rewards = []
        for mid in order:
            env = E.EpisodeEnv(tr_by[mid], tr_lab[mid], fee_rate=0.0)
            ep_rewards.append(run_episode_train(agent, env, epsilon_at(gi)))
            gi += 1

        val = M.run_backtest(D.GreedyPolicy(agent), eps, manifest, "val", fee_rate=0.0)
        tag = ""
        if val["mean_pnl"] > best_val:
            best_val = val["mean_pnl"]
            agent.save(ckpt_path)
            tag = "  <- best (saved)"
        print(f"epoch {epoch:2d} | eps={epsilon_at(gi):.3f} | train_meanR={np.mean(ep_rewards):+.4f} "
              f"| val_meanPnL={val['mean_pnl']:+.4f} sharpe={val['sharpe']:+.3f} "
              f"trades={val['n_trades']} win={val['win_rate']:.0%}{tag}")
        history.append({"epoch": epoch, "eps": round(epsilon_at(gi), 4),
                        "train_meanR": float(np.mean(ep_rewards)),
                        "val_mean_pnl": val["mean_pnl"], "val_sharpe": val["sharpe"],
                        "val_trades": val["n_trades"], "val_win_rate": val["win_rate"]})

    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── final head-to-head on TEST with the best checkpoint ──────────────────
    agent.load(ckpt_path)
    sigma = B.estimate_sigma_per_min(eps[eps.split == "train"])
    contenders = {
        "random": B.RandomPolicy(),
        "always_down": B.AlwaysSidePolicy(E.BUY_DOWN),
        "fairval_t1": B.FairValuePolicy(tier=1, threshold=0.05, sigma_per_min=sigma),
        "fairval_t1+exit": B.FairValuePolicy(tier=1, threshold=0.05, sigma_per_min=sigma,
                                             exit_on_converge=True),
        f"DQN ({args.variant})": D.GreedyPolicy(agent),
    }
    # If the other variant's checkpoint exists, include it for a direct comparison.
    other_path = C.DQN_CKPT_VANILLA if is_double else C.DQN_CKPT
    if os.path.exists(other_path):
        other = D.DoubleDQNAgent(device=args.device, double=not is_double)
        other.load(other_path)
        contenders[f"DQN ({'vanilla' if is_double else 'double'})"] = D.GreedyPolicy(other)
    print("\n===== TEST: DQN vs baselines =====")
    results = []
    for name, pol in contenders.items():
        r = M.run_backtest(pol, eps, manifest, "test", fee_rate=0.0)
        r["name"] = name
        results.append(r)
    print(M.metrics_table(results))


if __name__ == "__main__":
    main()
