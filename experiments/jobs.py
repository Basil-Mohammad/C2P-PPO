"""Experiment manifest.

Each job is (env, method, seed, overrides, tag). Groups:
  main         all methods x all environments x 10 seeds
  ablation     8 ablations x {cartpole, arm} x 10 seeds and pointgoal x 5 seeds
  multi        multi-multiplier C2P-PPO on the arm (2 propositions) x 10 seeds
  sensitivity  one-at-a-time sweep of 10 hyperparameters x 2 values x {cartpole, arm} x 5 seeds

Usage:
  python experiments/jobs.py --group main --env cartpole --list
"""
from __future__ import annotations

import argparse
import json

SEEDS = list(range(10))
MAIN_METHODS = ["ppo", "ppo_lag", "cpo", "pcpo", "focops", "shielded_ppo", "safety_layer_v2", "lcrl_ppo", "c2p_ppo"]
ENVS = ["cartpole", "pointgoal", "cargoal", "arm"]
ABLATIONS = ["c2p_no_lagrangian", "c2p_no_shield", "c2p_fixed_alpha", "c2p_no_rnd", "c2p_no_is",
             "c2p_no_shaping", "c2p_geometric", "c2p_no_inadm_penalty"]
SENSITIVITY = {
    "beta": [0.5, 5.0],
    "alpha0": [0.5, 0.95],
    "lambda_lr": [0.01, 0.2],
    "lambda_init": [0.0, 1.0],
    "lambda_r": [0.3, 3.0],
    "n_candidates": [4, 64],
    "tau_safe": [0.3, 0.7],
    "k_lookahead": [5, 20],
    "is_clip": [1.5, 5.0],
    "kappa_inadm": [0.3, 3.0],
}


def jobs(group: str, env: str | None = None):
    out = []
    if group == "main":
        for e in ([env] if env else ENVS):
            for m in MAIN_METHODS:
                for s in SEEDS:
                    out.append(dict(env=e, method=m, seed=s, overrides={}, tag=""))
    elif group == "ablation":
        for e in ([env] if env else ["cartpole", "arm", "pointgoal"]):
            for m in ABLATIONS:
                for s in (SEEDS if e != "pointgoal" else SEEDS[:5]):   # 5 seeds on the costly navigation task
                    out.append(dict(env=e, method=m, seed=s, overrides={}, tag=""))
    elif group == "slayer_fix":   # re-run of the corrected safety-layer baseline on the finished environments
        for e in (["cartpole", "arm"] if not env else [env]):
            for s in SEEDS:
                out.append(dict(env=e, method="safety_layer_v2", seed=s, overrides={}, tag=""))
    elif group == "multi":
        for s in SEEDS:
            out.append(dict(env="arm", method="c2p_multi", seed=s, overrides={}, tag=""))
    elif group == "sensitivity":
        for e in ([env] if env else ["cartpole", "arm"]):
            for k, vals in SENSITIVITY.items():
                for v in vals:
                    for s in range(5):
                        out.append(dict(env=e, method="c2p_ppo", seed=s, overrides={k: v}, tag=f"{k}={v}"))
    else:
        raise KeyError(group)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--env", default=None)
    a = ap.parse_args()
    J = jobs(a.group, a.env)
    print(json.dumps(J, indent=0))
    print(f"# {len(J)} jobs")
