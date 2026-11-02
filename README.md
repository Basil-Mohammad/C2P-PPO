# C2P-PPO: CTL-Constrained Proximal Policy Optimization with Adaptive Shielding

Reference implementation and complete experimental pipeline for the article
*CTL-Constrained Proximal Policy Optimization with Adaptive Safety Shielding*
(B. M. Alzboun, B. H. Abed-alguni).

## What is implemented

| Component | File | Paper |
|---|---|---|
| CTL safety specification `AG(p1 & ... & pm)`, robustness margins, k-step labels | `c2p/ctl.py` | Sec. 4-5 |
| Level-3 safety surrogate trained on Level-2 Monte-Carlo labels | `c2p/algos/safety_models.py` | Sec. 5 |
| Admissible set + adaptive tilting shield realised by SIR | `c2p/algos/c2p_ppo.py` | Sec. 6.2-6.5 |
| Unbiased importance weights `c_t = Z_hat / w` (truncated) | `c2p/algos/c2p_ppo.py` | Lemma 7.4 |
| Blocked-action penalty (virtual transitions) | `c2p/algos/base.py`, `c2p_ppo.py` | Sec. 6.7 |
| Potential-based CTL-robustness shaping | `c2p/algos/c2p_ppo.py` | Sec. 6.8 |
| Lagrangian on cost advantages, Robbins-Monro multiplier steps | `c2p/algos/base.py` | Sec. 6.9 |
| One multiplier per atomic proposition | `C2PPPOMulti` | Sec. 6.10 |
| Baselines: PPO, PPO-Lagrangian, CPO (full), PCPO, FOCOPS, Shielded PPO, Safety Layer, LCRL-PPO | `c2p/algos/` | Sec. 8.2 |
| Environments: CartPole-Track (Safe-Control-Gym), SafetyPointGoal1, SafetyHopperVelocity, SafetyHalfCheetahVelocity (Safety-Gymnasium), SafeArmReach-6DoF (MuJoCo, UR5e kinematics) | `c2p/envs/` | Sec. 8.1 |

All methods share network sizes, rollout lengths, step budgets and PPO
hyperparameters (`c2p/registry.py`, `c2p/algos/base.py::Hyper`).

## Installation

```bash
bash experiments/setup_env.sh ./venv     # Python 3.10, pinned versions (requirements-lock.txt)
./venv/bin/python -m pytest -q tests      # unit tests (reduction to PPO, unbiased IS weights, ...)
```

## Running

```bash
# one run
./venv/bin/python -m c2p.train --env arm --method c2p_ppo --seed 0 --out results
# a group of runs (resumable, parallel)
./venv/bin/python experiments/run_jobs.py --group main --env cartpole --workers 4 --out results
# Kaggle notebooks, one per shard
python experiments/make_kaggle_notebooks.py
```

Environments: `cartpole`, `pointgoal`, `hopper`, `cheetah`, `arm`.
Methods: `ppo`, `ppo_lag`, `cpo`, `pcpo`, `focops`, `shielded_ppo`, `safety_layer_v2`,
`lcrl_ppo`, `c2p_ppo`, `c2p_multi`, and the ablations `c2p_no_lagrangian`,
`c2p_no_shield`, `c2p_fixed_alpha`, `c2p_no_rnd`, `c2p_no_is`, `c2p_no_shaping`,
`c2p_geometric`, `c2p_no_inadm_penalty`.

Each run writes `episodes.csv`, `updates.csv`, `evals.csv`, `summary.json` and
the final policy.

## Data and analysis

`data/` contains the collected results of all 860 runs reported in the article
(see `data/README.md`). The scripts in `analysis/` compute the statistics (IQM,
bootstrap confidence intervals, Holm-corrected tests) and generate every table
and figure of the article from these files.

## Citation

If you use this code, please cite the article *CTL-Constrained Proximal Policy
Optimization with Adaptive Safety Shielding* by B. M. Alzboun and
B. H. Abed-alguni.
