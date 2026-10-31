# Experimental data

Collected results of all runs reported in the revised manuscript (860 runs: five environments,
nine methods, ablation, sensitivity and multi-constraint studies; ten seeds per main configuration).

| File | Content |
|---|---|
| `c2p_runs.csv.gz` | one row per run: final evaluation (return, violation rate, success, safety rate; with and without shield for C2P-PPO), training violation rate, wall-clock time |
| `c2p_evals.csv.gz` | periodic evaluations during training |
| `c2p_curves.csv.gz` | training curves aggregated into 40 blocks per run (return, violation rate, shield intervention rate) |
| `c2p_updates.csv.gz` | training diagnostics (Lagrange multiplier, adaptive weight, importance weights, surrogate recall), up to 100 updates per run |

Run names: `<method>` for the main comparison, `c2p_no_*`, `c2p_fixed_alpha`, `c2p_geometric` for the ablation,
`c2p_multi` for the multi-constraint variant and `c2p_ppo_<parameter>=<value>` for the sensitivity analysis.
All tables and figures of the manuscript are generated from these files:

```
python analysis/make_results.py --data <unzipped dir> --out results --envs cartpole pointgoal hopper cheetah arm
python analysis/make_ablation.py <unzipped dir> results
python analysis/make_sensitivity.py <unzipped dir> results
```
