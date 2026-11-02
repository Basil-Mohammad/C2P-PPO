"""Aggregate statistics following Agarwal et al. (2021).

IQM with stratified percentile bootstrap CIs, Welch t and Mann--Whitney U
tests with Holm correction, Cohen's d and probability of improvement.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def iqm(x) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def bootstrap_ci(x, stat=iqm, n_boot=10_000, alpha=0.05, rng=0):
    x = np.asarray(x, float)
    g = np.random.default_rng(rng)
    idx = g.integers(0, len(x), size=(n_boot, len(x)))
    b = np.array([stat(x[i]) for i in idx])
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def prob_improvement(a, b, higher_is_better=True) -> float:
    """P(X > Y) with ties counted 1/2 (Mann--Whitney normalised)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not higher_is_better:
        a, b = -a, -b
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load_runs(root: str) -> pd.DataFrame:
    """One row per run with final-evaluation and training-time metrics."""
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        s = json.load(open(f))
        d = os.path.dirname(f)
        fe = s["final_eval"]
        E = pd.read_csv(os.path.join(d, "episodes.csv"))
        last = E[E.step > 0.9 * s["total_steps"]]
        r = {"env": s["env"], "run": os.path.basename(os.path.dirname(d)), "method": s["method"], "tag": s.get("tag", ""),
             "seed": s["seed"], "return": fe["eval_return"], "viol_rate": fe["eval_viol_rate"],
             "success": fe.get("eval_success", np.nan), "safe_rate": fe.get("eval_safe_rate", np.nan),
             "train_cost_rate": s["cost_rate"], "cum_violations": s["cum_violations"],
             "late_train_viol": float(last.violations.sum() / max(last.length.sum(), 1)),
             "late_train_return": float(last["return"].mean()), "wall_h": s["wall_time_s"] / 3600,
             "budget": s["budget"]}
        for k in ("eval_return_noshield", "eval_viol_rate_noshield", "eval_success_noshield", "eval_safe_rate_noshield"):
            if k in fe:
                r[k.replace("eval_", "")] = fe[k]
        rows.append(r)
    return pd.DataFrame(rows)


def summary_table(df, metrics, ref="c2p_ppo"):
    out = []
    for (env, run), g in df.groupby(["env", "run"]):
        row = {"env": env, "run": run, "n": len(g)}
        for m in metrics:
            row[m] = iqm(g[m])
            row[m + "_lo"], row[m + "_hi"] = bootstrap_ci(g[m].values)
        out.append(row)
    return pd.DataFrame(out)


def pairwise(df, metric, ref="c2p_ppo", higher_is_better=True):
    out = []
    for env, g in df.groupby("env"):
        a = g[g.run == ref][metric].values
        rows = []
        for run, h in g.groupby("run"):
            if run == ref:
                continue
            b = h[metric].values
            t = stats.ttest_ind(a, b, equal_var=False)
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"env": env, "baseline": run, "metric": metric, "delta_iqm": iqm(a) - iqm(b),
                         "p_welch": t.pvalue, "p_mwu": u.pvalue, "cohen_d": cohen_d(a, b),
                         "p_improve": prob_improvement(a, b, higher_is_better)})
        if rows:
            R = pd.DataFrame(rows)
            R["p_welch_holm"] = holm(R.p_welch)
            R["p_mwu_holm"] = holm(R.p_mwu)
            out.append(R)
    return pd.concat(out) if out else pd.DataFrame()
