"""Figures for the revised manuscript (PDF, vector).

python analysis/figures.py --results <results_dir> --out <paper/media>
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stats import bootstrap_ci, iqm, load_runs  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "figure.dpi": 150})

LABEL = {"ppo": "PPO", "ppo_lag": "PPO-Lag", "cpo": "CPO", "pcpo": "PCPO", "focops": "FOCOPS",
         "shielded_ppo": "Shielded PPO", "safety_layer": "Safety layer", "lcrl_ppo": "LCRL-PPO",
         "c2p_ppo": "C2P-PPO (ours)", "c2p_multi": "C2P-PPO multi-$\\lambda$",
         "c2p_no_lagrangian": "w/o Lagrangian", "c2p_no_shield": "w/o shield", "c2p_fixed_alpha": "w/o adaptive $\\alpha$",
         "c2p_no_rnd": "w/o RND", "c2p_no_is": "w/o IS weights", "c2p_no_shaping": "w/o shaping",
         "c2p_geometric": "geometric mixture", "c2p_no_inadm_penalty": "w/o blocked-action penalty"}
ORDER = ["ppo", "ppo_lag", "cpo", "pcpo", "focops", "shielded_ppo", "safety_layer", "lcrl_ppo", "c2p_ppo"]
COLORS = dict(zip(ORDER, ["#9e9e9e", "#4c72b0", "#55a868", "#8172b2", "#64b5cd", "#ccb974", "#937860",
                          "#da8bc3", "#c44e52"]))
ENV_TITLE = {"cartpole": "CartPole-Track", "pointgoal": "SafetyPointGoal1", "cargoal": "SafetyCarGoal1",
             "arm": "SafeArmReach-6DoF"}
BUDGET = {"cartpole": 0.05, "pointgoal": 0.025, "cargoal": 0.025, "arm": 0.01}


def curves(results, env, run, col, fname="evals.csv"):
    fs = sorted(glob.glob(os.path.join(results, env, run, "seed*", fname)))
    if not fs:
        return None, None
    D = [pd.read_csv(f) for f in fs]
    steps = D[0]["step"].values
    L = min(len(d) for d in D)
    M = np.stack([d[col].values[:L] for d in D])
    return steps[:L], M


def band(ax, x, M, color, label, lw=1.4):
    m = np.array([iqm(M[:, i]) for i in range(M.shape[1])])
    ci = np.array([bootstrap_ci(M[:, i], n_boot=2000) for i in range(M.shape[1])])
    ax.plot(x, m, color=color, label=label, lw=lw)
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, lw=0)


def fig_learning(results, out, envs):
    fig, axes = plt.subplots(2, len(envs), figsize=(3.0 * len(envs), 4.6))
    axes = np.atleast_2d(axes).reshape(2, len(envs))
    for j, env in enumerate(envs):
        for run in ORDER:
            x, R = curves(results, env, run, "eval_return")
            if x is None:
                continue
            _, V = curves(results, env, run, "eval_viol_rate")
            lw = 2.0 if run == "c2p_ppo" else 1.1
            band(axes[0, j], x / 1e6, R, COLORS[run], LABEL[run], lw)
            band(axes[1, j], x / 1e6, V, COLORS[run], LABEL[run], lw)
        axes[1, j].axhline(BUDGET[env], color="k", ls=":", lw=1)
        axes[0, j].set_title(ENV_TITLE[env])
        axes[1, j].set_xlabel("environment steps (M)")
    axes[0, 0].set_ylabel("evaluation return")
    axes[1, 0].set_ylabel("evaluation violation rate")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "fig_learning_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(df, out, envs):
    fig, axes = plt.subplots(1, len(envs), figsize=(3.0 * len(envs), 2.8))
    for ax, env in zip(np.atleast_1d(axes), envs):
        g = df[df.env == env]
        for run in ORDER:
            h = g[g.run == run]
            if h.empty:
                continue
            vx, ry = iqm(h.viol_rate), iqm(h["return"])
            vlo, vhi = bootstrap_ci(h.viol_rate.values, n_boot=2000)
            rlo, rhi = bootstrap_ci(h["return"].values, n_boot=2000)
            ax.errorbar(vx, ry, xerr=[[vx - vlo], [vhi - vx]], yerr=[[ry - rlo], [rhi - ry]],
                        fmt="*" if run == "c2p_ppo" else "o", ms=10 if run == "c2p_ppo" else 5,
                        color=COLORS[run], label=LABEL[run], capsize=2, lw=0.8)
        ax.axvspan(0, BUDGET[env], color="#2ca02c", alpha=0.08, lw=0)
        ax.axvline(BUDGET[env], color="k", ls=":", lw=1)
        ax.set_title(ENV_TITLE[env]); ax.set_xlabel("violation rate (IQM)")
    np.atleast_1d(axes)[0].set_ylabel("return (IQM)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "fig_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_ablation(df, out, envs):
    runs = ["c2p_ppo", "c2p_no_lagrangian", "c2p_no_shield", "c2p_fixed_alpha", "c2p_no_rnd", "c2p_no_is",
            "c2p_no_shaping", "c2p_geometric", "c2p_no_inadm_penalty"]
    fig, axes = plt.subplots(2, len(envs), figsize=(3.2 * len(envs), 5.0))
    axes = np.atleast_2d(axes).reshape(2, len(envs))
    for j, env in enumerate(envs):
        g = df[df.env == env]
        for i, (met, lab) in enumerate([("return", "evaluation return"), ("viol_rate", "violation rate")]):
            ax = axes[i, j]
            vals, los, his, names = [], [], [], []
            for r in runs:
                h = g[g.run == r]
                if h.empty:
                    continue
                v = iqm(h[met]); lo, hi = bootstrap_ci(h[met].values, n_boot=2000)
                vals.append(v); los.append(v - lo); his.append(hi - v); names.append(LABEL[r])
            y = np.arange(len(vals))
            ax.barh(y, vals, xerr=[los, his], color=["#c44e52"] + ["#8c8c8c"] * (len(vals) - 1), capsize=2)
            ax.set_yticks(y); ax.set_yticklabels(names if j == 0 else [])
            ax.invert_yaxis(); ax.set_xlabel(lab)
            if met == "viol_rate":
                ax.axvline(BUDGET[env], color="k", ls=":", lw=1)
            if i == 0:
                ax.set_title(ENV_TITLE[env])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ablation.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(df, out, envs):
    s = df[df.run.str.startswith("c2p_ppo_")].copy()
    if s.empty:
        return
    s["param"] = s.tag.str.split("=").str[0]
    s["value"] = s.tag.str.split("=").str[1].astype(float)
    params = sorted(s.param.unique())
    fig, axes = plt.subplots(len(envs), len(params), figsize=(1.6 * len(params), 2.2 * len(envs)), squeeze=False)
    from c2p.algos.base import Hyper
    from c2p.registry import ENVS
    for i, env in enumerate(envs):
        base = df[(df.env == env) & (df.run == "c2p_ppo")]
        for j, p in enumerate(params):
            ax = axes[i, j]
            default = ENVS[env]["hp"].get(p, getattr(Hyper(), p))
            pts = [(default, base)] + [(v, g) for v, g in s[(s.env == env) & (s.param == p)].groupby("value")]
            pts.sort(key=lambda t: t[0])
            xs = np.arange(len(pts))
            ax2 = ax.twinx()
            ax.plot(xs, [iqm(g["return"]) for _, g in pts], "o-", color="#4c72b0", ms=3)
            ax2.plot(xs, [iqm(g["viol_rate"]) for _, g in pts], "s--", color="#c44e52", ms=3)
            ax2.axhline(BUDGET[env], color="#c44e52", ls=":", lw=0.8)
            ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v, _ in pts], fontsize=7)
            if i == 0:
                ax.set_title(p.replace("_", r"\_") if False else p, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{ENV_TITLE[env]}\nreturn", color="#4c72b0")
            if j == len(params) - 1:
                ax2.set_ylabel("violation rate", color="#c44e52")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_dynamics(results, out, envs):
    cols = [("lambda", r"multiplier $\lambda_t$"), ("alpha", r"shield weight $\alpha_t$"),
            ("intervention", "intervention rate"), ("is_w_mean", r"mean IS weight $c_t$")]
    fig, axes = plt.subplots(len(cols), len(envs), figsize=(3.0 * len(envs), 7.0), squeeze=False)
    for j, env in enumerate(envs):
        fs = sorted(glob.glob(os.path.join(results, env, "c2p_ppo", "seed*", "updates.csv")))
        es = sorted(glob.glob(os.path.join(results, env, "c2p_ppo", "seed*", "episodes.csv")))
        if not fs:
            continue
        U = [pd.read_csv(f) for f in fs]
        for i, (c, lab) in enumerate(cols):
            ax = axes[i, j]
            if c == "intervention":
                E = [pd.read_csv(f) for f in es]
                grid = np.linspace(0, min(e.step.max() for e in E), 60)
                M = np.stack([np.interp(grid, e.step, e.intervention_rate.rolling(20, min_periods=1).mean()) for e in E])
                band(ax, grid / 1e6, M, "#c44e52", "")
            else:
                L = min(len(u) for u in U)
                M = np.stack([u[c].values[:L] for u in U])
                band(ax, U[0].step.values[:L] / 1e6, M, "#c44e52", "")
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(ENV_TITLE[env])
            if i == len(cols) - 1:
                ax.set_xlabel("environment steps (M)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dynamics.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_runs(a.results)
    envs = [e for e in ["cartpole", "pointgoal", "cargoal", "arm"] if e in set(df.env)]
    fig_learning(a.results, a.out, envs)
    fig_tradeoff(df[df.run.isin(ORDER)], a.out, envs)
    abl_envs = [e for e in envs if (df[df.env == e].run.str.startswith("c2p_no")).any()]
    if abl_envs:
        fig_ablation(df, a.out, abl_envs)
    fig_sensitivity(df, a.out, [e for e in envs if (df[df.env == e].tag != "").any()])
    fig_dynamics(a.results, a.out, envs)


if __name__ == "__main__":
    main()
