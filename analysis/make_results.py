"""Tables (LaTeX) and figures (PDF) for Section 8 from the merged CSV files.

python analysis/make_results.py --data <dir with c2p_*.csv> --out <paper/results> [--envs cartpole arm ...]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from stats import bootstrap_ci, cohen_d, holm, iqm, prob_improvement  # noqa: E402

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5})

ORDER = ["ppo", "ppo_lag", "cpo", "pcpo", "focops", "lcrl_ppo", "shielded_ppo", "safety_layer_v2", "c2p_ppo"]
LABEL = {"ppo": "PPO", "ppo_lag": "PPO-Lag", "cpo": "CPO", "pcpo": "PCPO", "focops": "FOCOPS",
         "lcrl_ppo": "LCRL-PPO", "shielded_ppo": "Shielded PPO", "safety_layer_v2": "Safety layer",
         "c2p_ppo": "C2P-PPO (ours)"}
COLORS = {"ppo": "#9e9e9e", "ppo_lag": "#4c72b0", "cpo": "#55a868", "pcpo": "#8172b2", "focops": "#64b5cd",
          "lcrl_ppo": "#da8bc3", "shielded_ppo": "#ccb974", "safety_layer_v2": "#937860", "c2p_ppo": "#c44e52"}
ENV_TITLE = {"cartpole": "CartPole-Track", "pointgoal": "SafetyPointGoal1", "cargoal": "SafetyCarGoal1",
             "arm": "SafeArmReach-6DoF"}
BUDGET = {"cartpole": 0.05, "pointgoal": 0.025, "cargoal": 0.025, "arm": 0.01}


def fmt(v, lo, hi, d=1):
    return f"{v:.{d}f} \\scriptsize{{[{lo:.{d}f}, {hi:.{d}f}]}}"


def main_table(R, envs, out):
    """One block per environment; bold = best per column (direction in header), dagger = IQM violation > budget."""
    lines = []
    for env in envs:
        g = R[R.env == env]
        rows = []
        for m in ORDER:
            h = g[g.run == m]
            if len(h) < 3:
                continue
            r = {"m": m, "n": len(h)}
            for k, col in [("ret", "eval_return"), ("viol", "eval_viol_rate"), ("tcr", "train_cost_rate")]:
                r[k] = iqm(h[col]); r[k + "_ci"] = bootstrap_ci(h[col].values)
            r["safe"] = h.eval_safe_rate.mean()
            r["succ"] = h.eval_success.mean()
            r["feas"] = int((h.eval_viol_rate <= BUDGET[env]).sum())
            rows.append(r)
        T = pd.DataFrame(rows)
        best = {"ret": T.ret.max(), "viol": T.viol.min(), "tcr": T.tcr.min(), "safe": T.safe.max(),
                "succ": T.succ.max(), "feas": T.feas.max()}
        nav = env in ("pointgoal", "cargoal", "arm")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{ENV_TITLE[env]}}} "
                     f"(budget $\\delta={BUDGET[env]}$)}}\\\\")
        for _, r in T.iterrows():
            dig = 3 if env != "cartpole" else 3
            def b(key, s):
                return f"\\textbf{{{s}}}" if np.isclose(r[key], best[key]) else s
            dag = "$^\\dagger$" if r.viol > BUDGET[env] else ""
            cells = [LABEL[r.m] + dag,
                     b("ret", fmt(r.ret, *r.ret_ci, d=1)),
                     b("viol", fmt(r.viol, *r.viol_ci, d=dig)),
                     b("safe", f"{r.safe:.2f}"),
                     (b("succ", f"{r.succ:.2f}") if nav else "--"),
                     b("tcr", fmt(r.tcr, *r.tcr_ci, d=dig)),
                     b("feas", f"{r.feas}/{r.n}")]
            if r.m == "c2p_ppo":
                cells = ["\\rowcolor{red!6}" + cells[0]] + cells[1:]
            lines.append(" & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    open(os.path.join(out, "tab_main_rows.tex"), "w").write("\n".join(lines))


def tests_table(R, envs, out):
    rows = []
    for env in envs:
        g = R[R.env == env]
        a = g[g.run == "c2p_ppo"]
        res = []
        for m in ORDER[:-1]:
            h = g[g.run == m]
            if len(h) < 3:
                continue
            r = {"env": env, "m": m}
            for key, col, hib in [("ret", "eval_return", True), ("viol", "eval_viol_rate", False),
                                  ("tcr", "train_cost_rate", False)]:
                r[key + "_p"] = stats.mannwhitneyu(a[col], h[col], alternative="two-sided").pvalue
                r[key + "_pi"] = prob_improvement(a[col], h[col], hib)
                r[key + "_d"] = cohen_d(a[col], h[col])
            res.append(r)
        D = pd.DataFrame(res)
        for key in ["ret", "viol", "tcr"]:
            D[key + "_padj"] = holm(D[key + "_p"])
        rows.append(D)
    D = pd.concat(rows)
    D.to_csv(os.path.join(out, "pairwise_tests.csv"), index=False)
    lines = []
    for env in envs:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{ENV_TITLE[env]}}}}}\\\\")
        for _, r in D[D.env == env].iterrows():
            def c(key):
                p, pi = r[key + "_padj"], r[key + "_pi"]
                ps = "<0.001" if p < 0.001 else f"{p:.3f}"
                if p < 0.05 and pi > 0.5:        # significant in favour of C2P-PPO
                    return f"\\textbf{{{ps}}} & \\textbf{{{pi:.2f}}}"
                if p < 0.05:                      # significant in favour of the baseline
                    return f"\\underline{{{ps}}} & \\underline{{{pi:.2f}}}"
                return f"{ps} & {pi:.2f}"
            lines.append(f"{LABEL[r.m]} & {c('ret')} & {c('viol')} & {c('tcr')}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    open(os.path.join(out, "tab_tests_rows.tex"), "w").write("\n".join(lines))
    return D


def band(ax, x, M, color, label, lw):
    m = np.array([iqm(M[:, i]) for i in range(M.shape[1])])
    ci = np.array([bootstrap_ci(M[:, i], n_boot=2000) for i in range(M.shape[1])])
    ax.plot(x, m, color=color, label=label, lw=lw, zorder=3 if lw > 1.5 else 2)
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.13, lw=0)


def learning_curves(E, C, envs, out):
    fig, axes = plt.subplots(3, len(envs), figsize=(3.3 * len(envs), 6.6), squeeze=False)
    for j, env in enumerate(envs):
        for m in ORDER:
            e = E[(E.env == env) & (E.run == m)]
            c = C[(C.env == env) & (C.run == m)]
            if e.seed.nunique() < 3:
                continue
            lw = 2.0 if m == "c2p_ppo" else 1.0
            P = e.pivot_table(index="seed", columns="step", values="eval_return").dropna(axis=1)
            V = e.pivot_table(index="seed", columns="step", values="eval_viol_rate").dropna(axis=1)
            band(axes[0, j], P.columns.values / 1e6, P.values, COLORS[m], LABEL[m], lw)
            band(axes[1, j], V.columns.values / 1e6, V.values, COLORS[m], LABEL[m], lw)
            c = c.assign(rate=c.n_violations / c.n_steps)
            TV = c.pivot_table(index="seed", columns="block", values="rate").dropna(axis=1)
            xs = c.groupby("block").step.mean().loc[TV.columns].values / 1e6
            band(axes[2, j], xs, TV.values, COLORS[m], LABEL[m], lw)
        for i in (1, 2):
            axes[i, j].axhline(BUDGET[env], color="k", ls=":", lw=1)
            axes[i, j].set_yscale("symlog", linthresh=BUDGET[env] / 2)
            axes[i, j].set_ylim(bottom=0)
        axes[0, j].set_title(ENV_TITLE[env])
        axes[2, j].set_xlabel("environment steps (millions)")
    axes[0, 0].set_ylabel("evaluation return")
    axes[1, 0].set_ylabel("evaluation violation rate")
    axes[2, 0].set_ylabel("training violation rate")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "fig_learning_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def tradeoff(R, envs, out):
    fig, axes = plt.subplots(1, len(envs), figsize=(3.3 * len(envs), 2.9), squeeze=False)
    for ax, env in zip(axes[0], envs):
        g = R[R.env == env]
        for m in ORDER:
            h = g[g.run == m]
            if len(h) < 3:
                continue
            vx, ry = iqm(h.eval_viol_rate), iqm(h.eval_return)
            vlo, vhi = bootstrap_ci(h.eval_viol_rate.values, n_boot=4000)
            rlo, rhi = bootstrap_ci(h.eval_return.values, n_boot=4000)
            star = m == "c2p_ppo"
            ax.errorbar(vx, ry, xerr=[[vx - vlo], [vhi - vx]], yerr=[[ry - rlo], [rhi - ry]],
                        fmt="*" if star else "o", ms=11 if star else 5, color=COLORS[m], label=LABEL[m],
                        capsize=2, lw=0.8, zorder=3 if star else 2)
        ax.axvspan(0, BUDGET[env], color="#2ca02c", alpha=0.09, lw=0)
        ax.axvline(BUDGET[env], color="k", ls=":", lw=1)
        ax.set_xscale("symlog", linthresh=BUDGET[env] / 2)
        ax.set_xlim(left=0)
        ax.set_title(ENV_TITLE[env]); ax.set_xlabel("evaluation violation rate (IQM)")
    axes[0, 0].set_ylabel("evaluation return (IQM)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(out, "fig_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def dynamics(U, C, envs, out):
    rows = [("lambda", r"multiplier $\lambda_t$"), ("alpha", r"shield weight $\alpha_t$"),
            ("intervention_rate", "shield intervention rate"), ("sc_unsafe_recall", "surrogate recall (unsafe)")]
    fig, axes = plt.subplots(len(rows), len(envs), figsize=(3.3 * len(envs), 7.2), squeeze=False)
    for j, env in enumerate(envs):
        u = U[(U.env == env) & (U.run == "c2p_ppo")]
        c = C[(C.env == env) & (C.run == "c2p_ppo")]
        for i, (col, lab) in enumerate(rows):
            ax = axes[i, j]
            if col == "intervention_rate":
                P = c.pivot_table(index="seed", columns="block", values=col).dropna(axis=1)
                xs = c.groupby("block").step.mean().loc[P.columns].values / 1e6
            else:
                uu = u.dropna(subset=[col])
                grid = np.linspace(uu.step.min(), uu.step.max(), 40)
                P = pd.DataFrame({s: np.interp(grid, d.step, d[col]) for s, d in uu.groupby("seed")}).T
                xs = grid / 1e6
            band(ax, xs, P.values, "#c44e52", "", 1.6)
            if i == 0:
                ax.set_title(ENV_TITLE[env])
            if j == 0:
                ax.set_ylabel(lab)
            if i == len(rows) - 1:
                ax.set_xlabel("environment steps (millions)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dynamics.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["cartpole", "pointgoal", "cargoal", "arm"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = pd.read_csv(os.path.join(a.data, "c2p_runs.csv"))
    E = pd.read_csv(os.path.join(a.data, "c2p_evals.csv"))
    C = pd.read_csv(os.path.join(a.data, "c2p_curves.csv"))
    U = pd.read_csv(os.path.join(a.data, "c2p_updates.csv"))
    envs = [e for e in a.envs if e in set(R.env)]
    main_table(R, envs, a.out)
    D = tests_table(R, envs, a.out)
    learning_curves(E, C, envs, a.out)
    tradeoff(R, envs, a.out)
    dynamics(U, C, envs, a.out)
    print(D.round(4).to_string())


if __name__ == "__main__":
    main()
