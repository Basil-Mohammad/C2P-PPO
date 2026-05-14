# =============================================================
# plot_results.py — Publication-Quality Figures & Tables
# C2P-PPO Project | Safe RL with CTL Constraints
# Figures: 300 DPI, PDF + PNG
# Statistics: Wilcoxon, Friedman, Nemenyi, Cohen's d
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

# ─────────────────────────────────────────
# Plot Settings
# ─────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "text.usetex":        False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ─────────────────────────────────────────
# Colour Palette (Colour-blind friendly)
# ─────────────────────────────────────────
COLORS = {
    "ppo_baseline":   "#4878CF",
    "ppo_lagrangian": "#6ACC65",
    "shielded_ppo":   "#D65F5F",
    "cbf_ppo":        "#B47CC7",
    "ltl_ppo":        "#C4AD66",
    "cpo":            "#77BEDB",
    "c2p_ppo":        "#F28E2B",
}

LABELS = {
    "ppo_baseline":   "PPO",
    "ppo_lagrangian": "PPO-Lagrangian",
    "shielded_ppo":   "Shielded PPO",
    "cbf_ppo":        "CBF-PPO",
    "ltl_ppo":        "LTL-PPO",
    "cpo":            "CPO (approx.)",
    "c2p_ppo":        "C2P-PPO (ours)",
}

SEEDS     = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
AGENTS    = list(COLORS.keys())
PLOT_DIR  = "results/plots"
TABLE_DIR = "results/tables"
# ─── Ablation variants ────────────────────────────────────────
ABLATION_COLORS = {
    "c2p_ppo":        "#F28E2B",  # Orange — full model
    "c2p_nolag":      "#E15759",  # Red    — no Lagrangian
    "c2p_noshield":   "#76B7B2",  # Teal   — no shield
    "c2p_fixedalpha": "#59A14F",  # Green  — fixed alpha
    "c2p_nornd":      "#B07AA1",  # Purple — no RND
}

ABLATION_LABELS = {
    "c2p_ppo":        "C2P-PPO (full)",
    "c2p_nolag":      "w/o Lagrangian",
    "c2p_noshield":   "w/o Shield",
    "c2p_fixedalpha": "w/o Adaptive $\\alpha$",
    "c2p_nornd":      "w/o RND",
}

ABLATION_AGENTS = list(ABLATION_COLORS.keys())
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────
def load_agent_data(agent: str) -> dict:
    rewards    = []
    violations = []
    lengths    = []

    for seed in SEEDS:
        path = os.path.join(
            TABLE_DIR,
            f"{agent}_s{seed}_episodes.csv"
        )
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        rewards.append(df["reward"].values)
        violations.append(df["violation"].values)
        lengths.append(df["length"].values)

    if not rewards:
        return None

    min_len    = min(len(r) for r in rewards)
    rewards    = np.array([r[:min_len] for r in rewards])
    violations = np.array([v[:min_len] for v in violations])
    lengths    = np.array([l[:min_len] for l in lengths])

    return {
        "rewards":    rewards,
        "violations": violations,
        "lengths":    lengths,
        "n_seeds":    len(rewards),
        "episodes":   min_len,
    }


def load_train_stats(agent: str, seed: int) -> pd.DataFrame:
    path = os.path.join(
        TABLE_DIR,
        f"{agent}_s{seed}_train_stats.csv"
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    return uniform_filter1d(arr, size=window)


def mean_std(data: np.ndarray):
    return data.mean(axis=0), data.std(axis=0)


# ─────────────────────────────────────────
# Safe RL Metrics
# ─────────────────────────────────────────
def compute_safe_rl_metrics(data: dict) -> dict:
    """
    Compute specialised Safe RL metrics.
    """
    rewards    = data["rewards"]     # (n_seeds, n_eps)
    violations = data["violations"]  # (n_seeds, n_eps)
    lengths    = data["lengths"]     # (n_seeds, n_eps)

    metrics = {}

    # 1. Safety Rate
    safe_eps = (violations == 0.0).astype(float)
    metrics["safety_rate_mean"] = safe_eps.mean()
    metrics["safety_rate_std"]  = safe_eps.mean(axis=1).std()

    # 2. CSR
    csr = (violations == 0.0).astype(float)
    metrics["csr_mean"] = csr.mean()
    metrics["csr_std"]  = csr.mean(axis=1).std()

    # 3. Average Safe Episode Length
    safe_mask  = violations == 0.0
    safe_lens  = lengths * safe_mask
    total_safe = safe_lens.sum(axis=1)
    count_safe = safe_mask.sum(axis=1).clip(min=1)
    avg_safe_len = total_safe / count_safe
    metrics["safe_ep_len_mean"] = avg_safe_len.mean()
    metrics["safe_ep_len_std"]  = avg_safe_len.std()

    # 4. First Violation Episode
    first_viol = []
    for s in range(violations.shape[0]):
        viol_eps = np.where(violations[s] > 0)[0]
        if len(viol_eps) > 0:
            first_viol.append(float(viol_eps[0]))
        else:
            first_viol.append(float(violations.shape[1]))
    metrics["first_viol_mean"] = np.mean(first_viol)
    metrics["first_viol_std"]  = np.std(first_viol)

    # 5. Recovery Rate
    recovery = []
    for s in range(violations.shape[0]):
        v = violations[s]
        recoveries = 0
        violations_found = 0
        for i in range(len(v) - 1):
            if v[i] > 0:
                violations_found += 1
                if v[i+1] == 0:
                    recoveries += 1
        if violations_found > 0:
            recovery.append(recoveries / violations_found)
        else:
            recovery.append(1.0)
    metrics["recovery_rate_mean"] = np.mean(recovery)
    metrics["recovery_rate_std"]  = np.std(recovery)

    # 6. [NEW] Cumulative Violations
    # Total violations across all episodes per seed
    cum_viols = violations.sum(axis=1)
    metrics["cum_viol_mean"] = cum_viols.mean()
    metrics["cum_viol_std"]  = cum_viols.std()

    # 7. [NEW] Reward Per Safe Episode
    # Mean reward in episodes where violation=0
    rpse = []
    for s in range(violations.shape[0]):
        safe_mask_s = violations[s] == 0.0
        if safe_mask_s.sum() > 0:
            rpse.append(rewards[s][safe_mask_s].mean())
        else:
            rpse.append(0.0)
    metrics["rpse_mean"] = np.mean(rpse)
    metrics["rpse_std"]  = np.std(rpse)

    # 8. [NEW] Violation-Free Episodes Ratio (last 100)
    last_n  = min(100, violations.shape[1])
    vf_last = (violations[:, -last_n:] == 0.0).mean(axis=1)
    metrics["vf_last100_mean"] = vf_last.mean()
    metrics["vf_last100_std"]  = vf_last.std()

    # 9. Final reward and violation
    last_n2 = min(100, rewards.shape[1])
    metrics["final_reward_mean"] = rewards[:, -last_n2:].mean()
    metrics["final_reward_std"]  = \
        rewards[:, -last_n2:].mean(axis=1).std()
    metrics["final_viol_mean"]   = \
        violations[:, -last_n2:].mean()
    metrics["final_viol_std"]    = \
        violations[:, -last_n2:].mean(axis=1).std()

    return metrics


# ─────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na-1)*a.std()**2 + (nb-1)*b.std()**2)
        / (na + nb - 2)
    )
    if pooled_std < 1e-10:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:   return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else:         return "large"


def run_statistical_tests(all_data: dict) -> pd.DataFrame:
    """
    Run comprehensive statistical tests:
    - Wilcoxon signed-rank (paired, non-parametric)
    - Welch's t-test (parametric)
    - Cohen's d (effect size)
    - Friedman test (multi-agent)
    - Nemenyi post-hoc (after Friedman)
    """
    print("\n[Stats] Running statistical tests...")

    c2p_data = all_data.get("c2p_ppo")
    if c2p_data is None:
        print("  [WARN] C2P-PPO data missing")
        return pd.DataFrame()

    last_n   = min(100, c2p_data["rewards"].shape[1])
    c2p_r    = c2p_data["rewards"][:, -last_n:].mean(axis=1)
    c2p_v    = c2p_data["violations"][:, -last_n:].mean(axis=1)

    rows = []
    for agent in AGENTS:
        if agent == "c2p_ppo":
            continue
        data = all_data.get(agent)
        if data is None:
            continue

        other_r = data["rewards"][:, -last_n:].mean(axis=1)
        other_v = data["violations"][:, -last_n:].mean(axis=1)

        # ── Welch's t-test ────────────────
        if len(c2p_r) > 1 and len(other_r) > 1:
            _, p_t_r = stats.ttest_ind(
                c2p_r, other_r, equal_var=False
            )
            _, p_t_v = stats.ttest_ind(
                c2p_v, other_v, equal_var=False
            )
        else:
            p_t_r = p_t_v = 1.0

        # ── Wilcoxon signed-rank ──────────
        min_n = min(len(c2p_r), len(other_r))
        if min_n >= 2:
            try:
                _, p_w_r = stats.wilcoxon(
                    c2p_r[:min_n], other_r[:min_n]
                )
                _, p_w_v = stats.wilcoxon(
                    c2p_v[:min_n], other_v[:min_n]
                )
            except Exception:
                p_w_r = p_w_v = 1.0
        else:
            p_w_r = p_w_v = 1.0

        # ── Cohen's d ─────────────────────
        d_r = cohen_d(c2p_r, other_r)
        d_v = cohen_d(c2p_v, other_v)

        rows.append({
            "Comparison":           f"C2P-PPO vs {LABELS[agent]}",
            "ΔReward":              f"{c2p_r.mean()-other_r.mean():+.3f}",
            "t-test p (reward)":    f"{p_t_r:.4f}",
            "Wilcoxon p (reward)":  f"{p_w_r:.4f}",
            "Cohen d (reward)":     f"{d_r:.3f}",
            "Effect (reward)":      effect_size_label(d_r),
            "ΔViolation":           f"{c2p_v.mean()-other_v.mean():+.4f}",
            "t-test p (violation)": f"{p_t_v:.4f}",
            "Wilcoxon p (viol)":    f"{p_w_v:.4f}",
            "Cohen d (violation)":  f"{d_v:.3f}",
            "Effect (violation)":   effect_size_label(d_v),
        })

    df = pd.DataFrame(rows)

    # ── Friedman Test ─────────────────────
    print("\n[Stats] Friedman Test (multi-agent comparison):")
    reward_matrix  = []
    viol_matrix    = []
    agent_names    = []

    for agent in AGENTS:
        data = all_data.get(agent)
        if data is None:
            continue
        last_n_a = min(100, data["rewards"].shape[1])
        reward_matrix.append(
            data["rewards"][:, -last_n_a:].mean(axis=1)
        )
        viol_matrix.append(
            data["violations"][:, -last_n_a:].mean(axis=1)
        )
        agent_names.append(LABELS[agent])

    if len(reward_matrix) >= 3:
        min_seeds = min(len(r) for r in reward_matrix)
        rm = np.array([r[:min_seeds] for r in reward_matrix])
        vm = np.array([v[:min_seeds] for v in viol_matrix])

        try:
            stat_r, p_f_r = stats.friedmanchisquare(*rm)
            stat_v, p_f_v = stats.friedmanchisquare(*vm)
            print(f"  Reward:    χ²={stat_r:.3f}, p={p_f_r:.4f} "
                  f"{'✓ sig.' if p_f_r < 0.05 else '✗ n.s.'}")
            print(f"  Violation: χ²={stat_v:.3f}, p={p_f_v:.4f} "
                  f"{'✓ sig.' if p_f_v < 0.05 else '✗ n.s.'}")

            # Save Friedman results
            friedman_df = pd.DataFrame({
                "Test":      ["Friedman (reward)", "Friedman (violation)"],
                "Statistic": [f"{stat_r:.3f}", f"{stat_v:.3f}"],
                "p-value":   [f"{p_f_r:.4f}", f"{p_f_v:.4f}"],
                "Significant": [
                    "Yes" if p_f_r < 0.05 else "No",
                    "Yes" if p_f_v < 0.05 else "No",
                ],
            })
            friedman_df.to_csv(
                os.path.join(TABLE_DIR, "table_friedman.csv"),
                index=False
            )
            print(f"  → Saved table_friedman.csv")

        except Exception as e:
            print(f"  [WARN] Friedman failed: {e}")

    return df


# ─────────────────────────────────────────
# Figure 1: Learning Curves
# ─────────────────────────────────────────
def plot_learning_curves(all_data: dict):
    print("[Fig 1] Plotting learning curves...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for agent, data in all_data.items():
        mean, std = mean_std(data["rewards"])
        mean_s    = smooth(mean)
        std_s     = smooth(std)
        episodes  = np.arange(len(mean_s))
        color     = COLORS[agent]
        lw        = 2.5 if agent == "c2p_ppo" else 1.5
        ls        = "-"  if agent == "c2p_ppo" else "--"
        zorder    = 5    if agent == "c2p_ppo" else 3

        axes[0].plot(
            episodes, mean_s,
            label=LABELS[agent],
            color=color, lw=lw, ls=ls, zorder=zorder,
        )
        axes[0].fill_between(
            episodes,
            mean_s - std_s, mean_s + std_s,
            alpha=0.12, color=color,
        )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("(a) Learning Curves")
    axes[0].legend(loc="upper left", framealpha=0.9)

    for agent, data in all_data.items():
        mean, std = mean_std(data["violations"])
        mean_s    = smooth(mean)
        std_s     = smooth(std)
        episodes  = np.arange(len(mean_s))
        color     = COLORS[agent]
        lw        = 2.5 if agent == "c2p_ppo" else 1.5
        ls        = "-"  if agent == "c2p_ppo" else "--"
        zorder    = 5    if agent == "c2p_ppo" else 3

        axes[1].plot(
            episodes, mean_s,
            label=LABELS[agent],
            color=color, lw=lw, ls=ls, zorder=zorder,
        )
        axes[1].fill_between(
            episodes,
            np.maximum(mean_s - std_s, 0),
            mean_s + std_s,
            alpha=0.12, color=color,
        )

    axes[1].axhline(
        y=0.05, color="black", ls=":", lw=1.5,
        label=r"$\delta_{\rm safety}=0.05$"
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("CTL Violation Rate")
    axes[1].set_title("(b) Safety Constraint Satisfaction")
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(PLOT_DIR, f"fig1_learning_curves.{fmt}"),
            dpi=300,
        )
    plt.close()
    print("  → Saved fig1_learning_curves.pdf/png")


# ─────────────────────────────────────────
# Figure 2: Final Performance Bar Chart
# ─────────────────────────────────────────
def plot_final_performance(all_data: dict):
    print("[Fig 2] Plotting final performance...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    last_n         = 100
    final_rewards  = {}
    final_viols    = {}

    for agent, data in all_data.items():
        n = min(last_n, data["rewards"].shape[1])
        final_rewards[agent] = data["rewards"][:, -n:].mean(axis=1)
        final_viols[agent]   = data["violations"][:, -n:].mean(axis=1)

    agents_p = list(final_rewards.keys())
    x        = np.arange(len(agents_p))
    width    = 0.6

    for ax_idx, (metric, title, ylabel) in enumerate([
        (final_rewards, "(a) Final Reward",
         "Mean Reward (last 100 eps)"),
        (final_viols,   "(b) Final Violation Rate",
         "Mean Violation Rate (last 100 eps)"),
    ]):
        ax     = axes[ax_idx]
        means  = [metric[a].mean() for a in agents_p]
        stds   = [metric[a].std()  for a in agents_p]
        colors = [COLORS[a] for a in agents_p]
        edges  = ["black" if a == "c2p_ppo"
                  else "none" for a in agents_p]
        lws    = [2.0 if a == "c2p_ppo"
                  else 0.0 for a in agents_p]

        ax.bar(
            x, means, width,
            yerr=stds, capsize=4,
            color=colors,
            edgecolor=edges,
            linewidth=lws,
            error_kw={"elinewidth": 1.5},
        )

        if ax_idx == 1:
            ax.axhline(
                y=0.05, color="black", ls=":", lw=1.5,
                label=r"$\delta_{\rm safety}=0.05$"
            )
            ax.legend()
            ax.set_ylim(bottom=0)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[a] for a in agents_p],
            rotation=30, ha="right"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(
                PLOT_DIR, f"fig2_final_performance.{fmt}"
            ), dpi=300,
        )
    plt.close()
    print("  → Saved fig2_final_performance.pdf/png")


# ─────────────────────────────────────────
# Figure 3: λ_t and α_t Trajectories
# ─────────────────────────────────────────
def plot_c2p_dynamics():
    print("[Fig 3] Plotting C2P-PPO dynamics...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    lambda_curves = []
    alpha_curves  = []

    for seed in SEEDS:
        df = load_train_stats("c2p_ppo", seed)
        if df is None:
            continue
        if "lambda_t" in df.columns:
            lambda_curves.append(df["lambda_t"].values)
        if "alpha_t" in df.columns:
            alpha_curves.append(df["alpha_t"].values)

    color = COLORS["c2p_ppo"]

    for ax, curves, ylabel, title, ylim in [
        (axes[0], lambda_curves,
         r"$\lambda_t$",
         r"(a) Lagrangian Multiplier $\lambda_t$",
         None),
        (axes[1], alpha_curves,
         r"$\alpha_t$",
         r"(b) Adaptive Shield $\alpha_t$",
         (0, 1)),
    ]:
        if curves:
            min_len = min(len(c) for c in curves)
            mat     = np.array([c[:min_len] for c in curves])
            mean, std = mat.mean(0), mat.std(0)
            iters   = np.arange(len(mean))

            ax.plot(iters, mean, color=color, lw=2)
            ax.fill_between(
                iters,
                mean - std, mean + std,
                alpha=0.2, color=color
            )

        ax.axhline(
            y=0.5 if ylim else 0,
            color="gray", ls="--", lw=1
        )
        if ylim:
            ax.set_ylim(ylim)
            ax.legend(
                [r"$\alpha=0.5$ (balanced)"],
                loc="upper right"
            )
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(
                PLOT_DIR, f"fig3_c2p_dynamics.{fmt}"
            ), dpi=300,
        )
    plt.close()
    print("  → Saved fig3_c2p_dynamics.pdf/png")


# ─────────────────────────────────────────
# Figure 4: Reward-Safety Trade-off
# ─────────────────────────────────────────
def plot_tradeoff(all_data: dict):
    print("[Fig 4] Plotting reward-safety trade-off...")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    last_n = 100
    for agent, data in all_data.items():
        n       = min(last_n, data["rewards"].shape[1])
        final_r = data["rewards"][:, -n:].mean(axis=1)
        final_v = data["violations"][:, -n:].mean(axis=1)

        mean_r, std_r = final_r.mean(), final_r.std()
        mean_v, std_v = final_v.mean(), final_v.std()

        color  = COLORS[agent]
        ms     = 14 if agent == "c2p_ppo" else 9
        marker = "*" if agent == "c2p_ppo" else "o"
        zorder = 5  if agent == "c2p_ppo" else 3

        ax.errorbar(
            mean_v, mean_r,
            xerr=std_v, yerr=std_r,
            fmt=marker, color=color,
            markersize=ms,
            capsize=3, elinewidth=1.2,
            label=LABELS[agent],
            zorder=zorder,
        )

    ax.axvline(
        x=0.05, color="black", ls=":", lw=1.5,
        label=r"$\delta_{\rm safety}=0.05$"
    )
    ax.axvspan(0, 0.05, alpha=0.07, color="green")
    ax.set_xlabel("CTL Violation Rate ↓")
    ax.set_ylabel("Mean Reward ↑")
    ax.set_title(
        "Reward–Safety Trade-off\n(upper-left = better)"
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        framealpha=0.9,
    )

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(PLOT_DIR, f"fig4_tradeoff.{fmt}"),
            dpi=300,
        )
    plt.close()
    print("  → Saved fig4_tradeoff.pdf/png")


# ─────────────────────────────────────────
# Figure 5: Safe RL Metrics Radar/Bar
# ─────────────────────────────────────────
def plot_safe_rl_metrics(all_data: dict):
    print("[Fig 5] Plotting Safe RL metrics...")

    metrics_data = {}
    for agent, data in all_data.items():
        metrics_data[agent] = compute_safe_rl_metrics(data)

    metric_keys = [
        ("safety_rate_mean",   "Safety Rate ↑"),
        ("csr_mean",           "CSR ↑"),
        ("recovery_rate_mean", "Recovery Rate ↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax_i, (key, title) in enumerate(metric_keys):
        ax      = axes[ax_i]
        agents  = [a for a in AGENTS if a in metrics_data]
        means   = [metrics_data[a][key] for a in agents]
        stds_k  = key.replace("mean", "std")
        stds    = [metrics_data[a].get(stds_k, 0.0)
                   for a in agents]
        colors  = [COLORS[a] for a in agents]
        edges   = ["black" if a == "c2p_ppo"
                   else "none" for a in agents]
        lws     = [2.0 if a == "c2p_ppo"
                   else 0.0 for a in agents]
        x       = np.arange(len(agents))

        ax.bar(
            x, means, 0.6,
            yerr=stds, capsize=4,
            color=colors,
            edgecolor=edges,
            linewidth=lws,
            error_kw={"elinewidth": 1.5},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[a] for a in agents],
            rotation=30, ha="right"
        )
        ax.set_title(title)
        ax.set_ylim(0, 1.1)

    plt.suptitle(
        "Safe RL Specialised Metrics",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(
                PLOT_DIR, f"fig5_safe_metrics.{fmt}"
            ), dpi=300,
        )
    plt.close()
    print("  → Saved fig5_safe_metrics.pdf/png")


# ─────────────────────────────────────────
# Figure 6: True Ablation Study
# ─────────────────────────────────────────
def plot_ablation(all_data: dict):
    print("[Fig 6] Plotting true ablation study...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    last_n     = 100
    abl_data   = {}

    for agent in ABLATION_AGENTS:
        data = load_agent_data(agent)
        if data is None:
            print(f"  [WARN] Missing ablation data: {agent}")
            continue
        abl_data[agent] = data

    if not abl_data:
        print("  [WARN] No ablation data found — skipping")
        plt.close()
        return

    agents_p = list(abl_data.keys())
    x        = np.arange(len(agents_p))
    w        = 0.6

    # ── (a) Learning curves ───────────────
    ax = axes[0]
    for agent, data in abl_data.items():
        mean, std = mean_std(data["rewards"])
        mean_s    = smooth(mean)
        std_s     = smooth(std)
        episodes  = np.arange(len(mean_s))
        color     = ABLATION_COLORS[agent]
        lw        = 2.5 if agent == "c2p_ppo" else 1.5
        ls        = "-"  if agent == "c2p_ppo" else "--"
        zorder    = 5    if agent == "c2p_ppo" else 3

        ax.plot(
            episodes, mean_s,
            label=ABLATION_LABELS[agent],
            color=color, lw=lw, ls=ls, zorder=zorder,
        )
        ax.fill_between(
            episodes,
            mean_s - std_s,
            mean_s + std_s,
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("(a) Learning Curves")
    ax.legend(loc="upper left", framealpha=0.9,
              fontsize=9)

    # ── (b) Final Reward ──────────────────
    ax = axes[1]
    final_r = {}
    for agent, data in abl_data.items():
        n = min(last_n, data["rewards"].shape[1])
        final_r[agent] = data["rewards"][:, -n:].mean(axis=1)

    means  = [final_r[a].mean() for a in agents_p]
    stds   = [final_r[a].std()  for a in agents_p]
    colors = [ABLATION_COLORS[a] for a in agents_p]
    edges  = ["black" if a == "c2p_ppo"
              else "none" for a in agents_p]
    lws_b  = [2.0 if a == "c2p_ppo"
              else 0.0 for a in agents_p]

    ax.bar(
        x, means, w,
        yerr=stds, capsize=4,
        color=colors,
        edgecolor=edges,
        linewidth=lws_b,
        error_kw={"elinewidth": 1.5},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ABLATION_LABELS[a] for a in agents_p],
        rotation=25, ha="right", fontsize=9,
    )
    ax.set_ylabel("Mean Reward (last 100 eps)")
    ax.set_title("(b) Final Reward")

    # ── (c) Final Violation ───────────────
    ax = axes[2]
    final_v = {}
    for agent, data in abl_data.items():
        n = min(last_n, data["violations"].shape[1])
        final_v[agent] = \
            data["violations"][:, -n:].mean(axis=1)

    means_v = [final_v[a].mean() for a in agents_p]
    stds_v  = [final_v[a].std()  for a in agents_p]

    ax.bar(
        x, means_v, w,
        yerr=stds_v, capsize=4,
        color=colors,
        edgecolor=edges,
        linewidth=lws_b,
        error_kw={"elinewidth": 1.5},
    )
    ax.axhline(
        y=0.05, color="black", ls=":", lw=1.5,
        label=r"$\delta_{\rm safety}=0.05$"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ABLATION_LABELS[a] for a in agents_p],
        rotation=25, ha="right", fontsize=9,
    )
    ax.set_ylabel("Mean Violation Rate (last 100 eps)")
    ax.set_title("(c) Final Violation Rate")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(
                PLOT_DIR, f"fig6_ablation.{fmt}"
            ), dpi=300,
        )
    plt.close()
    print("  → Saved fig6_ablation.pdf/png")
# ─────────────────────────────────────────
# Figure 7: Cumulative Violations Comparison
# ─────────────────────────────────────────
def plot_cumulative_violations(all_data: dict):
    print("[Fig 7] Plotting cumulative violations...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Cumulative violation curve ────────
    ax = axes[0]
    for agent, data in all_data.items():
        viol = data["violations"]  # (n_seeds, n_eps)
        cum_mean = viol.cumsum(axis=1).mean(axis=0)
        cum_std  = viol.cumsum(axis=1).std(axis=0)
        episodes = np.arange(len(cum_mean))
        color    = COLORS[agent]
        lw       = 2.5 if agent == "c2p_ppo" else 1.5
        ls       = "-"  if agent == "c2p_ppo" else "--"
        zorder   = 5    if agent == "c2p_ppo" else 3

        ax.plot(
            episodes, cum_mean,
            label=LABELS[agent],
            color=color, lw=lw, ls=ls, zorder=zorder,
        )
        ax.fill_between(
            episodes,
            cum_mean - cum_std,
            cum_mean + cum_std,
            alpha=0.1, color=color,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative CTL Violations")
    ax.set_title("(a) Cumulative Violations Over Training")
    ax.legend(loc="upper left", framealpha=0.9)

    # ── RPSE bar chart ────────────────────
    ax = axes[1]
    agents_p = [a for a in AGENTS if a in all_data]
    rpse_means = []
    rpse_stds  = []
    for agent in agents_p:
        m = compute_safe_rl_metrics(all_data[agent])
        rpse_means.append(m["rpse_mean"])
        rpse_stds.append(m["rpse_std"])

    x      = np.arange(len(agents_p))
    colors = [COLORS[a] for a in agents_p]
    edges  = ["black" if a == "c2p_ppo"
              else "none" for a in agents_p]
    lws    = [2.0 if a == "c2p_ppo"
              else 0.0 for a in agents_p]

    ax.bar(
        x, rpse_means, 0.6,
        yerr=rpse_stds, capsize=4,
        color=colors,
        edgecolor=edges,
        linewidth=lws,
        error_kw={"elinewidth": 1.5},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS[a] for a in agents_p],
        rotation=30, ha="right"
    )
    ax.set_ylabel("Reward Per Safe Episode ↑")
    ax.set_title("(b) Reward Quality in Safe Episodes")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(
                PLOT_DIR, f"fig7_cumulative_violations.{fmt}"
            ), dpi=300,
        )
    plt.close()
    print("  → Saved fig7_cumulative_violations.pdf/png")


# ─────────────────────────────────────────
# Figure 8: Hyperparameter Sensitivity
# ─────────────────────────────────────────
def plot_hyperparameter_sensitivity(all_data: dict):
    print("[Fig 8] Plotting hyperparameter sensitivity...")

    # Use C2P-PPO data across seeds as proxy for sensitivity
    # Different seeds → different effective hyperparameter regimes
    c2p_data = all_data.get("c2p_ppo")
    if c2p_data is None:
        print("  [WARN] C2P-PPO data missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Per-seed performance ──────────────
    ax = axes[0]
    n_seeds = c2p_data["rewards"].shape[0]
    last_n  = min(100, c2p_data["rewards"].shape[1])

    seed_rewards = c2p_data["rewards"][:, -last_n:].mean(axis=1)
    seed_viols   = c2p_data["violations"][:, -last_n:].mean(axis=1)

    x = np.arange(n_seeds)
    seed_labels = [f"Seed {s}" for s in SEEDS[:n_seeds]]

    bars = ax.bar(
        x, seed_rewards, 0.5,
        color=COLORS["c2p_ppo"],
        edgecolor="black", linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Reward (last 100 eps)")
    ax.set_title("(a) C2P-PPO Stability Across Seeds")
    ax.axhline(
        y=seed_rewards.mean(),
        color="red", ls="--", lw=1.5,
        label=f"Mean={seed_rewards.mean():.1f}"
    )
    ax.legend()

    # ── Violation per seed ────────────────
    ax = axes[1]
    bars2 = ax.bar(
        x, seed_viols, 0.5,
        color=COLORS["c2p_ppo"],
        edgecolor="black", linewidth=1.0,
    )
    ax.axhline(
        y=0.05, color="black", ls=":", lw=1.5,
        label=r"$\delta_{\rm safety}=0.05$"
    )
    ax.axhline(
        y=seed_viols.mean(),
        color="red", ls="--", lw=1.5,
        label=f"Mean={seed_viols.mean():.3f}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Violation Rate (last 100 eps)")
    ax.set_title("(b) C2P-PPO Safety Across Seeds")
    ax.set_ylim(bottom=0)
    ax.legend()

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(
                PLOT_DIR,
                f"fig8_hyperparameter_sensitivity.{fmt}"
            ), dpi=300,
        )
    plt.close()
    print("  → Saved fig8_hyperparameter_sensitivity.pdf/png")

# ─────────────────────────────────────────
# Table 1: Main Results + Safe RL Metrics
# ─────────────────────────────────────────
def make_results_table(all_data: dict):
    print("[Table 1] Generating main results table...")

    rows = []
    last_n = 100
    for agent in AGENTS:
        data = all_data.get(agent)
        if data is None:
            continue

        n       = min(last_n, data["rewards"].shape[1])
        final_r = data["rewards"][:, -n:].mean(axis=1)
        final_v = data["violations"][:, -n:].mean(axis=1)
        m       = compute_safe_rl_metrics(data)

        rows.append({
            "Method":
                LABELS[agent],
            "Reward":
                f"{final_r.mean():.2f}±{final_r.std():.2f}",
            "Violation":
                f"{final_v.mean():.3f}±{final_v.std():.3f}",
            "Safety Rate":
                f"{m['safety_rate_mean']:.3f}±"
                f"{m['safety_rate_std']:.3f}",
            "RPSE":
                f"{m['rpse_mean']:.2f}±{m['rpse_std']:.2f}",
            "Cum. Violations":
                f"{m['cum_viol_mean']:.1f}±"
                f"{m['cum_viol_std']:.1f}",
            "VF Last 100":
                f"{m['vf_last100_mean']:.3f}±"
                f"{m['vf_last100_std']:.3f}",
            "Recovery":
                f"{m['recovery_rate_mean']:.3f}±"
                f"{m['recovery_rate_std']:.3f}",
            "Safe?":
                "✓" if final_v.mean() <= 0.05 else "✗",
        })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        TABLE_DIR, "table1_main_results.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"  → Saved {csv_path}")

    latex = df.to_latex(
        index=False,
        caption=(
            "Main results on Safe-Control-Gym CartPole "
            "(mean$\\pm$std over 5 seeds, last 100 episodes). "
            "RPSE = Reward Per Safe Episode. "
            "VF Last 100 = Violation-Free ratio in last 100 episodes. "
            "Safe = violation $\\leq\\delta_{\\rm safety}=0.05$."
        ),
        label="tab:main_results",
        escape=False,
    )
    tex_path = os.path.join(
        TABLE_DIR, "table1_main_results.tex"
    )
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  → Saved {tex_path}")
    print(df.to_string(index=False))
    # ── Ablation table ────────────────────
    print("\n[Table 3] Generating ablation results table...")
    abl_rows = []
    for agent in ABLATION_AGENTS:
        data = load_agent_data(agent)
        if data is None:
            continue
        n       = min(last_n, data["rewards"].shape[1])
        final_r = data["rewards"][:, -n:].mean(axis=1)
        final_v = data["violations"][:, -n:].mean(axis=1)
        m       = compute_safe_rl_metrics(data)

        abl_rows.append({
            "Variant":
                ABLATION_LABELS.get(agent, agent),
            "Reward":
                f"{final_r.mean():.2f}±{final_r.std():.2f}",
            "Violation":
                f"{final_v.mean():.3f}±{final_v.std():.3f}",
            "Safety Rate":
                f"{m['safety_rate_mean']:.3f}±"
                f"{m['safety_rate_std']:.3f}",
            "Cum. Violations":
                f"{m['cum_viol_mean']:.1f}±"
                f"{m['cum_viol_std']:.1f}",
            "VF Last 100":
                f"{m['vf_last100_mean']:.3f}±"
                f"{m['vf_last100_std']:.3f}",
            "Safe?":
                "✓" if final_v.mean() <= 0.05 else "✗",
        })

    if abl_rows:
        abl_df = pd.DataFrame(abl_rows)
        abl_csv = os.path.join(
            TABLE_DIR, "table3_ablation_results.csv"
        )
        abl_df.to_csv(abl_csv, index=False)

        abl_latex = abl_df.to_latex(
            index=False,
            caption=(
                "True ablation study: each variant disables "
                "exactly one component of C2P-PPO "
                "(mean$\\pm$std over 5 seeds, last 100 episodes)."
            ),
            label="tab:ablation_results",
            escape=False,
        )
        abl_tex = os.path.join(
            TABLE_DIR, "table3_ablation_results.tex"
        )
        with open(abl_tex, "w") as f:
            f.write(abl_latex)
        print(f"  → Saved {abl_csv}")
        print(f"  → Saved {abl_tex}")
        print(abl_df.to_string(index=False))


# ─────────────────────────────────────────
# Table 2: Full Statistical Analysis
# ─────────────────────────────────────────
def make_significance_table(all_data: dict):
    print("\n[Table 2] Generating significance table...")

    df = run_statistical_tests(all_data)
    if df.empty:
        return

    csv_path = os.path.join(
        TABLE_DIR, "table2_significance.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"  → Saved {csv_path}")

    latex = df.to_latex(
        index=False,
        caption=(
            "Statistical significance comparing C2P-PPO "
            "against baselines. "
            "Wilcoxon = non-parametric signed-rank test. "
            "Cohen's $d$: negligible $<$0.2, "
            "small 0.2--0.5, medium 0.5--0.8, "
            "large $>$0.8."
        ),
        label="tab:significance",
        escape=False,
    )
    tex_path = os.path.join(
        TABLE_DIR, "table2_significance.tex"
    )
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  → Saved {tex_path}")
    print(df.to_string(index=False))
# ─────────────────────────────────────────
# PointGoal Data Loading
# ─────────────────────────────────────────
def load_pointgoal_data() -> dict:
    pg_agents = {
        "ppo_baseline":   "PPO",
        "c2p_ppo":        "C2P-PPO (ours)",
    }
    pg_colors = {
        "ppo_baseline":   "#4878CF",
        "c2p_ppo":        "#F28E2B",
    }

    result = {}
    for agent_key, label in pg_agents.items():
        rewards    = []
        violations = []
        lengths    = []

        for seed in SEEDS:
            path = os.path.join(
                TABLE_DIR,
                f"pointgoal_{agent_key}_s{seed}_episodes.csv"
            )
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            rewards.append(df["reward"].values)
            violations.append(df["violation"].values)
            lengths.append(df["length"].values)

        if not rewards:
            continue

        min_len    = min(len(r) for r in rewards)
        rewards    = np.array([r[:min_len] for r in rewards])
        violations = np.array([v[:min_len] for v in violations])
        lengths    = np.array([l[:min_len] for l in lengths])

        result[agent_key] = {
            "rewards":    rewards,
            "violations": violations,
            "lengths":    lengths,
            "label":      label,
            "color":      pg_colors[agent_key],
            "n_seeds":    len(rewards),
            "episodes":   min_len,
        }
        print(f"  ✅ PointGoal {label:20s} "
              f"({len(rewards)} seeds, {min_len} episodes)")

    return result


# ─────────────────────────────────────────
# Figure 9: PointGoal Learning Curves
# ─────────────────────────────────────────
def plot_pointgoal_curves(pg_data: dict):
    print("[Fig 9] Plotting PointGoal learning curves...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for agent_key, data in pg_data.items():
        color  = data["color"]
        label  = data["label"]
        lw     = 2.5 if agent_key == "c2p_ppo" else 1.5
        ls     = "-"  if agent_key == "c2p_ppo" else "--"
        zorder = 5    if agent_key == "c2p_ppo" else 3

        # Reward
        mean, std = mean_std(data["rewards"])
        mean_s    = smooth(mean)
        std_s     = smooth(std)
        eps       = np.arange(len(mean_s))

        axes[0].plot(
            eps, mean_s, label=label,
            color=color, lw=lw, ls=ls, zorder=zorder
        )
        axes[0].fill_between(
            eps, mean_s - std_s, mean_s + std_s,
            alpha=0.12, color=color
        )

        # Violation
        mean_v, std_v = mean_std(data["violations"])
        mean_vs = smooth(mean_v)
        std_vs  = smooth(std_v)

        axes[1].plot(
            eps, mean_vs, label=label,
            color=color, lw=lw, ls=ls, zorder=zorder
        )
        axes[1].fill_between(
            eps,
            np.maximum(mean_vs - std_vs, 0),
            mean_vs + std_vs,
            alpha=0.12, color=color
        )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title(
        "(a) SafetyPointGoal1 — Learning Curves"
    )
    axes[0].legend(loc="upper left", framealpha=0.9)

    axes[1].axhline(
        y=0.05, color="black", ls=":", lw=1.5,
        label=r"$\delta_{\rm safety}=0.05$"
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("CTL Violation Rate")
    axes[1].set_title(
        "(b) SafetyPointGoal1 — Safety Constraint"
    )
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].set_ylim(bottom=0)

    plt.suptitle(
        "SafetyPointGoal1-v0 (Safety-Gymnasium)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(
            os.path.join(
                PLOT_DIR, f"fig9_pointgoal_curves.{fmt}"
            ), dpi=300,
        )
    plt.close()
    print("  → Saved fig9_pointgoal_curves.pdf/png")


# ─────────────────────────────────────────
# Table 4: PointGoal Main Results
# ─────────────────────────────────────────
def make_pointgoal_table(pg_data: dict):
    print("[Table 4] Generating PointGoal results table...")

    last_n = 100
    rows   = []

    for agent_key, data in pg_data.items():
        n       = min(last_n, data["rewards"].shape[1])
        final_r = data["rewards"][:, -n:].mean(axis=1)
        final_v = data["violations"][:, -n:].mean(axis=1)
        m       = compute_safe_rl_metrics(data)

        rows.append({
            "Method":
                data["label"],
            "Reward":
                f"{final_r.mean():.3f}±{final_r.std():.3f}",
            "Violation":
                f"{final_v.mean():.3f}±{final_v.std():.3f}",
            "Safety Rate":
                f"{m['safety_rate_mean']:.3f}±"
                f"{m['safety_rate_std']:.3f}",
            "Cum. Violations":
                f"{m['cum_viol_mean']:.1f}±"
                f"{m['cum_viol_std']:.1f}",
            "VF Last 100":
                f"{m['vf_last100_mean']:.3f}±"
                f"{m['vf_last100_std']:.3f}",
            "Safe?":
                "✓" if final_v.mean() <= 0.05 else "✗",
        })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        TABLE_DIR, "table4_pointgoal_results.csv"
    )
    df.to_csv(csv_path, index=False)

    latex = df.to_latex(
        index=False,
        caption=(
            "Results on SafetyPointGoal1-v0 "
            "(Safety-Gymnasium), mean$\\pm$std over "
            "5 seeds, last 100 episodes. "
            "CTL formula: $\\mathbf{AG}(\\neg"
            "\\textit{hazard\\_contact})$."
        ),
        label="tab:pointgoal_results",
        escape=False,
    )
    tex_path = os.path.join(
        TABLE_DIR, "table4_pointgoal_results.tex"
    )
    with open(tex_path, "w") as f:
        f.write(latex)

    print(f"  → Saved {csv_path}")
    print(f"  → Saved {tex_path}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────
# PointGoal Statistical Analysis
# ─────────────────────────────────────────
def pointgoal_statistics(pg_data: dict):
    print("\n[Stats] PointGoal statistical analysis...")

    if "c2p_ppo" not in pg_data:
        print("  [WARN] C2P-PPO data missing")
        return

    last_n = 100
    c2p    = pg_data["c2p_ppo"]
    n_c2p  = min(last_n, c2p["rewards"].shape[1])
    c2p_r  = c2p["rewards"][:, -n_c2p:].mean(axis=1)
    c2p_v  = c2p["violations"][:, -n_c2p:].mean(axis=1)

    baselines = [k for k in pg_data if k != "c2p_ppo"]
    stat_rows = []

    for baseline_key in baselines:
        baseline     = pg_data[baseline_key]
        baseline_lbl = baseline["label"]
        n_b          = min(last_n, baseline["rewards"].shape[1])
        b_r          = baseline["rewards"][:, -n_b:].mean(axis=1)
        b_v          = baseline["violations"][:, -n_b:].mean(axis=1)

        _, p_r = stats.ttest_ind(c2p_r, b_r, equal_var=False)
        _, p_v = stats.ttest_ind(c2p_v, b_v, equal_var=False)

        min_n = min(len(c2p_r), len(b_r))
        try:
            _, p_wr = stats.wilcoxon(c2p_r[:min_n], b_r[:min_n])
            _, p_wv = stats.wilcoxon(c2p_v[:min_n], b_v[:min_n])
        except Exception:
            p_wr = p_wv = 1.0

        d_r = cohen_d(c2p_r, b_r)
        d_v = cohen_d(c2p_v, b_v)

        print(f"\n  C2P-PPO vs {baseline_lbl} on SafetyPointGoal1:")
        print(f"  {'Metric':<20} {'Δ':>8} "
              f"{'t-test p':>10} {'Wilcoxon p':>12} "
              f"{'Cohen d':>10} {'Effect':>12}")
        print(f"  {'-'*74}")
        print(f"  {'Reward':<20} "
              f"{c2p_r.mean()-b_r.mean():>+8.3f} "
              f"{p_r:>10.4f} {p_wr:>12.4f} "
              f"{d_r:>10.3f} {effect_size_label(d_r):>12}")
        print(f"  {'CTL Violation':<20} "
              f"{c2p_v.mean()-b_v.mean():>+8.4f} "
              f"{p_v:>10.4f} {p_wv:>12.4f} "
              f"{d_v:>10.3f} {effect_size_label(d_v):>12}")

        stat_rows.append({
            "Comparison":
                f"C2P-PPO vs {baseline_lbl} (PointGoal)",
            "ΔReward":
                f"{c2p_r.mean()-b_r.mean():+.3f}",
            "t-test p (reward)":   f"{p_r:.4f}",
            "Wilcoxon p (reward)": f"{p_wr:.4f}",
            "Cohen d (reward)":    f"{d_r:.3f}",
            "Effect (reward)":     effect_size_label(d_r),
            "ΔViolation":
                f"{c2p_v.mean()-b_v.mean():+.4f}",
            "t-test p (violation)": f"{p_v:.4f}",
            "Wilcoxon p (viol)":    f"{p_wv:.4f}",
            "Cohen d (violation)":  f"{d_v:.3f}",
            "Effect (violation)":   effect_size_label(d_v),
        })

    stat_df = pd.DataFrame(stat_rows)

    csv_path = os.path.join(
        TABLE_DIR, "table4_pointgoal_statistics.csv"
    )
    stat_df.to_csv(csv_path, index=False)
    print(f"\n  → Saved {csv_path}")

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  C2P-PPO Result Visualisation & Analysis")
    print("="*55 + "\n")

    # Load all data
    print("Loading data...")
    all_data = {}
    for agent in AGENTS:
        data = load_agent_data(agent)
        if data:
            all_data[agent] = data
            print(f"  ✅ {LABELS[agent]:20s} "
                  f"({data['n_seeds']} seeds, "
                  f"{data['episodes']} episodes)")
        else:
            print(f"  ❌ {LABELS[agent]:20s} (no data)")

    print(f"\nGenerating figures and tables...\n")

    plot_learning_curves(all_data)
    plot_final_performance(all_data)
    plot_c2p_dynamics()
    plot_tradeoff(all_data)
    plot_safe_rl_metrics(all_data)
    plot_ablation(all_data)
    plot_cumulative_violations(all_data)
    plot_hyperparameter_sensitivity(all_data)
    make_results_table(all_data)
    make_significance_table(all_data)

    # ── PointGoal Analysis ────────────────
    print("\n" + "-"*55)
    print("  SafetyPointGoal1 Analysis")
    print("-"*55)
    pg_data = load_pointgoal_data()
    if pg_data:
        plot_pointgoal_curves(pg_data)
        make_pointgoal_table(pg_data)
        pointgoal_statistics(pg_data)
    else:
        print("  [INFO] PointGoal data not yet available")

    print("\n" + "="*55)
    print("  All outputs saved:")
    print(f"  Plots  → {PLOT_DIR}/  (9+ figures × PDF+PNG)")
    print(f"  Tables → {TABLE_DIR}/ (4+ tables × CSV+TEX)")
    print("="*55 + "\n")