"""Ablation table: IQM per variant and Holm-adjusted Mann-Whitney p-values against the full method."""
import sys, os
import pandas as pd
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(__file__))
from stats import iqm, holm
data, out = sys.argv[1], sys.argv[2]
R = pd.read_csv(os.path.join(data, "c2p_runs.csv"))
V = [("c2p_no_lagrangian", "w/o Lagrangian"), ("c2p_no_shield", "w/o shield"),
     ("c2p_no_inadm_penalty", "w/o blocked-action penalty"), ("c2p_no_is", "w/o importance weights"),
     ("c2p_fixed_alpha", "fixed $\\alpha$"), ("c2p_no_shaping", "w/o reward shaping"),
     ("c2p_no_rnd", "w/o RND"), ("c2p_geometric", "geometric mixture")]
ENVS = [("cartpole", "CartPole-Track", 0.05), ("hopper", "SafetyHopperVelocity", 0.025),
        ("cheetah", "SafetyHalfCheetahVelocity", 0.025), ("arm", "SafeArmReach-6DoF", 0.01)]
def p(x): return f"{x:.3f}" if x >= 0.001 else "$<$0.001"
lines = []
for env, title, B in ENVS:
    g = R[R.env == env]; base = g[g.run == "c2p_ppo"]
    rows = []
    for m, lab in V:
        h = g[g.run == m]
        if len(h) < 3:          # variant not run in this environment
            continue
        rows.append(dict(lab=lab, ret=iqm(h.eval_return), viol=iqm(h.eval_viol_rate), tcr=iqm(h.train_cost_rate),
                         feas=(h.eval_viol_rate <= B).sum(), n=len(h),
                         pr=mannwhitneyu(h.eval_return, base.eval_return).pvalue,
                         pv=mannwhitneyu(h.eval_viol_rate, base.eval_viol_rate).pvalue,
                         pt=mannwhitneyu(h.train_cost_rate, base.train_cost_rate).pvalue))
    T = pd.DataFrame(rows)
    for c in ["pr", "pv", "pt"]:
        T[c] = holm(T[c].values)
    lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}} (budget $\\delta={B}$)}}\\\\")
    lines.append(f"C2P-PPO (full) & {iqm(base.eval_return):.1f} & -- & {iqm(base.eval_viol_rate):.4f} & -- & "
                 f"{iqm(base.train_cost_rate):.4f} & -- & {(base.eval_viol_rate <= B).sum()}/{len(base)} \\\\")
    for _, r in T.iterrows():
        b = lambda v, s: f"\\textbf{{{s}}}" if v < 0.05 else s
        lines.append(f"{r.lab} & {r.ret:.1f} & {b(r.pr, p(r.pr))} & {r.viol:.4f} & {b(r.pv, p(r.pv))} & "
                     f"{r.tcr:.4f} & {b(r.pt, p(r.pt))} & {r.feas}/{r.n} \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
open(os.path.join(out, "tab_ablation_rows.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
