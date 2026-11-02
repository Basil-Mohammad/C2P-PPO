"""Sensitivity table: one-at-a-time variation of ten hyperparameters (5 seeds per setting)
against the default configuration on the same seeds; Holm correction over the 20 settings."""
import sys, os
import pandas as pd
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(__file__))
from stats import iqm, holm
data, out = sys.argv[1], sys.argv[2]
R = pd.read_csv(os.path.join(data, "c2p_runs.csv"))
P = [("beta", "$\\beta$", 2.0), ("alpha0", "$\\alpha_0$", 0.8), ("lambda_lr", "$\\eta_\\lambda$", 0.05),
     ("lambda_init", "$\\lambda_0$", 0.1), ("lambda_r", "$\\lambda_R$", 1.0), ("n_candidates", "$M$", 16),
     ("tau_safe", "$\\tau_{\\mathrm{safe}}$", 0.5), ("k_lookahead", "$k$", 10), ("is_clip", "$\\bar c$", 2.0),
     ("kappa_inadm", "$\\kappa$", 1.0)]
ENVS = [("cartpole", 0.05), ("arm", 0.01)]
res = {}
for env, B in ENVS:
    g = R[R.env == env]
    base = g[(g.run == "c2p_ppo") & (g.seed < 5)]
    rows = []
    for k, lab, d in P:
        for v in sorted({float(x.split("=")[1]) for x in g.run if x.startswith(f"c2p_ppo_{k}=")}):
            name = f"c2p_ppo_{k}={v:g}" if f"c2p_ppo_{k}={v:g}" in set(g.run) else f"c2p_ppo_{k}={v}"
            h = g[g.run == name]
            rows.append(dict(k=k, lab=lab, d=d, v=v, n=len(h), ret=iqm(h.eval_return), viol=iqm(h.eval_viol_rate),
                             tcr=iqm(h.train_cost_rate), feas=int((h.eval_viol_rate <= B).sum()),
                             pr=mannwhitneyu(h.eval_return, base.eval_return).pvalue,
                             pv=mannwhitneyu(h.eval_viol_rate, base.eval_viol_rate).pvalue,
                             pt=mannwhitneyu(h.train_cost_rate, base.train_cost_rate).pvalue))
    T = pd.DataFrame(rows)
    for c in ["pr", "pv", "pt"]:
        T[c + "h"] = holm(T[c].values)
    res[env] = (T, dict(ret=iqm(base.eval_return), viol=iqm(base.eval_viol_rate), tcr=iqm(base.train_cost_rate),
                        feas=int((base.eval_viol_rate <= B).sum())))
    T.to_csv(os.path.join(out, f"sensitivity_{env}.csv"), index=False)
def cell(x, p, d):
    s = f"{x:.{d}f}"
    return f"\\textbf{{{s}}}" if p < 0.05 else s
lines = []
bc, ba = res["cartpole"][1], res["arm"][1]
lines.append(f"\\multicolumn{{2}}{{@{{}}l}}{{default}} & {bc['ret']:.1f} & {bc['viol']:.3f} & {bc['tcr']:.3f} & {bc['feas']}/5 & "
             f"{ba['ret']:.1f} & {ba['viol']:.4f} & {ba['tcr']:.4f} & {ba['feas']}/5 \\\\ \\midrule")
Tc, Ta = res["cartpole"][0], res["arm"][0]
for i in range(len(Tc)):
    c, a = Tc.iloc[i], Ta.iloc[i]
    par = f"{c.lab} (default {c.d:g})" if i % 2 == 0 else ""
    lines.append(f"{par} & {c.v:g} & {cell(c.ret, c.prh, 1)} & {cell(c.viol, c.pvh, 3)} & {cell(c.tcr, c.pth, 3)} & {c.feas}/{c.n} & "
                 f"{cell(a.ret, a.prh, 1)} & {cell(a.viol, a.pvh, 4)} & {cell(a.tcr, a.pth, 4)} & {a.feas}/{a.n} \\\\")
open(os.path.join(out, "tab_sensitivity_rows.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
for env in res:
    T = res[env][0]; print(env); print(T[["k", "v", "ret", "viol", "tcr", "feas", "pr", "pv", "pt", "prh", "pvh", "pth"]].round(4).to_string())
