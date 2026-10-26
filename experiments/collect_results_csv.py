# C2P-PPO: collect all finished Kaggle results into a few SMALL plain CSV files.
# Paste into one cell of a notebook whose Inputs are the finished notebooks.
# Output (in /kaggle/working):
#   c2p_runs.csv     one row per run (final evaluation + training summary)
#   c2p_evals.csv    periodic evaluations of every run (learning curves)
#   c2p_curves.csv   training episodes averaged in 40 blocks per run
#   c2p_updates.csv  per-update diagnostics (lambda, alpha, IS weights, ...), max 100 rows per run
#   c2p_errors.csv   tail of the error log of every incomplete run
import csv, glob, io, json, os, re, zipfile
import pandas as pd

W = "/kaggle/working"
PAT = re.compile(r"results/([^/]+)/([^/]+)/seed(\d+)/([^/]+)$")
LOG = re.compile(r"results/_logs/([^/]+)\.log$")
files, logs = {}, {}

def add(name, read):
    m = PAT.search(name)
    if m and m.group(4) in ("summary.json", "evals.csv", "updates.csv", "episodes.csv"):
        files.setdefault((m.group(1), m.group(2), int(m.group(3))), {})[m.group(4)] = read()
    m = LOG.search(name)
    if m:
        logs[m.group(1)] = read()

for z in glob.glob("/kaggle/input/**/results_*.zip", recursive=True):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            add(n, lambda n=n: zf.read(n))
for f in glob.glob("/kaggle/input/**/results/**/*.*", recursive=True):
    add(f, lambda f=f: open(f, "rb").read())

runs, evals, curves, updates, errors = [], [], [], [], []
for (env, run, seed), d in sorted(files.items()):
    key = dict(env=env, run=run, seed=seed)
    if "summary.json" not in d:
        lg = [v for k, v in logs.items() if k.startswith(f"{env}_{run}_") and k.endswith(f"_s{seed}")]
        errors.append({**key, "log_tail": (lg[0].decode(errors="ignore")[-1500:] if lg else "")})
        continue
    s = json.loads(d["summary.json"]); fe = s.get("final_eval", {})
    runs.append({**key, **{k: v for k, v in fe.items()}, "train_cost_rate": s.get("cost_rate"),
                 "cum_violations": s.get("cum_violations"), "episodes": s.get("episodes"),
                 "wall_h": s.get("wall_time_s", 0) / 3600, "budget": s.get("budget")})
    if "evals.csv" in d:
        e = pd.read_csv(io.BytesIO(d["evals.csv"]))
        evals.append(e.assign(**key))
    if "episodes.csv" in d:
        e = pd.read_csv(io.BytesIO(d["episodes.csv"]))
        e["block"] = (e["step"] / e["step"].max() * 40).clip(upper=39.999).astype(int)
        cols = [c for c in ["step", "return", "viol_rate", "any_violation", "success", "length",
                            "intervention_rate", "fallback_rate"] if c in e]
        g = e.groupby("block")[cols].mean().reset_index()
        g["n_violations"] = e.groupby("block")["violations"].sum().values
        g["n_steps"] = e.groupby("block")["length"].sum().values
        curves.append(g.assign(**key))
    if "updates.csv" in d:
        u = pd.read_csv(io.BytesIO(d["updates.csv"]))
        if len(u) > 100:
            u = u.iloc[pd.Series(range(len(u))).iloc[::max(1, len(u) // 100)].values]
        updates.append(u.assign(**key))

pd.DataFrame(runs).to_csv(f"{W}/c2p_runs.csv", index=False)
for name, lst in [("evals", evals), ("curves", curves), ("updates", updates)]:
    (pd.concat(lst) if lst else pd.DataFrame()).round(5).to_csv(f"{W}/c2p_{name}.csv", index=False)
pd.DataFrame(errors).to_csv(f"{W}/c2p_errors.csv", index=False, quoting=csv.QUOTE_ALL)

R = pd.DataFrame(runs)
print(R.groupby(["env", "run"]).size().to_string() if len(R) else "no complete runs")
print(f"\ncomplete: {len(runs)}   incomplete: {len(errors)}")
for f in sorted(glob.glob(f"{W}/c2p_*.csv")):
    print(f"{os.path.basename(f):18s} {os.path.getsize(f) / 1e6:6.2f} MB")
