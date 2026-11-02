# C2P-PPO: collect the results of all finished Kaggle sessions into ONE light file.
# Paste this whole cell into a new Kaggle notebook. Add as Input the finished
# notebooks (Add Input -> Your Work -> Notebooks). No internet or GPU needed.
# Output: /kaggle/working/c2p_results_light.zip  (+ a coverage report printed below)
import glob, io, json, os, re, zipfile, collections

OUT = "/kaggle/working/c2p_results_light.zip"
KEEP = ("summary.json", "evals.csv", "updates.csv", "episodes.csv")
PAT = re.compile(r"results/([^/]+)/([^/]+)/seed(\d+)/([^/]+)$")

runs = {}          # (env, run, seed) -> {fname: bytes}
sources = []

def add(path_in_results, data):
    m = PAT.search(path_in_results)
    if not m or m.group(4) not in KEEP:
        return
    env, run, seed, fn = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    runs.setdefault((env, run, seed), {})[fn] = data

# 1) zipped outputs: results_*.zip
for z in sorted(glob.glob("/kaggle/input/**/results_*.zip", recursive=True)):
    sources.append(z)
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.endswith(KEEP):
                add(n, zf.read(n))
# 2) outputs that Kaggle already extracted: .../results/<env>/<run>/seed<k>/*
for f in glob.glob("/kaggle/input/**/results/*/*/seed*/*", recursive=True):
    if f.endswith(KEEP):
        sources.append(os.path.dirname(f))
        add(f, open(f, "rb").read())

complete = {k: v for k, v in runs.items() if "summary.json" in v}
partial = sorted(k for k in runs if k not in complete)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    rows = []
    for (env, run, seed), files in sorted(complete.items()):
        for fn, data in files.items():
            zf.writestr(f"results/{env}/{run}/seed{seed}/{fn}", data)
        s = json.loads(files["summary.json"])
        fe = s.get("final_eval", {})
        rows.append({"env": env, "run": run, "seed": seed, "return": fe.get("eval_return"),
                     "viol_rate": fe.get("eval_viol_rate"), "success": fe.get("eval_success"),
                     "train_cost_rate": s.get("cost_rate"), "wall_h": round(s.get("wall_time_s", 0) / 3600, 2)})
    buf = io.StringIO()
    keys = ["env", "run", "seed", "return", "viol_rate", "success", "train_cost_rate", "wall_h"]
    buf.write(",".join(keys) + "\n")
    for r in rows:
        buf.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    zf.writestr("summary_all_runs.csv", buf.getvalue())

# coverage report
cov = collections.Counter((e, r) for (e, r, s) in complete)
print(f"sources found: {len(set(sources))}")
print(f"complete runs: {len(complete)}   incomplete (no summary.json): {len(partial)}\n")
print(f"{'environment':<12}{'method':<34}{'seeds':>6}")
for (e, r), n in sorted(cov.items()):
    print(f"{e:<12}{r:<34}{n:>6}")
if partial:
    print("\nincomplete runs (will be re-run automatically if the session is restarted):")
    for k in partial[:50]:
        print("  ", k)
print(f"\nwritten {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
