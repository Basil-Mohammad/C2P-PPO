"""Merge collected CSV batches; later batches override earlier ones; drop superseded runs."""
import sys, glob, os
import pandas as pd
DROP = {"safety_layer"}          # superseded by safety_layer_v2
def merge(dirs, name):
    parts = []
    for i, d in enumerate(dirs):
        f = os.path.join(d, f"c2p_{name}.csv")
        if os.path.exists(f) and os.path.getsize(f) > 2:
            x = pd.read_csv(f); x["_batch"] = i; parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    df = df[~df.run.isin(DROP)]
    last = df.groupby(["env", "run", "seed"])["_batch"].transform("max")
    return df[df._batch == last].drop(columns="_batch")
if __name__ == "__main__":
    out, dirs = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    for n in ["runs", "evals", "curves", "updates"]:
        merge(dirs, n).to_csv(os.path.join(out, f"c2p_{n}.csv"), index=False)
    print(pd.read_csv(os.path.join(out, "c2p_runs.csv")).groupby(["env", "run"]).size().unstack(0).fillna(0).astype(int))
