"""Resumable parallel job runner (one single-threaded process per core).

  python experiments/run_jobs.py --group main --env pointgoal --shard 0 --nshards 4 \
      --out /kaggle/working/results --workers 4 --hours 11.5

Jobs whose summary.json already exists are skipped, so a session that is
interrupted can simply be restarted with the same arguments. The job list is
split deterministically into --nshards disjoint shards so that several
sessions can run in parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.jobs import jobs  # noqa: E402


def _done(out, j):
    name = j["method"] + (f"_{j['tag']}" if j["tag"] else "")
    return os.path.exists(os.path.join(out, j["env"], name, f"seed{j['seed']}", "summary.json"))


def _estimate_h(j):
    """Duration estimates (hours) on a 4-core Kaggle CPU session, measured on the first batch
    (navigation: C2P-PPO 5.4 h, baselines 3.6-4.2 h; CartPole/arm: 0.6-1.5 h)."""
    nav = j["env"] in ("pointgoal", "cargoal")
    c2p = j["method"].startswith("c2p")
    if nav:
        return 5.8 if c2p else 4.6
    return 1.6 if c2p else 1.1


def _run(j, out, root, steps=None):
    cmd = [sys.executable, "-m", "c2p.train", "--env", j["env"], "--method", j["method"], "--seed", str(j["seed"]),
           "--out", out] + (["--steps", str(steps)] if steps else [])
    if j["tag"]:
        cmd += ["--tag", j["tag"]]
    if j["overrides"]:
        cmd += ["--set"] + [f"{k}={json.dumps(v)}" for k, v in j["overrides"].items()]
    env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    log_dir = os.path.join(out, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    tag = f"{j['env']}_{j['method']}_{j['tag']}_s{j['seed']}".replace("=", "-")
    with open(os.path.join(log_dir, tag + ".log"), "w") as f:
        f.write(p.stdout[-20000:] + "\n---STDERR---\n" + p.stderr[-20000:])
    err = "" if p.returncode == 0 else (p.stderr or p.stdout)[-3000:]
    return tag, p.returncode, time.time() - t0, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--hours", type=float, default=11.5, help="stop submitting new jobs after this many hours")
    ap.add_argument("--steps", type=int, default=None, help="override the step budget (testing only)")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    J = jobs(a.group, a.env)
    J = [j for i, j in enumerate(J) if i % a.nshards == a.shard]
    todo = [j for j in J if not _done(a.out, j)]
    # longest jobs first to reduce tail latency; within equal duration, lower seeds first
    todo.sort(key=lambda j: (-_estimate_h(j), j["seed"]))
    print(f"[runner] shard {a.shard}/{a.nshards}: {len(J)} jobs, {len(todo)} to run, {a.workers} workers", flush=True)
    t_start, deadline = time.time(), time.time() + a.hours * 3600
    n_ok = n_err = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs, it = {}, iter(todo)
        for _ in range(a.workers):
            j = next(it, None)
            if j is not None and time.time() + _estimate_h(j) * 3600 < deadline:
                futs[ex.submit(_run, j, a.out, root, a.steps)] = j
        while futs:
            for f in as_completed(list(futs)):
                futs.pop(f)
                tag, rc, dt, err = f.result()
                print(f"[runner] {'OK ' if rc == 0 else 'ERR'} {tag} ({dt / 60:.1f} min, "
                      f"elapsed {(time.time() - t_start) / 3600:.2f} h)", flush=True)
                if rc != 0:
                    n_err += 1
                    if n_err <= 2:
                        print("[runner] ---- error output ----\n" + err + "\n[runner] ----------------------", flush=True)
                    if n_err >= 4 and n_ok == 0:
                        print("[runner] the first jobs all failed; stopping (see error output above)", flush=True)
                        for g in futs:
                            g.cancel()
                        futs.clear()
                        break
                else:
                    n_ok += 1
                while True:
                    j = next(it, None)
                    if j is None:
                        break
                    if time.time() + _estimate_h(j) * 3600 < deadline:
                        futs[ex.submit(_run, j, a.out, root, a.steps)] = j
                        break
                    print(f"[runner] skip (not enough time left): {j['env']} {j['method']} s{j['seed']}", flush=True)
                break
    left = [j for j in J if not _done(a.out, j)]
    print(f"[runner] finished; {len(left)} jobs of this shard still missing", flush=True)


if __name__ == "__main__":
    main()
