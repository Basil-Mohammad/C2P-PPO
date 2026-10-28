"""One-line Kaggle entry point.

In a Kaggle notebook with the dataset 'c2p-code' attached (CPU, Internet on):

    !python $(find /kaggle/input -name kaggle_run.py | head -1) main_cartpole_shard0of2

Steps: copy code -> install pinned Python 3.10 environment -> unit tests ->
restore results of earlier runs of the same session (if attached as input) ->
run the shard with 4 workers under a hard time limit -> zip the results into
/kaggle/working (Output tab). Uses only the standard library, so it runs with
Kaggle's own Python.

    python kaggle_run.py --list      # all session names
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_kaggle_notebooks import PLAN  # noqa: E402

W = "/kaggle/working"
os.environ["MPLBACKEND"] = "Agg"   # the Jupyter inline backend is invalid in subprocesses
CODE = os.path.join(W, "code")
VENV = os.path.join(W, "venv")


def sessions():
    out = {}
    for name, group, env, nshards, hours in PLAN:
        for k in range(nshards):
            out[f"{name}_shard{k}of{nshards}"] = (group, env, k, nshards, hours)
    return out


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        sys.exit(f"[kaggle_run] command failed ({r.returncode}): {cmd}")
    return r.returncode


def main():
    S = sessions()
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "--list"):
        print("Sessions:\n  " + "\n  ".join(S))
        return
    tag = sys.argv[1]
    if tag not in S:
        sys.exit(f"Unknown session '{tag}'. Use --list.")
    group, env, shard, nshards, hours = S[tag]

    # 1. code
    src = os.path.dirname(HERE)
    if os.path.abspath(src) != CODE:
        shutil.rmtree(CODE, ignore_errors=True)
        shutil.copytree(src, CODE)
    print(f"[kaggle_run] session {tag}: group={group} env={env} shard={shard}/{nshards}", flush=True)

    # 2. environment (pinned Python 3.10)
    sh(f"bash {CODE}/experiments/setup_env.sh {VENV}")
    py = os.path.join(VENV, "bin", "python")

    # 3. tests
    sh(f"cd {CODE} && {py} -m pytest -q tests 2>&1 | tail -3")
    sh("nproc")

    # 3b. smoke test: a very short training run in this session's environment
    smoke_env = env or "arm"
    smoke_method = "safety_layer_v2" if group in ("slayer_fix", "fill_pointgoal") else "c2p_ppo"
    rc = sh(f"cd {CODE} && PYTHONWARNINGS=ignore {py} -m c2p.train --env {smoke_env} --method {smoke_method} --seed 0 "
            f"--out /tmp/smoke --steps 3000 --set steps_per_update=1000 shield_warmup_updates=1 2>&1 | tail -25", check=False)
    if not os.path.exists(f"/tmp/smoke/{smoke_env}/{smoke_method}/seed0/summary.json"):
        sys.exit("[kaggle_run] smoke test failed; see the error output above")
    print("[kaggle_run] smoke test passed", flush=True)

    # 4. restore earlier results of this session, if attached as input
    for z in glob.glob(f"/kaggle/input/**/results_{tag}.zip", recursive=True):
        print("[kaggle_run] restoring", z)
        sh(f"cd {W} && unzip -qo '{z}'", check=False)

    # 5. run (hard limit keeps the session below Kaggle's 12 h so that step 6 always runs)
    env_arg = f"--env {env} " if env else ""
    sh(f"cd {CODE} && timeout {int(hours * 3600 + 1800)} {py} experiments/run_jobs.py --group {group} {env_arg}"
       f"--shard {shard} --nshards {nshards} --workers 4 --hours {hours} --out {W}/results", check=False)

    # 6. package
    sh(f"cd {W} && zip -qr results_{tag}.zip results -x 'results/*/*/*/final_policy.pt'", check=False)
    sh(f"cd {W} && zip -qr policies_{tag}.zip results -i '*final_policy.pt'", check=False)
    sh(f"ls -la {W}/*.zip", check=False)
    print(f"\n[kaggle_run] done. Download results_{tag}.zip from the Output tab.")


if __name__ == "__main__":
    main()
