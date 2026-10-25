"""Generate one Kaggle notebook per experiment shard.

Each notebook: installs the pinned environment (Python 3.10 via uv), obtains
the code (attached Kaggle dataset 'c2p-code' if present, otherwise the GitHub
branch), runs its shard with 4 workers, and zips the results into
/kaggle/working so they appear in the notebook's Output tab.
"""
from __future__ import annotations

import json
import os

REPO = "https://github.com/Basil-Mohammad/C2P-PPO.git"
BRANCH = "revision-1"

# (name, group, env, nshards, hours)
PLAN = [
    ("main_pointgoal", "main", "pointgoal", 5, 10.5),
    ("main_cargoal", "main", "cargoal", 5, 10.5),
    ("main_cartpole", "main", "cartpole", 2, 11.0),
    ("main_arm", "main", "arm", 2, 11.0),
    ("ablation_cartpole", "ablation", "cartpole", 2, 11.0),
    ("ablation_arm", "ablation", "arm", 2, 11.0),
    ("ablation_pointgoal", "ablation", "pointgoal", 3, 10.5),
    ("sensitivity_cartpole", "sensitivity", "cartpole", 2, 11.0),
    ("sensitivity_arm", "sensitivity", "arm", 2, 11.0),
    ("multi_arm", "multi", None, 1, 11.0),
]


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def notebook(name, group, env, shard, nshards, hours):
    env_arg = f"--env {env} " if env else ""
    tag = f"{name}_shard{shard}of{nshards}"
    cells = [
        cell(f"# C2P-PPO revision experiments: `{tag}`\n\n"
             "Settings: **Accelerator = None (CPU)**, **Internet = On**, **Persistence = Files**.\n"
             "Run with *Save Version -> Save & Run All (Commit)* so that the session runs in the background "
             "and the results zip appears in the Output tab. If the session stops early, simply re-run it: "
             "finished runs are skipped.", "markdown"),
        cell("import os, glob, subprocess\n"
             "W = '/kaggle/working'\n"
             "CODE = W + '/code'\n"
             "# Locate the attached dataset wherever Kaggle mounted it (layout differs between Kaggle versions)\n"
             "hits = glob.glob('/kaggle/input/**/c2p/train.py', recursive=True)\n"
             "print('found:', hits)\n"
             "if hits:\n"
             "    src = os.path.dirname(os.path.dirname(hits[0]))\n"
             "    subprocess.run(f'rm -rf {CODE} && cp -r {src} {CODE}', shell=True, check=True)\n"
             "else:\n"
             "    # the dataset may hold the zip itself\n"
             "    zips = glob.glob('/kaggle/input/**/c2p-code*.zip', recursive=True)\n"
             "    assert zips, 'Dataset c2p-code not attached: use Add Input -> Your Datasets -> c2p-code'\n"
             "    subprocess.run(f'rm -rf {CODE} && mkdir -p {CODE} && cd {CODE} && unzip -q {zips[0]}', shell=True, check=True)\n"
             "print(sorted(os.listdir(CODE)))"),
        cell("!bash /kaggle/working/code/experiments/setup_env.sh /kaggle/working/venv 2>&1 | tail -5"),
        cell("!nproc && free -g | head -2"),
        cell(f"!cd /kaggle/working/code && /kaggle/working/venv/bin/python -m pytest -q tests 2>&1 | tail -2"),
        cell("# Resume: results of earlier versions of this notebook attached as input are restored first\n"
             "import glob, subprocess\n"
             f"for z in glob.glob('/kaggle/input/**/results_{tag}.zip', recursive=True):\n"
             "    subprocess.run(f'cd /kaggle/working && unzip -qo {z}', shell=True)\n"
             "    print('restored', z)"),
        cell(f"!cd /kaggle/working/code && timeout {int(hours * 3600 + 1800)} /kaggle/working/venv/bin/python "
             f"experiments/run_jobs.py --group {group} {env_arg}--shard {shard} --nshards {nshards} --workers 4 "
             f"--hours {hours} --out /kaggle/working/results; echo exit=$?"),
        cell(f"!cd /kaggle/working && zip -qr results_{tag}.zip results -x 'results/*/*/*/final_policy.pt' && "
             f"zip -qr policies_{tag}.zip results -i '*final_policy.pt' && ls -la *.zip"),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main(outdir="kaggle_notebooks"):
    os.makedirs(outdir, exist_ok=True)
    names = []
    for name, group, env, nshards, hours in PLAN:
        for k in range(nshards):
            nb = notebook(name, group, env, k, nshards, hours)
            fn = os.path.join(outdir, f"{name}_shard{k}of{nshards}.ipynb")
            json.dump(nb, open(fn, "w"), indent=1)
            names.append(fn)
    print("\n".join(names))
    print(f"{len(names)} notebooks")


if __name__ == "__main__":
    main()
