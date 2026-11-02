#!/usr/bin/env bash
# Reproducible environment (Python 3.10) for all experiments.
# Usage: bash experiments/setup_env.sh /path/to/venv
set -euo pipefail
# Kaggle images may redirect uv to the system interpreter; disable that
unset UV_SYSTEM_PYTHON UV_PYTHON UV_PROJECT_ENVIRONMENT || true
VENV=${1:-./venv}
CODE=$(cd "$(dirname "$0")/.." && pwd)
pip -q install -U uv
# Force a uv-managed CPython 3.10 (MuJoCo 2.3.3 and Safety-Gymnasium 1.0.0 require Python <= 3.10/3.11)
uv python install 3.10
rm -rf "$VENV"
uv venv -q --python 3.10 --python-preference only-managed "$VENV"
export VIRTUAL_ENV="$VENV"
"$VENV/bin/python" -c "import sys; print('venv python', sys.version); assert sys.version_info[:2] == (3, 10), 'expected Python 3.10'"
# CPU build of PyTorch is sufficient (small MLPs, one thread per process)
uv pip install -q --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu || uv pip install -q --python "$VENV/bin/python" torch
uv pip install -q --python "$VENV/bin/python" -r "$CODE/requirements-lock.txt"
# Safety-Gymnasium 1.0.0 pins an old pygame that does not build; its runtime deps are in the lock file.
uv pip install -q --python "$VENV/bin/python" --no-deps "safety-gymnasium==1.0.0"
# Safe-Control-Gym, pinned commit
mkdir -p "$VENV/src"
if [ ! -d "$VENV/src/safe-control-gym" ]; then
  git clone -q https://github.com/utiasDSL/safe-control-gym.git "$VENV/src/safe-control-gym"
  git -C "$VENV/src/safe-control-gym" checkout -q 6b5391d014f36fdfa0f9d22d92c77387e5274308
fi
uv pip install -q --python "$VENV/bin/python" --no-deps -e "$VENV/src/safe-control-gym"
"$VENV/bin/python" - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
import torch, mujoco, gymnasium, safety_gymnasium
from safe_control_gym.utils.registration import make
print("OK torch", torch.__version__, "mujoco", mujoco.__version__, "gymnasium", gymnasium.__version__)
EOF
