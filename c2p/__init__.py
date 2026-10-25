import os as _os

# Jupyter/Kaggle export MPLBACKEND=module://matplotlib_inline..., which is invalid
# in a plain subprocess and breaks the import of Safe-Control-Gym. Use a headless backend.
if _os.environ.get("MPLBACKEND", "").startswith("module://") or "MPLBACKEND" not in _os.environ:
    _os.environ["MPLBACKEND"] = "Agg"
