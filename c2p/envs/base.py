"""Common interface for constrained environments.

Every environment exposes
  reset(seed) -> obs
  step(a)     -> obs, reward, rho, terminated, truncated, info
where rho in R^m is the vector of robustness margins of the atomic
propositions of phi = AG(p_1 & ... & p_m), evaluated at the successor state.
Actions are expected in [-1, 1]^d and rescaled internally.
"""
from __future__ import annotations

import numpy as np

from c2p.ctl import SafetySpec


class SafeEnv:
    name: str = "base"
    obs_dim: int
    act_dim: int
    max_episode_steps: int
    spec: SafetySpec

    def reset(self, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def robustness(self) -> np.ndarray:
        """rho(s) at the current state (used by state-only filters)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
