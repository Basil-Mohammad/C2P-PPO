"""CTL safety specifications of the form AG(p_1 & ... & p_m).

Each atomic proposition p_j is evaluated on a state through a quantitative
(robustness) margin rho_j(s): p_j holds at s iff rho_j(s) >= 0. The Boolean
labelling L(s) used for model checking is recovered by thresholding rho_j.
This is the quantitative semantics adopted in robust temporal logics
(Donze & Maler, 2010; Nayak et al., 2025) restricted to atomic propositions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SafetySpec:
    """phi = AG(p_1 & ... & p_m) together with its per-step cost budget."""
    predicates: list[str]
    budget: float                        # delta_safety: bound on E[CTLCost] (per step)
    potential_clip: float = 1.0          # rho is clipped to [-c, c] for shaping
    formula: str = field(default="")

    def __post_init__(self):
        if not self.formula:
            self.formula = "AG(" + " & ".join(self.predicates) + ")"

    @property
    def m(self) -> int:
        return len(self.predicates)

    @staticmethod
    def labels(rho: np.ndarray) -> np.ndarray:
        """Boolean labelling L(s): 1 if p_j holds."""
        return (np.asarray(rho) >= 0.0).astype(np.float32)

    @staticmethod
    def cost_vector(rho: np.ndarray) -> np.ndarray:
        """Per-predicate immediate cost 1[s' does not satisfy p_j]."""
        return (np.asarray(rho) < 0.0).astype(np.float32)

    @staticmethod
    def ctl_cost(rho: np.ndarray) -> float:
        """Immediate conjunctive CTLCost(s,a,s') = 1[s' does not satisfy p_1 & ... & p_m]."""
        return float(np.any(np.asarray(rho) < 0.0))

    def potential(self, rho: np.ndarray) -> float:
        """Phi(s) = clip(min_j rho_j(s)): robustness of the conjunction."""
        c = self.potential_clip
        return float(np.clip(np.min(rho), -c, c))


def k_step_safety_labels(cost: np.ndarray, done: np.ndarray, k: int):
    """Level-2 Monte-Carlo labels from collected trajectories.

    y_t = 1 if no violation occurs at steps t+1..t+k (i.e. in the costs
    recorded for transitions t..t+k-1) within the same episode.
    Returns (labels, valid) where valid=0 marks samples whose k-step window is
    cut by the end of the buffer (their label is unknown and they are not used).
    Episodes that end before k steps without violation are labelled safe.
    """
    n = len(cost)
    y = np.ones(n, np.float32)
    valid = np.ones(n, np.float32)
    for t in range(n):
        for j in range(t, min(t + k, n)):
            if cost[j] > 0:
                y[t] = 0.0
                break
            if done[j]:
                break
        else:
            if t + k > n:
                valid[t] = 0.0
    return y, valid
