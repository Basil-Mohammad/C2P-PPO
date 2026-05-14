# =============================================================
# lagrangian.py — Lagrangian Multiplier for CTL Constraints
# C2P-PPO Project | Safe RL with CTL Constraints
# =============================================================

import torch
import numpy as np
from typing import List, Dict


# ─────────────────────────────────────────
# Lagrangian Multiplier
# ─────────────────────────────────────────
class LagrangianMultiplier:
    """
    Projected stochastic subgradient update for λ_t.

    Update rule:
      λ_{t+1} = max(0, λ_t + η_λ · (E[CTLCost] - δ_safety))

    Convergence guaranteed under Robbins-Monro conditions:
      Σ η_λ = ∞  and  Σ η_λ² < ∞
    (satisfied by η_λ,t = η_0 / (t+1))

    Properties:
      - λ_t ≥ 0 always  (projection onto [0,∞))
      - λ_t → λ* as t → ∞
      - At optimum: either λ*=0 (constraint satisfied)
        or E[CTLCost] = δ_safety (constraint active)
    """

    def __init__(self, cfg: dict):
        s = cfg["safety"]
        self.lambda_t      = s["lambda_0"]
        self.lambda_lr     = s["lambda_lr"]
        self.delta_safety  = s["delta_safety"]

        # History for convergence analysis
        self.lambda_history:    List[float] = [self.lambda_t]
        self.violation_history: List[float] = []
        self.update_count = 0

        # Decaying learning rate option
        self.use_decay = False
        self.lambda_lr_0 = self.lambda_lr

    # ── Main Update ───────────────────────
    def update(self, mean_ctl_cost: float) -> float:
        """
        Update λ_t via projected subgradient ascent.

        Args:
            mean_ctl_cost: E_t[CTLCost(s_t, a_t)] from current batch

        Returns:
            new λ_t value
        """
        self.violation_history.append(mean_ctl_cost)
        self.update_count += 1

        # Optional: decaying learning rate η_0 / (t+1)
        if self.use_decay:
            lr = self.lambda_lr_0 / (self.update_count + 1)
        else:
            lr = self.lambda_lr

        # Subgradient step
        constraint_violation = mean_ctl_cost - self.delta_safety
        self.lambda_t = max(
            0.0,
            self.lambda_t + lr * constraint_violation
        )

        self.lambda_history.append(self.lambda_t)
        return self.lambda_t

    # ── Status ────────────────────────────
    def is_constraint_active(self) -> bool:
        """Returns True if constraint is currently being enforced."""
        return self.lambda_t > 1e-6

    def is_constraint_satisfied(self, tol: float = 0.01) -> bool:
        """
        Returns True if recent violations are within tolerance.
        Uses last 10 updates for assessment.
        """
        if len(self.violation_history) < 10:
            return False
        recent = self.violation_history[-10:]
        return float(np.mean(recent)) <= self.delta_safety + tol

    def get_stats(self) -> Dict:
        return {
            "lambda":            self.lambda_t,
            "update_count":      self.update_count,
            "constraint_active": self.is_constraint_active(),
            "satisfied":         self.is_constraint_satisfied(),
            "mean_violation":    float(np.mean(
                self.violation_history[-20:]
            )) if self.violation_history else 0.0,
        }

    # ── Save / Load ───────────────────────
    def save(self, path: str):
        torch.save({
            "lambda_t":          self.lambda_t,
            "lambda_history":    self.lambda_history,
            "violation_history": self.violation_history,
            "update_count":      self.update_count,
        }, path)
        print(f"[Lagrangian] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path)
        self.lambda_t          = ckpt["lambda_t"]
        self.lambda_history    = ckpt["lambda_history"]
        self.violation_history = ckpt["violation_history"]
        self.update_count      = ckpt["update_count"]
        print(f"[Lagrangian] Checkpoint loaded ← {path}")