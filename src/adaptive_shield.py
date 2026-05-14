# =============================================================
# adaptive_shield.py — Exploration-Aware Adaptive Shielding
# C2P-PPO Project | Safe RL with CTL Constraints
# =============================================================

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Tuple


# ─────────────────────────────────────────
# RND Networks (Random Network Distillation)
# ─────────────────────────────────────────
class RNDNetwork(nn.Module):
    """Fixed target network for RND exploration bonus."""
    def __init__(self, obs_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        # Fixed — never updated
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class RNDPredictor(nn.Module):
    """Trainable predictor network for RND."""
    def __init__(self, obs_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────
# Adaptive Shield
# ─────────────────────────────────────────
class AdaptiveShield:
    """
    Exploration-Aware Adaptive Shield for C2P-PPO.

    Computes the shielded behaviour policy:
      π̃(a|s) ∝ π_old(a|s) · [α_t · Safe(s,a) + (1-α_t) · Explore(s,a)]

    Where:
      - Safe(s,a)    from CTLSafetyMapper
      - Explore(s,a) from RND bonus
      - α_t          adapts based on running violation rate
    """

    def __init__(
        self,
        cfg:     dict,
        obs_dim: int,
        act_dim: int,
        device:  torch.device,
    ):
        self.cfg     = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device  = device

        # Shielding hyperparameters
        s = cfg["shielding"]
        self.alpha_t      = s["alpha_0"]
        self.beta         = s["beta"]
        self.window_size  = s["window_size"]
        self.c_threshold  = s["c_threshold"]

        # Safety mask threshold
        self.eps_mask = cfg["safety"]["eps_mask"]

        # Violation history buffer (sliding window)
        self.violation_buffer = deque(maxlen=self.window_size)
        # Initialise with zeros (no violations)
        for _ in range(self.window_size):
            self.violation_buffer.append(0.0)

        # RND networks
        self.rnd_target    = RNDNetwork(obs_dim).to(device)
        self.rnd_predictor = RNDPredictor(obs_dim).to(device)
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd_predictor.parameters(), lr=1e-4
        )

        # Running stats for RND normalisation
        self.rnd_mean = 0.0
        self.rnd_std  = 1.0
        self.rnd_count = 0

        # Stats
        self.alpha_history = []
        self.fallback_count = 0

    # ── α_t Update ────────────────────────
    def update_alpha(self, episode_violation_rate: float):
        """
        Update α_t based on running violation rate.
        α_t = σ(β · (C_violation - C_threshold))

        Properties:
          α_t → 1 when violations high  (prioritise safety)
          α_t → 0 when violations low   (allow exploration)
          α_t ∈ (0,1) always            (sigmoid guarantee)
        """
        self.violation_buffer.append(episode_violation_rate)
        c_violation = float(np.mean(self.violation_buffer))

        x = self.beta * (c_violation - self.c_threshold)
        self.alpha_t = float(torch.sigmoid(torch.tensor(x)).item())
        self.alpha_history.append(self.alpha_t)
        return self.alpha_t

    # ── RND Exploration Bonus ─────────────
    @torch.no_grad()
    def exploration_bonus(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute RND bonus for each action (state-based).
        Explore(s,a) = ||f(s) - f_target(s)||² (normalised)
        Note: RND is state-based; same bonus for all actions.
        """
        target_feat = self.rnd_target(obs.unsqueeze(0))
        pred_feat   = self.rnd_predictor(obs.unsqueeze(0))
        bonus = (pred_feat - target_feat).pow(2).sum().item()

        # Online normalisation
        self.rnd_count += 1
        self.rnd_mean  += (bonus - self.rnd_mean) / self.rnd_count
        self.rnd_std    = max(
            np.sqrt(
                (self.rnd_std ** 2 * (self.rnd_count - 1)
                 + (bonus - self.rnd_mean) ** 2)
                / self.rnd_count
            ),
            1e-8,
        )

        # Normalise to [0,1] approximately via sigmoid
        normalised = torch.sigmoid(
            torch.tensor((bonus - self.rnd_mean) / self.rnd_std)
        ).item()

        # Return same bonus for all actions (shape: act_dim,)
        return torch.full(
            (self.act_dim,), normalised,
            dtype=torch.float32, device=self.device
        )

    # ── Shielded Policy ───────────────────
    def compute_shielded_policy(
        self,
        obs:         torch.Tensor,
        pi_old_probs: torch.Tensor,
        safety_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Compute π̃(a|s) and advantage mask.

        Args:
            obs:          (obs_dim,) current observation
            pi_old_probs: (act_dim,) π_old action probabilities
            safety_probs: (act_dim,) Safe(s,a) from CTLSafetyMapper

        Returns:
            pi_tilde:     (act_dim,) shielded distribution
            fallback:     True if all actions were unsafe (Z=0)
        """
        # Exploration bonus
        explore = self.exploration_bonus(obs)

        # Combined weight
        weight = (self.alpha_t * safety_probs
                  + (1 - self.alpha_t) * explore)

        # Unnormalised shielded policy
        pi_tilde_unnorm = pi_old_probs * weight

        # Normalisation constant Z(s)
        Z = pi_tilde_unnorm.sum()

        fallback = False
        if Z.item() < 1e-10:
            # Edge case: all actions unsafe → fallback to π_old
            pi_tilde = pi_old_probs.clone()
            fallback = True
            self.fallback_count += 1
        else:
            pi_tilde = pi_tilde_unnorm / Z

        return pi_tilde, fallback

    # ── Advantage Mask ────────────────────
    def advantage_mask(
        self,
        pi_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns mask = 1 for actions with π̃(a|s) > eps_mask.
        Used to zero out advantages for shield-blocked actions.
        Distribution shift correction (Section 4d — Option A).
        """
        return (pi_tilde > self.eps_mask).float()

    # ── Update RND Predictor ──────────────
    def update_rnd(self, obs_batch: torch.Tensor) -> float:
        """Update RND predictor network."""
        with torch.no_grad():
            target_feat = self.rnd_target(obs_batch)
        pred_feat = self.rnd_predictor(obs_batch)

        loss = (pred_feat - target_feat).pow(2).mean()
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        return loss.item()

    # ── Save / Load ───────────────────────
    def save(self, path: str):
        torch.save({
            "rnd_predictor":   self.rnd_predictor.state_dict(),
            "rnd_optimizer":   self.rnd_optimizer.state_dict(),
            "alpha_t":         self.alpha_t,
            "alpha_history":   self.alpha_history,
            "violation_buffer": list(self.violation_buffer),
            "rnd_mean":        self.rnd_mean,
            "rnd_std":         self.rnd_std,
            "rnd_count":       self.rnd_count,
            "fallback_count":  self.fallback_count,
        }, path)
        print(f"[Shield] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.rnd_predictor.load_state_dict(ckpt["rnd_predictor"])
        self.rnd_optimizer.load_state_dict(ckpt["rnd_optimizer"])
        self.alpha_t         = ckpt["alpha_t"]
        self.alpha_history   = ckpt["alpha_history"]
        self.violation_buffer = deque(
            ckpt["violation_buffer"], maxlen=self.window_size
        )
        self.rnd_mean        = ckpt["rnd_mean"]
        self.rnd_std         = ckpt["rnd_std"]
        self.rnd_count       = ckpt["rnd_count"]
        self.fallback_count  = ckpt["fallback_count"]
        print(f"[Shield] Checkpoint loaded ← {path}")