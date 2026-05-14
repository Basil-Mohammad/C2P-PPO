# =============================================================
# ctl_safety.py — CTL-to-Action Safety Mapping
# C2P-PPO Project | Safe RL with CTL Constraints
# =============================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ─────────────────────────────────────────
# CTL Formula Types
# ─────────────────────────────────────────
class CTLFormula:
    """
    Represents a CTL safety formula of type AG(p).
    For CartPole: AG(|pole_angle| < θ_max ∧ |cart_pos| < x_max)
    """
    def __init__(self, formula_type: str = "AG_safety"):
        self.formula_type = formula_type

    def check(self, obs: np.ndarray) -> bool:
        """
        Check if CTL formula is satisfied at current state.
        Returns True if SAFE, False if VIOLATED.
        """
        if self.formula_type == "AG_safety":
            return self._cartpole_safety(obs)
        else:
            raise ValueError(f"Unknown formula: {self.formula_type}")

    def _cartpole_safety(self, obs: np.ndarray) -> bool:
        """
        CartPole safety constraint:
        AG(|pole_angle| < 0.2 ∧ |cart_position| < 2.4)
        """
        # obs = [cart_pos, cart_vel, pole_angle, pole_vel]
        cart_pos   = obs[0] if len(obs) > 0 else 0.0
        pole_angle = obs[2] if len(obs) > 2 else 0.0

        cart_safe  = abs(cart_pos)   < 2.4
        angle_safe = abs(pole_angle) < 0.2
        return cart_safe and angle_safe


# ─────────────────────────────────────────
# Safety Critic Network (Level 3 — Learned)
# ─────────────────────────────────────────
class SafetyCritic(nn.Module):
    """
    Neural network surrogate for Safe(s,a) ∈ [0,1].
    Trained via binary cross-entropy against CTL labels.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([obs, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────
# CTL Safety Mapper
# ─────────────────────────────────────────
class CTLSafetyMapper:
    """
    Computes Safe(s,a) ∈ [0,1] at three levels:
      Level 1 — Exact   (formula check, known model)
      Level 2 — Approx  (k-step lookahead)
      Level 3 — Learned (neural network surrogate)
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

        # CTL formula
        self.formula = CTLFormula(
            formula_type=cfg["ctl"]["formula"]
        )

        # Safety threshold τ
        self.tau_safe = cfg["safety"]["tau_safe"]

        # Safety critic (Level 3)
        self.critic = SafetyCritic(obs_dim, act_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4
        )
        self.bce_loss = nn.BCELoss()

        # Stats
        self.update_count = 0

    # ── Level 1: Exact ────────────────────
    def safe_exact(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
    ) -> float:
        """
        Returns 1.0 if next_obs satisfies CTL formula, 0.0 otherwise.
        Uses the transition (obs → next_obs) directly.
        """
        return 1.0 if self.formula.check(next_obs) else 0.0

    # ── Level 2: k-step lookahead ─────────
    def safe_lookahead(
        self,
        env,
        obs:    np.ndarray,
        action: int,
        k:      int = 5,
        K:      int = 10,
    ) -> float:
        """
        Monte Carlo estimate of safety over k steps.
        Runs K rollouts from (obs, action) and checks CTL.
        """
        safe_count = 0
        cfg_k = self.cfg["ctl"]["lookahead_k"]
        cfg_K = self.cfg["ctl"]["n_rollouts"]
        k = cfg_k
        K = cfg_K

        for _ in range(K):
            # Save env state (requires env.save/load or copy)
            try:
                state = env.save_state()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                curr_obs = env.get_obs()

                satisfied = True
                for step in range(k - 1):
                    if done:
                        break
                    if not self.formula.check(curr_obs):
                        satisfied = False
                        break
                    rand_action = env.action_space.sample()
                    _, _, terminated, truncated, _ = env.step(rand_action)
                    done = terminated or truncated
                    curr_obs = env.get_obs()

                if satisfied and self.formula.check(curr_obs):
                    safe_count += 1

                env.load_state(state)
            except Exception:
                # Fallback: use Level 1 if env does not support save/load
                return self.safe_exact(obs, obs)

        return safe_count / K

    # ── Level 3: Learned surrogate ────────
    def safe_nn(
        self,
        obs:    torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        """
        Returns Safe_NN(s,a;ψ) ∈ [0,1] for all actions.
        obs: (obs_dim,) tensor
        Returns: (act_dim,) tensor of safety probabilities
        """
        obs_batch = obs.unsqueeze(0).expand(self.act_dim, -1)

        # One-hot encode all actions
        one_hot = torch.eye(self.act_dim).to(self.device)

        with torch.no_grad():
            safety_probs = self.critic(obs_batch, one_hot)

        return safety_probs  # shape: (act_dim,)

    # ── CTL Cost ──────────────────────────
    def ctl_cost(
        self,
        obs:      np.ndarray,
        action:   int,
        next_obs: np.ndarray,
    ) -> float:
        """
        CTLCost(s,a) = 1 if action leads to CTL violation, 0 otherwise.
        Uses Level 1 (exact) against tau_safe threshold.
        """
        safety = self.safe_exact(obs, next_obs)
        return float(safety < self.tau_safe)

    # ── Update Safety Critic ──────────────
    def update_critic(
        self,
        obs_batch:      torch.Tensor,
        action_batch:   torch.Tensor,
        next_obs_batch: torch.Tensor,
    ) -> float:
        """
        Update safety critic using BCE loss against Level 1 labels.
        """
        # Generate labels from Level 1
        labels = []
        for i in range(next_obs_batch.shape[0]):
            nobs = next_obs_batch[i].cpu().numpy()
            label = self.safe_exact(
                obs_batch[i].cpu().numpy(), nobs
            )
            labels.append(label)

        labels_t = torch.tensor(
            labels, dtype=torch.float32
        ).to(self.device)

        # One-hot encode actions
        one_hot = torch.zeros(
            action_batch.shape[0], self.act_dim
        ).to(self.device)
        one_hot.scatter_(
            1, action_batch.unsqueeze(1).long(), 1.0
        )

        # Forward + loss
        preds = self.critic(obs_batch, one_hot)
        loss  = self.bce_loss(preds, labels_t)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        return loss.item()

    # ── Save / Load ───────────────────────
    def save(self, path: str):
        torch.save({
            "critic_state":    self.critic.state_dict(),
            "optimizer_state": self.critic_optimizer.state_dict(),
            "update_count":    self.update_count,
        }, path)
        print(f"[Safety] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(ckpt["critic_state"])
        self.critic_optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )
        self.update_count = ckpt["update_count"]
        print(f"[Safety] Checkpoint loaded ← {path}")