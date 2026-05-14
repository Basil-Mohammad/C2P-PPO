# =============================================================
# ablations.py — True Ablation Variants of C2P-PPO
# C2P-PPO Project | Safe RL with CTL Constraints
#
# Each variant disables exactly ONE component of C2P-PPO:
#   1. C2P-PPO-NoLag:        λ_t = 0 (no Lagrangian)
#   2. C2P-PPO-NoShield:     α_t = 0 (no shielding)
#   3. C2P-PPO-FixedAlpha:   α_t = fixed (no adaptation)
#   4. C2P-PPO-NoRND:        Explore = 0 (no RND bonus)
# =============================================================

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Tuple
from collections import deque

from src.ppo_base        import ActorCritic, RolloutBuffer
from src.c2p_ppo         import (
    C2PRolloutBuffer,
    ContinuousCTLSafetyMapper,
    ContinuousAdaptiveShield,
)
from src.lagrangian      import LagrangianMultiplier


# ─────────────────────────────────────────
# Base class shared by all ablations
# ─────────────────────────────────────────
class AblationBase:
    """Shared infrastructure for all C2P-PPO ablation variants."""

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        self.cfg     = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device  = torch.device(
            cfg["experiment"]["device"]
        )

        p = cfg["ppo"]
        self.lr            = p["lr"]
        self.gamma         = p["gamma"]
        self.gae_lambda    = p["gae_lambda"]
        self.clip_epsilon  = p["clip_epsilon"]
        self.epochs        = p["epochs"]
        self.batch_size    = p["batch_size"]
        self.value_coef    = p["value_coef"]
        self.entropy_coef  = p["entropy_coef"]
        self.max_grad_norm = p["max_grad_norm"]

        s = cfg["safety"]
        self.lambda_r = s["lambda_r"]
        self.eta      = s["eta"]

        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )

        self.safety_mapper = ContinuousCTLSafetyMapper(
            cfg, obs_dim
        )
        self.buffer      = C2PRolloutBuffer()
        self.iteration   = 0
        self.train_stats: Dict = {}

    def shape_reward(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        next_obs: np.ndarray,
        r_env:    float,
    ) -> Tuple[float, float]:
        return self.safety_mapper.shape_reward(
            obs, next_obs, r_env, self.lambda_r
        )

    def _ppo_update(
        self,
        last_obs:    np.ndarray,
        lambda_val:  float = 0.0,
    ) -> Dict:
        """Core PPO update shared across ablations."""
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae_c2p(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        (obs_t, actions_t, old_lp_t,
         next_obs_t, ctl_cost_t) = \
            self.buffer.get_c2p_tensors()

        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)
        ctl_cost_t = ctl_cost_t.to(self.device)

        total_loss = total_pg = total_vf = 0.0
        total_ent  = total_safety = 0.0
        n_updates  = 0
        dataset_size = obs_t.shape[0]

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(
                0, dataset_size, self.batch_size
            ):
                idx = indices[start: start + self.batch_size]

                log_probs, entropy, values = \
                    self.ac.evaluate_actions(
                        obs_t[idx], actions_t[idx]
                    )

                ratio = torch.exp(log_probs - old_lp_t[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * advantages[idx]
                pg_loss     = -torch.min(surr1, surr2).mean()
                vf_loss     = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss    = -entropy.mean()
                safety_loss = torch.tensor(lambda_val) \
                    * ctl_cost_t[idx].mean()

                loss = (pg_loss
                        + self.value_coef   * vf_loss
                        + self.entropy_coef * ent_loss
                        + safety_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss   += loss.item()
                total_pg     += pg_loss.item()
                total_vf     += vf_loss.item()
                total_ent    += entropy.mean().item()
                total_safety += safety_loss.item()
                n_updates    += 1

        mean_ctl = ctl_cost_t.mean().item()
        self.buffer.clear()
        self.iteration += 1

        return {
            "loss":        total_loss   / n_updates,
            "pg_loss":     total_pg     / n_updates,
            "vf_loss":     total_vf     / n_updates,
            "entropy":     total_ent    / n_updates,
            "safety_loss": total_safety / n_updates,
            "ctl_cost":    mean_ctl,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "iteration":   self.iteration,
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "train_stats": self.train_stats,
        }, path)
        print(f"[{self.__class__.__name__}] "
              f"Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.iteration   = ckpt["iteration"]
        self.train_stats = ckpt["train_stats"]


# ─────────────────────────────────────────
# Ablation 1: No Lagrangian (λ_t = 0)
# ─────────────────────────────────────────
class C2PPPO_NoLag(AblationBase):
    """
    C2P-PPO without Lagrangian multiplier.
    Shield is active but no dual-variable constraint enforcement.
    Tests: does the Lagrangian contribute beyond the shield?
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        # Shield still active
        self.shield = ContinuousAdaptiveShield(cfg)

    @torch.no_grad()
    def select_action(
        self,
        obs:      np.ndarray,
        prev_obs: np.ndarray = None,
    ) -> Tuple:
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()

        safety_margin = self.safety_mapper.safety_margin(obs)
        shielded, adv_mask = self.shield.scale_action(
            action_np, obs, safety_margin
        )

        return (
            shielded,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
            log_prob.item(),
            adv_mask,
        )

    def update(self, last_obs: np.ndarray) -> Dict:
        # λ_t = 0: no Lagrangian penalty
        stats = self._ppo_update(last_obs, lambda_val=0.0)
        stats["lambda_t"] = 0.0
        stats["alpha_t"]  = self.shield.alpha_t
        self.train_stats  = stats
        return stats


# ─────────────────────────────────────────
# Ablation 2: No Shield (α_t = 0)
# ─────────────────────────────────────────
class C2PPPO_NoShield(AblationBase):
    """
    C2P-PPO without behaviour-policy shield.
    Lagrangian is active but no action modification during collection.
    Tests: does the shield contribute beyond the Lagrangian?
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        self.lagrangian = LagrangianMultiplier(cfg)

    @torch.no_grad()
    def select_action(
        self,
        obs:      np.ndarray,
        prev_obs: np.ndarray = None,
    ) -> Tuple:
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()

        # NO shield: action used as-is
        action_clipped = np.clip(action_np, -10.0, 10.0)

        return (
            action_clipped,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
            log_prob.item(),
            1.0,  # no masking
        )

    def update(self, last_obs: np.ndarray) -> Dict:
        lambda_val = self.lagrangian.lambda_t
        stats = self._ppo_update(last_obs, lambda_val)
        mean_ctl = stats["ctl_cost"]
        self.lagrangian.update(mean_ctl)
        stats["lambda_t"] = self.lagrangian.lambda_t
        stats["alpha_t"]  = 0.0
        self.train_stats  = stats
        return stats


# ─────────────────────────────────────────
# Ablation 3: Fixed Alpha (no adaptation)
# ─────────────────────────────────────────
class C2PPPO_FixedAlpha(AblationBase):
    """
    C2P-PPO with α_t fixed at α_0 throughout training.
    Both shield and Lagrangian active, but no online α adaptation.
    Tests: does adaptive α_t contribute?
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        self.lagrangian = LagrangianMultiplier(cfg)
        # Fixed alpha — never updated
        self.alpha_fixed = cfg["shielding"]["alpha_0"]

    def _safety_margin(self, obs: np.ndarray) -> float:
        return self.safety_mapper.safety_margin(obs)

    def _shield_action(
        self,
        action: np.ndarray,
        obs:    np.ndarray,
    ) -> np.ndarray:
        """Shield with fixed α — same logic as ContinuousAdaptiveShield
        but α never changes."""
        safety_margin = self._safety_margin(obs)
        if safety_margin > 0.5:
            return np.clip(action, -10.0, 10.0)

        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
        corrective = np.array([
            -np.sign(pole_angle) * 5.0
            - np.sign(cart_pos)  * 2.0
        ])
        blend    = self.alpha_fixed * (1.0 - safety_margin)
        shielded = (1.0 - blend) * action + blend * corrective
        return np.clip(shielded, -10.0, 10.0)

    @torch.no_grad()
    def select_action(
        self,
        obs:      np.ndarray,
        prev_obs: np.ndarray = None,
    ) -> Tuple:
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()
        shielded  = self._shield_action(action_np, obs)

        return (
            shielded,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
            log_prob.item(),
            1.0,
        )

    def update(self, last_obs: np.ndarray) -> Dict:
        lambda_val = self.lagrangian.lambda_t
        stats = self._ppo_update(last_obs, lambda_val)
        self.lagrangian.update(stats["ctl_cost"])
        stats["lambda_t"] = self.lagrangian.lambda_t
        stats["alpha_t"]  = self.alpha_fixed  # constant
        self.train_stats  = stats
        return stats


# ─────────────────────────────────────────
# Ablation 4: No RND (Explore = 0)
# ─────────────────────────────────────────
class C2PPPO_NoRND(AblationBase):
    """
    C2P-PPO without RND exploration bonus.
    Shield uses Safety only: π̃ ∝ π_old · Safe(s,a)
    Tests: does the RND exploration bonus contribute?
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        self.lagrangian = LagrangianMultiplier(cfg)

        s = cfg["shielding"]
        self.alpha_0     = s["alpha_0"]
        self.beta        = s["beta"]
        self.window_size = s["window_size"]
        self.c_threshold = s["c_threshold"]
        self.alpha_t     = self.alpha_0

        from collections import deque
        self.violation_buffer = deque(maxlen=self.window_size)
        for _ in range(self.window_size):
            self.violation_buffer.append(0.0)

    def update_alpha(self, viol_rate: float):
        self.violation_buffer.append(viol_rate)
        c_viol = float(np.mean(self.violation_buffer))
        x = self.beta * (c_viol - self.c_threshold)
        self.alpha_t = float(
            torch.sigmoid(torch.tensor(x)).item()
        )

    def _shield_action(
        self,
        action:        np.ndarray,
        obs:           np.ndarray,
        safety_margin: float,
    ) -> np.ndarray:
        """Shield without RND: pure safety-based scaling."""
        if safety_margin > 0.5:
            return np.clip(action, -10.0, 10.0)

        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
        corrective = np.array([
            -np.sign(pole_angle) * 5.0
            - np.sign(cart_pos)  * 2.0
        ])
        # No RND: blend = alpha_t * (1 - margin), no explore term
        blend    = self.alpha_t * (1.0 - safety_margin)
        shielded = (1.0 - blend) * action + blend * corrective
        return np.clip(shielded, -10.0, 10.0)

    @torch.no_grad()
    def select_action(
        self,
        obs:      np.ndarray,
        prev_obs: np.ndarray = None,
    ) -> Tuple:
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()

        safety_margin = self.safety_mapper.safety_margin(obs)
        shielded = self._shield_action(
            action_np, obs, safety_margin
        )

        return (
            shielded,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
            log_prob.item(),
            1.0,
        )

    def update(self, last_obs: np.ndarray) -> Dict:
        lambda_val = self.lagrangian.lambda_t
        stats = self._ppo_update(last_obs, lambda_val)
        self.lagrangian.update(stats["ctl_cost"])
        stats["lambda_t"] = self.lagrangian.lambda_t
        stats["alpha_t"]  = self.alpha_t
        self.train_stats  = stats
        return stats