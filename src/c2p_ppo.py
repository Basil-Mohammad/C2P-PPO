# =============================================================
# c2p_ppo.py — C2P-PPO: Full Algorithm (Continuous Actions)
# CTL-Constrained PPO with Adaptive Shielding
# =============================================================

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Tuple

from src.ppo_base        import ActorCritic, RolloutBuffer
from src.ctl_safety      import CTLSafetyMapper
from src.adaptive_shield import AdaptiveShield
from src.lagrangian      import LagrangianMultiplier


# ─────────────────────────────────────────
# C2P-PPO Rollout Buffer (extended)
# ─────────────────────────────────────────
class C2PRolloutBuffer(RolloutBuffer):
    def __init__(self):
        super().__init__()
        self.next_obs:  list = []
        self.ctl_costs: list = []
        self.pi_old:    list = []
        self.adv_masks: list = []

    def clear(self):
        super().clear()
        self.next_obs  = []
        self.ctl_costs = []
        self.pi_old    = []
        self.adv_masks = []

    def add_c2p(
        self,
        obs, action, log_prob, reward, value, done,
        next_obs, ctl_cost, pi_old_prob, adv_mask,
    ):
        self.add(obs, action, log_prob, reward, value, done)
        self.next_obs.append(next_obs)
        self.ctl_costs.append(ctl_cost)
        self.pi_old.append(pi_old_prob)
        self.adv_masks.append(adv_mask)

    def compute_gae_c2p(
        self,
        last_value: torch.Tensor,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages, returns = self.compute_gae(
            last_value, gamma, gae_lambda
        )
        masks      = torch.tensor(
            self.adv_masks, dtype=torch.float32
        )
        advantages = advantages * masks
        return advantages, returns

    def get_c2p_tensors(self):
        obs_t, actions_t, old_lp_t = self.get_tensors()
        next_obs_t = torch.stack(self.next_obs)
        ctl_cost_t = torch.tensor(
            self.ctl_costs, dtype=torch.float32
        )
        return obs_t, actions_t, old_lp_t, next_obs_t, ctl_cost_t


# ─────────────────────────────────────────
# Safety Mapper (Continuous Version)
# ─────────────────────────────────────────
class ContinuousCTLSafetyMapper:
    """
    CTL Safety Mapper for continuous action spaces.
    Safe(s) is state-based for continuous actions.
    """
    def __init__(self, cfg: dict, obs_dim: int):
        self.cfg      = cfg
        self.obs_dim  = obs_dim
        self.tau_safe = cfg["safety"]["tau_safe"]
        self.eta      = cfg["safety"]["eta"]

    def is_safe_state(self, obs: np.ndarray) -> bool:
        """
        CTL formula: AG(|pole_angle| < 0.2 ∧ |cart_pos| < 2.4)
        obs = [cart_pos, cart_vel, pole_angle, pole_vel]
        """
        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
        return abs(cart_pos) < 2.4 and abs(pole_angle) < 0.2

    def safety_margin(self, obs: np.ndarray) -> float:
        """
        Returns safety margin ∈ [0,1].
        1.0 = perfectly safe, 0.0 = at boundary.
        """
        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0

        cart_margin  = max(0.0, 1.0 - abs(cart_pos)  / 2.4)
        angle_margin = max(0.0, 1.0 - abs(pole_angle) / 0.2)
        return min(cart_margin, angle_margin)

    def ctl_cost(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
    ) -> float:
        """CTLCost = 1 if next state violates CTL formula."""
        return 0.0 if self.is_safe_state(next_obs) else 1.0

    def shape_reward(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        r_env:    float,
        lambda_r: float,
    ) -> Tuple[float, float]:
        """R_total = R_env + λ_R · R_CTL"""
        cost  = self.ctl_cost(obs, next_obs)
        r_ctl = -self.eta if cost > 0 else 0.0
        return r_env + lambda_r * r_ctl, cost


# ─────────────────────────────────────────
# Continuous Adaptive Shield
# ─────────────────────────────────────────
class ContinuousAdaptiveShield:
    """
    Adaptive shield for continuous action spaces.
    Applies safety-aware noise scaling to actions.
    α_t controls how much safety margin affects exploration.
    """
    def __init__(self, cfg: dict):
        s = cfg["shielding"]
        self.alpha_t     = s["alpha_0"]
        self.beta        = s["beta"]
        self.window_size = s["window_size"]
        self.c_threshold = s["c_threshold"]

        from collections import deque
        self.violation_buffer = deque(maxlen=self.window_size)
        for _ in range(self.window_size):
            self.violation_buffer.append(0.0)

        self.alpha_history  = []
        self.fallback_count = 0

    def update_alpha(self, episode_violation_rate: float):
        """Update α_t based on running violation rate."""
        self.violation_buffer.append(episode_violation_rate)
        c_violation = float(np.mean(self.violation_buffer))
        x = self.beta * (c_violation - self.c_threshold)
        self.alpha_t = float(
            torch.sigmoid(torch.tensor(x)).item()
        )
        self.alpha_history.append(self.alpha_t)
        return self.alpha_t

    def scale_action(
        self,
        action:        np.ndarray,
        obs:           np.ndarray,
        safety_margin: float,
        act_low:       float = -10.0,
        act_high:      float =  10.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Safety-aware action correction.
        When unsafe: blend action with corrective action.
        When safe:   use action as-is.
        """
        # If safe enough, no correction needed
        if safety_margin > 0.5:
            return np.clip(action, act_low, act_high), 1.0
        # Corrective action: push toward safe region
        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
        # Counteract pole angle and cart position
        corrective = np.array([
            -np.sign(pole_angle) * 5.0
            - np.sign(cart_pos)  * 2.0
            ])
        # Blend factor: more correction when more unsafe
        blend = self.alpha_t * (1.0 - safety_margin)
        shielded = (1.0 - blend) * action + blend * corrective
        shielded = np.clip(shielded, act_low, act_high)
        return shielded, 1.0

# ─────────────────────────────────────────
# C2P-PPO Agent (Continuous)
# ─────────────────────────────────────────
class C2PPPOAgent:
    """
    CTL-Constrained PPO with Adaptive Shielding.
    Continuous action space version for Safe-Control-Gym.
    """

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

        # ── Networks ──────────────────────
        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )

        # ── Safety Components ─────────────
        self.safety_mapper = ContinuousCTLSafetyMapper(
            cfg, obs_dim
        )
        self.shield     = ContinuousAdaptiveShield(cfg)
        self.lagrangian = LagrangianMultiplier(cfg)
        self.buffer     = C2PRolloutBuffer()

        self.iteration   = 0
        self.train_stats: Dict = {}

    # ── Select Action ─────────────────────
    @torch.no_grad()
    def select_action(
        self,
        obs:      np.ndarray,
        prev_obs: np.ndarray = None,
    ) -> Tuple:
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)

        # Sample from π_old
        action, log_prob, entropy, value = \
            self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()

        # Store π_old log_prob
        pi_old_lp = log_prob.item()

        # Safety margin for shield
        safety_margin = self.safety_mapper.safety_margin(obs)

        # Apply continuous shield
        shielded_action, adv_mask = self.shield.scale_action(
            action_np, obs, safety_margin
        )

        return (
            shielded_action,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
            pi_old_lp,
            adv_mask,
        )

    # ── Shape Reward ──────────────────────
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

    # ── C2P-PPO Update ────────────────────
    def update(self, last_obs: np.ndarray) -> Dict:
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
         next_obs_t, ctl_cost_t) = self.buffer.get_c2p_tensors()

        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)
        ctl_cost_t = ctl_cost_t.to(self.device)

        lambda_t = self.lagrangian.lambda_t

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

                # Standard PPO ratio
                ratio = torch.exp(log_probs - old_lp_t[idx])

                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * advantages[idx]
                pg_loss = -torch.min(surr1, surr2).mean()

                vf_loss  = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss     = -entropy.mean()
                safety_loss  = lambda_t * ctl_cost_t[idx].mean()

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

        mean_ctl_cost = ctl_cost_t.mean().item()
        self.lagrangian.update(mean_ctl_cost)
        self.buffer.clear()
        self.iteration += 1

        self.train_stats = {
            "loss":        total_loss   / n_updates,
            "pg_loss":     total_pg     / n_updates,
            "vf_loss":     total_vf     / n_updates,
            "entropy":     total_ent    / n_updates,
            "safety_loss": total_safety / n_updates,
            "lambda_t":    self.lagrangian.lambda_t,
            "alpha_t":     self.shield.alpha_t,
            "ctl_cost":    mean_ctl_cost,
        }
        return self.train_stats

    # ── Checkpoint ────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "iteration":   self.iteration,
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "lambda_t":    self.lagrangian.lambda_t,
            "alpha_t":     self.shield.alpha_t,
            "train_stats": self.train_stats,
        }, path)
        print(f"[C2P-PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.lagrangian.lambda_t = ckpt["lambda_t"]
        self.shield.alpha_t      = ckpt["alpha_t"]
        self.iteration           = ckpt["iteration"]
        self.train_stats         = ckpt["train_stats"]
        print(f"[C2P-PPO] Checkpoint loaded ← {path}")