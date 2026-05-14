# =============================================================
# baselines.py — Baseline Agents for Comparison
# C2P-PPO Project | Safe RL with CTL Constraints
#
# Baselines:
#   1. PPO-Lagrangian  (Ray et al., 2019)
#   2. Shielded PPO    (Alshiekh et al., 2018)
#   3. CBF-PPO         (Chow et al., 2018)
# =============================================================

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Tuple

from src.ppo_base   import ActorCritic, RolloutBuffer
from src.lagrangian import LagrangianMultiplier


# ─────────────────────────────────────────
# 1. PPO-Lagrangian
# ─────────────────────────────────────────
class PPOLagrangianAgent:
    """
    PPO with scalar Lagrangian constraint (Ray et al., 2019).
    Differs from C2P-PPO:
      - No CTL formula (scalar cost only)
      - No adaptive shielding
      - No exploration-awareness
      - No distribution shift correction
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        self.cfg    = cfg
        self.device = torch.device(
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

        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        self.optimizer  = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.lagrangian = LagrangianMultiplier(cfg)
        self.buffer     = RolloutBuffer()

        # Cost buffer for Lagrangian
        self.costs:      list = []
        self.train_stats: Dict = {}

    def _is_safe(self, obs: np.ndarray) -> float:
        """Scalar safety cost (same CTL formula for fairness)."""
        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
        violated   = (
            abs(cart_pos) >= 2.4 or abs(pole_angle) >= 0.2
        )
        return 1.0 if violated else 0.0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = self.ac.get_action(obs_t)
        return (
            action.cpu().numpy(),
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
        )

    def add_cost(
        self, obs: np.ndarray, next_obs: np.ndarray
    ):
        self.costs.append(self._is_safe(next_obs))

    def update(self, last_obs: np.ndarray) -> Dict:
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        obs_t, actions_t, old_lp_t = self.buffer.get_tensors()
        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        mean_cost = float(np.mean(self.costs)) \
            if self.costs else 0.0
        lambda_t  = self.lagrangian.lambda_t

        total_loss = total_pg = total_vf = 0.0
        total_ent  = 0.0
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

                pg_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss = -entropy.mean()

                # Lagrangian penalty (scalar cost)
                loss = (pg_loss
                        + self.value_coef   * vf_loss
                        + self.entropy_coef * ent_loss
                        + lambda_t * mean_cost)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += entropy.mean().item()
                n_updates  += 1

        self.lagrangian.update(mean_cost)
        self.buffer.clear()
        self.costs = []

        self.train_stats = {
            "loss":     total_loss / n_updates,
            "pg_loss":  total_pg   / n_updates,
            "vf_loss":  total_vf   / n_updates,
            "entropy":  total_ent  / n_updates,
            "lambda_t": self.lagrangian.lambda_t,
            "ctl_cost": mean_cost,
        }
        return self.train_stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "lambda_t":    self.lagrangian.lambda_t,
        }, path)
        print(f"[PPO-Lag] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.lagrangian.lambda_t = ckpt["lambda_t"]
        print(f"[PPO-Lag] Checkpoint loaded ← {path}")


# ─────────────────────────────────────────
# 2. Shielded PPO
# ─────────────────────────────────────────
class ShieldedPPOAgent:
    """
    PPO with runtime shield (Alshiekh et al., 2018).
    Differs from C2P-PPO:
      - Hard shield only (no Lagrangian)
      - No adaptive α (fixed shield)
      - No exploration bonus
      - No distribution shift correction
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        self.cfg    = cfg
        self.device = torch.device(
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

        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        self.optimizer   = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.buffer      = RolloutBuffer()
        self.train_stats: Dict = {}
        self.costs:       list = []

    def _safety_margin(self, obs: np.ndarray) -> float:
        """How close are we to constraint boundary?"""
        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
        cart_m  = max(0.0, 1.0 - abs(cart_pos)  / 2.4)
        angle_m = max(0.0, 1.0 - abs(pole_angle) / 0.2)
        return min(cart_m, angle_m)

    def _shield_action(
        self, action: np.ndarray, obs: np.ndarray
    ) -> np.ndarray:
        """
        Fixed shield: scale action toward 0 when near boundary.
        No adaptation — shield strength is constant.
        """
        margin = self._safety_margin(obs)
        # Scale down action when unsafe (margin < 0.3)
        if margin < 0.3:
            scale = margin / 0.3
            action = action * scale
        return np.clip(action, -10.0, 10.0)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()

        # Apply fixed shield
        shielded = self._shield_action(action_np, obs)

        return (
            shielded,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
        )

    def add_cost(
        self, obs: np.ndarray, next_obs: np.ndarray
    ):
        cart_pos   = float(next_obs[0])
        pole_angle = float(next_obs[2])
        violated   = (
            abs(cart_pos) >= 2.4 or abs(pole_angle) >= 0.2
        )
        self.costs.append(1.0 if violated else 0.0)

    def update(self, last_obs: np.ndarray) -> Dict:
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        obs_t, actions_t, old_lp_t = self.buffer.get_tensors()
        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        mean_cost = float(np.mean(self.costs)) \
            if self.costs else 0.0

        total_loss = total_pg = total_vf = 0.0
        total_ent  = 0.0
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

                pg_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss = -entropy.mean()
                loss     = (pg_loss
                            + self.value_coef   * vf_loss
                            + self.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += entropy.mean().item()
                n_updates  += 1

        self.buffer.clear()
        self.costs = []

        self.train_stats = {
            "loss":     total_loss / n_updates,
            "pg_loss":  total_pg   / n_updates,
            "vf_loss":  total_vf   / n_updates,
            "entropy":  total_ent  / n_updates,
            "ctl_cost": mean_cost,
        }
        return self.train_stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, path)
        print(f"[Shield-PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        print(f"[Shield-PPO] Checkpoint loaded ← {path}")


# ─────────────────────────────────────────
# 3. CBF-PPO
# ─────────────────────────────────────────
class CBFPPOAgent:
    """
    PPO with Control Barrier Function safety filter.
    CBF projects actions onto the safe set.
    Differs from C2P-PPO:
      - CBF projection (not CTL shield)
      - No Lagrangian
      - No adaptive exploration
      - No temporal logic
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        self.cfg    = cfg
        self.device = torch.device(
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

        # CBF parameters
        self.cbf_gamma = 0.5  # CBF decay rate

        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        self.optimizer   = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.buffer      = RolloutBuffer()
        self.train_stats: Dict = {}
        self.costs:       list = []

    def _cbf_value(self, obs: np.ndarray) -> float:
        """
        CBF h(x) = min(2.4 - |cart_pos|, 0.2 - |pole_angle|)
        Safe set: h(x) >= 0
        """
        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
        return min(
            2.4 - abs(cart_pos),
            0.2 - abs(pole_angle)
        )

    def _cbf_project(
        self, action: np.ndarray, obs: np.ndarray
    ) -> np.ndarray:
        """
        Project action to satisfy CBF constraint:
        h_dot + γ·h(x) >= 0
        Simplified: scale action based on CBF value.
        """
        h = self._cbf_value(obs)

        if h > 0.1:
            # Safely inside — no projection needed
            return action

        # Near boundary — project toward safe action
        cart_pos   = float(obs[0]) if len(obs) > 0 else 0.0
        pole_angle = float(obs[2]) if len(obs) > 2 else 0.0

        # Safe direction: push cart/pole toward center
        safe_action = np.array(
            [-np.sign(cart_pos) * 2.0
             - np.sign(pole_angle) * 3.0]
        )

        # Blend: more projection when closer to boundary
        alpha = max(0.0, min(1.0, h / 0.1))
        projected = alpha * action + (1 - alpha) * safe_action
        return np.clip(projected, -10.0, 10.0)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = self.ac.get_action(obs_t)
        action_np = action.cpu().numpy()

        # Apply CBF projection
        safe_action = self._cbf_project(action_np, obs)

        return (
            safe_action,
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
        )

    def add_cost(
        self, obs: np.ndarray, next_obs: np.ndarray
    ):
        cart_pos   = float(next_obs[0])
        pole_angle = float(next_obs[2])
        violated   = (
            abs(cart_pos) >= 2.4 or abs(pole_angle) >= 0.2
        )
        self.costs.append(1.0 if violated else 0.0)

    def update(self, last_obs: np.ndarray) -> Dict:
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        obs_t, actions_t, old_lp_t = self.buffer.get_tensors()
        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        mean_cost = float(np.mean(self.costs)) \
            if self.costs else 0.0

        total_loss = total_pg = total_vf = 0.0
        total_ent  = 0.0
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

                pg_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss = -entropy.mean()
                loss     = (pg_loss
                            + self.value_coef   * vf_loss
                            + self.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += entropy.mean().item()
                n_updates  += 1

        self.buffer.clear()
        self.costs = []

        self.train_stats = {
            "loss":     total_loss / n_updates,
            "pg_loss":  total_pg   / n_updates,
            "vf_loss":  total_vf   / n_updates,
            "entropy":  total_ent  / n_updates,
            "ctl_cost": mean_cost,
        }
        return self.train_stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, path)
        print(f"[CBF-PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        print(f"[CBF-PPO] Checkpoint loaded ← {path}")
# ─────────────────────────────────────────
# 4. LTL-PPO
# ─────────────────────────────────────────
class LTLPPOAgent:
    """
    PPO with LTL reward shaping (Hasanbeig et al., 2019).
    Differs from C2P-PPO:
      - LTL (linear) not CTL (branching)
      - Reward shaping only (no shield, no Lagrangian)
      - No adaptive exploration
      - No distribution shift correction
    LTL formula: G(safe) — globally safe
    Implemented as progress-based reward shaping.
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        self.cfg    = cfg
        self.device = torch.device(
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

        # LTL shaping weight
        self.ltl_weight = cfg["safety"]["lambda_r"]
        self.eta        = cfg["safety"]["eta"]

        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        self.optimizer   = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.buffer      = RolloutBuffer()
        self.train_stats: Dict = {}
        self.costs:       list = []

    def _ltl_progress(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
    ) -> float:
        """
        LTL-based reward shaping.
        Formula: G(|pole_angle| < 0.2 ∧ |cart_pos| < 2.4)
        R_LTL = +η if safe, -η if violated.
        Linear (no branching quantifiers unlike CTL).
        """
        cart_pos   = float(next_obs[0])
        pole_angle = float(next_obs[2])
        safe = (
            abs(cart_pos)   < 2.4
            and abs(pole_angle) < 0.2
        )
        return self.eta if safe else -self.eta

    def _ctl_cost(self, next_obs: np.ndarray) -> float:
        cart_pos   = float(next_obs[0])
        pole_angle = float(next_obs[2])
        return float(
            abs(cart_pos) >= 2.4
            or abs(pole_angle) >= 0.2
        )

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        return (
            action.cpu().numpy(),
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
        )

    def shape_reward(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        r_env:    float,
    ) -> Tuple[float, float]:
        ltl_r  = self._ltl_progress(obs, next_obs)
        r_total = r_env + self.ltl_weight * ltl_r
        cost    = self._ctl_cost(next_obs)
        return r_total, cost

    def update(self, last_obs: np.ndarray) -> Dict:
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        obs_t, actions_t, old_lp_t = \
            self.buffer.get_tensors()
        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        mean_cost = float(np.mean(self.costs)) \
            if self.costs else 0.0

        total_loss = total_pg = total_vf = 0.0
        total_ent  = 0.0
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

                pg_loss  = -torch.min(surr1, surr2).mean()
                vf_loss  = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss = -entropy.mean()
                loss     = (pg_loss
                            + self.value_coef   * vf_loss
                            + self.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += entropy.mean().item()
                n_updates  += 1

        self.buffer.clear()
        self.costs = []

        self.train_stats = {
            "loss":     total_loss / n_updates,
            "pg_loss":  total_pg   / n_updates,
            "vf_loss":  total_vf   / n_updates,
            "entropy":  total_ent  / n_updates,
            "ctl_cost": mean_cost,
        }
        return self.train_stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.ac.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }, path)
        print(f"[LTL-PPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        print(f"[LTL-PPO] Checkpoint loaded ← {path}")


# ─────────────────────────────────────────
# 5. CPO (Approximated)
# ─────────────────────────────────────────
class CPOAgent:
    """
    Approximated CPO (Achiam et al., 2017).
    Uses first-order constraint update instead of
    second-order Fisher matrix (computational reasons).
    Noted explicitly in paper as approximated CPO.

    Differs from C2P-PPO:
      - No CTL formula (scalar constraint)
      - No adaptive shielding
      - No exploration bonus
      - Trust region via KL penalty (not clipping)
    """

    def __init__(self, cfg: dict, obs_dim: int, act_dim: int):
        self.cfg    = cfg
        self.device = torch.device(
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

        # CPO-specific
        self.kl_coef      = 0.2   # KL penalty coefficient
        self.cost_limit   = cfg["safety"]["delta_safety"]
        self.lagrangian   = LagrangianMultiplier(cfg)

        self.ac = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)

        # Store old policy for KL computation
        self.ac_old = ActorCritic(
            obs_dim, act_dim,
            continuous=True,
            act_low=-10.0,
            act_high=10.0,
        ).to(self.device)
        self.ac_old.load_state_dict(self.ac.state_dict())

        self.optimizer   = torch.optim.Adam(
            self.ac.parameters(), lr=self.lr
        )
        self.buffer      = RolloutBuffer()
        self.train_stats: Dict = {}
        self.costs:       list = []

    def _ctl_cost(self, next_obs: np.ndarray) -> float:
        cart_pos   = float(next_obs[0])
        pole_angle = float(next_obs[2])
        return float(
            abs(cart_pos) >= 2.4
            or abs(pole_angle) >= 0.2
        )

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.tensor(
            obs, dtype=torch.float32
        ).to(self.device)
        action, log_prob, _, value = \
            self.ac.get_action(obs_t)
        return (
            action.cpu().numpy(),
            log_prob.detach(),
            value.detach(),
            obs_t.detach(),
        )

    def add_cost(
        self, obs: np.ndarray, next_obs: np.ndarray
    ):
        self.costs.append(self._ctl_cost(next_obs))

    def _kl_divergence(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between old and new policy."""
        from torch.distributions import Normal

        with torch.no_grad():
            mean_old, _ = self.ac_old.forward(obs)
            std_old = self.ac_old.actor_log_std.exp()

        mean_new, _ = self.ac.forward(obs)
        std_new = self.ac.actor_log_std.exp()

        dist_old = Normal(mean_old, std_old)
        dist_new = Normal(mean_new, std_new)

        kl = torch.distributions.kl_divergence(
            dist_old, dist_new
        ).sum(-1).mean()
        return kl

    def update(self, last_obs: np.ndarray) -> Dict:
        last_obs_t = torch.tensor(
            last_obs, dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            _, last_value = self.ac.forward(last_obs_t)

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        obs_t, actions_t, old_lp_t = \
            self.buffer.get_tensors()
        obs_t      = obs_t.to(self.device)
        actions_t  = actions_t.to(self.device)
        old_lp_t   = old_lp_t.to(self.device)
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        mean_cost = float(np.mean(self.costs)) \
            if self.costs else 0.0
        lambda_t  = self.lagrangian.lambda_t

        total_loss = total_pg = total_vf = 0.0
        total_ent  = total_kl = 0.0
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
                pg_loss = -torch.min(surr1, surr2).mean()

                vf_loss  = nn.functional.mse_loss(
                    values, returns[idx]
                )
                ent_loss = -entropy.mean()

                # KL penalty (CPO trust region)
                kl = self._kl_divergence(
                    obs_t[idx], actions_t[idx]
                )

                # Lagrangian constraint
                cost_loss = lambda_t * mean_cost

                loss = (pg_loss
                        + self.value_coef   * vf_loss
                        + self.entropy_coef * ent_loss
                        + self.kl_coef      * kl
                        + cost_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += entropy.mean().item()
                total_kl   += kl.item()
                n_updates  += 1

        # Update old policy
        self.ac_old.load_state_dict(self.ac.state_dict())
        self.lagrangian.update(mean_cost)
        self.buffer.clear()
        self.costs = []

        self.train_stats = {
            "loss":     total_loss / n_updates,
            "pg_loss":  total_pg   / n_updates,
            "vf_loss":  total_vf   / n_updates,
            "entropy":  total_ent  / n_updates,
            "kl":       total_kl   / n_updates,
            "lambda_t": self.lagrangian.lambda_t,
            "ctl_cost": mean_cost,
        }
        return self.train_stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state":     self.ac.state_dict(),
            "ac_old_state":    self.ac_old.state_dict(),
            "optim_state":     self.optimizer.state_dict(),
            "lambda_t":        self.lagrangian.lambda_t,
        }, path)
        print(f"[CPO] Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["model_state"])
        self.ac_old.load_state_dict(ckpt["ac_old_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.lagrangian.lambda_t = ckpt["lambda_t"]
        print(f"[CPO] Checkpoint loaded ← {path}")