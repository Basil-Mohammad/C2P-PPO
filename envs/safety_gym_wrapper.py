# =============================================================
# safety_gym_wrapper.py — Safety-Gymnasium PointGoal Wrapper
# C2P-PPO Project | Safe RL with CTL Constraints
#
# Wraps SafetyPointGoal1-v0 to integrate with C2P-PPO:
#   - Unified step() API (obs, reward, cost, done, info)
#   - CTL formula: AG(no_hazard_collision)
#   - Continuous action space [-1, 1]^2
# =============================================================

import numpy as np
import gymnasium
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────
# CTL Formula for PointGoal
# ─────────────────────────────────────────
class PointGoalCTL:
    """
    CTL safety formula for SafetyPointGoal1:
      φ = AG(¬hazard_collision)

    In Safety-Gymnasium, cost=1 means the agent
    contacted a hazard — a direct CTL violation.
    The branching-time property we enforce is:
      AG(cost = 0)
    i.e., along ALL paths, the agent never contacts
    a hazard. This requires CTL (not LTL) because
    we also verify:
      EF(AG(cost=0)) — there EXISTS a trajectory
      from which the agent can avoid hazards forever,
      which guides exploration toward recoverable states.
    """

    def __init__(self, cost_threshold: float = 0.5):
        self.cost_threshold = cost_threshold

    def is_violated(self, cost: float) -> bool:
        """Returns True if CTL formula is violated."""
        return cost >= self.cost_threshold

    def safety_margin(
        self,
        obs:  np.ndarray,
        cost: float,
    ) -> float:
        """
        Safety margin ∈ [0, 1].
        Uses lidar observations (indices 0:16 in PointGoal1)
        to estimate proximity to hazards.
        1.0 = far from hazards, 0.0 = in contact.
        """
        # PointGoal obs structure (60-dim):
        # [0:16]   hazard lidar (16 beams)
        # [16:32]  goal lidar
        # [32:48]  vases lidar
        # [48:52]  velocimeter
        # [52:56]  gyro
        # [56:60]  accelerometer
        if len(obs) >= 16:
            hazard_lidar = obs[0:16]
            # Lidar: 1.0=no obstacle, 0.0=obstacle close
            min_clearance = hazard_lidar.min()
            margin = float(min_clearance)
        else:
            margin = 1.0 - float(cost)

        return float(np.clip(margin, 0.0, 1.0))


# ─────────────────────────────────────────
# Safety-Gymnasium Wrapper
# ─────────────────────────────────────────
class SafetyGymnasiumWrapper:
    """
    Wrapper for SafetyPointGoal1-v0 that provides
    a unified interface compatible with C2P-PPO.

    Key differences from Safe-Control-Gym:
      - obs_dim: 60 (vs 4 for CartPole)
      - act_dim: 2 continuous in [-1, 1]
      - cost provided directly by environment
      - No need for learned safety surrogate
    """

    def __init__(
        self,
        env_name:       str   = "SafetyPointGoal1-v0",
        cost_threshold: float = 0.5,
        max_episode_steps: int = 500,
    ):
        import safety_gymnasium
        self.env = safety_gymnasium.make(
            env_name,
            max_episode_steps=max_episode_steps,
        )
        self.ctl = PointGoalCTL(cost_threshold)

        # Expose standard spaces
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

        # Stats
        self.episode_cost   = 0.0
        self.episode_steps  = 0

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        obs = np.array(obs, dtype=np.float32).flatten()
        self.episode_cost  = 0.0
        self.episode_steps = 0
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, float, bool, dict]:
        """
        Returns: (obs, reward, ctl_cost, done, info)
        ctl_cost ∈ {0.0, 1.0} — direct CTL violation flag
        """
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        obs, reward, cost, terminated, truncated, info = \
            self.env.step(action)

        obs  = np.array(obs, dtype=np.float32).flatten()
        done = terminated or truncated

        # CTL cost: 1 if hazard contact, 0 otherwise
        ctl_cost = float(self.ctl.is_violated(cost))

        self.episode_cost  += ctl_cost
        self.episode_steps += 1

        info["ctl_cost"]       = ctl_cost
        info["raw_cost"]       = cost
        info["episode_cost"]   = self.episode_cost
        info["safety_margin"]  = self.ctl.safety_margin(
            obs, cost
        )

        return obs, float(reward), ctl_cost, done, info

    def safety_margin(self, obs: np.ndarray) -> float:
        """Proxy safety margin from lidar observations."""
        if len(obs) >= 16:
            return float(np.clip(obs[0:16].min(), 0.0, 1.0))
        return 1.0

    def close(self):
        self.env.close()

    def action_space_sample(self) -> np.ndarray:
        return self.env.action_space.sample()


# ─────────────────────────────────────────
# Safety Mapper for PointGoal
# ─────────────────────────────────────────
class PointGoalSafetyMapper:
    """
    CTL Safety Mapper for SafetyPointGoal1.
    Uses environment-provided cost directly
    (no learned surrogate needed at Level 1).

    CTL formula: AG(¬hazard_contact)
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        s             = cfg["safety"]
        self.lambda_r = s["lambda_r"]
        self.eta      = s["eta"]

    def ctl_cost(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        env_cost: float,
    ) -> float:
        """CTLCost = env_cost (direct from environment)."""
        return float(env_cost > 0.5)

    def safety_margin(
        self,
        obs:      np.ndarray,
        env_cost: float = 0.0,
    ) -> float:
        """Safety margin from lidar + cost signal."""
        if len(obs) >= 16:
            lidar_margin = float(
                np.clip(obs[0:16].min(), 0.0, 1.0)
            )
        else:
            lidar_margin = 1.0
        cost_penalty = float(env_cost > 0.5)
        return float(lidar_margin * (1.0 - cost_penalty))

    def shape_reward(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        r_env:    float,
        env_cost: float,
    ):
        """R_total = R_env + λ_R · R_CTL"""
        cost    = self.ctl_cost(obs, next_obs, env_cost)
        r_ctl   = -self.eta if cost > 0 else 0.0
        r_total = r_env + self.lambda_r * r_ctl
        return r_total, cost