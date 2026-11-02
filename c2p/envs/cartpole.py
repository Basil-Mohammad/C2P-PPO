"""Safe-Control-Gym CartPole trajectory tracking with a CTL position/angle constraint.

Task: the cart must track the reference x_ref(t) = 1.5 sin(2 pi t / T), T = 10 s,
using the default exponential quadratic tracking reward of Safe-Control-Gym
(Yuan et al., 2022). Safety specification:

    phi = AG( |x| <= 1.0  &  |theta| <= 0.2 )

The admissible region is strictly contained in the termination region of the
environment (|x| <= 2.4, |theta| <= pi/2), and the reference leaves the safe
region during roughly 45% of every period. Hence reward maximisation and
constraint satisfaction are in genuine conflict: an agent that tracks the
reference perfectly violates phi, whereas a safe agent must saturate at the
boundary.
"""
from __future__ import annotations

import numpy as np

from c2p.ctl import SafetySpec
from c2p.envs.base import SafeEnv


class CartPoleTrack(SafeEnv):
    name = "CartPoleTrack"
    X_MAX = 1.0
    THETA_MAX = 0.2

    def __init__(self, budget: float = 0.05):
        from safe_control_gym.utils.registration import make  # explicit failure if missing
        self.env = make(
            "cartpole", task="traj_tracking", episode_len_sec=10, ctrl_freq=50,
            obs_goal_horizon=1, randomized_init=True, done_on_out_of_bound=True,
            task_info={"trajectory_type": "circle", "num_cycles": 1,
                       "trajectory_plane": "zx", "trajectory_position_offset": [0, 0],
                       "trajectory_scale": 1.5},
        )
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])
        self.low = self.env.action_space.low.astype(np.float64)
        self.high = self.env.action_space.high.astype(np.float64)
        self.max_episode_steps = 500
        self.spec = SafetySpec(predicates=["|x|<=1.0", "|theta|<=0.2"], budget=budget)
        self._state = None

    def _rho(self, s) -> np.ndarray:
        x, th = float(s[0]), float(np.arctan2(np.sin(s[2]), np.cos(s[2])))
        return np.array([(self.X_MAX - abs(x)) / self.X_MAX,
                         (self.THETA_MAX - abs(th)) / self.THETA_MAX], np.float32)

    def robustness(self) -> np.ndarray:
        return self._rho(self._state)

    def reset(self, seed=None):
        out = self.env.reset(seed=seed)
        obs = out[0] if isinstance(out, tuple) else out
        self._state = np.asarray(self.env.state, np.float64)
        return np.asarray(obs, np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        a = self.low + 0.5 * (a + 1.0) * (self.high - self.low)
        obs, rew, done, info = self.env.step(a)
        self._state = np.asarray(self.env.state, np.float64)
        rho = self._rho(self._state)
        terminated = bool(info.get("out_of_bounds", False))
        truncated = bool(done and not terminated)
        return np.asarray(obs, np.float32), float(rew), rho, terminated, truncated, {}

    def close(self):
        self.env.close()
