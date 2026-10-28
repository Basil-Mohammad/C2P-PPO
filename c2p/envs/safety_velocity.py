"""Safety-Gymnasium velocity tasks (Ji et al., 2023), e.g. SafetyHalfCheetahVelocity-v1.

phi = AG(v_x <= v_max): the robot must run forward as fast as possible while
its forward velocity stays below the task's velocity threshold.  The
robustness of the atomic proposition is the normalised velocity margin
rho = (v_max - v_x) / v_max, so that rho < 0 coincides exactly with the native
Safety-Gymnasium cost (cost = 1[v_x > v_max]).
"""
from __future__ import annotations

import numpy as np

from c2p.ctl import SafetySpec
from c2p.envs.base import SafeEnv


class SafetyVelocity(SafeEnv):
    def __init__(self, env_id: str = "SafetyHalfCheetahVelocity-v1", budget: float = 0.025):
        import safety_gymnasium  # explicit failure if missing
        self.name = env_id.replace("-v1", "")
        self.env = safety_gymnasium.make(env_id)
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])
        self.low = self.env.action_space.low.astype(np.float64)
        self.high = self.env.action_space.high.astype(np.float64)
        self.max_episode_steps = 1000
        self.vmax = float(self.env.unwrapped._velocity_threshold)
        # 25 cost units per 1000-step episode, the standard Safety-Gymnasium budget
        self.spec = SafetySpec(predicates=["v_x<=v_max"], budget=budget)
        self._v = 0.0

    def _rho(self) -> np.ndarray:
        return np.array([(self.vmax - self._v) / self.vmax], np.float32)

    def robustness(self) -> np.ndarray:
        return self._rho()

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self._v = float(self.env.unwrapped.data.qvel[0])
        return np.asarray(obs, np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        a = self.low + 0.5 * (a + 1.0) * (self.high - self.low)
        obs, rew, cost, term, trunc, info = self.env.step(a)
        self._v = float(info.get("x_velocity", self.env.unwrapped.data.qvel[0]))
        rho = self._rho()
        # the native cost is authoritative for the Boolean label
        if cost > 0 and rho[0] >= 0:
            rho[0] = -1e-3
        if cost <= 0 and rho[0] < 0:
            rho[0] = 1e-3
        return np.asarray(obs, np.float32), float(rew), rho, bool(term), bool(trunc), {"x_velocity": self._v}

    def close(self):
        self.env.close()
