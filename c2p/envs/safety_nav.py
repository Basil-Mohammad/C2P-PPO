"""Safety-Gymnasium navigation tasks (Ji et al., 2023): SafetyPointGoal1, SafetyCarGoal1.

phi = AG(not hazard), where the robustness of the atomic proposition is the
signed distance from the agent to the closest hazard boundary, normalised by
the hazard radius. Violation coincides with the native Safety-Gymnasium
hazard cost (cost_hazards > 0).
"""
from __future__ import annotations

import numpy as np

from c2p.ctl import SafetySpec
from c2p.envs.base import SafeEnv


class SafetyNav(SafeEnv):
    def __init__(self, env_id: str = "SafetyPointGoal1-v0", budget: float = 0.025):
        import safety_gymnasium  # explicit failure if missing
        self.name = env_id.replace("-v0", "")
        self.env = safety_gymnasium.make(env_id)
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])
        self.low = self.env.action_space.low.astype(np.float64)
        self.high = self.env.action_space.high.astype(np.float64)
        self.max_episode_steps = 1000
        # Budget 0.025 per step == 25 cost units per 1000-step episode, the
        # standard Safety-Gymnasium benchmark budget.
        self.spec = SafetySpec(predicates=["!hazard"], budget=budget)

    def _rho(self) -> np.ndarray:
        task = self.env.unwrapped.task
        a = np.asarray(task.agent.pos)[:2]
        hp = np.asarray(task.hazards.pos)[:, :2]
        r = float(task.hazards.size)
        d = np.linalg.norm(hp - a, axis=1).min() - r
        return np.array([d / r], np.float32)

    def robustness(self) -> np.ndarray:
        return self._rho()

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return np.asarray(obs, np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        a = self.low + 0.5 * (a + 1.0) * (self.high - self.low)
        obs, rew, cost, term, trunc, info = self.env.step(a)
        rho = self._rho()
        # The native hazard cost is authoritative for the Boolean label.
        if info.get("cost_hazards", cost) > 0 and rho[0] >= 0:
            rho[0] = -1e-3
        if info.get("cost_hazards", cost) <= 0 and rho[0] < 0:
            rho[0] = 1e-3
        out = {"goal_met": bool(info.get("goal_met", False))}
        return np.asarray(obs, np.float32), float(rew), rho, bool(term), bool(trunc), out

    def close(self):
        self.env.close()
