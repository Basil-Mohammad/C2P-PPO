"""PPO, PPO-Lagrangian (Ray et al., 2019) and LCRL-PPO (Hasanbeig et al., 2019)."""
from __future__ import annotations

import numpy as np
import torch

from c2p.algos.base import OnPolicyAgent


def _std(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)


class PPO(OnPolicyAgent):
    name = "PPO"

    def update(self, buf, last_v, last_vc):
        adv, ret, cadv, cret = self.advantages(buf, last_v, last_vc)
        T = buf.tensors(self.device)
        adv, ret, cret = (torch.as_tensor(x, device=self.device) for x in (adv, ret, cret))
        st = self.ppo_policy_update(T["obs"], T["act"], T["logp"], _std(adv))
        self.update_critics(T["obs"], ret, cret)
        self.n_updates += 1
        return st


class PPOLagrangian(OnPolicyAgent):
    """Clipped surrogate on (A^R - lambda A^C)/(1 + lambda); lambda by projected ascent."""
    name = "PPO-Lag"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lam = self.hp.lambda_init

    def update(self, buf, last_v, last_vc):
        adv, ret, cadv, cret = self.advantages(buf, last_v, last_vc)
        T = buf.tensors(self.device)
        avg_cost = float(T["cost"].mean())
        self.lam = self.lagrange_step(self.lam, avg_cost)
        adv, cadv, ret, cret = (torch.as_tensor(x, device=self.device) for x in (adv, cadv, ret, cret))
        a = (_std(adv) - self.lam * (cadv - cadv.mean())) / (1.0 + self.lam)
        st = self.ppo_policy_update(T["obs"], T["act"], T["logp"], a)
        self.update_critics(T["obs"], ret, cret)
        self.n_updates += 1
        st["lambda"] = self.lam
        return st


class LCRLPPO(PPO):
    """Automaton-guided PPO for the safety LTL formula G p.

    The limit-deterministic Buchi automaton of G p has an accepting state u=0
    and an absorbing rejecting sink u=1. The policy acts on the product state
    (s, u) and receives r_env - eta * 1[u' = sink], i.e. a persistent penalty
    once the specification is irrecoverably violated. No shield and no
    Lagrangian multiplier are used.
    """
    name = "LCRL-PPO"

    def __init__(self, obs_dim, act_dim, spec, hp, device="cpu"):
        super().__init__(obs_dim + 1, act_dim, spec, hp, device)
        self.u = 0.0

    def augment_obs(self, obs):
        return np.concatenate([obs, [self.u]]).astype(np.float32)

    def track(self, rho_next):
        if np.any(np.asarray(rho_next) < 0):
            self.u = 1.0

    def shape_reward(self, r, rho_prev, rho_next, terminated):
        return r - self.hp.lcrl_eta * self.u

    def reset_episode(self):
        self.u = 0.0
