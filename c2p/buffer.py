"""On-policy rollout storage with GAE for reward and cost streams."""
from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, n_preds: int):
        self.size = size
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)        # pre-clipping Gaussian sample
        self.logp = np.zeros(size, np.float32)                  # log pi_old(a|s)
        self.rew = np.zeros(size, np.float32)                   # reward used for learning
        self.cost = np.zeros(size, np.float32)                  # CTLCost (conjunctive, Boolean)
        self.cost_vec = np.zeros((size, n_preds), np.float32)   # per-predicate violation indicators
        self.cost_learn = np.zeros(size, np.float32)            # cost with bootstrap at time-limit truncation
        self.cost_vec_learn = np.zeros((size, n_preds), np.float32)
        self.cval_vec = np.zeros((size, n_preds), np.float32)
        self.last_cvec = np.zeros(n_preds, np.float32)
        self.val = np.zeros(size, np.float32)
        self.cval = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)                  # episode boundary (terminated OR truncated)
        self.term = np.zeros(size, np.float32)                  # true termination (no bootstrap)
        self.is_w = np.ones(size, np.float32)                   # c_t = pi_old / pi_tilde (truncated)
        self.mask = np.ones(size, np.float32)                   # advantage mask
        self.ep_id = np.zeros(size, np.int64)
        self.raw_obs = np.zeros((size, obs_dim), np.float32)    # un-normalised observation
        self.exec_act = np.zeros((size, act_dim), np.float32)   # action actually executed (clipped)
        self.rho_prev = np.zeros((size, n_preds), np.float32)
        self.rho_next = np.zeros((size, n_preds), np.float32)
        self.ptr = 0

    def add(self, **kw) -> None:
        i = self.ptr
        for k, v in kw.items():
            if k in ("cand", "cand_logp", "cand_inadm"):
                if not hasattr(self, k):
                    v = np.asarray(v)
                    setattr(self, k, np.zeros((self.size,) + v.shape, np.float32))
            getattr(self, k)[i] = v
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def reset(self) -> None:
        self.ptr = 0
        self.is_w[:] = 1.0
        self.mask[:] = 1.0

    @staticmethod
    def gae(rew, val, last_val, done, term, gamma, lam):
        """GAE(gamma, lambda). Truncated episodes are bootstrapped beforehand
        (the bootstrap value is folded into the final reward), so here a
        boundary simply cuts the recursion."""
        n = len(rew)
        adv = np.zeros(n, np.float32)
        last = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                nv = 0.0 if done[t] else last_val
            else:
                nv = 0.0 if done[t] else val[t + 1]
            delta = rew[t] + gamma * nv - val[t]
            last = delta + gamma * lam * (0.0 if done[t] else 1.0) * last
            adv[t] = last
        ret = adv + val
        return adv, ret

    def tensors(self, device="cpu") -> dict[str, torch.Tensor]:
        n = self.ptr
        names = ["obs", "next_obs", "act", "logp", "rew", "cost", "cost_vec", "val", "cval",
                 "done", "term", "is_w", "mask"]
        return {k: torch.as_tensor(getattr(self, k)[:n], device=device) for k in names}
