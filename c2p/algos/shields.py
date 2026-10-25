"""Filter-based baselines.

ShieldedPPO -- post-posed shield in the spirit of Alshiekh et al. (2018): an
    action flagged unsafe by the learned safety critic is replaced by the
    safest of M policy samples. The shield is binary and non-adaptive, and the
    executed action is used for the PPO update without any correction for the
    induced distribution shift.
SafetyLayerPPO -- Dalal et al. (2018): a learned linear model of the
    constraint dynamics, rho_j(s') ~ rho_j(s) + g_j(s)^T a, is used to project
    each action onto the half-space that keeps the most critical predicate
    above a margin, in closed form.
"""
from __future__ import annotations

import numpy as np
import torch

from c2p.algos.ppo_family import PPO
from c2p.algos.safety_models import LinearSafetyModel, SafetyCriticTrainer


class ShieldedPPO(PPO):
    name = "Shielded-PPO"

    def __init__(self, obs_dim, act_dim, spec, hp, device="cpu"):
        super().__init__(obs_dim, act_dim, spec, hp, device)
        self.sc = SafetyCriticTrainer(obs_dim, act_dim, hp, self.device)

    @property
    def shield_on(self):
        return self.sc.trained and self.n_updates >= self.hp.shield_warmup_updates

    @torch.no_grad()
    def _safest(self, o, d):
        cand = d.sample((self.hp.n_candidates,)).squeeze(1)       # M x act
        p = self.sc.prob(o.expand(cand.shape[0], -1), cand.clamp(-1, 1))
        return cand[int(torch.argmax(p))], p

    @torch.no_grad()
    def act(self, obs_n, env=None):
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        d = self.actor.dist(o)
        a = d.sample()[0]
        intervened = 0.0
        if self.shield_on and float(self.sc.prob(o, a.clamp(-1, 1).unsqueeze(0))) < self.hp.tau_safe:
            a, _ = self._safest(o, d)
            intervened = 1.0
        return {"act": a.cpu().numpy(), "exec": a.clamp(-1, 1).cpu().numpy(),
                "logp": float(d.log_prob(a.unsqueeze(0)).sum(-1)), "v": float(self.vr(o)),
                "vc": float(self.vc(o)), "is_w": 1.0, "mask": 1.0, "intervened": intervened}

    @torch.no_grad()
    def act_eval(self, obs_n, env=None):
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        d = self.actor.dist(o)
        a = d.mean[0]
        if self.shield_on and float(self.sc.prob(o, a.clamp(-1, 1).unsqueeze(0))) < self.hp.tau_safe:
            a, _ = self._safest(o, d)
        return a.clamp(-1, 1).cpu().numpy()

    def update(self, buf, last_v, last_vc):
        n = buf.ptr
        o, a, y = self.sc.add(buf.raw_obs[:n], buf.exec_act[:n], buf.cost[:n], buf.done[:n])
        st = self.sc.evaluate(self.norm(o), a, y)
        st.update(super().update(buf, last_v, last_vc))
        st.update(self.sc.train(self.norm))
        return st


class SafetyLayerPPO(PPO):
    name = "SafetyLayer-PPO"

    def __init__(self, obs_dim, act_dim, spec, hp, device="cpu"):
        super().__init__(obs_dim, act_dim, spec, hp, device)
        self.g = LinearSafetyModel(obs_dim, act_dim, spec.m).to(self.device)
        self.g_opt = torch.optim.Adam(self.g.parameters(), lr=1e-3)
        cap = 200_000
        self.D = {"o": np.zeros((cap, obs_dim), np.float32), "a": np.zeros((cap, act_dim), np.float32),
                  "r0": np.zeros((cap, spec.m), np.float32), "r1": np.zeros((cap, spec.m), np.float32)}
        self.n, self.ptr, self.cap, self.trained = 0, 0, cap, False

    def _project(self, o, a, rho):
        """Closed-form projection for the most critical predicate (Dalal et al., 2018, Eq. 6)."""
        G = self.g(o)[0]                                           # m x act
        rho = torch.as_tensor(np.clip(rho, -1.0, 2.0), device=self.device, dtype=torch.float32)
        need = self.hp.sl_margin - rho - G @ a                     # constraint: rho + G a >= margin
        lam = torch.clamp(need / (G.pow(2).sum(-1) + 1e-6), min=0.0)
        j = int(torch.argmax(lam))
        return a + lam[j] * G[j], float(lam[j] > 0)

    @torch.no_grad()
    def act(self, obs_n, env=None):
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        d = self.actor.dist(o)
        a = d.sample()[0]
        inter = 0.0
        if self.trained and env is not None:
            a, inter = self._project(o, a.clamp(-1, 1), env.robustness())
        return {"act": a.cpu().numpy(), "exec": a.clamp(-1, 1).cpu().numpy(),
                "logp": float(d.log_prob(a.unsqueeze(0)).sum(-1)), "v": float(self.vr(o)),
                "vc": float(self.vc(o)), "is_w": 1.0, "mask": 1.0, "intervened": inter}

    @torch.no_grad()
    def act_eval(self, obs_n, env=None):
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        a = self.actor.dist(o).mean[0].clamp(-1, 1)
        if self.trained and env is not None:
            a, _ = self._project(o, a, env.robustness())
        return a.clamp(-1, 1).cpu().numpy()

    def update(self, buf, last_v, last_vc):
        n = buf.ptr
        for k, src in (("o", buf.raw_obs), ("a", buf.exec_act), ("r0", buf.rho_prev), ("r1", buf.rho_next)):
            arr = src[:n] if k != "r0" and k != "r1" else np.clip(src[:n], -1.0, 2.0)
            idx = (self.ptr + np.arange(n)) % self.cap
            self.D[k][idx] = arr
        self.ptr = (self.ptr + n) % self.cap
        self.n = min(self.n + n, self.cap)
        st = super().update(buf, last_v, last_vc)
        o = torch.as_tensor(self.norm(self.D["o"][:self.n]), device=self.device)
        a = torch.as_tensor(self.D["a"][:self.n], device=self.device)
        r0 = torch.as_tensor(self.D["r0"][:self.n], device=self.device)
        r1 = torch.as_tensor(self.D["r1"][:self.n], device=self.device)
        losses = []
        perm = torch.randperm(self.n)
        for s in range(0, min(self.n, 50_000), 512):
            i = perm[s:s + 512]
            pred = r0[i] + (self.g(o[i]) @ a[i].unsqueeze(-1)).squeeze(-1)
            loss = (pred - r1[i]).pow(2).mean()
            self.g_opt.zero_grad()
            loss.backward()
            self.g_opt.step()
            losses.append(loss.item())
        self.trained = self.n_updates >= self.hp.shield_warmup_updates
        st["sl_loss"] = float(np.mean(losses))
        return st
