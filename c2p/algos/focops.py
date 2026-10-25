"""FOCOPS: First Order Constrained Optimization in Policy Space (Zhang, Vuong & Ross, 2020)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from c2p.algos.base import OnPolicyAgent


class FOCOPS(OnPolicyAgent):
    name = "FOCOPS"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.nu = 0.0

    def update(self, buf, last_v, last_vc):
        hp = self.hp
        adv, ret, cadv, cret = self.advantages(buf, last_v, last_vc)
        T = buf.tensors(self.device)
        obs, act, logp_old = T["obs"], T["act"], T["logp"]
        avg_cost = float(T["cost"].mean())
        # nu <- Proj_[0, nu_max]( nu + alpha_nu (J_C - d) / d )
        self.nu = float(np.clip(self.nu + hp.focops_nu_lr * (avg_cost - self.spec.budget) / self.spec.budget,
                                0.0, hp.focops_nu_max))
        adv = torch.as_tensor(adv, device=self.device)
        cadv = torch.as_tensor(cadv, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        cadv = cadv - cadv.mean()
        a_tot = adv - self.nu * cadv
        with torch.no_grad():
            d0 = self.actor.dist(obs)
            old_mu, old_std = d0.loc.clone(), d0.scale.clone()
        n = obs.shape[0]
        kl_last = 0.0
        for _ in range(hp.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, hp.minibatch):
                i = perm[s:s + hp.minibatch]
                dist = self.actor.dist(obs[i])
                kl = kl_divergence(dist, Normal(old_mu[i], old_std[i])).sum(-1)
                ratio = torch.exp(dist.log_prob(act[i]).sum(-1) - logp_old[i])
                loss = ((kl - (1.0 / hp.focops_lam) * ratio * a_tot[i]) * (kl.detach() <= hp.focops_eta).float()).mean()
                self.opt_pi.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), hp.max_grad_norm)
                self.opt_pi.step()
            with torch.no_grad():
                kl_last = kl_divergence(Normal(old_mu, old_std), self.actor.dist(obs)).sum(-1).mean().item()
            if kl_last > hp.target_kl:
                break
        self.update_critics(obs, torch.as_tensor(ret, device=self.device), torch.as_tensor(cret, device=self.device))
        self.n_updates += 1
        return {"kl": kl_last, "nu": self.nu}
