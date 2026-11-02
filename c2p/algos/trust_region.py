"""Second-order constrained trust-region methods.

CPO  -- Constrained Policy Optimization (Achiam et al., 2017), full
        implementation with Fisher-vector products, conjugate gradient, the
        dual solution of Achiam et al. (2017, Appendix 10.2), infeasibility
        recovery and backtracking line search.
PCPO -- Projection-Based Constrained Policy Optimization (Yang et al., 2020),
        KL-projection variant.

Constraint units: J_C is the expected per-step CTLCost, so the linearised
change of J_C induced by a new policy is E_{pi_old}[ratio * A^C].
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import kl_divergence

from c2p.algos.base import OnPolicyAgent


def flat_params(m):
    return torch.cat([p.data.view(-1) for p in m.parameters()])


def set_flat_params(m, flat):
    i = 0
    for p in m.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def flat_grad(y, m, retain=False, create=False):
    g = torch.autograd.grad(y, list(m.parameters()), retain_graph=(retain or create), create_graph=create, allow_unused=True)
    return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1) for gi, p in zip(g, m.parameters())])


def conjugate_gradient(Avp, b, iters=15, tol=1e-10):
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / (p @ Ap + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rr = r @ r
        if new_rr < tol:
            break
        p = r + (new_rr / rr) * p
        rr = new_rr
    return x


class _TrustRegionBase(OnPolicyAgent):
    def _setup(self, buf, last_v, last_vc):
        adv, ret, cadv, cret = self.advantages(buf, last_v, last_vc)
        T = buf.tensors(self.device)
        adv = torch.as_tensor(adv, device=self.device)
        cadv = torch.as_tensor(cadv, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        cadv = cadv - cadv.mean()
        with torch.no_grad():
            old_dist = self.actor.dist(T["obs"])
            old_dist = torch.distributions.Normal(old_dist.loc.clone(), old_dist.scale.clone())
        return T, adv, cadv, torch.as_tensor(ret, device=self.device), torch.as_tensor(cret, device=self.device), old_dist

    def _fvp_fn(self, obs, old_dist):
        def fvp(v):
            kl = kl_divergence(old_dist, self.actor.dist(obs)).sum(-1).mean()
            g = flat_grad(kl, self.actor, create=True)
            return flat_grad(g @ v, self.actor) + self.hp.cg_damping * v
        return fvp

    def _surrogates(self, obs, act, logp_old, adv, cadv):
        ratio = torch.exp(self.actor.log_prob(obs, act) - logp_old)
        return (ratio * adv).mean(), (ratio * cadv).mean()

    def _kl(self, obs, old_dist):
        with torch.no_grad():
            return kl_divergence(old_dist, self.actor.dist(obs)).sum(-1).mean().item()


class CPO(_TrustRegionBase):
    name = "CPO"

    def update(self, buf, last_v, last_vc):
        T, adv, cadv, ret, cret, old_dist = self._setup(buf, last_v, last_vc)
        obs, act, logp_old = T["obs"], T["act"], T["logp"]
        delta = self.hp.tr_delta
        c = float(T["cost"].mean()) - self.spec.budget      # J_C - d

        loss_r, loss_c = self._surrogates(obs, act, logp_old, adv, cadv)
        g = flat_grad(loss_r, self.actor, retain=True)      # ascent direction (reward)
        b = flat_grad(loss_c, self.actor)                   # cost gradient
        fvp = self._fvp_fn(obs, old_dist)
        x = conjugate_gradient(fvp, g, self.hp.cg_iters)
        q = float(g @ x)

        if float(b @ b) <= 1e-8 and c < 0:
            case = 4
            r_ = s_ = 0.0
        else:
            p = conjugate_gradient(fvp, b, self.hp.cg_iters)
            r_, s_ = float(g @ p), float(b @ p)
            A = q - r_ ** 2 / (s_ + 1e-8)
            B = 2 * delta - c ** 2 / (s_ + 1e-8)
            if c < 0 and B < 0:
                case = 3
            elif c < 0 <= B:
                case = 2
            elif c >= 0 and B >= 0:
                case = 1
            else:
                case = 0

        if case in (3, 4):
            step = np.sqrt(2 * delta / (q + 1e-8)) * x
        elif case in (1, 2):
            lam_a = np.sqrt(max(A, 0.0) / (B + 1e-8))
            lam_b = np.sqrt(q / (2 * delta))
            ce = c + 1e-8
            if c < 0:
                lam_a_s = float(np.clip(lam_a, 0, max(r_ / ce, 0)))
                lam_b_s = float(np.clip(lam_b, max(r_ / ce, 0), np.inf))
            else:
                lam_a_s = float(np.clip(lam_a, max(r_ / ce, 0), np.inf))
                lam_b_s = float(np.clip(lam_b, 0, max(r_ / ce, 0)))
            f_a = -0.5 * (A / (lam_a_s + 1e-8) + B * lam_a_s) - r_ * c / (s_ + 1e-8)
            f_b = -0.5 * (q / (lam_b_s + 1e-8) + 2 * delta * lam_b_s)
            lam_s = lam_a_s if f_a >= f_b else lam_b_s
            nu_s = max(lam_s * c - r_, 0.0) / (s_ + 1e-8)
            step = (x - nu_s * p) / (lam_s + 1e-8)
        else:  # infeasible: pure recovery step decreasing the constraint
            step = -np.sqrt(2 * delta / (s_ + 1e-8)) * p

        theta0 = flat_params(self.actor)
        with torch.no_grad():
            r0, c0 = (float(v) for v in self._surrogates(obs, act, logp_old, adv, cadv))
        accepted, frac = False, 1.0
        for _ in range(self.hp.ls_steps):
            set_flat_params(self.actor, theta0 + frac * step)
            with torch.no_grad():
                r1, c1 = (float(v) for v in self._surrogates(obs, act, logp_old, adv, cadv))
            kl = self._kl(obs, old_dist)
            improve_ok = True if case <= 1 else (r1 - r0) > 0
            cost_ok = (c1 - c0) <= max(-c, 0.0)
            if np.isfinite(kl) and kl <= delta and improve_ok and cost_ok:
                accepted = True
                break
            frac *= self.hp.ls_decay
        if not accepted:
            set_flat_params(self.actor, theta0)
        self.update_critics(obs, ret, cret)
        self.n_updates += 1
        return {"kl": self._kl(obs, old_dist), "cpo_case": case, "ls_accept": float(accepted)}


class PCPO(_TrustRegionBase):
    name = "PCPO"

    def update(self, buf, last_v, last_vc):
        T, adv, cadv, ret, cret, old_dist = self._setup(buf, last_v, last_vc)
        obs, act, logp_old = T["obs"], T["act"], T["logp"]
        delta = self.hp.tr_delta
        c = float(T["cost"].mean()) - self.spec.budget
        loss_r, loss_c = self._surrogates(obs, act, logp_old, adv, cadv)
        g = flat_grad(loss_r, self.actor, retain=True)
        b = flat_grad(loss_c, self.actor)
        fvp = self._fvp_fn(obs, old_dist)
        x = conjugate_gradient(fvp, g, self.hp.cg_iters)
        p = conjugate_gradient(fvp, b, self.hp.cg_iters)
        q, s_ = float(g @ x), float(b @ p)
        step = np.sqrt(2 * delta / (q + 1e-8)) * x                  # reward improvement step
        viol = float(b @ step) + c
        step = step - max(0.0, viol / (s_ + 1e-8)) * p               # KL projection onto linearised constraint
        theta0 = flat_params(self.actor)
        frac, accepted = 1.0, False
        for _ in range(self.hp.ls_steps):
            set_flat_params(self.actor, theta0 + frac * step)
            kl = self._kl(obs, old_dist)
            if np.isfinite(kl) and kl <= 2 * delta:
                accepted = True
                break
            frac *= self.hp.ls_decay
        if not accepted:
            set_flat_params(self.actor, theta0)
        self.update_critics(obs, ret, cret)
        self.n_updates += 1
        return {"kl": self._kl(obs, old_dist), "ls_accept": float(accepted)}
