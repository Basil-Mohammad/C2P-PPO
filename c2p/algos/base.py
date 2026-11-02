"""Shared machinery for all on-policy agents (PPO family and trust-region family)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from c2p.buffer import RolloutBuffer
from c2p.nets import Critic, GaussianActor, RunningMeanStd


@dataclass
class Hyper:
    # common
    steps_per_update: int = 4000
    gamma: float = 0.99
    lam: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    epochs: int = 10
    minibatch: int = 256
    clip: float = 0.2
    target_kl: float = 0.02          # early stopping => D_KL(pi_new||pi_old) controlled (Pinsker => TV)
    max_grad_norm: float = 0.5
    hidden: tuple = (64, 64)
    # Lagrangian (PPO-Lag, C2P-PPO)
    lambda_init: float = 0.1
    lambda_lr: float = 0.05          # eta_0 in eta_t = eta_0 / (1 + t / lambda_tau)
    lambda_tau: float = 100.0
    lambda_max: float = 100.0
    # trust region (CPO, PCPO)
    tr_delta: float = 0.01
    cg_iters: int = 15
    cg_damping: float = 0.1
    ls_steps: int = 15
    ls_decay: float = 0.8
    # FOCOPS
    focops_lam: float = 1.5
    focops_nu_lr: float = 0.01
    focops_nu_max: float = 2.0
    focops_eta: float = 0.02
    # shields / safety critic (Shielded PPO, C2P-PPO)
    n_candidates: int = 16           # M
    tau_safe: float = 0.5
    k_lookahead: int = 10            # Level-2 horizon k
    safety_buffer: int = 200_000
    safety_epochs: int = 2
    safety_lr: float = 1e-3
    shield_warmup_updates: int = 2
    # C2P-PPO adaptive shield
    alpha0: float = 0.8
    beta: float = 2.0
    window: int = 10                 # W episodes
    c_threshold_scale: float = 1.0   # C_threshold = scale * delta_safety
    is_clip: float = 2.0             # truncation level c_bar for c_t = pi_old/pi_tilde
    lambda_r: float = 1.0            # weight of potential-based CTL shaping
    kappa_inadm: float = 1.0         # weight of the inadmissible-mass penalty
    rnd_lr: float = 1e-4
    # LCRL-PPO
    lcrl_eta: float = 1.0
    # Safety layer
    sl_margin: float = 0.1
    # flags (ablations)
    flags: dict = field(default_factory=dict)


class OnPolicyAgent:
    """Gaussian actor with reward critic V^R and cost critic V^C."""
    name = "base"
    uses_cost_critic = True

    def __init__(self, obs_dim: int, act_dim: int, spec, hp: Hyper, device="cpu"):
        self.obs_dim, self.act_dim, self.spec, self.hp = obs_dim, act_dim, spec, hp
        self.device = torch.device(device)
        self.actor = GaussianActor(obs_dim, act_dim, hp.hidden).to(self.device)
        self.vr = Critic(obs_dim, hp.hidden).to(self.device)
        self.vc = Critic(obs_dim, hp.hidden).to(self.device)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=hp.lr_actor)
        self.opt_v = torch.optim.Adam(list(self.vr.parameters()) + list(self.vc.parameters()), lr=hp.lr_critic)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.n_updates = 0
        self.stats: dict = {}

    # ------------------------------------------------------------- acting
    def norm(self, obs: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(obs)

    @torch.no_grad()
    def act(self, obs_n: np.ndarray, env=None) -> dict:
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        d = self.actor.dist(o)
        a = d.sample()
        return {"act": a[0].cpu().numpy(), "exec": np.clip(a[0].cpu().numpy(), -1, 1),
                "logp": float(d.log_prob(a).sum(-1)), "v": float(self.vr(o)), "vc": float(self.vc(o)),
                "is_w": 1.0, "mask": 1.0, "intervened": 0.0}

    @torch.no_grad()
    def act_eval(self, obs_n: np.ndarray, env=None) -> np.ndarray:
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        return np.clip(self.actor.dist(o).mean[0].cpu().numpy(), -1, 1)

    @torch.no_grad()
    def values(self, obs_n: np.ndarray) -> tuple[float, float]:
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        return float(self.vr(o)), float(self.vc(o))

    # hooks
    def shape_reward(self, r: float, rho_prev, rho_next, terminated: bool) -> float:
        return r

    def augment_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def track(self, rho_next) -> None:
        """Update any specification-automaton state after a transition."""
        pass

    def reset_episode(self) -> None:
        pass

    def on_transition(self, **kw) -> None:
        pass

    def on_episode_end(self, ep_violation_rate: float) -> None:
        pass

    # ------------------------------------------------------------- learning
    def advantages(self, buf: RolloutBuffer, last_v: float, last_vc: float):
        n = buf.ptr
        adv, ret = buf.gae(buf.rew[:n], buf.val[:n], last_v, buf.done[:n], buf.term[:n], self.hp.gamma, self.hp.lam)
        cadv, cret = buf.gae(buf.cost_learn[:n], buf.cval[:n], last_vc, buf.done[:n], buf.term[:n], self.hp.gamma, self.hp.lam)
        return adv, ret, cadv, cret

    def update_critics(self, obs, ret, cret) -> None:
        n = obs.shape[0]
        for _ in range(self.hp.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.hp.minibatch):
                i = perm[s:s + self.hp.minibatch]
                loss = (self.vr(obs[i]) - ret[i]).pow(2).mean() + (self.vc(obs[i]) - cret[i]).pow(2).mean()
                self.opt_v.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.vr.parameters()) + list(self.vc.parameters()), self.hp.max_grad_norm)
                self.opt_v.step()

    def ppo_policy_update(self, obs, act, logp_old, adv, weight=None, virtual=None) -> dict:
        """Clipped surrogate with optional per-sample weights w_t (importance
        weights c_t times advantage mask). KL early stopping enforces the
        trust region that PPO clipping only approximates.

        virtual = (obs_v, act_v, logp_v, coef): inadmissible candidate actions
        treated as virtual samples with advantage -1, contributing
        coef * mean_v max(r_v, clip(r_v)) to the loss (an unbiased, clipped
        estimate of the inadmissible probability mass of pi_theta)."""
        n = obs.shape[0]
        w = torch.ones(n, device=self.device) if weight is None else weight
        with torch.no_grad():
            d_old = self.actor.dist(obs)
            d_old = Normal(d_old.loc.clone(), d_old.scale.clone())
        kl_hist, clipfrac, n_steps, stopped = [], [], 0, False
        for _ in range(self.hp.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.hp.minibatch):
                i = perm[s:s + self.hp.minibatch]
                logp = self.actor.log_prob(obs[i], act[i])
                ratio = torch.exp(logp - logp_old[i])
                s1 = ratio * adv[i]
                s2 = torch.clamp(ratio, 1 - self.hp.clip, 1 + self.hp.clip) * adv[i]
                loss = -(w[i] * torch.min(s1, s2)).sum() / (w[i].sum() + 1e-8)
                if virtual is not None and virtual[0].shape[0] > 0:
                    ov, av, lv, coef = virtual
                    j = torch.randint(0, ov.shape[0], (min(len(i), ov.shape[0]),))
                    rv = torch.exp(self.actor.log_prob(ov[j], av[j]) - lv[j])
                    loss = loss + coef * torch.max(rv, torch.clamp(rv, 1 - self.hp.clip, 1 + self.hp.clip)).mean()
                self.opt_pi.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.hp.max_grad_norm)
                self.opt_pi.step()
                n_steps += 1
                with torch.no_grad():
                    clipfrac.append(((ratio - 1).abs() > self.hp.clip).float().mean().item())
            with torch.no_grad():   # exact KL(pi_old || pi_theta) of the Gaussian policies on the batch states
                kl = kl_divergence(d_old, self.actor.dist(obs)).sum(-1).mean().item()
            kl_hist.append(kl)
            if kl > 1.5 * self.hp.target_kl:
                stopped = True
                break
        return {"kl": kl_hist[-1], "clipfrac": float(np.mean(clipfrac)), "pi_steps": n_steps, "early_stop": float(stopped)}

    def update(self, buf: RolloutBuffer, last_v: float, last_vc: float) -> dict:
        raise NotImplementedError

    # Lagrangian multiplier: projected stochastic (sub)gradient ascent with
    # Robbins--Monro step sizes eta_t = eta_0 / (1 + t / tau).
    def lagrange_step(self, lam: float, avg_cost: float) -> float:
        eta = self.hp.lambda_lr / (1.0 + self.n_updates / self.hp.lambda_tau)
        return float(np.clip(lam + eta * (avg_cost - self.spec.budget) / max(self.spec.budget, 1e-8),
                             0.0, self.hp.lambda_max))

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "vr": self.vr.state_dict(), "vc": self.vc.state_dict(),
                "obs_rms": self.obs_rms.state_dict()}
