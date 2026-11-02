"""Learned safety models shared by shield-based agents.

SafetyCriticTrainer: Level-3 surrogate trained on Level-2 k-step Monte-Carlo
labels harvested from the agent's own trajectories (Section 5).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from c2p.ctl import k_step_safety_labels
from c2p.nets import SafetyCritic


class SafetyCriticTrainer:
    def __init__(self, obs_dim, act_dim, hp, device):
        self.hp, self.device = hp, device
        self.net = SafetyCritic(obs_dim, act_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=hp.safety_lr)
        cap = hp.safety_buffer
        self.raw_obs = np.zeros((cap, obs_dim), np.float32)
        self.act = np.zeros((cap, act_dim), np.float32)
        self.y = np.zeros(cap, np.float32)
        self.n, self.ptr, self.cap = 0, 0, cap
        self.trained = False

    @torch.no_grad()
    def prob(self, obs_n: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(obs_n, act)

    def add(self, raw_obs, act_exec, cost, done):
        y, valid = k_step_safety_labels(cost, done, self.hp.k_lookahead)
        idx = np.where(valid > 0)[0]
        for j in idx:
            self.raw_obs[self.ptr] = raw_obs[j]
            self.act[self.ptr] = act_exec[j]
            self.y[self.ptr] = y[j]
            self.ptr = (self.ptr + 1) % self.cap
            self.n = min(self.n + 1, self.cap)
        return raw_obs[idx], act_exec[idx], y[idx]

    @torch.no_grad()
    def evaluate(self, obs_n, act, y) -> dict:
        """Out-of-sample diagnostics on freshly collected data (before training on it)."""
        if not self.trained or len(y) == 0:
            return {}
        p = self.net(torch.as_tensor(obs_n, device=self.device), torch.as_tensor(act, device=self.device)).cpu().numpy()
        pred = (p >= self.hp.tau_safe).astype(np.float32)
        unsafe = y < 0.5
        out = {"sc_acc": float((pred == y).mean()),
               "sc_brier": float(np.mean((p - y) ** 2)),
               "sc_unsafe_frac": float(unsafe.mean())}
        if unsafe.any():
            out["sc_unsafe_recall"] = float((pred[unsafe] == 0).mean())
        if (~unsafe).any():
            out["sc_safe_recall"] = float((pred[~unsafe] == 1).mean())
        return out

    def train(self, normalize) -> dict:
        if self.n < 1000:
            return {}
        n = self.n
        obs = torch.as_tensor(normalize(self.raw_obs[:n]), device=self.device)
        act = torch.as_tensor(self.act[:n], device=self.device)
        y = torch.as_tensor(self.y[:n], device=self.device)
        n_unsafe = float((y < 0.5).sum())
        w_unsafe = float(np.clip((n - n_unsafe) / max(n_unsafe, 1.0), 1.0, 10.0))
        losses = []
        for _ in range(self.hp.safety_epochs):
            perm = torch.randperm(n)
            for s in range(0, min(n, 50_000), 512):
                i = perm[s:s + 512]
                logit = self.net.logits(obs[i], act[i])
                wt = torch.where(y[i] < 0.5, torch.full_like(y[i], w_unsafe), torch.ones_like(y[i]))
                loss = (wt * F.binary_cross_entropy_with_logits(logit, y[i], reduction="none")).mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                losses.append(loss.item())
        self.trained = True
        return {"sc_loss": float(np.mean(losses))}


class LinearSafetyModel(nn.Module):
    """Safety layer (Dalal et al., 2018): rho_j(s') ~ rho_j(s) + g_j(s)^T a."""

    def __init__(self, obs_dim, act_dim, m):
        super().__init__()
        from c2p.nets import mlp
        self.m, self.act_dim = m, act_dim
        self.g = mlp(obs_dim, m * act_dim, (64, 64), act=nn.ReLU)

    def forward(self, obs):
        return self.g(obs).view(-1, self.m, self.act_dim)
