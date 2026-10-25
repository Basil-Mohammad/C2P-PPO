"""C2P-PPO: CTL-Constrained PPO with Adaptive Shielding.

Behaviour policy (Section 6.2):
    pi_tilde(a|s) = pi_old(a|s) w_t(s,a) / Z_t(s),
    w_t(s,a)      = 1[Safe(s,a) >= tau_safe] * (alpha_t Safe(s,a) + (1 - alpha_t) Explore(s,a)),
realised exactly in the limit M -> infinity by sampling-importance-resampling
(SIR) over M candidates drawn from pi_old. If no candidate is admissible the
shield falls back to the safest candidate and the transition is masked.

Policy update (Section 6.4):
    L(theta) = E_{pi_tilde}[ c_t m_t min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) ],
    r_t = pi_theta / pi_old,   c_t = min(pi_old / pi_tilde, c_bar) = min(Z/w, c_bar),
    A_t = (A^R_t - lambda_t A^C_t) / (1 + lambda_t),
so the trust region is still centred on pi_old while the fixed weights c_t
correct the sampling distribution of the shielded data.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from c2p.algos.base import OnPolicyAgent
from c2p.algos.safety_models import SafetyCriticTrainer
from c2p.nets import RND


def _std(x):
    return (x - x.mean()) / (x.std() + 1e-8)


class C2PPPO(OnPolicyAgent):
    name = "C2P-PPO"

    def __init__(self, obs_dim, act_dim, spec, hp, device="cpu"):
        super().__init__(obs_dim, act_dim, spec, hp, device)
        f = hp.flags
        self.use_lagrangian = not f.get("no_lagrangian", False)
        self.use_shield = not f.get("no_shield", False)
        self.adaptive_alpha = not f.get("fixed_alpha", False)
        self.use_rnd = not f.get("no_rnd", False)
        self.use_is = not f.get("no_is", False)
        self.use_shaping = not f.get("no_shaping", False)
        self.use_inadm_penalty = not f.get("no_inadm_penalty", False)
        self.mixture = f.get("mixture", "linear")         # 'linear' (ours) | 'geometric'
        self.lam = hp.lambda_init if self.use_lagrangian else 0.0
        self.alpha = hp.alpha0
        self.c_threshold = hp.c_threshold_scale * spec.budget
        self.viol_window = deque(maxlen=hp.window)
        self.sc = SafetyCriticTrainer(obs_dim, act_dim, hp, self.device)
        self.rnd = RND(obs_dim, act_dim).to(self.device)
        self.rnd_opt = torch.optim.Adam(self.rnd.pred.parameters(), lr=hp.rnd_lr)

    # ----------------------------------------------------------- adaptive alpha
    def on_episode_end(self, ep_violation_rate):
        self.viol_window.append(ep_violation_rate)
        if not self.adaptive_alpha:
            return
        c = float(np.mean(self.viol_window))
        x = self.hp.beta * (c - self.c_threshold) / max(self.c_threshold, 1e-8) \
            + np.log(self.hp.alpha0 / (1 - self.hp.alpha0))
        self.alpha = float(1.0 / (1.0 + np.exp(-x)))

    # ----------------------------------------------------------- shaping
    def shape_reward(self, r, rho_prev, rho_next, terminated):
        if not self.use_shaping:
            return r
        phi0 = self.spec.potential(rho_prev)
        phi1 = 0.0 if terminated else self.spec.potential(rho_next)
        return r + self.hp.lambda_r * (self.hp.gamma * phi1 - phi0)

    # ----------------------------------------------------------- shield
    @property
    def shield_on(self):
        return self.use_shield and self.sc.trained and self.n_updates >= self.hp.shield_warmup_updates

    def _weights(self, safe, explore, hard: bool = True):
        a = self.alpha
        if self.mixture == "geometric":
            mix = safe.clamp_min(1e-6) ** a * explore.clamp_min(1e-3) ** (1 - a)
        else:
            mix = a * safe + (1 - a) * explore
        return (safe >= self.hp.tau_safe).float() * mix if hard else mix

    @torch.no_grad()
    def act(self, obs_n, env=None):
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        d = self.actor.dist(o)
        v, vc = float(self.vr(o)), float(self.vc(o))
        M = self.hp.n_candidates
        if not self.shield_on:
            a = d.sample()[0]
            return {"act": a.cpu().numpy(), "exec": a.clamp(-1, 1).cpu().numpy(),
                    "logp": float(d.log_prob(a.unsqueeze(0)).sum(-1)), "v": v, "vc": vc,
                    "is_w": 1.0, "mask": 1.0, "intervened": 0.0, "fallback": 0.0,
                    "cand": np.zeros((M, self.act_dim), np.float32), "cand_logp": np.zeros(M, np.float32),
                    "cand_inadm": np.zeros(M, np.float32)}
        cand = d.sample((M,)).squeeze(1)                              # M x act, i.i.d. from pi_old
        ce = cand.clamp(-1, 1)
        oo = o.expand(M, -1)
        safe = self.sc.prob(oo, ce)
        explore = self.rnd.bonus(oo, ce) if self.use_rnd else torch.zeros_like(safe)
        w = self._weights(safe, explore)
        intervened = float(safe[0] < self.hp.tau_safe)                # raw pi_old sample would be blocked
        fb = 0.0
        if float(w.sum()) <= 0:
            # No admissible candidate (typically the state already violates phi):
            # soft mode, i.e. the same tilting without the hard admissibility filter.
            w = self._weights(safe, explore, hard=False)
            fb = 1.0
        if float(w.sum()) > 0:
            j = int(torch.multinomial(w / w.sum(), 1))
            c = float(w.mean() / w[j]) if self.use_is else 1.0         # pi_old/pi_tilde = Z/w (SIR estimate)
            c, mask = min(c, self.hp.is_clip), 1.0
        else:                                                         # degenerate: all weights vanish
            j = int(torch.argmax(safe))
            c, mask = 1.0, 0.0
        a = cand[j]
        # inadmissible candidates (not in soft mode): virtual samples for the inadmissible-mass penalty
        inadm = (safe < self.hp.tau_safe).float() * (1.0 - fb)
        return {"act": a.cpu().numpy(), "exec": ce[j].cpu().numpy(),
                "logp": float(d.log_prob(a.unsqueeze(0)).sum(-1)), "v": v, "vc": vc,
                "is_w": c, "mask": mask, "intervened": intervened, "fallback": fb,
                "cand": cand.cpu().numpy(), "cand_logp": d.log_prob(cand.unsqueeze(1)).sum(-1)[:, 0].cpu().numpy(),
                "cand_inadm": inadm.cpu().numpy()}

    @torch.no_grad()
    def act_eval(self, obs_n, env=None, shield: bool = True):
        """Deployment: mean action, shielded by the safety critic (alpha = 1)."""
        o = torch.as_tensor(obs_n, device=self.device).unsqueeze(0)
        d = self.actor.dist(o)
        a = d.mean[0].clamp(-1, 1)
        if shield and self.shield_on and float(self.sc.prob(o, a.unsqueeze(0))) < self.hp.tau_safe:
            cand = d.sample((self.hp.n_candidates,)).squeeze(1).clamp(-1, 1)
            a = cand[int(torch.argmax(self.sc.prob(o.expand(cand.shape[0], -1), cand)))]
        return a.cpu().numpy()

    # ----------------------------------------------------------- learning
    def _virtual(self, buf, obs_t):
        """Inadmissible candidates as virtual samples; coefficient chosen so that
        the penalty estimates kappa * E_s[pi_theta(A \\ Adm(s) | s)]."""
        if not (self.use_inadm_penalty and self.use_shield and hasattr(buf, "cand_inadm")):
            return None
        n, M = buf.ptr, self.hp.n_candidates
        t_idx, i_idx = np.nonzero(buf.cand_inadm[:n] > 0)
        if len(t_idx) == 0:
            return None
        ov = obs_t[torch.as_tensor(t_idx, device=self.device)]
        av = torch.as_tensor(buf.cand[t_idx, i_idx], device=self.device)
        lv = torch.as_tensor(buf.cand_logp[t_idx, i_idx], device=self.device)
        coef = self.hp.kappa_inadm * len(t_idx) / (n * M)
        self._inadm_frac = len(t_idx) / (n * M)
        return ov, av, lv, coef

    def _train_rnd(self, obs_t, exec_act):
        """RND predictor regression on executed state-action pairs."""
        n = obs_t.shape[0]
        x_a = torch.as_tensor(exec_act, device=self.device)
        with torch.no_grad():
            self.rnd.update_stats(self.rnd.error(obs_t, x_a))
        perm = torch.randperm(n)
        for s in range(0, n, 512):
            i = perm[s:s + 512]
            loss = self.rnd.error(obs_t[i], x_a[i]).mean()
            self.rnd_opt.zero_grad()
            loss.backward()
            self.rnd_opt.step()

    def update(self, buf, last_v, last_vc):
        n = buf.ptr
        o_lab, a_lab, y_lab = self.sc.add(buf.raw_obs[:n], buf.exec_act[:n], buf.cost[:n], buf.done[:n])
        st = self.sc.evaluate(self.norm(o_lab), a_lab, y_lab)

        adv, ret, cadv, cret = self.advantages(buf, last_v, last_vc)
        T = buf.tensors(self.device)
        avg_cost = float(T["cost"].mean())
        if self.use_lagrangian:
            self.lam = self.lagrange_step(self.lam, avg_cost)
        adv, cadv, ret, cret = (torch.as_tensor(x, device=self.device) for x in (adv, cadv, ret, cret))
        mask = T["mask"]
        a_comb = (_std(adv) - self.lam * (cadv - cadv.mean())) / (1.0 + self.lam)
        weight = T["is_w"] * mask
        st.update(self.ppo_policy_update(T["obs"], T["act"], T["logp"], a_comb, weight=weight,
                                         virtual=self._virtual(buf, T["obs"])))
        self.update_critics(T["obs"], ret, cret)

        self._train_rnd(T["obs"], buf.exec_act[:n])

        st.update(self.sc.train(self.norm))
        self.n_updates += 1
        w = T["is_w"][mask > 0]
        st.update({"lambda": self.lam, "alpha": self.alpha,
                   "is_w_mean": float(w.mean()) if len(w) else 1.0,
                   "is_w_trunc": float((w >= self.hp.is_clip - 1e-6).float().mean()) if len(w) else 0.0,
                   "ess_frac": float(w.sum() ** 2 / (w.pow(2).sum() + 1e-8) / max(len(w), 1)) if len(w) else 1.0,
                   "mask_frac": float((mask == 0).float().mean()),
                   "inadm_mass_old": float(getattr(self, "_inadm_frac", 0.0))})
        return st


class C2PPPOMulti(C2PPPO):
    """C2P-PPO with one cost critic and one Lagrange multiplier per atomic proposition.

    phi = AG(p_1 & ... & p_m) is enforced through the m constraints
    E[1(s' violates p_j)] <= delta / m. By the union bound, their joint
    satisfaction implies E[CTLCost] <= delta for the conjunction.
    """
    name = "C2P-PPO (multi-lambda)"

    def __init__(self, obs_dim, act_dim, spec, hp, device="cpu"):
        super().__init__(obs_dim, act_dim, spec, hp, device)
        from c2p.nets import mlp
        self.vcm = mlp(obs_dim, spec.m, hp.hidden).to(self.device)
        self.opt_vcm = torch.optim.Adam(self.vcm.parameters(), lr=hp.lr_critic)
        self.lams = np.full(spec.m, hp.lambda_init if self.use_lagrangian else 0.0)
        self.budget_j = spec.budget / spec.m

    @torch.no_grad()
    def values_vec(self, obs_n):
        return self.vcm(torch.as_tensor(obs_n, device=self.device).unsqueeze(0))[0].cpu().numpy()

    @torch.no_grad()
    def act(self, obs_n, env=None):
        out = super().act(obs_n, env)
        out["vc_vec"] = self.values_vec(obs_n)
        return out

    def update(self, buf, last_v, last_vc):
        n = buf.ptr
        o_lab, a_lab, y_lab = self.sc.add(buf.raw_obs[:n], buf.exec_act[:n], buf.cost[:n], buf.done[:n])
        st = self.sc.evaluate(self.norm(o_lab), a_lab, y_lab)
        adv, ret, _, _ = self.advantages(buf, last_v, last_vc)
        cadvs, crets = [], []
        for j in range(self.spec.m):
            a_j, r_j = buf.gae(buf.cost_vec_learn[:n, j], buf.cval_vec[:n, j], float(buf.last_cvec[j]),
                               buf.done[:n], buf.term[:n], self.hp.gamma, self.hp.lam)
            cadvs.append(a_j); crets.append(r_j)
            if self.use_lagrangian:
                eta = self.hp.lambda_lr / (1.0 + self.n_updates / self.hp.lambda_tau)
                g = (float(buf.cost_vec[:n, j].mean()) - self.budget_j) / self.budget_j
                self.lams[j] = float(np.clip(self.lams[j] + eta * g, 0.0, self.hp.lambda_max))
        T = buf.tensors(self.device)
        adv = torch.as_tensor(adv, device=self.device)
        ret = torch.as_tensor(ret, device=self.device)
        cadv = torch.as_tensor(np.stack(cadvs, 1), device=self.device)
        cret = torch.as_tensor(np.stack(crets, 1), device=self.device)
        lam = torch.as_tensor(self.lams, device=self.device, dtype=torch.float32)
        a_comb = (_std(adv) - ((cadv - cadv.mean(0)) * lam).sum(1)) / (1.0 + lam.sum())
        st.update(self.ppo_policy_update(T["obs"], T["act"], T["logp"], a_comb, weight=T["is_w"] * T["mask"],
                                         virtual=self._virtual(buf, T["obs"])))
        # critics: reward critic (base) + per-predicate cost critic
        self.update_critics(T["obs"], ret, torch.as_tensor(buf.gae(buf.cost_learn[:n], buf.cval[:n], last_vc,
                            buf.done[:n], buf.term[:n], self.hp.gamma, self.hp.lam)[1], device=self.device))
        for _ in range(self.hp.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.hp.minibatch):
                i = perm[s:s + self.hp.minibatch]
                loss = (self.vcm(T["obs"][i]) - cret[i]).pow(2).mean()
                self.opt_vcm.zero_grad(); loss.backward(); self.opt_vcm.step()
        self._train_rnd(T["obs"], buf.exec_act[:n])
        st.update(self.sc.train(self.norm))
        self.n_updates += 1
        self.lam = float(self.lams.sum())
        st.update({f"lambda_{j}": float(l) for j, l in enumerate(self.lams)})
        st.update({"lambda": self.lam, "alpha": self.alpha})
        return st
