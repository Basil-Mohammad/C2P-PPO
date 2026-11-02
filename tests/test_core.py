import numpy as np
import pytest
import torch

from c2p.buffer import RolloutBuffer
from c2p.ctl import SafetySpec, k_step_safety_labels
from c2p.registry import make_hyper, method_class


class _Spec(SafetySpec):
    pass


def _agent(flags=None, M=4096, **hp_over):
    spec = SafetySpec(predicates=["p"], budget=0.05)
    hp = make_hyper("cartpole", "c2p_ppo", dict(n_candidates=M, **hp_over))
    if flags:
        hp.flags = flags
    ag = method_class("c2p_ppo")(3, 1, spec, hp)
    ag.sc.trained = True
    ag.n_updates = 10
    return ag


def test_k_step_labels():
    cost = np.array([0, 0, 1, 0, 0, 0, 0], np.float32)
    done = np.array([0, 0, 0, 0, 1, 0, 0], np.float32)
    y, v = k_step_safety_labels(cost, done, k=2)
    # t=0 window {0,1}: safe; t=1 window {1,2}: unsafe; t=2 itself unsafe
    assert y.tolist()[:5] == [1, 0, 0, 1, 1]
    # t=5: window {5,6} complete -> valid; t=6: window cut by buffer end -> invalid
    assert v[5] == 1 and v[6] == 0


def test_gae_matches_naive():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=6).astype(np.float32), rng.normal(size=6).astype(np.float32)
    done = np.array([0, 0, 1, 0, 0, 0], np.float32)
    adv, _ = RolloutBuffer.gae(r, v, 0.7, done, done, 0.9, 1.0)
    # with lambda = 1, advantage = discounted return - V
    ret0 = r[0] + 0.9 * r[1] + 0.81 * r[2]
    assert adv[0] == pytest.approx(ret0 - v[0], rel=1e-5)
    ret3 = r[3] + 0.9 * r[4] + 0.81 * r[5] + 0.9 ** 3 * 0.7
    assert adv[3] == pytest.approx(ret3 - v[3], rel=1e-5)


def test_reduction_to_ppo_when_everything_safe():
    """Safe = 1, alpha = 1, no RND: pi_tilde = pi_old and c_t = 1 exactly."""
    ag = _agent(flags={"no_rnd": True}, M=16)
    ag.alpha = 1.0
    ag.sc.prob = lambda o, a: torch.ones(a.shape[0])
    obs = np.zeros(3, np.float32)
    ws = [ag.act(obs)["is_w"] for _ in range(200)]
    assert np.allclose(ws, 1.0)


def test_sir_importance_weights_are_unbiased():
    """E_{pi_tilde}[c f(a)] = E_{pi_old}[f(a)] when weights are not truncated."""
    torch.manual_seed(0)
    ag = _agent(flags={"no_rnd": True}, M=64, is_clip=1e9, tau_safe=0.0)
    ag.alpha = 1.0
    # smooth safety score that strongly prefers negative actions
    ag.sc.prob = lambda o, a: torch.sigmoid(-3.0 * a[:, 0])
    obs = np.zeros(3, np.float32)
    acts, cs = [], []
    for _ in range(4000):
        out = ag.act(obs)
        acts.append(out["exec"][0]); cs.append(out["is_w"])
    acts, cs = np.array(acts), np.array(cs)
    with torch.no_grad():
        mu = float(ag.actor.dist(torch.zeros(1, 3)).mean[0, 0])
    # behaviour mean is shifted towards safe (negative) actions ...
    assert acts.mean() < mu - 0.05
    # ... and the self-normalised IS estimate recovers the pi_old mean
    est = np.sum(cs * acts) / np.sum(cs)
    assert abs(est - mu) < 0.05


def test_fallback_soft_mode_and_mask():
    # nothing admissible but some safety mass: soft SIR, transition kept
    ag = _agent(flags={"no_rnd": True}, M=8)
    ag.sc.prob = lambda o, a: torch.full((a.shape[0],), 0.1)
    out = ag.act(np.zeros(3, np.float32))
    assert out["fallback"] == 1.0 and out["mask"] == 1.0 and out["is_w"] == pytest.approx(1.0)
    # all weights vanish: degenerate case is masked
    ag.sc.prob = lambda o, a: torch.zeros(a.shape[0])
    out = ag.act(np.zeros(3, np.float32))
    assert out["mask"] == 0.0


def test_alpha_controller():
    ag = _agent()
    for _ in range(10):
        ag.on_episode_end(ag.c_threshold)
    assert ag.alpha == pytest.approx(ag.hp.alpha0, abs=1e-6)       # balance -> alpha0
    for _ in range(10):
        ag.on_episode_end(10 * ag.c_threshold)
    assert ag.alpha > 0.99                                           # many violations -> shield
    for _ in range(10):
        ag.on_episode_end(0.0)
    assert ag.alpha < ag.hp.alpha0                                   # no violations -> exploration


def test_lagrange_projection():
    ag = _agent()
    assert ag.lagrange_step(0.0, 0.0) == 0.0
    assert ag.lagrange_step(0.0, 1.0) > 0.0


def test_potential_shaping_is_zero_sum_on_cycle():
    ag = _agent()
    rho = [np.array([0.3]), np.array([0.7]), np.array([0.3])]
    g = ag.hp.gamma
    s = ag.shape_reward(0.0, rho[0], rho[1], False) + g * ag.shape_reward(0.0, rho[1], rho[2], False)
    # sum_t gamma^t F_t = gamma^2 Phi(s_2) - Phi(s_0)
    assert s == pytest.approx(ag.hp.lambda_r * (g * g * 0.3 - 0.3), rel=1e-6)


def test_blocked_action_penalty_reduces_inadmissible_probability():
    torch.manual_seed(0)
    ag = _agent(M=8)
    obs = torch.zeros(64, 3)
    act = ag.actor.dist(obs).sample()
    logp = ag.actor.log_prob(obs, act).detach()
    # virtual samples: positive actions are "inadmissible"
    av = torch.full((64, 1), 0.8)
    lv = ag.actor.log_prob(obs, av).detach()
    before = float(lv.mean())
    ag.ppo_policy_update(obs, act, logp, torch.zeros(64), virtual=(obs, av, lv, 1.0))
    after = float(ag.actor.log_prob(obs, av).mean())
    assert after < before
    # and the trust region is respected: probability ratio not far below 1 - eps per update
    assert np.exp(after - before) > 0.5
