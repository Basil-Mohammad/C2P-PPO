"""Environment / method registry and per-environment hyperparameters.

All methods share the same network sizes, PPO-family hyperparameters,
rollout length and step budget within an environment.
"""
from __future__ import annotations

from dataclasses import replace

from c2p.algos.base import Hyper
from c2p.algos.c2p_ppo import C2PPPO, C2PPPOMulti
from c2p.algos.focops import FOCOPS
from c2p.algos.ppo_family import PPO, LCRLPPO, PPOLagrangian
from c2p.algos.shields import SafetyLayerPPO, ShieldedPPO
from c2p.algos.trust_region import CPO, PCPO

METHODS = {
    "ppo": PPO, "ppo_lag": PPOLagrangian, "cpo": CPO, "pcpo": PCPO, "focops": FOCOPS,
    "shielded_ppo": ShieldedPPO, "safety_layer": SafetyLayerPPO, "lcrl_ppo": LCRLPPO, "c2p_ppo": C2PPPO, "c2p_multi": C2PPPOMulti,
}

ABLATIONS = {
    "c2p_no_lagrangian": {"no_lagrangian": True},
    "c2p_no_shield": {"no_shield": True},
    "c2p_fixed_alpha": {"fixed_alpha": True},
    "c2p_no_rnd": {"no_rnd": True},
    "c2p_no_is": {"no_is": True},
    "c2p_no_shaping": {"no_shaping": True},
    "c2p_geometric": {"mixture": "geometric"},
    "c2p_no_inadm_penalty": {"no_inadm_penalty": True},
}

# env name -> (constructor kwargs, total steps, Hyper overrides)
ENVS = {
    "cartpole": dict(total_steps=1_000_000, eval_every=50_000,
                     hp=dict(steps_per_update=5000, lambda_r=1.0, lcrl_eta=1.0)),
    "pointgoal": dict(total_steps=2_000_000, eval_every=200_000,
                      hp=dict(steps_per_update=20000, lambda_r=0.1, lcrl_eta=0.05, minibatch=1024)),
    "cargoal": dict(total_steps=2_000_000, eval_every=200_000,
                    hp=dict(steps_per_update=20000, lambda_r=0.1, lcrl_eta=0.05, minibatch=1024)),
    "arm": dict(total_steps=1_000_000, eval_every=50_000,
                hp=dict(steps_per_update=8000, lambda_r=1.0, lcrl_eta=0.5, minibatch=512)),
}


def make_env(name: str):
    if name == "cartpole":
        from c2p.envs.cartpole import CartPoleTrack
        return CartPoleTrack()
    if name == "pointgoal":
        from c2p.envs.safety_nav import SafetyNav
        return SafetyNav("SafetyPointGoal1-v0")
    if name == "cargoal":
        from c2p.envs.safety_nav import SafetyNav
        return SafetyNav("SafetyCarGoal1-v0")
    if name == "arm":
        from c2p.envs.arm import SafeArmReach
        return SafeArmReach()
    raise KeyError(name)


def make_hyper(env_name: str, method: str, overrides: dict | None = None) -> Hyper:
    hp = replace(Hyper(), **ENVS[env_name]["hp"])
    flags = dict(ABLATIONS.get(method, {}))
    hp = replace(hp, flags=flags)
    if overrides:
        hp = replace(hp, **overrides)
    return hp


def method_class(method: str):
    if method in ABLATIONS:
        return C2PPPO
    return METHODS[method]
