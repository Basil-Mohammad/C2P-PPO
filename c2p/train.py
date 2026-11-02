"""Single-run trainer.

Usage:
  python -m c2p.train --env cartpole --method c2p_ppo --seed 0 --out results/
  python -m c2p.train --env arm --method c2p_ppo --seed 0 --set beta=5 --tag beta5

Outputs (in <out>/<env>/<method>[_<tag>]/seed<k>/):
  episodes.csv  one row per training episode
  updates.csv   one row per policy update (diagnostics, lambda, alpha, ...)
  evals.csv     periodic deterministic evaluation on fixed initial conditions
  summary.json  final metrics and run metadata
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from c2p.buffer import RolloutBuffer
from c2p.registry import ENVS, make_env, make_hyper, method_class


def _parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def evaluate(agent, env, seeds, shield=True):
    rets, costs, succ, lens = [], [], [], []
    saved_u = getattr(agent, "u", None)
    for sd in seeds:
        obs = env.reset(seed=int(sd))
        agent.reset_episode()
        R = C = G = L = 0
        done = False
        while not done:
            on = agent.norm(agent.augment_obs(obs))
            if hasattr(agent, "use_shield") and not shield:
                a = agent.act_eval(on, env, shield=False)
            else:
                a = agent.act_eval(on, env)
            obs, r, rho, term, trunc, info = env.step(a)
            agent.track(rho)
            R += r
            C += env.spec.ctl_cost(rho)
            G += int(info.get("goal_met", False))
            L += 1
            done = term or trunc
        rets.append(R); costs.append(C); succ.append(float(G > 0)); lens.append(L)
    if saved_u is not None:
        agent.u = saved_u
    costs, lens = np.asarray(costs, float), np.asarray(lens, float)
    return {"eval_return": float(np.mean(rets)), "eval_cost": float(np.mean(costs)),
            "eval_viol_rate": float(np.mean(costs / lens)), "eval_success": float(np.mean(succ)),
            "eval_safe_rate": float(np.mean(costs == 0)), "eval_len": float(np.mean(lens))}


def train(env_name, method, seed, out, total_steps=None, overrides=None, tag="", eval_episodes=10, device="cpu"):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    cfg = ENVS[env_name]
    total_steps = int(total_steps or cfg["total_steps"])
    env, eval_env = make_env(env_name), make_env(env_name)
    spec = env.spec
    hp = make_hyper(env_name, method, overrides)
    Agent = method_class(method)
    agent = Agent(env.obs_dim, env.act_dim, spec, hp, device)
    run_name = method + (f"_{tag}" if tag else "")
    odir = os.path.join(out, env_name, run_name, f"seed{seed}")
    os.makedirs(odir, exist_ok=True)

    obs_dim_eff = agent.obs_dim
    buf = RolloutBuffer(hp.steps_per_update, obs_dim_eff, env.act_dim, spec.m)
    eval_seeds = [10_000 + i for i in range(eval_episodes)]   # identical across methods and seeds

    ep_rows, up_rows, ev_rows = [], [], []
    obs = env.reset(seed=int(rng.integers(2 ** 31)))
    agent.reset_episode()
    rho_prev = env.robustness()
    ep = dict(R=0.0, L=0, V=0, G=0, I=0, FB=0, P=np.zeros(spec.m))
    n_ep, t0, next_eval = 0, time.time(), 0

    for step in range(1, total_steps + 1):
        obs_a = agent.augment_obs(obs)
        agent.obs_rms.update(obs_a)
        on = agent.norm(obs_a)
        out_a = agent.act(on, env)
        obs2, r, rho, term, trunc, info = env.step(out_a["exec"])
        agent.track(rho)
        cost = spec.ctl_cost(rho)
        r_learn = agent.shape_reward(r, rho_prev, rho, term)
        c_learn = cost
        cvec = spec.cost_vector(rho)
        cvec_learn = cvec.copy()
        if trunc and not term:  # time-limit: bootstrap (partial-episode bootstrapping)
            on2 = agent.norm(agent.augment_obs(obs2))
            v2, vc2 = agent.values(on2)
            r_learn += hp.gamma * v2
            c_learn += hp.gamma * vc2
            if hasattr(agent, "values_vec"):
                cvec_learn = cvec + hp.gamma * agent.values_vec(on2)
        buf.add(obs=on, act=out_a["act"], logp=out_a["logp"], rew=r_learn, cost=cost,
                cost_vec=cvec, cost_learn=c_learn, cost_vec_learn=cvec_learn,
                cval_vec=out_a.get("vc_vec", np.zeros(spec.m, np.float32)), val=out_a["v"], cval=out_a["vc"],
                done=float(term or trunc), term=float(term), is_w=out_a["is_w"], mask=out_a["mask"],
                ep_id=n_ep, raw_obs=obs_a, exec_act=out_a["exec"], rho_prev=rho_prev, rho_next=rho,
                **({k: out_a[k] for k in ("cand", "cand_logp", "cand_inadm")} if "cand" in out_a else {}))
        ep["R"] += r; ep["L"] += 1; ep["V"] += int(cost); ep["G"] += int(info.get("goal_met", False))
        ep["I"] += out_a.get("intervened", 0.0); ep["FB"] += out_a.get("fallback", 0.0)
        ep["P"] += spec.cost_vector(rho)
        obs, rho_prev = obs2, rho

        if term or trunc:
            vr = ep["V"] / ep["L"]
            row = {"episode": n_ep, "step": step, "return": ep["R"], "length": ep["L"], "violations": ep["V"],
                   "viol_rate": vr, "any_violation": int(ep["V"] > 0), "goals": ep["G"], "success": int(ep["G"] > 0),
                   "intervention_rate": ep["I"] / ep["L"], "fallback_rate": ep["FB"] / ep["L"],
                   "time": time.time() - t0}
            for j, nm in enumerate(spec.predicates):
                row[f"viol_p{j}"] = int(ep["P"][j])
            ep_rows.append(row)
            agent.on_episode_end(vr)
            n_ep += 1
            obs = env.reset(seed=int(rng.integers(2 ** 31)))
            agent.reset_episode()
            rho_prev = env.robustness()
            ep = dict(R=0.0, L=0, V=0, G=0, I=0, FB=0, P=np.zeros(spec.m))

        if buf.full:
            lv, lvc = agent.values(agent.norm(agent.augment_obs(obs)))
            if hasattr(agent, "values_vec"):
                buf.last_cvec[:] = agent.values_vec(agent.norm(agent.augment_obs(obs)))
            if buf.done[buf.ptr - 1]:
                lv = lvc = 0.0
                buf.last_cvec[:] = 0.0
            batch_cost = float(buf.cost[:buf.ptr].mean())
            st = agent.update(buf, lv, lvc)
            st.update({"update": agent.n_updates, "step": step, "batch_cost": batch_cost, "time": time.time() - t0})
            up_rows.append(st)
            buf.reset()

        if step >= next_eval:
            ev = {"step": step, **evaluate(agent, eval_env, eval_seeds)}
            if hasattr(agent, "use_shield"):
                ev.update({k + "_noshield": v for k, v in evaluate(agent, eval_env, eval_seeds, shield=False).items()})
            ev_rows.append(ev)
            next_eval += cfg["eval_every"]
            pd.DataFrame(ep_rows).to_csv(os.path.join(odir, "episodes.csv"), index=False)
            pd.DataFrame(up_rows).to_csv(os.path.join(odir, "updates.csv"), index=False)
            pd.DataFrame(ev_rows).to_csv(os.path.join(odir, "evals.csv"), index=False)

    ev = {"step": total_steps, **evaluate(agent, eval_env, eval_seeds)}
    if hasattr(agent, "use_shield"):
        ev.update({k + "_noshield": v for k, v in evaluate(agent, eval_env, eval_seeds, shield=False).items()})
    ev_rows.append(ev)
    E, U, V = pd.DataFrame(ep_rows), pd.DataFrame(up_rows), pd.DataFrame(ev_rows)
    E.to_csv(os.path.join(odir, "episodes.csv"), index=False)
    U.to_csv(os.path.join(odir, "updates.csv"), index=False)
    V.to_csv(os.path.join(odir, "evals.csv"), index=False)
    summary = {"env": env_name, "method": method, "tag": tag, "seed": seed, "total_steps": total_steps,
               "episodes": n_ep, "wall_time_s": time.time() - t0, "budget": spec.budget,
               "formula": spec.formula, "hyper": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(hp).items()},
               "platform": platform.platform(), "torch": torch.__version__,
               "cum_violations": int(E["violations"].sum()), "cost_rate": float(E["violations"].sum() / E["length"].sum()),
               "final_eval": V.iloc[-1].to_dict()}
    with open(os.path.join(odir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    torch.save(agent.state_dict(), os.path.join(odir, "final_policy.pt"))
    env.close(); eval_env.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s = train(a.env, a.method, a.seed, a.out, a.steps, _parse_overrides(a.set), a.tag)
    print(json.dumps({k: s[k] for k in ("env", "method", "seed", "cum_violations", "cost_rate", "wall_time_s")}))


if __name__ == "__main__":
    main()
