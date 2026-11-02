"""SafeArmReach-6DoF: a six-joint manipulator reaching task with spatial safety constraints.

The kinematic chain follows the Denavit--Hartenberg parameters of the
Universal Robots UR5e (d1 = 0.1625 m, a2 = 0.425 m, a3 = 0.3922 m,
d4 = 0.1333 m, d5 = 0.0997 m, d6 = 0.0996 m). The arm is simulated in MuJoCo
(Todorov et al., 2012) with joint position servos and internal gravity
compensation, as provided by industrial controllers; the policy commands
joint velocities in [-1, 1]^6, scaled to at most 1 rad/s per joint.

Task: move the tool centre point (TCP) from a home pose on one side of the
workspace to a randomly sampled target on the other side. A spherical
keep-out zone (e.g. a human co-worker's protected volume or a fragile
fixture) is placed, with random jitter, on the straight-line path, so the
reward-greedy motion is unsafe and the agent must learn a detour.

Safety specification with two atomic propositions:
    phi = AG( clear_keepout & clear_table )
  clear_keepout: every monitored point of the arm is outside the keep-out sphere
                 (inflated by the link radius);
  clear_table:   every monitored point of the forearm and wrist is at least
                 3 cm above the table surface.
"""
from __future__ import annotations

import numpy as np

from c2p.ctl import SafetySpec
from c2p.envs.base import SafeEnv

_XML = """
<mujoco model="safe_arm_6dof">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <default>
    <joint damping="40" armature="0.2"/>
    <geom contype="1" conaffinity="2" density="800" rgba="0.75 0.78 0.82 1"/>
    <position kp="600" forcerange="-150 150"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="table" type="plane" size="1.5 1.5 0.05" contype="2" conaffinity="1" rgba="0.35 0.32 0.30 1"/>
    <body name="keepout" mocap="true" pos="0.45 0 0.3">
      <geom name="keepout_g" type="sphere" size="0.13" contype="0" conaffinity="0" rgba="0.9 0.2 0.2 0.35"/>
    </body>
    <body name="target" mocap="true" pos="0.45 0.35 0.2">
      <site name="target_s" type="sphere" size="0.03" rgba="0.2 0.8 0.2 0.8"/>
    </body>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.07 0.05" pos="0 0 0.05" contype="0" conaffinity="0"/>
      <body name="shoulder" pos="0 0 0.1625">
        <joint name="j1" axis="0 0 1" range="-3.1416 3.1416"/>
        <geom type="cylinder" size="0.06 0.07" contype="0" conaffinity="0"/>
        <body name="upper_arm" pos="0 0.138 0">
          <joint name="j2" axis="0 1 0" range="-3.1416 3.1416"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.425" size="0.05" contype="0"/>
          <site name="m_ua" pos="0 0 0.2125" size="0.01"/>
          <body name="forearm" pos="0 -0.131 0.425">
            <joint name="j3" axis="0 1 0" range="-2.8 2.8"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.3922" size="0.04"/>
            <site name="m_el" pos="0 0 0" size="0.01"/>
            <site name="m_fa" pos="0 0 0.196" size="0.01"/>
            <body name="wrist1" pos="0 0 0.3922">
              <joint name="j4" axis="0 1 0" range="-3.1416 3.1416"/>
              <geom type="capsule" fromto="0 0 0 0 0.1333 0" size="0.035"/>
              <site name="m_w1" pos="0 0 0" size="0.01"/>
              <body name="wrist2" pos="0 0.1333 0">
                <joint name="j5" axis="0 0 1" range="-3.1416 3.1416"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.0997" size="0.035"/>
                <site name="m_w2" pos="0 0 0" size="0.01"/>
                <body name="wrist3" pos="0 0 0.0997">
                  <joint name="j6" axis="0 1 0" range="-3.1416 3.1416"/>
                  <geom type="capsule" fromto="0 0 0 0 0.0996 0" size="0.03"/>
                  <site name="m_w3" pos="0 0 0" size="0.01"/>
                  <site name="tcp" pos="0 0.15 0" size="0.015" rgba="0.1 0.1 0.9 1"/>
                  <geom type="cylinder" fromto="0 0.0996 0 0 0.15 0" size="0.02"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" ctrlrange="-3.1416 3.1416"/>
    <position joint="j2" ctrlrange="-3.1416 3.1416"/>
    <position joint="j3" ctrlrange="-2.8 2.8"/>
    <position joint="j4" ctrlrange="-3.1416 3.1416"/>
    <position joint="j5" ctrlrange="-3.1416 3.1416"/>
    <position joint="j6" ctrlrange="-3.1416 3.1416"/>
  </actuator>
</mujoco>
"""

# Monitored points and their capsule radii (for the keep-out inflation).
_SITES = [("m_ua", 0.05), ("m_el", 0.04), ("m_fa", 0.04), ("m_w1", 0.035),
          ("m_w2", 0.035), ("m_w3", 0.03), ("tcp", 0.02)]
_TABLE_SITES = ["m_el", "m_fa", "m_w1", "m_w2", "m_w3", "tcp"]


class SafeArmReach(SafeEnv):
    name = "SafeArmReach6DoF"
    HOME_Q = np.array([-0.903, 0.137, 2.005, -1.148, -1.594, 0.0])  # IK: TCP at (0.45, -0.35, 0.30)
    KEEPOUT_R = 0.13
    TABLE_MIN = 0.03
    REACH_TOL = 0.05
    QDOT_MAX = 1.0
    FRAME_SKIP = 25          # 20 Hz control

    def __init__(self, budget: float = 0.01, home_q: np.ndarray | None = None):
        import mujoco  # explicit failure if missing
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        if home_q is not None:
            self.HOME_Q = np.asarray(home_q, float)
        self.site_ids = [self.model.site(n).id for n, _ in _SITES]
        self.site_r = np.array([r for _, r in _SITES])
        self.table_ids = [self.model.site(n).id for n in _TABLE_SITES]
        self.tcp_id = self.model.site("tcp").id
        self.keep_mocap = self.model.body("keepout").mocapid[0]
        self.tgt_mocap = self.model.body("target").mocapid[0]
        self.jlo = self.model.jnt_range[:, 0].copy()
        self.jhi = self.model.jnt_range[:, 1].copy()
        self.dt = self.model.opt.timestep * self.FRAME_SKIP
        self.max_episode_steps = 200
        self.act_dim = 6
        self.obs_dim = 6 * 2 + 6 + 3 * 4 + 1
        self.spec = SafetySpec(predicates=["clear_keepout", "clear_table"], budget=budget)
        self.rng = np.random.default_rng(0)
        self.t = 0

    # -------------------------------------------------------------- geometry
    def _tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def _rho(self) -> np.ndarray:
        pts = self.data.site_xpos[self.site_ids]
        c = self.data.mocap_pos[self.keep_mocap]
        d_keep = np.linalg.norm(pts - c, axis=1) - self.site_r - self.KEEPOUT_R
        z = self.data.site_xpos[self.table_ids][:, 2]
        return np.array([d_keep.min() / self.KEEPOUT_R,
                         (z.min() - self.TABLE_MIN) / self.TABLE_MIN], np.float32)

    def robustness(self) -> np.ndarray:
        return self._rho()

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[:6]
        qd = self.data.qvel[:6]
        tcp = self._tcp()
        tgt = self.data.mocap_pos[self.tgt_mocap]
        kc = self.data.mocap_pos[self.keep_mocap]
        return np.concatenate([np.sin(q), np.cos(q), qd / self.QDOT_MAX, tcp, tgt, tgt - tcp,
                               kc - tcp, [np.linalg.norm(tgt - tcp)]]).astype(np.float32)

    # -------------------------------------------------------------- api
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        self.mj.mj_resetData(m, d)
        q0 = self.HOME_Q + self.rng.uniform(-0.05, 0.05, 6)
        d.qpos[:6] = q0
        d.ctrl[:6] = q0
        d.mocap_pos[self.keep_mocap] = [self.rng.uniform(0.40, 0.50), self.rng.uniform(-0.05, 0.05),
                                        self.rng.uniform(0.25, 0.33)]
        d.mocap_pos[self.tgt_mocap] = [self.rng.uniform(0.35, 0.55), self.rng.uniform(0.28, 0.42),
                                       self.rng.uniform(0.15, 0.35)]
        self.mj.mj_forward(m, d)
        self.t = 0
        self.prev_dist = float(np.linalg.norm(d.mocap_pos[self.tgt_mocap] - self._tcp()))
        self.reached = False
        return self._obs()

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        tgt_q = np.clip(self.data.ctrl[:6] + a * self.QDOT_MAX * self.dt, self.jlo, self.jhi)
        self.data.ctrl[:6] = tgt_q
        for _ in range(self.FRAME_SKIP):
            self.mj.mj_step(self.model, self.data)
        self.t += 1
        dist = float(np.linalg.norm(self.data.mocap_pos[self.tgt_mocap] - self._tcp()))
        at_goal = dist < self.REACH_TOL
        rew = 10.0 * (self.prev_dist - dist) + (1.0 if at_goal else 0.0) - 0.01 * float(a @ a)
        self.prev_dist = dist
        self.reached |= at_goal
        rho = self._rho()
        truncated = self.t >= self.max_episode_steps
        info = {"goal_met": bool(at_goal), "dist": dist}
        return self._obs(), float(rew), rho, False, truncated, info
