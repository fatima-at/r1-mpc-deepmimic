import numpy as np
import mujoco

from mpc.env import load_model
from mpc.actuators import name_mask, actuator_joint_qposadr, actuator_joint_range
from mpc.sensors import find_foot_bodies_model, foot_sensor_slices


DEFAULT_REWARD = {
    "pose": 0.55,
    "vel": 0.10,
    "endeff": 0.20,
    "root": 0.15,
}
DEFAULT_TASK = {
    "track": 0.35,
    "along": 0.30,
    "cross": 0.15,
    "heading": 0.20,
}
DEFAULT_REG = {
    "slip": 0.40,
    "smooth": 0.30,
    "torque": 0.20,
    "accel": 0.10,
}
DEFAULT_SCALES = {
    "pose": 2.0,
    "vel": 0.10,
    "endeff": 40.0,
    "root": 10.0,
    "task": 8.0,
    "along": 4.0,
    "cross": 20.0,
    "heading": 10.0,
    "slip": 20.0,
    "smooth": 2.0,
    "accel": 1.0,
    "torque": 1.0,
}
DEFAULT_CFG = {
    "imitation_weight": 0.50,
    "task_weight": 0.30,
    "reg_weight": 0.20,
    "action_scale": 0.18,
    "arm_action_scale": 0.60,
    "arm_joints": ("shoulder_pitch",),
    "residual": True,
    "fall_height": 0.50,
    "fall_upright": 0.40,
    "max_pose_error": 0.35,
    "max_foot_error": 0.15,
    "max_along_error": 1.00,
    "obs_future": (1, 4, 10),
    "obs_history": 3,
    "early_termination": True,
    "rsi": True,
    "max_steps": 300,
    "vel_filter": True,
    "push_interval": 2.5,
    "push_vel": 0.35,
    "obs_noise": 0.01,
    "init_noise": 0.02,
    "domain_rand": False,
}


def quat_to_mat(q):
    m = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def wrap_angle(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class DeepMimicEnv:
    def __init__(self, reference, model_path="assets/r1/scene.xml", model=None,
                 cfg=None, reward_weights=None, scales=None, task_weights=None,
                 reg_weights=None, seed=0):
        self.model = model if model is not None else load_model(model_path)
        self.ref = reference
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.rw = dict(DEFAULT_REWARD)
        if reward_weights:
            self.rw.update(reward_weights)
        self.tw = dict(DEFAULT_TASK)
        if task_weights:
            self.tw.update(task_weights)
        self.gw = dict(DEFAULT_REG)
        if reg_weights:
            self.gw.update(reg_weights)
        self.sc = dict(DEFAULT_SCALES)
        if scales:
            self.sc.update(scales)

        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        leg_keys = ["hip", "knee", "ankle"]
        self.mask = name_mask(self.model, leg_keys)
        leg_idx = np.flatnonzero(self.mask)
        arm_keys = list(self.cfg["arm_joints"])
        arm_idx = (np.flatnonzero(name_mask(self.model, arm_keys))
                   if arm_keys else np.array([], dtype=np.int64))
        self.act_idx = np.concatenate([leg_idx, arm_idx]).astype(np.int64)
        self.n_leg = len(leg_idx)
        self.n_arm = len(arm_idx)
        self.n_act = len(self.act_idx)
        self.imit_sel = np.arange(self.n_leg)

        adr = actuator_joint_qposadr(self.model)
        self.q_adr = adr[self.act_idx]
        self.v_adr = self.q_adr - 1
        self.imit_q_adr = self.q_adr[self.imit_sel]
        self.imit_v_adr = self.v_adr[self.imit_sel]
        self.scale_vec = np.concatenate([
            np.full(self.n_leg, self.cfg["action_scale"]),
            np.full(self.n_arm, self.cfg["arm_action_scale"])])
        lo, hi = actuator_joint_range(self.model)
        self.ctrl_lo = lo[self.act_idx]
        self.ctrl_hi = hi[self.act_idx]
        fr = np.abs(self.model.actuator_forcerange[self.act_idx, 1])
        self.tau_max = np.where(fr > 1e-6, fr, 1.0)

        self.ctrl_dt = float(self.ref.dt)
        self.n_substeps = max(1, int(round(self.ctrl_dt / self.model.opt.timestep)))

        feet = find_foot_bodies_model(self.model)
        self.foot_slices = foot_sensor_slices(self.model, feet)
        self.n_feet = len(self.foot_slices)
        self.foot_bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                            for n in feet]

        self._clip = 0
        self._k0 = 0
        self._k = 0
        self._steps = 0
        self.vel_alpha = min(1.0, self.ctrl_dt / self.ref.gait_period)
        self._vel_ema = np.zeros(3)
        self._heading_ref = 0.0
        self._pos_ref = np.zeros(2)
        self.last_action = np.zeros(self.n_act)
        self.prev_action = np.zeros(self.n_act)
        self._hist = None
        self._next_push = 0
        self.obs_dim = len(self.reset())

    def _ref_qpos(self, k):
        c = self.ref.clips[self._clip]["qpos"]
        return c[self.ref.index(self._clip, k)].astype(np.float64)

    def _ref_qvel(self, k):
        c = self.ref.clips[self._clip]["qvel"]
        return c[self.ref.index(self._clip, k)].astype(np.float64)

    def _ref_ctrl(self, k):
        c = self.ref.clips[self._clip]["qref"]
        return c[self.ref.index(self._clip, k)].astype(np.float64)

    def _feet_local(self, qpos, foot_world):
        mat = quat_to_mat(qpos[3:7])
        rel = np.asarray(foot_world, dtype=np.float64) - qpos[:3][None, :]
        return (rel @ mat).ravel()

    def _feet_local_sim(self):
        w = np.array([self.data.sensordata[p] for p, _ in self.foot_slices])
        return self._feet_local(np.asarray(self.data.qpos, dtype=np.float64), w)

    def _feet_local_ref(self, k):
        c = self.ref.clips[self._clip]
        kk = self.ref.index(self._clip, k)
        return self._feet_local(c["qpos"][kk].astype(np.float64), c["foot_pos"][kk])

    def foot_contacts(self):
        flags = np.zeros(self.n_feet)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            for gid in (con.geom1, con.geom2):
                bid = int(self.model.geom_bodyid[gid])
                for i, fb in enumerate(self.foot_bodies):
                    if bid == fb:
                        flags[i] = 1.0
        return flags

    def foot_slip(self):
        con = self.foot_contacts()
        tot = 0.0
        for i, (_, v) in enumerate(self.foot_slices):
            if con[i] > 0.5:
                vel = np.asarray(self.data.sensordata[v], dtype=np.float64)
                tot += float(vel[0] ** 2 + vel[1] ** 2)
        return tot

    def yaw(self):
        m = quat_to_mat(self.data.qpos[3:7])
        return float(np.arctan2(m[1, 0], m[0, 0]))

    def path_error(self):
        h = self._heading_ref
        d = np.asarray(self.data.qpos[:2], dtype=np.float64) - self._pos_ref
        along = float(np.cos(h) * d[0] + np.sin(h) * d[1])
        cross = float(-np.sin(h) * d[0] + np.cos(h) * d[1])
        return along, cross, wrap_angle(self.yaw() - h)

    def phase(self):
        return self.ref.phase(self._clip, self._k)

    def command(self):
        return np.asarray(self.ref.clips[self._clip]["command"], dtype=np.float64)

    def observation(self):
        cur = self._observation_now()
        h = int(self.cfg["obs_history"])
        if h <= 1:
            return cur
        if self._hist is None:
            self._hist = [cur.copy() for _ in range(h - 1)]
        n = float(self.cfg["obs_noise"]) if self.cfg["domain_rand"] else 0.0
        if n > 0:
            cur = cur + self.rng.normal(0, n, cur.shape)
        out = np.concatenate([cur] + self._hist)
        self._hist = [cur.copy()] + self._hist[:h - 2]
        return out

    def _observation_now(self):
        d = self.data
        mat = quat_to_mat(d.qpos[3:7])
        gravity = mat.T @ np.array([0.0, 0.0, -1.0])
        lin = mat.T @ np.array(d.qvel[0:3])
        ang = np.array(d.qvel[3:6])
        ph = self.phase()
        along, cross, dyaw = self.path_error()

        parts = [
            np.array([d.qpos[2]]),
            gravity,
            lin,
            ang,
            d.qpos[self.q_adr],
            d.qvel[self.v_adr],
            self.last_action,
            self.prev_action,
            np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]),
            self.command(),
            self._vel_ema - self.command(),
            np.array([along, cross, np.sin(dyaw), np.cos(dyaw)]),
            self.foot_contacts(),
        ]
        for h in self.cfg["obs_future"]:
            rq = self._ref_qpos(self._k + h)
            parts.append(rq[self.q_adr] - d.qpos[self.q_adr])
            parts.append(np.array([rq[2] - d.qpos[2]]))
        return np.concatenate(parts).astype(np.float64)

    def reset(self, clip=None, frame=None, rng=None):
        rng = self.rng if rng is None else rng
        if clip is None or frame is None:
            if self.cfg["rsi"]:
                self._clip, self._k0 = self.ref.sample_start(rng)
            else:
                self._clip = int(rng.integers(len(self.ref)))
                self._k0 = self.ref.clips[self._clip]["settle"]
        else:
            self._clip, self._k0 = int(clip), int(frame)
        self._k = self._k0
        self._steps = 0

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._ref_qpos(self._k)
        self.data.qvel[:] = self._ref_qvel(self._k)
        self.data.ctrl[:] = self._ref_ctrl(self._k)
        mujoco.mj_forward(self.model, self.data)

        if self.cfg["domain_rand"]:
            n = float(self.cfg["init_noise"])
            if n > 0:
                self.data.qpos[self.q_adr] += rng.normal(0, n, self.n_act)
                self.data.qvel[self.v_adr] += rng.normal(0, 5 * n, self.n_act)
                mujoco.mj_forward(self.model, self.data)
            iv = max(1, int(self.cfg["push_interval"] / self.ctrl_dt))
            self._next_push = int(rng.integers(iv // 2, 2 * iv))
        self._hist = None
        self._vel_ema = self._body_vel()
        self._heading_ref = self.yaw()
        self._pos_ref = np.asarray(self.data.qpos[:2], dtype=np.float64).copy()
        self.last_action = np.zeros(self.n_act)
        self.prev_action = np.zeros(self.n_act)
        return self.observation()

    def _body_vel(self):
        d = self.data
        mat = quat_to_mat(d.qpos[3:7])
        lin = mat.T @ np.array(d.qvel[0:3])
        return np.array([lin[0], lin[1], float(d.qvel[5])])

    def _advance_path(self):
        cmd = self.command()
        h = self._heading_ref
        fwd = np.array([np.cos(h), np.sin(h)])
        lat = np.array([-np.sin(h), np.cos(h)])
        self._pos_ref = self._pos_ref + (fwd * cmd[0] + lat * cmd[1]) * self.ctrl_dt
        self._heading_ref = wrap_angle(h + cmd[2] * self.ctrl_dt)

    def _apply(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        base = self._ref_ctrl(self._k)
        ctrl = base.copy()
        if self.cfg["residual"]:
            ctrl[self.act_idx] = base[self.act_idx] + self.scale_vec * a
        else:
            mid = 0.5 * (self.ctrl_hi + self.ctrl_lo)
            half = 0.5 * (self.ctrl_hi - self.ctrl_lo)
            ctrl[self.act_idx] = mid + half * a
        ctrl[self.act_idx] = np.clip(ctrl[self.act_idx], self.ctrl_lo, self.ctrl_hi)
        return ctrl, a

    def reward(self, action):
        d = self.data
        k = self._k
        rq, rv = self._ref_qpos(k), self._ref_qvel(k)

        dq = d.qpos[self.imit_q_adr] - rq[self.imit_q_adr]
        r_pose = np.exp(-self.sc["pose"] * float(np.sum(dq ** 2)))

        dv = d.qvel[self.imit_v_adr] - rv[self.imit_v_adr]
        r_vel = np.exp(-self.sc["vel"] * float(np.sum(dv ** 2)))

        de = self._feet_local_sim() - self._feet_local_ref(k)
        r_end = np.exp(-self.sc["endeff"] * float(np.sum(de ** 2)))

        mat = quat_to_mat(d.qpos[3:7])
        rmat = quat_to_mat(rq[3:7])
        dz = float(d.qpos[2] - rq[2])
        dup = float(np.sum((mat[:, 2] - rmat[:, 2]) ** 2))
        r_root = np.exp(-self.sc["root"] * (dz * dz + dup))

        imit = (self.rw["pose"] * r_pose + self.rw["vel"] * r_vel
                + self.rw["endeff"] * r_end + self.rw["root"] * r_root)

        v = self._vel_ema if self.cfg["vel_filter"] else self._body_vel()
        e = v - self.command()
        r_track = np.exp(-self.sc["task"] * float(np.sum(e ** 2)))
        along, cross, dyaw = self.path_error()
        r_along = np.exp(-self.sc["along"] * along * along)
        r_cross = np.exp(-self.sc["cross"] * cross * cross)
        r_head = np.exp(-self.sc["heading"] * dyaw * dyaw)
        task = (self.tw["track"] * r_track + self.tw["along"] * r_along
                + self.tw["cross"] * r_cross + self.tw["heading"] * r_head)

        r_slip = np.exp(-self.sc["slip"] * self.foot_slip())
        da = action - self.last_action
        r_smooth = np.exp(-self.sc["smooth"] * float(np.sum(da ** 2)))
        d2a = action - 2.0 * self.last_action + self.prev_action
        r_accel = np.exp(-self.sc["accel"] * float(np.sum(d2a ** 2)))
        tau = np.asarray(d.actuator_force[self.act_idx], dtype=np.float64)
        r_torque = np.exp(-self.sc["torque"]
                          * float(np.sum((tau / self.tau_max) ** 2)))
        reg = (self.gw["slip"] * r_slip + self.gw["smooth"] * r_smooth
               + self.gw["torque"] * r_torque + self.gw["accel"] * r_accel)

        total = (self.cfg["imitation_weight"] * imit
                 + self.cfg["task_weight"] * task
                 + self.cfg["reg_weight"] * reg)
        terms = {"pose": r_pose, "vel": r_vel, "endeff": r_end, "root": r_root,
                 "track": r_track, "along": r_along, "cross": r_cross,
                 "heading": r_head,
                 "slip": r_slip, "smooth": r_smooth, "accel": r_accel,
                 "torque": r_torque, "imitation": imit, "task": task, "reg": reg}
        return float(total), terms

    def terminated(self):
        if not self.cfg["early_termination"]:
            return False, ""
        d = self.data
        if not np.all(np.isfinite(d.qpos)):
            return True, "nan"
        if float(d.qpos[2]) < self.cfg["fall_height"]:
            return True, "height"
        mat = quat_to_mat(d.qpos[3:7])
        if float(mat[2, 2]) < self.cfg["fall_upright"]:
            return True, "upright"
        dq = (d.qpos[self.imit_q_adr]
              - self._ref_qpos(self._k)[self.imit_q_adr])
        if float(np.sqrt(np.mean(dq ** 2))) > self.cfg["max_pose_error"]:
            return True, "pose"
        de = self._feet_local_sim() - self._feet_local_ref(self._k)
        if float(np.sqrt(np.mean(de ** 2))) > self.cfg["max_foot_error"]:
            return True, "foot"
        along, _, _ = self.path_error()
        if abs(along) > self.cfg["max_along_error"]:
            return True, "along"
        return False, ""

    def _maybe_push(self):
        if not self.cfg["domain_rand"]:
            return
        self._next_push -= 1
        if self._next_push > 0:
            return
        iv = max(1, int(self.cfg["push_interval"] / self.ctrl_dt))
        self._next_push = int(self.rng.integers(iv // 2, 2 * iv))
        mag = float(self.cfg["push_vel"]) * float(self.rng.uniform(0.3, 1.0))
        ang = float(self.rng.uniform(0, 2 * np.pi))
        self.data.qvel[0] += mag * np.cos(ang)
        self.data.qvel[1] += mag * np.sin(ang)

    def step(self, action):
        self._maybe_push()
        ctrl, a = self._apply(action)
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._k += 1
        self._steps += 1
        self._vel_ema += self.vel_alpha * (self._body_vel() - self._vel_ema)
        self._advance_path()

        rew, terms = self.reward(a)
        self.prev_action = self.last_action
        self.last_action = a

        term, why = self.terminated()
        if self.ref.cyclic:
            truncated = self._steps >= self.cfg["max_steps"]
        else:
            horizon = self.ref.n_frames(self._clip) - max(self.cfg["obs_future"]) - 1
            truncated = self._k >= horizon or self._steps >= self.cfg["max_steps"]
        obs = self.observation()
        info = {"terms": terms, "reason": why, "phase": self.phase(),
                "clip": self._clip, "frame": self._k, "steps": self._steps}
        return obs, rew, bool(term), bool(truncated), info
