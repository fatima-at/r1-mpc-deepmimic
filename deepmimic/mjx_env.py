import functools

import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from flax import struct

from mpc.env import load_model
from mpc.actuators import name_mask, actuator_joint_qposadr, actuator_joint_range
from deepmimic.env import (DEFAULT_REWARD, DEFAULT_TASK, DEFAULT_REG,
                           DEFAULT_SCALES, DEFAULT_CFG)


@struct.dataclass
class RefData:
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    qref: jnp.ndarray
    foot: jnp.ndarray
    length: jnp.ndarray
    command: jnp.ndarray
    logits: jnp.ndarray


@struct.dataclass
class State:
    data: mjx.Data
    clip: jnp.ndarray
    k: jnp.ndarray
    steps: jnp.ndarray
    vel_ema: jnp.ndarray
    heading_ref: jnp.ndarray
    pos_ref: jnp.ndarray
    last_action: jnp.ndarray
    prev_action: jnp.ndarray
    hist: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    truncated: jnp.ndarray
    key: jnp.ndarray
    metrics: dict


def pack_reference(ref, speed_bias=0.0):
    n = len(ref)
    lengths = np.array([ref.n_frames(i) for i in range(n)], dtype=np.int32)
    mx = int(lengths.max())
    nq = ref.clips[0]["qpos"].shape[1]
    nv = ref.clips[0]["qvel"].shape[1]
    nu = ref.clips[0]["qref"].shape[1]
    nf = ref.clips[0]["foot_pos"].shape[1]

    qpos = np.zeros((n, mx, nq), np.float32)
    qvel = np.zeros((n, mx, nv), np.float32)
    qref = np.zeros((n, mx, nu), np.float32)
    foot = np.zeros((n, mx, nf, 3), np.float32)
    cmd = np.zeros((n, 3), np.float32)
    for i, c in enumerate(ref.clips):
        L = lengths[i]
        qpos[i, :L] = c["qpos"]
        qvel[i, :L] = c["qvel"]
        qref[i, :L] = c["qref"]
        foot[i, :L] = c["foot_pos"]
        cmd[i] = c["command"]

    v = np.abs(np.array(ref.speeds))
    w = 1.0 + float(speed_bias) * v / max(v.max(), 1e-9)
    logits = np.log(w / w.sum()).astype(np.float32)

    return RefData(qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel),
                   qref=jnp.asarray(qref), foot=jnp.asarray(foot),
                   length=jnp.asarray(lengths), command=jnp.asarray(cmd),
                   logits=jnp.asarray(logits))


def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap_angle(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class MJXWalker:
    def __init__(self, reference, model_path="assets/r1/scene.xml", cfg=None,
                 speed_bias=0.0):
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.rw = dict(DEFAULT_REWARD)
        self.tw = dict(DEFAULT_TASK)
        self.gw = dict(DEFAULT_REG)
        self.sc = dict(DEFAULT_SCALES)

        self.mj_model = load_model(model_path, strip_mesh_collision=True)
        self.model = mjx.put_model(self.mj_model)
        self.ref = pack_reference(reference, speed_bias)
        self.n_clips = len(reference)

        m = self.mj_model
        leg_idx = np.flatnonzero(name_mask(m, ["hip", "knee", "ankle"]))
        arm_keys = list(self.cfg["arm_joints"])
        arm_idx = (np.flatnonzero(name_mask(m, arm_keys)) if arm_keys
                   else np.array([], dtype=np.int64))
        self.act_idx = np.concatenate([leg_idx, arm_idx]).astype(np.int32)
        self.n_leg = len(leg_idx)
        self.n_arm = len(arm_idx)
        self.n_act = len(self.act_idx)

        adr = actuator_joint_qposadr(m)
        self.q_adr = jnp.asarray(adr[self.act_idx].astype(np.int32))
        self.v_adr = self.q_adr - 1
        self.imit_q = self.q_adr[:self.n_leg]
        self.imit_v = self.v_adr[:self.n_leg]
        lo, hi = actuator_joint_range(m)
        self.ctrl_lo = jnp.asarray(lo[self.act_idx].astype(np.float32))
        self.ctrl_hi = jnp.asarray(hi[self.act_idx].astype(np.float32))
        fr = np.abs(m.actuator_forcerange[self.act_idx, 1])
        self.tau_max = jnp.asarray(np.where(fr > 1e-6, fr, 1.0).astype(np.float32))
        self.scale_vec = jnp.asarray(np.concatenate([
            np.full(self.n_leg, self.cfg["action_scale"]),
            np.full(self.n_arm, self.cfg["arm_action_scale"])]).astype(np.float32))
        self.act_idx_j = jnp.asarray(self.act_idx)

        feet = [n for n in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                            for b in range(m.nbody))
                if n and ("ankle_roll" in n or "foot" in n)][:2]
        self.foot_bodies = jnp.asarray(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in feet],
            dtype=jnp.int32)
        fg = np.zeros((len(feet), m.ngeom), dtype=bool)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            for i, fb in enumerate(self.foot_bodies):
                if b == int(fb):
                    fg[i, g] = True
        self.foot_geom = jnp.asarray(fg)
        self.foot_root = jnp.asarray(
            [int(m.body_rootid[int(b)]) for b in self.foot_bodies],
            dtype=jnp.int32)

        self.ctrl_dt = float(reference.dt)
        self.n_substeps = max(1, int(round(self.ctrl_dt / m.opt.timestep)))
        self.gait_period = float(reference.gait_period)
        self.vel_alpha = min(1.0, self.ctrl_dt / self.gait_period)
        self.nu = m.nu

        self.obs_history = int(self.cfg["obs_history"])
        self.single_obs = self._single_obs_dim()
        self.obs_dim = self.single_obs * self.obs_history

    def _single_obs_dim(self):
        return (1 + 3 + 3 + 3 + 2 * self.n_act + 2 * self.n_act + 2 + 3 + 3
                + 4 + 2 + len(self.cfg["obs_future"]) * (self.n_act + 1))

    def _idx(self, clip, k):
        return jnp.mod(k, self.ref.length[clip])

    def _ref_qpos(self, clip, k):
        return self.ref.qpos[clip, self._idx(clip, k)]

    def _ref_qvel(self, clip, k):
        return self.ref.qvel[clip, self._idx(clip, k)]

    def _ref_qref(self, clip, k):
        return self.ref.qref[clip, self._idx(clip, k)]

    def _ref_foot(self, clip, k):
        return self.ref.foot[clip, self._idx(clip, k)]

    def _phase(self, clip, k):
        return self._idx(clip, k) / self.ref.length[clip]

    def _feet_local(self, qpos, foot_world):
        mat = quat_to_mat(qpos[3:7])
        rel = foot_world - qpos[:3][None, :]
        return (rel @ mat).ravel()

    def _foot_world(self, data):
        return data.xipos[self.foot_bodies]

    def _foot_contacts(self, data):
        con = data._impl.contact
        active = con.dist < 0.0
        g1 = con.geom[:, 0]
        g2 = con.geom[:, 1]
        out = []
        for i in range(self.foot_geom.shape[0]):
            mask = self.foot_geom[i]
            hit = active & (mask[g1] | mask[g2])
            out.append(jnp.any(hit).astype(jnp.float32))
        return jnp.stack(out)

    def _foot_vel(self, data):
        pos = data.xipos[self.foot_bodies]
        com = data.subtree_com[self.foot_root]
        cv = data.cvel[self.foot_bodies]
        ang = cv[:, 0:3]
        lin = cv[:, 3:6]
        return lin + jnp.cross(ang, pos - com)

    def _body_vel(self, data):
        mat = quat_to_mat(data.qpos[3:7])
        lin = mat.T @ data.qvel[0:3]
        return jnp.array([lin[0], lin[1], data.qvel[5]])

    def _yaw(self, data):
        mat = quat_to_mat(data.qpos[3:7])
        return jnp.arctan2(mat[1, 0], mat[0, 0])

    def _path_error(self, data, heading_ref, pos_ref):
        d = data.qpos[:2] - pos_ref
        along = jnp.cos(heading_ref) * d[0] + jnp.sin(heading_ref) * d[1]
        cross = -jnp.sin(heading_ref) * d[0] + jnp.cos(heading_ref) * d[1]
        return along, cross, wrap_angle(self._yaw(data) - heading_ref)

    def _single_obs(self, data, clip, k, vel_ema, heading_ref, pos_ref,
                    last_action, prev_action):
        mat = quat_to_mat(data.qpos[3:7])
        gravity = mat.T @ jnp.array([0.0, 0.0, -1.0])
        lin = mat.T @ data.qvel[0:3]
        ph = self._phase(clip, k)
        along, cross, dyaw = self._path_error(data, heading_ref, pos_ref)
        cmd = self.ref.command[clip]

        parts = [
            data.qpos[2][None],
            gravity,
            lin,
            data.qvel[3:6],
            data.qpos[self.q_adr],
            data.qvel[self.v_adr],
            last_action,
            prev_action,
            jnp.array([jnp.sin(2 * jnp.pi * ph), jnp.cos(2 * jnp.pi * ph)]),
            cmd,
            vel_ema - cmd,
            jnp.array([along, cross, jnp.sin(dyaw), jnp.cos(dyaw)]),
            self._foot_contacts(data),
        ]
        for h in self.cfg["obs_future"]:
            rq = self._ref_qpos(clip, k + h)
            parts.append(rq[self.q_adr] - data.qpos[self.q_adr])
            parts.append((rq[2] - data.qpos[2])[None])
        return jnp.concatenate(parts)

    def _stack(self, cur, hist):
        return jnp.concatenate([cur, hist.ravel()])

    def reset(self, key):
        key, k1, k2 = jax.random.split(key, 3)
        clip = jax.random.categorical(k1, self.ref.logits)
        k0 = jax.random.randint(k2, (), 0, self.ref.length[clip])

        data = mjx.make_data(self.model)
        data = data.replace(qpos=self._ref_qpos(clip, k0),
                            qvel=self._ref_qvel(clip, k0),
                            ctrl=self._ref_qref(clip, k0))
        data = mjx.forward(self.model, data)

        vel_ema = self._body_vel(data)
        heading_ref = self._yaw(data)
        pos_ref = data.qpos[:2]
        zero = jnp.zeros(self.n_act)
        cur = self._single_obs(data, clip, k0, vel_ema, heading_ref, pos_ref,
                               zero, zero)
        hist = jnp.tile(cur, (self.obs_history - 1, 1))
        obs = self._stack(cur, hist)

        return State(data=data, clip=clip, k=k0, steps=jnp.int32(0),
                     vel_ema=vel_ema, heading_ref=heading_ref, pos_ref=pos_ref,
                     last_action=zero, prev_action=zero, hist=hist, obs=obs,
                     reward=jnp.float32(0), done=jnp.float32(0),
                     truncated=jnp.float32(0), key=key,
                     metrics={k: jnp.float32(0) for k in
                              ("pose", "vel", "endeff", "root", "track",
                               "along", "cross", "heading", "slip", "smooth",
                               "accel", "torque", "imitation", "task", "reg")})

    def step(self, state, action):
        a = jnp.clip(action, -1.0, 1.0)
        base = self._ref_qref(state.clip, state.k)
        ctrl = base.at[self.act_idx_j].set(
            jnp.clip(base[self.act_idx_j] + self.scale_vec * a,
                     self.ctrl_lo, self.ctrl_hi))

        data = state.data.replace(ctrl=ctrl)

        def one(d, _):
            return mjx.step(self.model, d), None

        data, _ = jax.lax.scan(one, data, None, length=self.n_substeps)

        k = state.k + 1
        steps = state.steps + 1
        vel_ema = state.vel_ema + self.vel_alpha * (self._body_vel(data)
                                                    - state.vel_ema)
        cmd = self.ref.command[state.clip]
        h = state.heading_ref
        fwd = jnp.array([jnp.cos(h), jnp.sin(h)])
        lat = jnp.array([-jnp.sin(h), jnp.cos(h)])
        pos_ref = state.pos_ref + (fwd * cmd[0] + lat * cmd[1]) * self.ctrl_dt
        heading_ref = wrap_angle(h + cmd[2] * self.ctrl_dt)

        rq = self._ref_qpos(state.clip, k)
        rv = self._ref_qvel(state.clip, k)
        dq = data.qpos[self.imit_q] - rq[self.imit_q]
        r_pose = jnp.exp(-self.sc["pose"] * jnp.sum(dq ** 2))
        dv = data.qvel[self.imit_v] - rv[self.imit_v]
        r_vel = jnp.exp(-self.sc["vel"] * jnp.sum(dv ** 2))

        fw = self._foot_world(data)
        de = (self._feet_local(data.qpos, fw)
              - self._feet_local(rq, self._ref_foot(state.clip, k)))
        r_end = jnp.exp(-self.sc["endeff"] * jnp.sum(de ** 2))

        mat = quat_to_mat(data.qpos[3:7])
        rmat = quat_to_mat(rq[3:7])
        dz = data.qpos[2] - rq[2]
        dup = jnp.sum((mat[:, 2] - rmat[:, 2]) ** 2)
        r_root = jnp.exp(-self.sc["root"] * (dz * dz + dup))

        imit = (self.rw["pose"] * r_pose + self.rw["vel"] * r_vel
                + self.rw["endeff"] * r_end + self.rw["root"] * r_root)

        e = vel_ema - cmd
        r_track = jnp.exp(-self.sc["task"] * jnp.sum(e ** 2))
        along, cross, dyaw = self._path_error(data, heading_ref, pos_ref)
        r_along = jnp.exp(-self.sc["along"] * along * along)
        r_cross = jnp.exp(-self.sc["cross"] * cross * cross)
        r_head = jnp.exp(-self.sc["heading"] * dyaw * dyaw)
        task = (self.tw["track"] * r_track + self.tw["along"] * r_along
                + self.tw["cross"] * r_cross + self.tw["heading"] * r_head)

        contacts = self._foot_contacts(data)
        fvel = self._foot_vel(data)
        slip = jnp.sum(contacts * jnp.sum(fvel[:, :2] ** 2, axis=1))
        r_slip = jnp.exp(-self.sc["slip"] * slip)
        da = a - state.last_action
        r_smooth = jnp.exp(-self.sc["smooth"] * jnp.sum(da ** 2))
        d2a = a - 2.0 * state.last_action + state.prev_action
        r_accel = jnp.exp(-self.sc["accel"] * jnp.sum(d2a ** 2))
        tau = data.actuator_force[self.act_idx_j] / self.tau_max
        r_torque = jnp.exp(-self.sc["torque"] * jnp.sum(tau ** 2))
        reg = (self.gw["slip"] * r_slip + self.gw["smooth"] * r_smooth
               + self.gw["torque"] * r_torque + self.gw["accel"] * r_accel)

        reward = (self.cfg["imitation_weight"] * imit
                  + self.cfg["task_weight"] * task
                  + self.cfg["reg_weight"] * reg)

        fell = ((data.qpos[2] < self.cfg["fall_height"])
                | (mat[2, 2] < self.cfg["fall_upright"])
                | (jnp.sqrt(jnp.mean(dq ** 2)) > self.cfg["max_pose_error"])
                | (jnp.sqrt(jnp.mean(de ** 2)) > self.cfg["max_foot_error"])
                | (jnp.abs(along) > self.cfg["max_along_error"])
                | ~jnp.all(jnp.isfinite(data.qpos)))
        done = fell.astype(jnp.float32)
        truncated = (steps >= self.cfg["max_steps"]).astype(jnp.float32)

        cur = self._single_obs(data, state.clip, k, vel_ema, heading_ref,
                               pos_ref, a, state.last_action)
        hist = jnp.concatenate([cur[None], state.hist[:-1]], axis=0)
        obs = self._stack(cur, hist)

        metrics = {"pose": r_pose, "vel": r_vel, "endeff": r_end,
                   "root": r_root, "track": r_track, "along": r_along,
                   "cross": r_cross, "heading": r_head, "slip": r_slip,
                   "smooth": r_smooth, "accel": r_accel, "torque": r_torque,
                   "imitation": imit, "task": task, "reg": reg}

        return state.replace(data=data, k=k, steps=steps, vel_ema=vel_ema,
                             heading_ref=heading_ref, pos_ref=pos_ref,
                             last_action=a, prev_action=state.last_action,
                             hist=hist, obs=obs, reward=reward, done=done,
                             truncated=truncated, metrics=metrics)

    def reset_if_done(self, state):
        key, sub = jax.random.split(state.key)
        fresh = self.reset(sub)
        flag = jnp.maximum(state.done, state.truncated) > 0.5

        def pick(a, b):
            return jnp.where(flag, a, b)

        out = jax.tree.map(pick, fresh, state.replace(key=key))
        return out.replace(reward=state.reward, done=state.done,
                           truncated=state.truncated, metrics=state.metrics)
