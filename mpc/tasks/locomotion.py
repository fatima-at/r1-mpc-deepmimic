import numpy as np
import mujoco

from mpc.cem import quat_to_mat, control_bounds
from mpc.actuators import nominal_ctrl as actuator_nominal_ctrl
from mpc.gait import derive_gait_period
from mpc.sensors import (
    find_foot_bodies_model,
    foot_sensor_slices,
    foot_axis_slices,
    rest_foot_axis,
    rest_foot_offset,
    rest_foot_height,
)


DEFAULT_WEIGHTS = {
    "lin_vel": 1.0,
    "yaw_vel": 0.5,
    "height": 3.0,
    "upright": 3.0,
    "posture": 0.3,
    "joint_vel": 0.05,
    "ctrl": 0.1,
    "smooth": 0.2,
    "fall": 100.0,
    "swing": 3.0,
    "stance": 3.0,
    "slip": 1.0,
    "over": 6.0,
    "flat": 0.5,
    "step": 2.0,
    "heading": 2.0,
    "cross": 3.0,
    "pushoff": 1.5,
}

GAIT = {
    "period": None,
    "duty": 0.60,
    "clearance": 0.05,
    "over_margin": 0.03,
    "min_speed": 0.05,
    "pushoff": 0.35,
    "early_frac": 0.25,
    "late_frac": 0.70,
}

DEFAULT_TOLERANCES = {
    "lin_vel": 0.10,
    "lat_vel": 0.30,
    "yaw_vel": 0.20,
    "height": 0.02,
    "upright": 0.05,
    "posture": 0.20,
    "joint_vel": 2.00,
    "ctrl": 0.20,
    "smooth": 0.05,
    "swing": 0.02,
    "stance": 0.01,
    "slip": 0.15,
    "over": 0.03,
    "flat": 0.10,
    "step": 0.03,
    "heading": 0.10,
    "cross": 0.05,
    "pushoff": 0.10,
}


class LocomotionTask:
    def __init__(
        self,
        model,
        weights=None,
        tolerances=None,
        gait=None,
        q_nominal=None,
        vel_mode="mean",
        keyframe="standing",
        height_target=None,
        fall_height=None,
    ):
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("first joint must be a free floating base")

        self.model = model
        self.nq = model.nq
        self.nv = model.nv
        self.nu = model.nu
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.tol = dict(DEFAULT_TOLERANCES)
        if tolerances:
            self.tol.update(tolerances)

        self.q_nominal = (
            np.array(q_nominal, dtype=np.float64)
            if q_nominal is not None
            else self._nominal_qpos(keyframe)
        )
        self.qj_nominal = self.q_nominal[7:]
        self.height_nominal = float(self.q_nominal[2])
        self.height_target = (
            self.height_nominal if height_target is None else float(height_target)
        )
        self.fall_height = (
            0.5 * self.height_nominal if fall_height is None else float(fall_height)
        )

        lo, hi = control_bounds(model)
        self.ctrl_span = np.maximum(hi - lo, 1e-6)
        self.u_nominal = actuator_nominal_ctrl(model, self.q_nominal)

        self.vel_mode = vel_mode
        self.gait = dict(GAIT)
        if gait:
            self.gait.update(gait)
        if self.gait.get("period") is None:
            info = derive_gait_period(model, duty=self.gait["duty"])
            self.gait["period"] = info["period_gait"] if info else 1.5
            self.gait_info = info
        else:
            self.gait_info = None
        self.foot_names = find_foot_bodies_model(model)
        self.foot_slices = foot_sensor_slices(model, self.foot_names)
        self.foot_axes = foot_axis_slices(model, self.foot_names)
        self.foot_axis_rest = (
            rest_foot_axis(model, self.foot_names) if self.foot_names else []
        )
        self.foot_offset = (
            rest_foot_offset(model, self.foot_names) if self.foot_names else []
        )
        self.foot_rest_z = (
            rest_foot_height(model, self.foot_names) if self.foot_names else None
        )
        self.ankle_adr = []
        for nm in self.foot_names:
            side = "left" if "left" in nm else "right"
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_pitch_joint"
            )
            self.ankle_adr.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)
        self.has_gait = len(self.foot_slices) == 2

    def _nominal_qpos(self, keyframe):
        if keyframe is not None and self.model.nkey > 0:
            for i in range(self.model.nkey):
                if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i) == keyframe:
                    return np.array(self.model.key_qpos[i], dtype=np.float64)
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        return np.array(data.qpos, dtype=np.float64)

    def nominal_ctrl(self):
        return self.u_nominal.copy()

    def gait_cost(self, sensordata, times, command, qpos=None, mat=None):
        g = self.gait
        t = self.tol
        w = self.weights

        speed = float(np.hypot(command[0], command[1]))
        walking = speed >= g["min_speed"]
        duty = float(np.clip(g["duty"], 0.05, 0.95))
        base_pos = qpos[..., 0:3] if qpos is not None else None

        total = 0.0
        for i, (pos_sl, vel_sl) in enumerate(self.foot_slices):
            z = sensordata[:, :, pos_sl][:, :, 2] - self.foot_rest_z[i]
            vxy = sensordata[:, :, vel_sl][:, :, :2]

            if walking:
                phase = (times / g["period"] + 0.5 * i) % 1.0
                stance = phase < duty
                u = np.clip(phase / duty, 0.0, 1.0)
                s = np.clip((phase - duty) / (1.0 - duty), 0.0, 1.0)
                z_des = np.where(stance, 0.0, g["clearance"] * np.sin(np.pi * s))
                early = stance & (u < g["early_frac"])
                late = stance & (u >= g["late_frac"])
                mid = stance & ~early & ~late
                flat_scale = np.where(mid, 1.0, np.where(stance, 0.15, 0.30))
                height_scale = np.where(
                    mid, 1.0, np.where(early, 0.30, np.where(late, 0.20, 1.0))
                )
            else:
                stance = np.ones_like(z, dtype=bool)
                late = np.zeros_like(z, dtype=bool)
                u = np.zeros_like(z)
                s = np.zeros_like(z)
                z_des = np.zeros_like(z)
                flat_scale = np.ones_like(z)
                height_scale = np.ones_like(z)

            c_swing = ((z - z_des) / t["swing"]) ** 2
            c_stance = (z / t["stance"]) ** 2
            c_slip = np.sum((vxy / t["slip"]) ** 2, axis=-1)

            over = np.maximum(z - (z_des + g["over_margin"]), 0.0)
            c_over = (over / t["over"]) ** 2

            total = total + np.where(
                stance,
                w["stance"] * height_scale * c_stance + w["slip"] * c_slip,
                w["swing"] * c_swing,
            )
            total = total + w["over"] * c_over

            if walking and base_pos is not None and mat is not None and self.foot_offset:
                p_world = sensordata[:, :, pos_sl]
                r_body = np.einsum("...ji,...j->...i", mat, p_world - base_pos)
                step_len = speed * g["period"] * 0.5
                x_des = self.foot_offset[i][0] + step_len * (s - 0.5)
                y_des = self.foot_offset[i][1]
                c_step = (
                    ((r_body[..., 0] - x_des) / t["step"]) ** 2
                    + ((r_body[..., 1] - y_des) / t["step"]) ** 2
                )
                total = total + np.where(stance, 0.0, w["step"] * c_step)

            if walking and qpos is not None and self.ankle_adr[i] >= 0:
                ramp = np.clip(
                    (u - g["late_frac"]) / max(1.0 - g["late_frac"], 1e-6), 0.0, 1.0
                )
                ank_des = g["pushoff"] * ramp
                ank = qpos[..., self.ankle_adr[i]]
                c_push = ((ank - ank_des) / t["pushoff"]) ** 2
                total = total + np.where(late, w["pushoff"] * c_push, 0.0)

            if self.foot_axes and self.foot_axis_rest:
                zax = sensordata[:, :, self.foot_axes[i]]
                align = np.einsum("...k,k->...", zax, self.foot_axis_rest[i])
                total = total + (
                    w["flat"] * flat_scale * ((1.0 - align) / t["flat"]) ** 2
                )
        return total

    def cost(self, qpos, qvel, ctrl, command, last_ctrl=None, sensordata=None,
             times=None, heading_ref=None, t_now=None, pos_ref=None):
        command = np.asarray(command, dtype=np.float64)
        w = self.weights

        height = qpos[..., 2]
        quat = qpos[..., 3:7]
        mat = quat_to_mat(quat)

        lin_vel_world = qvel[..., 0:3]
        lin_vel_body = np.einsum("...ji,...j->...i", mat, lin_vel_world)
        ang_vel_body = qvel[..., 3:6]

        t = self.tol
        n_j = max(qpos.shape[-1] - 7, 1)

        if self.vel_mode == "mean":
            c_lin = 0.0
            c_yaw = 0.0
        else:
            c_lin = (
                ((lin_vel_body[..., 0] - command[0]) / t["lin_vel"]) ** 2
                + ((lin_vel_body[..., 1] - command[1]) / t["lat_vel"]) ** 2
            )
            c_yaw = ((ang_vel_body[..., 2] - command[2]) / t["yaw_vel"]) ** 2
        c_height = ((height - self.height_target) / t["height"]) ** 2
        c_upright = ((1.0 - mat[..., 2, 2]) / t["upright"]) ** 2
        c_posture = np.mean(
            ((qpos[..., 7:] - self.qj_nominal) / t["posture"]) ** 2, axis=-1
        )
        c_jvel = np.mean((qvel[..., 6:] / t["joint_vel"]) ** 2, axis=-1)

        du = ctrl - self.u_nominal
        c_ctrl = np.mean((du / t["ctrl"]) ** 2, axis=-1)

        if last_ctrl is None:
            prev = du[:, :1]
        else:
            prev = np.broadcast_to(
                np.asarray(last_ctrl, dtype=np.float64) - self.u_nominal,
                (du.shape[0], 1, du.shape[2]),
            )
        d_ctrl = np.diff(du, axis=1, prepend=prev)
        c_smooth = np.mean((d_ctrl / t["smooth"]) ** 2, axis=-1)

        c_fall = (height < self.fall_height).astype(np.float64)

        if heading_ref is not None and times is not None and t_now is not None:
            psi = np.arctan2(mat[..., 1, 0], mat[..., 0, 0])
            psi_des = heading_ref + command[2] * (times - t_now)
            err = np.arctan2(np.sin(psi - psi_des), np.cos(psi - psi_des))
            c_heading = (err / t["heading"]) ** 2
        else:
            c_heading = 0.0

        if (pos_ref is not None and heading_ref is not None
                and times is not None and t_now is not None):
            ch, sh = np.cos(heading_ref), np.sin(heading_ref)
            dt_h = times - t_now
            vx_w = command[0] * ch - command[1] * sh
            vy_w = command[0] * sh + command[1] * ch
            px_d = pos_ref[0] + vx_w * dt_h
            py_d = pos_ref[1] + vy_w * dt_h
            cross = -(qpos[..., 0] - px_d) * sh + (qpos[..., 1] - py_d) * ch
            c_cross = (cross / t["cross"]) ** 2
        else:
            c_cross = 0.0

        per_step = (
            w["lin_vel"] * c_lin
            + w["yaw_vel"] * c_yaw
            + w["height"] * c_height
            + w["upright"] * c_upright
            + w["posture"] * c_posture
            + w["joint_vel"] * c_jvel
            + w["ctrl"] * c_ctrl
            + w["smooth"] * c_smooth
            + w["fall"] * c_fall
            + w["heading"] * c_heading
            + w["cross"] * c_cross
        )
        if self.has_gait and sensordata is not None and times is not None:
            per_step = per_step + self.gait_cost(
                sensordata, times, command, qpos=qpos, mat=mat
            )
        total = per_step.sum(axis=1)

        if self.vel_mode == "mean":
            n_steps = lin_vel_body.shape[1]
            v_mean = np.mean(lin_vel_body, axis=1)
            w_mean = np.mean(ang_vel_body[..., 2], axis=1)
            e_fwd = (v_mean[..., 0] - command[0]) / t["lin_vel"]
            e_lat = (v_mean[..., 1] - command[1]) / t["lat_vel"]
            e_yaw = (w_mean - command[2]) / t["yaw_vel"]
            total = total + n_steps * (
                w["lin_vel"] * (e_fwd**2 + e_lat**2) + w["yaw_vel"] * e_yaw**2
            )
        return total
