import numpy as np
import mujoco

from mpc.actuators import to_position_actuators
from mpc.sensors import find_foot_bodies, add_foot_sensors


def load_model(
    path,
    unpin_base=True,
    position_actuators=True,
    kp_scale=10.0,
    kv_ratio=0.2,
    implicit=True,
    foot_sensors=True,
    strip_mesh_collision=False,
):
    spec = mujoco.MjSpec.from_file(str(path))
    if position_actuators:
        to_position_actuators(spec, kp_scale=kp_scale, kv_ratio=kv_ratio)
    feet = []
    if foot_sensors:
        feet = find_foot_bodies(spec)
        if feet:
            add_foot_sensors(spec, feet)
    if strip_mesh_collision:
        for g in spec.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                g.contype = 0
                g.conaffinity = 0

    model = spec.compile()

    if implicit:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    if unpin_base:
        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD:
                continue
            if model.eq_obj1id[i] == 0 or model.eq_obj2id[i] == 0:
                model.eq_active0[i] = 0
    return model


def keyframe_id(model, name):
    if name is None:
        return -1
    for i in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == name:
            return i
    return -1


def reset(model, data, keyframe="standing"):
    mujoco.mj_resetData(model, data)
    kid = keyframe_id(model, keyframe)
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    return data


def settle(model, data, ctrl, seconds=0.0):
    n = int(round(seconds / model.opt.timestep))
    data.ctrl[:] = ctrl
    for _ in range(n):
        mujoco.mj_step(model, data)
    return data


def base_state(model, data):
    quat = np.array(data.qpos[3:7], dtype=np.float64)
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    mat = mat.reshape(3, 3)
    lin_body = mat.T @ np.array(data.qvel[0:3], dtype=np.float64)
    ang_body = np.array(data.qvel[3:6], dtype=np.float64)
    return {
        "height": float(data.qpos[2]),
        "upright": float(mat[2, 2]),
        "lin_vel_body": lin_body,
        "ang_vel_body": ang_body,
    }
