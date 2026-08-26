import numpy as np
import mujoco


DEFAULT_CROUCH = {
    "hip_pitch": -0.20,
    "knee": 0.40,
    "ankle_pitch": -0.20,
}


def joint_qposadr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        return -1
    return int(model.jnt_qposadr[jid])


def joint_angles_qpos(model, angles):
    q = np.array(model.qpos0, dtype=np.float64)
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        for key, val in angles.items():
            if key in nm:
                adr = int(model.jnt_qposadr[j])
                if model.jnt_limited[j]:
                    lo, hi = model.jnt_range[j]
                    q[adr] = float(np.clip(val, lo, hi))
                else:
                    q[adr] = float(val)
                break
    return q


def crouch_qpos(model, angles=None, settle_s=1.5, drop_height=None):
    from mpc.actuators import nominal_ctrl

    angles = dict(DEFAULT_CROUCH) if angles is None else dict(angles)
    q = joint_angles_qpos(model, angles)
    if drop_height is not None:
        q[2] = float(drop_height)

    u = nominal_ctrl(model, q)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = u
    for _ in range(int(round(settle_s / model.opt.timestep))):
        mujoco.mj_step(model, data)

    settled = np.array(data.qpos, dtype=np.float64)
    settled[0] = 0.0
    settled[1] = 0.0
    return settled
