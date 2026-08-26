import numpy as np
import mujoco


def to_position_actuators(spec, kp_scale=10.0, kv_ratio=0.2, kp_min=20.0):
    for act in spec.actuators:
        torque_limit = float(np.max(np.abs(act.ctrlrange)))
        if torque_limit <= 0.0:
            torque_limit = float(np.max(np.abs(act.forcerange)))
        if torque_limit <= 0.0:
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm = [0.0] * 10
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.biasprm = [0.0] * 10
            act.forcerange = [0.0, 0.0]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
            continue

        kp = max(kp_scale * torque_limit, kp_min)
        kv = kv_ratio * kp

        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm = [kp] + [0.0] * 9
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm = [0.0, -kp, -kv] + [0.0] * 7
        act.forcerange = [-torque_limit, torque_limit]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


def actuator_joint_qposadr(model):
    adr = np.full(model.nu, -1, dtype=np.int64)
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = model.actuator_trnid[i, 0]
        adr[i] = model.jnt_qposadr[jid]
    return adr


def actuator_joint_range(model):
    lo = np.full(model.nu, -np.inf)
    hi = np.full(model.nu, np.inf)
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = model.actuator_trnid[i, 0]
        if model.jnt_limited[jid]:
            lo[i] = model.jnt_range[jid, 0]
            hi[i] = model.jnt_range[jid, 1]
    return lo, hi


def is_position_mode(model):
    if model.nu == 0:
        return False
    return bool(
        np.all(np.asarray(model.actuator_biastype) == mujoco.mjtBias.mjBIAS_AFFINE)
    )


def frozen_mask(model):
    return np.asarray(model.actuator_gainprm[:, 0], dtype=np.float64) == 0.0


def name_mask(model, patterns):
    mask = np.zeros(model.nu, dtype=bool)
    if not patterns:
        return ~mask
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        mask[i] = any(p in nm for p in patterns)
    return mask


def nominal_ctrl(model, q_nominal):
    if not is_position_mode(model):
        return np.zeros(model.nu, dtype=np.float64)
    adr = actuator_joint_qposadr(model)
    out = np.zeros(model.nu, dtype=np.float64)
    valid = adr >= 0
    out[valid] = np.asarray(q_nominal, dtype=np.float64)[adr[valid]]
    return out
