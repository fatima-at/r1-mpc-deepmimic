import numpy as np
import mujoco


def subtree_bodies(model, root):
    out = [root]
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b in out:
                continue
            if int(model.body_parentid[b]) in out:
                out.append(b)
                changed = True
    return out


def leg_pendulum_period(model, hip_joint):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_joint)
    if jid < 0:
        return None
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    pivot = np.array(data.xanchor[jid], dtype=np.float64)
    axis = np.array(data.xaxis[jid], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    bodies = subtree_bodies(model, int(model.jnt_bodyid[jid]))
    m_tot = 0.0
    first = np.zeros(3)
    i_axis = 0.0
    for b in bodies:
        m = float(model.body_mass[b])
        if m <= 0.0:
            continue
        r = np.array(data.xipos[b], dtype=np.float64) - pivot
        rot = np.array(data.ximat[b], dtype=np.float64).reshape(3, 3)
        i_local = np.diag(np.array(model.body_inertia[b], dtype=np.float64))
        i_world = rot @ i_local @ rot.T
        d_perp2 = float(r @ r - (r @ axis) ** 2)
        i_axis += float(axis @ i_world @ axis) + m * d_perp2
        m_tot += m
        first += m * r

    if m_tot <= 0.0 or i_axis <= 0.0:
        return None
    com = first / m_tot
    d = float(np.linalg.norm(com - (com @ axis) * axis))
    if d <= 1e-9:
        return None

    g = float(abs(model.opt.gravity[2])) or 9.81
    t_full = 2.0 * np.pi * np.sqrt(i_axis / (m_tot * g * d))
    return {
        "mass": m_tot,
        "inertia_about_hip": i_axis,
        "com_dist": d,
        "period_full": t_full,
        "period_swing": 0.5 * t_full,
    }


def derive_gait_period(model, duty=0.6, hip_joint=None):
    if hip_joint is None:
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "hip_pitch" in nm and "left" in nm:
                hip_joint = nm
                break
    if hip_joint is None:
        return None
    info = leg_pendulum_period(model, hip_joint)
    if info is None:
        return None
    info["hip_joint"] = hip_joint
    info["duty"] = duty
    info["period_gait"] = info["period_swing"] / max(1.0 - duty, 1e-6)
    return info
