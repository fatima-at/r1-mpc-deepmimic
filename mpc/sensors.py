import numpy as np
import mujoco


FOOT_KEYS = ("ankle_roll", "foot")


def find_foot_bodies(spec):
    names = []
    for body in spec.bodies:
        nm = body.name or ""
        if any(k in nm for k in FOOT_KEYS):
            names.append(nm)
    left = [n for n in names if "left" in n or n.startswith("L")]
    right = [n for n in names if "right" in n or n.startswith("R")]
    if len(left) == 1 and len(right) == 1:
        return [left[0], right[0]]
    return names[:2]


def find_foot_bodies_model(model):
    sensed = []
    for i in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if not any(k in nm for k in FOOT_KEYS):
            continue
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"footpos_{nm}") >= 0:
            sensed.append(nm)
    return sensed


def add_foot_sensors(spec, foot_names):
    for nm in foot_names:
        p = spec.add_sensor()
        p.name = f"footpos_{nm}"
        p.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
        p.objtype = mujoco.mjtObj.mjOBJ_BODY
        p.objname = nm

        v = spec.add_sensor()
        v.name = f"footvel_{nm}"
        v.type = mujoco.mjtSensor.mjSENS_FRAMELINVEL
        v.objtype = mujoco.mjtObj.mjOBJ_BODY
        v.objname = nm

        z = spec.add_sensor()
        z.name = f"footzax_{nm}"
        z.type = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
        z.objtype = mujoco.mjtObj.mjOBJ_BODY
        z.objname = nm
    return foot_names


def foot_sensor_slices(model, foot_names):
    out = []
    for nm in foot_names:
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"footpos_{nm}")
        vid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"footvel_{nm}")
        if pid < 0 or vid < 0:
            return []
        pa = int(model.sensor_adr[pid])
        va = int(model.sensor_adr[vid])
        out.append((slice(pa, pa + 3), slice(va, va + 3)))
    return out


def foot_axis_slices(model, foot_names):
    out = []
    for nm in foot_names:
        zid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"footzax_{nm}")
        if zid < 0:
            return []
        za = int(model.sensor_adr[zid])
        out.append(slice(za, za + 3))
    return out


def rest_foot_axis(model, foot_names):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    out = []
    for nm in foot_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"footzax_{nm}")
        if sid < 0:
            return []
        a = int(model.sensor_adr[sid])
        out.append(np.array(data.sensordata[a:a + 3], dtype=np.float64))
    return out


def rest_foot_offset(model, foot_names):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base = np.array(data.qpos[0:3], dtype=np.float64)
    out = []
    for nm in foot_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            return []
        out.append(np.array(data.xipos[bid], dtype=np.float64) - base)
    return out


def rest_foot_height(model, foot_names):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    zs = []
    for nm in foot_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        zs.append(float(data.xipos[bid][2]) if bid >= 0 else 0.0)
    return np.array(zs, dtype=np.float64)
