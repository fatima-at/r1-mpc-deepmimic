import numpy as np


BANDS = {
    "clearance_p90": (0.03, 0.08),
    "clearance_max": (0.0, 0.15),
    "frac_high": (0.0, 0.02),
    "duty_asym": (0.0, 0.10),
    "double_support": (0.15, 0.35),
    "flight": (0.0, 0.02),
    "base_h_std": (0.0, 0.02),
    "step_len": (0.15, 0.35),
    "cadence": (1.0, 1.7),
    "yaw_drift": (0.0, 1.0),
    "lat_drift": (0.0, 0.05),
}

SCALES = {
    "clearance_p90": 0.03,
    "clearance_max": 0.05,
    "frac_high": 0.02,
    "duty_asym": 0.05,
    "double_support": 0.10,
    "flight": 0.02,
    "base_h_std": 0.01,
    "step_len": 0.10,
    "cadence": 0.5,
    "yaw_drift": 1.0,
    "lat_drift": 0.05,
}


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def debounce(sig, dt, min_air=0.12, min_contact=0.08):
    x = (np.asarray(sig) > 0.5).astype(np.int8).copy()
    n_air = max(1, int(round(min_air / dt)))
    n_con = max(1, int(round(min_contact / dt)))

    i = 0
    while i < len(x):
        if x[i] == 0:
            j = i
            while j < len(x) and x[j] == 0:
                j += 1
            if i > 0 and j < len(x) and (j - i) <= n_air:
                x[i:j] = 1
            i = j
        else:
            i += 1

    i = 0
    while i < len(x):
        if x[i] == 1:
            j = i
            while j < len(x) and x[j] == 1:
                j += 1
            if (j - i) < n_con:
                x[i:j] = 0
            i = j
        else:
            i += 1
    return x.astype(np.float64)


def episode_metrics(qpos, foot_pos, foot_contact, times, achieved_vx):
    z = foot_pos[:, :, 2] - foot_pos[0, :, 2]
    n_feet = foot_contact.shape[1]
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.02
    foot_contact = np.stack(
        [debounce(foot_contact[:, i], dt) for i in range(n_feet)], axis=1
    )
    duty = foot_contact.mean(axis=0)

    touchdowns = 0
    for i in range(n_feet):
        sig = foot_contact[:, i]
        touchdowns += int(np.sum((sig[1:] > 0.5) & (sig[:-1] < 0.5)))
    dur = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    dist = float(np.hypot(qpos[-1, 0] - qpos[0, 0], qpos[-1, 1] - qpos[0, 1]))
    dyaw = np.degrees(yaw_of(qpos[-1, 3:7]) - yaw_of(qpos[0, 3:7]))

    return {
        "vx": float(achieved_vx),
        "clearance_p90": float(np.mean(np.percentile(z, 90, axis=0))),
        "clearance_max": float(z.max()),
        "frac_high": float(np.mean(z > 0.15)),
        "duty_asym": float(abs(duty[0] - duty[1])),
        "double_support": float(np.mean(foot_contact.sum(axis=1) == n_feet)),
        "flight": float(np.mean(foot_contact.sum(axis=1) == 0)),
        "base_h_std": float(qpos[:, 2].std()),
        "cadence": touchdowns / max(dur, 1e-9),
        "step_len": dist / touchdowns if touchdowns else 0.0,
        "yaw_drift": abs(float(dyaw)) / max(dur, 1e-9),
        "lat_drift": abs(float(qpos[-1, 1] - qpos[0, 1])),
    }


def violation(metrics):
    total = 0.0
    parts = {}
    for k, (lo, hi) in BANDS.items():
        v = metrics.get(k)
        if v is None or not np.isfinite(v):
            parts[k] = 5.0
            total += 5.0
            continue
        s = SCALES[k]
        d = max(lo - v, 0.0) + max(v - hi, 0.0)
        parts[k] = d / s
        total += d / s
    return total, parts
