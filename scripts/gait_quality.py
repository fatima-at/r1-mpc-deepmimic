import argparse
import glob

import numpy as np
import h5py

from mpc.metrics import debounce


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def episode_metrics(g):
    fp = np.array(g["foot_pos"])
    c = np.array(g["foot_contact"])
    q = np.array(g["qpos"])
    t = np.array(g["time"])
    z = fp[:, :, 2] - fp[0, :, 2]

    n_feet = c.shape[1]
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.02
    c = np.stack([debounce(c[:, i], dt) for i in range(n_feet)], axis=1)
    duty = c.mean(axis=0)

    touchdowns = 0
    for i in range(n_feet):
        sig = c[:, i]
        touchdowns += int(np.sum((sig[1:] > 0.5) & (sig[:-1] < 0.5)))
    dur = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    dist = float(np.hypot(q[-1, 0] - q[0, 0], q[-1, 1] - q[0, 1]))
    dyaw = np.degrees(yaw_of(q[-1, 3:7]) - yaw_of(q[0, 3:7]))

    return {
        "cadence": touchdowns / max(dur, 1e-9),
        "step_len": dist / touchdowns if touchdowns else float("nan"),
        "yaw_drift": abs(float(dyaw)) / max(dur, 1e-9),
        "lat_drift": abs(float(q[-1, 1] - q[0, 1])),
        "vx": float(g.attrs["achieved_lin"][0]),
        "clearance_p90": float(np.mean(np.percentile(z, 90, axis=0))),
        "clearance_max": float(z.max()),
        "frac_high": float(np.mean(z > 0.15)),
        "duty_asym": float(abs(duty[0] - duty[1])),
        "double_support": float(np.mean(c.sum(axis=1) == n_feet)),
        "flight": float(np.mean(c.sum(axis=1) == 0)),
        "base_h_std": float(q[:, 2].std()),
        "travel": float(q[-1, 0] - q[0, 0]),
    }


def summarize(paths, label, vmin):
    rows = []
    for path in paths:
        with h5py.File(path, "r") as f:
            if "episodes" not in f:
                continue
            for name in f["episodes"]:
                m = episode_metrics(f["episodes"][name])
                if abs(m["vx"]) >= vmin:
                    rows.append(m)
    if not rows:
        print(f"{label}: no episodes with |vx| >= {vmin}")
        return None

    keys = ["vx", "clearance_p90", "clearance_max", "frac_high",
            "duty_asym", "double_support", "flight", "base_h_std",
            "cadence", "step_len", "yaw_drift", "lat_drift"]
    agg = {k: np.array([r[k] for r in rows]) for k in keys}
    print(f"\n{label}   ({len(rows)} episodes with |vx| >= {vmin})")
    print(f"  {'metric':<16} {'mean':>8} {'median':>8} {'worst':>8}   target")
    targets = {
        "vx": "-",
        "clearance_p90": "0.03-0.08",
        "clearance_max": "< 0.15",
        "frac_high": "~0.00",
        "duty_asym": "< 0.10",
        "double_support": "0.15-0.35",
        "flight": "~0.00",
        "base_h_std": "< 0.02",
        "cadence": "1.0-1.7 /s",
        "step_len": "0.15-0.35 m",
        "yaw_drift": "< 1 deg/s",
        "lat_drift": "< 0.05 m",
    }
    for k in keys:
        v = agg[k]
        worst = v.min() if k == "double_support" else v.max()
        print(f"  {k:<16} {v.mean():>8.3f} {np.median(v):>8.3f} {worst:>8.3f}   "
              f"{targets[k]}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+",
                    help="one or more h5 files or globs; each is a group")
    ap.add_argument("--min-speed", type=float, default=0.15)
    args = ap.parse_args()

    for spec in args.datasets:
        paths = sorted(glob.glob(spec)) or [spec]
        summarize(paths, spec, args.min_speed)


if __name__ == "__main__":
    main()
