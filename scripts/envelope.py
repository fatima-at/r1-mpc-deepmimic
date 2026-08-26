import argparse
import time

import numpy as np
import mujoco

from mpc.cem import CEMPlanner
from mpc.env import load_model, reset, base_state
from mpc.actuators import name_mask
from mpc.posture import crouch_qpos
from mpc.tasks.locomotion import LocomotionTask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--commands", type=float, nargs="*",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--settle", type=float, default=1.5)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--elite", type=int, default=10)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--n-knots", type=int, default=10)
    p.add_argument("--n-thread", type=int, default=8)
    p.add_argument("--vel-mode", default="mean", choices=["mean", "instant"])
    p.add_argument("--crouch", action="store_true")
    p.add_argument("--gait", action="store_true")
    return p.parse_args()


def episode(args, model, cmd, seed):
    data = mujoco.MjData(model)
    reset(model, data, "none")
    gait_cfg = None if args.gait else {"min_speed": 1e9}
    qnom = crouch_qpos(model) if args.crouch else None
    task = LocomotionTask(model, gait=gait_cfg, q_nominal=qnom,
                          vel_mode=args.vel_mode)
    if qnom is not None:
        data.qpos[:] = qnom
        mujoco.mj_forward(model, data)

    planner = CEMPlanner(
        model, task,
        horizon=args.horizon, n_samples=args.samples, n_elite=args.elite,
        n_iters=args.iters, n_knots=args.n_knots, n_thread=args.n_thread,
        seed=seed, free_mask=name_mask(model, ["hip", "knee", "ankle"]),
    )

    command = np.array([cmd, 0.0, 0.0])
    n = int(round(args.duration / planner.ctrl_dt))
    settle = int(round(args.settle / planner.ctrl_dt))
    vx, lift = [], []
    fell = False
    for _ in range(n):
        action = planner.plan(data, command)
        data.ctrl[:] = action
        for _ in range(planner.n_substeps):
            mujoco.mj_step(model, data)
        vx.append(base_state(model, data)["lin_vel_body"][0])
        if task.foot_slices:
            zs = [float(data.sensordata[sl][2]) - task.foot_rest_z[i]
                  for i, (sl, _) in enumerate(task.foot_slices)]
            lift.append(max(zs))
        if data.qpos[2] < task.fall_height:
            fell = True
            break
    planner.close()

    vx = np.array(vx)
    lift = np.array(lift) if lift else np.zeros(1)
    steady = vx[settle:] if len(vx) > settle else np.array([np.nan])
    return {
        "fell": fell,
        "vx": float(np.mean(steady)),
        "lift_frac": float(np.mean(lift > 0.01)),
        "lift_max": float(np.max(lift)),
    }


def main():
    args = parse_args()
    model = load_model(args.model)
    t0 = time.perf_counter()

    print("CONFIG  " + "  ".join(
        f"{k}={v}" for k, v in sorted(vars(args).items()) if k != "model"))
    print()
    print(f"{'cmd':>6} {'achieved':>20} {'err':>8} {'surv':>6} "
          f"{'lift_frac':>10} {'lift_max':>9}")

    rows = []
    for cmd in args.commands:
        res = [episode(args, model, cmd, s) for s in args.seeds]
        surv = [r for r in res if not r["fell"]]
        n_ok = len(surv)
        if n_ok:
            v = np.array([r["vx"] for r in surv])
            lf = float(np.mean([r["lift_frac"] for r in surv]))
            lm = float(np.max([r["lift_max"] for r in surv]))
            shown = f"{v.mean():+.3f} +/- {v.std():.3f}"
            err = f"{v.mean() - cmd:+.3f}"
        else:
            v = np.array([np.nan])
            lf = lm = float("nan")
            shown = f"{'--':>18}"
            err = f"{'--':>8}"
        rows.append((cmd, float(np.nanmean(v)), n_ok))
        print(f"{cmd:>6.2f} {shown:>20} {err:>8} {n_ok}/{len(args.seeds):>4} "
              f"{lf:>10.2f} {lm:>9.3f}")

    print()
    ok = [(c, v) for c, v, n in rows if n == len(args.seeds) and not np.isnan(v)]
    if len(ok) >= 2:
        good = [c for c, v in ok if abs(v - c) <= 0.10]
        print(f"commands tracked within 0.10 m/s at all seeds: "
              f"{good if good else 'none'}")
        if good:
            print(f"recommended dataset command range: 0.0 to {max(good):.2f} m/s")
    print(f"total wall time {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
