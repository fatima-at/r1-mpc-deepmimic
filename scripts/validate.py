import argparse
import time

import numpy as np
import mujoco

from mpc.cem import CEMPlanner
from mpc.env import load_model, reset, base_state
from mpc.actuators import name_mask
from mpc.tasks.locomotion import LocomotionTask


def make(model_path, seed, horizon, samples, elite, iters, threads, free_joints,
         n_knots, vel_mode, gait):
    model = load_model(model_path)
    data = mujoco.MjData(model)
    reset(model, data, "standing")
    gait_cfg = None if gait else {"min_speed": 1e9}
    task = LocomotionTask(model, gait=gait_cfg, vel_mode=vel_mode)
    planner = CEMPlanner(
        model,
        task,
        horizon=horizon,
        n_samples=samples,
        n_elite=elite,
        n_iters=iters,
        n_thread=threads,
        seed=seed,
        free_mask=name_mask(model, free_joints),
        n_knots=n_knots,
    )
    return model, data, task, planner


def episode(model, data, task, planner, command, duration, push=None):
    n = int(round(duration / planner.ctrl_dt))
    push_tick = None if push is None else int(round(push[0] / planner.ctrl_dt))
    heights, uprights, vels = [], [], []
    fell_at = None
    for tick in range(n):
        if push_tick is not None and tick == push_tick:
            data.qvel[0] += push[1]
        action = planner.plan(data, command)
        data.ctrl[:] = action
        for _ in range(planner.n_substeps):
            mujoco.mj_step(model, data)
        st = base_state(model, data)
        heights.append(st["height"])
        uprights.append(st["upright"])
        vels.append(st["lin_vel_body"].copy())
        if st["height"] < task.fall_height:
            fell_at = data.time
            break
    return {
        "survived": fell_at is None,
        "fell_at": fell_at,
        "t_end": data.time,
        "height": np.array(heights),
        "upright": np.array(uprights),
        "vel": np.array(vels),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--elite", type=int, default=20)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-knots", type=int, default=10)
    ap.add_argument("--vel-mode", default="instant", choices=["mean", "instant"])
    ap.add_argument("--no-gait", action="store_true")
    ap.add_argument("--free-joints", nargs="*", default=["hip", "knee", "ankle"])
    ap.add_argument("--v4-duration", type=float, default=8.0)
    ap.add_argument("--v4-seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--v5-duration", type=float, default=4.0)
    ap.add_argument("--v5-push", type=float, default=0.5)
    ap.add_argument("--v6-duration", type=float, default=4.0)
    ap.add_argument("--v6-commands", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--v6-settle", type=float, default=1.5)
    args = ap.parse_args()

    cfg = dict(
        horizon=args.horizon,
        samples=args.samples,
        elite=args.elite,
        iters=args.iters,
        threads=args.threads,
        free_joints=args.free_joints,
        n_knots=args.n_knots,
        vel_mode=args.vel_mode,
        gait=not args.no_gait,
    )
    print("CONFIG  " + "  ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
    t0 = time.perf_counter()

    print("=" * 72)
    print("V4  STANDING BALANCE   (zero command)")
    print("=" * 72)
    v4 = []
    for seed in args.v4_seeds:
        m, d, tk, pl = make(args.model, seed, **cfg)
        r = episode(m, d, tk, pl, np.zeros(3), args.v4_duration)
        pl.close()
        h, u = r["height"], r["upright"]
        ok = r["survived"] and abs(h[-1] - tk.height_target) < 0.05 and u.min() > 0.90
        v4.append(ok)
        print(f"  seed {seed}: survived {r['survived']}  t_end {r['t_end']:.2f}s  "
              f"h_end {h[-1]:.3f} (nom {tk.height_target:.3f})  h_min {h.min():.3f}  "
              f"upright_min {u.min():.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"  V4 {'PASS' if all(v4) else 'FAIL'}")
    print()

    print("=" * 72)
    print(f"V5  DISTURBANCE RECOVERY   (+{args.v5_push} m/s impulse at t=1.0s)")
    print("=" * 72)
    m, d, tk, pl = make(args.model, 0, **cfg)
    r = episode(m, d, tk, pl, np.zeros(3), args.v5_duration, push=(1.0, args.v5_push))
    pl.close()
    v5 = r["survived"] and r["upright"].min() > 0.80
    print(f"  survived {r['survived']}  t_end {r['t_end']:.2f}s  "
          f"h_min {r['height'].min():.3f}  upright_min {r['upright'].min():.3f}  "
          f"{'PASS' if v5 else 'FAIL'}")
    print()

    print("=" * 72)
    print("V6  COMMAND TRACKING")
    print("=" * 72)
    print(f"  {'cmd vx':>8} {'achieved':>10} {'err':>8} {'outcome':>9} "
          f"{'t_end':>7} {'h_min':>8}")
    valid = []
    n_fell = 0
    settle_ticks = int(round(args.v6_settle / 0.02))
    for c in args.v6_commands:
        m, d, tk, pl = make(args.model, 0, **cfg)
        r = episode(m, d, tk, pl, np.array([c, 0.0, 0.0]), args.v6_duration)
        pl.close()
        if r["survived"] and len(r["vel"]) > settle_ticks:
            vx = float(np.mean(r["vel"][settle_ticks:, 0]))
            valid.append((c, vx))
            shown = f"{vx:>10.3f}"
            err = f"{vx - c:>8.3f}"
        else:
            n_fell += 1
            shown = f"{'--':>10}"
            err = f"{'--':>8}"
        print(f"  {c:>8.2f} {shown} {err} "
              f"{('ok' if r['survived'] else 'FELL'):>9} "
              f"{r['t_end']:>7.2f} {r['height'].min():>8.3f}")

    if len(valid) >= 2:
        monotone = all(
            valid[i + 1][1] >= valid[i][1] - 0.02 for i in range(len(valid) - 1)
        )
        max_err = max(abs(v - c) for c, v in valid)
    else:
        monotone = False
        max_err = float("nan")
    span = (max(v for _, v in valid) - min(v for _, v in valid)) if len(valid) >= 2 else 0.0
    v6 = n_fell == 0 and monotone and span > 0.10
    print(f"  surviving {len(valid)}/{len(args.v6_commands)}   "
          f"monotone {monotone}   achieved span {span:.3f}   "
          f"(gain {span / max(args.v6_commands[-1] - args.v6_commands[0], 1e-9):.2f})   "
          f"{'PASS' if v6 else 'FAIL'}")
    print("  criterion: no falls, monotone response, usable achieved range")
    print("  (absolute tracking error is corrected downstream by relabelling)")
    print("  (achieved velocity reported only for surviving episodes)")
    print()

    print("=" * 72)
    print(f"V4 {'PASS' if all(v4) else 'FAIL'}   "
          f"V5 {'PASS' if v5 else 'FAIL'}   "
          f"V6 {'PASS' if v6 else 'FAIL'}")
    print(f"total wall time {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
