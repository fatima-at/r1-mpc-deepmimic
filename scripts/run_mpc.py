import argparse
import time

import numpy as np
import mujoco

from mpc.cem import CEMPlanner
from mpc.env import load_model, reset, base_state
from mpc.actuators import name_mask
from mpc.tasks.locomotion import LocomotionTask
from mpc.posture import crouch_qpos


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--keyframe", default="standing")
    p.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--elite", type=int, default=10)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--ctrl-dt", type=float, default=0.02)
    p.add_argument("--n-thread", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--kp-scale", type=float, default=10.0)
    p.add_argument("--kv-ratio", type=float, default=0.2)
    p.add_argument("--torque-actuators", action="store_true")
    p.add_argument("--free-joints", nargs="*", default=["hip", "knee", "ankle"])
    p.add_argument("--n-knots", type=int, default=5)
    p.add_argument("--weight", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--no-gait", action="store_true")
    p.add_argument("--crouch", action="store_true")
    p.add_argument("--vel-mode", default="mean", choices=["mean", "instant"])
    p.add_argument("--settle", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    model = load_model(
        args.model,
        position_actuators=not args.torque_actuators,
        kp_scale=args.kp_scale,
        kv_ratio=args.kv_ratio,
    )
    data = mujoco.MjData(model)
    reset(model, data, args.keyframe)

    overrides = {k: float(v) for k, v in (w.split('=') for w in args.weight)}
    gait_cfg = {'min_speed': 1e9} if args.no_gait else None
    qnom = crouch_qpos(model) if args.crouch else None
    task = LocomotionTask(model, keyframe=args.keyframe, weights=overrides or None, gait=gait_cfg, q_nominal=qnom, vel_mode=args.vel_mode)
    free = name_mask(model, args.free_joints)
    planner = CEMPlanner(
        model,
        task,
        free_mask=free,
        n_knots=args.n_knots,
        horizon=args.horizon,
        n_samples=args.samples,
        n_elite=args.elite,
        n_iters=args.iters,
        ctrl_dt=args.ctrl_dt,
        n_thread=args.n_thread,
        seed=args.seed,
    )

    command = np.array(args.command, dtype=np.float64)
    n_ticks = int(round(args.duration / planner.ctrl_dt))

    print(f"model      {args.model}")
    print(f"nq {model.nq}  nv {model.nv}  nu {model.nu}")
    print(f"physics dt {model.opt.timestep:.4f}  ctrl dt {planner.ctrl_dt:.4f}")
    print(f"substeps   {planner.n_substeps}  horizon {planner.horizon} ticks")
    print(f"rollouts   {args.samples} x {args.iters} iters per tick")
    if overrides:
        print(f"weights    {overrides}")
    print(f"posture    {'CROUCH' if args.crouch else 'qpos0'}  base z {task.height_target:.3f}")
    print(f"gait       {'OFF' if args.no_gait else 'ON'}  feet {task.foot_names}")
    print(f"free       {int(free.sum())}/{model.nu} actuators: {args.free_joints}")
    print(f"command    vx {command[0]:.2f}  vy {command[1]:.2f}  yaw {command[2]:.2f}")
    print()

    viewer = None
    if args.render:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    log = []
    t_start = time.perf_counter()
    try:
        for tick in range(n_ticks):
            t0 = time.perf_counter()
            action = planner.plan(data, command)
            plan_ms = 1e3 * (time.perf_counter() - t0)

            data.ctrl[:] = action
            for _ in range(planner.n_substeps):
                mujoco.mj_step(model, data)

            st = base_state(model, data)
            log.append((data.time, st["height"], st["lin_vel_body"][0], plan_ms))

            if viewer is not None:
                viewer.sync()

            if tick % 10 == 0:
                print(
                    f"t {data.time:5.2f}  h {st['height']:.3f}  "
                    f"up {st['upright']:+.3f}  "
                    f"vx {st['lin_vel_body'][0]:+.3f}  "
                    f"cost {planner.last_cost:9.2f}  plan {plan_ms:6.1f} ms"
                )

            if st["height"] < task.fall_height:
                print(f"\nfell at t = {data.time:.2f} s")
                break
    finally:
        if viewer is not None:
            viewer.close()
        planner.close()

    wall = time.perf_counter() - t_start
    log = np.array(log)
    print()
    print(f"simulated  {log[-1, 0]:.2f} s in {wall:.1f} s wall")
    print(f"mean plan  {log[:, 3].mean():.1f} ms per tick")
    mask = log[:, 0] >= args.settle
    if not mask.any():
        mask = np.ones(len(log), dtype=bool)
    print(f"mean vx    {log[:, 2].mean():+.3f} m/s  (command {command[0]:+.3f})")
    print(f"steady vx  {log[mask, 2].mean():+.3f} m/s  (after {args.settle:.1f}s)")
    print(f"min height {log[:, 1].min():.3f} m")
    print(f"RESULT vx={log[mask, 2].mean():+.4f} minh={log[:, 1].min():.4f} "
          f"tend={log[-1, 0]:.2f} fell={int(log[-1, 1] < task.fall_height)}")


if __name__ == "__main__":
    main()
