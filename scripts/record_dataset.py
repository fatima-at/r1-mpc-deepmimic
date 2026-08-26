import argparse
import os

import numpy as np
import h5py
import mujoco

from mpc.cem import CEMPlanner
from mpc.env import load_model, reset, base_state
from mpc.actuators import name_mask
from mpc.posture import crouch_qpos
from mpc.sensors import foot_sensor_slices, find_foot_bodies_model
from mpc.tasks.locomotion import LocomotionTask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="assets/r1/scene.xml")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--command-lo", type=float, default=0.0)
    p.add_argument("--command-hi", type=float, default=0.5)
    p.add_argument("--commands", type=float, nargs="*", default=None)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--settle", type=float, default=1.5)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--elite", type=int, default=20)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--n-knots", type=int, default=10)
    p.add_argument("--n-thread", type=int, default=8)
    p.add_argument("--vel-mode", default="instant", choices=["mean", "instant"])
    p.add_argument("--crouch", action="store_true")
    p.add_argument("--no-gait", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-falls", action="store_true")
    p.add_argument("--weight", nargs="*", default=[], metavar="KEY=VAL")
    return p.parse_args()


def foot_contacts(model, data, foot_body_ids):
    flags = np.zeros(len(foot_body_ids), dtype=np.float32)
    for c in range(data.ncon):
        con = data.contact[c]
        for gid in (con.geom1, con.geom2):
            bid = int(model.geom_bodyid[gid])
            for i, fb in enumerate(foot_body_ids):
                if bid == fb:
                    flags[i] = 1.0
    return flags


def record_episode(args, model, command, seed):
    data = mujoco.MjData(model)
    reset(model, data, "none")

    gait_cfg = {"min_speed": 1e9} if args.no_gait else None
    qnom = crouch_qpos(model) if args.crouch else None
    overrides = {k: float(v) for k, v in (w.split("=") for w in args.weight)}
    task = LocomotionTask(model, gait=gait_cfg, q_nominal=qnom,
                          vel_mode=args.vel_mode, weights=overrides or None)
    if qnom is not None:
        data.qpos[:] = qnom
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    planner = CEMPlanner(
        model, task,
        horizon=args.horizon, n_samples=args.samples, n_elite=args.elite,
        n_iters=args.iters, n_knots=args.n_knots, n_thread=args.n_thread,
        seed=seed, free_mask=name_mask(model, ["hip", "knee", "ankle"]),
    )

    foot_names = find_foot_bodies_model(model)
    foot_slices = foot_sensor_slices(model, foot_names)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in foot_names]

    n = int(round(args.duration / planner.ctrl_dt))
    rec = {k: [] for k in (
        "time", "qpos", "qvel", "qref", "tau", "lin_vel_body", "ang_vel_body",
        "foot_pos", "foot_vel", "foot_contact")}
    fell = False

    for _ in range(n):
        action = planner.plan(data, command)
        data.ctrl[:] = action
        for _ in range(planner.n_substeps):
            mujoco.mj_step(model, data)

        st = base_state(model, data)
        rec["time"].append(data.time)
        rec["qpos"].append(data.qpos.copy())
        rec["qvel"].append(data.qvel.copy())
        rec["qref"].append(action.copy())
        rec["tau"].append(data.actuator_force.copy())
        rec["lin_vel_body"].append(st["lin_vel_body"].copy())
        rec["ang_vel_body"].append(st["ang_vel_body"].copy())
        rec["foot_pos"].append(
            np.array([data.sensordata[p] for p, _ in foot_slices]))
        rec["foot_vel"].append(
            np.array([data.sensordata[v] for _, v in foot_slices]))
        rec["foot_contact"].append(foot_contacts(model, data, foot_ids))

        if data.qpos[2] < task.fall_height:
            fell = True
            break

    planner.close()
    out = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}

    settle_i = int(round(args.settle / planner.ctrl_dt))
    steady = out["lin_vel_body"][settle_i:] if len(out["time"]) > settle_i else None
    achieved = (steady.mean(axis=0) if steady is not None and len(steady)
                else np.full(3, np.nan, dtype=np.float32))
    yaw = out["ang_vel_body"][settle_i:, 2] if steady is not None and len(steady) else None

    meta = {
        "command": np.asarray(command, dtype=np.float32),
        "achieved_lin": achieved.astype(np.float32),
        "achieved_yaw": np.float32(np.mean(yaw) if yaw is not None and len(yaw) else np.nan),
        "seed": seed,
        "fell": bool(fell),
        "t_end": float(out["time"][-1]) if len(out["time"]) else 0.0,
        "settle_index": settle_i,
        "ctrl_dt": planner.ctrl_dt,
    }
    return out, meta


def main():
    args = parse_args()
    model = load_model(args.model)

    if args.commands is not None:
        cmds = list(args.commands)
    else:
        rng = np.random.default_rng(1234 + args.start_index)
        cmds = list(rng.uniform(args.command_lo, args.command_hi, args.episodes))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    kept = 0
    with h5py.File(args.out, "w") as f:
        g_meta = f.create_group("meta")
        g_meta.attrs["model"] = args.model
        g_meta.attrs["nq"] = model.nq
        g_meta.attrs["nv"] = model.nv
        g_meta.attrs["nu"] = model.nu
        g_meta.attrs["vel_mode"] = args.vel_mode
        g_meta.attrs["samples"] = args.samples
        g_meta.attrs["horizon"] = args.horizon
        g_meta.attrs["n_knots"] = args.n_knots
        g_meta.attrs["crouch"] = args.crouch
        g_meta.attrs["gait"] = not args.no_gait
        g_meta.attrs["weight_overrides"] = " ".join(args.weight)
        g_meta.attrs["joint_names"] = np.array(
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
             for j in range(model.njnt)], dtype=h5py.string_dtype())
        g_meta.attrs["actuator_names"] = np.array(
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
             for a in range(model.nu)], dtype=h5py.string_dtype())

        eps = f.create_group("episodes")
        for i, cx in enumerate(cmds):
            idx = args.start_index + i
            command = np.array([float(cx), 0.0, 0.0])
            seed = args.seed + idx
            out, meta = record_episode(args, model, command, seed)

            status = "FELL" if meta["fell"] else "ok"
            print(f"ep {idx:04d}  cmd {cx:+.3f}  achieved "
                  f"{meta['achieved_lin'][0]:+.3f}  t_end {meta['t_end']:.2f}  "
                  f"{status}", flush=True)

            if meta["fell"] and not args.keep_falls:
                continue

            g = eps.create_group(f"ep_{idx:04d}")
            for k, v in out.items():
                g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            g.attrs["command"] = meta["command"]
            g.attrs["achieved_lin"] = meta["achieved_lin"]
            g.attrs["achieved_yaw"] = meta["achieved_yaw"]
            g.attrs["seed"] = meta["seed"]
            g.attrs["fell"] = meta["fell"]
            g.attrs["t_end"] = meta["t_end"]
            g.attrs["settle_index"] = meta["settle_index"]
            g.attrs["ctrl_dt"] = meta["ctrl_dt"]
            kept += 1
            g_meta.attrs["n_episodes"] = kept
            f.flush()

        g_meta.attrs["n_episodes"] = kept

    print(f"wrote {kept}/{len(cmds)} episodes to {args.out}")


if __name__ == "__main__":
    main()
