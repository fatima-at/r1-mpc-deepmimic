import argparse

import numpy as np
import torch
import mujoco

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv, DEFAULT_CFG
from deepmimic.policy import ActorCritic
from mpc.metrics import debounce, BANDS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", default="data/qcheck/gait_v6.h5")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--max-seam", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def foot_contacts(model, data, foot_ids):
    flags = np.zeros(len(foot_ids), dtype=np.float32)
    for c in range(data.ncon):
        con = data.contact[c]
        for gid in (con.geom1, con.geom2):
            bid = int(model.geom_bodyid[gid])
            for i, fb in enumerate(foot_ids):
                if bid == fb:
                    flags[i] = 1.0
    return flags


def metrics(Q, Z, C, A, dt):
    n_feet = C.shape[1]
    C = np.stack([debounce(C[:, i], dt) for i in range(n_feet)], axis=1)
    duty = C.mean(axis=0)
    td = sum(int(np.sum((C[1:, i] > 0.5) & (C[:-1, i] < 0.5)))
             for i in range(n_feet))
    dur = len(Z) * dt
    dist = float(np.hypot(Q[-1, 0] - Q[0, 0], Q[-1, 1] - Q[0, 1]))
    z = Z - Z.min(axis=0, keepdims=True)
    d1 = np.diff(A, axis=0) if len(A) > 1 else np.zeros((1, A.shape[1]))
    d2 = np.diff(A, n=2, axis=0) if len(A) > 2 else np.zeros((1, A.shape[1]))
    return {
        "vx": float(Q[-1, 0] - Q[0, 0]) / dur,
        "clearance_p90": float(np.mean(np.percentile(z, 90, axis=0))),
        "clearance_max": float(z.max()),
        "duty_asym": float(abs(duty[0] - duty[1])),
        "double_support": float(np.mean(C.sum(axis=1) == n_feet)),
        "flight": float(np.mean(C.sum(axis=1) == 0)),
        "base_h_std": float(Q[:, 2].std()),
        "cadence": td / max(dur, 1e-9),
        "step_len": dist / td if td else float("nan"),
        "lat_drift": abs(float(Q[-1, 1] - Q[0, 1])),
        "act_rate": float(np.sqrt(np.mean(d1 ** 2)) / dt),
        "act_accel": float(np.sqrt(np.mean(d2 ** 2)) / dt ** 2),
    }


def main():
    args = parse_args()
    ref = ReferenceMotions(args.dataset, cyclic=True, max_seam=args.max_seam)
    cfg = dict(DEFAULT_CFG)
    cfg["max_steps"] = args.steps
    env = DeepMimicEnv(ref, cfg=cfg, seed=args.seed)
    foot_ids = [int(env.model.geom_bodyid[env.model.sensor_objid[
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, n)]])
        if False else 0 for n in []]
    names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(env.model.nbody)]
    foot_ids = [i for i, n in enumerate(names)
                if n and ("ankle_roll" in n or "foot" in n)][:2]

    ac = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ac = ActorCritic(env.obs_dim, env.n_act,
                         hidden=tuple(ck["cfg"]["ppo"]["hidden"]),
                         log_std_init=ck["cfg"]["ppo"]["log_std_init"])
        ac.load_state_dict(ck["model"])
        ac.eval()

    print("REFERENCE clips (what DeepMimic is being asked to copy):")
    print(f"  {'clip':<9} {'speed':>7} {'clear_p90':>10} {'dbl_sup':>8} "
          f"{'duty_asym':>10} {'cadence':>8}")
    for c in range(len(ref)):
        C = ref.clips[c]["foot_contact"]
        fz = ref.clips[c]["foot_pos"][:, :, 2]
        z = fz - fz.min(axis=0, keepdims=True)
        Cd = np.stack([debounce(C[:, i], env.ctrl_dt) for i in range(2)], axis=1)
        duty = Cd.mean(axis=0)
        td = sum(int(np.sum((Cd[1:, i] > 0.5) & (Cd[:-1, i] < 0.5)))
                 for i in range(2))
        dur = len(C) * env.ctrl_dt
        print(f"  {ref.clips[c]['name']:<9} {ref.clips[c]['command'][0]:>7.3f} "
              f"{np.mean(np.percentile(z, 90, axis=0)):>10.3f} "
              f"{np.mean(Cd.sum(axis=1) == 2):>8.3f} "
              f"{abs(duty[0] - duty[1]):>10.3f} {td / dur:>8.3f}")
    print()

    rows = {}
    for label, pol in (("zero (MPC replay)", lambda o: np.zeros(env.n_act)),
                       ("trained policy",
                        (lambda o: ac.act(o[None, :], deterministic=True)[0][0])
                        if ac else None)):
        if pol is None:
            continue
        agg = []
        for c in range(len(ref)):
            obs = env.reset(clip=c, frame=0)
            Q = [env.data.qpos.copy()]
            Z, C, A = [], [], []
            for _ in range(args.steps):
                a = pol(obs)
                obs, r, term, trunc, info = env.step(a)
                Q.append(env.data.qpos.copy())
                Z.append([env.data.sensordata[p][2] for p, _ in env.foot_slices])
                C.append(foot_contacts(env.model, env.data, foot_ids))
                A.append(env.data.ctrl[env.act_idx].copy())
                if term or trunc:
                    break
            if len(Z) < 20:
                continue
            agg.append(metrics(np.array(Q), np.array(Z), np.array(C),
                               np.array(A), env.ctrl_dt))
        rows[label] = agg

    keys = ["vx", "clearance_p90", "duty_asym", "double_support", "flight",
            "base_h_std", "cadence", "step_len", "lat_drift",
            "act_rate", "act_accel"]
    print(f"  {'metric':<16}" + "".join(f"{k:>20}" for k in rows) +
          f"{'target band':>16}")
    for k in keys:
        line = f"  {k:<16}"
        for label in rows:
            v = [r[k] for r in rows[label] if np.isfinite(r[k])]
            line += f"{np.mean(v):>20.3f}" if v else f"{'-':>20}"
        b = BANDS.get(k)
        line += f"{f'{b[0]}-{b[1]}' if b else '-':>16}"
        print(line)


if __name__ == "__main__":
    main()
