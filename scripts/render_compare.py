import argparse
import os

import numpy as np
import torch
import mujoco
import imageio.v2 as iio

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv, DEFAULT_CFG
from deepmimic.policy import ActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="data/qcheck/gait_v6.h5")
    p.add_argument("--max-seam", type=float, default=0.08)
    p.add_argument("--clip", type=int, default=None)
    p.add_argument("--speed", type=float, default=0.28)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", default="data/dm/compare.mp4")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--width", type=int, default=420)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--distance", type=float, default=2.6)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def frames_from(model, qpos_list, args, tint=None):
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.distance = args.distance
    out = []
    with mujoco.Renderer(model, args.height, args.width) as r:
        for q in qpos_list:
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [q[0], q[1], 0.45]
            r.update_scene(data, cam)
            img = r.render().copy()
            if tint is not None:
                img[:6, :, :] = tint
            out.append(img)
    return out


def main():
    args = parse_args()
    ref = ReferenceMotions(args.dataset, cyclic=True, max_seam=args.max_seam)
    cfg = dict(DEFAULT_CFG)
    cfg["max_steps"] = args.steps
    env = DeepMimicEnv(ref, cfg=cfg, seed=args.seed)

    c = args.clip if args.clip is not None else int(
        np.argmin(np.abs(ref.speeds - args.speed)))
    label = float(ref.speeds[c])

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ac = ActorCritic(env.obs_dim, env.n_act,
                     hidden=tuple(ck["cfg"]["ppo"]["hidden"]),
                     log_std_init=ck["cfg"]["ppo"]["log_std_init"])
    ac.load_state_dict(ck["model"])
    ac.eval()

    obs = env.reset(clip=c, frame=0)
    pol_q = [env.data.qpos.copy()]
    q0 = env.data.qpos[:2].copy()
    for _ in range(args.steps):
        a, _, _ = ac.act(obs[None, :], deterministic=True)
        obs, r, term, trunc, info = env.step(a[0])
        pol_q.append(env.data.qpos.copy())
        if term or trunc:
            break
    reason = info["reason"] if info["reason"] else "reached cap"
    dist = float(env.data.qpos[0] - q0[0])
    dur = (len(pol_q) - 1) * env.ctrl_dt

    n = ref.n_frames(c)
    dx = ref.clips[c]["cycle"]["dx"]
    ref_q = []
    for i in range(len(pol_q)):
        q = ref.clips[c]["qpos"][i % n].astype(np.float64).copy()
        q[0] += dx * (i // n)
        ref_q.append(q)

    idx = range(0, len(pol_q), args.every)
    fp = frames_from(env.model, [pol_q[i] for i in idx], args,
                     tint=np.array([80, 140, 255], dtype=np.uint8))
    fr = frames_from(env.model, [ref_q[i] for i in idx], args,
                     tint=np.array([120, 255, 140], dtype=np.uint8))
    both = [np.concatenate([b, a], axis=1) for b, a in zip(fr, fp)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    iio.mimsave(args.out, both, fps=args.fps, quality=8, macro_block_size=1)
    print(f"clip {ref.clips[c]['name']}  label {label:+.3f} m/s")
    print(f"LEFT = reference (green bar)   RIGHT = policy (blue bar)")
    print(f"policy survived {len(pol_q) - 1} steps ({dur:.2f} s), {reason}")
    print(f"policy travelled {dist:+.2f} m  mean speed {dist / max(dur, 1e-9):+.3f} m/s")
    print(f"wrote {args.out}  ({len(both)} frames)")


if __name__ == "__main__":
    main()
