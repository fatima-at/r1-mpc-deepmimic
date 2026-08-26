import argparse

import numpy as np
import torch
import mujoco
import imageio.v2 as iio

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv, DEFAULT_CFG
from deepmimic.policy import ActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", default="data/qcheck/gait_v6.h5")
    p.add_argument("--max-seam", type=float, default=0.08)
    p.add_argument("--clip", type=int, default=None)
    p.add_argument("--speed", type=float, default=0.30)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--skip", type=int, default=8)
    p.add_argument("--start", type=int, default=60)
    p.add_argument("--out", default="data/dm/strip.png")
    p.add_argument("--width", type=int, default=260)
    p.add_argument("--height", type=int, default=340)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-8.0)
    p.add_argument("--distance", type=float, default=2.0)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def render(model, qpos_list, args):
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
            out.append(r.render().copy())
    return out


def main():
    args = parse_args()
    ref = ReferenceMotions(args.dataset, cyclic=True, max_seam=args.max_seam)
    cfg = dict(DEFAULT_CFG)
    cfg["max_steps"] = 900
    env = DeepMimicEnv(ref, cfg=cfg, seed=args.seed)

    c = args.clip if args.clip is not None else int(
        np.argmin(np.abs(ref.speeds - args.speed)))
    label = float(ref.speeds[c])

    if args.reference:
        n = ref.n_frames(c)
        qp = [ref.clips[c]["qpos"][i % n].astype(np.float64)
              for i in range(0, args.frames * args.skip, args.skip)]
        adv = ref.clips[c]["cycle"]["dx"]
        for j, q in enumerate(qp):
            q[0] += adv * ((j * args.skip) // n)
        tag = f"reference clip {ref.clips[c]['name']} label {label:+.3f}"
        traj = qp
    else:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ac = ActorCritic(env.obs_dim, env.n_act,
                         hidden=tuple(ck["cfg"]["ppo"]["hidden"]),
                         log_std_init=ck["cfg"]["ppo"]["log_std_init"])
        ac.load_state_dict(ck["model"])
        ac.eval()
        obs = env.reset(clip=c, frame=0)
        traj_all = [env.data.qpos.copy()]
        for _ in range(900):
            a, _, _ = ac.act(obs[None, :], deterministic=True)
            obs, r, term, trunc, info = env.step(a[0])
            traj_all.append(env.data.qpos.copy())
            if term or trunc:
                break
        idx = [min(len(traj_all) - 1, args.start + j * args.skip)
               for j in range(args.frames)]
        traj = [traj_all[i] for i in idx]
        tag = (f"policy clip {ref.clips[c]['name']} label {label:+.3f} "
               f"survived {len(traj_all)}")

    frames = render(env.model, traj, args)
    strip = np.concatenate(frames, axis=1)
    iio.imwrite(args.out, strip)
    print(tag)
    print(f"wrote {args.out}  {strip.shape[1]}x{strip.shape[0]}  "
          f"{len(frames)} frames, every {args.skip} ticks "
          f"({args.skip * env.ctrl_dt:.2f} s apart)")


if __name__ == "__main__":
    main()
