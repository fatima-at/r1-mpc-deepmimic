import argparse

import numpy as np
import torch
import mujoco

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv, DEFAULT_CFG
from deepmimic.policy import ActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", default="data/qcheck/gait_v6.h5")
    p.add_argument("--model", default="assets/r1/scene.xml")
    p.add_argument("--out", default="data/dm/policy.mp4")
    p.add_argument("--clip", type=int, default=None)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=520)
    p.add_argument("--height", type=int, default=380)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--distance", type=float, default=2.6)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--acyclic", action="store_true")
    p.add_argument("--max-seam", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    ref = ReferenceMotions(args.dataset, cyclic=not args.acyclic,
                           max_seam=args.max_seam)
    cfg = dict(DEFAULT_CFG)
    cfg["max_steps"] = args.steps
    env = DeepMimicEnv(ref, model_path=args.model, cfg=cfg, seed=args.seed)

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ac = ActorCritic(env.obs_dim, env.n_act,
                         hidden=tuple(ck["cfg"]["ppo"]["hidden"]),
                         log_std_init=ck["cfg"]["ppo"]["log_std_init"])
        ac.load_state_dict(ck["model"])
        ac.eval()

        def policy(o):
            a, _, _ = ac.act(o[None, :], deterministic=not args.stochastic)
            return a[0]
    else:
        def policy(o):
            return np.zeros(env.n_act)

    c = args.clip if args.clip is not None else int(np.argmax(ref.speeds))
    obs = env.reset(clip=c, frame=ref.clips[c]["settle"])
    qpos, rewards = [env.data.qpos.copy()], []
    reason = "truncated"
    for _ in range(args.steps):
        obs, r, term, trunc, info = env.step(policy(obs))
        qpos.append(env.data.qpos.copy())
        rewards.append(r)
        if term or trunc:
            reason = info["reason"] if info["reason"] else "truncated"
            break

    qpos = np.array(qpos)
    dist = float(qpos[-1, 0] - qpos[0, 0])
    dur = len(rewards) * env.ctrl_dt
    print(f"clip {ref.clips[c]['name']}  label {ref.clips[c]['command'][0]:+.3f} m/s")
    print(f"survived {len(rewards)} steps ({dur:.2f} s), ended: {reason}")
    print(f"travelled {dist:+.3f} m  mean speed {dist / max(dur, 1e-9):+.3f} m/s  "
          f"mean reward {np.mean(rewards):.3f}")

    model = env.model
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.distance = args.distance

    frames = []
    with mujoco.Renderer(model, args.height, args.width) as r:
        for k in range(0, len(qpos), args.every):
            data.qpos[:] = qpos[k]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [qpos[k, 0], qpos[k, 1], 0.45]
            r.update_scene(data, cam)
            frames.append(r.render().copy())

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if args.out.endswith(".mp4"):
        import imageio.v2 as iio
        iio.mimsave(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
    else:
        import imageio.v2 as iio
        iio.mimsave(args.out, frames, duration=1.0 / args.fps)
    print(f"wrote {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
