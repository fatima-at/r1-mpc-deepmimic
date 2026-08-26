import argparse

import numpy as np
import h5py
import mujoco

from mpc.env import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qcheck/gait_v6.h5")
    p.add_argument("--model", default="assets/r1/scene.xml")
    p.add_argument("--episode", default=None)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--out", default="data/qcheck/walk.gif")
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=440)
    p.add_argument("--height", type=int, default=330)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--distance", type=float, default=2.4)
    p.add_argument("--track", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.data, "r") as f:
        names = sorted(f["episodes"].keys())
        if args.episode:
            name = args.episode
        elif args.speed is not None:
            name = min(names, key=lambda n: abs(
                float(f["episodes"][n].attrs["achieved_lin"][0]) - args.speed))
        else:
            name = max(names, key=lambda n: float(
                f["episodes"][n].attrs["achieved_lin"][0]))
        g = f["episodes"][name]
        qpos = np.array(g["qpos"])
        vx = float(g.attrs["achieved_lin"][0])

    model = load_model(args.model)
    data = mujoco.MjData(model)

    cam = mujoco.MjvCamera()
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    cam.distance = args.distance

    idx = range(0, len(qpos), args.every)
    frames = []
    with mujoco.Renderer(model, args.height, args.width) as r:
        for k in idx:
            data.qpos[:] = qpos[k]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if args.track:
                cam.lookat[:] = [qpos[k, 0], qpos[k, 1], 0.45]
            else:
                cam.lookat[:] = [qpos[len(qpos) // 2, 0], 0.0, 0.45]
            r.update_scene(data, cam)
            frames.append(r.render().copy())

    if args.out.endswith(".mp4"):
        import imageio.v2 as iio
        iio.mimsave(args.out, frames, fps=args.fps, quality=8,
                    macro_block_size=1)
    else:
        from PIL import Image
        imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE,
                                           colors=128) for f in frames]
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0, optimize=True)

    print(f"episode {name}   vx {vx:+.3f} m/s   {len(frames)} frames")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
