import argparse
import glob

import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="e.g. 'data/parts/*.h5'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relabel", action="store_true",
                    help="overwrite command with the achieved velocity")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"no files matching {args.pattern}")
        return

    n_out = 0
    cmds, achs = [], []
    with h5py.File(args.out, "w") as dst:
        meta_written = False
        eps = dst.create_group("episodes")
        for path in files:
            with h5py.File(path, "r") as src:
                if not meta_written and "meta" in src:
                    src.copy("meta", dst)
                    meta_written = True
                if "episodes" not in src:
                    continue
                for name in src["episodes"]:
                    g = src["episodes"][name]
                    if name in eps:
                        continue
                    src.copy(g, eps, name=name)
                    a = eps[name].attrs
                    cmds.append(float(a["command"][0]))
                    achs.append(float(a["achieved_lin"][0]))
                    if args.relabel:
                        new = np.array(a["achieved_lin"], dtype=np.float32)
                        a["command_original"] = a["command"]
                        a["command"] = np.array(
                            [new[0], new[1], float(a["achieved_yaw"])],
                            dtype=np.float32)
                    n_out += 1
        dst["meta"].attrs["n_episodes"] = n_out
        dst["meta"].attrs["relabeled"] = bool(args.relabel)

    cmds = np.array(cmds)
    achs = np.array(achs)
    print(f"merged {n_out} episodes from {len(files)} files -> {args.out}")
    if n_out:
        print(f"commanded  min {cmds.min():+.3f}  max {cmds.max():+.3f}")
        print(f"achieved   min {achs.min():+.3f}  max {achs.max():+.3f}")
        ok = np.isfinite(cmds) & np.isfinite(achs) & (cmds > 1e-6)
        if ok.any():
            print(f"gain (achieved/commanded)  mean {np.mean(achs[ok]/cmds[ok]):.3f}")
        print(f"relabeled: {args.relabel}")


if __name__ == "__main__":
    main()
