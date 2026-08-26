import argparse

import numpy as np

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv, DEFAULT_CFG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/qcheck/gait_v6.h5")
    p.add_argument("--states", type=int, default=12)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--action-scale", type=float, default=0.30)
    p.add_argument("--noise", type=float, nargs="*",
                   default=[0.0, 0.05, 0.15, 0.35, 0.70])
    p.add_argument("--corr", type=int, default=5)
    p.add_argument("--acyclic", action="store_true")
    p.add_argument("--max-seam", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def episode(env, sigma, rng, corr):
    env_obs = env.observation()
    n, fell = 0, False
    a = np.zeros(env.n_act)
    while True:
        if n % corr == 0:
            a = rng.normal(0.0, sigma, env.n_act) if sigma > 0 else np.zeros(env.n_act)
        _, _, term, trunc, info = env.step(a)
        n += 1
        if term or trunc:
            fell = bool(term)
            break
    return n, fell


def main():
    args = parse_args()
    ref = ReferenceMotions(args.dataset, cyclic=not args.acyclic,
                           max_seam=args.max_seam)
    cfg = dict(DEFAULT_CFG)
    cfg["max_steps"] = args.max_steps
    cfg["action_scale"] = args.action_scale
    env = DeepMimicEnv(ref, cfg=cfg, seed=args.seed)

    pick = np.random.default_rng(args.seed)
    states = []
    for _ in range(args.states):
        c, k = ref.sample_start(pick)
        states.append((c, k))

    print(f"action sensitivity: {args.states} fixed RSI states x {args.seeds} "
          f"action seeds, cap {args.max_steps} steps")
    print(f"residual scale {args.action_scale} rad, noise held {args.corr} steps")
    print()
    header = f"  {'state':<14}" + "".join(f"{s:>17}" for s in
                                          [f"sigma={x}" for x in args.noise])
    print(header)

    rows = []
    for (c, k) in states:
        line = f"  {ref.clips[c]['name']}/{k:<5}"
        row = []
        for sigma in args.noise:
            lens, falls = [], []
            for s in range(args.seeds):
                env.reset(clip=c, frame=k)
                rng = np.random.default_rng(1000 * s + 7)
                n, f = episode(env, sigma, rng, args.corr)
                lens.append(n)
                falls.append(f)
            lens = np.array(lens)
            row.append((lens.mean(), lens.max(), np.mean(falls)))
            line += f"  {lens.mean():>5.0f}/{lens.max():>3.0f} f{np.mean(falls):>4.2f}"
        rows.append(row)
        print(line)

    print()
    print("aggregate (mean over states):")
    print(f"  {'sigma':>7} {'mean len':>9} {'best len':>9} {'fall rate':>10} "
          f"{'headroom':>9}")
    for i, sigma in enumerate(args.noise):
        m = np.mean([r[i][0] for r in rows])
        b = np.mean([r[i][1] for r in rows])
        f = np.mean([r[i][2] for r in rows])
        print(f"  {sigma:>7.2f} {m:>9.1f} {b:>9.1f} {f:>10.2f} {b - m:>9.1f}")

    base_mean = np.mean([r[0][0] for r in rows])
    best_any = np.mean([max(r[i][1] for i in range(len(args.noise)))
                        for r in rows])
    print()
    print(f"zero-action mean length      {base_mean:.1f}")
    print(f"best-of-any-noise mean length {best_any:.1f}")
    print(f"achievable improvement        {best_any - base_mean:+.1f} steps "
          f"({100 * (best_any / max(base_mean, 1e-9) - 1):+.0f}%)")


if __name__ == "__main__":
    main()
