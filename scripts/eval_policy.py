import argparse
import json

import numpy as np
import torch

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv, DEFAULT_CFG
from deepmimic.policy import ActorCritic
from deepmimic.vecenv import TERM_KEYS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", default="data/qcheck/gait_v6.h5")
    p.add_argument("--model", default="assets/r1/scene.xml")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--from-start", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--acyclic", action="store_true")
    p.add_argument("--max-seam", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def build(args, cfg_over=None):
    ref = ReferenceMotions(args.dataset, cyclic=not args.acyclic,
                           max_seam=args.max_seam)
    cfg = dict(DEFAULT_CFG)
    cfg["max_steps"] = args.max_steps
    if cfg_over:
        cfg.update(cfg_over)
    return ref, DeepMimicEnv(ref, model_path=args.model, cfg=cfg, seed=args.seed)


def run(env, ref, policy, n_ep, from_start, rng, deterministic=True):
    rets, lens, falls = [], [], []
    acc = {k: [] for k in TERM_KEYS}
    per_clip = {}
    for e in range(n_ep):
        if from_start:
            c = e % len(ref)
            env.reset(clip=c, frame=ref.clips[c]["settle"])
        else:
            env.reset(rng=rng)
        c = env._clip
        obs = env.observation()
        tot, n = 0.0, 0
        fell = False
        loc = {k: [] for k in TERM_KEYS}
        while True:
            a = policy(obs, deterministic)
            obs, r, term, trunc, info = env.step(a)
            tot += r
            n += 1
            for k in TERM_KEYS:
                loc[k].append(info["terms"][k])
            if term or trunc:
                fell = bool(term)
                break
        rets.append(tot)
        lens.append(n)
        falls.append(1.0 if fell else 0.0)
        for k in TERM_KEYS:
            acc[k].append(float(np.mean(loc[k])))
        per_clip.setdefault(c, []).append((n, fell))
    return (np.array(rets), np.array(lens), np.array(falls),
            {k: float(np.mean(v)) for k, v in acc.items()}, per_clip)


def main():
    args = parse_args()
    ref, env = build(args)
    rng = np.random.default_rng(args.seed)

    policies = [("zero action", lambda o, d: np.zeros(env.n_act))]
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ac = ActorCritic(env.obs_dim, env.n_act,
                         hidden=tuple(ck["cfg"]["ppo"]["hidden"]),
                         log_std_init=ck["cfg"]["ppo"]["log_std_init"])
        ac.load_state_dict(ck["model"])
        ac.eval()

        def pol(o, d):
            a, _, _ = ac.act(o[None, :], deterministic=d)
            return a[0]

        tag = "policy (det)" if not args.stochastic else "policy (stoch)"
        policies.append((tag, pol))
        if "rec" in ck:
            print(f"checkpoint from iter {ck['rec']['iter']} "
                  f"steps {ck['rec']['steps']:,} train-return {ck['rec']['ret']:.2f}")

    mode = "clip start" if args.from_start else "RSI"
    print(f"{args.episodes} episodes, {mode}, max {args.max_steps} steps, "
          f"{'stochastic' if args.stochastic else 'deterministic'}")
    print()
    print(f"{'policy':<16} {'return':>9} {'length':>8} {'fall':>6} "
          f"{'imit':>6} {'task':>6} {'pose':>6} {'endeff':>6} {'root':>6}")

    results = {}
    for name, pol in policies:
        rng = np.random.default_rng(args.seed)
        R, L, F, T, pc = run(env, ref, pol, args.episodes, args.from_start, rng,
                             deterministic=not args.stochastic)
        results[name] = (R, L, F, T, pc)
        print(f"{name:<16} {R.mean():>9.2f} {L.mean():>8.1f} {F.mean():>6.2f} "
              f"{T['imitation']:>6.3f} {T['task']:>6.3f} {T['pose']:>6.3f} "
              f"{T['endeff']:>6.3f} {T['root']:>6.3f}")

    print()
    print("survival by clip (mean length / fall rate):")
    names = list(results.keys())
    hdr = "  clip     label  " + "  ".join(f"{n:>18}" for n in names)
    print(hdr)
    for c in range(len(ref)):
        row = f"  {ref.clips[c]['name']:<8} {ref.clips[c]['command'][0]:>+5.2f}  "
        for n in names:
            pc = results[n][4].get(c, [])
            if pc:
                ln = np.mean([x[0] for x in pc])
                fr = np.mean([1.0 if x[1] else 0.0 for x in pc])
                row += f"  {ln:>10.1f} / {fr:>4.2f}"
            else:
                row += f"  {'-':>17}"
        print(row)


if __name__ == "__main__":
    main()
