import argparse
import json
import os
import time

import numpy as np
import torch

from deepmimic.vecenv import VecEnv
from deepmimic.ppo import PPO, DEFAULT_PPO
from deepmimic.env import DEFAULT_CFG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/qcheck/gait_v6.h5")
    p.add_argument("--model", default="assets/r1/scene.xml")
    p.add_argument("--out", default="runs/dm")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--steps-per-env", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="*", default=[256, 256])
    p.add_argument("--log-std-init", type=float, default=-2.5)
    p.add_argument("--action-scale", type=float, default=0.18)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--acyclic", action="store_true")
    p.add_argument("--absolute", action="store_true")
    p.add_argument("--no-rsi", action="store_true")
    p.add_argument("--min-speed", type=float, default=None)
    p.add_argument("--max-seam", type=float, default=0.08)
    p.add_argument("--speed-bias", type=float, default=4.0)
    p.add_argument("--obs-history", type=int, default=3)
    p.add_argument("--domain-rand", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    env_cfg = dict(DEFAULT_CFG)
    env_cfg.update({
        "action_scale": args.action_scale,
        "residual": not args.absolute,
        "rsi": not args.no_rsi,
        "max_steps": args.max_steps,
        "obs_history": args.obs_history,
        "domain_rand": args.domain_rand,
    })
    ref_kw = {"cyclic": not args.acyclic, "max_seam": args.max_seam,
              "speed_bias": args.speed_bias}
    if args.min_speed is not None:
        ref_kw["min_speed"] = args.min_speed

    vec = VecEnv(args.n_envs, args.dataset, model_path=args.model,
                 cfg=env_cfg, ref_kw=ref_kw, seed=args.seed * 1000,
                 n_workers=args.n_workers)

    ppo_cfg = dict(DEFAULT_PPO)
    ppo_cfg.update({
        "steps_per_env": args.steps_per_env,
        "lr": args.lr,
        "gamma": args.gamma,
        "clip": args.clip,
        "ent_coef": args.ent_coef,
        "epochs": args.epochs,
        "minibatches": args.minibatches,
        "hidden": tuple(args.hidden),
        "log_std_init": args.log_std_init,
    })
    algo = PPO(vec, cfg=ppo_cfg, seed=args.seed)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        algo.ac.load_state_dict(ck["model"])
        if "opt" in ck:
            algo.opt.load_state_dict(ck["opt"])
        if "lr" in ck:
            for g in algo.opt.param_groups:
                g["lr"] = ck["lr"]
        algo.total_steps = int(ck.get("total_steps", 0))
        print(f"resumed {args.resume} at {algo.total_steps:,} steps "
              f"lr {algo.opt.param_groups[0]['lr']:.2e}")

    cfg_dump = {"args": vars(args), "env": {k: list(v) if isinstance(v, tuple) else v
                                            for k, v in env_cfg.items()},
                "ppo": {k: list(v) if isinstance(v, tuple) else v
                        for k, v in ppo_cfg.items()},
                "obs_dim": vec.obs_dim, "act_dim": vec.act_dim}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=2)

    log_path = os.path.join(args.out, "train.jsonl")
    logf = open(log_path, "a")
    best = {"ret": -1e18}
    t0 = time.perf_counter()

    print(f"obs {vec.obs_dim}  act {vec.act_dim}  envs {args.n_envs}  "
          f"batch {args.n_envs * args.steps_per_env}  target {args.total_steps:,}")
    print(f"{'iter':>5} {'steps':>10} {'sps':>6} {'ret':>8} {'len':>6} "
          f"{'fall':>5} {'imit':>6} {'task':>6} {'reg':>6} {'slip':>6} "
          f"{'smth':>6} {'std':>5} {'kl':>7} {'ev':>6} {'lr':>9}")

    def on_log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"{rec['iter']:>5} {rec['steps']:>10,} {rec['sps']:>6.0f} "
              f"{rec['ret']:>8.2f} {rec['len']:>6.1f} {rec['fall']:>5.2f} "
              f"{rec['imitation']:>6.3f} {rec['task']:>6.3f} {rec['reg']:>6.3f} "
              f"{rec['slip']:>6.3f} {rec['smooth']:>6.3f} {rec['std']:>5.3f} "
              f"{rec['kl']:>7.4f} {rec['ev']:>6.3f} {rec['lr']:>9.2e}",
              flush=True)
        if np.isfinite(rec["ret"]) and rec["ret"] > best["ret"]:
            best["ret"] = rec["ret"]
            torch.save({"model": algo.ac.state_dict(), "cfg": cfg_dump,
                        "rec": rec, "opt": algo.opt.state_dict(),
                        "total_steps": algo.total_steps,
                        "lr": algo.opt.param_groups[0]["lr"]},
                       os.path.join(args.out, "best.pt"))
        if rec["iter"] % args.save_every == 0:
            torch.save({"model": algo.ac.state_dict(), "cfg": cfg_dump,
                        "rec": rec, "opt": algo.opt.state_dict(),
                        "total_steps": algo.total_steps,
                        "lr": algo.opt.param_groups[0]["lr"]},
                       os.path.join(args.out, "last.pt"))

    try:
        algo.learn(args.total_steps, log_every=args.log_every, on_log=on_log)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        torch.save({"model": algo.ac.state_dict(), "cfg": cfg_dump,
                    "opt": algo.opt.state_dict(),
                    "total_steps": algo.total_steps,
                    "lr": algo.opt.param_groups[0]["lr"]},
                   os.path.join(args.out, "last.pt"))
        logf.close()
        vec.close()

    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min   "
          f"best return {best['ret']:.2f}   saved to {args.out}")


if __name__ == "__main__":
    main()
