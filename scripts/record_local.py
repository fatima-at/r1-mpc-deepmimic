import argparse
import os
import subprocess
import sys
import time

import h5py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--commands", type=float, nargs="*",
                   default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    p.add_argument("--duration", type=float, default=14.0)
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--elite", type=int, default=20)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--n-knots", type=int, default=10)
    p.add_argument("--n-thread", type=int, default=14)
    p.add_argument("--parts", default="data/parts")
    p.add_argument("--out", default="data/qcheck/gait_v6.h5")
    p.add_argument("--model", default="assets/r1/scene.xml")
    p.add_argument("--python", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def part_ok(path):
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return "episodes" in f and len(f["episodes"]) > 0
    except OSError:
        return False


def main():
    args = parse_args()
    py = args.python or sys.executable
    os.makedirs(args.parts, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    jobs = []
    i = 0
    for s in args.seeds:
        for c in args.commands:
            jobs.append((i, c, s))
            i += 1

    per_plan = 0.0298 * args.samples * args.iters / 2.0
    est = len(jobs) * args.duration / 0.02 * per_plan / 3600.0
    print(f"{len(jobs)} episodes x {args.duration:.0f}s   "
          f"samples {args.samples} iters {args.iters} threads {args.n_thread}")
    print(f"estimated total {est:.1f} h   "
          f"(~{est * 60 / max(len(jobs), 1):.0f} min per episode)")
    usable = args.duration - args.settle
    print(f"expected cycles per episode ~{max(0, int(usable / 1.48) - 1)}   "
          f"total ~{len(jobs) * max(0, int(usable / 1.48) - 1)}")
    print()

    done = sum(1 for (i, c, s) in jobs
               if part_ok(os.path.join(args.parts, f"ep_{i:04d}.h5")))
    print(f"already complete: {done}/{len(jobs)}")
    if args.dry_run:
        return

    t0 = time.perf_counter()
    for n, (idx, cmd, seed) in enumerate(jobs):
        path = os.path.join(args.parts, f"ep_{idx:04d}.h5")
        if part_ok(path):
            print(f"[{n + 1}/{len(jobs)}] ep_{idx:04d} cmd {cmd:+.2f} "
                  f"seed {seed}  SKIP (exists)", flush=True)
            continue
        print(f"[{n + 1}/{len(jobs)}] ep_{idx:04d} cmd {cmd:+.2f} seed {seed} "
              f"... ", end="", flush=True)
        t1 = time.perf_counter()
        cp = subprocess.run([
            py, "-u", "scripts/record_dataset.py",
            "--model", args.model, "--out", path,
            "--commands", str(cmd), "--start-index", str(idx),
            "--seed", str(seed), "--duration", str(args.duration),
            "--settle", str(args.settle), "--horizon", str(args.horizon),
            "--n-knots", str(args.n_knots), "--iters", str(args.iters),
            "--samples", str(args.samples), "--elite", str(args.elite),
            "--vel-mode", "instant", "--n-thread", str(args.n_thread),
        ], capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": os.getcwd()})
        el = (time.perf_counter() - t1) / 60.0
        tail = [x for x in cp.stdout.strip().splitlines() if x.strip()]
        status = tail[-2] if len(tail) >= 2 else (tail[-1] if tail else "?")
        if cp.returncode != 0:
            print(f"FAILED ({el:.1f} min)")
            print(cp.stderr.strip()[-800:])
        else:
            print(f"{el:.1f} min   {status}")
        remaining = len(jobs) - (n + 1)
        avg = (time.perf_counter() - t0) / 60.0 / (n + 1)
        print(f"      elapsed {(time.perf_counter() - t0) / 60:.0f} min, "
              f"~{remaining * avg / 60:.1f} h left", flush=True)

    parts = [os.path.join(args.parts, f)
             for f in sorted(os.listdir(args.parts)) if f.endswith(".h5")]
    parts = [p for p in parts if part_ok(p)]
    print(f"\nmerging {len(parts)} parts -> {args.out}")
    subprocess.run([py, "scripts/merge_dataset.py",
                    os.path.join(args.parts, "*.h5"), "--out", args.out],
                   env={**os.environ, "PYTHONPATH": os.getcwd()})


if __name__ == "__main__":
    main()
