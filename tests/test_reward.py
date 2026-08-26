import numpy as np

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv


def main():
    ref = ReferenceMotions("data/qcheck/gait_v6.h5")
    env = DeepMimicEnv(ref, seed=0)
    print(f"velocity filter alpha {env.vel_alpha:.4f}  "
          f"time constant {1 / env.vel_alpha:.0f} steps "
          f"({ref.gait_period:.3f} s)")
    print()

    print("task reward vs time, zero action, start of clip (settle frame):")
    print(f"  {'step':>5} {'vx_inst':>9} {'vx_ema':>9} {'label':>9} {'r_task':>8} "
          f"{'r_imit':>8}")
    env.reset(clip=2, frame=ref.clips[2]["settle"])
    label = env.command()
    for i in range(220):
        _, _, term, trunc, info = env.step(np.zeros(env.n_act))
        if i % 20 == 0 or i == 219:
            inst = env._body_vel()
            print(f"  {i:>5} {inst[0]:>9.3f} {env._vel_ema[0]:>9.3f} "
                  f"{label[0]:>9.3f} {info['terms']['task']:>8.4f} "
                  f"{info['terms']['imitation']:>8.4f}")
        if term or trunc:
            print(f"  ended at step {i}: {info['reason'] or 'truncated'}")
            break
    print()

    print("per-clip asymptotic task reward (zero action, last 40 steps):")
    print(f"  {'clip':>8} {'label':>8} {'ema':>8} {'r_task':>8} {'r_imit':>8} "
          f"{'steps':>6}")
    for c in range(len(ref)):
        e = DeepMimicEnv(ref, model=env.model, cfg={"max_steps": 400}, seed=0)
        e.reset(clip=c, frame=ref.clips[c]["settle"])
        tasks, imits = [], []
        n = 0
        while True:
            _, _, term, trunc, info = e.step(np.zeros(e.n_act))
            tasks.append(info["terms"]["task"])
            imits.append(info["terms"]["imitation"])
            n += 1
            if term or trunc:
                break
        k = min(40, len(tasks))
        print(f"  {ref.clips[c]['name']:>8} {e.command()[0]:>8.3f} "
              f"{e._vel_ema[0]:>8.3f} {np.mean(tasks[-k:]):>8.4f} "
              f"{np.mean(imits[-k:]):>8.4f} {n:>6}")


if __name__ == "__main__":
    main()
