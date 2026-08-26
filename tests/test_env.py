import numpy as np

from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv


def rollout(env, policy, n_ep=8, rng=None):
    rets, lens, reasons = [], [], {}
    term_terms = []
    for _ in range(n_ep):
        env.reset()
        total, n = 0.0, 0
        last = None
        while True:
            obs, r, term, trunc, info = env.step(policy(env, rng))
            total += r
            n += 1
            last = info
            if term or trunc:
                key = info["reason"] if info["reason"] else "truncated"
                reasons[key] = reasons.get(key, 0) + 1
                break
        rets.append(total)
        lens.append(n)
        term_terms.append(last["terms"])
    return np.array(rets), np.array(lens), reasons, term_terms


def main():
    ref = ReferenceMotions("data/qcheck/gait_v6.h5")
    env = DeepMimicEnv(ref, seed=0)
    print(f"obs dim {env.obs_dim}   act dim {env.n_act}   "
          f"ctrl_dt {env.ctrl_dt}   substeps {env.n_substeps}")
    print(f"clips {len(ref)}   frames/clip {ref.n_frames(0)}   "
          f"max episode {ref.n_frames(0) - max(env.cfg['obs_future']) - 1}")
    print()

    rng = np.random.default_rng(0)
    policies = [
        ("zero action (replay of MPC targets)", lambda e, g: np.zeros(e.n_act)),
        ("small noise (0.1)", lambda e, g: g.normal(0, 0.1, e.n_act)),
        ("full random", lambda e, g: g.uniform(-1, 1, e.n_act)),
    ]
    for label, pol in policies:
        R, L, why, _ = rollout(env, pol, n_ep=10, rng=rng)
        print(f"{label}")
        print(f"   return {R.mean():8.2f} +/- {R.std():5.2f}   "
              f"length {L.mean():6.1f} (max {L.max()})   "
              f"per-step {R.sum() / L.sum():.3f}")
        print(f"   endings {why}")
    print()

    print("reward breakdown, zero action, mid-clip:")
    env.reset(clip=0, frame=120)
    acc = {}
    for _ in range(50):
        _, _, _, _, info = env.step(np.zeros(env.n_act))
        for k, v in info["terms"].items():
            acc.setdefault(k, []).append(v)
    for k, v in acc.items():
        print(f"   {k:<11} {np.mean(v):.4f}")
    print()

    print("determinism check (same seed, same start):")
    a = DeepMimicEnv(ref, model=env.model, seed=3)
    b = DeepMimicEnv(ref, model=env.model, seed=3)
    o1, o2 = a.reset(clip=1, frame=100), b.reset(clip=1, frame=100)
    same_obs = np.allclose(o1, o2)
    ra = rb = 0.0
    for i in range(60):
        act = np.sin(np.arange(a.n_act) + i * 0.1) * 0.5
        _, r1, _, _, _ = a.step(act)
        _, r2, _, _, _ = b.step(act)
        ra += r1
        rb += r2
    print(f"   identical reset obs {same_obs}   returns {ra:.6f} vs {rb:.6f}   "
          f"match {abs(ra - rb) < 1e-9}")
    print()

    print("residual vs absolute action mode, zero action:")
    for mode in (True, False):
        e = DeepMimicEnv(ref, model=env.model, cfg={"residual": mode}, seed=0)
        R, L, why, _ = rollout(e, lambda x, g: np.zeros(x.n_act), n_ep=6, rng=rng)
        print(f"   residual={mode!s:<5} return {R.mean():8.2f}   "
              f"length {L.mean():6.1f}   {why}")
    print()

    print("RSI coverage over 300 resets:")
    ph, cl = [], []
    for _ in range(300):
        env.reset()
        ph.append(env.phase())
        cl.append(env._clip)
    ph = np.array(ph)
    hist = np.histogram(ph, bins=5, range=(0, 1))[0]
    print(f"   phase histogram (5 bins) {hist}   clips visited {len(set(cl))}/{len(ref)}")


if __name__ == "__main__":
    main()
