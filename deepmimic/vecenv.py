import multiprocessing as mp

import numpy as np

from mpc.env import load_model
from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv


TERM_KEYS = ("pose", "vel", "endeff", "root", "track", "along", "cross", "heading",
             "slip", "smooth", "accel", "torque", "imitation", "task", "reg")


def _worker(remote, parent, ref_path, ref_kw, model_path, cfg, seed, n_local):
    parent.close()
    ref = ReferenceMotions(ref_path, **ref_kw)
    shared = load_model(model_path)
    envs = [DeepMimicEnv(ref, model=shared, cfg=cfg, seed=seed + i)
            for i in range(n_local)]
    n_obs, n_act = envs[0].obs_dim, envs[0].n_act
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "step":
                obs = np.zeros((n_local, n_obs), dtype=np.float32)
                final = np.zeros((n_local, n_obs), dtype=np.float32)
                rew = np.zeros(n_local, dtype=np.float32)
                term = np.zeros(n_local, dtype=bool)
                trunc = np.zeros(n_local, dtype=bool)
                terms = np.zeros((n_local, len(TERM_KEYS)), dtype=np.float32)
                lens = np.zeros(n_local, dtype=np.int64)
                for i, e in enumerate(envs):
                    o, r, td, tc, info = e.step(payload[i])
                    rew[i], term[i], trunc[i] = r, td, tc
                    lens[i] = info["steps"]
                    for j, k in enumerate(TERM_KEYS):
                        terms[i, j] = info["terms"][k]
                    if td or tc:
                        final[i] = o
                        o = e.reset()
                    obs[i] = o
                remote.send((obs, rew, term, trunc, final, terms, lens))
            elif cmd == "reset":
                remote.send(np.array([e.reset() for e in envs], dtype=np.float32))
            elif cmd == "spec":
                remote.send((n_obs, n_act))
            elif cmd == "close":
                remote.close()
                break
    except (KeyboardInterrupt, EOFError):
        pass


class VecEnv:
    def __init__(self, n_envs, ref_path, model_path="assets/r1/scene.xml",
                 cfg=None, ref_kw=None, seed=0, n_workers=None):
        ctx = mp.get_context("spawn")
        if n_workers is None:
            n_workers = min(n_envs, mp.cpu_count())
        n_workers = max(1, min(n_workers, n_envs))
        base, extra = divmod(n_envs, n_workers)
        self.counts = [base + (1 if i < extra else 0) for i in range(n_workers)]
        self.n_envs = n_envs
        self.n_workers = n_workers
        self.slices = []
        o = 0
        for c in self.counts:
            self.slices.append(slice(o, o + c))
            o += c

        self.remotes, self.procs = [], []
        ref_kw = ref_kw or {}
        s = seed
        for i, c in enumerate(self.counts):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker,
                            args=(b, a, ref_path, ref_kw, model_path, cfg, s, c),
                            daemon=True)
            p.start()
            b.close()
            self.remotes.append(a)
            self.procs.append(p)
            s += c
        self.remotes[0].send(("spec", None))
        self.obs_dim, self.act_dim = self.remotes[0].recv()
        self.closed = False

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes], axis=0)

    def step(self, actions):
        for r, sl in zip(self.remotes, self.slices):
            r.send(("step", actions[sl]))
        out = [r.recv() for r in self.remotes]
        obs = np.concatenate([o[0] for o in out], axis=0)
        rew = np.concatenate([o[1] for o in out], axis=0)
        term = np.concatenate([o[2] for o in out], axis=0)
        trunc = np.concatenate([o[3] for o in out], axis=0)
        final = np.concatenate([o[4] for o in out], axis=0)
        terms = np.concatenate([o[5] for o in out], axis=0)
        lens = np.concatenate([o[6] for o in out], axis=0)
        return obs, rew, term, trunc, final, terms, lens

    def close(self):
        if getattr(self, "closed", True):
            return
        self.closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self):
        self.close()
