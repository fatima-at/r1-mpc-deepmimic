import numpy as np
import h5py

from deepmimic.cycles import extract_cycle, extract_cycles


class ReferenceMotions:
    def __init__(self, path, gait_period=None, min_speed=None, max_speed=None,
                 use_relabelled=True, cyclic=False, verbose=False,
                 max_seam=0.20, max_per_clip=None,
                 cycle_lo=0.75, cycle_hi=1.25, speed_bias=0.0):
        self.path = str(path)
        clips = []
        with h5py.File(self.path, "r") as f:
            meta = dict(f["meta"].attrs)
            for name in sorted(f["episodes"].keys()):
                g = f["episodes"][name]
                a = g.attrs
                if bool(a.get("fell", False)):
                    continue
                cmd = np.array(a["command"], dtype=np.float64)
                ach = np.array(a["achieved_lin"], dtype=np.float64)
                yaw = float(a.get("achieved_yaw", 0.0))
                if not np.isfinite(yaw):
                    yaw = 0.0
                hindsight = np.array([ach[0], ach[1], yaw], dtype=np.float64)
                label = hindsight if use_relabelled else cmd
                v = float(ach[0])
                if min_speed is not None and v < min_speed:
                    continue
                if max_speed is not None and v > max_speed:
                    continue
                clips.append({
                    "name": name,
                    "qpos": np.array(g["qpos"], dtype=np.float32),
                    "qvel": np.array(g["qvel"], dtype=np.float32),
                    "qref": np.array(g["qref"], dtype=np.float32),
                    "foot_pos": np.array(g["foot_pos"], dtype=np.float32),
                    "foot_contact": np.array(g["foot_contact"], dtype=np.float32),
                    "time": np.array(g["time"], dtype=np.float64),
                    "command": label.astype(np.float32),
                    "commanded": cmd.astype(np.float32),
                    "achieved": ach.astype(np.float32),
                    "settle": int(a.get("settle_index", 0)),
                    "dt": float(a.get("ctrl_dt", 0.02)),
                })
        if not clips:
            raise ValueError(f"no usable clips in {self.path}")
        self.meta = meta
        self.dt = clips[0]["dt"]
        self.gait_period = float(gait_period) if gait_period else 1.538
        self.cyclic = bool(cyclic)

        if self.cyclic:
            kept = []
            for c in clips:
                cycs = extract_cycles(c, self.dt, self.gait_period,
                                      lo=cycle_lo, hi=cycle_hi,
                                      max_seam=max_seam,
                                      max_per_clip=max_per_clip,
                                      verbose=verbose)
                for j, cyc in enumerate(cycs):
                    d = dict(c)
                    for k in ("qpos", "qvel", "qref", "foot_pos", "foot_contact"):
                        d[k] = cyc[k]
                    d["settle"] = 0
                    d["cycle"] = cyc
                    d["name"] = f"{c['name']}c{j}"
                    kept.append(d)
            if not kept:
                raise ValueError(f"no extractable gait cycles in {self.path}")
            clips = kept

        self.clips = clips
        self.speeds = np.array([c["achieved"][0] for c in clips])
        self._w = None
        if speed_bias:
            self.set_speed_bias(speed_bias)

    def __len__(self):
        return len(self.clips)

    def n_frames(self, i):
        return self.clips[i]["qpos"].shape[0]

    def index(self, i, k):
        n = self.n_frames(i)
        if self.cyclic:
            return int(k) % n
        return int(np.clip(k, 0, n - 1))

    def phase(self, i, k):
        if self.cyclic:
            return float((int(k) % self.n_frames(i)) / self.n_frames(i))
        return float((self.clips[i]["time"][k] / self.gait_period) % 1.0)

    def clip_for_speed(self, v, rng=None):
        d = np.abs(self.speeds - v)
        near = np.flatnonzero(d <= d.min() + 1e-9)
        if rng is not None and len(near) > 1:
            return int(rng.choice(near))
        return int(near[0])

    def set_speed_bias(self, alpha):
        v = np.abs(self.speeds)
        w = 1.0 + float(alpha) * v / max(v.max(), 1e-9)
        self._w = w / w.sum()

    def sample_clip(self, rng):
        if getattr(self, "_w", None) is None:
            return int(rng.integers(len(self.clips)))
        return int(rng.choice(len(self.clips), p=self._w))

    def sample_start(self, rng, settle=True, margin=1):
        i = self.sample_clip(rng)
        if self.cyclic:
            return i, int(rng.integers(self.n_frames(i)))
        lo = self.clips[i]["settle"] if settle else 0
        hi = self.n_frames(i) - margin - 1
        if hi <= lo:
            lo, hi = 0, max(1, self.n_frames(i) - 2)
        k = int(rng.integers(lo, hi))
        return i, k

    def frame(self, i, k):
        c = self.clips[i]
        k = self.index(i, k)
        return {
            "qpos": c["qpos"][k],
            "qvel": c["qvel"][k],
            "qref": c["qref"][k],
            "foot_pos": c["foot_pos"][k],
            "foot_contact": c["foot_contact"][k],
            "phase": self.phase(i, k),
            "command": c["command"],
        }

    def summary(self):
        lines = [f"{len(self.clips)} clips from {self.path}"
                 f"{'  (cyclic)' if self.cyclic else ''}",
                 f"  dt {self.dt:.3f}s   gait period {self.gait_period:.3f}s",
                 f"  speeds {self.speeds.min():+.3f} .. {self.speeds.max():+.3f} m/s",
                 f"  frames per clip {min(self.n_frames(i) for i in range(len(self)))}"
                 f" .. {max(self.n_frames(i) for i in range(len(self)))}",
                 f"  total frames {sum(self.n_frames(i) for i in range(len(self)))}"]
        return "\n".join(lines)
