import numpy as np

from mpc.metrics import debounce


CYCLE_FIELDS = ("qpos", "qvel", "qref", "foot_pos", "foot_contact")


def strike_indices(contact, dt):
    sig = debounce(np.asarray(contact, dtype=float), dt)
    rise = np.flatnonzero((sig[1:] > 0.5) & (sig[:-1] < 0.5)) + 1
    return rise, sig


def seam_error(clip, a, b):
    q = clip["qpos"]
    dq = np.abs(q[b, 7:] - q[a, 7:])
    dv = np.abs(clip["qvel"][b, 6:] - clip["qvel"][a, 6:])
    dz = abs(float(q[b, 2] - q[a, 2]))
    return float(np.sqrt(np.mean(dq ** 2))), float(np.sqrt(np.mean(dv ** 2))), dz


def extract_cycle(clip, dt, gait_period, foot=0, lo=0.5, hi=2.0, verbose=False):
    settle = clip["settle"]
    rises, _ = strike_indices(clip["foot_contact"][:, foot], dt)
    rises = rises[rises >= settle]
    if len(rises) < 2:
        return None

    lo_n = int(lo * gait_period / dt)
    hi_n = int(hi * gait_period / dt)
    cands = []
    for a, b in zip(rises[:-1], rises[1:]):
        n = int(b - a)
        if n < lo_n or n > hi_n:
            continue
        eq, ev, ez = seam_error(clip, a, b)
        cands.append({"a": int(a), "b": int(b), "n": n,
                      "seam_q": eq, "seam_v": ev, "seam_z": ez})
    if not cands:
        return None

    med = np.median([c["n"] for c in cands])
    for c in cands:
        c["score"] = c["seam_q"] + 0.05 * c["seam_v"] + abs(c["n"] - med) / med
    best = min(cands, key=lambda c: c["score"])
    if verbose:
        print(f"    {len(cands)} candidate cycles, lengths "
              f"{[c['n'] for c in cands]}, chose {best['n']} "
              f"(seam_q {best['seam_q']:.4f} rad)")

    a, b = best["a"], best["b"]
    out = {k: np.array(clip[k][a:b], dtype=np.float32) for k in CYCLE_FIELDS}
    out["length"] = best["n"]
    out["duration"] = best["n"] * dt
    out["seam_q"] = best["seam_q"]
    out["seam_v"] = best["seam_v"]
    out["seam_z"] = best["seam_z"]
    out["n_candidates"] = len(cands)
    q = clip["qpos"]
    out["dx"] = float(q[b, 0] - q[a, 0])
    out["dy"] = float(q[b, 1] - q[a, 1])
    out["start"] = a
    return out


def extract_cycles(clip, dt, gait_period, foot=0, lo=0.5, hi=2.0,
                   max_seam=0.20, max_per_clip=None, verbose=False):
    settle = clip["settle"]
    rises, _ = strike_indices(clip["foot_contact"][:, foot], dt)
    rises = rises[rises >= settle]
    if len(rises) < 2:
        return []

    lo_n = int(lo * gait_period / dt)
    hi_n = int(hi * gait_period / dt)
    cands = []
    for a, b in zip(rises[:-1], rises[1:]):
        n = int(b - a)
        if n < lo_n or n > hi_n:
            continue
        eq, ev, ez = seam_error(clip, a, b)
        if eq > max_seam:
            continue
        cands.append({"a": int(a), "b": int(b), "n": n,
                      "seam_q": eq, "seam_v": ev, "seam_z": ez})
    if not cands:
        return []

    med = np.median([c["n"] for c in cands])
    for c in cands:
        c["score"] = c["seam_q"] + 0.05 * c["seam_v"] + abs(c["n"] - med) / med
    cands.sort(key=lambda c: c["score"])
    if max_per_clip:
        cands = cands[:max_per_clip]

    q = clip["qpos"]
    out = []
    for best in cands:
        a, b = best["a"], best["b"]
        cyc = {k: np.array(clip[k][a:b], dtype=np.float32) for k in CYCLE_FIELDS}
        cyc["length"] = best["n"]
        cyc["duration"] = best["n"] * dt
        cyc["seam_q"] = best["seam_q"]
        cyc["seam_v"] = best["seam_v"]
        cyc["seam_z"] = best["seam_z"]
        cyc["dx"] = float(q[b, 0] - q[a, 0])
        cyc["dy"] = float(q[b, 1] - q[a, 1])
        cyc["start"] = a
        out.append(cyc)
    if verbose:
        print(f"    {len(out)} cycles kept, lengths {[c['length'] for c in out]}, "
              f"seams {[round(c['seam_q'], 3) for c in out]}")
    return out
