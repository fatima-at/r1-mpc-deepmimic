import argparse

import numpy as np
import mujoco

from mpc.cem import CEMPlanner
from mpc.env import load_model, reset
from mpc.actuators import name_mask
from mpc.tasks.locomotion import LocomotionTask


def build(model_path, horizon=10, n_samples=32, n_elite=4, n_iters=3, seed=0):
    model = load_model(model_path)
    data = mujoco.MjData(model)
    reset(model, data)
    task = LocomotionTask(model)
    planner = CEMPlanner(
        model,
        task,
        horizon=horizon,
        n_samples=n_samples,
        n_elite=n_elite,
        n_iters=n_iters,
        n_thread=4,
        seed=seed,
        free_mask=name_mask(model, ["hip", "knee", "ankle"]),
    )
    return model, data, task, planner


def reference_rollout(model, data, ctrl_seq, n_substeps):
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qvel[:] = data.qvel
    d.act[:] = data.act
    d.time = data.time
    mujoco.mj_forward(model, d)

    H = ctrl_seq.shape[0]
    qpos = np.zeros((H, model.nq))
    qvel = np.zeros((H, model.nv))
    for t in range(H):
        d.ctrl[:] = ctrl_seq[t]
        for _ in range(n_substeps):
            mujoco.mj_step(model, d)
        qpos[t] = d.qpos
        qvel[t] = d.qvel
    return qpos, qvel


def v1_rollout_fidelity(model, data, planner):
    rng = np.random.default_rng(1)
    nominal = planner.task.nominal_ctrl()
    noise = 0.05 * (planner.hi - planner.lo) * rng.standard_normal(
        (planner.horizon, planner.nu)
    )
    ctrl_seq = np.clip(nominal + noise, planner.lo, planner.hi)

    state0 = np.empty(planner.nstate)
    mujoco.mj_getState(model, data, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    qpos_b, qvel_b = planner._rollout_states(state0[None], ctrl_seq[None])
    qpos_r, qvel_r = reference_rollout(model, data, ctrl_seq, planner.n_substeps)

    dq = np.abs(qpos_b[0] - qpos_r).max()
    dv = np.abs(qvel_b[0] - qvel_r).max()
    ok = dq < 1e-9 and dv < 1e-9
    print(f"V1 rollout fidelity      max|dqpos| {dq:.3e}  max|dqvel| {dv:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def v2_cost_sanity(model, data, task, planner):
    rng = np.random.default_rng(2)
    nominal = planner.task.nominal_ctrl()
    state0 = np.empty(planner.nstate)
    mujoco.mj_getState(model, data, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    n_perturb = 16
    seqs = np.tile(nominal, (n_perturb + 1, planner.n_knots, 1))
    scale = 0.30 * (planner.hi - planner.lo)
    seqs[1:] += scale * rng.standard_normal((n_perturb, planner.n_knots, planner.nu))
    np.clip(seqs, planner.lo, planner.hi, out=seqs)

    init = np.tile(state0, (n_perturb + 1, 1))
    costs = planner._evaluate(init, seqs, np.zeros(3))

    n_better = int(np.sum(costs[1:] < costs[0]))
    ok = n_better <= n_perturb // 4
    v2_cost_sanity.scale = float(np.median(costs[1:]))
    print(f"V2 cost sanity           nominal {costs[0]:10.3f}   "
          f"perturbed median {np.median(costs[1:]):10.3f}   "
          f"{n_better}/{n_perturb} beat nominal  {'PASS' if ok else 'FAIL'}")
    return ok


def v3_improvement(model, data, planner, scale):
    rng = np.random.default_rng(3)

    planner.reset()
    planner.mean += 0.20 * (planner.hi - planner.lo) * rng.standard_normal(
        (planner.n_knots, planner.nu)
    )
    np.clip(planner.mean, planner.lo, planner.hi, out=planner.mean)
    planner.plan(data, np.zeros(3))
    c_perturbed = planner.iter_costs
    ok_a = c_perturbed[-1] < c_perturbed[0]
    print(f"V3a improves from perturbed start  {[round(x, 2) for x in c_perturbed]}  "
          f"{'PASS' if ok_a else 'FAIL'}")

    planner.reset()
    planner.plan(data, np.zeros(3))
    c_nominal = planner.iter_costs
    tol = max(1.5 * c_nominal[0], 0.05 * scale)
    ok_b = c_nominal[-1] <= tol
    print(f"V3b stable at optimum              {[round(x, 2) for x in c_nominal]}  "
          f"tol {tol:.2f}  {'PASS' if ok_b else 'FAIL'}")

    return ok_a and ok_b


def v8_spline(planner):
    rng = np.random.default_rng(8)
    knots = planner.task.nominal_ctrl() + 0.1 * rng.standard_normal(
        (1, planner.n_knots, planner.nu)
    )
    seq = planner.expand(knots)
    at_knots = seq[0][planner.knot_ticks]
    err = np.abs(at_knots - knots[0]).max()
    inside = seq.max() <= planner.hi.max() + 1e-9
    ok = err < 1e-9 and inside and seq.shape[1] == planner.horizon
    print(f"V8 spline expansion      knots {planner.n_knots} -> {seq.shape[1]} ticks  "
          f"max|err at knots| {err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def v7_determinism(model_path):
    results = []
    for _ in range(2):
        m, d, t, p = build(model_path, seed=7)
        a = p.plan(d, np.zeros(3))
        p.close()
        results.append(a)
    dmax = np.abs(results[0] - results[1]).max()
    ok = dmax == 0.0
    print(f"V7 determinism           max|da| {dmax:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model, data, task, planner = build(args.model)
    print(f"model nq {model.nq}  nv {model.nv}  nu {model.nu}  "
          f"substeps {planner.n_substeps}")
    print(f"position mode: nominal action range "
          f"[{planner.task.nominal_ctrl().min():.3f}, "
          f"{planner.task.nominal_ctrl().max():.3f}]")
    print()

    results = [
        v1_rollout_fidelity(model, data, planner),
        v2_cost_sanity(model, data, task, planner),
        v3_improvement(model, data, planner, v2_cost_sanity.scale),
        v8_spline(planner),
    ]
    planner.close()
    results.append(v7_determinism(args.model))

    print()
    print("ALL PASS" if all(results) else "FAILURES PRESENT")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
