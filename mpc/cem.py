import numpy as np
import mujoco
from mujoco import rollout as mj_rollout

from mpc.actuators import is_position_mode, actuator_joint_range, frozen_mask


def quat_to_mat(quat):
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n < 1e-12, 1.0, n)
    w, x, y, z = w / n, x / n, y / n, z / n
    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mat[..., 0, 1] = 2.0 * (x * y - w * z)
    mat[..., 0, 2] = 2.0 * (x * z + w * y)
    mat[..., 1, 0] = 2.0 * (x * y + w * z)
    mat[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mat[..., 1, 2] = 2.0 * (y * z - w * x)
    mat[..., 2, 0] = 2.0 * (x * z - w * y)
    mat[..., 2, 1] = 2.0 * (y * z + w * x)
    mat[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mat


def base_yaw(quat):
    m = quat_to_mat(np.asarray(quat, dtype=np.float64)[None])[0]
    return float(np.arctan2(m[1, 0], m[0, 0]))


def control_bounds(model):
    if is_position_mode(model):
        lo, hi = actuator_joint_range(model)
        bad = ~np.isfinite(lo) | ~np.isfinite(hi)
        lo = np.where(bad, -np.pi, lo)
        hi = np.where(bad, np.pi, hi)
        frozen = frozen_mask(model)
        lo = np.where(frozen, 0.0, lo)
        hi = np.where(frozen, 0.0, hi)
        return lo, hi

    lo = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64).copy()
    hi = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64).copy()
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    if not limited.all():
        flo = np.asarray(model.actuator_forcerange[:, 0], dtype=np.float64)
        fhi = np.asarray(model.actuator_forcerange[:, 1], dtype=np.float64)
        flimited = np.asarray(model.actuator_forcelimited, dtype=bool)
        fallback_lo = np.where(flimited, flo, -1.0)
        fallback_hi = np.where(flimited, fhi, 1.0)
        lo = np.where(limited, lo, fallback_lo)
        hi = np.where(limited, hi, fallback_hi)
    return lo, hi


class CEMPlanner:
    def __init__(
        self,
        model,
        task,
        horizon=25,
        n_samples=100,
        n_elite=10,
        n_iters=2,
        ctrl_dt=0.02,
        init_std_frac=0.25,
        min_std_frac=0.02,
        init_std_rad=0.10,
        min_std_rad=0.02,
        momentum=0.1,
        temperature=0.5,
        n_thread=8,
        seed=0,
        free_mask=None,
        n_knots=5,
    ):
        self.model = model
        self.task = task
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.n_elite = int(min(n_elite, n_samples))
        self.n_iters = int(n_iters)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)

        self.n_substeps = max(1, int(round(ctrl_dt / model.opt.timestep)))
        self.ctrl_dt = self.n_substeps * model.opt.timestep
        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv
        self.nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        self.lo, self.hi = control_bounds(model)
        if free_mask is not None:
            nom = np.asarray(task.nominal_ctrl(), dtype=np.float64)
            free = np.asarray(free_mask, dtype=bool)
            self.lo = np.where(free, self.lo, nom)
            self.hi = np.where(free, self.hi, nom)
        span = self.hi - self.lo
        if is_position_mode(model):
            self.init_std = np.minimum(init_std_rad, 0.25 * span)
            self.min_std = np.minimum(min_std_rad, 0.05 * span)
        else:
            self.init_std = init_std_frac * span
            self.min_std = min_std_frac * span

        self.n_knots = int(min(max(2, n_knots), self.horizon))
        if self.n_knots >= self.horizon:
            self.n_knots = self.horizon
        kt = np.unique(
            np.round(np.linspace(0, self.horizon - 1, self.n_knots)).astype(np.int64)
        )
        self.knot_ticks = kt
        self.n_knots = len(kt)
        ticks = np.arange(self.horizon)
        k0 = np.clip(np.searchsorted(kt, ticks, side="right") - 1, 0, self.n_knots - 2)
        k1 = k0 + 1
        self._k0 = k0
        self._k1 = k1
        self._kf = ((ticks - kt[k0]) / (kt[k1] - kt[k0]).astype(np.float64))[None, :, None]

        self.n_thread = max(1, int(n_thread))
        self._datas = [mujoco.MjData(model) for _ in range(self.n_thread)]
        self._roll = mj_rollout.Rollout(nthread=self.n_thread)

        self.last_cost = np.inf
        self.reset()

    def reset(self):
        nominal = self.task.nominal_ctrl()
        self.mean = np.tile(nominal, (self.n_knots, 1))
        self.std = np.tile(self.init_std, (self.n_knots, 1))
        self.last_action = nominal.copy()
        self.heading_ref = None
        self.pos_ref = None
        self._t_now = 0.0
        self.last_cost = np.inf

    def expand(self, knots):
        return knots[:, self._k0, :] * (1.0 - self._kf) + knots[:, self._k1, :] * self._kf

    def close(self):
        self._roll.close()

    def _rollout_states(self, initial_state, ctrl_seq):
        ctrl = np.repeat(ctrl_seq, self.n_substeps, axis=1)
        state, sensordata = self._roll.rollout(
            self.model, self._datas, initial_state, control=ctrl
        )
        sub = slice(self.n_substeps - 1, None, self.n_substeps)
        tick = state[:, sub, :]
        qpos = tick[:, :, 1 : 1 + self.nq]
        qvel = tick[:, :, 1 + self.nq : 1 + self.nq + self.nv]
        self._last_time = tick[:, :, 0]
        self._last_sensor = sensordata[:, sub, :] if sensordata is not None else None
        return qpos, qvel

    def _evaluate(self, initial_state, knots, command):
        seq = self.expand(knots)
        qpos, qvel = self._rollout_states(initial_state, seq)
        cost = self.task.cost(
            qpos,
            qvel,
            seq,
            command,
            last_ctrl=self.last_action,
            sensordata=self._last_sensor,
            times=self._last_time,
            heading_ref=self.heading_ref,
            t_now=self._t_now,
            pos_ref=self.pos_ref,
        )
        return np.where(np.isfinite(cost), cost, 1e12)

    def plan(self, data, command):
        state0 = np.empty(self.nstate, dtype=np.float64)
        mujoco.mj_getState(
            self.model, data, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        initial_state = np.tile(state0, (self.n_samples, 1))

        self._t_now = float(data.time)
        if self.heading_ref is None:
            self.heading_ref = base_yaw(data.qpos[3:7])
        if self.pos_ref is None:
            self.pos_ref = np.array(data.qpos[0:2], dtype=np.float64)

        costs = None
        self.iter_costs = []
        for _ in range(self.n_iters):
            noise = self.rng.standard_normal((self.n_samples, self.n_knots, self.nu))
            samples = self.mean[None] + self.std[None] * noise
            samples[0] = self.mean
            np.clip(samples, self.lo, self.hi, out=samples)

            costs = self._evaluate(initial_state, samples, command)
            self.iter_costs.append(float(costs.min()))
            elite_idx = np.argpartition(costs, self.n_elite - 1)[: self.n_elite]
            elites = samples[elite_idx]
            elite_costs = costs[elite_idx]

            adv = elite_costs - elite_costs.min()
            scale = adv.mean() + 1e-9
            weights = np.exp(-adv / (self.temperature * scale))
            weights /= weights.sum()

            new_mean = np.einsum("i,ijk->jk", weights, elites)
            new_var = np.einsum("i,ijk->jk", weights, (elites - new_mean) ** 2)

            self.mean = (1.0 - self.momentum) * new_mean + self.momentum * self.mean
            self.std = np.maximum(np.sqrt(new_var), self.min_std)

        self.last_cost = float(costs.min())
        action = np.clip(self.mean[0].copy(), self.lo, self.hi)
        self.last_action = action.copy()
        ch, sh = np.cos(self.heading_ref), np.sin(self.heading_ref)
        self.pos_ref = self.pos_ref + self.ctrl_dt * np.array([
            command[0] * ch - command[1] * sh,
            command[0] * sh + command[1] * ch,
        ])
        self.heading_ref += float(command[2]) * self.ctrl_dt
        self._shift()
        return action

    def _shift(self):
        full = self.expand(self.mean[None])[0]
        full[:-1] = full[1:]
        self.mean = full[self.knot_ticks].copy()
        self.std = np.tile(self.init_std, (self.n_knots, 1))
