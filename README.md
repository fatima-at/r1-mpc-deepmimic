# R1 MPC → DeepMimic

Walking controllers for the Unitree R1 in MuJoCo.

The pipeline has three stages. A sampling-based MPC produces walking
trajectories, those trajectories become reference motions, and a reinforcement
learning policy is trained to track them. The trained policy is intended as the
expert for a later distillation step into a spiking network.


## Install

```bash
python -m venv venv
venv/bin/python -m pip install -e .
```

On Windows use `venv/Scripts/python.exe` instead of `venv/bin/python`.

For the RL stage you also need PyTorch:

```bash
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For the MJX (GPU) stage:

```bash
venv/bin/python -m pip install jax jaxlib mujoco-mjx flax
```

## Stage 1 — MPC

A cross-entropy method planner runs in a receding horizon over a spline
parameterisation of the joint targets. The cost combines velocity tracking,
posture, and an explicit gait schedule (swing clearance, stance, double
support). Gait period and PD gains are derived from the robot's inertia rather
than tuned by hand; see `mpc/gait.py`.

Run it live:

```bash
python scripts/run_mpc.py --model assets/r1/scene.xml --command 0.6
```

Check the speed envelope:

```bash
python scripts/envelope.py --model assets/r1/scene.xml --gait
```

## Stage 2 — recording

Records MPC rollouts at a range of commanded speeds. Each episode stores
`qpos`, `qvel`, the position targets, joint torques, and foot state.


```bash
python scripts/record_local.py --commands 0.2 0.4 0.6 0.8 1.0 --duration 14
```

Episodes should be long. Whole gait cycles are extracted from each recording,
and the number of usable cycles is roughly `(duration - settle) / period - 1`,
so a 14 s episode yields about seven cycles while a 6 s episode yields one.

Check gait quality against target bands:

```bash
python scripts/gait_quality.py data/qcheck/gait_v6.h5
```

## Stage 3 — DeepMimic

Single gait cycles are extracted from the recordings and looped. The policy
outputs residual joint targets on top of the recorded MPC targets, and is
rewarded for tracking the reference pose, velocity, foot placement and root
state, for following the commanded velocity and path, and for smooth,
slip-free, low-torque motion. Training uses PPO.

```bash
python scripts/train_deepmimic.py --dataset data/qcheck/gait_v6.h5 --out runs/exp
```

Training is resumable:

```bash
python scripts/train_deepmimic.py --resume runs/exp/last.pt --out runs/exp
```

Evaluate and render:

```bash
python scripts/eval_policy.py --checkpoint runs/exp/best.pt --dataset data/qcheck/gait_v6.h5
python scripts/policy_gait.py --checkpoint runs/exp/best.pt --dataset data/qcheck/gait_v6.h5
python scripts/render_compare.py --checkpoint runs/exp/best.pt --out out.mp4
```

`render_compare.py` puts the reference and the policy side by side, which is
usually the fastest way to see what a policy is actually doing.

An MJX version of the environment is in `deepmimic/mjx_env.py` for GPU
training. It matches the CPU environment step for step; see
`tests/test_mjx_parity.py`.

## Layout

```
mpc/          CEM planner, cost terms, actuator and sensor setup
deepmimic/    reference motions, cycle extraction, RL environment, PPO
scripts/      entry points for running, recording, training, evaluating
tests/        environment and planner checks
assets/r1/    Unitree R1 description
```

Datasets, checkpoints and videos are not tracked. Regenerate them with the
commands above.

## Notes

The R1 model is loaded unmodified. Torque actuators are converted to position
servos, foot sensors are added, and mesh collisions can be stripped for MJX —
all applied in memory at load time in `mpc/env.py`.

## License

BSD 3-Clause, see `LICENSE`.

The Unitree R1 description in `assets/r1/` is copyright Unitree Robotics and
distributed under BSD 3-Clause; see `assets/r1/LICENSE` and
`assets/r1/SOURCE.md`.
