import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

from mpc.env import load_model
from deepmimic.reference import ReferenceMotions
from deepmimic.env import DeepMimicEnv, DEFAULT_CFG
from deepmimic.mjx_env import MJXWalker


def seed_state(env, ref, clip, frame, key):
    st = jax.jit(env.reset)(key)
    st = st.replace(clip=jnp.int32(clip), k=jnp.int32(frame))
    d = st.data.replace(
        qpos=jnp.asarray(ref.clips[clip]["qpos"][frame], dtype=jnp.float32),
        qvel=jnp.asarray(ref.clips[clip]["qvel"][frame], dtype=jnp.float32),
        ctrl=jnp.asarray(ref.clips[clip]["qref"][frame], dtype=jnp.float32))
    d = mjx.forward(env.model, d)
    cur = env._single_obs(d, jnp.int32(clip), jnp.int32(frame),
                          env._body_vel(d), env._yaw(d), d.qpos[:2],
                          jnp.zeros(env.n_act), jnp.zeros(env.n_act))
    hist = jnp.tile(cur, (env.obs_history - 1, 1))
    return st.replace(data=d, vel_ema=env._body_vel(d), heading_ref=env._yaw(d),
                      pos_ref=d.qpos[:2], hist=hist,
                      obs=jnp.concatenate([cur, hist.ravel()]),
                      last_action=jnp.zeros(env.n_act),
                      prev_action=jnp.zeros(env.n_act))


def main():
    ref = ReferenceMotions("data/qcheck/gait_v6.h5", cyclic=True, max_seam=0.08)
    cfg = dict(DEFAULT_CFG)
    cfg["max_steps"] = 300
    mj = load_model("assets/r1/scene.xml", strip_mesh_collision=True)
    cpu = DeepMimicEnv(ref, model=mj, cfg=cfg, seed=0)
    env = MJXWalker(ref, cfg=cfg)

    clip, frame = 40, 0
    cpu.reset(clip=clip, frame=frame)
    st = seed_state(env, ref, clip, frame, jax.random.PRNGKey(0))
    stepf = jax.jit(env.step)

    rng = np.random.default_rng(3)
    print(f"{'step':>5} {'|dqpos|':>10} {'dz':>10} {'rew_cpu':>9} "
          f"{'rew_mjx':>9} {'drew':>9}")
    info = None
    for i in range(60):
        a = rng.uniform(-0.3, 0.3, cpu.n_act).astype(np.float32)
        _, rc, _, _, info = cpu.step(a)
        st = stepf(st, jnp.asarray(a))
        if i == 0 or (i + 1) % 10 == 0:
            dq = float(np.linalg.norm(np.array(st.data.qpos) - cpu.data.qpos))
            dz = float(st.data.qpos[2]) - float(cpu.data.qpos[2])
            print(f"{i + 1:>5} {dq:>10.5f} {dz:>10.5f} {rc:>9.4f} "
                  f"{float(st.reward):>9.4f} {float(st.reward) - rc:>9.4f}")

    print()
    print("reward term comparison after 60 steps:")
    print(f"  {'term':<10} {'cpu':>9} {'mjx':>9} {'diff':>9}")
    for k in ("pose", "vel", "endeff", "root", "track", "along", "cross",
              "heading", "slip", "smooth", "torque", "imitation", "task"):
        c = info["terms"][k]
        m = float(st.metrics[k])
        print(f"  {k:<10} {c:>9.4f} {m:>9.4f} {m - c:>9.4f}")


if __name__ == "__main__":
    main()
