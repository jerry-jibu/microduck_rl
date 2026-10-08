"""Microduck Brax/MJX environment (v0) — pure JAX, runs on the Intel Arc A380
via the oneAPI backend (set ONEAPI_DEVICE_SELECTOR=level_zero:0).

Port of the mjlab velocity task's core contract to a standard Brax env:

- Real MJCF: scene_walk_mjx.xml (box-sole collision variant — the 7900-vertex
  sole mesh hulls explode the batched XLA working set to GiB/env; see
  scripts_dev_make_mjx_variant.py). Feet-only collisions, as in the official
  'walk' family. Solver 25/10 baked into the scene.
- Official 61D actor observation layout (mjlab_microduck invariant):
  48 proprioception [joint_pos(14), joint_vel(14), base_ang_vel(3, body frame),
  projected_gravity(3), last_action(14)] + 13D command block
  [twist(3), head_pose(4) zero-padded, body_pose(6) zero-padded].
- 50 Hz policy over the 500 Hz physics: 10 MJX steps per env step.
- XML position actuators (kp=50): action = STAND ctrl + 0.5 * clipped action.
  NOTE: the official policy family trains under the BAM voltage actuator
  (kp_fw=200, voltage sag, friction) — this PD model is the v0 stand-in;
  porting BAM into JAX is the v1 milestone before any sim2real claim.

Frames (empirically verified on CPU MuJoCo, scene_walk.xml):
- free-joint qvel[0:3] is WORLD-frame linear velocity,
- qvel[3:6] is BODY-frame angular velocity (gyro at the imu site confirms),
- imu site quat is identity, so sensor views == trunk body frame.

Not ported (v1+): BAM actuator + voltage/friction DR, obs noise/delay/bias
stack, reward curriculum, mass/CoM DR, symmetry features.
"""

import mujoco
from mujoco import mjx

from brax import base
from brax.envs.base import Env, State

import jax
from jax import numpy as jp

DEFAULT_XML = "./src/mjlab_microduck/robot/microduck/scene_walk_mjx.xml"

N_JOINTS = 14          # 14 XL330 servos; qpos[0:7] free joint, qpos[7:21] joints
OBS_SIZE = 61          # official actor obs contract
N_PHYSICS_STEPS = 10   # 0.002 s * 10 = 20 ms -> 50 Hz policy
ACTION_SCALE = 0.5     # rad of joint-target offset per unit action
TRACK_STD = 0.30       # m/s, velocity tracking Gaussian width
ALIVE_BONUS = 0.25
ACTION_RATE_WEIGHT = 0.005  # deliberately tiny (repo lesson: no smoothness tax during skill discovery)
FALL_Z = 0.08          # m; STAND trunk height is 0.12 — 0.12 would insta-terminate
TILT_PROJ_Z = -0.5     # projected-gravity z above this (~60° tilt) = fallen

# -- sim2real v1: obs noise (values from official velocity cfg, uniform jitter) --
# microduck_velocity_env_cfg.py: base_ang_vel ±0.03, gravity ±0.01,
# joint_pos ±0.001, joint_vel ±0.25
OBS_NOISE_SCALES = {
    "joint_pos": 0.001,
    "joint_vel": 0.25,
    "base_ang_vel": 0.03,
    "proj_grav": 0.01,
}
# -- sim2real v1: actuator control delay (microduck_constants.py
#    _BAM_ACTUATOR_KWARGS: delay_min_lag=3, delay_max_lag=6 at 50 Hz) --
ACT_DELAY_MIN, ACT_DELAY_MAX = 3, 6
ACT_QUEUE_LEN = ACT_DELAY_MAX + 1  # FIFO depth; index -1 is the newest entry


def _quat_rotate_inverse(q: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
    """Rotate v from the world frame into the body frame of quaternion q=(w,x,y,z)."""
    w = q[0]
    u = q[1:]
    return (2.0 * w * w - 1.0) * v - 2.0 * w * jp.cross(u, v) + 2.0 * u * jp.dot(u, v)


class MicroduckWalkEnv(Env):
    """Forward-walking velocity task for the Microduck biped, Brax/MJX style."""

    def __init__(self, xml_path: str = DEFAULT_XML, reset_noise: float = 0.03,
                 obs_noise: bool = True, action_delay: bool = True):
        model = mujoco.MjModel.from_xml_path(xml_path)
        self._mjx_model = mjx.put_model(model)

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if key_id < 0:
            raise ValueError("STAND keyframe not found — is this the walk scene?")
        self._qpos0 = jp.array(model.key_qpos[key_id])  # (21,)
        self._ctrl0 = jp.array(model.key_ctrl[key_id])  # (14,)

        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        # one host->device transfer here; reset() must never call put_data (not
        # jit/vmap-compatible)
        self._init_data = mjx.put_data(model, data)

        self._reset_noise = reset_noise
        self._obs_noise = obs_noise
        self._action_delay = action_delay

    # -- brax Env contract ---------------------------------------------------
    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def action_size(self) -> int:
        return int(self._mjx_model.nu)

    @property
    def observation_size(self) -> int:
        return OBS_SIZE

    # -- internals ------------------------------------------------------------
    def _projected_gravity(self, quat: jp.ndarray) -> jp.ndarray:
        return _quat_rotate_inverse(quat, jp.array([0.0, 0.0, -1.0]))

    def _noisy(self, value: jp.ndarray, scale: float, key: jp.ndarray) -> jp.ndarray:
        if not self._obs_noise or scale == 0.0:
            return value
        return value + jax.random.uniform(key, value.shape, minval=-scale, maxval=scale)

    def _get_obs(self, data: base.State, last_action: jp.ndarray, command: jp.ndarray,
                 noise_key: jp.ndarray) -> jp.ndarray:
        nk = jax.random.split(noise_key, 4) if self._obs_noise else [None] * 4
        joint_pos = data.qpos[7:] - self._qpos0[7:]
        joint_vel = data.qvel[6:]
        base_ang_vel = data.qvel[3:6]  # body frame (verified)
        proj_grav = self._projected_gravity(data.qpos[3:7])
        joint_pos = self._noisy(joint_pos, OBS_NOISE_SCALES["joint_pos"], nk[0])
        joint_vel = self._noisy(joint_vel, OBS_NOISE_SCALES["joint_vel"], nk[1])
        base_ang_vel = self._noisy(base_ang_vel, OBS_NOISE_SCALES["base_ang_vel"], nk[2])
        proj_grav = self._noisy(proj_grav, OBS_NOISE_SCALES["proj_grav"], nk[3])
        proprio = jp.concatenate([joint_pos, joint_vel, base_ang_vel, proj_grav, last_action])
        return jp.concatenate([proprio, command])  # (48,)+(13,) = 61

    # -- API -------------------------------------------------------------------
    def reset(self, rng: jp.ndarray) -> State:
        rng_xy, rng_quat, rng_joint, rng_cmd, rng_aux = jax.random.split(rng, 5)
        qpos = self._qpos0
        qpos = qpos.at[0:3].add(jax.random.uniform(rng_xy, (3,), minval=-0.02, maxval=0.02))
        quat = self._qpos0[3:7] + jax.random.uniform(rng_quat, (4,), minval=-0.02, maxval=0.02)
        qpos = qpos.at[3:7].set(quat / jp.linalg.norm(quat))
        qpos = qpos.at[7:].add(
            jax.random.uniform(rng_joint, (N_JOINTS,), minval=-self._reset_noise, maxval=self._reset_noise)
        )
        data = self._init_data.replace(qpos=qpos, qvel=jp.zeros(20), ctrl=self._ctrl0)

        # command block: [vx, vy, wz] + head_pose(4) + body_pose(6) zero-padded.
        # vx ~ U(0, 0.6): includes near-zero so "stand" is trained for free.
        vx = jax.random.uniform(rng_cmd, (), minval=0.0, maxval=0.6)
        command = jp.zeros(13).at[0].set(vx)

        obs = self._get_obs(data, jp.zeros(self.action_size), command, rng_aux)
        zero = jp.zeros(())
        metrics = {"vx": zero, "reward_tracking": zero, "reward_alive": zero, "action_rate": zero}
        # info keys are carry structure for brax's rollout scan — constant set
        info = {
            "command": command,
            "last_action": jp.zeros(self.action_size),
            "noise_rng": rng_aux,
            # actuator delay FIFO: [-1] is the newest emitted action; applied is
            # [-1-lag] with lag ~ U{3..6} (microduck_constants.py BAM delays)
            "act_queue": jp.zeros((ACT_QUEUE_LEN, self.action_size)),
            "act_lag": jax.random.randint(rng_aux, (), ACT_DELAY_MIN, ACT_DELAY_MAX + 1),
        }
        return State(pipeline_state=data, obs=obs, reward=zero, done=zero, metrics=metrics, info=info)

    def step(self, state: State, action: jp.ndarray) -> State:
        command = state.info["command"]
        last_action = state.info["last_action"]
        noise_rng, next_noise_rng = jax.random.split(state.info["noise_rng"])

        act = jp.clip(action, -1.0, 1.0)

        # actuator control delay: the queue holds emitted actions, [-1] newest;
        # the servo sees the action emitted `lag` steps ago (lag fixed per env).
        # Positive index — dynamic negative indexing is unsupported in JAX.
        if self._action_delay:
            applied = state.info["act_queue"][ACT_QUEUE_LEN - 1 - state.info["act_lag"]]
        else:
            applied = act

        target = self._ctrl0 + ACTION_SCALE * applied

        def body(d, _):
            return mjx.step(self._mjx_model, d), None

        data, _ = jax.lax.scan(body, state.pipeline_state.replace(ctrl=target), None, length=N_PHYSICS_STEPS)

        vx = data.qvel[0]  # world-frame forward speed (robot faces +x)
        r_tracking = jp.exp(-jp.square((vx - command[0]) / TRACK_STD))
        r_alive = jp.float32(ALIVE_BONUS)
        r_action_rate = -ACTION_RATE_WEIGHT * jp.sum(jp.square(act - last_action))
        reward = r_tracking + r_alive + r_action_rate

        proj_grav_z = self._projected_gravity(data.qpos[3:7])[2]
        done = jp.float32((data.qpos[2] < FALL_Z) | (proj_grav_z > TILT_PROJ_Z))

        obs = self._get_obs(data, act, command, noise_rng)
        # in-place updates ONLY: brax's rollout scan requires the State's
        # info/metrics dicts to keep an identical key structure across steps —
        # replacing the dicts here would drop the training wrappers' keys
        # (steps, episode_metrics, first_pipeline_state, ...)
        state.info["command"] = command
        state.info["last_action"] = act
        state.info["noise_rng"] = next_noise_rng
        if self._action_delay:
            state.info["act_queue"] = jp.concatenate(
                [state.info["act_queue"][1:], act[None]], axis=0
            )
        state.metrics["vx"] = vx
        state.metrics["reward_tracking"] = r_tracking
        state.metrics["reward_alive"] = r_alive
        state.metrics["action_rate"] = r_action_rate
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )


# =============================================================================
# Self-test: parallel reset/step on the selected JAX device + sanity checks.
#   ONEAPI_DEVICE_SELECTOR=level_zero:0 python3 microduck_brax_env.py [N_ENVS]
# =============================================================================
if __name__ == "__main__":
    import sys
    import time

    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    env = MicroduckWalkEnv()
    print(f"action_size={env.action_size} observation_size={env.observation_size} "
          f"physics steps/decision={N_PHYSICS_STEPS} (50 Hz policy)")

    rngs = jax.random.split(jax.random.PRNGKey(0), num_envs)
    t0 = time.time()
    states = jax.block_until_ready(jax.vmap(env.reset)(rngs))
    print(f"reset({num_envs}) compiled+ran in {time.time()-t0:.1f}s, obs shape {states.obs.shape}")
    assert states.obs.shape == (num_envs, OBS_SIZE), "obs must be 61D per the family contract"

    @jax.jit
    def batch_step(states, actions):
        return jax.vmap(env.step)(states, actions)

    # sanity 1: standing still from STAND, zero action — how long until fall?
    t0 = time.time()
    s = states
    for _ in range(100):  # 2 s of policy time
        s = batch_step(s, jp.zeros((num_envs, env.action_size)))
    s.done.block_until_ready()
    alive = int((s.done == 0).sum())
    print(f"sanity: zero-action STAND after 2 s -> {alive}/{num_envs} envs upright "
          f"(kp=50 stance is marginal; the POLICY must learn to balance) "
          f"[compiled+ran {time.time()-t0:.1f}s]")

    # throughput: random actions
    actions = jax.random.uniform(jax.random.PRNGKey(1), (num_envs, env.action_size), minval=-1, maxval=1)
    n_iter = 50
    t0 = time.time()
    s = states
    for _ in range(n_iter):
        s = batch_step(s, actions)
    s.obs.block_until_ready()
    dt = time.time() - t0
    print(f"🔥 throughput: {num_envs * n_iter / dt:.0f} env-steps/s "
          f"({dt/n_iter*1000:.1f} ms per 50 Hz policy step, batch={num_envs})")
