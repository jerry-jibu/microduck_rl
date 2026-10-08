"""CPU tests for the Intel-A380 Brax track — no GPU needed.

Run:  JAX_PLATFORMS=cpu python3 tests/test_brax_env_cpu.py
(or pytest tests/test_brax_env_cpu.py)

Locks in the invariants of the port: 61D obs contract, upright gravity,
finite rewards, padded command block, and the MJX variant XML's physics knobs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must precede jax import

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from microduck_brax_env import OBS_SIZE, MicroduckWalkEnv

XML = "src/mjlab_microduck/robot/microduck/scene_walk_mjx.xml"


def test_env_constructs_and_resets():
    env = MicroduckWalkEnv(XML)
    assert env.observation_size == OBS_SIZE == 61
    assert env.action_size == 14
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (61,)
    assert bool(jp.isfinite(state.obs).all()), "obs must be finite at reset"
    assert float(state.done) == 0.0


def test_upright_projected_gravity_at_reset():
    env = MicroduckWalkEnv(XML)
    state = env.reset(jax.random.PRNGKey(1))
    # obs layout: [joint_pos(14), joint_vel(14), base_ang_vel(3), proj_grav(3), ...]
    proj_grav = np.asarray(state.obs[31:34])
    assert np.allclose(proj_grav, [0, 0, -1], atol=0.15), \
        f"upright gravity projection should be ~(0,0,-1), got {proj_grav}"


def test_step_finite_and_command_block_padded():
    env = MicroduckWalkEnv(XML)
    state = env.reset(jax.random.PRNGKey(2))
    action = jp.zeros(env.action_size)
    for _ in range(3):
        state = env.step(state, action)
        assert state.obs.shape == (61,)
        assert bool(jp.isfinite(state.obs).all())
        assert bool(jp.isfinite(state.reward))
        assert float(state.done) in (0.0, 1.0)
        # command block lives in obs[48:61]; head_pose/body_pose slots must stay 0
        cmd = np.asarray(state.obs[48:61])
        assert np.all(cmd[3:] == 0.0), "unused command slots must stay zero-padded"
        # metrics are carried for logging
        assert "vx" in state.metrics


def test_fall_termination_triggers():
    """Drive the trunk below FALL_Z by setting a huge downward velocity."""
    env = MicroduckWalkEnv(XML)
    state = env.reset(jax.random.PRNGKey(3))
    broken = state.replace(
        pipeline_state=state.pipeline_state.replace(
            qpos=state.pipeline_state.qpos.at[2].set(0.05)  # below FALL_Z=0.08
        )
    )
    out = env.step(broken, jp.zeros(env.action_size))
    # one physics batch later the trunk is still low -> done must fire
    assert float(out.done) == 1.0, "a trunk below FALL_Z must terminate"


def test_mjx_variant_xml():
    m = mujoco.MjModel.from_xml_path(XML)
    assert m.opt.iterations == 25 and m.opt.ls_iterations == 10, \
        "MJX-tuned solver must stay baked into the scene variant"
    assert m.nq == 21 and m.nu == 14, "topology must match the walk model"
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX, \
            "soles must stay box primitives (mesh hulls explode XLA memory)"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "side") >= 0, \
        "tracking camera needed by rollout_microduck.py"
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    assert kid >= 0 and m.key_ctrl.shape[1] == 14


def test_step_preserves_foreign_state_keys():
    """brax training wrappers stash keys in info (steps, episode_metrics, ...);
    step() must not wipe them — lax.scan carry structure depends on it."""
    env = MicroduckWalkEnv(XML)
    state = env.reset(jax.random.PRNGKey(5))
    state.info["wrapper_steps"] = jp.zeros(())
    state.metrics["wrapper_reward"] = jp.zeros(())
    out = env.step(state, jp.zeros(env.action_size))
    assert "wrapper_steps" in out.info and "command" in out.info
    assert "wrapper_reward" in out.metrics and "vx" in out.metrics


def test_obs_noise_within_documented_bounds():
    """Noisy and clean envs must differ by no more than the documented scales
    (velocity cfg: joint_pos .001, joint_vel .25, base_ang_vel .03, grav .01)."""
    from microduck_brax_env import OBS_NOISE_SCALES
    noisy = MicroduckWalkEnv(XML, obs_noise=True)
    clean = MicroduckWalkEnv(XML, obs_noise=False)
    rng = jax.random.PRNGKey(7)
    for _ in range(3):
        s_noisy = noisy.reset(rng)
        s_clean = clean.reset(rng)
        # identical seed -> identical physics; diff is pure obs noise
        segs = {
            "joint_pos": (0, 14), "joint_vel": (14, 28),
            "base_ang_vel": (28, 31), "proj_grav": (31, 34),
        }
        for name, (a, b) in segs.items():
            diff = np.abs(np.asarray(s_noisy.obs[a:b]) - np.asarray(s_clean.obs[a:b]))
            assert diff.max() <= OBS_NOISE_SCALES[name] + 1e-6, \
                f"{name} noise exceeded documented scale"
        s_noisy = noisy.step(s_noisy, jp.zeros(noisy.action_size))
        s_clean = clean.step(s_clean, jp.zeros(clean.action_size))


def test_action_delay_defers_application():
    """With delay on (lag ∈ {3..6}), step-1 ctrl must still be STAND, and after
    8 steps of constant +1 the queue is fully flushed: ctrl = STAND + 0.5."""
    from microduck_brax_env import ACTION_SCALE
    env = MicroduckWalkEnv(XML, action_delay=True)
    state = env.reset(jax.random.PRNGKey(8))
    out1 = env.step(state, jp.ones(env.action_size))
    np.testing.assert_allclose(
        np.asarray(out1.pipeline_state.ctrl), np.asarray(env._ctrl0), atol=1e-9,
        err_msg="ctrl moved on step 1 despite ≥3-step actuator delay")
    for _ in range(8):  # > ACT_DELAY_MAX: queue fully flushed with +1
        state = env.step(state, jp.ones(env.action_size))
    np.testing.assert_allclose(
        np.asarray(state.pipeline_state.ctrl),
        np.asarray(env._ctrl0) + ACTION_SCALE, atol=1e-6,
        err_msg="ctrl did not reach the queued action after 8 steps")


def test_standing_ctrl_holds_height_initially():
    """With STAND ctrl the trunk must not sink immediately (spawn is at 0.12)."""
    env = MicroduckWalkEnv(XML)
    state = env.reset(jax.random.PRNGKey(4))
    for _ in range(5):  # 100 ms — too short to topple even a marginal stance
        state = env.step(state, jp.zeros(env.action_size))
        assert state.pipeline_state.qpos[2] > 0.10, \
            "trunk sank instantly — check soles/ctrl"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
