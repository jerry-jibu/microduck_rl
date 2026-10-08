"""Render a trained Microduck policy rollout as an animated GIF.

Two modes:
  --replay runs/xxx/rollout_final.npz   view the trajectory saved by training
                                        (no JAX, no GPU — works anywhere)
  --checkpoint runs/xxx/params.pkl      roll out the policy fresh (uses JAX
                                        device, GPU by default)

Examples:
    python3 rollout_microduck.py --replay runs/smoke/rollout_final.npz
    ONEAPI_DEVICE_SELECTOR=level_zero:0 python3 rollout_microduck.py \
        --checkpoint runs/smoke/params.pkl --steps 250
    MUJOCO_GL=egl python3 ...   # if no display; auto-falls back to EGL anyway

Output: <source>_view.gif next to the source file.
"""

import argparse
from pathlib import Path

import numpy as np

import mujoco

XML = "./src/mjlab_microduck/robot/microduck/scene_walk_mjx.xml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="rollout_final.npz saved by train_microduck_brax.py")
    src.add_argument("--checkpoint", help="params.pkl saved by train_microduck_brax.py")
    p.add_argument("--steps", type=int, default=250, help="live rollout length (250 = 5 s)")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out", help="output GIF path (default: alongside source)")
    p.add_argument("--viewer", action="store_true",
                   help="open an interactive mujoco viewer instead of writing a GIF")
    return p.parse_args()


def load_model():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    return m, d, kid


def run_live(checkpoint, steps):
    """Roll out the trained policy on the JAX device, return (qpos, qvel, rewards)."""
    import pickle

    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from microduck_brax_env import MicroduckWalkEnv

    with open(checkpoint, "rb") as f:
        params = pickle.load(f)
    env = MicroduckWalkEnv()
    # normalize MUST be wired explicitly outside ppo.train — default is
    # identity, which silently skips the observation normalizer
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size, action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    qp, qv, rw = [], [], []
    for i in range(steps):
        action, _ = policy(state.obs, rng)
        state = env.step(state, action)
        qp.append(np.asarray(state.pipeline_state.qpos))
        qv.append(np.asarray(state.pipeline_state.qvel))
        rw.append(float(state.reward))
        if float(state.done) > 0:
            print(f"terminated (fallen) at step {i}")
            break
    return np.array(qp), np.array(qv), np.array(rw)


def render_gif(m, d, kid, qpos_hist, out_path, fps, w, h):
    from PIL import Image

    renderer = mujoco.Renderer(m, h, w)
    frames = []
    try:
        for i, q in enumerate(qpos_hist):
            d.qpos[:] = q
            d.qvel[:] = 0  # visuals only
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera="side")
            frames.append(Image.fromarray(renderer.render().copy()))
    finally:
        renderer.close()
    dur_ms = max(1, int(1000 / fps))
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=dur_ms, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames, {len(frames)/fps:.1f} s)")


def main():
    args = parse_args()
    m, d, kid = load_model()

    if args.replay:
        z = np.load(args.replay)
        qpos_hist = z["qpos"]
        src = Path(args.replay)
        print(f"replaying {len(qpos_hist)} saved steps")
    else:
        qpos_hist, _, rewards = run_live(args.checkpoint, args.steps)
        src = Path(args.checkpoint)
        print(f"live rollout: {len(qpos_hist)} steps, sum reward {rewards.sum():.2f}, "
              f"distance {qpos_hist[-1][0] - qpos_hist[0][0]:+.3f} m")

    out = Path(args.out) if args.out else src.with_name(src.stem + "_view.gif")

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(m, d) as v:
            for q in qpos_hist:
                if not v.is_running():
                    break
                d.qpos[:] = q
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                v.sync()
                import time
                time.sleep(1.0 / args.fps)
    else:
        render_gif(m, d, kid, qpos_hist, out, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
