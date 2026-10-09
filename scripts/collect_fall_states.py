"""Collect a fall-state library for the reverse curriculum.

Rolls out selected checkpoints (their own deterministic policies) in vmap'd
parallel envs, captures the states just BEFORE each fall (the "starting to
fall" instants the policy must learn to recover from), filters out states that
are already past the termination thresholds, and saves:

    runs/fall_library/fall_states.npz  {qpos: (N, 21), qvel: (N, 20)}

Backend notes (Intel oneAPI, hard-won):
- the rollout MUST be jitted — bare vmap dispatches per primitive (~80 min);
- the scan must be CHUNKED (25-step chunks, Python loop between chunks) — a
  single scan(250) program compiles pathologically slowly here;
- params are ARGUMENTS of the jitted chunk so all checkpoints share one
  compile.
GPU only (CPU serial rollout is ~9 s/step). Trained-condition falls: env
noise/delay stay on, actions come from each checkpoint's own policy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PRE_FALL_STATES = 4  # capture the K states right before the done step
CHUNK = 25           # physics steps per jitted chunk (250 = 10 chunks)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", default="runs/fall_library/fall_states.npz")
    p.add_argument("--envs", type=int, default=32, help="vmap width per checkpoint")
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--keep", type=int, default=2000, help="cap on library size")
    return p.parse_args()


def main():
    args = parse_args()
    import pickle

    import jax
    from jax import numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from microduck_brax_env import FALL_Z, MicroduckWalkEnv, TILT_PROJ_Z

    print(f"devices: {jax.devices()}", flush=True)
    env = MicroduckWalkEnv()  # noise/delay on, matching training conditions
    action_size = env.action_size

    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size, action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize)
    policy_apply = networks.policy_network.apply  # stable bound method:
    # captured once below, so params-as-arguments keep ONE compile for all
    # checkpoints (identical architecture across checkpoints)

    # one jitted chunk reused by ALL checkpoints; deterministic policy ==
    # tanh(loc), loc = logits[:, :action_size] (verified numerically against
    # make_policy, see scripts/export_brax_onnx.py)
    @jax.jit
    def chunk_rollout(norm_params, pol_params, state, rng):
        def body(carry, _):
            state, rng = carry
            act_rng, rng = jax.random.split(rng)
            logits = policy_apply(norm_params, pol_params, state.obs)
            act = jp.tanh(logits[..., :action_size])
            state = env.step(state, act)
            return (state, rng), (state.pipeline_state.qpos,
                                  state.pipeline_state.qvel, state.done)

        (state, rng), outs = jax.lax.scan(body, (state, rng), None, length=CHUNK)
        return state, rng, outs

    # params broadcast (None); state/rng are the vmapped batch axes
    chunk_rollout_batched = jax.jit(
        jax.vmap(chunk_rollout, in_axes=(None, None, 0, 0)))

    n_chunks = args.max_steps // CHUNK
    all_qpos, all_qvel = [], []

    for ckpt in args.checkpoints:
        with open(ckpt, "rb") as f:
            params = pickle.load(f)
        norm_params, pol_params = params[0], params[1]

        rngs = jax.random.split(jax.random.PRNGKey(0), args.envs)
        states = jax.jit(jax.vmap(env.reset))(rngs)
        rng = jax.random.split(jax.random.PRNGKey(1), args.envs)

        qpos_c, qvel_c, done_c = [], [], []
        for _ in range(n_chunks):
            states, rng, outs = chunk_rollout_batched(norm_params, pol_params, states, rng)
            qpos_c.append(outs[0])
            qvel_c.append(outs[1])
            done_c.append(outs[2])
        qpos = np.concatenate([np.asarray(x) for x in qpos_c], axis=1)
        qvel = np.concatenate([np.asarray(x) for x in qvel_c], axis=1)
        done = np.concatenate([np.asarray(x) for x in done_c], axis=1) > 0

        kept = 0
        for e in range(args.envs):
            falls = np.where(done[e])[0]
            if len(falls) == 0:
                continue
            t = falls[0]  # first termination
            for k in range(max(0, t - PRE_FALL_STATES), t):  # strictly pre-fall
                all_qpos.append(qpos[e, k])
                all_qvel.append(qvel[e, k])
                kept += 1
        print(f"{Path(ckpt).name}: {kept} pre-fall states "
              f"({len(falls)} falls / {args.envs} envs)", flush=True)

    qpos = np.array(all_qpos, dtype=np.float32)
    qvel = np.array(all_qvel, dtype=np.float32)

    # keep only still-alive states (spawning there must not insta-terminate):
    # trunk above FALL_Z and tilt not yet past the threshold. gravity z in
    # body frame from quat (w,x,y,z): g_z = 2(x²+y²) − 1 → −1 when upright
    quat = qpos[:, 3:7]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    x, y = quat[:, 1], quat[:, 2]
    g_z = 2.0 * (x * x + y * y) - 1.0
    alive = (qpos[:, 2] >= FALL_Z) & (g_z <= TILT_PROJ_Z)
    print(f"alive filter (z>={FALL_Z}, grav_z<={TILT_PROJ_Z}): {alive.sum()}/{len(qpos)}")
    qpos, qvel = qpos[alive], qvel[alive]

    if len(qpos) > args.keep:
        idx = np.random.default_rng(0).choice(len(qpos), args.keep, replace=False)
        qpos, qvel = qpos[idx], qvel[idx]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, qpos=qpos, qvel=qvel)
    print(f"wrote {out}: {len(qpos)} states", flush=True)


if __name__ == "__main__":
    main()
