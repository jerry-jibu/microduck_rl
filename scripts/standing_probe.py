"""Zero-action standing survival test: the physics-fault isolation probe.

Rolls N parallel envs from the STAND pose with ZERO action for T seconds and
reports the survival distribution. Run it on scene_walk_mjx.xml (kp=50) and
scene_walk_mjx_kp150.xml (kp=150): if survival jumps with stiffness, the
"policies plateau at ~12 steps because kp=50 stance is unsalvageable"
hypothesis is confirmed and training should move to the stiffer actuator.

Usage: ONEAPI_DEVICE_SELECTOR=level_zero:0 python scripts/standing_probe.py \
           <scene.xml> [--envs 32] [--seconds 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xml")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--chunk", type=int, default=25)
    args = p.parse_args()

    import jax
    from jax import numpy as jp
    from microduck_brax_env import MicroduckWalkEnv

    print(f"devices: {jax.devices()}", flush=True)
    env = MicroduckWalkEnv(args.xml, obs_noise=False, action_delay=False)

    n_steps = int(args.seconds / 0.02)  # 50 Hz policy steps
    n_chunks = n_steps // args.chunk

    @jax.jit
    def chunk(state, rng):
        def body(carry, _):
            state, rng = carry
            act = jp.zeros(env.action_size)
            state = env.step(state, act)
            return (state, rng), (state.pipeline_state.qpos[2], state.done)

        (state, rng), outs = jax.lax.scan(body, (state, rng), None, length=args.chunk)
        return state, rng, outs

    batched = jax.jit(jax.vmap(chunk, in_axes=(0, 0)))

    surv_all = []
    for seed in range(args.seeds):
        rngs = jax.random.split(jax.random.PRNGKey(seed), args.envs)
        states = jax.jit(jax.vmap(env.reset))(rngs)
        rng = jax.random.split(jax.random.PRNGKey(9), args.envs)
        zs, ds = [], []
        for _ in range(n_chunks):
            states, rng, outs = batched(states, rng)
            zs.append(outs[0])
            ds.append(outs[1])
        done = np.concatenate([np.asarray(x) > 0 for x in ds], axis=1)
        for e in range(args.envs):
            f = np.where(done[e])[0]
            surv_all.append(f[0] if len(f) else done.shape[1])

    surv = np.array(surv_all)
    n = args.envs * args.seeds
    full = int((surv >= done.shape[1]).sum())
    print(f"{Path(args.xml).name}: full-survival {full}/{n} ({full/n:.0%}), "
          f"median {int(np.median(surv))} steps ({np.median(surv)*0.02:.2f} s), "
          f"p90 {int(np.percentile(surv, 90))} steps, mean {surv.mean()*0.02:.2f} s")


if __name__ == "__main__":
    main()
