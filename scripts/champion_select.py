"""Champion-select evaluation: measure SURVIVAL, not mean reward.

Motivation: the fall-library collector found 17/32 envs surviving a full
5-second episode under the FINAL 300k-v2 checkpoint, while eval means hovered
~9.4 — the policy's behavior is multi-modal and mean-reward hides the best
modes. This script rolls every checkpoint deterministically (32 envs, 250
steps = 5 s) and reports the full-survival rate + survival distribution per
checkpoint, so "next step = better selection vs next step = physics" gets a
data-driven answer.

GPU required (~2 min per checkpoint after one compile).
Output: runs/champion_select/report.csv + console table sorted by full-survival.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--episode-length", type=int, default=250)
    p.add_argument("--chunk", type=int, default=25,
                   help="physics steps per jitted chunk (backend constraint)")
    p.add_argument("--out", default="runs/champion_select/report.csv")
    p.add_argument("--seeds", type=int, default=3,
                   help="reset-seed repeats per checkpoint (variance estimate)")
    return p.parse_args()


def main():
    args = parse_args()
    import jax
    from jax import numpy as jp
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from microduck_brax_env import MicroduckWalkEnv

    print(f"devices: {jax.devices()}", flush=True)
    env = MicroduckWalkEnv()
    # networks are built PER-ARCHITECTURE below: checkpoints from different
    # eras have different widths (32x4 vs 128x4) — probe the first checkpoint

    n_chunks = args.episode_length // args.chunk

    def build_for(ckpt):
        """Probe hidden widths from the pickle, build matching networks.
        params tree holds hidden_0..N where hidden_N is the OUTPUT layer
        (kernel -> 2*action): drop it from the hidden-width list."""
        with open(ckpt, "rb") as f:
            params = pickle.load(f)
        tree = params[1]["params"]
        hs = sorted((k for k in tree if k.startswith("hidden_")),
                    key=lambda s: int(s.split("_")[1]))
        widths = [tree[h]["kernel"].shape[1] for h in hs[:-1]]  # drop output
        nets = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size, action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_hidden_layer_sizes=tuple(widths))
        return params, nets.policy_network.apply

    batched_cache = {}

    def get_batched(policy_apply):
        if policy_apply not in batched_cache:
            @jax.jit
            def chunk_rollout(norm_params, pol_params, state, rng):
                def body(carry, _):
                    state, rng = carry
                    act_rng, rng = jax.random.split(rng)
                    logits = policy_apply(norm_params, pol_params, state.obs)
                    act = jp.tanh(logits[..., :env.action_size])
                    state = env.step(state, act)
                    return (state, rng), (state.pipeline_state.qpos,
                                          state.pipeline_state.qvel, state.done)

                (state, rng), outs = jax.lax.scan(
                    body, (state, rng), None, length=args.chunk)
                return state, rng, outs

            batched_cache[policy_apply] = jax.jit(
                jax.vmap(chunk_rollout, in_axes=(None, None, 0, 0)))
        return batched_cache[policy_apply]

    rows = []
    for ckpt in args.checkpoints:
        params, policy_apply = build_for(ckpt)

        def make_batched(apply_fn):
            @jax.jit
            def chunk_rollout(norm_params, pol_params, state, rng):
                def body(carry, _):
                    state, rng = carry
                    act_rng, rng = jax.random.split(rng)
                    logits = apply_fn(norm_params, pol_params, state.obs)
                    act = jp.tanh(logits[..., :env.action_size])
                    state = env.step(state, act)
                    return (state, rng), (state.pipeline_state.qpos,
                                          state.pipeline_state.qvel, state.done)

                (state, rng), outs = jax.lax.scan(
                    body, (state, rng), None, length=args.chunk)
                return state, rng, outs

            return jax.jit(jax.vmap(chunk_rollout, in_axes=(None, None, 0, 0)))

        batched = make_batched(policy_apply)
        norm_params, pol_params = params[0], params[1]

        surv_steps_all, full_all, reward_all = [], [], 0.0
        for seed in range(args.seeds):
            rngs = jax.random.split(jax.random.PRNGKey(100 + seed), args.envs)
            states = jax.jit(jax.vmap(env.reset))(rngs)
            rng = jax.random.split(jax.random.PRNGKey(7 + seed), args.envs)

            qpos_c, done_c, rew_c = [], [], []
            for _ in range(n_chunks):
                states, rng, outs = batched(norm_params, pol_params, states, rng)
                qpos_c.append(outs[0])
                done_c.append(outs[2])
                rew_c.append(0.0)  # filled below via done+qpos proxy-free path

            qpos = np.concatenate([np.asarray(x) for x in qpos_c], axis=1)
            done = np.concatenate([np.asarray(x) for x in done_c], axis=1) > 0

            # survival steps = first done index (or full length)
            for e in range(args.envs):
                falls = np.where(done[e])[0]
                surv_steps_all.append(falls[0] if len(falls) else done.shape[1])
            full_all.append(int((~done.any(axis=1)).sum()))

        surv = np.array(surv_steps_all)
        n = args.envs * args.seeds
        rows.append({
            "checkpoint": Path(ckpt).name,
            "full_survival_rate": round(sum(full_all) / n, 3),
            "median_survival_steps": int(np.median(surv)),
            "p90_survival_steps": int(np.percentile(surv, 90)),
            "mean_survival_steps": round(float(surv.mean()), 1),
            "n_rollouts": n,
        })
        print(f"{Path(ckpt).name}: full={rows[-1]['full_survival_rate']:.0%} "
              f"median={rows[-1]['median_survival_steps']} "
              f"p90={rows[-1]['p90_survival_steps']} "
              f"mean={rows[-1]['mean_survival_steps']}", flush=True)

    rows.sort(key=lambda r: -r["full_survival_rate"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== ranking by full-survival rate ===")
    for r in rows:
        print(f"{r['full_survival_rate']:.0%}  {r['checkpoint']}  "
              f"(median {r['median_survival_steps']} steps)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
