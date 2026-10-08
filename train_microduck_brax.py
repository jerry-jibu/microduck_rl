"""Train the Microduck walk policy with Brax PPO on the Intel Arc A380.

Usage (from the repo root, venv activated):
    ONEAPI_DEVICE_SELECTOR=level_zero:0 python3 train_microduck_brax.py \
        --num-timesteps 4096 --out-dir runs/smoke      # pipeline smoke test
    ONEAPI_DEVICE_SELECTOR=level_zero:0 python3 train_microduck_brax.py \
        --num-timesteps 200000 --out-dir runs/overnight

Outputs in --out-dir:
    training_log.csv      one row per eval (steps, episode reward, ...)
    params_step*.pkl      policy params snapshot at each eval
    params.pkl            final params (load with rollout_microduck.py)
    rollout_final.npz     greedy rollout trajectory of the final policy
    summary.json          config + wall time + final metrics

Honest expectation on the A380: ~7 policy env-steps/s at 32 envs. A 200k-step
run is roughly a day. Use it to WATCH learning happen, not to train a deployable
policy (that is what the repo's HF Jobs pipeline is for).
"""

import argparse
import csv
import dataclasses
import json
import pickle
import time
from pathlib import Path

import jax
import numpy as np
import brax_jax11_compat  # noqa: F401  (must precede brax.training imports)
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train

from microduck_brax_env import MicroduckWalkEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="runs/run_%s" % time.strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--num-timesteps", type=int, default=4096,
                   help="total env steps. smoke=4096, overnight=2e5")
    p.add_argument("--num-envs", type=int, default=32,
                   help="parallel envs. 40 is the A380 memory ceiling")
    p.add_argument("--episode-length", type=int, default=250,
                   help="max steps per episode (250 = 5 s at 50 Hz)")
    p.add_argument("--unroll-length", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--num-updates-per-batch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy-cost", type=float, default=5e-3,
                   help="exploration bonus. higher = try crazier actions longer")
    p.add_argument("--discounting", type=float, default=0.97)
    p.add_argument("--num-evals", type=int, default=5,
                   help="evaluations during training (each eval costs env steps!)")
    p.add_argument("--num-eval-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=200,
                   help="greedy rollout saved for viewing (200 = 4 s)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(dataclasses.asdict(args) if dataclasses.is_dataclass(args) else vars(args),
                  f, indent=2)
    print(f"JAX devices: {jax.devices()}")
    print(f"config: {vars(args)}")

    env = MicroduckWalkEnv()
    print(f"env: obs={env.observation_size} action={env.action_size} (50 Hz policy)")

    rows = []

    def progress_fn(num_steps, metrics):
        # metrics: dict of scalar jnp arrays (eval/... and train/...)
        flat = {k: float(v) for k, v in metrics.items() if hasattr(v, "item")}
        row = {"num_steps": num_steps, "wall_s": round(time.time() - T0, 1), **flat}
        rows.append(row)
        if rows and len(rows) > 1:
            keys = rows[0].keys() | row.keys()
        else:
            keys = row.keys()
        with open(out / "training_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(keys))
            w.writeheader()
            w.writerows(rows)
        er = flat.get("eval/episode_reward", float("nan"))
        el = flat.get("eval/episode_length", float("nan"))
        print(f"[eval] steps={num_steps} wall={row['wall_s']}s "
              f"episode_reward={er:.3f} episode_length={el:.1f}", flush=True)

    def policy_params_fn(num_steps, make_policy, params):
        # snapshot at every eval so you can diff behaviors across training
        with open(out / f"params_step{num_steps}.pkl", "wb") as f:
            pickle.dump(params, f)

    T0 = time.time()
    make_policy, params, metrics = ppo_train.train(
        environment=env,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=True,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        seed=args.seed,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )
    train_s = time.time() - T0
    print(f"training done in {train_s/60:.1f} min")

    with open(out / "params.pkl", "wb") as f:
        pickle.dump(params, f)

    # greedy rollout of the final policy, saved for viewing without jax
    # normalize MUST be wired explicitly outside ppo.train — default is
    # identity, which silently skips the observation normalizer
    from brax.training.acme import running_statistics

    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size, action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    policy = make_policy(params, deterministic=True)
    rng = jax.random.PRNGKey(args.seed)
    from jax import numpy as jp

    state = env.reset(rng)
    qpos_hist, qvel_hist, reward_hist = [], [], []
    for _ in range(args.rollout_steps):
        action, _ = policy(state.obs, rng)
        state = env.step(state, action)
        qpos_hist.append(np.asarray(state.pipeline_state.qpos))
        qvel_hist.append(np.asarray(state.pipeline_state.qvel))
        reward_hist.append(float(state.reward))
        if float(state.done) > 0:
            break
    np.savez(out / "rollout_final.npz",
             qpos=np.array(qpos_hist), qvel=np.array(qvel_hist),
             reward=np.array(reward_hist), ctrl_home=np.asarray(env._ctrl0))
    steps_done = len(qpos_hist)
    print(f"final greedy rollout: {steps_done} steps ({steps_done/50:.1f} s), "
          f"fell={'YES' if float(state.done) > 0 else 'no'}, "
          f"sum reward={sum(reward_hist):.2f}")

    with open(out / "summary.json", "w") as f:
        json.dump({**vars(args),
                   "train_wall_s": round(train_s, 1),
                   "rollout_steps": steps_done,
                   "rollout_fell": bool(float(state.done) > 0),
                   "rollout_sum_reward": sum(reward_hist)}, f, indent=2)
    print(f"artifacts in {out}/  ->  view with rollout_microduck.py")


if __name__ == "__main__":
    main()
