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
import functools
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
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="phase-1 lr (also the only lr without --lr-phases)")
    p.add_argument("--lr-phases", default="",
                   help="step-decay via phased restarts, e.g. "
                        "'120000:3e-4,90000:1e-4,90000:3e-5'. Each phase "
                        "resumes the previous params (normalizer+policy) with "
                        "a fresh optimizer and the lower lr")
    p.add_argument("--entropy-cost", type=float, default=5e-3,
                   help="exploration bonus. higher = try crazier actions longer")
    p.add_argument("--discounting", type=float, default=0.97)
    p.add_argument("--policy-layers", default="32,32,32,32",
                   help="policy MLP hidden widths, comma-separated")
    p.add_argument("--fall-states", default="",
                   help="fall_states.npz from scripts/collect_fall_states.py — "
                        "enables reverse-curriculum spawns")
    p.add_argument("--spawn-pfall", type=float, default=0.3,
                   help="probability a reset spawns from a pre-fall state")
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

    env = MicroduckWalkEnv(fall_states_path=args.fall_states or None,
                           spawn_pfall=args.spawn_pfall)
    print(f"env: obs={env.observation_size} action={env.action_size} (50 Hz policy)"
          + (f", reverse-curriculum p={args.spawn_pfall}" if args.fall_states else ""))

    layers = tuple(int(x) for x in args.policy_layers.split(","))
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=layers)

    # step-decay via phased restarts: each phase resumes params with a fresh
    # optimizer at the lower lr (brax has no decay schedule built in)
    phases = []
    if args.lr_phases:
        for spec in args.lr_phases.split(","):
            steps_s, lr_s = spec.split(":")
            phases.append((int(steps_s), float(lr_s)))
    else:
        phases = [(args.num_timesteps, args.learning_rate)]
    evals_per_phase = max(1, round(args.num_evals / len(phases)))

    rows = []
    offset = [0]  # cumulative steps from earlier phases

    def progress_fn(num_steps, metrics):
        # metrics: dict of scalar jnp arrays (eval/... and train/...)
        flat = {k: float(v) for k, v in metrics.items() if hasattr(v, "item")}
        row = {"num_steps": num_steps + offset[0], "wall_s": round(time.time() - T0, 1), **flat}
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
        print(f"[eval] steps={num_steps + offset[0]} wall={row['wall_s']}s "
              f"episode_reward={er:.3f} episode_length={el:.1f}", flush=True)

    def policy_params_fn(num_steps, make_policy, params):
        # snapshot at every eval so you can diff behaviors across training
        with open(out / f"params_step{num_steps + offset[0]}.pkl", "wb") as f:
            pickle.dump(params, f)

    T0 = time.time()
    params = None
    make_policy = None
    for i, (steps, lr) in enumerate(phases):
        print(f"=== phase {i + 1}/{len(phases)}: {steps} steps @ lr {lr} "
              f"(layers {layers}) ===", flush=True)
        make_policy, params, metrics = ppo_train.train(
            environment=env,
            num_timesteps=steps,
            learning_rate=lr,
            restore_params=params,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
            entropy_cost=args.entropy_cost,
            discounting=args.discounting,
            unroll_length=args.unroll_length,
            batch_size=args.batch_size,
            num_minibatches=args.num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            normalize_observations=True,
            num_evals=evals_per_phase,
            num_eval_envs=args.num_eval_envs,
            seed=args.seed + i,
            network_factory=network_factory,
            progress_fn=progress_fn,
            policy_params_fn=policy_params_fn,
        )
        with open(out / f"params_phase{i + 1}.pkl", "wb") as f:
            pickle.dump(params, f)
        offset[0] += steps
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
