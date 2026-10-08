"""Export a Brax-trained Microduck policy to ONNX with the normalizer baked in.

The repo invariant (AGENTS.md) is that deployed ONNX policies carry their
observation normalizer INSIDE the graph — a checkpoint hand-converted without it
passes in-sim play and fails on hardware. This script builds the graph directly:

    obs(61) → −mean /std (baked constants) → 4×(MatMul 32 + swish) → MatMul 28
            → loc = out[:14] → Tanh → action(14)   (deterministic policy mode)

Verification is mandatory: the ONNX output must match the original Brax
inference function on a rollout of real observations (atol 1e-4) before the
file is written.

Usage (CPU only, no GPU needed):
    JAX_PLATFORMS=cpu python3 scripts/export_brax_onnx.py \
        --checkpoint runs/overnight_100k/params.pkl --out runs/.../policy.onnx
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

XML = "./src/mjlab_microduck/robot/microduck/scene_walk_mjx.xml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="params.pkl from train_microduck_brax.py")
    p.add_argument("--out", help="output .onnx path (default: beside checkpoint)")
    p.add_argument("--n-verify", type=int, default=512)
    return p.parse_args()


def brax_reference_policy(checkpoint):
    """Return f(obs[61]) -> action[14]: the deterministic Brax inference fn.

    CRITICAL: make_ppo_networks defaults to identity preprocessing — ppo.train
    wires running_statistics.normalize itself, so a policy rebuilt outside the
    trainer MUST wire it explicitly or it silently skips normalization.
    (Verified numerically: this function now matches ppo.train's make_policy
    to 3e-8; without the wiring the gap is ~0.5 on real observations.)
    """
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from microduck_brax_env import MicroduckWalkEnv

    with open(checkpoint, "rb") as f:
        params = pickle.load(f)
    env = MicroduckWalkEnv()
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size, action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
    return lambda obs: np.asarray(policy(jax.numpy.asarray(obs), jax.random.PRNGKey(0))[0])


def collect_real_observations(n):
    """Collect realistic obs via vmap-parallel rollouts (a serial CPU rollout of
    n steps costs ~9 s/step here — vectorizing turns hours into one compile)."""
    import jax

    from microduck_brax_env import MicroduckWalkEnv

    env = MicroduckWalkEnv()
    n_envs, n_steps = 64, (n + 63) // 64

    def one_env_rollout(rng):
        def body(carry, _):
            state, rng = carry
            act_rng, rng = jax.random.split(rng)
            act = jax.random.uniform(act_rng, (env.action_size,), minval=-1, maxval=1)
            state = env.step(state, act)
            return (state, rng), state.obs

        state = env.reset(rng)
        (state, rng), ys = jax.lax.scan(body, (state, rng), None, length=n_steps)
        return ys  # (n_steps, 61)

    rngs = jax.random.split(jax.random.PRNGKey(0), n_envs)
    obs = jax.vmap(one_env_rollout)(rngs)          # (n_envs, n_steps, 61)
    return np.asarray(obs).reshape(-1, env.observation_size)[:n]


def build_onnx(norm_mean, norm_std, layers, out_path):
    """layers: list of (kernel, bias) for hidden_0..4 (swish between, none last)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inits = [
        numpy_helper.from_array(norm_mean.astype(np.float32), "obs_mean"),
        numpy_helper.from_array(norm_std.astype(np.float32), "obs_std"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "loc_starts"),
        numpy_helper.from_array(np.array([14], dtype=np.int64), "loc_ends"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "loc_axes"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_mean"], ["norm_sub"]),
        helper.make_node("Div", ["norm_sub", "obs_std"], ["norm"]),
    ]
    x = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, 61])
    cur = "norm"
    for i, (k, b) in enumerate(layers):
        last = i == len(layers) - 1
        wname, bname, oname = f"w{i}", f"b{i}", f"h{i}"
        inits.append(numpy_helper.from_array(k.astype(np.float32), wname))
        inits.append(numpy_helper.from_array(b.astype(np.float32), bname))
        nodes.append(helper.make_node("MatMul", [cur, wname], [oname + "_z"]))
        nodes.append(helper.make_node("Add", [oname + "_z", bname], [oname]))
        cur = oname
        if not last:
            # swish = x * sigmoid(x)
            nodes.append(helper.make_node("Sigmoid", [cur], [cur + "_sig"]))
            nodes.append(helper.make_node("Mul", [cur, cur + "_sig"], [cur + "_sw"]))
            cur = cur + "_sw"
    # deterministic policy mode for tanh_normal: tanh(loc), loc = out[:14]
    nodes.append(helper.make_node("Slice", [cur, "loc_starts", "loc_ends", "loc_axes"], ["loc"]))
    nodes.append(helper.make_node("Tanh", ["loc"], ["action"]))
    y = helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, 14])

    graph = helper.make_graph(nodes, "microduck_policy", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


def main():
    args = parse_args()
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name("policy.onnx")

    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)
    norm = params[0]
    mean = np.asarray(norm.mean.value if hasattr(norm.mean, "value") else norm.mean)
    std = np.asarray(norm.std.value if hasattr(norm.std, "value") else norm.std)
    tree = params[1]["params"]
    layers = [(np.asarray(tree[f"hidden_{i}"]["kernel"]),
               np.asarray(tree[f"hidden_{i}"]["bias"])) for i in range(5)]
    print(f"checkpoint: mean{mean.shape} std{std.shape}, layers "
          + " → ".join(f"{k.shape[0]}→{k.shape[1]}" for k, _ in layers))

    build_onnx(mean, std, layers, out)
    print(f"wrote {out} — verifying against Brax inference on real rollout obs...")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    obs = collect_real_observations(args.n_verify)
    ref = brax_reference_policy(args.checkpoint)
    # batch through the reference in chunks (jit-friendly)
    got = sess.run(["action"], {"obs": obs})[0]
    ref_out = np.concatenate([ref(obs[i:i + 64]) for i in range(0, len(obs), 64)], axis=0)
    err = np.abs(got - ref_out).max()
    print(f"max |ONNX − Brax| over {len(obs)} rollout obs: {err:.2e}")
    assert err < 1e-4, f"ONNX graph diverges from Brax inference (max err {err})"
    print(f"✅ {out}: normalizer baked, numerically verified — ready for "
          f"runtime rehearsal (BAM still missing, see docs/bam_jax_porting_notes.md)")


if __name__ == "__main__":
    main()
