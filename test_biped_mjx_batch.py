"""Microduck MJX throughput test on Intel Arc A380 — parameterized batch size.

Usage: python3 test_biped_mjx_batch.py [BATCH_SIZE=64] [N_STEPS=20]
Batch kept small by default: XLA/oneAPI first-compile of the batched collision
solver is the bottleneck; throughput scales with batch once compiled.
"""
import sys
import time

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
# MJX trains with far fewer solver iterations than CPU MuJoCo: the Newton
# solver's iterations unroll into the XLA graph, so the CPU-tuned 100/50 in
# the XML explodes the batched HLO (218 GiB working set -> segfault on a
# 6 GB GPU). 25/10 is the standard MJX training setting.
SOLVER_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 25
LS_ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 10
XML = sys.argv[5] if len(sys.argv) > 5 else "./src/mjlab_microduck/robot/microduck/scene_walk_mjx.xml"

xml_path = XML
print(f"正在读取官方 Microduck 双足步行场景: {xml_path}", flush=True)
mj_model = mujoco.MjModel.from_xml_path(xml_path)
if SOLVER_ITER > 0:
    mj_model.opt.iterations = SOLVER_ITER
    mj_model.opt.ls_iterations = LS_ITER
mj_data = mujoco.MjData(mj_model)
print(f"solver iterations={mj_model.opt.iterations} ls_iterations={mj_model.opt.ls_iterations}", flush=True)

print(f"正在将双足机器人拓扑编译转移至 GPU (MJX), 设备: {jax.devices()}", flush=True)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(f"开辟并行环境数量 (Batch Size): {BATCH_SIZE} ...", flush=True)
batch_mjx_data = jax.vmap(lambda _: mjx_data)(jp.arange(BATCH_SIZE))

key = jax.random.PRNGKey(42)
random_actions = jax.random.uniform(key, (BATCH_SIZE, mjx_model.nu), minval=-1.0, maxval=1.0)


@jax.jit
def parallel_step(data_batch, actions):
    data_batch = data_batch.replace(ctrl=actions)
    return jax.vmap(lambda d: mjx.step(mjx_model, d))(data_batch)


print("正在编译双足物理算子内核 (XLA 硬件编译)...", flush=True)
start = time.time()
compiled_data = parallel_step(batch_mjx_data, random_actions)
compiled_data.qpos.block_until_ready()
print(f"双足机器人首次编译与步进耗时: {time.time() - start:.4f} 秒", flush=True)

print(f"连续执行 {N_STEPS} 步硬件步进评估吞吐性能...", flush=True)
start = time.time()
for _ in range(N_STEPS):
    compiled_data = parallel_step(compiled_data, random_actions)
compiled_data.qpos.block_until_ready()
end = time.time()

fps = (BATCH_SIZE * N_STEPS) / (end - start)
print(f"🎉 真实双足机器人并行物理推演成功！", flush=True)
print(f"🔥 Intel Arc A380 双足仿真吞吐量 (FPS): {fps:.2f} 帧/秒", flush=True)
print(f"   (批量={BATCH_SIZE}, 每步 {(end-start)/N_STEPS*1000:.1f} ms, 物理步长 {mj_model.opt.timestep} s)", flush=True)
