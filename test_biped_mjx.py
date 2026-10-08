import time
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

# 指向官方的真实双足步行场景文件
xml_path = "./src/mjlab_microduck/robot/microduck/scene_walk.xml"

print(f"正在读取官方 Microduck 双足步行场景: {xml_path}")
try:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
except Exception as e:
    print(f"解析 XML 出错: {e}")
    print("提示：如果遇到 global 坐标报错，请在终端执行 sed 清洗属性后再运行本脚本。")
    exit(1)

print("正在将双足机器人拓扑编译转移至 Intel Arc A380 (MJX/oneAPI)...")
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# 在显存里开辟 1024 个小鸭子同时进行步态探索
BATCH_SIZE = 1024
print(f"开辟并行环境数量 (Batch Size): {BATCH_SIZE} ...")
batch_mjx_data = jax.vmap(lambda _: mjx_data)(jp.arange(BATCH_SIZE))

# 为这 1024 个双足机器人的关节生成随机的控制电压/速度信号
# mjx_model.nu 会自动映射这只机器人的真实执行器（Actuators）数量
key = jax.random.PRNGKey(42)
random_actions = jax.random.uniform(key, (BATCH_SIZE, mjx_model.nu), minval=-1.0, maxval=1.0)

@jax.jit
def parallel_step(data_batch, actions):
    # 将神经网络产生的步态控制量赋给电机
    data_batch = data_batch.replace(ctrl=actions)
    # 调用底层 Intel 独显的 XLA 机器码进行并行物理步进
    return jax.vmap(lambda d: mjx.step(mjx_model, d))(data_batch)

print("正在编译双足物理算子内核 (XLA 硬件编译)...")
start = time.time()
compiled_data = parallel_step(batch_mjx_data, random_actions)
compiled_data.qpos.block_until_ready()
print(f"双足机器人首次编译与步进耗时: {time.time() - start:.4f} 秒")

print("连续执行 10 步硬件步进评估吞吐性能...")
start = time.time()
for _ in range(10):
    compiled_data = parallel_step(compiled_data, random_actions)
compiled_data.qpos.block_until_ready()
end = time.time()

fps = (BATCH_SIZE * 10) / (end - start)
print(f"🎉 真实双足机器人并行物理推演成功！")
print(f"🔥 Intel Arc A380 双足仿真吞吐量 (FPS): {fps:.2f} 帧/秒")
