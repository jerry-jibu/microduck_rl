# Microduck × Intel Arc A380：探索全记录（2026-10-07 → 10-11）

> 一份给后来者的完整档案：我们如何在一块 6 GB 的 Intel 独显上跑通双足机器人
> RL 训练闭环、踩过哪些坑、用实验排除了哪些假设、以及当前悬而未决的问题。
> 所有数字均为本机实测。配套工具与代码均已入库（fork: jerry-jibu/microduck_rl，
> 分支 develop）。

---

## 0. 一图流总结

```
目标：在 Intel Arc A380 上复现 Pollen Robotics Microduck 双足机器人的 RL 训练管线
成果：
  ✅ MJX 物理仿真在 A380 上跑通（75.7 env-steps/s 裸物理，~3-5 steps/s 全链路 PPO）
  ✅ 纯 JAX Brax 环境（官方 61 维观测契约 + 两个 sim2real 特性）
  ✅ PPO 训练三连跑（100k / 300k / 300k v2），曲线、checkpoint、GIF 全链路
  ✅ ONNX 导出（归一化烘焙 + 数值验证 2.5e-7）
  ✅ 五课中文新手课程 + 用户指南 + 诊断工具箱
  ❌ 策略卡死在 ~12 步（0.25 s）存活平台——6 个 checkpoint × 96 回放 0% 站满
当前病灶（实验指向）：kp=50 位置伺服下 STAND 站姿物理性不可自救
下一步：kp 对照实验（进行中）→ BAM 电压执行器移植（需官方 bam 包源码）
```

---

## 1. 环境搭建（2026-10-07）

**硬件**：Intel Arc A380（6 GB 显存，实测 3.38 TFLOP/s fp32）+ 核显 UHD 770。
**软件**：`~/robotics_env` venv——jax 0.11.2 + jax-oneapi-plugin（实验性）+
mujoco/mjx 3.15.0 + brax 0.14.2。

三个当天就锁定的配置事实：

| 事实 | 证据 |
|---|---|
| `ONEAPI_DEVICE_SELECTOR=level_zero:0` 是唯一正确写法 | `:0`=A380、`:1`=核显（XLA 无法编译，报 `target: Xe1`）；三段式 `level_zero:gpu:0` 直接 abort（"Too many colons"） |
| 该后端**首次编译极慢**且 OOM 时**段错误而非报错** | 单环境步编译 28 s；错误日志只在 rematerialization 警告里给真实工作集 |
| 机器有两块 Intel GPU，必须把 JAX 锁到独显 | 已写入 venv activate（第 71 行） |

---

## 2. 让真实鸭模在 MJX 上跑起来：三只拦路虎

官方模型：`src/mjlab_microduck/robot/microduck/`（onshape-to-robot 导出，
robot_walk.xml + scene_walk.xml，STAND/SIT/FOLD 关键帧，nq=21/nu=14）。

| 拦路虎 | 症状 | 解法 | 效果 |
|---|---|---|---|
| **求解器迭代** | XML 100/50 是 CPU 配置；牛顿迭代展开进 XLA 图 | 烘 25/10 进变体场景 | 工作集 218→37.5 GiB（batch 64→16） |
| **脚掌网格**（主因） | 碰撞体只有 2 个脚掌，但 sole 网格 7900 顶点，凸包-平面接触内核爆炸（2.3 GiB/环境） | 盒图元替换（等尺寸 bbox）——对齐官方 walk 家族"feet-only"策略 | 0.084 GiB/环境（28 倍降） |
| **批量上限** | batch 64 段错误（5.35 GiB > ~4 GB 预算） | batch 32 为安全值（40 极限） | ✅ 75.7 env-steps/s |

生成器：`scripts_dev_make_mjx_variant.py` → `scene_walk_mjx.xml`
（盒脚 + 求解器 25/10 + 跟踪相机）。**教训：MJX 不是"换个 GPU 的 MuJoCo"，
CPU 精细配置会数量级放大 XLA 内存。**

---

## 3. 纯 JAX Brax 环境（microduck_brax_env.py）

对官方 mjlab 管线契约的忠实移植（帧语义全部 CPU 实证）：

- **61 维观测**：joint_pos(14) + joint_vel(14) + base_ang_vel(3, 机体系) +
  重力投影(3) + last_action(14) + 命令块(13，vx 采样其余零填充)。
  帧语义实测：`qvel[0:3]` 世界系线速度、`qvel[3:6]` 机体系角速度（gyro 交叉验证）。
- **动作**：STAND ctrl ± 0.5 rad 目标角偏移（位置伺服追，与真机同构）。
- **摔倒判定**：z < 0.08 m 或倾角 >60°（0.12 是站高，不能当阈值）。
- **奖励**：速度跟踪高斯(σ=0.3) + 活着 0.25 + 微量动作平滑。
- **sim2real v1**：观测噪声（官方数值：joint_pos ±0.001 / joint_vel ±0.25 /
  gyro ±0.03 / grav ±0.01）+ 3-6 步执行器延迟 FIFO（BAM delay 参数）。

踩过的集成坑（每个都有回归测试）：
- brax 0.14 调 jax 0.11 已删除的 `device_put_replicated` → shim
  （`brax_jax11_compat.py`，`PartitionSpec` 要用全名）；
- `state.replace(info={...})` 整字典替换会扫掉训练包装器的键 →
  scan carry 结构断裂 → 必须原位更新；
- **训练器外重建推理必须手动接 `running_statistics.normalize`**
  （默认 identity 静默跳过归一化——靠 ONNX 对拍门抓到的，见 §6）。

测试套件 `tests/test_brax_env_cpu.py`：**10 项 CPU 测试**（30 秒跑完），
锁定观测布局、噪声边界、延迟语义、carry 结构、变体 XML 配置。

---

## 4. 训练三连跑与性能账本

| 跑 | 步数 | 时长 | 配置 | 结果 |
|---|---|---|---|---|
| smoke | 4,096 | 23 min | 32 宽网络，无噪声 | 6.6→7.8（管线验证） |
| overnight_100k | 107,520 | 5.8 h | 同上 | 6.6→峰值 9.4@46k→8.6 收盘，2 次塌陷 |
| overnight_300k_v1 | 304,640 | 11.8 h | 同上，250 步回合 | 峰值 9.94@130k（历史新高），3 次塌陷 |
| overnight_300k_v2 | 210k+183k | 4.7 h + 7.3 h | **128×4 网络 + 反向课程(p=0.3) + lr 三阶段** | 峰值 9.25（更难环境），塌陷依旧且**对齐阶段边界** |

性能实测：裸物理 75.7 steps/s（batch 32）；PPO 全链路 3.1-7.2 steps/s
（梯度更新吃掉一半；无并行 CPU 任务时更快）。预算公式：`时长 ≈ 步数/3 秒 + 4 分钟`。

**运维事故两起**（都已写进指南 §10）：
1. 训练运行中并行跑 CPU 端 vmap 导出 → XLA 编译吃 37 GB 主机内存 →
   内核 OOM 连带杀掉训练。**纪律：长训练期间禁并行 CPU JAX 编译。**
2. brax 无内置 lr 衰减 → 自制"阶段重启式衰减"（restore_params 接力）→
   阶段 2 的 CSV 新指标列炸掉 DictWriter（损失 5 小时）→ 修复 + 加
   `--resume-params` 热启动接力。

---

## 5. 工具箱（都是这次踩坑的产物）

| 工具 | 用途 |
|---|---|
| `train_microduck_brax.py` | PPO 封装：CSV 曲线/逐 eval checkpoint/多阶段 lr/反向课程接线 |
| `rollout_microduck.py` | 回放 npz 或现场推理 → 跟踪相机 GIF / 交互 viewer |
| `scripts/export_brax_onnx.py` | ONNX 导出，归一化烘焙 + 对拍门（1e-4 硬校验） |
| `scripts/collect_fall_states.py` | 摔倒前状态库采集（分块 jit——单 scan(250) 编译不完） |
| `scripts/champion_select.py` | **站满率评估器**（跨架构自动探测，按生存率排名） |
| `scripts/standing_probe.py` | 零动作站立存活探针（物理病灶隔离） |
| `test_biped_mjx_batch.py` | 裸物理吞吐门禁 |
| `tests/test_brax_env_cpu.py` | 10 项 CPU 契约测试 |

后端经验法则：**vmap 必须套 jit**（裸 vmap 逐原语派发，80 分钟 vs 2 分钟）；
**scan 要分块**（25 步/块，长 scan 单图编译病态）；参数做成 jit 入参可跨
checkpoint 共享编译；vmap 内切片用省略号；多平台 context 崩溃 → 一切纯
CPU 活加 `JAX_PLATFORMS=cpu`。

---

## 6. ONNX 导出（部署链路的第一环）

`scripts/export_brax_onnx.py`：直接构图
`obs → (x−μ)/σ（烘焙常量） → 4×(MatMul+swish) → MatMul → tanh(loc[:14])`，
强制与 brax 推理在 512 条真实轨迹观测上对拍（门槛 1e-4）。

**对拍门抓到的真 bug**：brax 的 `make_ppo_networks` 默认 identity 预处理——
`ppo.train` 内部自己接归一化，外部重建推理时必须显式传入
`preprocess_observations_fn=running_statistics.normalize`，否则静默跳过。
修复前 rollout 21 步摔是 bug 产物；修复后同一 checkpoint 28 步、单步奖励 0.72。
最终验证：**max err 2.53e-7**。产物：`runs/smoke/policy.onnx`（25 KB）。
（诚实边界：BAM 缺失，只可运行时演练，不可上真机。）

---

## 7. 诊断战役：平台是真的吗？

**现象**：三代训练、两种网络宽度（32×4 / 128×4）、有无反向课程、有无 lr
衰减——所有策略的中位存活钉死在 **11-13 步（0.22-0.26 s）**，eval 峰值
9.3-9.9 对应"活 38 步左右"，距离 250 步满回合量级差 10 倍。

三个假设，逐一检验（2026-10-11 更新）：

| 假设 | 检验 | 结论 |
|---|---|---|
| H1 阶段重启制造塌陷 | 塌陷精确对齐阶段边界（90,000 步处 -2.52） | ✅ 塌陷归因于 optimizer 状态重置，**不再用 phased restart** |
| H2 多峰分布：好策略被平均分掩盖 | `champion_select.py`：6 checkpoint × 96 确定性回放（3 种子） | ✗ **0% 站满，中位 12 步**——选择假设否决 |
| H3 物理层：kp 太低站姿不可自救 | 两次检验（详见下） | **部分证实，但被 H4 修正** |

**H3 的曲折**：第一次 GPU 对照（kp50 vs kp150 零动作存活）结果逐位相同——
发现 sed 改的 kp="50" 在 `microduck` 默认类里，**14 个执行器实际继承
`chosen_actuator` 类的 kp=0.55**，实验无效（教训：改 XML 前用
`actuator_gainprm` 验证改到了生效配置）。改用 CPU kp 扫描（秒级/配置）：
**kp ∈ {0.55, 1.65, 5, 15, 50, 150} 无一能让 STAND 自稳**（8/8 全摔，
tilt 155-173°）——被动站立在此机器人上不可实现，与真机一致（官方 stand
也是训练出的策略，非关键帧保持）。

**H4（新假设，当前最佳解释）：学习失败而非物理不可解。**
CPU CEM 神谕搜索（常量 14 维动作，8 轮×60 样本）：

| 配置 | CEM 最优常量动作存活 |
|---|---|
| scale=0.5, kp=0.55（当前环境） | **1.46 s**（零动作 0.24 s 的 6 倍！） |
| scale=1.5, kp=0.55（大力矩权限） | 1.18 s（更差——权限不是瓶颈） |
| scale=1.5, kp=5.0 | 1.48 s（无提升） |

**关键推理链**：① 静态前倾姿态（CEM 找到的）在策略动作空间内；② 训练出的
策略存活与零动作完全一致（都是 12 步）→ 策略连这个 6× 改进都没发现；
③ 加大权限无改善 → 不是力矩问题。**病灶锁定：学习设置**。嫌疑排序：
探索熵过早坍缩（entropy_cost 5e-3 下 std 收缩→困在零动作附近）、
观测归一化与近常数观测分布的交互、信用分配（12 步回合 × unroll 5）。

**下一步实验**（按嫌疑排序，全部 CPU/GPU 各 ~1 h）：
1. 只留 alive 奖励的简化任务跑 100k——PPO 若连"活得更久"都学不到
   （CEM 8 轮就找到 6×），则是学习设置问题实锤；
2. entropy_cost 1e-2 + init_std 1.5（强制更久的暴力探索）；
3. 关闭 normalize_observations 对照（61 维观测中 48 维近常数，
   Welford 归一化可能把噪声放大成主信号）。

---

## 8. 教学体系（tutorials/）

五课中文 HTML 课程（Tufte 风格 + 即时反馈测验组件 + 61 维观测点选探索器）：
概念 → 跑通管线 → 观测解剖 → 奖励工程 → 曲线诊断。四份参考卡（术语表、
命令速查、观测布局、奖励八诫）。MISSION.md 定义四条结业判据。
入口：`tutorials/index.html`；工具书：`docs/intel_arc_a380_guide.md`。

---

## 9. 悬而未决（按优先级，2026-10-11 更新）

1. **学习设置诊断**（当前主战场，见 §7 H4）：简化任务对照 / 熵参数 /
   归一化开关，三个各 ~1 h 的实验；
2. **BAM 电压执行器 JAX 移植**：施工图 `docs/bam_jax_porting_notes.md`。
   先决条件：从官方 uv 环境拿回 `bam` 包源码（本机无）；验收门：与 PyTorch
   原实现对拍 atol 1e-5；
3. **塌陷治理**：放弃 phased restart（H1 实锤）；候选——brax 的
   ADAPTIVE_KL lr 模式或单阶段恒定 lr；
4. **官方 61 维 term 顺序逐项比对**（影响与官方策略热交换）；
5. 域随机化（质量/摩擦/推搡）——sim2real 前必须补齐的一整层。

## 10. 产物索引

- 训练产物：`runs/`（smoke / overnight_100k / overnight_300k_v1 /
  overnight_300k_v2(+_resume) / fall_library / champion_select）
- 诊断报告：`runs/champion_select/report.csv`
- GIF：`runs/*/params_step*_view.gif`（每代最佳策略的可视回放）
- 文档：本档案、`docs/intel_arc_a380_guide.md`（操作手册）、
  `docs/bam_jax_porting_notes.md`（移植施工图）
- 代码：全部入库 fork `jerry-jibu/microduck_rl` develop 分支
  （mjx-intel 系列 commit：80c5b73 → 8d986e0+）
