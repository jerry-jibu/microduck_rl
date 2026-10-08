# Microduck 双足机器人 RL 仿真 — Intel Arc A380 本地开发指南

> 面向**没有机器人行业经验**的软件工程师。目标：在这台装有 Intel Arc A380 的
> 工作站上，从零跑通"双足机器人强化学习训练→看到小鸭子学会走路"的完整闭环，
> 并理解每一步在做什么。
>
> 所有数值（吞吐、显存、编译时间）均为本机实测，测试日期 2026-10-07。
>
> 🎓 **零基础？** 先读 `tutorials/index.html` 的五课中文课程（概念 → 跑通
> 管线 → 读观测 → 改奖励 → 诊断曲线），本指南作工具书用。

---

## 1. 这是什么：10 分钟背景知识

**小鸭子是谁**：Microduck 是 Pollen Robotics 的双足机器人——高约 25 cm、重约
800 g，腿上有 10 个 + 脖子/头部 4 个 Dynamixel XL330 舵机（共 14 个执行器）。
官方团队用 NVIDIA GPU 训练它的行走策略，部署到真机上。本指南讲的是**在 Intel
独立显卡上重建同一套训练管线**，用于学习和实验。

**强化学习训练机器人，一句话版本**：让 32 只虚拟小鸭子同时随机乱动，动作
"好"（往前走、不摔）给正分，动作"差"（摔倒、抖动）给负分；一个神经网络
（策略, policy）看传感器数据（观测, observation）决定每个关节发多少力
（动作, action），用 PPO 算法不断调整网络参数让得分变高。几千次迭代后，
乱动的鸭子变成会走路的鸭子。

| 术语 | 意思 | 在本仓库的位置 |
|---|---|---|
| MJCF | 机器人的 XML 物理描述（关节、质量、碰撞体） | `src/mjlab_microduck/robot/microduck/*.xml` |
| MuJoCo | 物理引擎，读入 MJCF 后模拟机器人 | pip 包 `mujoco` 3.15 |
| MJX | MuJoCo 的 JAX 版：物理计算编译成 GPU 代码 | `mujoco.mjx` |
| JAX / XLA | Google 的数组计算框架 + 编译器 | pip 包 `jax` 0.11.2 |
| oneAPI 插件 | 让 XLA 把代码编译到 Intel GPU | `jax-oneapi-plugin`（实验性！） |
| PPO | 最常用的策略梯度 RL 算法 | brax 自带 |
| Brax | 基于 JAX 的 RL 库（环境接口 + PPO 实现） | pip 包 `brax` 0.14.2 |
| episode | 一条命：从出生（reset）到摔倒或到时限 | 50 Hz 决策 × 若干秒 |
| 域随机化 (DR) | 训练时随机扰动摩擦/质量等，让策略适应真机 | v1 未移植，见 §9 |

**数据流**：

```
你的 Python 脚本
  └─ JAX 描述计算图（环境 step + 神经网络）
       └─ XLA 编译器 → oneAPI 后端 → Intel GPU 机器码
            └─ Arc A380（6 GB 显存）并行仿真 32 只鸭子
```

**为什么第一次运行很慢**：XLA 要先把"整个物理 step + 奖励计算"编译成 GPU
机器码（一次 10 秒～几分钟），之后每次步进就快了。编译慢 ≠ 运行慢，别 Ctrl+C。

---

## 2. 环境前提（本机已配好，这里供复查）

- GPU：Intel Arc A380（6 GB）。机器上还有核显 UHD 770，**必须把 JAX 锁到独显**。
- venv：`~/robotics_env`，**activate 脚本第 71 行已写**
  `export ONEAPI_DEVICE_SELECTOR=level_zero:0`（`:0` 是 A380，`:1` 是核显，
  XLA 无法为核显编译）。每次 `source ~/robotics_env/bin/activate` 自动生效。
- 三段式写法 `level_zero:gpu:0` 会让 SYCL 直接崩溃（"Too many colons"），别用。
- 验证：`python3 -c "import jax; print(jax.devices())"` → `[OneapiDevice(id=0)]`。

---

## 3. 十分钟上手：四条命令

都在仓库根目录 `~/src/github.com/pollen-robotics/microduck_rl` 下执行。

### ① CPU 冒烟（30 秒，不动 GPU）

```bash
JAX_PLATFORMS=cpu python3 tests/test_brax_env_cpu.py
```

期望输出 `6/6 passed`。这一步验证代码逻辑（观测维度、奖励有限、摔倒判定、
XML 配置），改了任何环境代码后**先跑这个**。

### ② GPU 物理门禁（约 1 分钟）

```bash
python3 test_biped_mjx_batch.py 32 20 0 0
```

期望输出 `🎉 ... 吞吐量 (FPS): ~75 帧/秒`。参数依次是：批量 32、测 20 步、
后两个 0 表示"用 XML 里烘好的求解器配置"。这一步验证 A380 物理链路。

### ③ 训练冒烟（约 30-40 分钟，挂后台）

```bash
nohup python3 train_microduck_brax.py \
    --num-timesteps 4096 --episode-length 100 --num-evals 3 \
    --out-dir runs/smoke > runs_smoke.log 2>&1 &
tail -f runs_smoke.log     # Ctrl+C 退出跟踪不影响训练
```

前 5-15 分钟是 XLA 编译（正常），之后每 1-2 分钟打印一行 eval 结果。
这个配置只够验证管线通畅，鸭子大概率还不会走路——见 §6 的时长预算。

### ④ 看结果

```bash
# 看训练曲线
column -s, -t runs/smoke/training_log.csv | less -S

# 把最终策略的 rollout 渲染成 GIF（纯 CPU，秒出）
python3 rollout_microduck.py --replay runs/smoke/rollout_final.npz
# 或加载 checkpoint 现场推理一段新的：
python3 rollout_microduck.py --checkpoint runs/smoke/params.pkl --steps 250
# 有桌面时想交互式看 3D：
python3 rollout_microduck.py --replay runs/smoke/rollout_final.npz --viewer
```

产出的 `*_view.gif` 用浏览器打开，你会看到小鸭子的 3D 回放。

---

## 4. 文件地图（本 track 新增的部分）

| 文件 | 作用 |
|---|---|
| `microduck_brax_env.py` | **核心**：Brax 风格环境。61 维观测、奖励、摔倒判定、50 Hz 决策。改环境从这里进 |
| `train_microduck_brax.py` | PPO 训练入口：CSV 曲线、checkpoint、最终 rollout |
| `brax_jax11_compat.py` | 兼容 shim：brax 0.14.2 用了 jax 0.11 已移除的 `device_put_replicated`（Intel 插件锁死 jax 0.11，无法降级），进程内补回。训练脚本已自动引入，无需手动调用 |
| `rollout_microduck.py` | 回放/推理 → GIF 或交互窗口 |
| `test_biped_mjx_batch.py` | 裸物理吞吐测试（无 RL），调性能用 |
| `scripts_dev_make_mjx_variant.py` | 生成 MJX 专用 XML 变体（见 §8） |
| `tests/test_brax_env_cpu.py` | 环境的 CPU 测试套件 |
| `src/.../scene_walk_mjx.xml` + `robot_walk_mjx.xml` | **生成的** MJX 变体场景（盒脚、求解器 25/10、跟踪相机） |
| `runs/` | 训练产物（曲线、checkpoint、GIF） |

与官方 mjlab 管线的对应：`microduck_brax_env.py` 里的观测布局/奖励思想来自
`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` 和
`src/mjlab_microduck/tasks/mdp.py`（读懂官方版本后可以回来补全本项目）。

---

## 5. 观测与动作：这份环境的"感官"和"肌肉"

**61 维观测**（策略网络每 20 ms 看一眼的向量，顺序是硬契约，别改）：

| 段 | 维度 | 内容 |
|---|---|---|
| joint_pos | 14 | 14 个关节角度（相对 STAND 姿态的偏移） |
| joint_vel | 14 | 关节角速度 |
| base_ang_vel | 3 | 躯干角速度（机体系） |
| projected_gravity | 3 | 重力方向投影到机身坐标——机器人的"倾斜感知"，直立时 = (0,0,-1) |
| last_action | 14 | 上一步发出的动作（让网络知道自己刚干了什么） |
| 命令块 | 13 | [前进速度 vx, 侧移 vy, 转向 wz] + 头部姿态(4) + 身体姿态(6) |

其中命令块只用了前 3 个槽里的 vx（训练时从 0~0.6 m/s 随机抽），其余 10 个槽
**零填充占位**——这是官方家族的接口约定，保证以后新技能可以热插拔。

**14 维动作**：每个值 ∈ [-1,1]，映射为"在 STAND 站姿基础上 ±0.5 rad 的关节
目标偏移"。位置伺服（PD 控制器，kp=50）负责追这个目标。

**奖励**（每个决策步）：
- `+exp(-((实际前进速度 - 命令 vx)/0.30)²)` —— 跟踪指令，主要分数
- `+0.25` —— 活着奖励（不摔倒就有）
- `-0.005·Σ(动作变化量)²` —— 微量平滑

**摔倒判定**：躯干高度 < 0.08 m 或倾角 > 60°（重力投影 z > -0.5）。

---

## 6. 训练时长预算（本机实测，先算钱再开跑）

实测吞吐：**PPO 全链路约 3 次 env 步/秒**（32 环境、50 Hz 决策；裸物理能到
7.5，PPO 的梯度更新和网络推理吃掉一半以上）。首个 eval 含编译约 4 分钟。

```
训练时长 ≈ num_timesteps / 3 秒  +  约 4 分钟
4096 步  ≈ 23 分钟（实测冒烟：6.6 → 7.8 分，鸭子还不会走）
10 万步  ≈ 9 小时                （白天挂机：能明显站得更久）
50 万步  ≈ 1.9 天                （过夜×2：可能走出小碎步）
300 万步 ≈ 11 天                 （官方入门级行走水平，建议挪去 HF Jobs）
```

**怎么判断在学**（看 `training_log.csv` 的 `eval/episode_reward`；冒烟跑的
参照系：起点 6.6 ≈ 摔倒前活 ~35 步）：
1. 前期：奖励低且平 → 鸭子在摔；
2. 然后：`eval/episode_length` 开始上升 → 能撑更久不摔（第一个里程碑！）；
3. 中期：reward 爬升 → 站稳 + 开始试探迈步；
4. 后期：reward 接近 1.25×episode_length → 稳定行走并跟踪速度。

每次 eval 都会存 `params_step*.pkl`，可以拿不同阶段的 checkpoint 渲染 GIF
对比行为，直观看到"从乱抖到站住"的过程。

⚠️ **纪律**（来自官方仓库的血泪教训）：训练日志里任何惩罚项均值必须 ≤ 0；
如果总奖励在涨但 episode_length 不涨，说明网络在刷分不是在学走路——减少
奖励项、先让"不摔"成立，再谈别的。

---

## 7. 实验菜单：改什么，怎么改

**改代码前**：跑一遍 `JAX_PLATFORMS=cpu python3 tests/test_brax_env_cpu.py`；
**改参数前**：把当前配置记下来（runs/*/summary.json 自动记了）；**一次只改
一件事**。

入门实验（按序做，每个都是一晚）：

1. **摘掉活着奖励**：`microduck_brax_env.py` 里 `ALIVE_BONUS = 0.0`。
   预期：学得更慢甚至摆烂——体会"稠密正反馈"的价值。
2. **收紧速度跟踪**：`TRACK_STD` 从 0.30 改成 0.15。预期：奖励更难拿，
   步态更激进——体会"高斯的宽度 = 你还在乎的误差尺度"。
3. **加平滑税**：`ACTION_RATE_WEIGHT` 从 0.005 加到 0.05。预期：抖动减少、
   步子变小——体会"正则化压制行为"。
4. **只学走直线**：reset 里 vx 从 `U(0, 0.6)` 改成固定 `0.4`。预期：收敛
   明显变快——体会"任务越简单越快"。
5. **加硬摔惩罚**：在 `step()` 里 done 分支加 `-5.0`。观察是否真的有用
   （官方的经验：终局惩罚往往不如"死了就没分"来得有效）。

**不要动的东西**（改了要么崩、要么破坏与官方家族的兼容）：
- `OBS_SIZE = 61` 和观测顺序——这是全家策略热插拔的契约；
- `qvel[0:3]` 世界系 / `qvel[3:6]` 机体系的帧语义（已实测锁定，测试覆盖）；
- `N_PHYSICS_STEPS = 10`（50 Hz 决策是部署频率）；
- `scene_walk_mjx.xml` 里的盒脚和求解器配置（§8 解释了为什么）；
- 命令块的零填充槽。

---

## 8. 性能账本：为什么你的卡跑不了 4096 只鸭子

官方 NVIDIA 管线一次训练 4096 个环境；A380 上限是 **约 40 个**。原因与实测：

| 隔离实验 | 内存工作集 | 结果 |
|---|---|---|
| 原版 XML（脚掌 = 7900 顶点网格凸包），batch 64，求解器 100/50 | 218 GiB | 段错误 |
| 原版 XML，batch 16，求解器 25/10 | 37.5 GiB | OOM |
| **脚掌换盒图元**，batch 64 | 5.35 GiB | 段错误（差一点） |
| **盒脚 + batch 32（当前配置）** | ~2.7 GiB | ✅ 75.7 steps/s |
| 求解器 10/5（代替 25/10） | 更小 | 90.7 steps/s（+20%） |

三个教训：
1. **MJX 不是"换个 GPU 就能跑"的 MuJoCo**：CPU 版配置（高迭代次数、精细
   网格碰撞）会以数量级放大 XLA 编译后的内存。MJX 生态的标准做法就是
   图元碰撞 + 低迭代求解器。
2. **本后端是内核延迟瓶颈**：32 环境和 1 环境的单步耗时几乎一样（批量几乎
   免费），所以提升吞吐靠"加到显存装不下的最大批量"，而不是减环境。
3. **每环境 84 MB vs NVIDIA 生态 ~10 MB**：这是实验性 oneAPI 后端的代码生成
   质量问题，会随版本改善。真正的训练请用官方 `--hf-jobs`（NVIDIA 云），
   本地卡用来开发和验证想法。

---

## 9. 诚实声明：本地训练的策略不能直接上真机

本地环境与官方管线的差异清单（2026-10-08 更新，**✅=已落地**）：

1. **执行器**（最大缺口）：官方用 BAM 电压模型（XL330 反电动势、
   Coulomb/Stribeck 摩擦、6.5-8.2 V 电压跌落）；本地用 XML 的 kp=50 位置
   伺服。其中**控制延迟（3-6 步）✅ 已移植**（默认开启）；BAM 本体移植
   施工图见 `docs/bam_jax_porting_notes.md`（需官方 bam 包源码）。
2. **域随机化**：官方随机化质量/摩擦/推搡上百项；本地零随机化 → 策略
   会"过拟合仿真"。
3. **观测噪声 ✅**：官方实测噪声值已按原数移植（joint_pos ±0.001、
   joint_vel ±0.25、base_ang_vel ±0.03、重力 ±0.01，默认开启）；官方的
   观测延迟包络（0-1 lag）未移植。
4. **ONNX 导出 ✅**：`scripts/export_brax_onnx.py` 把归一化器烘进图
   （Constant 节点），并强制与 brax 推理逐点对拍（atol 1e-4）后才落盘。
   产物形状与官方 61D/14D 契约一致，但 term 顺序未经官方渲染比对，
   且 BAM 缺失——只可做运行时演练，不可上真机。

所以本地的定位是：**理解 RL 训练全流程、做想法的快速实验、培养直觉**。
真机级训练回到官方管线（`uv run train ... --hf-jobs`）。

另外一个已知事实：XML 的 kp=50 伺服下，STAND 站姿**不是自稳的**（零动作
2 秒后 32 只鸭子只剩 ~6 只站着；官方靠 BAM 固件刚度 kp_fw=200 才稳）。所以
训练初期看到大量摔倒不是 bug——鸭子要先学会"主动平衡"。

---

## 10. 故障排查表

| 症状 | 原因 | 处理 |
|---|---|---|
| 进程消失，日志里有 `EXIT=139` 或 rematerialization 警告 | 显存爆了（batch 太大） | 降 batch：32 是安全值，40 是极限 |
| `RESOURCE_EXHAUSTED: Out of memory` | 同上 | 同上 |
| `Too many colons` 后进程崩溃 | `ONEAPI_DEVICE_SELECTOR` 写了三段式 | 用 `level_zero:0` |
| devices 里出现别的设备 / XLA 编译失败 | 选到了核显 | 确认 selector 是 `level_zero:0` |
| 第一次 step 卡住几分钟 | XLA 编译（正常） | 等。用 `tail -f` 看进度 |
| `target: Xe1` 架构不匹配 | 跑到核显上了 | 同 selector 项 |
| GIF 里只有地板没有鸭子 | 相机没跟踪 | 用仓库自带的 `side` 相机（已在变体 XML 里配好） |
| 日志出现 `Failed to import warp` | 本 track 不需要 warp | 无害，忽略 |
| 训练奖励一直 -0.75 左右不动 | 鸭子全摔了躺着（活着奖励没了+跟踪分 0） | 先确认 episode_length 在涨；或把 `ALIVE_BONUS` 调回 |
| **训练进程莫名消失，journalctl 有 oom-kill** | **主机内存被其他 XLA-CPU 编译挤爆**（实测一次 vmap 导出编译峰值 37 GB） | **纪律：长训练运行时不要并行跑任何 CPU 端 JAX 编译/导出**；重 CPU 活等训练结束或用 systemd-run 限内存 |
| CPU 测试挂了 | 环境契约被改坏 | 看 `tests/test_brax_env_cpu.py` 里对应的断言说明 |

---

## 11. 路线图

- **v1（进行中）**：✅ 观测噪声（官方数值）；✅ 执行器控制延迟 3-6 步；
  ✅ ONNX 导出（归一化烘焙 + 数值验证）。**待办**：BAM 电压执行器本体
  （施工图 `docs/bam_jax_porting_notes.md`，需官方 bam 包源码）；基础域
  随机化（质量/摩擦）；观测延迟包络。
- **v2**：官方 61 维 term 顺序逐项比对（当前按 mjlab 惯例实现，未经官方
  配置渲染比对）；与官方 `scripts/infer_policy.py` 部署演练对齐。
- **随时可做**：更大的网络（`make_ppo_networks` 的 hidden sizes）、奖励课程
  （vx 范围从 0→0.6 逐步放开）、摔倒恢复任务（参考官方 velstand 的设计）。

## 12. 一页速查

```bash
source ~/robotics_env/bin/activate                 # 自动带 ONEAPI_DEVICE_SELECTOR
cd ~/src/github.com/pollen-robotics/microduck_rl

JAX_PLATFORMS=cpu python3 tests/test_brax_env_cpu.py          # 30 s 代码冒烟
python3 test_biped_mjx_batch.py 32 20 0 0                     # 1 min GPU 门禁
nohup python3 train_microduck_brax.py --num-timesteps 4096 \
    --episode-length 100 --num-evals 3 --out-dir runs/smoke \
    > runs_smoke.log 2>&1 &  tail -f runs_smoke.log           # ~35 min 训练冒烟
column -s, -t runs/smoke/training_log.csv | less -S           # 看曲线
python3 rollout_microduck.py --replay runs/smoke/rollout_final.npz   # 看 GIF
python3 rollout_microduck.py --replay runs/smoke/rollout_final.npz --viewer  # 交互看
```
