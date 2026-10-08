# BAM 执行器 JAX 移植规范（v1 待办）

> 状态：**未移植**。本文档是移植作业的施工图，不是移植声明。
> 本地 Brax 环境目前用 XML 的 kp=50 位置伺服代替 BAM（教学版）。

## 为什么必须移植

官方管线全部策略在 BAM（voltage-controlled XL330 模型）下训练：电压控制律、
反电动势、Coulomb/Stribeck/负载相关摩擦、电池电压跌落。它承载了真机伺服的
主要未建模动态。没有它，本地策略与官方策略**不可比**，也不具备 sim2real
讨论资格。

## 为什么现在不能移植（2026-10-08 检查）

权威实现在 `bam.mjlab.BamActuator`（PyTorch，随官方 uv 环境分发），**本机
没有该包的源码**（无 .venv、无 uv 缓存、无独立仓库副本）。凭参数记忆重写
电机模型等于编造——XL330 的具体控制律/摩擦参数必须逐行对拍原文。
移植时第一步：从官方开发环境取回 `bam` 包源码（或其上游仓库）。

## 已知的确定事实（来自本仓库源码，可作移植的验收锚点）

来源 `src/mjlab_microduck/robot/microduck_constants.py`（L156-182）：

- 型号：`motor_name="xl330", model="m6"`
- 固件刚度：`kp_fw=200`（microduck 保留值；microban 用 125）
- 电池电压：`vin_range=(6.5, 8.2)` 每环境启动时采样一次
- 负载压降：`vin_drop_gain_range=(0.0, 0.2)`，`V_drop = gain · Σ|τ|`，
  硬底线 `vin_min=6.0`
- 控制延迟：`delay_min_lag=3, delay_max_lag=6`（50 Hz 步）——**已移植**
  （见 `microduck_brax_env.py` act_queue / act_lag，CPU 测试覆盖）
- 摩擦：BAM 在 `_compute_friction_budget` 内算 Coulomb + Stribeck +
  负载相关项，速度无关项乘 per-env `friction_scale`；MuJoCo 的
  `dof_frictionloss` 在 BAM 下清零（源注释，friction_dr_bam.py）
- 编码器：backlash 变体中固件位置环读 `qpos[主关节] + qpos[backlash 关节]`
  （BacklashEncoderBamActuator）
- 目标选择：`target_names_expr=(r"^(?!passive_).*",)` → 14 个舵机关节

## 移植施工顺序（拿回源码后）

1. **对拍读源**：`BamActuator.compute()` / `_compute_friction_budget()` /
   电压控制律逐行抄进 `bam_actuator_jax.py`（纯函数，`vmap` 可批）。
2. **单元对拍**：官方 uv 环境里以相同输入（电压/位置/速度网格）跑 PyTorch
   版，dump 数值表 → JAX 版在同一网格上 assert allclose(atol=1e-5)。
   没有这一步不许接线。
3. **接进 env**：`MicroduckWalkEnv(use_bam=True)`——每步先算
   `ctrl_cmd → 电压 → 力矩`，作为 `data.ctrl` 之外的施加力矩
   （`mjx.apply` generalized force）注入 scan 循环；延迟队列沿用现有实现。
4. **电压/摩擦 DR**：`vin`/`friction_scale` per-env 常量在 reset 采样
   （non-accumulating：reset 重采样，遵循仓库 DR 纪律）。
5. **A/B 冒烟**：`use_bam=True/False` 各 4096 步，BAM 版站姿应明显更稳
   （官方 kp_fw=200 vs XML kp=50 的差距）。
6. **文档**：更新 `docs/intel_arc_a380_guide.md` §9 差异表与 `AGENTS.md`
   相关注记。

## 临时替代的诚实边界

在 BAM 落地前：本地策略的性能数字**不可**与官方 wandb 指标并排比较；
"本地练好了"只支持流程结论，不支持动力学结论。
