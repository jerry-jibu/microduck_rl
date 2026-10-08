# Resources（可信资源库）

> 教学内容必须锚定高质量外部资源，不许只靠参数记忆。按信任度排序。

## 一手权威（课程引用的首选）

| 资源 | 用途 | 链接 |
|---|---|---|
| MuJoCo 官方文档 | MJCF 格式、物理模型、传感器定义 | https://mujoco.readthedocs.io/ |
| MJX 文档 | MJX 与 CPU MuJoCo 的差异、训练配置建议（求解器迭代等） | https://mujoco.readthedocs.io/en/stable/mjx/ |
| Brax 仓库 | brax 环境接口、PPO 训练器、官方 notebook | https://github.com/google/brax |
| JAX 文档 | jit/vmap/scan、设备与 sharding | https://jax.readthedocs.io/ |
| OpenAI Spinning Up | RL 核心概念（策略/PPO/奖励设计）最权威入门 | https://spinningup.openai.com/ |
| mujoco_playground | DeepMind 官方 MJX 小机器人训练库——本 track 的盒脚/低迭代做法的出处 | https://github.com/google-deepmind/mujoco_playground |

## 本仓库内（第二优先级，环境特定的真相）

| 资源 | 用途 |
|---|---|
| `docs/intel_arc_a380_guide.md` | 本机性能账本、故障排查、实验菜单 |
| `AGENTS.md` | 官方管线的全部血泪教训（奖励符号纪律、sim2real footguns） |
| `microduck_brax_env.py` | 61 维观测/奖励/摔倒判定的实现真相 |
| `src/mjlab_microduck/tasks/mdp.py` | 官方奖励函数全家（进阶阅读） |
| `docs/sitstand_policy.md`、`docs/velstand_policy.md` | 官方策略训练的实战记录 |

## 社区（获得智慧的场所）

| 社区 | 适合问什么 |
|---|---|
| Brax GitHub Issues | brax API/训练器报错（先搜后问） |
| MuJoCo Discussion Forum | MJX 内存/编译/物理问题，DeepMind 工程师出没 |
| robotsrowse / r/robotics | 泛机器人 RL 经验交流 |
