# 四条臂的地形对照（2026-10-30 夜）

写在这里的判据是**在看到任何结果之前**定的。这样第二天不会有人（包括我）拿结果去凑解释。

## 问题

从平面策略续训到生成地面，500 iter 内熵从 −2 涨到 +13、`mean_std` 从 0.22 涨到 0.46、奖励从 106 掉到 41。而同一策略在地形上回放更好（MAE 0.0328 vs 0.0384），物理量（摩擦、接触模型、出生高度、稳态姿态）逐项一致。

所以问题不是「地形难」，是**地形改变了训练动力学**。四条臂各回答一个不同的子问题，缺一条就分不清是哪一层。

## 四条臂

| 臂 | 起点 | 地面 | 粗糙度 | 迭代 | 回答什么 |
|---|---|---|---|---|---|
| **B** | 从零 | 生成（`--terrain`） | 5 mm | 3000 | 地形到底能不能训 |
| **A** | 从零 | 平面 | — | 3000 | 同代码下的平地参照（当前代码相对 `06-20-54` 有没有漂） |
| **C** | 从零 | 生成 | 10 mm | 3000 | 结论随振幅变不变 |
| **D** | `06-20-54/model_9999` | 生成 | 5 mm | 3000 | 续训是「崩了」还是「只是慢」 |

共同条件：`task jumper.tripod`、`seed 2`、`num_envs 4096`、`headless`、任务默认（含 `push_robot` 课程），串行跑，同一把 GPU 锁。

**顺序是 B A C D**：B 走的是新代码路径（`HfRandomUniformTerrainCfg` 的随机分支，5 mm 从未真跑过），先跑它，坏了当晚就知道。

## 判据（先写，后看）

记 `末 100 条均值`，数据源是各 run 目录的 tensorboard events。参照量 `A`。

- **R1 地形可训**：`B.reward ≥ 0.8 × A.reward` 且 `B.mean_std ≤ 1.5 × A.mean_std` 且 `B.entropy < 0`
  → 地形能训，L0 的 500 iter 判决作废；阶梯每级预算改为 ≥3000 iter，且从零在地形上起，不再从平面策略续。
- **R2 地形阻断**：`B.entropy > 0` 或 `B.reward < 0.5 × A.reward`
  → 停在诊断上：把同一 `qpos` 注入平面与地形两个 env，逐步比观测与逐项奖励，定位是哪一项在生成地面上取值不同。**这条需要我在场，不能挂机做。**
- **R3 振幅无关**：`|C.reward − B.reward| < A 的末 100 条自波动`（用 A 自己的极差估）
  → 粗糙度不是变量，问题出在「生成地面 vs 平面」这个二值差异，阶梯的振幅维度没有意义，要换设计。
- **R4 续训只是慢**：`D.reward ≥ 0.8 × A.reward` 且 `D.entropy < 0`
  → 崩的是「500 iter 之后就看结论」这个做法，不是地形；闸门与阶梯的每级时长都要按 3000 iter 重设。

R1 与 R4 可以同时成立（从零能训、续训也能恢复），R2 与 R3 不矛盾（地形阻断且振幅无关是同一件事的两面）。

## 这四条臂不做的事

- 不跑 benchmark（checkpoint 还没选，跑评测等于给四个未定级的策略发成绩单）。
- 不调奖励、不改代码（除了已经在 `feat/terrain-switch` 上的地形开关本身）。
- 不做 3 种子：种子方差是用来量化**已确认的方向**的，不是用来找方向的。

## 明早的核验清单

1. `manifests/terrain-4arms.result.json`：四条 `ok` 是否为 true，有没有 `problems`。
2. 四条 `status.json`：`status`、`rc`、`elapsed_s`、`limit_min_w`/`limit_max_w`、`mem_mib`。
3. 末 100 条对照表：`Train/mean_reward`、`Metrics/twist/error_vel_xy`、`error_vel_yaw`、`Episode_Reward/tripod_gait`、`Policy/mean_std`、`Loss/entropy`、`Metrics/slip_velocity_mean`、`Episode_Termination/fell_over`（**计数**，除以 `envs × steps` 才是率）。
4. 按 R1–R4 逐条写下命中/未命中，再决定下一步。
