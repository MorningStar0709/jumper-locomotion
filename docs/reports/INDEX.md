# Reports — index / 报告索引

The ten reports behind the numbers, each with a one-line summary in both languages. The
reports themselves are written in Chinese; every table, figure and number in them is
language-independent.

这十篇报告支撑 README 里的每个数字。正文为中文，附英文一句摘要；表格与数字与语言无关。

| # | Report | English | 中文 |
|:-:|:-------|:--------|:-----|
| 1 | [Terrain scan and ceiling](research-terrain-scan-ceiling-20261031.md) | Where the baseline policy stops walking, and the three tool defects that made earlier scans meaningless. | 基线策略在哪里走不动了，以及三个让早期扫描毫无意义的工具缺陷。 |
| 2 | [Gate, L0–L1](research-l0-gate-terrain-paved-20261031.md) | Building the gate itself: why yaw MAE was the wrong quantity, and the resolution limits of a replay. | 闸门怎么建的：为什么偏航 MAE 是错的量，以及回放的分辨力边界。 |
| 3 | [Terrain ladder](research-terrain-ladder-20261101.md) | Twelve training arms, walkable-versus-trainable curves, and why 3.5 cm destroyed a working 3 cm gait. | 十二个训练臂、"能走"与"能训"两条曲线，以及 3.5 cm 为何毁掉 3 cm 的步态。 |
| 4 | [3 cm mixture](research-night-3cm-mix-20261103.md) | The 1:4 mixture proportion, the audit trail of what pairing can and cannot resolve at 5 mm. | 1:4 混合配比，以及 5 mm 上配对能分辨什么、不能分辨什么。 |
| 5 | [From-scratch mixture](research-mix14-from-scratch-20261104.md) | Whether a mixture can replace "walk there first" — measured, it cannot. | 混合地面能否替代"先走过去"——实测不能。 |
| 6 | [Heading drift](research-heading-drift-20261102.md) | How far a policy drifts over 10 m, measured directly, and why integrating the mean yaw rate lies. | 直走 10 米实际偏多少，以及为什么用平均偏航率积分会骗人。 |
| 7 | [Heading loop](research-heading-loop-20261102.md) | The loop's effect in simulation, the knee in its gain at 2.0, and what a longer lookahead buys. | 回路在仿真里的效果、增益在 2.0 处的拐点，以及更长前瞻买到什么。 |
| 8 | [Heading hold in the controller](research-heading-hold-controller-20261102.md) | The same loop in Rust: feed-forward, contract validation, and the two things it refuses. | 同一回路在 Rust 控制器里的实现：前馈、契约校验，以及它拒绝的两类输入。 |
| 9 | [Pursuit and the line loop](research-pursuit-line-20261104.md) | Foot odometry, forward kinematics, and the line loop that lost to the IMU by an order of magnitude. | 足端里程计、正解算，以及输给 IMU 一个数量级的直线回路。 |
| 10 | [Heading work, collected](research-heading-pursuit-20261104.md) | The whole heading thread in one place, with the one claim that was withdrawn and why. | 航向这条线的合集，含一处被撤回的结论与撤回原因。 |

Two conventions used throughout: a number without its conditions is not a number, and a
difference below the measurement's resolution is not a difference — see
[`../EXPLORATION.md`](../EXPLORATION.md).

全文坚持两条：没有条件的数字不算数字；低于分辨力的差异不算差异——见
[`../EXPLORATION.md`](../EXPLORATION.md)。