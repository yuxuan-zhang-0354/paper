# 多无人机动态察打生命周期主线实验设计（v2）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-14
- Verification Status: SPEC-REVIEWED; IMPLEMENTATION-UNVERIFIED
- Version Label: dynamic_lifecycle_mainline_v2
- Probability Registry: `preregistration/first_batch.json`
- Registry SHA-256: `314f923560a280221149613fffd7f51eb358150658e11bebf89640d9311cb57e`

日期：2026-07-14  
状态：已完成三路独立规格审查与修订，待用户复核；尚未授权实现或启动 D2  
实验类型：事件驱动、matched-model、配对 Monte Carlo 仿真

## 1. 实验目标、算法定位与主假设

验证以下完整闭环：

\[
\text{隐藏战场}
\rightarrow\text{共享联合 belief}
\rightarrow\text{Dynamic Task Manager}
\rightarrow\text{idle-only Johnson-CBBA}
\rightarrow\text{并行 commit-next}
\rightarrow\text{物理/观测事件}
\rightarrow\text{belief 更新与重规划}.
\]

算法定位为：

> 以 target-local 一步 Bayesian rollout 生成动态原子任务，并以 Johnson-Warped CBBA 作为冻结 epoch 分配器的滚动时域启发式框架。

它不是 POMDP 解法，不是联合最优分解，也不继承动态外层的全局最优或近似保证。

主假设为：相对使用同一任务效用、同一 fleet screening 和同一 Johnson-Warped 分配器、但只在 \(t=0\) 规划一次的冻结路径基线 `B1m`，完整方法 `P` 能提高真实归一化综合效用。该比较检验完整动态闭环的总效应；Recon、BDA 和事件触发等单项机制只由预先指定的消融支持，不由 `P-B1m` 单独归因。

主效应判据在查看动态主线结果前冻结为：

1. `P-B1m` 的等 cell 权重平均配对归一化效用增益至少为 `0.01`；
2. 分层配对 bootstrap 95% CI 下界大于 0；
3. confirmation 的完整性检查及全部 correctness Gates 为 0。

D1 pilot 不构成确认性证据；只有冻结生成分布和独立 manifest 后的未见 D2 scenarios 才能用于主结论。

## 2. 范围边界

本批包含：

- 静态二维目标、同构 UAV、固定速度和直线飞行；
- 多 epoch 随机物理状态、Recon/BDA 观测和 Attack 转移；
- 并行非抢占动作、目标锁、共享 belief 和事件触发；
- 动态 mode screening、固定原子任务和 Johnson-Warped CBBA；
- 连续攻击、可选 BDA、重复侦察和跨 UAV 序贯接续；
- 航程、弹药、截止期硬约束及真实综合效用；
- counter-based common random numbers 与配对比较。

本批不包含：

- 通信延迟、丢包、带宽约束；
- 移动目标、障碍、碰撞规避和返航；
- 多 UAV 同步联合攻击；
- POMDP 精确求解、强化学习或神经网络；
- model mismatch；matched-model 通过后另写规格；
- ranked-mode fallback；它只保留为附录消融，不进入 `P`。

## 3. 冻结参数、状态与信息边界

### 3.1 冻结 matched-model 参数

配置 ID 为 `recon_damage_plus_010_r2_a6_b3`，矩阵约定为“观测为行、真实状态为列”：

\[
V_H=100,\quad V_L=30,\quad
\pi_H=0.40,\quad \pi_L=0.75,
\]

\[
\delta_R=4,\quad\delta_A=2,\quad\delta_B=1.5,
\quad \delta_{\min}=1.5,
\]

\[
\kappa_R=2,\quad\kappa_A=6,\quad\kappa_B=3,
\quad\beta=0.02,
\]

\[
\lambda_d=0.10,\qquad\lambda_m=0.50,
\qquad v=1.
\]

Recon 类别、Recon 毁伤和 BDA 毁伤矩阵分别为：

\[
M_R^C=
\begin{bmatrix}
0.65&0.15\\
0.35&0.85
\end{bmatrix},\quad
M_R^S=
\begin{bmatrix}
0.85&0.15\\
0.15&0.85
\end{bmatrix},
\]

\[
M_B^S=
\begin{bmatrix}
0.92&0.06\\
0.08&0.94
\end{bmatrix}.
\]

其中 \(\kappa_a\) 只表示动作服务成本，不包含飞行或弹药成本；\(\lambda_dL\) 和 \(\lambda_mM\) 分别表示飞行成本与弹药稀缺成本，因此不存在同一成本重复扣除。航程、弹药和截止期的硬约束负责可行性，软成本负责可行方案之间的偏好。

### 3.2 目标与 UAV 状态

目标 \(j\) 的隐藏状态为：

\[
X_j^{true}=(C_j,S_j),\quad
C_j\in\{H,L\},\quad S_j\in\{A,D\}.
\]

规划器只维护公共 belief：

\[
b_j=[b_j^{HA},b_j^{HD},b_j^{LA},b_j^{LD}]^\top
\in\Delta^3.
\]

每架 UAV 维护：当前位置、available/reserved/consumed 弹药、available/reserved/consumed 航程、idle/busy 状态和当前承诺动作。idle UAV 的 `finish_time=None`。busy 动作记录：agent、target、mode、出发位置、目标位置、commit 时刻、service-start 时刻、completion 时刻、travel、预留资源和 0-based 动作序号。

### 3.3 Public 与 private audit 严格分离

`PublicSnapshot` 仅含：当前时刻、公共 belief、目标位置、UAV 可用资源、busy action 的目标/终点/完成时刻、target locks、公开观测包和公开动作 ACK。

`PrivateAuditState/Event` 才可包含：真实类别、真实毁伤状态、随机 draw、Attack 成败、首次摧毁奖励、无效攻击标记、初始/最终 truth digest 和真实效用分解。

policy、Task Manager、CBBA、greedy 与 CEX API 不得接收 private 对象、真实奖励或 RNG。Attack 的 public completion 只给 ACK，并触发确定性 belief prediction；不得公开 hit、miss、destroyed 或 reward。日志可以同时保存 public 与 private 表，但 writer/evaluator 不得把 private 字段回传 policy。

初始真值按目标独立地从各自公共 belief 采样。初始 `D` 是残骸，仍保留真实类别。

## 4. Counter-Based Common Random Numbers

### 4.1 唯一键与生成器

固定：

```text
rng_version      = "sha256-u64-v1"
experiment_id    = "dynamic-lifecycle-mainline-v2"
generator_version= "d1-generator-v1"
```

一个 scenario 的全局标识为：

```text
(experiment_id, generator_version, cell_id, within_cell_seed)
```

哈希输入必须是一个完全扁平的 canonical JSON array：字符串均为 ASCII，整数均为十进制非负整数，UTF-8 编码，`ensure_ascii=true`，separators 固定为 `(',', ':')`，不允许对象、浮点数、嵌套 array 或尾随换行。字段顺序固定为：

```text
[rng_version, experiment_id, generator_version,
 cell_id, within_cell_seed,
 entity_namespace, entity_id,
 event_type, occurrence_index, subdraw_index]
```

其中 `entity_namespace` 只能是 `"scenario"`、`"target"` 或 `"agent"`，从而 target 0 与 agent 0 不会碰撞。`cell_id` 固定为 ASCII 形式，例如 `"N2-M3-Rtight"`。示例：

```text
["sha256-u64-v1","dynamic-lifecycle-mainline-v2","d1-generator-v1",
 "N2-M3-Rtight",7,"target",3,"attack",0,0]
```

规范序列化后的实际字节中没有上述换行或空格。

取 SHA-256 摘要前 8 字节为 big-endian 无符号整数 \(n\)，并映射为：

\[
U=(n+0.5)/2^{64}\in(0,1).
\]

不得使用 Python 内置 `hash()`、全局顺序 RNG 或由 worker 顺序决定的 seed。

### 4.2 Outcome 顺序与 ordinal

- initial truth 的 inverse-CDF 顺序：`HA, HD, LA, LD`；
- Recon 联合观测顺序：`(H,A), (H,D), (L,A), (L,D)`；
- BDA 观测顺序：`A, D`；
- Attack 在真实 `A` 时以 \(U<\pi_C\) 判为成功，真实 `D` 时 draw 不改变状态。

每个 method 内，对每个 `(target, action_type)` 使用独立的 0-based attempted-action ordinal。ordinal 在 commit 时分配并递增；即使真实目标已经 `D`，该次 Attack 仍占用 ordinal。动作不可取消，因此 committed attempt 必然完成。draw 只在 completion 时由 simulator 读取，policy 不得预取。

scenario 生成键同样冻结：agent/target 二维坐标分别使用各自 namespace、`event_type="initial_position"`、`occurrence_index=0`、`subdraw_index=0/1`；initial truth 使用 target namespace、`event_type="initial_truth"`、两个 index 均为 0；Recon/BDA/Attack 使用 target namespace、对应小写 event type、attempt ordinal 和 `subdraw_index=0`。belief archetype 分配是确定性的，不取随机数。

同 scenario 的所有方法共享初始几何、初始 belief、初始 truth 和相同 counter keys。不同策略可能执行不同动作数，或在不同隐藏毁伤状态下调用同一第 \(k\) 个 draw；这仍是合法 CRN coupling，只保证边缘转移核不变，不声称不同方法获得相同观测或相同真值轨迹。

## 5. 观测、Bayes 更新与 Attack 转移

### 5.1 Recon

对 \(o=(o^C,o^S)\)，联合 likelihood 为：

\[
Z_R(o\mid c,s)=M_R^C(o^C\mid c)M_R^S(o^S\mid s).
\]

\[
\eta_R(o\mid b)=\sum_{c,s}Z_R(o\mid c,s)b(c,s),
\]

\[
\mathcal T_R(b,o)(c,s)=
\frac{Z_R(o\mid c,s)b(c,s)}{\eta_R(o\mid b)}.
\]

Recon completion 时根据当前隐藏 \((C,S)\) 一次性采样联合观测并执行一次联合 Bayes 更新；不得拆成两个独立 marginal posterior。若 \(\eta_R=0\)，该观测在模型下不可能发生，立即记 Gate failure，不执行除法。

### 5.2 BDA

BDA 满足：

\[
Z_B(o^S\mid c,s)=M_B^S(o^S\mid s),
\quad o^B\perp C\mid S.
\]

\[
\eta_B(o\mid b)=\sum_{c,s}M_B^S(o\mid s)b(c,s).
\]

后验为：

\[
\mathcal T_B(b,o)(c,s)=
\frac{M_B^S(o\mid s)b(c,s)}
{\sum_{c',s'}M_B^S(o\mid s')b(c',s')}.
\]

BDA 可在首次 Attack 前执行。它没有直接类别通道，但联合 belief 中 \(C,S\) 相关时可以间接改变类别边缘。零 evidence 同样记 Gate failure。

### 5.3 Attack

真实 `D` 吸收。真实 `A` 在 completion 时按类别相关 Bernoulli 转移：

\[
P(D^+\mid H,A,Attack)=\pi_H,
\qquad
P(D^+\mid L,A,Attack)=\pi_L.
\]

public policy 不观察结果，只无条件执行：

\[
T_A b=
\begin{bmatrix}
(1-\pi_H)b^{HA}\\
b^{HD}+\pi_Hb^{HA}\\
(1-\pi_L)b^{LA}\\
b^{LD}+\pi_Lb^{LA}
\end{bmatrix}.
\]

target lock 保证 Attack 旅行和服务期间该目标没有并发动作，因此 completion 时的 prior 唯一。后续可以再次 Attack、Recon、BDA 或 Defer。

### 5.4 通信与 belief 一致性

Public observation/ACK 包即时、可靠、全连通广播，并以 `(scenario_id, target_id, action_type, ordinal)` 去重；每包只融合一次。batch barrier 后，各 UAV 以相同 event order 得到相同公共 belief。本文不研究通信时延、丢包或带宽，也不把 belief 同步本身作为主要创新。

## 6. 动作时间线、资源账本与事件状态机

### 6.1 唯一动作时间线

在公共时刻 \(t\) 将 idle UAV \(i\) 承诺到目标 \(j\) 的动作 \(a\) 时：

\[
t^c=t,\qquad
d_{ij}=\lVert x_i-x_j\rVert,
\]

\[
\widetilde t^s=t^c+d_{ij}/v,\qquad
\widetilde t^f=\widetilde t^s+\delta_a.
\]

实际 scheduler 时刻为 §6.4 的规范值

\[
t^s=\epsilon_t\operatorname{tick}(\widetilde t^s),
\qquad
t^f=\epsilon_t\operatorname{tick}(\widetilde t^f).
\]

只有同时满足

\[
t^f\le T_{max},\quad d_{ij}\le L_i^{avail},
\quad \nu_a\le m_i^{avail}
\]

的动作才可 commit，其中 \(\nu_A=1,\nu_R=\nu_B=0\)。

commit 时原子地：锁定目标；将 \(d_{ij}\) 与 \(\nu_a\) 从 available 转入该动作的 reserved；分配 action ordinal；将 UAV 标为 busy；建立唯一 completion event。commit 不更新 belief 或隐藏目标状态。

service cost 按绝对 \(t^s\) 记账。completion 时：读取 draw；进行 Recon/BDA 观测或 Attack 真值转移；更新 public belief；支付可能的首次摧毁奖励；把 reserved 转为 consumed 而不二次扣减；把 UAV 位置设为 \(x_j\)；标为 idle；解除 target lock。

### 6.2 Commit-Next-Only 与原子提交

除 `B1m` 的冻结 suffix 例外外，每个 planning epoch 中，分配器可以形成多任务 path，但每架 winning idle UAV 只 commit `path[0]`。所有 suffix 立即释放，不锁目标、不预留资源、不进入事件队列。

同一 epoch 最多为每个 target commit 一个动作。Task Manager 本身每 target 最多生成一个 fixed-mode task；B6 和 CEX 也必须显式满足 one-target-one-commit。所有 winners 先整体校验，若出现 target 重复、资源越界或 path 不一致，则整个 commit batch 不作部分写入并记 Gate failure。

### 6.3 Idle-Only 与 target lock

只有当前 idle UAV 参加当轮 Task Manager、screening 和分配。busy UAV 的资源已从 available 中排除，不能竞标、不能成为当前任务 witness，也不在本轮 continuation proxy 中使用。它完成后才以新位置和新 available resources 重新加入。

target 从 commit 到 completion 一直锁定，排除同步联合攻击；completion 后允许同一或另一 UAV 序贯接续。

### 6.4 规范时间轴与每个 method 独立的事件时钟

为避免 binary64 近相等产生多种 batch，所有 public clock times 使用整数 tick。固定：

\[
\epsilon_t=10^{-10},\qquad
\operatorname{tick}(r)=
\operatorname{round}_{half\text{-}even}(r/\epsilon_t).
\]

实现契约固定为 IEEE-754 binary64 距离（Python `math.hypot`），随后以 `Decimal.from_float(r/epsilon_t).to_integral_value(rounding=ROUND_HALF_EVEN)` 产生整数 tick；不得用平台默认近似相等或其他 rounding mode。

commit、预测 service-start、completion、periodic grid 和 \(T_{max}\) 在进入 scheduler 时均转换为整数 tick；公开时间为 `tick * epsilon_t`。硬时限、折扣时刻和 event equality 均使用该规范时间。不同事件只有 tick 完全相同才组成 batch，batch 当前时刻就是该唯一 tick，不再使用 min/max/tolerance 代表值。

每个 `(scenario, method)` 独立运行自己的事件队列；CRN 不要求方法间物理时钟同步。仿真将规划计算视为零物理时间，Johnson run 期间 snapshot 和任务集冻结。

初始化时在 tick 0 运行该方法允许的规划，并立即执行一次 progress/termination 检查：无 busy、无 pending suffix、无正可行任务则在 \(t=0\) 正常终止；存在正可行 pair 但零 commit 则立即记 stall Gate，绝不跳到 \(T_{max}\)。其后重复：

1. 取 `min(next completion, next permitted planning grid, T_max)`；
2. 先收集该整数 tick 的全部 completion；grid、completion 与 \(T_{max}\) 的同刻判断均使用 tick 精确相等；
3. 按 `(target_id, agent_id)` 固定顺序记录，但所有不同目标的物理结果均基于 batch 前各自状态采样；
4. 对每个 completion 依次完成 truth/observation、reward、belief、resource、position、phase、idle 和 unlock 更新；
5. 若当前时刻为 \(T_{max}\)，先完整结算该 batch，再终止且不再规划；
6. 否则按 method clock 至多规划一次并原子 commit；
7. 若仍有 busy completion，禁止因 idle UAV 当前无任务而提前终止。

动作完成时刻恰等于 \(T_{max}\) 合法并必须先结算。任何 action completion 都严格晚于其 commit，因为有限动作集满足 \(\delta_a\ge\delta_{\min}=1.5>0\)。

## 7. 动作效用、动态任务生成与 Johnson 接口

### 7.1 同深度一步 rollout

单次 Attack 的期望新增摧毁收益为：

\[
R_A(b)=V_H\pi_Hb^{HA}+V_L\pi_Lb^{LA}.
\]

共同 terminal Attack/Stop surrogate 为：

\[
A_T(b)=-\kappa_A+e^{-\beta\delta_A}R_A(b),
\qquad V_T(b)=\max\{0,A_T(b)\}.
\]

optimistic 一步效用在动作 service-start 时刻计量：

\[
Q_R^{opt}(b)=-\kappa_R+e^{-\beta\delta_R}
\sum_o\eta_R(o\mid b)V_T(\mathcal T_R(b,o)),
\]

\[
Q_B^{opt}(b)=-\kappa_B+e^{-\beta\delta_B}
\sum_o\eta_B(o\mid b)V_T(\mathcal T_B(b,o)),
\]

\[
Q_A^{opt}(b)=-\kappa_A+e^{-\beta\delta_A}
[R_A(b)+V_T(T_Ab)].
\]

无 continuation 的当前动作值为：

\[
Q_R^0=-\kappa_R,\quad Q_B^0=-\kappa_B,
\quad Q_A^0=-\kappa_A+e^{-\beta\delta_A}R_A(b).
\]

Defer 值恒为 0。

### 7.2 唯一的 target-level ammo-reachability proxy

为保持一个 target 在进入 CBBA 前只有一个 agent-independent fixed utility，主方法沿用第三批已验证的 winner-independent existence proxy，而不采用依赖最终 winner 的精确 continuation gate。

在 epoch \(t\)，只用当前 idle 集 \(\mathcal I_t\) 定义：

\[
g_{ja}(t)=1
\iff
\exists h\in\mathcal I_t:
\begin{cases}
m_h^{avail}\ge \nu_a+1,\\
d_{hj}\le L_h^{avail},\\
t+d_{hj}/v+\delta_a+\delta_A\le T_{max}.
\end{cases}
\]

它表示存在一架 seed-state witness UAV，可以从其当前位置完成当前 mode，并在目标处立即执行一次 terminal Attack。该 witness 不必成为 CBBA 最终 winner；多个目标也可共享同一 proxy witness 和同一未来弹药。这里不做真实预留，故它只是冻结的轻量 continuation proxy，不是可实现 residual value 或资源保证。

此外，terminal surrogate 不计未来 Attack 的 \(\lambda_m\) 或额外执行机机会成本；这些只在动作真正成为当前任务时进入路径评分。这是有意保留并通过消融检验的 myopic optimism。

gate 同时作用于 Recon、BDA 和 Attack 的 continuation：

\[
Q_{ja}^{gate}(t)=Q_a^0(b_j)
+g_{ja}(t)[Q_a^{opt}(b_j)-Q_a^0(b_j)].
\]

`optimistic` 令所有 \(g=1\)，只作消融；`no_continuation` 令所有 \(g=0\)，只作诊断下界。busy UAV 不参与 \(g\)，从而不存在对途中位置或未来 release state 的多重解释。

### 7.3 绝对时间路径评分与硬可行性

令 \(\mathcal J_t\) 为所有 unlocked targets。对 idle UAV \(i\) 的候选路径 \(p_i=(k_1,\ldots,k_q)\)，每个固定任务

\[
k=(j,a,x_j,\delta_a,\nu_a,Q_{ja}^{gate}(t)).
\]

从当前位置和 epoch 时刻递推相对 service-start \(\tau_{ik}^s\)、总飞行距离 \(\ell_i(p_i)\)、弹药和最终 completion。硬可行性为：

\[
F_{i,t}(p_i)=1
\iff
\begin{cases}
t+\tau_{iq}^s+\delta_{a_q}\le T_{max},\\
\ell_i(p_i)\le L_i^{avail},\\
\sum_{k\in p_i}\nu_k\le m_i^{avail},\\
\text{path 内 target 不重复}.
\end{cases}
\]

动态 epoch 的 raw planner score 使用绝对时刻：

\[
S_{i,t}(p_i)=
\sum_{k\in p_i}
e^{-\beta(t+\tau_{ik}^s)}Q_k
-\lambda_d\ell_i(p_i)
-\lambda_m\sum_{k\in p_i}\nu_k.
\]

这避免每次重规划把折扣时钟重置为 0。distance/ammo 软成本按设计不折扣，且 planner 与 evaluator 使用同一口径。

### 7.4 Fleet screening 与固定任务集

空路径单任务 score 为：

\[
\psi_{ija}(t)=
\begin{cases}
S_{i,t}((j,a)),&F_{i,t}((j,a))=1,\\
-\infty,&\text{otherwise}.
\end{cases}
\]

\[
\Psi_{ja}(t)=\max_{i\in\mathcal I_t}\psi_{ija}(t),
\qquad
a_j^{screen}=\arg\max_{a\in\{R,A,B\}}\Psi_{ja}(t).
\]

仅当最大值严格大于 \(\epsilon_s=10^{-12}\) 时生成一个固定任务；否则该 target 本 epoch Defer。三个 executable modes 的 tie 固定为 `Recon, Attack, BDA`，witness tie 取较小 agent ID；Defer 总是在最大 score 不超过 \(\epsilon_s\) 时选择。

screening、所有 \(Q_k\)、eligibility、\(F_{i,t}\)、\(S_{i,t}\) 和 tie-break 在一次 Johnson run 内完全冻结。busy UAV 不进入当前 screening。proxy 资源不会因同轮其他任务 winner 改变，因此 CBBA 接收的是固定 valuation；其代价是已明确承认的 terminal resource optimism。

### 7.5 Johnson-Warped 分配与进展

Johnson 输入唯一为：

\[
(\mathcal I_t,\mathcal K_t,F_{i,t},S_{i,t},
\text{deterministic ties},\text{epoch ID}).
\]

内部 raw bid 为路径插入边际，外部交换 prefix-warped bid；每轮 Bundle Build 完整重构。确定性 round cap 沿用已验证实现：

\[
R_{cap}=\max(100,10|\mathcal I_t|\max(1,|\mathcal K_t|)).
\]

cycle、超过 cap、winner conflict、不可行 path、bundle/path mismatch、warped monotonicity 或 replay mismatch 都是结构 Gate failure；不得用 wall-clock timeout 决定算法状态，也不得 fallback/retry 掩盖。

最终只 commit 每个 winning path 的首任务。正 screened 但未分配的任务本 epoch Defer。若至少一个首任务 commit，episode 继续；若没有 commit 且仍有 busy event，则等待最近 completion；若没有 busy event、仍存在正的单任务可行 pair，却 commit 集为空，则记 `allocation_stall/all_orphan` Gate failure。

`P` 优化的是上述 myopic surrogate，不声称精确最大化 \(E[J_{real}]\)。suffix 只影响当前首任务选择，是 receding-horizon lookahead，不是已承诺价值。

## 8. Defer 与终止

Defer 只表示本 epoch 不生成任务，不永久删除目标。belief、位置、资源或 lock 状态在后续 completion/grid 改变后，可以重新激活。

正常终止只允许发生在完成当前时刻 batch 之后：

1. 当前时刻为 \(T_{max}\)；
2. event queue 为空、无 pending frozen suffix，且不存在正效用可行任务；
3. 对 B5，event queue 为空且下一 grid 不早于 \(T_{max}\)，并且当前没有可在未来 grid 执行的正效用任务；
4. 对 B1m，所有 frozen paths 已执行完毕。

只要 event queue 中仍有 busy completion，就不得用“当前 idle UAV 无可行任务”终止。若 event queue 为空、存在正可行 pair 但没有 commit，属于结构性 no-progress Gate，不是正常结束。

由于 \(\delta_{\min}>0\)、动作不可抢占且 UAV 数有限，每架 UAV 的完成动作数至多为 \(\lfloor T_{max}/\delta_{\min}\rfloor+1\)，物理 completion 数有限；B5 的 grid 数也有限。event-count guard 使用由该上界加 grid 上界得到的确定性值，只检测实现错误。

## 9. 真实效用与记账

真实 evaluator 使用：

\[
J_{real}
=\sum_{j:\,\text{首次真实 }A\to D}
e^{-\beta t_j^D}V_{C_j}
-\sum_{a\in\mathcal A_{completed}}
e^{-\beta t_a^s}\kappa_a
-\lambda_dL_{actual}
-\lambda_mM_{fired}.
\]

规则：

- 初始残骸不产生摧毁奖励；
- 同一 target 只有首次真实 `A→D` 支付一次；
- 攻击真实 `D` 仍扣 service、飞行和弹药成本，但不支付价值；
- action service cost 按绝对 service-start \(t_a^s\) 折扣；
- 摧毁价值按 Attack completion \(t_j^D\) 折扣；
- \(L_{actual}\) 和 \(M_{fired}\) 不折扣，各资源只从 reserved 转 consumed 一次；
- 时间通过绝对折扣和 \(T_{max}\) 硬约束体现，不另加 \(-\lambda_tT\)。

归一化指标为：

\[
\widetilde J=\frac{J_{real}}{\sum_jV_{C_j}}.
\]

分母是 evaluator-only 的 gross scenario value，包含初始残骸的类别价值，因此不是 attainable-value normalization；所有 \(V_C>0\)，分母严格为正。该选择在配对内一致，初始残骸率必须分层报告。

辅助指标：首次摧毁价值、航程、弹药、makespan、动作数、无效攻击、初始残骸攻击、Recon/BDA、连续攻击、跨 UAV 接续、重规划/CBBA 轮数、终止原因、orphan/stall 和最终 joint-state Brier score。

## 10. 方法契约与比较方法

### 10.1 强制共享契约

除下表明确列出的唯一差异外，所有方法共享：scenario、public/private API、初始资源、kernel/duration、Bayes/Attack prediction、非抢占、target lock、资源账本、\(T_{max}\)、CRN、真实 evaluator、event batch 顺序和确定性 tie-break。

| 方法 | 可用 mode / phase | 任务生成 | 分配器 | 规划时钟 | 执行语义 |
|---|---|---|---|---|---|
| P | R/A/B/Defer | gated screening | Johnson-Warped | \(t=0\)+每个 completion batch | commit-next |
| B1m | R/A/B/Defer | \(t=0\) 同 P | Johnson-Warped | 仅 \(t=0\) | frozen paths auto-next |
| B2 | 固定 phase | 当前 phase | Johnson-Warped | \(t=0\)+completion | commit-next |
| B3 | A/Defer | Attack-only | Johnson-Warped | \(t=0\)+completion | commit-next |
| B4 | R/A/Defer | no-BDA screening | Johnson-Warped | \(t=0\)+completion | commit-next |
| B5 | R/A/B/Defer | 同 P | Johnson-Warped | absolute periodic grid | commit-next |
| B6 | R/A/B/Defer | 同 P | nearest-positive greedy | \(t=0\)+completion | one task/agent |
| CEX | R/A/B/Defer | all-mode exact | centralized exact | \(t=0\)+completion | commit-next |

### 10.2 P：完整方法

使用本规格 §§3–8 的完整 gated task manager、event trigger、idle-only Johnson-Warped 和 commit-next。

### 10.3 B1m：One-Shot Matched-Allocator CBBA（主基线）

在 \(t=0\) 使用与 P 完全相同的 public snapshot、gate、fleet screening、\(S_{i,0}\)、Johnson 参数和 ties，冻结各 UAV 的完整路径。此后正常产生物理事件并更新公共 belief，但控制器不读新 belief、不生成任务、不拍卖、不改路径。

每架 UAV 任一时刻只 active/lock/reserve 当前 frozen leg；当前 leg completion 后，在同一时刻不经拍卖原子启动其下一 frozen leg。pending suffix 不锁目标、不提前占资源，但因为全局 frozen assignment target-unique 且该 UAV 不执行其他任务，t=0 的完整路径可行性保证 suffix 可执行。pending suffix 非空时不得触发 no-positive termination；若意外不可行，记 Gate failure。

`P-B1m` 只改变是否利用新信息并动态再生成/再分配任务，不混入 Johnson-vs-standard allocator 差异。它检验完整动态闭环总效应，不隔离单项机制。

### 10.4 B2：Fixed-Order One-Pass Lifecycle CBBA

每 target 初始 phase 为：

```text
Recon -> Attack -> BDA -> TerminalAttack/Done -> Done
```

每个 epoch 只允许当前 phase。Recon/Attack/BDA 完成后无条件推进；BDA 完成后，使用同一 registered terminal Attack/Stop surrogate 和实际 fleet screening，只允许至多一次 TerminalAttack，随后 Done，不循环。若当前 phase 无正可行任务则本 epoch Defer，phase 不推进。它是固定 phase availability heuristic，不是该序列的动态规划最优控制器。

### 10.5 B3、B4、B5

- `B3 Attack-Only`：只允许 Attack/Defer；Attack completion 仍执行无观测 prediction 和事件重规划，可连续攻击。
- `B4 No-BDA`：允许 Recon/Attack/Defer；其余与 P 相同。
- `B5 Periodic`：其余与 P 相同，但规划集合固定为

\[
\mathcal G_\Delta=\{k\Delta:k=0,\ldots,\lfloor T_{max}/\Delta\rfloor\}.
\]

主间隔 \(\Delta=4\)，敏感性 \(\Delta\in\{2,8\}\)，不得挑选最佳间隔。off-grid completion 只完成 truth/belief/resource/lock 更新，idle UAV 原地零成本等待；下一事件为最近 completion、下一 grid 或 \(T_{max}\)。completion 与 grid 同刻时先处理完整 completion batch，再且仅规划一次。

### 10.6 B6：Nearest-Positive Greedy

使用与 P 相同的 screened fixed task set，不运行拍卖。候选 pair 的 marginal 专指基于 public snapshot、同一 \(Q\) 和 \(S_{i,t}\) 的 unwarped \(\Delta S_i\)，不得读取 truth 或 \(J_{real}\)。

从所有 \(\Delta S_i>\epsilon_s\) 的 pair 中按固定全序选择：

```text
(incremental_distance ascending,
 unwarped_public_marginal descending,
 target_id ascending,
 agent_id ascending)
```

选中后删除该 agent 及该 target 的全部 pairs，直到无 pair；每 agent 每 epoch 至多一个任务。它比较的是完整 one-task greedy allocator，不把差异只归因于通信或 consensus。

### 10.7 CEX：Centralized All-Mode Myopic Exact

只在 \(M\le5\) 运行。CEX 使用与 P 相同 public snapshot、target-level gated \(Q_{ja}\)、绝对时间 \(S_{i,t}\)、硬资源和 one-target-one-mode 约束，但绕过 independent fleet screening，联合枚举 mode、agent 和 path：

\[
\max\sum_{i\in\mathcal I_t}S_{i,t}(p_i).
\]

它只 commit 每条 path 首任务，并使用同一动态 event loop。不得读取 truth、未来观测、不同 continuation 或动态最优 value。exact solver 若超过确定性枚举界、出现不一致或 wall-clock 保护超时，记结构失败，不用 incumbent 替代。CEX 是当前 epoch myopic allocation benchmark，不是动态上界；不作 compute-matched 运行时间公平声明。

mode rank 固定为 `Recon=0, Attack=1, BDA=2`。对 agent ID 升序构造 solution key：

```text
(
  ((target_id, mode_rank), ... path of agent 0),
  ((target_id, mode_rank), ... path of agent 1),
  ...
)
```

empty path 为 `()`；objective 差不超过 \(\epsilon_s\) 时选择该 key 字典序最小的解。因此多个等值 path profiles 的 `path[0]` 唯一。完整枚举/DP 的确定性 profile 上界固定为

\[
E_{cap}=M!(1+3|\mathcal I_t|)^M,
\]

solver 必须在该上界内完成所有可行 profile 的等价精确搜索；超过上界、只返回 incumbent 或 tie key 不一致均记 Gate failure。

## 11. D0、D1 与 D2

### 11.1 D0：可执行定向证人

实现计划必须为每项建立固定 fixture、预期 public trace、预期 private audit trace 和断言；D0 通过前不跑随机 pilot：

1. 初始残骸零奖励；
2. 首次新摧毁只支付一次；
3. 攻击失败后连续攻击；
4. Attack ACK 不泄漏 outcome；
5. Recon 联合 Bayes；
6. BDA 可在首次 Attack 前执行；
7. target lock 防重叠；
8. busy UAV 不竞标、不进入 gate；
9. 同时 completion batch；
10. commit-next 释放 suffix；
11. 跨 UAV 序贯接续；
12. horizon/range/ammo 拒绝；
13. periodic 同刻先 completion 后 planning；
14. Defer 后因事件重新激活；
15. no-positive 正常终止；
16. counter replay 与跨方法初始真值共享；
17. busy event 存在时不得提前终止；
18. \(t^f=T_{max}\) 先结算后终止；
19. 正任务但零 commit 触发 allocation-stall Gate；
20. PublicEvent/PrivateAuditEvent 泄漏测试；
21. B1m frozen suffix auto-next 且不误终止；
22. dynamic absolute discount 不重置时钟。

概率/score/reward 公式容差固定为 \(10^{-10}\)，simplex 容差 \(10^{-12}\)，事件 batch 时间容差 \(10^{-10}\)。确定性 replay 的规范序列化字段要求逐字段一致。

### 11.2 D1：快速 pilot

初始 cells 为：

\[
c=(N,M,r),\quad
N\in\{2,3\},\quad M\in\{3,5\},\quad
r\in\{tight,loose\}.
\]

资源层是绑定层而非 ammo 与 horizon/range 的全因子，因此共 8 cells：

| 层 | 每 UAV 弹药 | \(T_{max}\) | 航程预算 |
|---|---:|---:|---:|
| tight | 1 | 16 | 18 |
| loose | 3 | 32 | 40 |

每 cell 20 个独立 scenario IDs；每个 scenario 运行全部适用方法。UAV 与 target 坐标按 counter generator 独立取自连续均匀 `[-6,6]^2`，并先生成一次再复制给方法。

四个 belief archetypes 冻结为：

```text
Recon  = (0.00, 0.42, 0.24, 0.34)
Attack = (1.00, 0.00, 0.00, 0.00)
BDA    = (0.26, 0.66, 0.00, 0.08)
Defer  = (0.00, 0.08, 0.06, 0.86)
```

对 episode 内 target 按 `(target_id + within_cell_seed) mod 4` 循环分配，从而在 seeds 间平衡；真实状态再从 belief 采样。

### 11.3 Effect-blind calibration

D1 最多允许 3 轮 versioned calibration。选择配置前不得解盲任何 method-labeled reward、`P-B1m`、win/loss、rank 或 CI。允许使用的信号仅为：

- D0/Gates、数值稳定性和运行时；
- generator-only 的 archetype/类别/初始残骸覆盖；
- 基于 public \(t=0\) snapshot 的 R/A/B/Defer screening 覆盖与单任务可行率；
- tight/loose 是否同时产生可行与受资源阻断的 public instances。

初始接受门槛为：R/A/B/Defer 在 pooled \(t=0\) target screenings 中均至少占 1%，两个资源层均至少出现一个有正可行任务和一个资源阻断 scenario，且全部 Gates 为 0。这里的“资源阻断 scenario”冻结定义为：存在同一无人机的两个不同目标 screened tasks，它们各自作为 singleton 均可行且得分为正，但按某一顺序组成二任务路径后因累计弹药、航程或 horizon 约束不可行。该定义检验 CBBA bundle/path 的累计资源约束，不要求 loose 层出现几何上不可能的 singleton 阻断。冻结第一轮满足门槛的配置；不得按动态真实效用选择。三轮均不满足则停止并由用户批准新规格，不继续自动调参。每轮原因、完整配置和结果全部保留。

第一轮满足门槛并将 generator/config digest 锁定后，才可解盲 D1 的 method-labeled 结果，用于方差估计和 exploratory 描述；不得再据此改变生成分布、D2 cell 权重或按效应符号/大小选择 D2 cells。D2 样本量可以使用配对差的方差，但不能使用其均值作为选分布依据。

不得调整：belief/Attack/Bayes 公式、policy、reward、\(\lambda\)、主指标、主效应阈值、Gates、baseline 定义或统计 estimand。

### 11.4 D2：Locked Confirmation 边界

D2 尚未授权。启动前必须创建独立只读 manifest，冻结：精确 cells、绑定资源层、生成分布、每 cell 样本数及精度依据、scenario ID 区间、所有 method 矩形、cell 权重、bootstrap seed/quantile convention、secondary contrast family 和失败策略。

候选范围仍为：

\[
N\in\{2,3,4\},\qquad M\in\{3,5,8\},
\]

建议每 cell 从 100 seeds 起；最终样本数由 D1 方差给出精度说明后，在查看任何 D2 输出前冻结。CEX 仅运行 \(M\le5\)。D2 IDs 与 D0/D1/calibration 完全不重叠；D2 结果不得反馈修改同一 confirmation。

## 12. Correctness Gates 与失败策略

任一结构 Gate 失败立即停止扩容并保留工件。

### G-DYN1：概率与信息因果

- belief 在 simplex 内；`D` 吸收；
- Bayes、\(T_A\) 与 registry 一致；
- public snapshot 不含 truth、reward、future draw 或 private digest；
- 保持 public history 不变而改变未观测 truth 时，下一 policy 决策不变；
- Attack outcome 只影响 private truth/reward，不进入 public ACK；
- observation packet 只融合一次。

### G-DYN2：事件、锁与资源

- commit/service/completion 时刻满足 §6；
- completion 时间严格推进，batch 内除外；
- busy UAV 不竞标，locked target 不生成/commit；
- 每 target 无重叠动作；commit batch 原子且 target-unique；
- suffix 不消耗资源，B1m 例外语义符合 §10.3；
- available/reserved/consumed 守恒，ammo/range/horizon 不越界；
- \(T_{max}\)、termination、event guard 和 no-progress 可解释。

### G-DYN3：奖励

- 初始残骸支付 0 次，每 target 新摧毁支付不超过 1 次；
- service、distance、ammo 和 reward 可从 private audit 逐项复算；
- planner 不读取 evaluator 字段；
- normalization denominator 正且仅 evaluator 可见。

### G-DYN4：分配与重放

- 每次 Johnson 的 cycle、round-cap、winner conflict、infeasible path、bundle/path mismatch、warped monotonicity 和 replay mismatch 为 0；
- 正任务、空 event queue、零 commit 必须成为明确 stall Gate；
- 同 seed 同方法规范字段重放一致；
- 跨方法 initial truth 和对应 counter keys 一致；
- 并行与单 worker 结果顺序和数值一致。

### 完整性与失败策略

D2 manifest 必须枚举完整 expected `(scenario_id, method)` 矩形。每个组合必须恰有一条 terminal record。Gate failure、event guard、solver/worker timeout、crash、NaN、duplicate、missing row 或 replay mismatch 均使 confirmation 为 `FAILED/INCOMPLETE`；不得删除、替换 seed 或对成功子集发布确认性 CI。算法/结构失败不得重试；任何 complete-case 数字只能标为 exploratory 并同时报告失败数与原因。

## 13. 统计 estimand 与分析

令 confirmation cell 集为 \(\mathcal C\)，scenario \(e\) 的配对差为：

\[
\Delta_e=\widetilde J_{e,P}-\widetilde J_{e,B1m}.
\]

主 estimand 固定为等 cell 权重：

\[
\tau=\frac1{|\mathcal C|}
\sum_{c\in\mathcal C}E(\Delta_e\mid c),
\]

估计量为各 cell 配对均值的等权平均。10,000 次 bootstrap 在每个 cell 内分别有放回抽取完整 scenario-level method pairs，再按相同 cell 权重聚合；报告 mean、95% percentile CI、median、p05/p95 和 win/tie/loss。tie 容差为 \(10^{-12}\)。bootstrap seed 和 quantile convention 在 D2 manifest 冻结。

主成功条件为：

\[
\widehat\tau\ge0.01,
\qquad CI_{0.025}>0,
\qquad \text{complete matrix and all Gates}=0.
\]

唯一 primary family 为 `P-B1m`。Holm secondary family 冻结为 `P-B2`、`P-B3`、`P-B4`、`P-B5(\Delta=4)` 和 `P-B6`；CEX 的 \(M\le5\) 比较单独报告，不与全规模 family 混合。B5 的 \(\Delta=2,8\) 只作预注册敏感性，不用于选择主间隔。

按 N、M、资源层、初始 action-region、初始残骸率和真实高/低价值比例作次要分层。D1 只报告覆盖、correctness 和 exploratory 描述，不作确认性声明，也不得用于 effect-based distribution selection。

## 14. 输出与并行

输出根目录：`results/dynamic_mainline/`。

```text
d0_witnesses/
d1_pilot/
calibration/run_<id>/
locked_confirmation/        # 仅在另行授权后创建
dynamic_manifest.json
records.csv
public_events.csv
private_audit_events.csv
summary.json
dynamic_verdict.md
```

episode-method 记录至少含：scenario/cell/method、initial/final truth digest（private）、public initial digest、termination/status、event/action/replan counts、最终 belief、效用分解、归一化效用、资源、handoff/continuous-attack/BDA、orphan/stall、Johnson Gates 和 replay audit。

runner 最多 22 workers；worker 输入仅为冻结的 scenario/config/method，返回内存记录，父进程唯一写文件。并行采用标准 `ProcessPoolExecutor`，按 manifest key 排序后写出确定性工件。episode 异常保留为 terminal failure row，不替换 seed 或返回 incumbent。

该 runner 是论文实验代码，不声称提供服务级进程治理、硬超时恢复、heartbeat 监控或安全隔离。public/private 通过固定 CSV schema 物理分文件；其目的为实验审计边界，而非对恶意输入的安全威胁模型。长运行由外层实验调度环境监控。

## 15. 理论声明边界

可以声明：

- belief、Attack、Bayes、资源账本、真实奖励和事件状态机按本规格唯一；
- \(\delta_{\min}>0\)、有限 horizon、非抢占和 no-progress 规则使 episode 物理事件数有限；
- 每个 frozen epoch 调用经过前批验证的 Johnson-Warped CBBA，任务集与 valuation 在该 run 内固定；
- CRN 对 adaptive policies 保持各方法正确的边缘转移核；
- Monte Carlo 结果支持冻结分布和 estimand 上的经验比较。

不能声明：

- 动态外层继承 Johnson 的单 epoch 整体收敛或性能界；
- target-level gate 精确预留未来弹药、执行机或时刻；
- 一步 continuation 等于 POMDP 最优 value；
- independent screening 有一般近似界；
- warped allocation 最大化 raw \(J_{real}\)；
- CEX 是动态全局上界；
- matched-model 结论自动代表模型失配、通信受限或移动目标。

## 16. 独立审查闭环

v1 的三路独立审查均给出 No-Go。v2 已处理其共同阻断项：

| v1 断点 | v2 处理位置 |
|---|---|
| busy action 前提前终止、\(T_{max}\) 同刻歧义 | §§6.4、8 |
| 正 duration 仍可能 Zeno | §§3.1、8 |
| 动态 epoch 折扣重置 | §7.3 |
| commit/travel/service/completion 混淆 | §6.1 |
| resource reservation 重复扣除 | §§3.2、6.1 |
| continuation gate 循环依赖 | §7.2 固定 target-level proxy |
| 概率 kernel 与 Bayes 未闭合 | §§3.1、5 |
| Public/private truth 泄漏 | §3.3 |
| CRN key/ordinal/outcome 顺序不唯一 | §4 |
| B1 与 commit-next 冲突、allocator 混杂 | §§10.1、10.3 |
| B2 terminal BDA 无实际 continuation | §10.4 |
| B5/B6/CEX 语义不唯一 | §§10.5–10.7 |
| all-orphan 无事件死锁 | §§7.5、8 |
| D1 cell 与 effect-based calibration | §§11.2–11.3 |
| estimand/bootstrap/缺失记录不唯一 | §§12–13 |

当前放行边界：用户批准 v2 后，可以进入 D0 与 D1 的实施计划；D2 仍必须等待独立 locked manifest 和再次授权。
