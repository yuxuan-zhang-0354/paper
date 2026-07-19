# 多无人机联合信念与 CBBA 验证框架设计

日期：2026-07-13  
状态：待用户审查  
目标平台：Windows，24 核 CPU，无 GPU 依赖

## 1. 目标

建立一个可复现的 CPU 仿真与性质验证框架，用可执行证据检验论文 D1.2 候选架构中的数学公式、决策代理、资源约束任务分配和动态闭环行为。

框架必须区分三类结论：

1. 可由解析证明支持的数学性质；
2. 可由有限状态穷举或性质测试证伪、但不能据此一般性证明的算法性质；
3. 必须通过 Monte Carlo 量化的工程性能。

第一实施批次只覆盖公式性质、belief simplex 决策区域和 DMG 截止期反例。只有第一批全部通过后，才实现 CBBA 与闭环 Monte Carlo。

## 2. 非目标

- 不求解完整 POMDP；
- 不训练强化学习或神经网络；
- 不在第一批实现动态图形界面；
- 不以实验替代一般性收敛证明；
- 不声称原始 CBBA 的 50% 近似界适用于资源约束动态问题；
- 不在第一批实现同步联合攻击。

## 3. 冻结的数学模型

### 3.1 目标状态与 belief

单目标隐藏状态按如下顺序编码：

\[
\mathcal X=(HA,HD,LA,LD).
\]

belief 为列向量：

\[
b=[b(HA),b(HD),b(LA),b(LD)]^{\mathsf T}\in\Delta^3.
\]

### 3.2 Attack 转移

每次 Attack 消耗一枚弹药，类别相关单次摧毁概率为 \(\pi_H,\pi_L\)，且主场景满足 \(\pi_H<\pi_L\)。

\[
(T_A b)(c,A)=(1-\pi_c)b(c,A),
\]

\[
(T_A b)(c,D)=b(c,D)+\pi_c b(c,A).
\]

### 3.3 Recon/BDA 更新

Recon 观测核：

\[
Z_R(o^C,o^S\mid c,s)=M_R^C(o^C\mid c)M_R^S(o^S\mid s).
\]

BDA 无直接类别通道：

\[
Z_B(o^S\mid c,s)=M_B^S(o^S\mid s).
\]

任一正预测概率观测使用标准 Bayes 更新；零预测概率观测视为模型下不可发生，不执行除法。

所有混淆矩阵统一采用“行表示观测、列表示真实状态”，每一列和为 1。二元观测顺序分别为 `(H,L)` 与 `(A,D)`。

### 3.4 状态转移奖励

目标价值只在真实的存活到摧毁转移上支付：

\[
g_j(x,A,x')=V_c\mathbf 1\{x=(c,A),x'=(c,D)\}.
\]

单次 Attack 的期望摧毁收益：

\[
R_A(b)=\sum_{c\in\{H,L\}}V_c\pi_c b(c,A).
\]

### 3.5 共同 terminal surrogate

折扣函数使用满足半群性质的指数折扣：

\[
d(t)=\exp(-\beta t).
\]

定义：

\[
A_T(b)=-\kappa_A+d(\delta_A)R_A(b),
\]

\[
V_T(b)=\max\{0,A_T(b)\}.
\]

四动作规划代理：

\[
Q^R(b)=-\kappa_R+d(\delta_R)
\sum_o\eta_R(o\mid b)V_T(\mathcal T_R(b,o)),
\]

\[
Q^B(b)=-\kappa_B+d(\delta_B)
\sum_o\eta_B(o\mid b)V_T(\mathcal T_B(b,o)),
\]

\[
Q^A(b)=-\kappa_A+d(\delta_A)[R_A(b)+V_T(T_A b)],
\]

\[
Q^{Df}(b)=0.
\]

上述 \(Q\) 是目标局部、资源乐观的 planning surrogate，不是全舰队可实现总价值，也不预留 terminal Attack 的弹药、执行机或时域。

## 4. 软件架构

采用纯 Python 包，核心模块相互独立：

```text
src/uav_lifecycle/
  belief.py       # 状态索引、观测预测、Bayes 更新
  attack.py       # Attack 转移、状态转移奖励
  rollout.py      # V_T 与四动作 Q
  simplex.py      # belief simplex 网格生成
  path_score.py   # 时间折扣路径评分与 DMG 反例
  scenarios.py    # 参数和可复现场景定义

experiments/
  run_properties.py
  run_belief_sweep.py
  reproduce_dmg_counterexample.py

tests/
  test_belief.py
  test_attack.py
  test_rollout.py
  test_joint_dependence.py
  test_dmg_counterexample.py

results/
  properties/
  belief_sweep/
  dmg_counterexample/
```

核心函数必须是无全局状态的纯函数。随机实验显式接收 seed；确定性性质测试不得依赖随机执行顺序。

## 5. 第一批实验

### 5.1 概率与转移性质

对手工边界案例和随机 belief 执行：

- Bayes 后验非负且归一；
- Attack 后 belief 非负且归一；
- Attack 保持类别边缘；
- Attack 后总存活概率不增加；
- 初始残骸不产生任务期新增摧毁奖励；
- 两次独立攻击的累计期望收益等于 \(V_c[1-(1-\pi_c)^2]\)；
- 吸收状态确保样本路径上摧毁奖励至多支付一次。

### 5.2 联合 belief 必要性

验证一般条件：令

\[
a_c=P(S=A\mid C=c),
\]

则攻击后 \(C\) 与 \(S'\) 独立当且仅当

\[
(1-\pi_H)a_H=(1-\pi_L)a_L
\]

（假定两类先验概率均为正）。

构造相同 \(P(C)\)、\(P(S)\) 但不同联合分布的 belief，验证其 Attack 价值不同。使用符合任务叙事的 \(\pi_H<\pi_L\) 数值例子验证：BDA 的“已摧毁”观测降低高价值类别后验，“仍存活”观测提高高价值类别后验。

### 5.3 共同 rollout

- 展开两次 Attack 的成本和奖励时间，验证折扣时刻一致；
- 在零传感成本、零时延的限定情形下验证一步信息价值非负；
- 验证 Recon/BDA 观测概率加和为 1；
- 对相同输入重复计算，要求 bitwise 或数值容差内一致。

### 5.4 belief simplex 决策区域

使用步长 0.02 枚举四维单纯形整数网格。每个点记录：

- 四个 \(Q\)；
- 最优动作；
- 次优动作与 margin；
- \(P(H),P(A)\) 和互信息/相关性辅助量。

首批参数不是最终论文标定值，而是预注册的模型验证参数族。固定基础量：

\[
V_H=100,\quad V_L=30,\quad
\pi_H=0.40,\quad \pi_L=0.75,
\]

\[
M_R^C=
\begin{bmatrix}
0.65&0.15\\
0.35&0.85
\end{bmatrix},\quad
M_R^S=
\begin{bmatrix}
0.75&0.25\\
0.25&0.75
\end{bmatrix},
\]

\[
M_B^S=
\begin{bmatrix}
0.92&0.06\\
0.08&0.94
\end{bmatrix}.
\]

基础动作时间与折扣：

\[
\delta_R=4,\quad\delta_A=2,\quad\delta_B=1.5,
\quad\beta=0.02.
\]

成本网格预先固定为：

\[
\kappa_R\in\{2,5,8\},\quad
\kappa_A\in\{6,12,20\},\quad
\kappa_B\in\{1,3,6\}.
\]

另做三个单因素传感器变体：

1. Recon 类别两列正确率各下降 0.10；
2. Recon 毁伤两列正确率各上升 0.10；
3. BDA 毁伤两列正确率各下降 0.10。

改变正确率时，将减少的概率质量转移到同列另一个观测，保持每列归一。首批参数族覆盖：

1. 类别不确定、毁伤基本确定；
2. 类别基本确定、毁伤不确定；
3. 高价值攻击更困难（\(\pi_H<\pi_L\)）；
4. BDA 更快且毁伤识别更准；
5. Recon 具有类别信息但毁伤识别较弱。

验收要求：在上述完整预注册参数族的汇总结果中，Recon、Attack、BDA、Defer 均至少出现一次；若某动作始终为空，报告参数事实并把“动作互不支配”标记为未验证，不在本批次结束后反向修改参数制造所需区域。

### 5.5 DMG 反例

精确复现已经审计的二维场景：

```text
O=(0,0), A=(-7,-7), K=(5,-2), J=(8,-2)
delta_A=1, delta_K=0, delta_J=3
U_A=100, U_K=50, U_J=40
discount base=0.98, hard completion deadline=29
```

要求在 \(10^{-6}\) 容差内复现：

\[
c_J^{raw}(p)=23.318506,
\quad c_J^{raw}(p')=22.519091,
\]

\[
c_J^F(p)=9.877665,
\quad c_J^F(p')=22.519091.
\]

从而同时验证 raw aggregate DMG 和 constrained DMG 违反。

## 6. 第二至第四批边界

第一批通过后依次实施：

1. Johnson full-reconstruction warped Bundle Build 与小规模收敛语料；
2. centralized all-mode 枚举/MILP、fleet screening 和 mode gap；
3. 完整事件触发闭环与论文基线；
4. 大规模 Monte Carlo、统计分析和论文图表。

ranked-mode fallback 不进入第一版主算法；仅当第二批测得 orphan rate 不可忽略时，作为独立增强与消融加入。

## 7. 并行与可复现性

- 默认 worker 数为 `min(22, os.cpu_count() - 2)`；
- 每个 job 是一个独立的 `(method, parameter_set, seed)`；
- 禁止嵌套进程池；
- 设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；
- worker 不直接写共享结果文件，由父进程统一聚合；
- 结果记录代码版本、配置哈希、seed、开始/结束时间和异常；
- pilot 每配置 100 seeds，正式实验根据方差和置信区间再决定 500–2000 seeds；
- 算法比较使用 common random numbers 或按 `(world_seed,target_id,action_type,action_index)` 派生的 counter-based seed。

第一批 belief sweep 是确定性枚举，不需要 22 个 worker 全开；并行只按参数组切分。

## 8. 测试与验收 Gate

### Gate A：公式正确性

所有单元测试、边界测试和性质测试通过。任何失败都阻止进入 CBBA 实现。

### Gate B：联合 belief 证据

攻击诱导相关性、BDA 间接类别更新和相同边缘不同攻击价值的数值证据全部可复现。

### Gate C：动作决策有效性

生成完整决策区域数据；不得通过删除不利参数点制造四动作均出现的假象。

### Gate D：DMG反例复现

固定反例数值在容差内完全匹配，并保存机器可读结果。

### 后续 Gate

- warped CBBA 测试语料无循环或未解释超时；
- 小规模 exact benchmark 报告 mode gap、orphan rate 和 witness mismatch；
- 完整闭环报告摧毁价值、距离、弹药、时长和行为统计；
- 统计比较使用配对差值、95% 置信区间、效应量及必要的多重比较校正。

## 9. 输出格式

每次实验至少输出：

1. `config.json`：完整参数与seed策略；
2. `records.csv` 或 `records.parquet`：逐job原始记录；
3. `summary.json`：验收指标；
4. `run.log`：运行状态和异常；
5. 图表只从保存的原始记录生成，不直接从内存临时结果生成。

第一批固定输出：

- `results/properties/summary.json`；
- `results/belief_sweep/decision_regions.csv`；
- `results/belief_sweep/action_counts.json`；
- `results/dmg_counterexample/reproduction.json`。

## 10. 风险控制

- 若合理参数下BDA区域始终为空，优先检查模型支配关系，不人为调参掩盖；
- 若 rollout 的 resource optimism 导致明显mode gap，将其作为消融并研究简单ammo availability gate；
- 若 Johnson 实现无法忠实复现Algorithm 2，删除收敛继承声明；
- 若并行结果不可重现，停止正式Monte Carlo并先修复随机数与聚合机制；
- 不把有限测试中“未发现循环”写成一般性收敛证明。
