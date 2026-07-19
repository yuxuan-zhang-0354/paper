# Ranked-Mode Fallback 设计

日期：2026-07-14  
状态：待用户复核书面规格（总体方案已于 2026-07-14 批准；已完成独立理论复核）  
依赖：第三批 All-Mode Screening 基础结果  

## 1. 触发依据与研究边界

基础 Fleet Screening + Johnson-Warped 算法已经冻结。未见 seeds 的 1620-instance holdout 中：

- Johnson Gate G2 失败数为 0；
- 平均 `Johnson / All-Mode Exact` 为 0.9977；
- task orphan rate 为 7.50%，不同规模约为 5.7%–8.7%。

因此满足预注册的 `task orphan rate > 1%` 条件，可以实现 ranked-mode fallback。该增强只处理固定 planning epoch 内的 task orphan，不引入动态观测、物理状态转移、神经网络或强化学习，也不修改 Johnson/CBBA 内部 bundle construction、winner reduction、consensus 和 warped-bid 规则。

## 2. Ranked Fleet Screening

定义两个用途不同的数值容差：

- \(\epsilon_{screen}=\texttt{mode\_allocation.TOL}=10^{-12}\)，只用于冻结的 screening 正值判断和并列规则；
- \(\epsilon_{score}=10^{-12}\)，只用于最终合法迭代的全局 near-maximum 选择。

对目标 \(j\)、模式 \(a\in\{R,A,B\}\) 和无人机 \(i\)，严格复用基础算法的单任务 screening primitive、可行性判断和 witness value：

\[
\psi_{ija}=S_i((j,a)).
\]

不可行时令 \(\psi_{ija}=-\infty\)，并定义：

\[
\Psi_{ja}=\max_i\psi_{ija}.
\]

每个目标只保留满足 \(\Psi_{ja}>\epsilon_{screen}\) 的模式，按以下固定规则排序：

1. \(\Psi_{ja}\) 降序；
2. 数值并列时 `Recon → Attack → BDA`；
3. witness agent 并列时取较小 agent ID。

目标 \(j\) 的有序候选列表记为：

\[
\mathcal L_j=(a_j^{(0)},a_j^{(1)},\ldots,a_j^{(K_j-1)}),
\qquad 0\le K_j\le3.
\]

基础算法使用首项 \(a_j^{(0)}\)。若列表为空，则该目标从一开始就是 Defer，不进入 fallback 分母。实现必须断言每个首项与冻结 `screen_modes` 的输出完全一致，并断言非空列表的目标集合与原基础正筛选目标集合完全相同；否则视为实现错误并停止实验。

## 3. 单调 fallback 状态

为每个基础正筛选目标维护候选指针：

\[
r_j\in\{0,1,\ldots,K_j\}.
\]

- 当 \(r_j<K_j\) 时，当前任务为 \((j,a_j^{(r_j)})\)；
- 当 \(r_j=K_j\) 时，该目标已穷尽候选，当前为 Defer；
- 指针只能增加，不能回退。

初始状态为所有非空列表的 \(r_j=0\)。每一轮使用当前固定 mode 集合从空 bundle 开始完整运行一次未经修改的 Johnson-Warped CBBA。

若该轮合法收敛，则找出当前 active task 中最终无人持有的目标集合 \(O^{(q)}\)。对所有 \(j\in O^{(q)}\) 同步执行：

\[
r_j\leftarrow r_j+1.
\]

其他目标的指针不变。随后重新生成固定任务集并从空 bundle 重跑 Johnson。

状态机规则为：

1. 合法轮后若当前无 active orphan，则停止；
2. 否则同步推进全部 active orphan；推进到 \(r_j=K_j\) 的目标变为 inactive/Defer；
3. 推进后仍运行下一轮，包括 active set 已为空的情形；该轮必然没有 active orphan并正常停止；
4. 若 Johnson 本轮未合法收敛，则立即停止推进并记录该轮 Gate failure。

若基础轮 \(q=0\) 非法，则不存在 fallback 输出，实例记为失败，非劣保证不适用。若后续轮 \(q>0\) 非法，则丢弃该非法轮，但保留并输出此前全部合法候选；实验仍按照 Gate policy 停止扩容。

因为每个目标最多有三个正候选，且每次继续迭代势函数

\[
P(\mathbf r)=\sum_{j\in\mathcal T_0}r_j
\]

严格增加，所以 Johnson 调用次数满足：

\[
Q\le 1+\sum_jK_j\le1+3M.
\]

这给出有限终止性，但不声称 fallback 获得全局最优 mode assignment。该有限终止性来自外层指针的有限偏序，或 inner Gate failure 后立即停止，不依赖 Johnson 对动态任务集的收敛证明。

## 4. 合法迭代与结果选择

一次迭代只有同时满足以下条件才是合法候选：

- Johnson 状态为 `converged`；
- winner conflict、不可行路径、bundle/path mismatch、warped monotonicity violation 和 replay mismatch 均为零；
- 最终效用由未 warped 的统一路径评分 \(S_i\) 复算。

基础迭代 \(q=0\) 与所有后续合法迭代都保留。令合法候选集合为 \(\mathcal Q_L\)，先计算：

\[
J_{max}=\max_{q\in\mathcal Q_L}J_q,
\qquad
\mathcal Q_{near}=\{q\in\mathcal Q_L:J_q\ge J_{max}-\epsilon_{score}\}.
\]

然后只在 \(\mathcal Q_{near}\) 中按以下顺序选择：

1. 基础正筛选目标中的未分配数量更少；
2. 按基础目标编号顺序构造实际分配向量 `assigned Recon / Attack / BDA / Defer`，按 `Recon → Attack → BDA → Defer` 排序；
3. 再并列时取更早迭代。

由于合法的基础迭代也进入候选集，因此只要基础 Johnson 合法收敛，就有：

\[
J_{fallback}\ge J_{max}-\epsilon_{score}
\ge J_{base}-\epsilon_{score}.
\]

这只是相对于基础算法的非劣保证，不是相对于 All-Mode Exact 的最优性保证。

## 5. Orphan 口径

禁止通过把任务切换为 Defer 来缩小统计分母。固定分母为基础轮中满足 \(\Psi_{ja_j^{(0)}}>\epsilon_{screen}\) 的目标集合：

\[
\mathcal T_0=\{j:K_j>0\}.
\]

令：

\[
\mathcal B=\{j\in\mathcal T_0:\text{基础迭代未以任一 mode 分配 }j\},
\]

\[
\mathcal F=\{j\in\mathcal T_0:\text{最终选中迭代未以任一 mode 分配 }j\}.
\]

则：

\[
r_{base}=|\mathcal B|/|\mathcal T_0|,\qquad
r_{fallback}=|\mathcal F|/|\mathcal T_0|,
\]

\[
\mathcal R=\mathcal B\setminus\mathcal F,\qquad
\mathcal N=\mathcal F\setminus\mathcal B.
\]

其中 \(\mathcal R\) 是 resolved base orphan，\(\mathcal N\) 是 newly unassigned target，不能用 \(|\mathcal B|-|\mathcal F|\) 代替。resolution rate 定义为 \(|\mathcal R|/|\mathcal B|\)；当 \(|\mathcal B|=0\) 时记为 NA。当 \(|\mathcal T_0|=0\) 时，单实例 unresolved rate 记为 NA，语料聚合使用总计数之比。报告：

- base orphan count/rate；
- fallback unresolved count/rate；
- orphan resolution count/rate 与 resolved target IDs；
- newly unassigned count/rate 与 target IDs；
- `selected_mode_switches` 和 `selected_defer_count`，只描述基础解到最终选中解的变化；
- `search_advances` 和 `search_exhausted_targets`，只描述完整搜索轨迹。

基础轮 cycle/timeout 或 Gate 失败时不产生正式 \(\mathcal B/\mathcal F\) 统计。后续尝试轮 Gate 失败时，该轮不产生 orphan 统计，但仍可对此前选中的合法结果报告 \(\mathcal F\)，并单独记录 attempted-run Gate failure。

## 6. 比较指标

基础算法结果不得覆盖。对每个实例配对记录：

- \(J^*_{all}\)、\(J_{base}\)、\(J_{fallback}\)；
- \(J_{fallback}-J_{base}\)；
- 两种算法相对 All-Mode Exact 的 ratio/regret；
- base/fallback unresolved rate；
- fallback 迭代数与 Johnson 总轮数；
- selected solution 的 mode switch，以及完整搜索轨迹的 advance/穷尽次数；
- 全部 Gate 计数。

Fallback 不重新定义第三批的 \(L_{screen},L_{alloc},L_{warp}\)。它作为基础 M4 之后的独立增强层报告，避免把动态 mode 切换混入原有三项损失分解。

## 7. 快速验证顺序

### Tier F0：定向证人

至少覆盖：

1. 基础无 orphan，fallback 不启动；
2. orphan 的第二 mode 可被分配；
3. 所有替代 mode 均不可分配并最终穷尽；
4. fallback 新模式扰动其他目标并产生新 orphan；
5. 多个 orphan 同步推进；
6. 后续迭代效用较低，最终选择基础迭代；
7. 相同分数下选择 unresolved 更少的迭代；
8. 完全确定性 replay。

### Tier F1：既有 holdout 配对复跑

先复用 `N,M={(2,4),(3,4),(3,5)}`、stratified profile 和 seeds 1000–1019。任一 fallback Gate 失败立即停止。

### Tier F2：是否进入 locked confirmation

只有同时满足以下条件才进入 100-seed confirmation：

- fallback Gate failures 为 0；
- 平均效用不低于基础算法（容差 \(10^{-9}\)）；
- unresolved rate 相对 base orphan rate 有可测下降；
- 运行时间仍适合 CPU 并行实验。

若 unresolved 没有下降，则保留负结果并放弃 fallback，不继续增加 mode-aware CBBA 复杂度。

## 8. 理论声明边界

本文可以声明：

- fallback 的候选指针单调，因此外层迭代有限终止；
- 每次 Johnson 调用期间任务集固定；不同外层迭代的任务集可以不同，并且每次均从空 bundle 启动独立 Johnson run；
- 在基础迭代合法时，保留最佳合法迭代使最终 surrogate score 相对基础算法非劣。

本文不能声明：

- fallback 找到全局最优 mode assignment；
- fallback 保留 Johnson 对动态任务集合的单次全局收敛证明；
- fallback 优化真实多 epoch 战场长期回报；
- orphan 降低必然导致真实任务效果提升。

## 9. 规格自审

- 无 TBD/TODO；
- fallback 只在预注册触发后加入；
- mode 排序、推进、终止、合法性和两阶段全局最终选择均唯一确定；
- Defer 不会改变基础 orphan 分母；
- Full-Raw 不参与 fallback；
- Johnson 内核和统一评分函数保持不变；
- calibration 与未见 seed 仍隔离。
