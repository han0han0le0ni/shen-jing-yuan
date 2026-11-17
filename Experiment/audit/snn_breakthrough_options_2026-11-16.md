# 破解方案评估：现有范式内 0.54 已被证为上限，下一步能借什么（2026-11-16）

> 目的：在 §6.22–§6.37 十条路径全部走完（0.54 = 该拓扑+该局部规则+该任务的上限）之后，
> 对着**公开可复现的开源方法与论文**，列出真正有希望突破的路线，并给出「与现有代码的距离 / 代价 / 第一条实验」。
> **本文不是验收**；阈值与判据不动；任何改动仍走预注册。

## 0. 先修正两个我此前说错的事实（读代码核对）

| 我此前说 | 代码实际 | 影响 |
|---|---|---|
| 「网络无回响（g=0）」 | **有** H→H 稀疏递归（`连接2`），每隐藏神经元 `K/2` 条，初始权重 **0.1–0.4** | 「加回递归」不是可行方案，**调储备池耦合强度**才是 |
| 「误差只到运动层」 | 对：误差项条件为 `post.type ∈ {M1,M2}`；H→M 已包含，隐藏层自身输入突触不包括（§6.36 已补开关） | 结论不变 |

其余关键事实：I→H 每输入神经元 20 条（1.0–3.0）、I→M 扇出 12、H→M 扇出 `K`（0.3–0.7）。

## 1. 三个「范式级」候选（按公开证据强度排序）

### A. e-prop（资格迹传播）—— **最贴合我们现状**

- **出处**：Bellec et al., *Nature Communications* 2020《A solution to the learning dilemma for recurrent networks of spiking neurons》；
  后续有 SpiNNaker2 上的在线实现（Frontiers in Neuroscience 2022）、JAX 开源实现（`jaxpike` 的 e-prop 指南）。
- **机制**：把 BPTT 的「时间反向」换成**向前携带的资格迹** + 逐神经元的**学习信号** `L_j`。
  它保留了局部性（O(1) 内存、可在线、可上神经形态芯片），但**修正了隐藏层信用分配**——
  正是 §6.36 用「反馈对齐式 `hid_dir`」失败的那件事。
- **与我们的距离：很近**。我们已经有：per-synapse 资格迹、三因子更新、误差信号、群体解码、
  逐 tick 在线更新。缺的只有两件：① 逐神经元学习信号 `L_j = Σ_k (∂E/∂z_k)·(∂z_k/∂h_j)`（用固定的反馈权重近似）；
  ② 输出层的「伪导数」项（e-prop 用 `∂z/∂h` 的替代函数）。
- **代价**：3–5 天（C 侧实现 + 逐位回归 + 预注册）；单次实验仍是 12 min/arm。
- **首条实验**：`error_gain>0` + e-prop 学习信号（替换 `hid_dir`），K=1，3 seed，判 `median ≤0.40`。

### B. 储备池重整（谱半径 / E-I 平衡 / 内在可塑性）—— **最便宜**

- **出处**：储备池计算与 LSM 的标准做法（Jaeger ESN；liquid state machine 综述；LSM+RL 在 cartpole 上可稳定到 reward>125）。
  核心要求：**谱半径 ρ≈0.9–1.5** 才有「衰落记忆」，否则网络只是逐 tick 的静态投影。
- **我们的现状**：H→H 权重 0.1–0.4、fan-out `K/2`。按 `PSP=25`、阈值间隔 10 mV 粗算，
  单个前突触后电位 2.5–10 mV，**递归驱动与阈值同量级但缺乏归一化**，谱半径未知且大概率 ≪1（驱动弱、记忆短）。
- **代价**：**最小**——加一个 `--recur-gain`（缩放 H→H 初始权重）或 `--recur-norm`（重标定到目标谱半径），
  默认 1.0 逐位不变；12 min/arm 评估。
- **风险**：这是**拓扑初始化改动**，按项目纪律**需签字**（与 §6.28 同级）。
- **首条实验**：`recur_gain ∈ {1, 3, 10}` × K=1+g=1，3 seed。

### C. 替代梯度 BPTT（snnTorch / SpikingJelly / Lava-DL）

- **出处**：PopSAN（CoRL 2020，Loihi 上 −140× 能耗）、ILC-SAN（TNNLS 2024）、Snn4RL 类基准、
  以及「替代梯度斜率」研究（浅斜率/调度斜率可让训练与最终性能提升约 2.1×）。
- **机制**：直接对脉冲做替代梯度反向传播，端到端训练全部权重——绕开所有局部规则的局限。
- **代价**：**最大**：新训练栈（Python/GPU）、序列长度与斜率超参、与 C 侧仿真器的对接。
  但可与 C 侧「共用任务定义」，用 Python 先跑通再迁回。
- **风险**：等于放弃「局部规则」前提；结论不能再与 B1/B2 的判据混谈。

## 2. 两个「局部学习」替代范式（研究性强，成本高）

| 方案 | 出处 | 机制 | 适配度 |
|---|---|---|---|
| **预测编码 SNN（PC）** | Lee, Dora, Mejías, Bohté, Pennartz, *Front. Comput. Neurosci.* 2024（spiking PC + feedforward gist） | 每层用「预测误差」驱动局部更新，无需反向传播 | 中：需要多层结构与每层误差，我们的三层可套，但要重构网络接口 |
| **平衡传播（EP）** | Scellier & Bengio 2017；Lin, Bal, Sengupta 2024（EP 扩展到卷积 SNN） | 用能量函数的两相（free / nudged）差值做局部更新 | 低-中：EP 要求网络收敛到平衡，我们是持续驱动的动态系统，不适配 |

## 3. 与本项目已有结论的对照（避免重复劳动）

| 已试 | 结论 | 与上述方案的关系 |
|---|---|---|
| 非线性/线性解码头（§6.29/§6.30） | hidden 有限可读、不足 | A/B 都是**改权重**，不是改读出，不重复 |
| 误差通道 motor-only（§6.31） | PDc 下有害、WTA 下有效 | A 会替换「误差怎么传到隐藏层」，直接承接 §6.36 的失败 |
| 隐藏层可塑 `hid_dir`（§6.36） | 不成立（R² 更差） | `hid_dir` 是 e-prop 学习信号的**粗糙近似**；A 是它的「正确版」 |
| 密度 8→16/32（§6.37） | 单调更差 | B 改的是**递归耦合**，不是扇入密度，不重复 |

## 4. 建议顺序（按「证据 × 便宜」）

1. **B（储备池重整，1 天）**：先确认「隐藏层有没有时间记忆」。**这是唯一能在不换范式的前提下，
   用一个参数决定成败的杠杆**，而且若记忆本来就极短，A 的成功率也会被严重限制——所以 B 是 A 的前置诊断。
2. **A（e-prop，3–5 天）**：若 B 显示「有记忆但读不出/学不会」，A 直接补上信用分配。
3. **C（BPTT）**：若 A 仍卡在 0.54，说明瓶颈在范式而非实现，再上 C。
4. 不建议：PC / EP（成本高、适配差）；再调解码/密度/误差增益（已证到头）。

## 5. 参考链接

- e-prop 原文与实现：`https://www.nature.com/articles/s41467-020-17236-y`；SpiNNaker2 实现：`https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1018006/full`；JAX 指南：`https://github.com/abdurrezzak/jaxpike`
- 储备池 / LSM：`https://grokipedia.com/page/Liquid_state_machine`（综述性，含 LSM+RL 控制案例）
- 预测编码 SNN：`https://lacuna.tiptreesystems.com/work/predictive-coding-with-spiking-neurons-and-feedforward-gist-signaling/`
- 平衡传播扩展：`https://scixplorer.org/abs/2024arXiv240502546L/abstract`
- PopSAN：`https://github.com/combra-lab/pop-spiking-deep-rl`；ILC-SAN：DOI `10.1109/TNNLS.2024.3352653`
- Nengo/NEF+PES：`https://www.nengo.ai/`；FORCE：`https://braintools.readthedocs.io/apis/generated/braintools.optim.FORCE.html`
