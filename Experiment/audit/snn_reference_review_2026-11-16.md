# 公开 SNN 项目借鉴评估（2026-11-16）

> 目的：在 §6.22–§6.30「四条路径全部失败」之后，不再自己造修法，而是对着**已被公开验证能跑闭环连续控制的 SNN 系统**
> 逐条对照，找出我们**结构上缺的那一环**，并给出可以立刻执行、且不触碰阈值/判据的动作清单。
>
> **本文不是 B2 验收**；B2 保持停止；**阈值零变更**；任何需要改学习规则/判据的动作都标注「需签字」。

## 0. 我们的现状快照（取自代码，非记忆）

| 维度 | 当前配置（`--preset m2`） | 出处 |
|---|---|---|
| 规模/拓扑 | 1000 / 3500 / 2×250，**avg_conn = 8**，`synapses = 115063`（≈0.8% 稠密度） | `host_b2` L149-167、运行日志 |
| 神经元 | LIF 离散：`V_rest −65`、`V_th −55`、`decay 0.95`、`refractory 3`、`PSP 25`；运动层异构 τ/阈/调谐 | `benchmark2_core.h` L40-44 |
| 侧抑制/回响 | `g = 0`（无回响兴奋） | §6.22 |
| 解码 | `decode_mode=3` PDc：`Σ s·pref_omega / Σ s`（不截断 k-WTA） | core L994-1004 |
| 学习信号 | R-STDP + 资格迹 + **单标量奖励**（滑动基线 200） | core L654-702 |
| 误差通道 | `error_gain = 0.0`、`novelty_gain = 0.0`、`cf_lambda = 0.0` —— **核心仍逐 tick 算出 `cf_err1/2`，但没有任何通往权重的路径** | core L474-476、L1258-1262 |
| 任务/指标 | 平面 2 连杆到达，budget 1200 tick，320 ep + 20 ep 评测 | §6.22 |

失败记录（一句话）：`avg_conn` 扫描 → 表达力不足（§6.23）；换输入编码 → 关闭（§6.26）；
直通/残差 → 自锁（§6.28）；非线性解码头 → 有限可读（§6.29）；速率码 → 未达线（§6.30）。

## 1. 公开项目盘点（做法 → 对我们哪条失败路径有效）

| 项目/方法 | 关键做法 | 对我们 |
|---|---|---|
| **Nengo / NEF + PES（hPES）** 自适应控制 | 用**群体编码**表示状态；解码器 `d` 用最小二乘初始化，再用**误差驱动规则**在线更新（`Δd ∝ E·a`，E 为控制误差）；误差小则不再学习 | 直接命中「分布漂移 + 无在线自适应」：我们的读出是**一次离线拟合**（§6.23–§6.30 全部如此），闭环一漂移就废。PES 的观点是「解码器必须被误差持续校正」 |
| **小脑自适应滤波器（Fujita / Dean-Porrill）** | 爬行纤维传递**逐 tick 有符号运动误差**，平行纤维–浦肯野突触按误差方向做 LTD | **我们代码里就是这个**（`cf_err = ω* − ω`，`error_gain` 增益、`tanh(pref_omega/0.15)` 定方向、资格迹做 credit），但 `error_gain = 0` → **从未启用** |
| **PopSAN**（`combra-lab/pop-spiking-deep-rl`，CoRL 2020；Loihi 部署，对比 Jetson TX2 每次推断能耗 −140×） | 状态→**群体编码**→脉冲骨干→**群体解码**为连续动作，外层套 DDPG 用替代梯度训练 | 我们的输入编码 + PDc 解码与它**同构** → 印证 §6.26/§6.29 的结论：编码/解码不是我们的瓶颈 |
| **PopSAN 高维版**（arXiv 2510.10516，2025） | Isaac Gym / PixMC / Franka 上做高维操作，能耗节省至 96.10% | 说明「同一套群体编码 + 梯度训练」可以扩到高维；我们卡住的是训练信号不是编码 |
| **ILC-SAN**（Ding Chen 等，TNNLS 2024，DOI `10.1109/TNNLS.2024.3352653`；SpikingJelly 有可跑教程） | 全脉冲 actor，**层内连接**增强表达，配 TD3 + 替代梯度 | 对应我们 §6.28 的教训：改拓扑必须同时保证「驱动不被摊薄」；他们靠**梯度**调权重而不是随机稀疏初始化 |
| **FORCE / RLS 读出**（Sussillo & Abbott 2009；`braintools.optim.FORCE`） | 对固定储备池的**线性读出**做**在线递归最小二乘**，误差反馈回网络 | 直接对应 §6.26「input 侧失败全部归因于分布漂移」——在线 RLS 正是为分布漂移设计的 |
| **snnTorch / SpikingJelly / Norse** | 替代梯度生态（`surrogate.BP` / ATan / Sigmoid 斜率可调） | 提供「训练输入/隐藏权重」的现成工具，是 T3 级动作 |
| **Lava（Intel 神经形态栈）** | 事件驱动多核部署 | 与我们当前目标（先把机制跑通）无关，只在「能效指标」上将来有用 |
| **序列 RL 的替代梯度斜率研究**（检索摘要：shallow/scheduled slope 使训练与最终性能提升约 2.1×） | 说明 SNN+RL 的瓶颈之一在**梯度设置** | 提醒：若走梯度路线，「斜率/序列长度」是要显式预注册的超参，不是随手调 |

> 检索式证据来自 2026-11-16 的网络检索与教程页（PopSAN GitHub、SpikingJelly ILC-SAN 教程、braintools FORCE 文档等），
> 未逐篇核对原文数值；引用仅用于「做法是否被公开验证」的判断。

## 2. 差距分析（把失败路径逐条对上）

1. **读出族（§6.29/§6.30）不是瓶颈，但「离线」是。**
   公开项目里没有任何一个靠**一次拟合、之后固定**的解码器跑闭环；Nengo/PES 与 FORCE 都把**误差在线灌进解码器**。
   我们的 DAgger（§6.26）本质上是「离线近似在线」，input 侧确实从 0.5637 → 0.2597（PASS），但 hidden 反而更差（1.72）
   ——这与「hidden 无边际信息」（§6.27）一致：**在线化救不了没有信息的特征**，但**只对 input 有效**这一点本身已经证明「分布漂移」是真因之一。
2. **拓扑（§6.28）失败的原因是「随机稀疏 + 无归一化」，不是「直通」本身。**
   公开项目的连接是密集/层内全连，且权重由梯度塑形；我们 `avg_conn=8`、`g=0`、随机初始化。
   §6.23（加大扇入反而更差）与 §6.28（改扇出直接自锁）正是「随机稀疏 + 静态读出」的典型病理。
3. **输入编码（§6.26）与 PDc 解码不是差距**：与 PopSAN 同构，且 §6.29 显示 input 侧还有 0.47→0.40 的余量但无落地路径。
4. **学习信号（从未测）= 真正的空白。**
   已读 `spec/judgments.md` 全文：`error_gain` / `cf_lambda` / 攀爬纤维 **0 次提及**。
   也就是说 §6.22–§6.30 换了读出、编码、拓扑、时序，唯独**没动过「往权重里灌什么信号」**；
   而公开项目无一例外用的是误差/梯度信号，不是「标量奖励 × 资格迹」。

**结论**：我们缺的不是更聪明的读出，而是**一个把误差直接送进突触的通道**——它已经在代码里，默认关着。

## 3. 可借鉴动作清单（按代价/风险排序）

| # | 动作 | 代价 | 是否需要签字 | 说明 |
|---|---|---|---|---|
| **A** | **T2k：打开攀爬纤维误差通道**（`--error-gain > 0`，其余全同 m2） | **零代码改动**（CLI 已支持）；3 seed × 320 ep ≈ 数小时 | 需负责人确认（学习信号变更，虽可 CLI 回退） | 对应 Nengo/PES 与小脑模型；已经写好预注册 `audit/t2k_prereg_2026-11-16.md` |
| **B** | 读出在线学习（FORCE/RLS/PES 风格，写进 `ext_beta` 或核心） | 中等：需实现在线更新 + 迹 | 需签字 | 对应「离线解码器必然输给分布漂移」；先做 A 再决定要不要 B |
| **C** | 输入/隐藏权重用替代梯度 BPTT 训练（snnTorch 移植） | 大：新训练栈、GPU、序列长度/斜率超参 | 需签字（等于换范式） | 公开项目的主流路线；与「不改学习规则」直接冲突，只能作为 T3 |
| **D** | Python 闭环仿真（2 连杆 + 稀疏 LIF + PDc + 在线 RLS） | 1–2 天 | 不需（纯工具） | 把 A/B 的迭代从「90 min/seed」降到分钟级；不改任何判据 |
| **E** | 指标与公开基准对齐：**成功率**（到达/试次）+ 多 seed 方差 + 能耗 | 小 | 不需 | 我们现在只报 median 距离，横向不可比 |

**推荐顺序：A（立刻）→ D（并行）→ 依 A 的结果决定 B/C。**

### 为什么 A 可以立刻做

- 参数链路已验证：`--preset m2 --error-gain 1.0` 解析后核心回显 `v6: error_gain=1.000`（20 ep 探针三档 `g=0/1/3` 均跑通，
  训练段 median 1.009 / 1.007 / 1.013，**因轮数远少于 320 ep，此处不作任何推断**）。
- 语义明确：只在 `NT_MOTOR_OMEGA*` 的输入突触上、且**突触前本 tick 发放**时生效，方向由 `tanh(pref_omega/0.15)` 给出，
  credit 用资格迹 → 与 R-STDP 共用同一套三因子机制，不是外挂。
- 可回退：`error_gain=0` 即回到 §6.22 逐位行为（`--seed 101 --episodes 2` 已确认旧路径未变，见 §6.30 的字节级校验）。

## 4. 明确不做

- 不再扫 `avg_conn`、不再换读出族、不再动 `direct-fan`、不再扫 `rate_tau`、不回 B2 调 lr/shaping/轮数。
- 不同时开 `novelty_gain`（会把「增益」与「惊讶度」混在一起，A 必须单变量）。
- 不因为 A 动了学习信号就顺手改阈值——A 的判定线沿用 `median ≤ 0.40`。

## 5. 参考

- PopSAN：`https://github.com/combra-lab/pop-spiking-deep-rl`；《Deep Reinforcement Learning with Population-Coded SNN for Continuous Control》(CoRL 2020)
- PopSAN 高维版：arXiv 2510.10516《Population-Coded Spiking Neural Networks for High-Dimensional Robotic Control》
- ILC-SAN：Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian，TNNLS 2024，DOI `10.1109/TNNLS.2024.3352653`；教程 `https://spikingjelly.readthedocs.io/zh_CN/latest/tutorials/cn/ilc_san.html`
- Nengo / NEF / PES(hPES)：`https://www.nengo.ai/`（解码器最小二乘初始化 + 误差驱动在线更新）
- FORCE / RLS 读出：Sussillo & Abbott, Neuron 2009；实现参考 `https://braintools.readthedocs.io/apis/generated/braintools.optim.FORCE.html`
- snnTorch `https://github.com/jeshraghian/snntorch`；SpikingJelly `https://github.com/fangwei123456/spikingjelly`；Norse `https://github.com/norse/norse`；Lava `https://github.com/lava-nano/lava`
