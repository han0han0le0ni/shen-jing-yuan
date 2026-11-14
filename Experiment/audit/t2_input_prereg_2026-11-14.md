# T2c · input 层读出上界 预注册（2026-11-14）

> 运行前落盘。承接 `audit/t2_decode_report_2026-11-14.md`（hidden 上界 0.9314）与
> `audit/t2_fanin_report_2026-11-14.md`（`avg_conn` 64 仍 1.8306，判「架构表达力不足」）。
> **不是 B2 验收**，不改变 B2 的既判 FAIL；阈值零变更。

## 0. 问题

fan-in 扫描证明「加大连接数」救不了 hidden 线性可读性。剩下要区分的是**信息在哪一层丢的**：

- 若 **input 层**线性读出 ≤0.40 → 目标/状态信息**在输入编码里就有**，是 hidden 把它搅乱了
  → 修的是**拓扑**（直通/残差、换 hidden 编码、改 fan-out 结构），不是编码；
- 若 **input 层**也 >0.40 → **输入群体编码本身不够**（分辨率/维度/编码方式）
  → 必须**换编码**，hidden 怎么改都无用。

## 1. 协议（除读出层外与 T2 逐项相同）

- 拓扑：`--preset m2`（input_n=1000 + PROP_N=56 → `input_count=1056`，hidden 3500，2×250 motor），
  `avg_conn=8`（**已按 §6.24 关闭扫描，固定回 8**），`prop_mode=raw`，seed=11，`--disable_reward`；
- 采集：20 episode × 1200 tick，每 10 tick 采 **input 层 1056 维二值发放** + 解析 ω*（2400 样本）；
  新增 host 开关 `--dump-layer input` / `--readout-layer input`（核心新增 `net2_set_ext_readout_src`）；
- 拟合：episode 0–13 训练 ridge（α ∈ {0.01,0.1,1,10}，训练集 3 折 CV），14–19 留出；
- 写回：β（1056×2）经 `--load-readout --readout-layer input` 写回，**权重零改动**；
- 评估：seed=999，30 episode × 1200 tick 闭环，同一套 median distance ratio。

## 2. 判定规则

- **参照**：`avg_conn=8` 的 hidden 上界 = 0.9314（§6.23 已测，不再重跑）；
- **PASS 条件**：input 上界 median **≤0.40**；
- **PASS** → 结论「信息在输入、hidden 破坏」，下一步只动拓扑（不换编码）；
- **FAIL** → 结论「输入编码不足」，下一步换编码（更高维/更密/时间编码），不再动 hidden 扇入；
- 若 input 上界 **≥ hidden 上界**（0.9314）→ 额外记为「读出层信息量不增」，直接指向编码维度/分辨率不足，
  但仍按 ≤0.40 判 PASS/FAIL，不改线。

## 3. 附带诊断（不参与判定）

1. **解析状态上界**：同一批样本上，用 6 维精确状态 `[θ1, θ2, end_x, end_y, target_x, target_y]`
   （target 从 dump CSV 按 episode 对齐）拟合 ridge → 单步 MAE。
   这是本流水线的「解析天花板」：若它也很差，说明**问题在 ω* 的瞬时线性可预测性**，与哪一层读出无关。
2. **input 密度**：1056 维平均发放率，检查是否饱和（>0.6）或死掉（<0.05）。
3. **单步 MAE vs 常数基线**，与 §6.23/§6.24 表同口径，便于横向比对。

## 4. 已知风险（必须写进报告）

- **input 层读出在口径上是「作弊」**：位置编码神经元由 `net2_set_target` 每 tick 写入目标位置，
  读出直接看到目标 → 它衡量的是**信息上界**，不是可用机制。PASS 只证明「信息在输入里」，
  **不证明**任何学习规则能提取它，也不构成 B2 可解的证据。
- `prop_mode=raw` 下 tag2/tag3 的 32 个本体感觉神经元恒静息 → 有效维度 1024，ridge 自动忽略零列。
- 仍只测**线性**读出；非线性上界未做（同一限制见 §6.23 §5）。
- 单 seed、无种子方差；结论只针对该拓扑 + `prop_mode=raw` + stride=10。

## 5. 复现

```
cd e:/shen_jing_yuan/Experiment
powershell -ExecutionPolicy Bypass -File .\build.ps1
py -3 tools/t2_input_upper.py
```
