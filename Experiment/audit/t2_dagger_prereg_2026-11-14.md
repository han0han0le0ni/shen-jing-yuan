# T2f · DAgger 式 on-policy 聚合上界 预注册（2026-11-14）

> 运行前落盘。承接：`audit/t2_coverage_report_2026-11-14.md`（0.40 闭环线不可归因于表达力；
> input 开环 93% 可解释却闭环 0.56–0.79）。
> **不是 B2 验收**；B2 既判 FAIL 不变；阈值零变更（0.40 仅作参考线，判定规则见 §2）。

## 0. 要回答的问题

T2c/T2d/T2e 已证明「读出在训练分布内很好、闭环就崩」。本实验直接检验**是否纯粹是分布漂移**：

用 DAgger（Dataset Aggregation）迭代 2 轮——
第 k 轮用**当前读出自己闭环走出来的状态**采集数据，并入训练集重新拟合 ridge。
若读出能自我纠偏，聚合 2 轮后闭环 median 应显著下降；若不降，说明瓶颈是**读出族/表征**，
不是分布。

## 1. 协议

- 起点数据：T2e 的均衡采集（`data_b2/t2bal/dump_{input,hidden}.bin`，`--drive-direct --retarget 60`，
  20 ep × 1200 tick，stride 10，seed 11）；
- 第 k 轮（k=1,2）：
  1. 用**上一轮 β** 闭环采集：`--load-readout beta_k --readout-layer L --dump-hidden dump_L_rk`
     （20 ep × 1200 tick，stride 10，seed 11，学习关闭）；
  2. **聚合**：把 T2e 起点数据与第 1..k 轮数据拼接（同一层），episode 0–13 训练 / 14–19 留出；
  3. 重新拟合 ridge（α ∈ {0.01,0.1,1,10}，训练集 3 折 CV）；
  4. 写回评估：seed=999，30 ep × 1200 tick 闭环，median distance ratio。
- 两层（input 1056 / hidden 3500）各自独立跑。
- 全部产物：`data_b2/t2dag/`。

## 2. 判定规则

- **PASS（分布漂移解释成立）**：input 层在 ≤2 轮内 median **≤0.40** → 表征/编码都不缺，
  瓶颈在**闭环控制**（无积分项/量化/1 tick 延迟）；
- **FAIL（读出族或表征不足）**：2 轮后 input 仍 **>0.40** → 线性读出族不够，进非线性读出（下一实验）；
- **附带读法**：`input 与 hidden 的差距`是否随轮次缩小——若 hidden 也追上来，说明 hidden 缺的只是分布；
  若差距保持，说明 hidden 的信息损失是结构性的（与 §6.25 结论一致）；
- **不作废条件**：若某一轮采集轨迹退化（新 dump 的 θ2 std <0.05），记为「读出把手臂开成静止」，
  照常纳入聚合、照常报告，不得丢弃该轮数据（丢弃会人为制造乐观偏差）。

## 3. 已知风险

- 读出若输出近零指令，采集到的仍是「停住」的状态 → 聚合价值有限，这本身是结论（读出不可用）。
- 聚合后训练集以「读出自诱导分布」为主，可能牺牲均衡覆盖 → 需同时报告各轮覆盖与 MAE/基线比。
- 仍只测线性读出、单 seed（seed 11 采集 / 999 评估，与 T2e 一致）。
- 起点数据是 `--drive-direct` 轨迹（解析控制器），非网络自身轨迹；这是上界的既定前提。

## 4. 复现

```
cd e:/shen_jing_yuan/Experiment
powershell -ExecutionPolicy Bypass -File .\build.ps1
py -3 tools/t2_dagger_upper.py
```
