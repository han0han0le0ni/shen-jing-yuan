# 历史 CSV 归档口径 · 2026-11-12

> 判据 `spec/judgments.md §5`：「日志只写 JSONL；CSV 停用；`results.csv` 之类旧格式归档到 `data/legacy/`」。

## 现状盘点

全仓 CSV 共 **334** 个（含本次新生成并已归档的文件）：

| 目录 | CSV 数 | 处理状态 |
|---|---|---|
| `data/legacy/b1/` | 125 | 本次新运行，已按 §5 归入归档目录 |
| `data_b2/` | 123 | 历史数据，仍在原目录（未移动） |
| `data/b1_m1/` | 48 | 历史数据，仍在原目录（未移动） |
| `data/multi_seed/` | 30 | 历史数据，仍在原目录（未移动） |
| `data/` | 8 | 历史数据，仍在原目录（未移动） |

## 本轮已做的

- 本次 B1 验收的 124 次运行，CSV 全部直接写入 `data/legacy/b1/`；对外主数据为 `data/b1_acceptance/*.jsonl`（每 trial 一行 + summary）。
- 每个运行写 `*.meta.json`，含源码哈希、二进制哈希、`seed_bundle` 四子种子、全部参数与判据版本；`seed_bundle` 为契约声明值，B1 C 运行时仍是单流 `srand(seed)`，该实现缺口已在验收报告 §6 明确标注。
- 两个新数据目录已加入 `.gitignore`（原始数据不入库，可由命令复现）。

## 历史数据为何暂不移动

现有工具链把路径写死，直接移动会破坏证据链与报告数字：

| 工具 | 硬编码路径 |
|---|---|
| `tools/gen_progress_report.py` | `data_b2/*.csv` |
| `tools/b1_report.py` | `data/b1_m1/sweep_*.jsonl` |
| `audit/scan_data.py` / `inspect_results.py` / `build_report.py` | `data_b2/` |
| `data/multi_seed/`、`data/` 根 | 早期 B1/B2 实验 CSV |

迁移顺序（**本轮不执行**，避免不可逆地改动历史证据）：① 把上述工具的输出目录参数化；② 回归验证报告数字不变；③ 再按原目录名移动到 `data/legacy/<原目录名>/`；④ 更新引用与 `.gitignore`。

## 历史 meta 缺口（未回填、未篡改）

`data/b1_m1/` 的 47 个 `*.csv.meta.json` 有超参与 build 时间，但缺源码哈希、二进制哈希、`seed_bundle` 四子种子、判据版本，且 seed 全为 12345。本次不回填这些字段（回填会伪造当时不可追溯的二进制哈希），只在此登记；新运行的 meta 已完整。

*本清单由 `tools/b1_acceptance.py --phase report` 生成，可重复核对。*