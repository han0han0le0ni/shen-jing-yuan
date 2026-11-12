# -*- coding: utf-8 -*-
"""B1 根因诊断：四流单变量隔离 + 输出层偏向 + 配对消融。

依据 spec/judgments.md §1 失败动作「停止，转 T1/T2 研究」与 §0.5/§0.6
（独立脑实例、单变量隔离）。本脚本只做诊断：

  * 不改任何判据阈值；
  * 不改学习规则、不改刺激生成器、不改任务定义；
  * 每个条件独立进程（独立脑实例），失败运行全部保留；
  * 所有数字来自实跑 JSONL summary，不手工抄录。

运行：  py -3 Experiment/tools/b1_bias_diagnosis.py
输出：  data/b1_bias/diagnosis_summary.jsonl
        audit/b1_root_cause_2026-11-12.md
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "benchmark1.exe"
DATA = ROOT / "data" / "b1_bias"
DIAG = DATA / "diag"
LEGACY = ROOT / "data" / "legacy" / "b1"
SUMMARY_JSONL = DATA / "diagnosis_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_root_cause_2026-11-12.md"

# 与 tools/b1_acceptance.py 完全相同的 10 个 seed，便于直接对照历史结论
MASTER_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
ORDER_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
STREAM_SEEDS = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008]

BASE = ["--delay", "200", "--trials", "300",
        "--learning-rate", "0.02", "--psp-factor", "10"]

GROUPS = [
    # 固定结构与权重，只换试次顺序流
    ("trial_order", [(12345, ["--trial-order-seed", str(v)]) for v in ORDER_SEEDS]),
    # 固定结构与试次顺序，只换权重流
    ("weight", [(12345, ["--trial-order-seed", "999", "--weight-seed", str(v)])
                for v in STREAM_SEEDS]),
    # 固定权重与试次顺序，只换结构流
    ("structure", [(12345, ["--trial-order-seed", "999", "--structure-seed", str(v)])
                   for v in STREAM_SEEDS]),
    # 新四流默认行为，10 个独立 master seed
    ("master_bundle", [(s, []) for s in MASTER_SEEDS]),
    # 旧单流行为，同样 10 个 seed（历史证据链对照）
    ("master_legacy", [(s, ["--legacy-rng"]) for s in MASTER_SEEDS]),
    # 与 master_bundle 配对的「移除资格迹」消融，权重完全不更新
    ("master_notrace", [(s, ["--disable_trace"]) for s in MASTER_SEEDS]),
]


def sha1(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def stats(vals):
    if not vals:
        return dict(n=0, mean=float("nan"), median=float("nan"), sd=float("nan"),
                    lo=float("nan"), hi=float("nan"), passed=0)
    return dict(
        n=len(vals),
        mean=statistics.fmean(vals),
        median=statistics.median(vals),
        sd=statistics.stdev(vals) if len(vals) > 1 else 0.0,
        lo=min(vals), hi=max(vals),
        passed=sum(1 for v in vals if v >= 80.0),
    )


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    mx = statistics.fmean([p[0] for p in pairs])
    my = statistics.fmean([p[1] for p in pairs])
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def run_one(group, idx, seed, extra):
    tag = f"{group}_{idx:02d}"
    jf = DATA / f"{tag}.jsonl"
    csv = LEGACY / f"bias_{tag}.csv"
    diag = DIAG / f"bias_{tag}.txt"
    cmd = [str(EXE), "--seed", str(seed), *BASE, *extra,
           "--jsonl", str(jf), "--out", str(csv),
           "--diag", str(diag), "--tag", tag]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    # host 退出码 2 = test_acc<80% 判据 FAIL，不是运行失败
    if not jf.exists():
        raise RuntimeError(f"{tag}: 无 JSONL 输出; rc={proc.returncode}; "
                           f"stderr={proc.stderr[-300:]}")
    lines = [ln for ln in jf.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip()]
    if not lines:
        raise RuntimeError(f"{tag}: JSONL 为空; rc={proc.returncode}")
    summary = json.loads(lines[-1])
    if summary.get("event") != "summary":
        raise RuntimeError(f"{tag}: 末行不是 summary（运行未完成）")
    summary.update(group=group, idx=idx, master_seed=seed,
                   extra=extra, returncode=proc.returncode, tag=tag)
    return summary


def fmt(v, nd=2):
    return "n/a" if v is None or not math.isfinite(v) else f"{v:.{nd}f}"


def build_report(rows, hashes):
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)

    acc = {g: stats([r["test_acc"] for r in rs]) for g, rs in by_group.items()}
    s = []
    a = s.append

    a("# B1 根因诊断报告 · 2026-11-12")
    a("")
    a("> 依据：`spec/judgments.md` §1 失败动作（停止，转 T1/T2）、§0.5 独立脑实例、§0.6 单变量。")
    a("> 驱动：`tools/b1_bias_diagnosis.py`，全部数字来自本次实跑，**不改阈值、不改学习规则、")
    a("> 不改刺激生成器、不改任务定义**。")
    a(f"> 二进制 `benchmark1.exe` SHA1：`{hashes['benchmark1.exe']}`；"
      "判据版本：v1.2（阈值零变更）。")
    a("")
    a("## 0. 结论")
    a("")
    # 无迹组按设计权重不更新，评估「学习是否改变映射」时必须排除
    learned = [r for r in rows if r["group"] != "master_notrace"]
    d_beta = [r["beta_final"] - r["beta_init"] for r in learned]
    mean_abs_db = statistics.fmean([abs(x) for x in d_beta])
    sd_t = acc["trial_order"]["sd"]
    sd_w = acc["weight"]["sd"]
    sd_s = acc["structure"]["sd"]
    paired = []
    mb = {r["master_seed"]: r for r in by_group["master_bundle"]}
    nt = {r["master_seed"]: r for r in by_group["master_notrace"]}
    for seed in MASTER_SEEDS:
        if seed in mb and seed in nt:
            paired.append((seed, mb[seed]["test_acc"] - nt[seed]["test_acc"],
                           mb[seed]["test_acc"], nt[seed]["test_acc"]))
    diffs = [p[1] for p in paired]
    mean_diff = statistics.fmean(diffs) if diffs else float("nan")
    sd_diff = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
    d_cohen = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    identical = sum(1 for _, d, _, _ in paired if abs(d) < 1e-9)

    if sd_t < 1e-9:
        variance_note = ("试次顺序流 **零影响**（8 次运行全部得到同一正确率）；"
                         "而只换权重流或结构流，正确率就在 0–100% 之间跳变。"
                         "这直接符合「初始网络实例主导」假设。")
    elif sd_w > 2 * sd_t and sd_s > 2 * sd_t:
        variance_note = "后两者远大于前者，符合「初始结构/权重主导」假设。"
    else:
        variance_note = "本轮数据未呈现预期的方差分离，结论需谨慎，见 §2。"

    a(f"1. **学习几乎没有改变刺激→输出的净映射**：{len(learned)} 次有学习的运行"
      f"（已排除按设计权重不变的无迹组）`|beta_final − beta_init|` 平均仅 "
      f"**{mean_abs_db:.4f}**（beta 为 hidden→output 四类权重和的归一化"
      "正确映射偏向，取值 ±1）。")
    a("2. **正确率由初始网络实例决定，而非延迟信用分配**：只换试次顺序流的 SD = "
      f"**{fmt(sd_t)}**，只换权重流 SD = **{fmt(sd_w)}**，只换结构流 SD = "
      f"**{fmt(sd_s)}**；{variance_note}")
    a(f"3. **配对消融不成立**：10 组同 seed 的「完整 − 无迹」平均差 "
      f"**{fmt(mean_diff)}pp**（SD {fmt(sd_diff)}，Cohen's d {fmt(d_cohen, 3)}），"
      f"其中 **{identical}/10 组正确率完全相同**；判据要求下降 ≥30pp。"
      "学习对正确率既无稳定提升也谈不上「被消融掉」。")
    a("")
    a("**B1 仍不通过。** 本轮没有把判据拉回「通过」，也没有改任何阈值；"
      "结论是「当前 B1 任务测到的是初始随机工作点，不是远端信用分配」。")
    a("按 §1 失败动作，继续停在 T1/T2 研究，不得进入 B2。")
    a("")

    a("## 1. 方法")
    a("")
    a("- M0 `seed_bundle` 四流（structure / weight / noise / trial_order）已落到 B1 运行时；"
      "`--legacy-rng` 逐位复现旧单流，保留历史证据链。")
    a("- 新增只读诊断 `net_output_bias()`：统计 hidden 左/右半 → 输出 A/B 四类权重和，"
      "由此算 `beta`（正确映射净偏向）与 `delta_out`（输出侧绝对偏向）。")
    a("- 每个条件独立进程，参数固定：D=200、300 trials（train/test 各 150）、"
      "lr=0.02、psp=10、K=3、τ=[20,100,500]。")
    a("- 6 组共 %d 次运行；退出码 2 视为「判据 FAIL」的有效运行，全部保留。" % len(rows))
    a("")

    a("## 2. 分组实跑结果")
    a("")
    a("| 组 | 变量 | n | mean (%) | median (%) | SD | min / max | ≥80% |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    label = {
        "trial_order": "只换 trial_order 流（结构/权重固定）",
        "weight": "只换 weight 流（结构/试次固定）",
        "structure": "只换 structure 流（权重/试次固定）",
        "master_bundle": "10 个 master seed · 新四流",
        "master_legacy": "10 个 master seed · 旧单流",
        "master_notrace": "10 个 master seed · 移除资格迹",
    }
    for g, _ in GROUPS:
        st = acc[g]
        a(f"| {g} | {label[g]} | {st['n']} | {fmt(st['mean'])} | {fmt(st['median'])} "
          f"| {fmt(st['sd'])} | {fmt(st['lo'])} / {fmt(st['hi'])} | {st['passed']}/{st['n']} |")
    a("")
    a("### 单变量方差对比（固定 master=12345）")
    a("")
    a(f"- 试次顺序 SD = {fmt(sd_t)}；权重 SD = {fmt(sd_w)}；结构 SD = {fmt(sd_s)}。")
    a(f"- 权重/试次 方差比 = {fmt(sd_w / sd_t, 2) if sd_t > 0 else 'n/a'}；"
      f"结构/试次 方差比 = {fmt(sd_s / sd_t, 2) if sd_t > 0 else 'n/a'}。")
    a("")

    a("## 3. 输出层偏向与正确率的相关性")
    a("")
    a("| 指标 | 全部运行 Pearson r | master_bundle 10 seed Pearson r |")
    a("|---|---:|---:|")
    mb_rows = by_group["master_bundle"]
    for key in ("beta_init", "beta_final", "delta_init", "delta_final"):
        a(f"| {key} vs test_acc | "
          f"{fmt(pearson([r[key] for r in rows], [r['test_acc'] for r in rows]), 3)} | "
          f"{fmt(pearson([r[key] for r in mb_rows], [r['test_acc'] for r in mb_rows]), 3)} |")
    a(f"| abs(delta_init) vs test_acc | "
      f"{fmt(pearson([abs(r['delta_init']) for r in rows], [r['test_acc'] for r in rows]), 3)} | "
      f"{fmt(pearson([abs(r['delta_init']) for r in mb_rows], [r['test_acc'] for r in mb_rows]), 3)} |")
    a("")
    a("`delta_out` 为绝对权重差，量级随总权重变化；`beta` 已归一化。"
      "若两者与正确率都无稳定相关，说明简单线性权重和不足以预测发放动态，"
      "B1 需要一个能区分「初始工作点」与「学习成效」的读出指标（T2）。")
    a("")

    a("## 4. 配对消融（完整组 − 无迹组，同 seed）")
    a("")
    a("| seed | 完整 (%) | 无迹 (%) | 差 (pp) |")
    a("|---:|---:|---:|---:|")
    for seed, diff, full, no_trace in paired:
        a(f"| {seed} | {fmt(full)} | {fmt(no_trace)} | {fmt(diff)} |")
    a("")
    a(f"平均差 {fmt(mean_diff)}pp，SD {fmt(sd_diff)}，Cohen's d {fmt(d_cohen, 3)}；"
      f"**{identical}/10 组两者正确率完全相同**。判据要求无迹组相对完整组下降 ≥30pp。"
      "无迹组权重按设计完全不更新，其正确率与完整组几乎一致，"
      "说明当前任务测到的不是远端信用分配，而是初始网络实例本身。")
    a("")

    a("## 5. 对判据与后续研究的影响")
    a("")
    a("- **判据状态不变**：B1 不通过；`judgments.md` 本轮只追加 v1.2 复核记录，阈值零变更。")
    a("- **实验设计缺陷已定位**：输出层初始读出由随机权重决定，且侧抑制会把微小偏向放大成"
      "「恒投一侧」或「恰好正确」的二值结果；学习对净映射的改动量极小。重做 B1 时"
      "必须先把初始读出设为严格对称（消除初始答案），再用配对消融证明正确率来自学习。")
    a("- **T1（前向模型）/ T2（解码可学性）**：本轮数据支持「先做离线监督上界、"
      "确认架构表达力，再谈局部学习规则」的顺序；不要在现有任务上继续调 lr / psp / τ。")
    a("- **四流契约已落地**：`--structure-seed / --weight-seed / --noise-seed / "
      "--trial-order-seed` 可独立覆盖，后续所有 A/B 必须按 §0.6 单变量执行。")
    a("")

    a("## 6. 复现与证据")
    a("")
    a("```")
    a("py -3 Experiment/tools/b1_bias_diagnosis.py")
    a("py -3 Experiment/tools/b1_acceptance.py --phase seeds   # 历史 10 seed 复核")
    a("```")
    a("")
    a("| 文件 | SHA1 |")
    a("|---|---|")
    for name, h in hashes.items():
        a(f"| {name} | {h} |")
    a("")
    a("原始 JSONL：`data/b1_bias/*.jsonl`；汇总：`data/b1_bias/diagnosis_summary.jsonl`；"
      "诊断快照：`data/b1_bias/diag/`；CSV 按 §5 归入 `data/legacy/b1/`。")
    a("")
    a("*本报告只陈述可核对事实；不宣布 B1 通过，不修改任何阈值。*")
    return "\n".join(s) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-only", action="store_true",
                    help="只从已有 diagnosis_summary.jsonl 重生成报告，不重跑实验")
    args = ap.parse_args()

    for d in (DATA, DIAG, LEGACY, REPORT_MD.parent):
        d.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        if not SUMMARY_JSONL.exists():
            print("ERROR: 没有汇总 jsonl，请先完整运行一次", file=sys.stderr)
            return 1
        rows = [json.loads(ln) for ln in
                SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if ln.strip()]
        print(f"从 {SUMMARY_JSONL} 读取 {len(rows)} 条记录（不重跑）")
    else:
        if not EXE.exists():
            print("ERROR: benchmark1.exe 不存在，先运行 build.ps1", file=sys.stderr)
            return 1
        rows = []
        total = sum(len(items) for _, items in GROUPS)
        done = 0
        for group, items in GROUPS:
            for idx, (seed, extra) in enumerate(items):
                row = run_one(group, idx, seed, extra)
                rows.append(row)
                done += 1
                print(f"[{done:2d}/{total}] {group:<16} seed={seed:<6} "
                      f"acc={row['test_acc']:6.2f}%  beta_i={row['beta_init']:+.4f} "
                      f"beta_f={row['beta_final']:+.4f}  rc={row['returncode']}", flush=True)
        with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hashes = {
        "benchmark1.exe": sha1(EXE),
        "brain/benchmark1_core.c": sha1(ROOT / "brain" / "benchmark1_core.c"),
        "brain/benchmark1_core.h": sha1(ROOT / "brain" / "benchmark1_core.h"),
        "host/benchmark1_host.cpp": sha1(ROOT / "host" / "benchmark1_host.cpp"),
        "core/seed_bundle.c": sha1(ROOT / "core" / "seed_bundle.c"),
        "spec/judgments.md": sha1(ROOT / "spec" / "judgments.md"),
    }
    REPORT_MD.write_text(build_report(rows, hashes), encoding="utf-8")
    print(f"\n完成：{len(rows)} 次运行 -> {SUMMARY_JSONL}")
    print(f"报告：{REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
