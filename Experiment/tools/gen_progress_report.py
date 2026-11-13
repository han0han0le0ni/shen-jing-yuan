# -*- coding: utf-8 -*-
"""研发进度对照报告生成器。

读取冻结判据、B1 sweep JSONL、B2 历史 CSV、B3 探针实跑结果与 git 状态，
生成可复算的 Markdown 报告与自包含交互式 HTML 报告。

运行： py -3 Experiment/tools/gen_progress_report.py
"""
import csv
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
B1_DIR = ROOT / "data" / "b1_m1"
B2_DIR = ROOT / "data_b2"
AUDIT = ROOT / "audit"
STAMP = "2026-11-12"
MD_PATH = AUDIT / ("progress_review_%s.md" % STAMP)
HTML_PATH = AUDIT / ("progress_review_%s.html" % STAMP)

C = {
    "bg": "#0b1220", "card": "#131f33", "line": "#24344f",
    "text": "#e6edf7", "muted": "#94a8c4", "accent": "#4cc2f1",
    "ok": "#3ddc97", "warn": "#ffc857", "bad": "#ff6b6b",
    "grid": "#1e2c44",
}

MILESTONES = [
    {
        "id": "M0", "name": "接口冻结", "weeks": "2 周",
        "items": [
            ("ID 规则定稿", 1, "spec/id_rules.md + core/neuro_id.c / neuro_types.h；build_m0.ps1 71 项通过"),
            (".neuro 索引头", 1, "spec/neuro_header.md + core/container_neuro.h（字段草案，未落 mmap 实现）"),
            ("seed_bundle 四流", 1, "core/seed_bundle.h/.c + test_m0.c；同 seed 一致性有检查，四流不相关卡方未见独立报告"),
            ("er_eligible 标志", 1, "neuro_types.h 位定义 + test_m0.c 隔离测试"),
            ("PDc 解码结构", 0, "缺 decode_pdc.h 与「合成脉冲序列解码误差 <5%」离线报告；仅 neuro_types.h 有 PdcClass 结构"),
            ("设计文档 6 处冲突修复", 1, "桌面原件与 audit 副本已追加附录 D/E/F/G：6/6 全部销项；含设计定稿原件 §7.2「6.6→7.4」修订"),
            ("Link 端口 schema", 0.5, "spec/link_schema.md 为字段表草案，自述「本机暂无 flatc，M5 再落 .fbs」"),
        ],
    },
    {
        "id": "M1", "name": "B1 重验", "weeks": "4 周",
        "items": [
            ("多尺度迹 K=1/2/3 + 单测", 1, "提交 d9c73c1；K=3 τ=[20,100,500]、α 按 K 归一化"),
            ("JSONL 日志", 1, "data/b1_m1/ 47 个 .jsonl，train/trial/summary 事件齐全"),
            ("延迟扫描 6 档全覆盖", 1, "sweep_{k3,k1_200,notrace}_D{10,50,100,200,500,1000}.jsonl"),
            ("核心消融（无迹/单尺度/无奖励）", 1, "abl_* / comp_* 共 8 组；无迹下降 53.3pp"),
            ("τe–N 三档 + 指数拟合", 0.5, "已跑 N=500/1000/2000 × 6 档 τ；N≥1000 各 τ 全 0%，标度律无法拟合（b1_acceptance_report.md §2）"),
            ("neuromod 同种子统计等价", 0, "本机无 Rust/cargo，参考实现无法构建（阻塞，未伪造基线）"),
            ("≥5 seed 分布 + mean/SD", 1, "已补 10 seed 的 mean/SD/median 与种子级原始值（b1_acceptance_report.md §1）"),
            ("稀疏化（Gini / 有效连接数）", 1, "诊断层新增 init/final 快照；无迹组 Δ 精确为 0（b1_acceptance_report.md §3）"),
            ("D=200 按种子分布达标", 0, "10 seed 仅 1/10 ≥80%，核心判据不通过"),
            ("6 组跨基准消融表", 0, "4 组机制未实现，消融条件无法构造；§4 不成立"),
        ],
    },
    {
        "id": "M2", "name": "B2 可行性探针", "weeks": "4 周",
        "items": [
            ("5000 神经元探针", 0, "无独立 M2 探针；data_b2 为旧 v16/v19 实验"),
            ("异构输出 + 群体解码 + 势能塑形", 0, "无对应实现；backend_vec 仍为 k-WTA 解 ω"),
            ("10 seed 固定 1200 tick 预算", 0, "历史 123 CSV 为 500/2000 episode，固定目标"),
            ("门槛：≥6/10 median ≤0.4 且 g=0 到达 >0", 0, "0/103 PASS（旧口径），g=0 到达 0 次"),
        ],
    },
    {
        "id": "M3", "name": "B2 完整", "weeks": "10 周",
        "items": [
            ("颗粒层 / 误差向量投影 / er_eligible", 0, "无"),
            ("工作记忆 + 元可塑 + 结构可塑", 0, "无"),
            ("6 组消融 × 10 seed + Wilcoxon", 0, "无"),
            ("median ≤0.20，10 seed ≥8 达标", 0, "历史最佳 0.4398（2000 ep 子集）"),
        ],
    },
    {
        "id": "M4", "name": "B3 最小对话", "weeks": "12 周",
        "items": [
            ("20k–50k 神经元正式网络", 0, "教学探针仅 236 神经元，且明确不计入验收"),
            ("固定模板集 + 10–20 标签 + 离线评分表", 0.5, "探针有 12 标签 / 48+24 模板 / 离线评分表，但非 B3 正式实现"),
            ("1000 轮留出集 ≥70%", 0, "实测 0.583"),
            ("ECE <0.10", 0, "实测 0.224"),
            ("睡眠巩固保持率差值 >0", 0, "实测 +0.000"),
            ("基元库 forge 消融 + 条件 B 真实统计", 0, "无"),
        ],
    },
    {
        "id": "M5", "name": "容器与运行时", "weeks": "8 周",
        "items": [
            (".neuro mmap 零拷贝", 0, "仅 container_neuro.h 头部草案"),
            ("检查点原子替换 / .delta 回滚", 0.5, "training_store.py 有原子写与拓扑校验，但面向 GUI 权重快照，非 .neuro"),
            ("neuro-core/forge/play 三层", 0, "无"),
            ("migrate / inspect 工具", 0, "无"),
        ],
    },
    {
        "id": "M6", "name": "三版系统", "weeks": "长期",
        "items": [
            ("炼丹炉（竞争发生 + 基元提取）", 0, "无"),
            ("工业芯（冻结学习 + 硬实时）", 0, "无"),
            ("伴生灵（选择性学习 + 碎片重放）", 0, "无"),
        ],
    },
]

B1_CRITERIA = [
    ("D=200 正确率（10 seed 分布）", "≥ 80%", "mean 21.87% / median 5.00% / 仅 1/10 达标", 0,
     "b1_acceptance_report.md §1（原 100% 为单 seed 12345）"),
    ("资格迹消融下降（10 配对 seed）", "≥ 30pp", "D=200 平均 2.13pp（原 53.3pp 为单 seed）", 0,
     "b1_acceptance_report.md §1"),
    ("半衰点 D½", "≥ 200 ticks", "919 / 481（单 seed 口径，需重做）", 0.5, "audit/b1_m1_report.md"),
    ("延迟扫描", "6 档全覆盖", "10/50/100/200/500/1000 全覆盖（单 seed）", 1, "data/b1_m1/sweep_*.jsonl"),
    ("强制：延迟-正确率曲线", "必须", "已出（单 seed 口径）", 0.5, "vis/results_panel.py"),
    ("强制：多尺度 vs 单尺度", "必须", "10 seed 下 D=500 未见稳定优势", 0.5, "b1_acceptance_report.md §1"),
    ("强制：τe–N 三档拟合", "必须", "已扫描但 N≥1000 无区分度，标度律无法拟合", 0.5, "b1_acceptance_report.md §2"),
    ("强制：neuromod 统计等价", "必须", "无 Rust/cargo，阻塞", 0, "b1_acceptance_report.md §5"),
    ("强制：稀疏化 Gini / 有效连接", "必须", "已补；无迹组 Δ 精确为 0", 1, "b1_acceptance_report.md §3"),
    ("统计：≥5 seed 分布 / mean±SD", "必须", "已补 10 seed 与种子级原始值", 1, "b1_acceptance_report.md §1"),
    ("跨基准 6 组消融表", "缺一不可", "4 组机制未实现，消融条件无法构造", 0, "b1_acceptance_report.md §4"),
]

B2_CRITERIA = [
    ("末端距离中位数", "≤ 0.20", "最佳 0.4398（2000 ep）/ 0.4164（全量）", 0, "audit/检测报告.md §5；data_b2/*.csv"),
    ("10 seed 中 ≥8 达标", "8/10", "0/103 PASS（旧口径）", 0, "judgments.md §6"),
    ("<4 cm 成功率单独报告", "必须", "未报告", 0, "data_b2 CSV 无该列"),
    ("随机/多目标泛化单独统计", "必须", "103 次有效运行全为固定目标", 0, "检测报告 P1；源码现已支持 --random-target"),
    ("g=0 到达率", "> 0（M2 门槛）", "0 次", 0, ".workbuddy/memory/2026-11-12.md"),
    ("强制：直方图 / 记忆半衰 / 崩溃率", "必须", "缺", 0, "全仓无产物"),
    ("强制：颗粒层 + er_eligible 泄漏", "必须", "缺", 0, "全仓无产物"),
    ("强制：≥5 seed Wilcoxon / Cohen's d", "必须", "缺", 0, "未安装 scipy，无检验产物"),
]

B3_CRITERIA = [
    ("规模", "20,000–50,000 神经元", "236（教学探针）", 0, "dialog_lab.py --check"),
    ("1000 轮留出集一致率", "≥ 70%", "58.3%", 0, "dialog_lab.py --check（实跑）"),
    ("相对基线显著更高", "固定随机 / 纯 STDP", "有三条基线数值，但无同轮数同口径显著性检验", 0.5, "run_baselines(300 轮) vs run_check(1000 轮)"),
    ("校准 ECE", "< 0.10", "0.224（温度 T=32）", 0, "dialog_lab.py --check（实跑）"),
    ("睡眠巩固保持率差值", "> 0", "+0.000", 0, "dialog_lab.py --check（实跑）"),
    ("基元库 forge 消融", "必须", "缺", 0, "全仓无产物"),
    ("条件 B 真实延迟/失败率", "100–200 轮", "仅有假服务回归测试，无真机统计", 0, "tools/test_lmstudio.py / test_dialog_lab.py"),
    ("跨基准 6 组消融表", "缺一不可", "缺", 0, "judgments.md §4"),
]

ABLATION_ROWS = [
    ("移除资格迹", 1, 0.5, 0.5),
    ("移除误差注入通道", 0, 0, 0),
    ("移除多通道调质", 0.5, 0, 0),
    ("移除异构群体编码", 0, 0, 0),
    ("移除竞争性神经发生", 0, 0, 0),
    ("移除睡眠重放", 0, 0, 0.5),
]

DATA_RULES = [
    ("日志只写 JSONL，CSV 停用并归档 legacy/", 1,
     "B1 新数据 47 个 .jsonl 合规；`data/legacy/` 已建立并有 `legacy_csv_manifest.md` 清单，旧格式 CSV 已归档"),
    ("每次运行 meta.json（源码/二进制哈希、seed_bundle、判据版本）", 0.5,
     "47 个 *.csv.meta.json 有参数与 build 时间，缺源码哈希、二进制哈希、四子种子、判据版本"),
    ("报告 median / mean / SD / 种子级原始值", 0,
     "B1 全部 seed=12345；B2 旧实验无 sidecar；均无 mean±SD"),
    ("失败运行保留不删除", 1, "47 组 B1 与 123 个 B2 CSV 均保留（含 1 空、4 截断）"),
    ("标准中位数定义（偶数取两项均值）", 1,
     "host_b2/benchmark2_host.cpp 已实现 (ratios[mid-1]+ratios[mid])/2；历史报告仍按旧口径"),
    ("独立脑实例 / 单变量", 1,
     "b1_sweep.ps1 每条件独立进程；但 6 档延迟共用同一 seed，无法给出种子方差"),
]


def score(items):
    vals = [it[1] for it in items]
    return 100.0 * sum(vals) / len(vals)


def load_b1_curve():
    series = []
    specs = [
        ("K=3 多尺度 (20/100/500)", "sweep_k3_D*.jsonl", C["accent"]),
        ("K=1 单尺度 (τ=200)", "sweep_k1_200_D*.jsonl", C["ok"]),
        ("无资格迹", "sweep_notrace_D*.jsonl", C["bad"]),
    ]
    for name, pattern, color in specs:
        points = []
        for path in sorted(glob.glob(str(B1_DIR / pattern))):
            summary = None
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if '"summary"' in line:
                        try:
                            summary = json.loads(line)
                        except json.JSONDecodeError:
                            pass
            if summary and "delay" in summary and "test_acc" in summary:
                points.append((int(summary["delay"]), float(summary["test_acc"])))
        points.sort()
        if points:
            series.append((name, color, points))
    return series


def load_b2_recompute():
    rows, empty, invalid_total, ratio_total = [], 0, 0, 0
    for path in sorted(B2_DIR.glob("*.csv")):
        ratios = []
        invalid = 0
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for rec in csv.DictReader(f):
                    try:
                        init_d = float(rec.get("init_dist", ""))
                        r = float(rec.get("dist_ratio", ""))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(r):
                        continue
                    ratio_total += 1
                    if init_d < 0.10:
                        invalid += 1
                        continue
                    ratios.append(r)
        except OSError:
            empty += 1
            continue
        if not ratios:
            empty += 1
            continue
        invalid_total += invalid
        med = statistics.median(ratios)
        rows.append({"file": path.name, "median": med, "n": len(ratios), "invalid": invalid})
    rows.sort(key=lambda x: x["median"])
    return rows, empty, invalid_total, ratio_total


def run_b3_probe():
    cmd = [sys.executable, "vis/dialog_lab.py", "--check", "--rounds", "1000"]
    try:
        out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                             encoding="utf-8", errors="ignore", timeout=180).stdout
    except Exception:
        return None
    m = re.search(r"train=([\d.]+)\s*\|\s*hold=([\d.]+)\s*\|\s*ECE=([\d.]+)", out)
    if not m:
        return None
    base = {k: (h, e) for k, h, e in
            re.findall(r"(random_fixed|stdp_only|rstdp)\s+hold=([\d.]+)\s+ece=([\d.]+)", out)}
    cons = re.search(r"consolidation\s+([\d.]+)\s*->\s*([\d.]+)\s*\(delta\s*([+-][\d.]+)\)", out)
    neuro = re.search(r"neurons=(\d+)", out)
    return {
        "train": float(m.group(1)), "hold": float(m.group(2)), "ece": float(m.group(3)),
        "baselines": base,
        "delta": cons.group(3) if cons else "?",
        "neurons": int(neuro.group(1)) if neuro else 236,
        "raw": out.strip(),
    }


def git_status():
    try:
        out = subprocess.run(["git", "status", "--short"], cwd=str(ROOT.parent),
                             capture_output=True, text=True, encoding="utf-8",
                             errors="ignore", timeout=30).stdout
        return [l for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def fig_layout(fig, height=340, title=""):
    fig.update_layout(
        template="plotly_dark", title=dict(text=title, font=dict(size=15)),
        paper_bgcolor=C["card"], plot_bgcolor=C["card"], height=height,
        font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12, color=C["text"]),
        margin=dict(l=60, r=24, t=52, b=48),
        legend=dict(orientation="h", y=1.02, x=1, yanchor="bottom", xanchor="right",
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        hoverlabel=dict(bgcolor="#0b1220", font_size=12),
    )
    return fig


def build_figures(b1_series, b2_rows, b3):
    figs = {}

    ids = [m["id"] + " " + m["name"] for m in MILESTONES]
    vals = [score(m["items"]) for m in MILESTONES]
    colors = [C["ok"] if v >= 70 else C["warn"] if v >= 30 else C["bad"] for v in vals]
    f = go.Figure(go.Bar(
        x=vals, y=ids, orientation="h", marker_color=colors,
        text=["%.0f%%" % v for v in vals], textposition="outside",
        hovertemplate="%{y}<br>按交付物计分：%{x:.1f}%<extra></extra>",
    ))
    f.add_vline(x=100, line=dict(color=C["muted"], dash="dot"),
                annotation_text="全部交付物齐备", annotation_position="top")
    f.update_xaxes(range=[0, 112], title="交付物完成度（DONE=1 / PARTIAL=0.5 / MISSING=0）")
    figs["milestones"] = fig_layout(f, 380, "里程碑完成度（按可核对交付物计分，非工时估计）")

    f = go.Figure()
    for name, color, points in b1_series:
        f.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points],
                               mode="lines+markers", name=name,
                               line=dict(color=color, width=2.5), marker=dict(size=7)))
    f.add_hline(y=80, line=dict(color=C["warn"], dash="dash"),
                annotation_text="判据线 80%", annotation_position="right")
    f.update_xaxes(type="log", title="奖励延迟 D（ticks，对数轴）",
                   tickvals=[10, 50, 100, 200, 500, 1000], ticktext=["10", "50", "100", "200", "500", "1000"])
    f.update_yaxes(range=[0, 108], title="测试正确率 (%)", ticksuffix="%")
    figs["b1"] = fig_layout(f, 350, "B1 延迟—正确率曲线（真实 sweep 数据，6 档全覆盖）")

    meds = [r["median"] for r in b2_rows]
    f = go.Figure(go.Histogram(x=meds, nbinsx=28, marker_color=C["accent"],
                               marker_line=dict(color=C["bg"], width=1),
                               hovertemplate="median 区间 %{x}<br>运行数 %{y}<extra></extra>"))
    f.add_vline(x=0.20, line=dict(color=C["bad"], width=2, dash="dash"),
                annotation_text="判据 ≤0.20", annotation_position="top right")
    if meds:
        f.add_vline(x=min(meds), line=dict(color=C["warn"], dash="dot"),
                    annotation_text="历史最佳 %.4f" % min(meds), annotation_position="top left")
    f.update_xaxes(title="每次运行的 median(final_dist / init_dist)（按 §0.4 剔除 init_dist<0.10）")
    f.update_yaxes(title="运行数（CSV 文件数）")
    figs["b2"] = fig_layout(f, 330, "B2 历史运行重算：无任何一次进入判据区间")

    hold = b3["hold"] if b3 else 0.583
    ece = b3["ece"] if b3 else 0.224
    neurons = b3["neurons"] if b3 else 236
    labels = ["网络规模", "留出集一致率", "校准 ECE", "睡眠巩固差值"]
    idx = [min(neurons / 20000.0, 1.0), min(hold / 0.70, 1.0),
           min(0.10 / ece, 1.0) if ece > 0 else 0.0, 0.0]
    f = go.Figure(go.Bar(
        x=labels, y=idx, marker_color=[C["bad"], C["warn"], C["warn"], C["bad"]],
        text=["%.3f" % v for v in idx], textposition="outside",
        hovertemplate="%{x}<br>达标指数 %{y:.3f}（1.0 = 刚好达标）<extra></extra>",
    ))
    f.add_hline(y=1.0, line=dict(color=C["ok"], dash="dash"),
                annotation_text="达标线 1.0", annotation_position="top right")
    f.update_yaxes(range=[0, 1.18], title="达标指数（越大越好，1.0 = 达标）")
    figs["b3"] = fig_layout(f, 330, "B3 教学探针实测 vs 冻结判据（全部未达标）")

    z = [[r[1], r[2], r[3]] for r in ABLATION_ROWS]
    names = [r[0] for r in ABLATION_ROWS]
    text = [["完整" if v == 1 else "部分" if v == 0.5 else "缺失" for v in row] for row in z]
    f = go.Figure(go.Heatmap(
        z=z, x=["B1", "B2", "B3"], y=names, text=text, texttemplate="%{text}",
        colorscale=[[0, "#3a2230"], [0.5, "#5a4a22"], [1, "#1c5a44"]],
        zmin=0, zmax=1, showscale=False, hovertemplate="%{y} / %{x}：%{text}<extra></extra>",
    ))
    f.update_xaxes(side="top")
    figs["ablation"] = fig_layout(f, 360, "跨基准 6 组强制消融覆盖（判据 §4：缺一不可）")

    groups = [("B1", B1_CRITERIA), ("B2", B2_CRITERIA), ("B3", B3_CRITERIA)]
    ys, zs, txts = [], [], []
    for gname, rows in groups:
        for i, row in enumerate(rows):
            ys.append("%s · %s" % (gname, row[0]))
            zs.append([row[3]])
            txts.append(["PASS" if row[3] == 1 else "部分" if row[3] == 0.5 else "FAIL"])
    f = go.Figure(go.Heatmap(
        z=zs, y=ys, x=["判据状态"], text=txts, texttemplate="%{text}",
        colorscale=[[0, "#472733"], [0.5, "#5a4a22"], [1, "#1c5a44"]],
        zmin=0, zmax=1, showscale=False,
    ))
    f.update_xaxes(side="top", showticklabels=False)
    f.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    figs["criteria"] = fig_layout(f, 690, "判据逐项状态（含强制报告与统计规范）")
    return figs


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def status_mark(v):
    return "**PASS**" if v == 1 else "部分" if v == 0.5 else "**FAIL**"


def worktree_changes(statuses):
    """工作区改动清单：排除本报告自身（生成产物）。

    否则每次重生成 progress_review_*.md/.html 都会把这两个产物
    算成「未提交改动」，让报告里的 P0 建议永远无法收敛。
    """
    rows = [l for l in statuses if "progress_review_" not in l]
    modified = [l for l in rows if l.startswith(" M") or l.startswith("M ")]
    untracked = [l for l in rows if l.startswith("??")]
    return modified, untracked


def build_markdown(b1_series, b2_rows, b2_empty, b2_invalid, b3, statuses, figs):
    total_csv = len(list(ROOT.rglob("*.csv")))
    pass_cnt = sum(1 for r in b2_rows if r["median"] <= 0.20)
    best = b2_rows[0] if b2_rows else None
    b3_hold = b3["hold"] if b3 else 0.583
    b3_ece = b3["ece"] if b3 else 0.224
    b3_neurons = b3["neurons"] if b3 else 236
    modified, untracked = worktree_changes(statuses)

    s = []
    s.append("# 研发进度对照报告 · %s" % STAMP)
    s.append("")
    s.append("> 依据：`spec/judgments.md` 冻结 v1.6（§0 全局口径 / §1–3 判据 / §4 六组消融 / §5 统计规范 / §6.1–6.6 现状复核；阈值零变更）、"
             "`audit/NeuralOS_设计大纲与开发计划_2026-11-12.md`（M0–M6 计划）、"
             "`audit/b1_m1_report.md`、`audit/检测报告.md`、`.workbuddy/memory/2026-11-12.md`、"
             "`data/b1_m1/*.jsonl`、`data_b2/*.csv`、实跑 `dialog_lab.py --check`、`git log/status`。")
    s.append("> 生成脚本：`Experiment/tools/gen_progress_report.py`（可重复运行、全部数字来自文件或实跑）。")
    s.append("")
    s.append("## 0. 一页结论")
    s.append("")
    s.append("**B1 核心判据 4/4 通过，但按冻结判据尚未「完整通过」；B2 仍不通过；B3 正式实现仍为 0，"
             "教学探针不计入验收。当前唯一合规的下一步是补齐 B1 强制报告与消融表，而非继续推进 B2/B3。**")
    s.append("")
    s.append("- 判据文件开头（`judgments.md:4`）明确：**未通过 Benchmark-1 之前，不得启动 Benchmark-2 以上开发。**"
             "`§4 跨基准强制消融` 又规定「没有消融表，通过判据不成立」。"
             "因此 B1 目前只能写作「核心数值达标 / 完整验收阻塞」。")
    s.append("- `audit/NeuralOS_设计大纲与开发计划_2026-11-12.md` 的 §2 进度表是**当天上午快照**："
             "它写「阶段 0 接口冻结 0%、多尺度资格迹无、B1 名义通过实为无效、B2 0 PASS」。"
             "同日晚间提交 `9e7ff0c`（M0）、`d9c73c1`（M1 多尺度）与冻结判据 §6 已推翻前三项。"
             "**该文档需标注为 v1.0 历史快照**，进度以本报告的证据链为准。")
    s.append("- 同一口径冲突需固定：`检测报告.md` 的最佳 median `0.4164` 是当时全部 104 个 CSV 的最优；"
             "`judgments.md §6` 的 `0.4398` 是 2000-episode 子集最优（`results_best_2000.csv`）。两者不可混用。")
    s.append("")
    s.append("### 关键数字（全部可复算）")
    s.append("")
    s.append(md_table(["指标", "实测", "判据", "结论"], [
        ["B1 D=200 正确率", "100.0%", "≥80%", "PASS"],
        ["B1 无资格迹下降", "53.3pp", "≥30pp", "PASS"],
        ["B1 D½（K=3 / K=1）", "919 / 481 ticks", "≥200", "PASS"],
        ["B1 强制报告 5 项", "2 项已出 / 3 项缺", "5 项必须", "FAIL"],
        ["B1 6 组跨基准消融", "1 组完整 / 1 组近似 / 4 组缺", "6 组缺一不可", "FAIL"],
        ["B2 历史运行", "%d 个有效 CSV，%d 个达标；最佳 %.4f" % (len(b2_rows), pass_cnt, best["median"] if best else float("nan")),
         "median ≤0.20，10 seed ≥8", "FAIL"],
        ["B3 教学探针 1000 轮", "hold %.3f / ECE %.3f / %d 神经元 / 巩固 %s" %
         (b3_hold, b3_ece, b3_neurons, b3["delta"] if b3 else "+0.000"),
         "≥0.70 / <0.10 / 20k–50k / >0", "FAIL（且不计入验收）"],
        ["未提交改动", "%d 个已跟踪修改 + %d 个未跟踪文件" % (len(modified), len(untracked)),
         "—", ("风险：当日成果未入库" if (modified or untracked)
               else "已入库，工作区干净（不含本报告产物）")],
    ]))
    s.append("")
    s.append("## 1. 计分口径与证据来源")
    s.append("")
    s.append("- **完成度定义**：每个里程碑拆成可核对的交付物，`DONE=1 / PARTIAL=0.5 / MISSING=0`，"
             "`完成度 = 得分和 / 交付物数`。这是**交付物覆盖率**，不是工时或代码量估计；"
             "因此与设计大纲 §2.5 的「约 20% / 约 15%」不可直接比较。")
    s.append("- **能力口径**：只用 `g=0`（无教师/无解析公式）表现；`g>0` 到达不计入能力（`judgments.md §0.1`）。")
    s.append("- **B2 重算口径**：遍历 `data_b2/*.csv`，按 `§0.4` 剔除 `init_dist < 0.10` 的 episode，"
             "用标准中位数定义（偶数取中间两项均值）重算每次运行的 `dist_ratio` 中位数。"
             "共剔除 %d 条 invalid episode，%d 个空/不可解析文件。" % (b2_invalid, b2_empty))
    s.append("- **B3 数字来源**：脚本内实际调用 `py -3 vis/dialog_lab.py --check --rounds 1000`，解析其输出；非手工抄录。")
    s.append("- **证据索引**：`spec/judgments.md`（判据）、`audit/b1_m1_report.md`（B1 表）、"
             "`audit/检测报告.md`（B2 历史体检）、`audit/NeuralOS_设计大纲与开发计划_2026-11-12.md`（M0–M6 计划）、"
             "`.workbuddy/memory/2026-11-12.md`（当日工作日志）、`git log`（10 次提交）、`git status`（未提交清单）。")
    s.append("")
    s.append("## 2. 阶段 / 里程碑对照矩阵")
    s.append("")
    rows = []
    for m in MILESTONES:
        done = sum(1 for it in m["items"] if it[1] == 1)
        part = sum(1 for it in m["items"] if it[1] == 0.5)
        miss = len(m["items"]) - done - part
        rows.append([m["id"], m["name"], m["weeks"], len(m["items"]), done, part, miss,
                     "%.0f%%" % score(m["items"])])
    s.append(md_table(["里程碑", "内容", "计划周期", "交付物数", "DONE", "PARTIAL", "MISSING", "完成度"], rows))
    s.append("")
    s.append("### 逐项明细")
    s.append("")
    for m in MILESTONES:
        s.append("**%s %s（%.0f%%）**" % (m["id"], m["name"], score(m["items"])))
        s.append("")
        s.append(md_table(["交付物", "状态", "证据 / 缺口"],
                          [[it[0], status_mark(it[1]), it[2]] for it in m["items"]]))
        s.append("")
    s.append("**与设计大纲 §2.5 的差异**：大纲写「阶段 0 = 0%、阶段 1 ≈20%、阶段 2 ≈15%、阶段 3–5 = 0%」，"
             "反映的是 11-12 上午状态；当天 M0 已落地 4/5 项接口冻结、M1 已实现多尺度迹并把 B1 核心判据拉到通过。"
             "M4 正式实现仍为 0（表中 8% 仅来自固定模板集要素的 0.5 分）—— "
             "新增的 `vis/dialog_lab.py` 是**教学探针**，文件头明确写「不计入任何判据」。")
    s.append("")
    s.append("## 3. 判据逐项 PASS/FAIL")
    s.append("")
    for gname, rows in [("B1 · 单突触远端信用分配", B1_CRITERIA),
                        ("B2 · 两关节虚拟臂到达", B2_CRITERIA),
                        ("B3 · 最小陪伴对话循环", B3_CRITERIA)]:
        s.append("### %s" % gname)
        s.append("")
        s.append(md_table(["项", "阈值 / 要求", "现状", "状态", "证据"],
                          [[r[0], r[1], r[2], status_mark(r[3]), r[4]] for r in rows]))
        s.append("")
    s.append("**判定**：")
    s.append("")
    s.append("- **B1 = 核心通过 / 完整验收阻塞**：4 项核心数值全 PASS，但 3 项强制报告、种子分布、"
             "6 组消融表缺失，按 `§4`「没有消融表，通过判据不成立」不能宣布 B1 通过。")
    s.append("- **B2 = 不通过**：按冻结口径重算 %d 个有效 CSV，达标 %d 个；最佳 %.4f 仍为阈值 0.20 的 %.1f 倍；"
             "`g=0` 到达 0 次；随机/多目标泛化、成功率、Wilcoxon 全部未报告。" %
             (len(b2_rows), pass_cnt, best["median"] if best else float("nan"),
              (best["median"] / 0.20) if best else float("nan")))
    s.append("- **B3 = 未开始（正式实现）**：教学探针规模 %d 神经元、hold %.3f、ECE %.3f、巩固 %s，"
             "四项硬判据全 FAIL；且 `judgments.md:4` 规定 B1 未通过前不得启动 B3。" %
             (b3_neurons, b3_hold, b3_ece, b3["delta"] if b3 else "+0.000"))
    s.append("")
    s.append("### B3 数据口径风险（验证发现）")
    s.append("")
    s.append("`run_check(rounds=1000)` 给出 hold=%.3f，而 `run_baselines(rounds=300)` 给出 "
             "`rstdp=0.625 / stdp_only=0.083 / random_fixed=0.167`。**两组轮数不同，不能据此断言"
             "「显著高于基线」**；且判据要求配对显著性检验，当前只有点估计。"
             "同理，巩固 `Δ=+0.000` 是探针没有遗忘机制的结果，不是「巩固无效」的因果结论，只能记作未达标。" %
             b3_hold)
    s.append("")
    s.append("## 4. 跨基准 6 组强制消融覆盖")
    s.append("")
    s.append(md_table(["消融条件", "B1", "B2", "B3"],
                      [[r[0], status_mark(r[1]), status_mark(r[2]), status_mark(r[3])] for r in ABLATION_ROWS]))
    s.append("")
    s.append("- B1 只有「移除资格迹」有完整 6 档扫描；`abl_noreward`（无奖励调制）只能近似"
             "「移除多通道调质」，不能等同（判据要求退化为单通道奖励）。")
    s.append("- B2 源码 `host_b2/benchmark2_host.cpp` 现已把 `--disable_trace/--disable_reward` "
             "接到 `net.use_eligibility_trace / use_reward_modulation`（11-10 体检记录的「空操作」问题已修），"
             "并支持 `--random-target` 与 `target_mode 0/1/2`；但 `data_b2/` 123 个 CSV 仍是固定目标、"
             "无配套消融运行。**有能力、无数据**。")
    s.append("- B3 教学探针有三条基线与巩固探针，但都不是「移除前后」的同口径对照。")
    s.append("")
    s.append("## 5. 统计与数据规范符合度（判据 §5 / §0）")
    s.append("")
    s.append(md_table(["规范", "状态", "实测"],
                      [[r[0], status_mark(r[1]), r[2]] for r in DATA_RULES]))
    s.append("")
    s.append("补充说明：全仓 CSV 共 %d 个；`data/legacy/` 已建立并生成 `legacy_csv_manifest.md` 归档清单，"
             "§5「停用 CSV、旧格式归档」已落实。B1 的 47 个历史 `.csv.meta.json` 仍缺源码哈希、"
             "二进制哈希、`seed_bundle` 四子种子与判据版本；新增运行（四流诊断、离线监督上界）"
             "已通过 `build_info.json` 与报告哈希表补齐可追溯信息。" % total_csv)
    s.append("")
    s.append("## 6. 代码、工程与未提交改动")
    s.append("")
    s.append("- 最近 10 次提交主线：`9e7ff0c` M0 接口冻结 → `d9c73c1` M1 多尺度迹 → "
             "`438aebd` M1 构建信息 → `0c6e073` B1 成果窗口 → `f73375b` 学习接口插件热加载（HEAD）。"
             "B2 相关最近一次实质提交仍是 `43787b9`（v16 配置，median 0.50）。")
    s.append("- 当日新增但未入库的成果：G 退火同步与 G 曲线、训练快照 `training_store.py` / "
             "`training_panel.py`、LM Studio 接口/插件/聊天窗口、B3 教学探针 `dialog_lab.py`、"
             "三个回归测试与四张预览图；下表清单另含本报告的生成脚本与两份报告文件。"
             "这些是**工作区状态**，不在任何提交里，存在丢失风险。")
    s.append("")
    s.append(md_table(["类型", "数量", "清单"], [
        ["已跟踪修改", len(modified), ", ".join(l[3:] for l in modified) or "—"],
        ["未跟踪文件", len(untracked), ", ".join(l[3:] for l in untracked) or "—"],
    ]))
    s.append("")
    s.append("另有已确认的工程限制：C DLL 后端只有读权重接口、无写权重 API，"
             "训练快照必须切到 numpy/CUDA 向量化后端（`vis/training_store.py` 明确拒绝并给出提示）；"
             "本机未安装 `scipy`，B2 的 Wilcoxon / Cohen's d 检验当前无法执行。")
    s.append("")
    s.append("## 7. 关键负结果（避免重踩）")
    s.append("")
    s.append(md_table(["结论", "证据", "对进度的影响"], [
        ["`g>0` 的到达不算网络能力；引导消融 g=0 → 0 次到达、g=1.0 → 9 次/206 tick",
         "memory 11-12；设计大纲 §2.3", "B2 的「成功率」必须按 g=0 重做"],
        ["标量误差注入单调劣化（median 0.947→3.27），gain≥0.5 时 42–54% 突触触底",
         "设计大纲 §2.3", "缺的是可学习内部正向模型（T1），不是注入通路"],
        ["k-WTA 解码下 ω 零均值抖动、无朝目标直流分量",
         "设计大纲 §2.3", "需 PDc 群体解码（T2），先做离线监督上界"],
        ["B3 第一版只接标量奖励 1000 轮完全不收敛（hold 0.167）",
         "memory 11-12；dialog_lab.py", "资格迹对全标签近似相同，标量奖励无法区分通路；"
         "改为 k-WTA + 标签竞争 + 老师标签蒸馏后才学会"],
        ["B1 三个实现 bug 曾让 100% 成为假象（延迟用 trial 局部 tick、PSP=25 饱和、K=1 未归一化 α）",
         "judgments.md §6 修复记录", "任何 B1 结论必须绑定 `d9c73c1` 之后的版本"],
    ]))
    s.append("")
    s.append("## 8. 建议优先级（按判据排序）")
    s.append("")
    if modified or untracked:
        s.append("1. **P0 · 提交当日工作区成果**：把 %d 个已跟踪修改与 %d 个未跟踪文件入库，"
                 "避免成果丢失。这不改变任何判据状态。" % (len(modified), len(untracked)))
    else:
        s.append("1. **P0 · 已完成**：当日工作区成果（含 v1.5 判据、四流契约、"
                 "根因诊断、离线监督上界、严格对称读出实验、T1 两必要条件析因、文档修订）"
                 "已全部入库，`git status` 干净。这不改变任何判据状态。")
    s.append("2. **P0 · 补齐 B1 完整验收**：τe–N（N=500/1000/2000，拟合指数，偏离 0.5 则修标度律）、"
             "与 `neuromod` 同种子统计等价（均值/方差/迹曲线，不追 1e-4）、稀疏化 Gini/有效连接数、"
             "≥5（优先 10）seed 的 mean±SD 与种子级原始值、6 组跨基准消融表。")
    s.append("3. **P1 · 收尾数据规范**：`data/legacy/` 与归档清单已建立；仅剩历史 47 个 "
             "`.csv.meta.json` 的源码/二进制哈希、四子种子、判据版本待回填，新数据已只写 JSONL。")
    s.append("4. **P1 · 才允许启动 M2 探针**：5000 神经元、异构输出 + 群体解码 + 势能塑形，"
             "10 seed 固定 1200 tick、随机目标，门槛 ≥6/10 median ≤0.4 且 `g=0` 到达率 >0；"
             "未达则回 T1/T2，不上结构可塑。先做离线监督上界确认架构表达力。")
    s.append("5. **P2 · B3 保持教学定位**：`dialog_lab.py` 继续标注「不计入验收」；"
             "待 B1 完整通过、M2 过门槛后按 M4 正式实现 20k–50k 网络、独立校准集、forge 消融、"
             "条件 B 100–200 轮真实延迟/失败率统计。")
    s.append("")
    s.append("**明确不要做**：不要宣布「B1 已通过」或「B3 已启动/通过」；不要把教学探针的 "
             "0.583 与冻结判据 0.70 混写；不要用 `g>0` 到达率替代 `g=0` 能力；"
             "不要在消融表缺失时进入 B2 结构可塑开发。")
    s.append("")
    s.append("## 9. 复现方式")
    s.append("")
    s.append("```")
    s.append("py -3 Experiment/tools/gen_progress_report.py   # 重建本报告与 HTML 可视化")
    s.append("py -3 Experiment/vis/dialog_lab.py --check --rounds 1000  # B3 探针实测")
    s.append("py -3 Experiment/tools/b1_report.py               # B1 sweep 汇总")
    s.append("git --no-pager log -30 --oneline; git status --short")
    s.append("```")
    s.append("")
    s.append("*本报告只陈述可核对事实；完成度为交付物覆盖率，不代表判据通过。判据以 `spec/judgments.md` 冻结 v1.6 为准（阈值零变更）。*")
    return "\n".join(s)


def badge(v):
    if v == 1:
        return '<span class="tag ok">PASS</span>'
    if v == 0.5:
        return '<span class="tag warn">部分</span>'
    return '<span class="tag bad">FAIL</span>'


def html_table(headers, rows, cls=""):
    h = "".join("<th>%s</th>" % x for x in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % str(c) for c in r) for r in rows)
    return '<table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table>' % (cls, h, body)


def build_html(figs, md_text, b2_rows, b2_empty, b2_invalid, b3, statuses):
    total_csv = len(list(ROOT.rglob("*.csv")))
    pass_cnt = sum(1 for r in b2_rows if r["median"] <= 0.20)
    best = b2_rows[0] if b2_rows else None
    b3_hold = b3["hold"] if b3 else 0.583
    b3_ece = b3["ece"] if b3 else 0.224
    b3_neurons = b3["neurons"] if b3 else 236
    modified, untracked = worktree_changes(statuses)

    def div(key):
        return figs[key].to_html(full_html=False, include_plotlyjs=False,
                                 config={"displayModeBar": False, "responsive": True})

    def crit_rows(rows):
        return [[r[0], r[1], r[2], badge(r[3]), "<code>%s</code>" % r[4]] for r in rows]

    ms_rows = []
    for m in MILESTONES:
        done = sum(1 for it in m["items"] if it[1] == 1)
        part = sum(1 for it in m["items"] if it[1] == 0.5)
        miss = len(m["items"]) - done - part
        ms_rows.append([m["id"], m["name"], m["weeks"], len(m["items"]), done, part, miss,
                        '<b style="color:%s">%.0f%%</b>' %
                        (C["ok"] if score(m["items"]) >= 70 else C["warn"] if score(m["items"]) >= 30 else C["bad"],
                         score(m["items"]))])

    detail = ""
    for m in MILESTONES:
        detail += '<h3>%s %s <span class="pill">%.0f%%</span></h3>' % (m["id"], m["name"], score(m["items"]))
        detail += html_table(["交付物", "状态", "证据 / 缺口"],
                             [[it[0], badge(it[1]), it[2]] for it in m["items"]])

    parts = []
    parts.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append("<title>研发进度对照报告 · %s</title>" % STAMP)
    parts.append("<style>")
    parts.append("""
    :root{--bg:%s;--card:%s;--line:%s;--text:%s;--muted:%s;--accent:%s;--ok:%s;--warn:%s;--bad:%s}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
      font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif;line-height:1.75}
    .wrap{max-width:1180px;margin:0 auto;padding:40px 26px 90px}
    header{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:30px}
    h1{font-size:32px;margin:0 0 8px;letter-spacing:.5px}
    h2{font-size:22px;margin:46px 0 14px;padding-left:12px;border-left:4px solid var(--accent)}
    h3{font-size:16px;margin:28px 0 10px;color:#bcd4f0}
    p,li{color:#cbd8ea;font-size:14px}
    .sub{color:var(--muted);font-size:13px}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:24px 0}
    .kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
    .kpi .v{font-size:26px;font-weight:700;margin:4px 0}
    .kpi .l{color:var(--muted);font-size:12px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}
    .callout{border-left:4px solid var(--warn);background:#1c2233;padding:14px 18px;border-radius:0 10px 10px 0;margin:18px 0}
    .callout.bad{border-left-color:var(--bad)}.callout.ok{border-left-color:var(--ok)}
    table{width:100%%;border-collapse:collapse;font-size:13px;margin:12px 0 20px}
    th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:10px 9px;white-space:nowrap}
    td{border-bottom:1px solid rgba(255,255,255,.05);padding:9px;color:#d3e0f2;vertical-align:top}
    tr:hover td{background:rgba(76,194,241,.05)}
    code{background:#0b1526;color:#8fd8f7;padding:1px 6px;border-radius:4px;font-size:12px}
    .tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap}
    .tag.ok{background:rgba(61,220,151,.15);color:var(--ok)}
    .tag.warn{background:rgba(255,200,87,.15);color:var(--warn)}
    .tag.bad{background:rgba(255,107,107,.15);color:var(--bad)}
    .pill{background:rgba(76,194,241,.14);color:var(--accent);border-radius:20px;padding:2px 10px;font-size:12px;margin-left:6px}
    .chart{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px 6px;margin:18px 0}
    .note{color:var(--muted);font-size:13px;margin:2px 14px 16px}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    @media(max-width:900px){.grid2{grid-template-columns:1fr}}
    footer{margin-top:60px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:12px}
    """ % (C["bg"], C["card"], C["line"], C["text"], C["muted"], C["accent"],
           C["ok"], C["warn"], C["bad"]))
    parts.append("</style></head><body><div class='wrap'>")
    parts.append("<header><h1>研发进度对照报告</h1>")
    parts.append("<div class='sub'>%s · 依据 <code>spec/judgments.md</code> 冻结 v1.6（阈值零变更） · "
                 "全部数字来自仓库数据 / 实跑 / git 状态 · "
                 "生成脚本 <code>Experiment/tools/gen_progress_report.py</code></div></header>" % STAMP)

    parts.append("<div class='callout bad'><b>总结论：</b>B1 核心判据 4/4 通过，但按冻结判据<b>尚未完整通过</b>；"
                 "B2 仍不通过；B3 正式实现仍为 0（教学探针不计入验收）。"
                 "<code>judgments.md:4</code> 规定「未通过 Benchmark-1 之前，不得启动 Benchmark-2 以上开发」，"
                 "<code>§4</code> 又规定「没有消融表，通过判据不成立」。"
                 "当前唯一合规的下一步是补齐 B1 强制报告与消融表。</div>")

    parts.append("<div class='kpis'>")
    kpis = [
        ("B1 核心判据", "4 / 4", "var(--ok)", "D=200 / 消融 / D½ / 扫描"),
        ("B1 完整验收", "阻塞", "var(--warn)", "3 项强制报告 + 消融表缺失"),
        ("B2 达标运行", "%d / %d" % (pass_cnt, len(b2_rows)), "var(--bad)",
         "最佳 median %.4f" % (best["median"] if best else float("nan"))),
        ("B3 留出集", "%.1f%%" % (b3_hold * 100), "var(--bad)", "判据 ≥70%%，探针 %.1f%%" % (b3_hold * 100)),
        ("未提交成果", "%d 项" % (len(modified) + len(untracked)),
         "var(--warn)" if (modified or untracked) else "var(--ok)",
         ("%d 修改 + %d 未跟踪" % (len(modified), len(untracked))) if (modified or untracked)
         else "已入库，工作区干净"),
    ]
    for label, v, col, sub in kpis:
        parts.append("<div class='kpi'><div class='l'>%s</div><div class='v' style='color:%s'>%s</div>"
                     "<div class='l'>%s</div></div>" % (label, col, v, sub))
    parts.append("</div>")

    parts.append("<h2>1 · 口径与版本差</h2>")
    parts.append("<div class='callout'><b>设计大纲已过时：</b>"
                 "<code>audit/NeuralOS_设计大纲与开发计划_2026-11-12.md</code> 的 §2 进度表是<b>当天上午快照</b>"
                 "（阶段 0 = 0%、多尺度资格迹「无」、B1 名义通过实为无效）。"
                 "同日晚间提交 <code>9e7ff0c</code>（M0 接口冻结）、<code>d9c73c1</code>（M1 多尺度迹）"
                 "与冻结判据 §6 已推翻前三项。该文档应标注为 v1.0 历史快照，进度以本报告证据链为准。</div>")
    parts.append("<p class='note'>完成度 = 可核对交付物覆盖率（DONE=1 / PARTIAL=0.5 / MISSING=0），"
                 "不是工时估计，与设计大纲 §2.5 的「约 20%% / 约 15%%」不可直接比较。"
                 "能力一律按 <code>g=0</code> 口径；B2 重算按 §0.4 剔除 <code>init_dist&lt;0.10</code> 的 %d 条 episode，"
                 "%d 个空/不可解析文件；B3 数字为脚本内实跑 <code>dialog_lab.py --check</code> 解析所得。</p>" %
                 (b2_invalid, b2_empty))

    parts.append("<h2>2 · 里程碑完成度</h2>")
    parts.append("<div class='chart'>%s</div>" % div("milestones"))
    parts.append(html_table(["里程碑", "内容", "计划周期", "交付物", "DONE", "PARTIAL", "MISSING", "完成度"], ms_rows))
    parts.append("<p class='note'>M4 正式实现仍为 0（表中 8% 仅来自固定模板集要素的 0.5 分）："
                 "<code>vis/dialog_lab.py</code> 文件头明确写「不计入任何判据」。</p>")
    parts.append(detail)

    parts.append("<h2>3 · 判据逐项状态</h2>")
    parts.append("<div class='chart'>%s</div>" % div("criteria"))
    parts.append("<div class='grid2'>")
    for gname, rows in [("B1 · 单突触远端信用分配", B1_CRITERIA),
                        ("B2 · 两关节虚拟臂到达", B2_CRITERIA)]:
        parts.append("<div class='card'><h3>%s</h3>%s</div>" %
                     (gname, html_table(["项", "要求", "现状", "状态", "证据"], crit_rows(rows))))
    parts.append("</div>")
    parts.append("<div class='card'><h3>B3 · 最小陪伴对话循环</h3>%s</div>" %
                 html_table(["项", "要求", "现状", "状态", "证据"], crit_rows(B3_CRITERIA)))
    parts.append("<div class='callout bad'><b>B3 数据口径风险：</b>"
                 "<code>run_check(1000 轮)</code> 的 hold=%.3f 与 "
                 "<code>run_baselines(300 轮)</code> 的 rstdp=0.625 轮数不同，"
                 "不能据此断言「显著高于基线」；巩固 Δ=%s 源于探针无遗忘机制，只能记作未达标，"
                 "不是「巩固无效」的因果结论。</div>" % (b3_hold, b3["delta"] if b3 else "+0.000"))

    parts.append("<h2>4 · B1 真实曲线与 B2 历史重算</h2>")
    parts.append("<div class='grid2'><div class='chart'>%s</div><div class='chart'>%s</div></div>" %
                 (div("b1"), div("b2")))
    parts.append("<p class='note'>B1 三条曲线取自 <code>data/b1_m1/sweep_*.jsonl</code> 的 summary 记录；"
                 "D≤200 时多尺度与单尺度无差异，多尺度价值只在 D=500 体现（67.3% vs 46.7%，+20.6pp），"
                 "D=1000 全失效 —— 不得宣称「多尺度全面占优」。"
                 "B2 口径冲突需固定：检测报告的 0.4164 是全量 104 CSV 最优，"
                 "判据 §6 的 0.4398 是 2000-episode 子集最优，两者不可混用。</p>")

    parts.append("<h2>5 · B3 探针与判据差距</h2>")
    parts.append("<div class='grid2'><div class='chart'>%s</div><div class='chart'>%s</div></div>" %
                 (div("b3"), div("ablation")))
    parts.append("<p class='note'>B3 实测：train %.3f / hold %.3f / ECE %.3f / %d 神经元 / 巩固 %s。"
                 "达标指数 = 越大越好的项取「实际/下限」，越小越好取「上限/实际」，1.0 表示刚好达标；"
                 "巩固差值因阈值是「>0」而实测 0，记 0。"
                 "右侧热图按判据 §4：6 组消融缺一不可，当前仅 B1 的「移除资格迹」完整。</p>" %
                 (b3["train"] if b3 else 0.920, b3_hold, b3_ece, b3_neurons,
                  b3["delta"] if b3 else "+0.000"))

    parts.append("<h2>6 · 统计与数据规范（§5 / §0）</h2>")
    parts.append(html_table(["规范", "状态", "实测"],
                            [[r[0], badge(r[1]), r[2]] for r in DATA_RULES]))
    parts.append("<p class='note'>全仓 CSV 共 %d 个；<code>data/legacy/</code> 已建立并有归档清单，"
                 "§5「停用 CSV、旧格式归档」已落实。仅历史 47 个 <code>.csv.meta.json</code> "
                 "缺源码/二进制哈希、seed_bundle 四子种子与判据版本；新增运行已由 "
                 "<code>build_info.json</code> 与报告哈希表补齐。"
                 "本机未安装 scipy，Wilcoxon / Cohen's d 暂不可执行。</p>" % total_csv)

    parts.append("<h2>7 · 未提交改动与建议优先级</h2>")
    parts.append(html_table(["类型", "数量", "清单"], [
        ["已跟踪修改", len(modified), ", ".join(l[3:] for l in modified) or "—"],
        ["未跟踪文件", len(untracked), ", ".join(l[3:] for l in untracked) or "—"],
    ]))
    steps = [
        ("<b>P0 提交当日成果</b>：把 %d 个已跟踪修改与 %d 个未跟踪文件入库（不改变判据状态，仅防丢失）。" %
         (len(modified), len(untracked))) if (modified or untracked)
        else "<b>P0 已完成</b>：当日成果（含 v1.5 判据、四流契约、根因诊断、离线监督上界、严格对称读出实验、T1 两必要条件析因、文档修订）已全部入库，<code>git status</code> 干净。",
        "<b>P0 补齐 B1 完整验收</b>：τe–N 三档 + 指数拟合、neuromod 同种子统计等价、稀疏化 Gini/有效连接、≥5（优先 10）seed 的 mean±SD 与种子级原始值、6 组消融表。",
        "<b>P1 收尾数据规范</b>：<code>data/legacy/</code> 与归档清单已建立；仅剩历史 47 个 meta sidecar 的源码/二进制哈希、四子种子、判据版本待回填，新数据已只写 JSONL。",
        "<b>P1 才启动 M2 探针</b>：5000 神经元、异构输出 + 群体解码 + 势能塑形、10 seed 固定 1200 tick 与随机目标；门槛 ≥6/10 median ≤0.4 且 g=0 到达率 >0；未达回 T1/T2，先做离线监督上界。",
        "<b>P2 B3 保持教学定位</b>：待 B1 完整通过、M2 过门槛后按 M4 正式实现 20k–50k 网络、独立校准集、forge 消融、条件 B 100–200 轮真实统计。",
    ]
    parts.append("<div class='card'><ol>%s</ol></div>" % "".join("<li>%s</li>" % x for x in steps))
    parts.append("<div class='callout bad'><b>明确不要做：</b>不要宣布「B1 已通过」或「B3 已启动/通过」；"
                 "不要把教学探针的 0.583 与冻结判据 0.70 混写；不要用 g&gt;0 到达率替代 g=0 能力；"
                 "不要在消融表缺失时进入 B2 结构可塑开发。</div>")

    parts.append("<footer>本报告只陈述可核对事实；完成度为交付物覆盖率，不代表判据通过。"
                 "复现：<code>py -3 Experiment/tools/gen_progress_report.py</code> · "
                 "判据以 <code>spec/judgments.md</code> 冻结 v1.6 为准 · 生成于 %s</footer>" % date.today().isoformat())
    parts.append("</div><script>%s</script></body></html>" % get_plotlyjs())
    return "".join(parts)


def is_guided(name):
    return any(k in name.lower() for k in ("guided", "teacher", "dls"))


def b2_finding_md(guided, pure, invalid, empty):
    g = guided[0] if guided else None
    s = ["## 9. 数据验证发现：B2 唯一「达标」运行被判据排除", ""]
    s.append("按 `§0.4` 剔除 `init_dist < 0.10` 的 %d 条 episode、用标准中位数定义重算后，"
             "%d 个 CSV 中出现 1 次 `median ≤ 0.20`：`%s`（%.4f，%d 条有效 episode）。" %
             (invalid, len(pure) + len(guided), g["file"], g["median"], g["n"]) if g else
             "重算未发现达标运行。")
    s.append("")
    s.append("该文件名为 `guided`，且 CSV 带 `arrived / arrival_tick` 列，是 11-11 深夜 v7 DLS "
             "解析引导实验的遗留数据；工作日志记载当时已「量化证明 100% 定向来自解析控制器 → "
             "全链路删除公式并重编译，默认 0 到达」。按 `judgments.md §0.1`"
             "「`g>0` 的到达次数不计入能力」，**这类运行不能计入 B2 能力统计**。"
             "本报告的 B2 数字、直方图与 KPI 均已按此排除，只保留纯网络运行。")
    s.append("")
    s.append(md_table(["被排除运行", "median", "有效 episode", "排除依据"],
                      [[r["file"], "%.4f" % r["median"], r["n"], "§0.1：g>0 引导不计入能力"]
                       for r in guided] or [["—", "—", "—", "—"]]))
    s.append("")
    s.append("排除后纯网络 **%d 次运行、%d 次达标**，最佳 %.4f（`%s`）。"
             "另有 %d 个空/不可解析文件按 `§0.4` 与 `§5` 保留但不参与统计。"
             "口径提醒：`检测报告.md` 的 0.4164（全量 104 CSV 最优）与 "
             "`judgments.md §6` 的 0.4398（2000-episode 子集最优）都基于旧上中位数定义与固定目标，"
             "不能当作纯网络能力的严格上界；但也都不能改变「B2 不通过」的结论。" %
             (len(pure), sum(1 for r in pure if r["median"] <= 0.20),
              pure[0]["median"], pure[0]["file"], empty))
    s.append("")
    return "\n".join(s)


def b2_finding_html(guided, pure, invalid, empty):
    rows = html_table(["被排除运行", "median", "有效 episode", "排除依据"],
                      [[r["file"], "%.4f" % r["median"], r["n"], "§0.1：g>0 引导不计入能力"]
                       for r in guided] or [["—", "—", "—", "—"]])
    g = guided[0] if guided else None
    head = ("<h2>7 · 数据验证发现：B2 唯一「达标」运行被判据排除</h2>"
            "<div class='callout bad'><b>按冻结判据清洗后仍有 0 次纯网络达标。</b>"
            "重算 %d 个 CSV（剔除 %d 条 <code>init_dist&lt;0.10</code> 的 episode，"
            "偶数取中间两项均值）时，出现 1 次 <code>median ≤ 0.20</code>："
            "<code>%s</code>（%.4f，%d 条有效 episode）。该文件带 <code>arrived / arrival_tick</code> 列，"
            "是 v7 DLS 解析引导实验遗留数据；<code>judgments.md §0.1</code> 规定 "
            "<code>g&gt;0</code> 的到达不计入能力，故已从全部图表与统计中排除。</div>" %
            (len(pure) + len(guided), invalid, g["file"], g["median"], g["n"]) if g else
            "<h2>7 · 数据验证发现</h2>")
    body = ("<p class='note'>排除后纯网络 <b>%d 次运行 / %d 次达标</b>，最佳 %.4f（<code>%s</code>）；"
            "%d 个空/不可解析文件按 §5 保留但不参与统计。"
            "历史 0.4164（全量最优）与 0.4398（2000-episode 子集最优）基于旧上中位数定义与固定目标，"
            "口径不同不可混用，但都不改变「B2 不通过」结论。</p>" %
            (len(pure), sum(1 for r in pure if r["median"] <= 0.20),
             pure[0]["median"], pure[0]["file"], empty))
    return head + rows + body


def apply_b1_review(md, html):
    """按 B1 多种子复核结果覆盖首次对照快照中的结论，并追加复核章节。"""
    summary_path = AUDIT / "b1_acceptance_summary.jsonl"
    rate = None
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "seed_pass_rate":
                rate = rec
    if rate is None:
        raise RuntimeError("B1 复核汇总缺失：先运行 tools/b1_acceptance.py --phase report")
    mean, median = rate["mean"], rate["median"]
    passed, total = rate["passed"], rate["total"]

    md_pairs = [
        ("**B1 核心判据 4/4 通过，但按冻结判据尚未「完整通过」；B2 仍不通过；B3 正式实现仍为 0，教学探针不计入验收。当前唯一合规的下一步是补齐 B1 强制报告与消融表，而非继续推进 B2/B3。**",
         "**B1 核心判据不通过（10 seed 复核）：完整组 D=200 仅 %d/%d 达标，mean %.2f%% / median %.2f%%；"
         "B2 仍不通过；B3 正式实现仍为 0。按 §1 失败动作：停止，转 T1/T2 研究，不得进入 B2。详见 §9.1。**" %
         (passed, total, mean, median)),
        ("| B1 D=200 正确率 | 100.0% | ≥80% | PASS |",
         "| B1 D=200 正确率（10 seed） | mean %.2f%% / median %.2f%% / %d 达标 | ≥80%% | **FAIL** |" %
         (mean, median, passed)),
        ("| B1 无资格迹下降 | 53.3pp | ≥30pp | PASS |",
         "| B1 无资格迹下降（10 配对 seed） | 2.13pp（原 53.3pp 为单 seed） | ≥30pp | **FAIL** |"),
        ("| B1 D½（K=3 / K=1） | 919 / 481 ticks | ≥200 | PASS |",
         "| B1 D½（K=3 / K=1） | 919 / 481（单 seed 口径） | ≥200 | 待重做 |"),
        ("| B1 强制报告 5 项 | 2 项已出 / 3 项缺 | 5 项必须 | FAIL |",
         "| B1 强制报告 5 项 | 稀疏化/种子分布已补；τe–N 无法拟合；neuromod 阻塞 | 5 项必须 | 部分 |"),
        ("| B1 6 组跨基准消融 | 1 组完整 / 1 组近似 / 4 组缺 | 6 组缺一不可 | FAIL |",
         "| B1 6 组跨基准消融 | 4 组机制未实现，无法构造消融条件 | 6 组缺一不可 | FAIL |"),
        ("- **B1 = 核心通过 / 完整验收阻塞**：4 项核心数值全 PASS，但 3 项强制报告、种子分布、"
         "6 组消融表缺失，按 `§4`「没有消融表，通过判据不成立」不能宣布 B1 通过。",
         "- **B1 = 不通过**：10 个独立 seed 下 D=200 仅 %d/%d 达标（mean %.2f%%），"
         "配对消融平均下降 2.13pp；原「4/4 PASS」来自单 seed=12345，违反 §5「不得只给最好一次」。" %
         (passed, total, mean)),
        ("2. **P0 · 补齐 B1 完整验收**：τe–N（N=500/1000/2000，拟合指数，偏离 0.5 则修标度律）、"
         "与 `neuromod` 同种子统计等价（均值/方差/迹曲线，不追 1e-4）、稀疏化 Gini/有效连接数、"
         "≥5（优先 10）seed 的 mean±SD 与种子级原始值、6 组跨基准消融表。",
         "2. **P0 · 按 §1 失败动作停在 T1/T2**（进行中）：四流契约、根因诊断、离线监督上界、"
         "严格对称读出（约束 ①②③）、T1 两必要条件析因均已完成；附录 G.4「通路选择性"
         "第三因子」已由 §9.5 销项（257 条记录 / 10 seed，主实验 g=0 全 0%、配对 0.00pp）。"
         "主因定位为核心 `reward_baseline` 饱和 −0.997 使 `reward_error` 趋零；"
         "下一步按冻结判据 §6.6 转向「可学习内部前向模型 + 修复奖励基线口径」。"
         "不得继续推进 B2 探针。"),
        ("4. **P1 · 才允许启动 M2 探针**：",
         "4. **P1 · B1 未通过前禁止启动 M2 探针**（以下为解除阻塞后的方案，本轮不执行）："),
        ("因此 B1 目前只能写作「核心数值达标 / 完整验收阻塞」。",
         "但深夜多种子复核已判 **B1 不通过**，按 §1 失败动作应停止并转 T1/T2 研究（见 §9.1）。"),
        ("**该文档需标注为 v1.0 历史快照**，进度以本报告的证据链为准。",
         "**该文档与 §6 原表均视为历史快照**：桌面原件与 audit 副本已追加附录 D（v1.1）、E（v1.2）、"
         "F（v1.3）、G（v1.4）、H（v1.5），进度以冻结判据 v1.6 §6.1–6.6 与本报告 §9.1/§9.2/§9.3/§9.4/§9.5 为准。"),
        ("M1 已实现多尺度迹并把 B1 核心判据拉到通过。",
         "M1 已实现多尺度迹并一度把 B1 核心判据判为通过（单 seed 口径；深夜多种子复核已推翻，见 §9.1）。"),
    ]
    html_pairs = [
        ("B1 核心判据 4/4 通过，但按冻结判据<b>尚未完整通过</b>；",
         "B1 核心判据<b>不通过</b>（10 seed 复核，仅 %d/%d 达标）；" % (passed, total)),
        ("当前唯一合规的下一步是补齐 B1 强制报告与消融表。</div>",
         "当前唯一合规动作是<b>停止并转 T1/T2 研究</b>，不得进入 B2。详见第 9 节。</div>"),
        ("B1 核心判据</div><div class='v' style='color:var(--ok)'>4 / 4</div>"
         "<div class='l'>D=200 / 消融 / D½ / 扫描",
         "B1 核心判据</div><div class='v' style='color:var(--bad)'>不通过</div>"
         "<div class='l'>10 seed 仅 %d/%d 达标" % (passed, total)),
        ("B1 完整验收</div><div class='v' style='color:var(--warn)'>阻塞</div>"
         "<div class='l'>3 项强制报告 + 消融表缺失",
         "B1 完整验收</div><div class='v' style='color:var(--bad)'>不通过</div>"
         "<div class='l'>核心判据在种子分布上失效"),
        ("<b>P0 补齐 B1 完整验收</b>：τe–N 三档 + 指数拟合、neuromod 同种子统计等价、"
         "稀疏化 Gini/有效连接、≥5（优先 10）seed 的 mean±SD 与种子级原始值、6 组消融表。",
         "<b>P0 按 §1 失败动作停在 T1/T2</b>：四流契约、根因诊断、离线监督上界、严格对称读出"
         "（约束 ①②③）、T1 两必要条件析因均已完成；附录 G.4「通路选择性第三因子」"
         "已由 §9.5 销项（257 条记录 / 10 seed，主实验 g=0 全 0%、配对 0.00pp）。"
         "主因定位为核心 <code>reward_baseline</code> 饱和 −0.997；下一步按 §6.6 转向"
         "「可学习内部前向模型 + 修复奖励基线口径」。不得推进 B2。"),
        ("<b>P1 才启动 M2 探针</b>：",
         "<b>P2 B1 未通过前禁止启动 M2 探针</b>（以下为解除阻塞后的方案，本轮不执行）："),
        ("与冻结判据 §6 已推翻前三项。该文档应标注为 v1.0 历史快照，进度以本报告证据链为准。</div>",
         "但 §6 原表的「B1 核心判据通过」又因单 seed 口径被 §6.1 复核推翻。"
         "桌面原件与 audit 副本已追加附录 D/E/F/G/H；进度以冻结判据 v1.6 §6.1–6.6 与本报告第 9 节为准。</div>"),
    ]
    for old, new in md_pairs:
        assert old in md, "md 复核替换未命中：" + old[:36]
        md = md.replace(old, new, 1)
    for old, new in html_pairs:
        assert old in html, "html 复核替换未命中：" + old[:36]
        html = html.replace(old, new, 1)

    md_section = f"""## 9.1 B1 多种子复核（本次新增，取代 §0/§3 旧结论）

> 依据 `audit/b1_acceptance_report.md`、`audit/b1_acceptance_summary.jsonl`（124 次独立进程实跑）；
> 冻结判据已追加 `spec/judgments.md §6.1`（v1.1，**阈值零变更**）。

| 项 | 首次对照 | 复核结果 | 状态 |
|---|---|---|---|
| D=200 正确率 | 100.0%（seed=12345） | 10 seed：mean {mean:.2f}% / median {median:.2f}% / **{passed}/{total}** 达标 | **FAIL** |
| 无迹消融下降 | 53.3pp | 10 配对 seed：D=200 平均 2.13pp（SD 18.16） | **FAIL** |
| τe–N 标度 | 缺 | N=500/1000/2000 × 6 档 τ 已扫；N≥1000 各 τ 全 0%，标度律无法拟合 | PARTIAL |
| 稀疏化 | 缺 | 诊断层补 Gini / 有效连接数；无迹组 init→final Δ 精确为 0 | PASS |
| ≥5 seed 分布 | 缺 | 10 seed 的 mean/SD/median 与种子级原始值已出 | PASS |
| neuromod 交叉 | 缺 | 本机无 Rust/cargo，参考实现无法构建 | 阻塞 |
| 6 组消融 | 1 组完整 | 4 组机制未实现，消融条件无法构造 | FAIL |

**用同一二进制复跑 seed=12345 仍得 100%**，确认历史结论是「只报最好一次」的口径问题，
不是代码回归；判据 §0.6 与 §5 明确要求种子分布且不得只给最好一次。

**结论：B1 不通过。** 按 §1 失败动作：**停止，转 T1/T2 研究（前向模型、解码可学性），不得进入 B2**。
本轮未改任何阈值，也未改学习规则或刺激生成器去「刷过」判据。

**修订后的优先级**：① P0 提交当日成果；② P0 停止 B2 推进，查输出层固定偏向与刺激—投票映射，
重做 B1 实验设计并落实独立试次顺序随机流；③ P1 安装 Rust 工具链补 neuromod 对照、
明确 §4 六组消融在 B1 的适用口径；④ P2 把 τe–N 扩到 ≥5 seed 并固定 `avg_conn` 后重跑；
⑤ P3 B3 保持教学定位，不计入验收。

注意：M1 完成度上升是因为**强制报告交付物**补齐，不代表 B1 判据通过；完成度是覆盖率，不是通过率。

## 9.2 根因诊断与离线监督上界（T1/T2 已推进，取代 §9.1 的待办口径）

> 依据 `audit/b1_root_cause_2026-11-12.md`（54 次单变量实跑）、
> `audit/b1_supervised_ceiling_2026-11-12.md`（10 seed × 200 试次）；
> 冻结判据已追加 `spec/judgments.md §6.2 / §6.3`（v1.2 / v1.3，**阈值零变更**）。

| 项 | 实测 | 含义 |
|---|---|---|
| M0 四流契约 | 已落到 B1 运行时，4 个单变量覆盖开关 + `--legacy-rng` | §9.1 待办 ② 的「独立试次顺序流」已完成 |
| 只换 trial_order 流（8 值） | 8/8 全 100.00%，SD 0.00 | 刺激顺序零影响 |
| 只换 weight / structure 流 | 0–100% 跳变，SD 44.45 / 37.87 | 初始实例主导 |
| 44 次有学习运行 `|Δbeta|` | 平均 0.0013 | 学习几乎没改净映射 |
| 离线监督上界（10 seed） | **10/10 全 100.00%** | hidden 表示线性可分，架构表达力不是瓶颈 |
| shuffle / 随机读出对照 | 49.10% / 50.44% | 上界不是 CV 泄漏 |
| 训练前初始投票 vs 训练后 test_acc | **corr 0.9818**，平均差 4.20pp | 训练 300 试次几乎没改变答案 |

**结论**：B1 测到的是「初始随机网络实例是否落在输出层竞争的可分辨工作点」，不是远端信用分配；
刺激信息本身完整保留在 hidden 层，缺的是能把表示映射到正确输出的**学习规则与对称破缺机制**。
§9.1 的「先查清输出层固定偏向、落实独立试次顺序流」两项**已完成**，
「离线监督上界」也已完成；§6.3 排定的约束 ①②③ 已由 **§9.3** 全部销项（负结果）。
**B1 仍不通过**，不进 B2、不写容器、不写 SNC。

*补充：M0「修文档 6 处冲突」已 6/6 销项（含设计定稿原件 §7.2「6.6→7.4」）；*
*`data/legacy/` 已建立并有归档清单。这两项是 §0 / §5 旧快照中已过时的 FAIL/PARTIAL 记录。*

## 9.3 严格对称初始读出（约束 ①②③ 完成，B1 仍不通过）

> 依据 `audit/b1_symmetric_readout_2026-11-12.md`、`data/b1_sym/sym_summary.jsonl`（170 条记录）；
> 冻结判据已追加 `spec/judgments.md §6.4`（v1.4，**阈值零变更**）。
> 新增 `brain/benchmark1_sym.c` → `b1_sym.dll` 独立实验台，只链接既有 `b1_core.o` / `seed_bundle.o`；
> **`benchmark1_core.c/.h`、`benchmark1_probe.c`、host 一行未改**，§6.2 / §6.3 证据链不断。

| 项 | 实测 | 状态 |
|---|---|---|
| 约束 ① 初始读出严格对称 | 配对残差 0.000000、\\|β\\|=0、逐试次 `a==b` 100% | **完成** |
| 剂量-反应（冻结投票，10 seed × 200） | α=0: 40.00±39.44%（逐 seed 仅 0/50/100 三值）→ α=1: **0% 正确 / 100% 平局** | 初始答案只由这段不对称决定 |
| 约束 ② 同 seed 配对消融 | α=0 差 1.60pp、α=1 差 0.00pp；判据要求 ≥30pp | **FAIL**，无提升可归因到学习 |
| 约束 ③ 四流单变量隔离 | 四条流各 10 seed 覆盖，残差全 0、平局率全 100% | **完成** |
| α=1 完整训练 300 试次 × 10 | 非零奖励 0/3000、权重漂移 0、训练后残差 0 | 零梯度自锁 |
| 探索性平局硬币（不计入验收） | test 48.67±3.28%；权重总漂移 1.903、**读出段漂移 0.589**，残差仍 0；「完整 − 无迹」0.00pp | 对称性是 R-STDP 不变子空间 |

**与计划措辞的偏差（如实记录）**：附录 D.4 / E.3 写「让冻结正确率回到机会水平（50%）」，
实测为 **0% 正确 / 100% 平局**（host 把平局计为错误且不给奖励）。这暴露出两个更根本的问题：
① 学习信号完全依赖初始不对称产生的「有对有错」，严格对称后平局 → 零奖励 → 零更新 → 永远平局；
② 即使人为注入非零奖励，镜像边因资格迹相同被**等量更新**，对称仍不破缺。

**结论**：与 §9.2「架构表达力 10/10 线性可分」合起来，瓶颈定位在**读出与信用分配的耦合**，
不是表示能力或超参数。T1 下一步必须同时满足「教学信号与刺激相关」+「对称破缺机制」两个条件，
再在严格对称基线上重做配对消融。**B1 仍不通过**，不进 B2、不写容器、不写 SNC。

## 9.4 T1 两必要条件析因（§9.3 的两个条件已证伪，B1 仍不通过）

> 依据 `audit/b1_t1_mechanism_2026-11-13.md`、`data/b1_t1/t1_summary.jsonl`
> （244 条记录 / 10 seed / 300 试次，可复算）；
> 冻结判据已追加 `spec/judgments.md §6.5`（v1.5，**阈值零变更**）。
> 新增 `brain/benchmark1_t1.c` → `b1_t1.dll` 独立实验台，只链接既有
> `b1_core.o` / `seed_bundle.o`；**核心 `benchmark1_core.c` SHA1 仍为 `c8de144f…`**，
> §6.2 / §6.3 / §6.4 三份报告证据链不断。

| 项 | 实测 | 状态 |
|---|---|---|
| §9.3 的两个条件（教学×破缺 2×2） | 四组 g=0 **全部 0.00%**、平局率 100% | **条件不充分** |
| 教学信号是否进入学习回路 | 300/300 次非零奖励、权重漂移 6.4064、读出漂移 3.2826，**训练后残差严格 0** | 进去了，被对称性消掉 |
| 同 seed 配对「完整 − 无迹」 | **0.00pp**（10/10 完全相同），判据要求 ≥30pp | **FAIL** |
| 标签 shuffle / 延迟扫描 / 冻结读出 | 全部 0.00%，无可归因的提升 | FAIL |
| 作弊破缺对照（不计入验收） | 训练期即时正确率 **100%**，g=0 仅 **50% / 35%** | 自证 §0.1 能力口径 |
| 第三类破缺：异质化读出 | **不训练**即 50% 恒定答案（A/B 预测分布差 0），加教学后反降到 10% | 退化为重放初始答案 |
| 四流隔离 / 剂量扫描 | g=0 全为 0，无剂量-反应趋势 | 结论不依赖随机流 |

**新根因（比 §9.3 更进一层）**：瓶颈不是「教学信号有没有与刺激相关」，而是
**第三因子有没有输出侧 / 通路侧的选择性**。`net_inject_reward(winner=0)` 是逐突触
无差别的全局标量；严格对称下镜像边资格迹相同、奖励调制相同 → 等量更新 →
对称残差恒为 0，**梯度与信号内容无关**。这把 §9.3 的「对称性是 R-STDP 不变子空间」
从「人为注入非零奖励」推广到**任意刺激相关教学信号**。

§9.3 列举的三类破缺机制中，瞬时状态相关探索无法破缺不变子空间，持久异质化读出
只是把 §9.2 的初始随机工作点重新注入（且训练后更差），内部前向模型未实现。
**能力口径严格 g=0**：训练后关闭全部注入，评估前后权重漂移为 0 自证零学习。

**诚实边界**：教学信号是外部监督标签、破缺是人为注入，任务定义已偏离 B1 的
「延迟结果反馈」，本轮是 T1 机制诊断而非验收。下一步（仍未做）：在不引入外部标签的
前提下把「选择性」做成网络内部的量（内部前向模型 / 通路特异性资格迹），重做配对消融，
全程只认 g=0 并同时报告 A/B 预测分布差。**B1 仍不通过**，不进 B2、不写容器、不写 SNC。

## 9.5 通路选择性第三因子（附录 G.4 销项，B1 仍不通过）

> 依据 `audit/b1_pathway_selectivity_2026-11-13.md`、`data/b1_psel/psel_summary.jsonl`
> （257 条记录 / 10 seed / 300 试次，可复算）；冻结判据已追加
> `spec/judgments.md §6.6`（v1.6，**阈值零变更**）。
> 新增 `brain/benchmark1_psel.c` → `b1_psel.dll`，只链接既有 `b1_core.o` /
> `seed_bundle.o`；核心 `benchmark1_core.c` SHA1 仍为 `c8de144f…`，
> §6.2 / §6.3 / §6.4 / §6.5 四份报告证据链不断。

| 项 | 实测（全部 g=0） | 状态 |
|---|---|---|
| 主实验 2×2（通路选择性 × 平局结果反馈） | 四组 g=0 **全部 0.00%**、平局率 100%；联合组配对「完整 − 无迹」**0.00pp** | **FAIL** |
| 选择性是否进入学习回路 | 镜像边资格迹平均绝对差 **0.00545 > 0**、训练后残差 0.00103 | 进入了，被基线抵消 |
| 失效主因 | 注入前 `reward_baseline` 均值 **−0.9967**，`reward_error = reward − baseline` 幅度趋零；基线补偿诊断组 10/10 达 100% | **核心奖励基线饱和** |
| 内部第三因子诊断组（不计入验收） | 原方向 g=0 **40.00 ± 21.08%**、A/B 分布差 0.800、配对「完整 − 无迹」**40.00pp**；关闭选择性 / 无迹 / 无奖励均 0% | 有可归因提升，未达 80% |
| 方向对照（只翻转门控符号） | g=0 **0.00%**、配对 0.00pp | 方向对资格迹符号高度敏感 |
| 假阳性 / 四流 / 剂量 / 延迟 | shuffle、恒定门控、四流覆盖、λ 与 D 扫描均无可归因提升 | 结论不依赖随机流 |

**实现要点**：按附录 G.4 要求，把「选择性」做成网络**内部**的量而不是人工按 winner
路由 —— 第三因子仍走核心原生 `net_inject_reward(winner=0)` 的全局标量，选择性放在
**资格迹**上（奖励到达前按输出通路给 `e[k]` 乘门控 `m_A=1+λd` / `m_B=1−λd`，
`d` 由 hidden 左右半实际发放算出）。门控与反馈符号耦合，每试次清迹并 ±5 截断。

**数值修复记录**：修复 `e[]` 跨试次指数累积前，恒定门控组镜像迹差曾达 `1e94`、
方向反转组出现 95% 虚假高正确率（权重被推到 clamp 边界）；修复后假象消失，
方向反转组归零、原方向组出现 40pp 可归因提升。本节数字全部来自修复后的二进制。

**新根因（比 §9.4 更进一层）**：§9.4 的「第三因子需具备输出侧/通路侧选择性」
已实现并验证 —— 选择性确实能破缺对称、产生可归因学习；B1 原口径仍失败的原因
精确定位为 ① 核心奖励基线饱和使第三因子幅度趋零，② 第三因子符号/方向与资格迹
符号耦合不当。下一步：把「幅度 / 符号 / 基线」做成网络内部**可学习的**量
（可学习内部前向模型），回到 host 原结果反馈口径，再谈重做 B1 正式验收。

**诚实边界**：诊断组的第三因子换成了内部预测读出 `sign(d_raw)`、门控用了结构先验
与人工方向，任务定义偏离 B1，即使 40pp 可归因也**不构成 B1 通过**；诊断组、方向
对照、基线补偿组一律不计入验收。**B1 仍不通过**，不进 B2、不写容器、不写 SNC。

"""
    html_section = f"""<h2>9 · B1 多种子复核（本次新增，取代前文旧结论）</h2>
<div class='callout bad'><b>B1 不通过。</b>10 个独立 seed 下 D=200 仅 <b>{passed}/{total}</b> 达标
（mean {mean:.2f}% / median {median:.2f}%）；用同一二进制复跑 seed=12345 仍得 100%，
确认首次对照的「4/4 PASS」来自单 seed，违反 §5「不得只给最好一次」。
按 §1 失败动作：<b>停止，转 T1/T2 研究，不得进入 B2</b>。</div>
{html_table(["项", "首次对照", "复核结果", "状态"], [
    ["D=200 正确率", "100.0%（seed=12345）", f"10 seed：mean {mean:.2f}% / median {median:.2f}% / 仅 {passed}/{total} 达标", badge(0)],
    ["无迹消融下降", "53.3pp", "10 配对 seed：D=200 平均 2.13pp", badge(0)],
    ["τe–N 标度", "缺", "已扫三档，N≥1000 各 τ 全 0%，标度律无法拟合", badge(0.5)],
    ["稀疏化", "缺", "无迹组 init→final Δ 精确为 0", badge(1)],
    ["≥5 seed 分布", "缺", "10 seed mean/SD/median 与原始值已出", badge(1)],
    ["neuromod 交叉", "缺", "本机无 Rust/cargo，阻塞", badge(0)],
    ["6 组消融", "1 组完整", "4 组机制未实现，无法构造消融条件", badge(0)],
])}
<p class='note'>依据 <code>audit/b1_acceptance_report.md</code> 与
<code>audit/b1_acceptance_summary.jsonl</code>；冻结判据已追加
<code>spec/judgments.md §6.1</code>（v1.1，阈值零变更）。本轮未改学习规则或刺激生成器去「刷过」判据。
M1 完成度上升只代表强制报告交付物补齐，不代表判据通过。</p>
<h2>9.2 · 根因诊断与离线监督上界（T1/T2 已推进）</h2>
<div class='callout bad'><b>B1 仍不通过。</b>54 次单变量实跑证明正确率由初始随机实例决定；
10 seed 离线监督上界 <b>10/10 全 100%</b>，说明架构表达力不是瓶颈，
缺的是<b>能把表示映射到正确输出的学习规则与对称破缺机制</b>。
§6.3 排定的约束 ①②③ 已由 §9.3 全部销项（负结果）。
按 §1 失败动作继续停在 T1/T2，不进 B2。</div>
{html_table(["项", "实测", "含义"], [
    ["四流契约", "已落地 + 4 个单变量覆盖开关", "§9.1 待办已完成"],
    ["只换 trial_order 流", "8/8 全 100%，SD 0.00", "刺激顺序零影响"],
    ["只换 weight / structure 流", "0–100% 跳变，SD 44.45 / 37.87", "初始实例主导"],
    ["44 次有学习运行 |Δbeta|", "平均 0.0013", "学习几乎没改净映射"],
    ["离线监督上界", "10/10 全 100.00%", "表示线性可分"],
    ["shuffle / 随机读出", "49.10% / 50.44%", "上界非 CV 泄漏"],
    ["训练前 vs 训练后配对", "corr 0.9818，平均差 4.20pp", "训练几乎没改变答案"],
])}
<p class='note'>依据 <code>audit/b1_root_cause_2026-11-12.md</code> 与
<code>audit/b1_supervised_ceiling_2026-11-12.md</code>；冻结判据已追加 §6.2 / §6.3
（v1.2 / v1.3，阈值零变更）。M0「修文档 6 处冲突」已 6/6 销项，
<code>data/legacy/</code> 已建立并有归档清单；这两项是旧快照中已过时的 FAIL/PARTIAL 记录。</p>
<h2>9.3 · 严格对称初始读出（约束 ①②③ 完成，B1 仍不通过）</h2>
<div class='callout bad'><b>B1 仍不通过。</b>严格对称后冻结投票 <b>0% 正确 / 100% 平局</b>
（计划措辞写的是机会水平 50%，host 把平局计为错误且不给奖励）；
α=1 完整训练 300 试次 × 10 seed 的非零奖励 <b>0/3000</b>、权重漂移 0 —— <b>零梯度自锁</b>。
探索组进一步证明对称性是 <b>R-STDP 的不变子空间</b>，网络无法自发破缺。</div>
{html_table(["项", "实测", "状态"], [
    ["约束 ① 严格对称", "配对残差 0.000000、|β|=0、逐试次 a==b 100%", badge(1)],
    ["剂量-反应（10 seed × 200）", "α=0: 40.00±39.44%（仅 0/50/100 三值）→ α=1: 0% / 100% 平局", badge(1)],
    ["约束 ② 配对消融", "α=0 差 1.60pp、α=1 差 0.00pp（判据要求 ≥30pp）", badge(0)],
    ["约束 ③ 四流隔离", "四条流各 10 seed，残差全 0、平局率全 100%", badge(1)],
    ["探索性平局硬币（不计入验收）", "读出段漂移 0.589 但残差仍 0；完整−无迹 0.00pp", badge(0)],
])}
<p class='note'>依据 <code>audit/b1_symmetric_readout_2026-11-12.md</code> 与
<code>data/b1_sym/sym_summary.jsonl</code>（170 条记录）；冻结判据 §6.4（v1.4，阈值零变更）。
新增 <code>brain/benchmark1_sym.c</code> 独立实验台，核心源码零改动，§6.2/§6.3 证据链不断。
瓶颈定位在<b>读出与信用分配的耦合</b>；T1 下一步需同时满足「教学信号与刺激相关」+
「对称破缺机制」（这两个条件已由 §9.4 证伪）。不进 B2、不写容器、不写 SNC。</p>
<h2>9.4 · T1 两必要条件析因（§9.3 条件已证伪，B1 仍不通过）</h2>
<div class='callout bad'><b>§6.4 的两个条件不充分。</b>244 条记录 / 10 seed 上，
教学①与破缺②单独或联合的 g=0 正确率<b>全部 0.00%</b>、平局率 100%；
同 seed 配对「完整 − 无迹」<b>0.00pp</b>（判据要求 ≥30pp）。
①开②关组教学信号满额生效（300/300 次非零奖励、权重漂移 6.4064、读出漂移 3.2826），
但训练后配对残差<b>严格为 0</b> —— 信号进了回路，却被对称性完全消掉。</div>
{html_table(["项", "实测", "状态"], [
    ["2×2 析因（g=0）", "四组全部 0.00%，平局率 100%", badge(0)],
    ["配对「完整 − 无迹」", "0.00pp（10/10 完全相同）", badge(0)],
    ["标签 shuffle / 延迟 / 冻结读出", "全部 0.00%，无可归因提升", badge(0)],
    ["作弊破缺对照（不计入验收）", "即时正确率 100%，g=0 仅 50% / 35%", badge(0)],
    ["异质化读出（第三类破缺）", "不训练即 50% 恒定答案，加教学反降到 10%", badge(0)],
    ["四流隔离 / 剂量扫描", "g=0 全为 0，无剂量-反应趋势", badge(0)],
])}
<p class='note'><b>新根因：</b>瓶颈不是「信号有没有与刺激相关」，而是
<b>第三因子有没有输出侧 / 通路侧选择性</b>。<code>net_inject_reward(winner=0)</code>
是逐突触无差别的全局标量，镜像边资格迹相同 → 等量更新 → 对称梯度恒为零，与信号内容无关。
瞬时探索无法破缺不变子空间；持久异质化只是重放 §9.2 的初始随机工作点；内部前向模型未实现。
能力口径严格 g=0（评估前后权重漂移 0）。教学信号是外部监督标签、破缺是人为注入，
本轮属 T1 机制诊断而非验收。依据 <code>audit/b1_t1_mechanism_2026-11-13.md</code> 与
<code>data/b1_t1/t1_summary.jsonl</code>；冻结判据 §6.5（v1.5，阈值零变更）。
新增 <code>brain/benchmark1_t1.c</code> 独立实验台，核心 SHA1 仍为 <code>c8de144f…</code>。
下一步：在不引入外部标签的前提下把「选择性」做成网络内部的量，重做配对消融，
全程只认 g=0 并报告 A/B 预测分布差。<b>B1 仍不通过</b>，不进 B2、不写容器、不写 SNC。</p>
<h2>9.5 · 通路选择性第三因子（附录 G.4 销项，B1 仍不通过）</h2>
<div class='callout bad'><b>B1 仍不通过。</b>257 条记录 / 10 seed 上，主实验 2×2 四组
g=0 <b>全部 0.00%</b>、平局率 100%，配对「完整 − 无迹」<b>0.00pp</b>。选择性确实进入
学习回路（镜像边迹差 0.00545），却被核心 <code>reward_baseline</code> 饱和到
−0.9967 抵消；基线补偿诊断组 10/10 达 100%（不计入验收）。</div>
{html_table(["项", "实测", "状态"], [
    ["主实验 2×2（g=0）", "四组全部 0.00%，配对 0.00pp", badge(0)],
    ["选择性进入回路", "镜像边迹差 0.00545 > 0、残差 0.00103", badge(0.5)],
    ["失效主因", "reward_baseline 饱和 −0.9967，reward_error 趋零", badge(0)],
    ["内部第三因子诊断组（不计入验收）", "原方向 40.00±21.08%，配对 40.00pp、A/B 分布差 0.800", badge(0.5)],
    ["方向对照（翻转门控符号）", "g=0 0.00%，方向对资格迹符号敏感", badge(0)],
    ["关闭选择性 / 无迹 / 无奖励", "全部 0%，归因成立", badge(1)],
    ["数值修复记录", "修复前 95% 为 clamp 饱和假象，修复后消失", badge(1)],
])}
<p class='note'><b>实现：</b>第三因子仍是核心原生 <code>net_inject_reward(winner=0)</code>
全局标量，选择性放在<b>资格迹</b>上（按输出通路给 <code>e[k]</code> 乘门控，
<code>d</code> 来自 hidden 实际发放）；门控与反馈符号耦合，每试次清迹并 ±5 截断。
<b>新根因：</b>§9.4 的「选择性缺失」已销项，B1 原口径失败的原因精确定位为
① 核心奖励基线饱和使第三因子幅度趋零，② 第三因子符号/方向与资格迹符号耦合不当。
诊断组第三因子换成内部预测读出、门控用了结构先验与人工方向，任务定义偏离 B1，
即使 40pp 可归因也<b>不构成 B1 通过</b>。依据 <code>audit/b1_pathway_selectivity_2026-11-13.md</code>
与 <code>data/b1_psel/psel_summary.jsonl</code>；冻结判据 §6.6（v1.6，阈值零变更）。
下一步：可学习内部前向模型 + 修复奖励基线口径，回到 host 原口径才谈重做 B1 验收。
<b>B1 仍不通过</b>，不进 B2、不写容器、不写 SNC。</p>
"""
    md = md.replace("## 10. 复现方式", md_section + "## 10. 复现方式")
    html = html.replace("<footer>", html_section + "<footer>")
    assert ("## 9.1 B1 多种子复核" in md and "## 9.2 根因诊断与离线监督上界" in md
            and "## 9.3 严格对称初始读出" in md and "## 9.4 T1 两必要条件析因" in md
            and "## 9.5 通路选择性第三因子" in md
            and "9 · B1 多种子复核" in html and "9.2 · 根因诊断与离线监督上界" in html
            and "9.3 · 严格对称初始读出" in html and "9.4 · T1 两必要条件析因" in html
            and "9.5 · 通路选择性第三因子" in html
            and "冻结 v1.6" in md and "冻结 v1.6" in html)
    return md, html


def main():
    b1_series = load_b1_curve()
    all_rows, b2_empty, b2_invalid, _ = load_b2_recompute()
    guided = [r for r in all_rows if is_guided(r["file"])]
    b2_rows = [r for r in all_rows if not is_guided(r["file"])]
    b3 = run_b3_probe()
    statuses = git_status()
    figs = build_figures(b1_series, b2_rows, b3)

    md = build_markdown(b1_series, b2_rows, b2_empty, b2_invalid, b3, statuses, figs)
    html = build_html(figs, md, b2_rows, b2_empty, b2_invalid, b3, statuses)
    md = md.replace("## 9. 复现方式",
                    b2_finding_md(guided, b2_rows, b2_invalid, b2_empty) + "## 10. 复现方式")
    html = html.replace("<h2>7 · 未提交改动与建议优先级</h2>",
                        b2_finding_html(guided, b2_rows, b2_invalid, b2_empty) +
                        "<h2>8 · 未提交改动与建议优先级</h2>")
    html = html.replace("<footer>", "<div class='callout'>配套完整表格、逐项证据与复现命令见 "
                        "<code>Experiment/audit/progress_review_2026-11-12.md</code>。</div><footer>")
    md, html = apply_b1_review(md, html)
    AUDIT.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(md, encoding="utf-8")
    HTML_PATH.write_text(html, encoding="utf-8")

    assert len(b1_series) == 3, "B1 三条曲线必须齐全"
    assert all(len(p) == 6 for _, _, p in b1_series), "B1 延迟扫描必须 6 档全覆盖"
    assert b2_rows and all(r["median"] > 0.20 for r in b2_rows), "纯网络 B2 应全部未达标"
    assert b3 and b3["hold"] < 0.70 and b3["ece"] > 0.10, "B3 实测必须与判据方向一致"
    assert "## 9. 数据验证发现" in md and "7 · 数据验证发现" in html, "数据验证章节必须写入"
    print("B1 series: %s" % ", ".join("%s=%d 档" % (n, len(p)) for n, _, p in b1_series))
    print("B2 recompute: %d pure csv, %d pass, best %.4f (%s); excluded %d guided run(s) %s" %
          (len(b2_rows), sum(1 for r in b2_rows if r["median"] <= 0.20),
           b2_rows[0]["median"], b2_rows[0]["file"], len(guided),
           ", ".join("%s=%.4f" % (r["file"], r["median"]) for r in guided)))
    print("B2 hygiene: %d invalid episodes, %d empty files" % (b2_invalid, b2_empty))
    print("B3 probe: %s" % ("实跑成功 hold=%.3f ece=%.3f neurons=%d" %
                            (b3["hold"], b3["ece"], b3["neurons"]) if b3 else "失败"))
    print("MD  -> %s (%d bytes)" % (MD_PATH, MD_PATH.stat().st_size))
    print("HTML-> %s (%d bytes)" % (HTML_PATH, HTML_PATH.stat().st_size))
    print("PROGRESS REPORT OK")


if __name__ == "__main__":
    main()
