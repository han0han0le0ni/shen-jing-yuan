"""生成「项目全面体检」报告：交互式 HTML（Plotly）+ Markdown 文本版。

用法: py audit/build_report.py
输入: audit/scan_results.json, Experiment/data*/ 原始 CSV
输出: audit/full_report.html, audit/检测报告.md
"""
from __future__ import annotations

import html as htmllib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = json.loads((HERE / "scan_results.json").read_text(encoding="utf-8"))
Q, S = R["quality"], R["summary"]

BLUE, ORANGE, GREEN, RED, GREY = "#2F6FB2", "#D97706", "#2E8B57", "#B23A3A", "#8A94A6"
THRESH = 0.20

# ---------------------------------------------------------------- 数据准备
b2 = pd.DataFrame([{k: v for k, v in r.items()
                    if k not in ("curve", "log", "flags", "path", "paired_log", "declared_episodes")}
                   for r in R["b2_runs"] if r.get("n")])
b2["mtime"] = pd.to_datetime(b2["mtime"])
b2 = b2.sort_values("mtime").reset_index(drop=True)


def bucket(n: int) -> str:
    if n >= 1500:
        return "2000 ep"
    if n >= 900:
        return "1000 ep"
    if n >= 400:
        return "500 ep"
    return "短跑 / 调试"


b2["group"] = b2["n"].map(bucket)
c500 = b2[(b2.n >= 400) & (b2.n < 900)]
c2000 = b2[b2.n >= 1500]
allruns = b2[b2.n >= 100]
b1 = pd.DataFrame([{k: v for k, v in r.items() if k not in ("flags", "path")}
                   for r in R["b1_runs"] if r.get("n")])
seeds = pd.DataFrame(R["b1_seeds"])
root_logs = pd.DataFrame(R["root_logs"])

best = b2.loc[b2.median_ratio.idxmin()]
latest = b2.iloc[-1]
med500 = c500.median_ratio.median()
med2000 = c2000.median_ratio.median()
best2000 = c2000.loc[c2000.median_ratio.idxmin()]

# ---------------------------------------------------------------- 图表
figs: dict[str, str] = {}


def save(key: str, fig: go.Figure, height: int = 430) -> None:
    fig.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", size=12, color="#243044"),
        title_font_size=16, margin=dict(l=70, r=30, t=70, b=60),
        legend=dict(orientation="h", y=1.08, x=0),
        hoverlabel=dict(font_size=12),
    )
    figs[key] = fig.to_html(full_html=False, include_plotlyjs=False,
                            config={"displaylogo": False, "responsive": True})


def thresh_line(fig: go.Figure) -> None:
    fig.add_hline(y=THRESH, line_dash="dash", line_color=RED, line_width=2,
                  annotation_text=f"PASS 阈值 {THRESH}", annotation_position="bottom right",
                  annotation_font_color=RED)


# 图 1：全部运行随时间的 median ratio
f = go.Figure()
for grp, color in (("500 ep", BLUE), ("1000 ep", ORANGE), ("2000 ep", GREEN), ("短跑 / 调试", GREY)):
    d = allruns[allruns.group == grp]
    if d.empty:
        continue
    f.add_trace(go.Scatter(x=d.mtime, y=d.median_ratio, mode="markers+lines", name=f"{grp}（n={len(d)}）",
                           marker=dict(size=8, color=color, line=dict(width=0.6, color="white")),
                           line=dict(width=1.2, color=color),
                           customdata=np.stack((d.file, d.n, d.mean_ratio), axis=-1),
                           hovertemplate="%{customdata[0]}<br>时间 %{x|%m-%d %H:%M}<br>"
                                         "episode %{customdata[1]}<br>中位数 %{y:.4f}<br>均值 %{customdata[2]:.4f}"
                                         "<extra></extra>"))
thresh_line(f)
f.update_layout(title="103 次有效 B2 运行从未达标：全部 median ratio > 0.20",
                xaxis_title="运行时间（文件修改时间）", yaxis_title="dist_ratio 中位数",
                yaxis=dict(range=[0, 2.0]))
save("fig1", f, 470)

# 图 2：500-episode 可比子集
f = go.Figure()
f.add_trace(go.Scatter(x=c500.mtime, y=c500.median_ratio, mode="lines+markers", name="500 ep 运行",
                       line=dict(color=BLUE, width=2), marker=dict(size=7, color=BLUE),
                       customdata=np.stack((c500.file, c500.n), axis=-1),
                       hovertemplate="%{customdata[0]}<br>%{x|%m-%d %H:%M}<br>中位数 %{y:.4f}<extra></extra>"))
recent = c500[c500.file.str.startswith(("v17", "v18", "v19"))]
f.add_trace(go.Scatter(x=recent.mtime, y=recent.median_ratio, mode="markers+text", name="最近三次调参",
                       marker=dict(size=12, color=RED, symbol="diamond"), text=recent.file,
                       textposition="top center", textfont=dict(size=10, color=RED),
                       hovertemplate="%{text}<br>中位数 %{y:.4f}<extra></extra>"))
thresh_line(f)
f.update_layout(title="同为 500 episode 的可比子集：最近 v17→v19 连续回退",
                xaxis_title="运行时间", yaxis_title="dist_ratio 中位数")
save("fig2", f)

# 图 3：2000-episode 子集
d = c2000.sort_values("mtime")
f = go.Figure()
f.add_trace(go.Bar(x=d.file, y=d.median_ratio, name="median ratio",
                   marker_color=[GREEN if v <= 0.52 else BLUE for v in d.median_ratio],
                   customdata=np.stack((d.n, d.frac_le_target, d.learning_delta), axis=-1),
                   hovertemplate="%{x}<br>episode %{customdata[0]}<br>中位数 %{y:.4f}<br>"
                                 "达标 episode 占比 %{customdata[1]:.2%}<br>"
                                 "首尾十分位改善 %{customdata[2]:.4f}<extra></extra>"))
f.add_hline(y=THRESH, line_dash="dash", line_color=RED, line_width=2,
            annotation_text="PASS 阈值 0.20", annotation_font_color=RED)
f.update_layout(title=f"长时间训练（≥1500 episode）最佳中位数 {best2000.median_ratio:.4f}，仍为阈值的 "
                      f"{best2000.median_ratio / THRESH:.1f} 倍",
                xaxis_title="运行文件（按时间排序）", yaxis_title="dist_ratio 中位数",
                xaxis=dict(tickangle=-35))
save("fig3", f, 460)

# 图 4：学习曲线
curves = [("results_best_2000.csv", "早期最佳 best_2000", GREEN, 101),
          ("v16_2000.csv", "最新长跑 v16_2000", BLUE, 101),
          ("v11_2000.csv", "v11_2000", ORANGE, 101),
          ("v19_decay.csv", "最新短跑 v19_decay", RED, 25)]
f = go.Figure()
for name, label, color, win in curves:
    p = ROOT / "data_b2" / name
    if not p.is_file():
        continue
    s = pd.read_csv(p)["dist_ratio"]
    smooth = s.rolling(win, center=True, min_periods=1).median()
    f.add_trace(go.Scatter(x=s.index, y=smooth, mode="lines", name=label,
                           line=dict(color=color, width=2),
                           hovertemplate=f"{label}<br>episode %{{x}}<br>滚动中位数 %{{y:.4f}}<extra></extra>"))
thresh_line(f)
f.update_layout(title="训练带来改善，但末期中位数仍远高于 0.20 阈值",
                xaxis_title="episode", yaxis_title="dist_ratio 滚动中位数",
                yaxis=dict(range=[0, 1.0]))
save("fig4", f, 460)

# 图 5：末期分布对比
f = go.Figure()
for name, label, color in (("results_best_2000.csv", "早期最佳 best_2000", GREEN),
                           ("v16_2000.csv", "最新长跑 v16_2000", BLUE)):
    s = pd.read_csv(ROOT / "data_b2" / name)["dist_ratio"]
    f.add_trace(go.Histogram(x=s, name=f"{label}（中位数 {s.median():.4f}）", opacity=0.62,
                             marker_color=color, nbinsx=45,
                             hovertemplate=f"{label}<br>ratio %{{x}}<br>次数 %{{y}}<extra></extra>"))
f.add_vline(x=THRESH, line_dash="dash", line_color=RED, line_width=2,
            annotation_text="0.20 阈值", annotation_font_color=RED)
f.update_layout(barmode="overlay", title="达标 episode 极少：v16_2000 仅 0.55%，早期最佳也只有 16.4%",
                xaxis_title="dist_ratio", yaxis_title="episode 数", xaxis=dict(range=[0, 1.2]))
save("fig5", f)

# 图 6：B1 多种子
f = go.Figure()
rng = np.random.default_rng(7)
for i, src in enumerate(("results.txt", "abl_trace.txt", "abl_reward.txt")):
    d = seeds[seeds.source == src]
    label = {"results.txt": "完整机制", "abl_trace.txt": "关闭资格迹",
             "abl_reward.txt": "关闭奖励调制"}[src]
    f.add_trace(go.Scatter(x=[i + 1 + rng.uniform(-0.11, 0.11) for _ in range(len(d))],
                           y=d.test_acc, mode="markers", name=f"{label}（10 种子）",
                           marker=dict(size=11, color=[BLUE, ORANGE, GREEN][i],
                                       line=dict(width=0.8, color="white")),
                           customdata=d.seed,
                           hovertemplate=label + "<br>seed %{customdata}<br>测试准确率 %{y:.2f}%<extra></extra>"))
f.add_hline(y=80, line_dash="dash", line_color=RED, line_width=2,
            annotation_text="PASS 阈值 80%", annotation_font_color=RED)
f.update_layout(title="B1：30 个种子全部 100%，连消融组也 100% —— 判据区分度不足",
                xaxis=dict(tickmode="array", tickvals=[1, 2, 3],
                           ticktext=["完整机制", "关闭资格迹", "关闭奖励调制"]),
                yaxis=dict(range=[0, 110], title="测试准确率 (%)"), showlegend=False)
save("fig6", f, 400)

# 图 7：数据质量问题计数
issues = [("无配对日志的 CSV", len(R["orphan_csvs"])), ("带质量标记的 B2 文件", len(Q["b2_bad_files"])),
          ("无最终结果的日志", len(Q["logs_without_final_results"])), ("孤立日志", len(R["orphan_logs"])),
          ("末行截断的文件", 4), ("ratio 偏差行（舍入放大）", Q["b2_ratio_mismatch_rows"]),
          ("reward 公式偏差行（版本差异）", Q["b2_reward_mismatch_rows"]),
          ("空 CSV", len(Q["b2_empty_files"])), ("B1 历史失败 CSV", S["b1_fail"])]
issues = sorted(issues, key=lambda x: x[1])
f = go.Figure(go.Bar(x=[v for _, v in issues], y=[k for k, _ in issues], orientation="h",
                     marker_color=[RED if k in ("空 CSV", "末行截断的文件", "无最终结果的日志") else BLUE
                                   for k, _ in issues],
                     text=[v for _, v in issues], textposition="outside",
                     hovertemplate="%{y}: %{x}<extra></extra>"))
f.update_layout(title="数据质量盘点：主数据完整，历史调试文件遗留较多",
                xaxis_title="数量", yaxis_title="", showlegend=False)
save("fig7", f, 440)

# ---------------------------------------------------------------- 结论数据
FINDINGS = [
    ("P0", "B2 突触数组容量估算不足 → 堆越界写",
     "brain_b2/benchmark2_core.c:167-173 按 avg_conn 估算 est；:183 固定 20 连接、:274/:294 各固定 12 连接，且所有 s_idx = synapse_count++ 无上限检查",
     "已动态复现：按当前源码重建 DLL，net2_init(36,200,10,10,avg_conn=3,seed=42) 后进程以 0xC0000374（STATUS_HEAP_CORRUPTION）退出；avg_conn=8 时正常（6168 synapses）",
     "按实际连接数精确计算容量或动态扩容；每次写入前检查 synapse_count < capacity"),
    ("P0", "可视化后端神经元数少算 PROP_N=56 → 越界写",
     "brain_b2/benchmark2_core.c:85 把 56 个本体感觉神经元计入 input_count；vis/backend_cpu.py:70 仍用 input+hidden+m1+m2 分配 3 个 double[N] 缓冲区；:79-84 分配",
     "已动态复现：按 N=256 分配而 C 写 312 个 double，net2_get_snapshot 后进程退出时 0xC0000005 访问冲突；vis/backend_vec.py:303 同样漏算",
     "由 C 侧返回真实 neuron_count（或让 topology 接口返回所需大小），Python 按返回值分配"),
    ("P0", "b2_core.dll 与当前源码 ABI 不同步",
     "b2_core.dll 时间 2026-11-10 01:46:35，源码 21:22–21:31；nm 显示仓库 DLL 缺 net2_inject_reward2（头文件 :189 已声明）；旧 DLL 小配置 neuron_count=256，重建 DLL=312",
     "ctypes 实测：仓库 DLL 无 net2_inject_reward2，且拓扑与当前 core 不一致；b2_debug.log 中 [REWARD_DEBUG] 打印在当前源码中已不存在",
     "重建并替换 DLL，加入构建脚本与版本戳；可视化启动前校验导出符号与结构版本"),
    ("P1", "benchmark2.exe 落后于最新 host 改动",
     "host_b2/benchmark2_host.cpp 21:31:48，benchmark2.exe 21:27:38（早 4 分 10 秒）；b2_core.o 21:27:37 介于 core.c 21:22:59 与 exe 之间",
     "当前 exe 不含最后一次 host 修改，直接用它复跑最新实验会得到旧行为",
     "统一构建入口，构建后写入 commit/源码哈希到日志；禁止手工多步编译"),
    ("P1", "无构建脚本、编译参数与超参不入日志",
     "全目录无 Makefile/CMake/bat/ps1；brain_b2/benchmark2_core.h:8 只有一行 g++ -O2 -D... 注释；host_b2/benchmark2_host.cpp:35、host/benchmark1_host.cpp:64、:256 硬编码 E:/shen_jing_yuan 绝对路径",
     "104 个 B2 CSV 中 54 个无配对日志，超参只能靠文件名 vN_* 猜；跨机器/跨目录不可复现",
     "补 Makefile/CMake；每次运行写 sidecar JSON（源码哈希、宏、seed、lr、wmax、目标模式）"),
    ("P1", "ctypes 测试脚本签名错误、文档与实现不符",
     "vis/test_ctypes.py:13,30 给 net2_init 传 6 参，头文件 :175-177 为 7 参（含 double avg_conn）；benchmark2_core.h:205-210 称 topology 返回所需大小/负值扩容，core.c:724-748 固定 return 0 且不检查缓冲区",
     "vis/backend_cpu.py:33-37 注释已记录过该 bug，但 test_ctypes.py 未修；调用者无法感知缓冲区不足",
     "统一签名与接口契约，补 ctypes 冒烟测试进构建流程"),
    ("P1", "B2 实验设计与注释/判据不一致",
     "host_b2/benchmark2_host.cpp:5 称随机生成 N 个目标，:47 TARGET_MODE 1 固定 (1.5,0.5)；:91-92 的 --disable_trace/--disable_reward 为空操作；:229 用 ratios[size/2]（上中位数）",
     "所有 103 次有效运行都在同一固定目标上评估，无泛化检验；消融只能靠编译期宏，PASS 判据无法隔离机制贡献",
     "评估阶段切换随机/多目标并单独统计；恢复运行时消融开关；中位数改用标准定义"),
    ("P1", "B1 消融对照无区分度",
     "data/multi_seed/results.txt、abl_trace.txt、abl_reward.txt 各 10 种子，测试准确率均 100.00%；判据仅 test_acc >= 80%（host/benchmark1_host.cpp:281）",
     "关闭资格迹或奖励调制后仍全部 100%，无法证明「延迟信用分配」机制有效；brain/benchmark1_core.c:387-453 确认开关确实生效",
     "提高任务难度/延迟，或改用学习速度、消融后收敛试次等有区分度的指标"),
    ("P2", "边界与数值稳定性缺陷",
     "core.c:109 /(grid_side-1)、:154/:164 /(motor_n-1)；host_b2:195/:226 final_dist/init_dist；host_b2:229/233-234 --episodes 0 时越界；data_b2/full_100ep.csv 第 3 行 init=0.0239 导致 ratio=5.77，历史文件最大 250.47",
     "极端参数会除零/UB；dist_ratio 在 init_dist≈0 时被放大，污染统计",
     "加参数校验；对 init_dist 设下限，或改用绝对末端距离为主指标"),
    ("P2", "复现性与状态重置不彻底",
     "core.c:16,83 全局 g_rand；net2_set_target 也消耗随机数；host_b2:124 L1/L2 未初始化；:130-134 只重置 arm 的 4 个字段，omega1/omega2/reward/max_reach 未重置",
     "网络实例间共享随机流，跨 episode 状态残留，结果对调用顺序敏感",
     "随机状态移入 NetworkState2；补全 episode 重置；加固定 seed 回归测试"),
    ("P2", "编译警告与可移植性",
     "g++ -Wall -Wextra：host_b2:83 nextd 未用、:124 L1/L2 未用、:213 缩进误导；host:77 nextd 未用；brain/benchmark1_core.c:477 未用参数；core.h:44-47 用 M_PI 但未定义 _USE_MATH_DEFINES",
     "MinGW gcc/g++ 16.1.0 编译通过（core 无警告），但 MSVC 下 M_PI 可能编译失败；警告掩盖真实问题",
     "清理死代码、显式定义 M_PI、开启 -Werror（至少对新增代码）"),
]

RECOMMENDATIONS = [
    ("立即", "修复 P0 内存安全", "修正突触容量估算并加写入边界检查；让 C 侧返回真实 neuron_count/synapse_count；重建 b2_core.dll 与 benchmark2.exe", "消除 2 个已复现的堆破坏/访问冲突"),
    ("立即", "建立单一构建入口", "新增 Makefile 或 build.ps1：gcc 编译 C 核心 → g++ 编译 host → 链接 → 生成 DLL；记录 gcc 版本、宏定义、源码哈希", "消除 exe/dll 与源码不同步"),
    ("本轮", "补齐实验元数据", "每次运行写 JSON sidecar（seed、episodes、ticks、lr 阶梯、wmax、sigma、TARGET_MODE、二进制哈希）；CSV 增加 run_id 列", "让 104 个 CSV 可追溯、可比较"),
    ("本轮", "修实验设计与判据", "评估阶段使用随机/多目标；恢复运行时消融；B1 增加有区分度的指标；中位数用标准定义；dist_ratio 增加 init_dist 下限", "让 PASS/FAIL 真正反映机制有效性"),
    ("随后", "清理与归档历史数据", "隔离或归档 mini*/full*/quick* 等调试文件，删除空文件与截断文件，补齐 8 个中断日志；统一 reward 裁剪规则", "减少误读，压缩 54 个孤立 CSV"),
    ("随后", "补回归与边界测试", "覆盖 episodes=0、motor_n=1、input_n=1、init_dist=0、小 hidden + 小 avg_conn；把 ctypes 冒烟测试纳入 CI", "防止内存问题回归"),
]

# ---------------------------------------------------------------- HTML
CSS = """
:root{--ink:#1d2634;--sub:#5b6779;--line:#e3e8ef;--bg:#f5f7fa;--card:#fff;--blue:#2f6fb2;--red:#b23a3a;--orange:#d97706;--green:#2e8b57}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:'Segoe UI','Microsoft YaHei',sans-serif;line-height:1.75}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 80px}
header{background:linear-gradient(120deg,#152238,#2f6fb2);color:#fff;padding:46px 24px 38px;margin-bottom:28px}
header .inner{max-width:1180px;margin:0 auto}header h1{margin:0 0 8px;font-size:30px;letter-spacing:.5px}
header p{margin:4px 0;color:#dce8f7;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:-52px 0 30px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:0 6px 18px rgba(20,40,70,.07)}
.kpi .label{font-size:12px;color:var(--sub);letter-spacing:.4px}.kpi .value{font-size:27px;font-weight:700;margin:4px 0}
.kpi .note{font-size:12px;color:var(--sub)}.kpi.bad .value{color:var(--red)}.kpi.warn .value{color:var(--orange)}.kpi.ok .value{color:var(--green)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px 28px;margin:22px 0;box-shadow:0 4px 14px rgba(20,40,70,.04)}
h2{font-size:21px;margin:0 0 6px;padding-bottom:10px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px;color:#2c3e57}p{font-size:14px;color:#33415a}
.lead{font-size:15px;color:#2c3e57;background:#f0f5fc;border-left:4px solid var(--blue);padding:12px 16px;border-radius:0 8px 8px 0}
.chart{margin:18px 0 8px;border:1px solid var(--line);border-radius:10px;padding:8px;overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0}th,td{border:1px solid var(--line);padding:9px 11px;text-align:left;vertical-align:top}
th{background:#f0f4fa;color:#2c3e57;font-weight:600}tr:nth-child(even) td{background:#fafbfd}
code{background:#f0f3f8;padding:1px 5px;border-radius:4px;font-family:Consolas,monospace;font-size:12px;color:#26364d}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;color:#fff}
.p0{background:var(--red)}.p1{background:var(--orange)}.p2{background:var(--blue)}.ok{background:var(--green)}
.tag{display:inline-block;padding:2px 8px;border-radius:5px;font-size:12px;background:#eef3fa;color:#2f6fb2;margin-right:6px}
ul,ol{font-size:14px;color:#33415a;padding-left:22px}li{margin:6px 0}
.small{font-size:12px;color:var(--sub)}footer{text-align:center;color:var(--sub);font-size:12px;margin-top:30px}
@media print{header{background:#152238!important;-webkit-print-color-adjust:exact}.kpi,section{box-shadow:none}section{break-inside:avoid}}
"""


def esc(x) -> str:
    return htmllib.escape(str(x))


def table(headers, rows) -> str:
    out = ["<table><thead><tr>"] + [f"<th>{esc(h)}</th>" for h in headers] + ["</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def badge(level: str) -> str:
    cls = {"P0": "p0", "P1": "p1", "P2": "p2"}[level]
    return f'<span class="badge {cls}">{level}</span>'


kpis = [
    ("B2 通过率", f"{S['b2_pass']} / {S['b2_pass'] + S['b2_fail']}", "阈值 median ≤ 0.20；无一达标", "bad"),
    ("B2 最佳中位数", f"{best.median_ratio:.4f}", f"{best.file}（阈值 0.20）", "warn"),
    ("B1 多种子通过率", f"{S['b1_seed_pass']} / {len(seeds)}", "全部 100%，含消融组", "ok"),
    ("已复现内存缺陷", "2", "堆损坏 + 访问冲突", "bad"),
    ("扫描数据规模", f"{Q['b2_csv_count'] + Q['b1_csv_count']} CSV", f"{Q['b2_total_episodes']:,} episode / {Q['b1_total_trials']:,} trial", ""),
]
kpi_html = "".join(
    f'<div class="kpi {cls}"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
    f'<div class="note">{esc(note)}</div></div>' for label, value, note, cls in kpis)

score_rows = [
    ["代码静态检测", '<span class="badge p0">不通过</span>', "11 项风险，其中 2 项已动态复现为内存破坏", "brain_b2/benchmark2_core.c、vis/backend_cpu.py"],
    ["构建状态", '<span class="badge p1">不通过</span>', "可编译且冒烟运行成功；但 DLL 落后约 19.7 小时、exe 落后 host 源码 4 分钟、无构建脚本", "gcc/g++ 16.1.0、nm 导出对比、时间戳"],
    ["数据与日志", '<span class="badge p1">部分通过</span>', "主数据无 NaN、无重复、公式自洽；历史文件有 1 空、4 截断、8 中断日志、54 孤立 CSV", "scan_results.json"],
    ["实验结果", '<span class="badge p0">B2 未达标</span>', f"B2 0 PASS / {S['b2_fail']} FAIL；B1 30/30 PASS 但消融组同为 100%", "PASS CHECK / PASS CRITERIA CHECK"],
]

body = []
body.append('<header><div class="inner"><h1>SNN 实验项目全面体检报告</h1>'
            f'<p>代码静态 · 构建状态 · 数据/日志 · 实验结果　|　检测时间 {esc(R["generated_at"])}</p>'
            '<p>只读检测：未修改任何实验数据；临时编译产物位于系统 TEMP 目录</p></div></header>')
body.append(f'<div class="wrap"><div class="kpis">{kpi_html}</div>')

body.append('<section id="s0"><h2>0　执行摘要</h2>'
            '<p class="lead">结论：项目可以编译、B1 基准全部通过，但 B2 到达任务在 103 次有效运行中从未达标；'
            '更严重的是当前源码存在两处已动态复现的内存破坏，且可视化 DLL 与源码 ABI 已脱节。'
            '建议先修 P0 内存安全与构建一致性，再继续调参。</p>'
            + table(["检测线", "结论", "要点", "主要证据"], score_rows)
            + '<h3>五个最关键的发现</h3><ol>'
            f'<li>B2 全部失败：{Q["b2_csv_count"]} 个 CSV 中 {S["b2_fail"]} 个 FAIL、1 个空文件、0 个 PASS；最佳中位数 '
            f'{best.median_ratio:.4f}（{esc(best.file)}），仍是阈值 0.20 的 {best.median_ratio / THRESH:.1f} 倍。</li>'
            f'<li>近期调参在回退：同为 500 episode 的可比子集里，v17=0.6390 → v18=0.6496 → v19=0.6776；'
            f'2000 episode 子集最佳为 {esc(best2000.file)} 的 {best2000.median_ratio:.4f}。</li>'
            '<li>当前源码编译出的 DLL 在 vis 小配置（36/200/10/10、avg_conn=3）下触发 0xC0000374 堆损坏；'
            '按旧 neuron_count 分配快照缓冲区则触发 0xC0000005 访问冲突。</li>'
            '<li>仓库 b2_core.dll 缺少 net2_inject_reward2 导出且不含 56 个本体感觉神经元，'
            '与 benchmark2_core.h / .c 已经不是同一套 ABI。</li>'
            '<li>B1 的 30 个种子（含两组消融）全部 100% 通过，说明 80% 准确率判据缺少区分度，'
            '不能作为「延迟信用分配有效」的证据。</li></ol>'
            '<p class="small">评级说明：P0 = 会导致错误结果或崩溃，须立即处理；P1 = 影响可信度/可复现性；P2 = 健壮性与维护性。</p>'
            '</section>')

body.append('<section id="s1"><h2>1　检测范围与方法</h2>'
            + table(["项目", "内容"],
                    [["数据来源", f'{Q["b2_csv_count"]} 个 data_b2 CSV + {Q["b2_log_count"]} 个 data_b2 日志；'
                                 f'{Q["b1_csv_count"]} 个 B1 CSV + 3 个多种子文本；{Q["root_log_count"]} 个根目录/其他日志'],
                     ["代码范围", "brain/、brain_b2/、host/、host_b2/ 的 C/C++ 源码与头文件；vis/ 的 4 个 Python 后端/测试脚本"],
                     ["构建验证", "MinGW-W64 gcc/g++ 16.1.0；-O2 -Wall -Wextra 分别编译 C 核心与 C++ host 并链接；另编译 -shared DLL；全部输出到 %TEMP%\\snn_audit"],
                     ["运行验证", "重建的 benchmark2.exe 跑 3 episode × 50 tick 冒烟；重建 DLL 用 ctypes 做 6 组配置初始化 + 缓冲区探针；benchmark1.exe 跑 --help"],
                     ["扫描器", "<code>audit/scan_data.py</code> 逐行解析 CSV、正则解析日志、检查产物时间戳与二进制标记，输出 <code>audit/scan_results.json</code>"],
                     ["复核", "<code>audit/inspect_results.py</code> 交叉核对 CSV 中位数与日志 FINAL RESULTS；<code>audit/check_ctypes.py</code> / <code>check_config.py</code> / <code>check_snapshot.py</code> 复现接口问题"]])
            + '<p>判据定义：B2 = <code>median(final_dist/init_dist) ≤ 0.20</code>（host_b2/benchmark2_host.cpp:243-251）；'
            'B1 = <code>后一半试次正确率 ≥ 80%</code>（host/benchmark1_host.cpp:279-289）。</p>'
            '<p class="small">未做：完整重跑 2000 episode 训练（耗时且会覆盖现有输出）；B1 完整运行（其诊断输出路径硬编码到 logs/）；'
            'GUI 可视化端到端运行（需 PySide6 显示环境）。</p></section>')

body.append('<section id="s2"><h2>2　代码静态检测</h2>'
            '<p class="lead">静态审阅共发现 11 项风险（P0 3 项、P1 5 项、P2 3 项）；其中 2 项内存问题已通过独立进程实测复现，不是理论推断。</p>'
            + table(["等级", "问题", "证据位置", "影响 / 复现结果", "修复方向"],
                    [[badge(lv), esc(title), f'<code>{esc(ev)}</code>', esc(impact), esc(action)]
                     for lv, title, ev, impact, action in FINDINGS])
            + '<h3>编译告警清单（-Wall -Wextra，gcc/g++ 16.1.0）</h3>'
            + table(["文件:行", "告警"],
                    [["host_b2/benchmark2_host.cpp:83", "nextd 定义但未使用"],
                     ["host_b2/benchmark2_host.cpp:124", "L1、L2 未使用（且未初始化）"],
                     ["host_b2/benchmark2_host.cpp:213", "if 缩进误导：satLO 判断与 satHI 写在同一行"],
                     ["host/benchmark1_host.cpp:77", "nextd 定义但未使用"],
                     ["brain/benchmark1_core.c:477", "lazy_decay_weights 的 net 参数未使用"],
                     ["brain_b2/benchmark2_core.c", "无告警（gcc -Wall -Wextra 编译通过）"]])
            + '</section>')

art = R["artifacts"]
body.append('<section id="s3"><h2>3　构建状态检测</h2>'
            '<p class="lead">构建能力本身健康：当前 B1/B2 源码都能用 gcc/g++ 16.1.0 编译、链接并运行；'
            '问题在产物同步与可复现性——可执行文件、DLL 与源码不在同一版本。</p>'
            '<h3>3.1　关键产物时间线</h3>'
            + table(["产物 / 源码", "修改时间", "字节", "说明"],
                    [[f'<code>{esc(k)}</code>', esc(v["mtime"]), f'{v["bytes"]:,}',
                      "源码最新（晚于 exe）" if k.endswith("host.cpp") else
                      ("DLL 落后约 19.7 小时" if k.endswith(".dll") else
                       ("exe 早于最新 host 改动" if k == "benchmark2.exe" else ""))]
                     for k, v in art.items() if k.startswith(("brain_b2", "host_b2", "benchmark2", "b2_core"))])
            + '<h3>3.2　实测结果</h3>'
            + table(["检测项", "结果"],
                    [["B2 编译", "core.c 无告警 → host.cpp 4 个告警 → 链接成功，生成 benchmark2_audit.exe（125,853 B）"],
                     ["B2 冒烟运行", "3 episode × 50 tick：Net 2256 neurons / 44300 synapses；正常退出，median 0.9540 → FAIL（短训练下符合预期）"],
                     ["B1 编译", "core.c 1 个告警、host.cpp 1 个告警；链接成功；--help 正常输出"],
                     ["重建 DLL 默认配置", "net2_init(100,2000,50,50,8,42) → 2256 neurons / 44401 synapses，与 v16_2000.log 完全一致（说明该数据来自当前 core；44300 的差异仅因冒烟 seed 不同）"],
                     ["DLL 导出符号", "仓库 b2_core.dll 缺 <code>net2_inject_reward2</code>；重建 DLL 具备。旧 DLL 小配置 neuron_count=256，重建 DLL=312（+PROP_N 56）"],
                     ["小配置初始化", "36/200/10/10 + avg_conn=3 → 0xC0000374 堆损坏；avg_conn=8 → 6168 synapses 正常；avg_conn=20 → 8590 正常"],
                     ["vis 快照缓冲区", "按 backend_cpu.py 的 N=256 分配，C 侧写 312 个 double ×3 → 进程退出时 0xC0000005"],
                     ["二进制标记", "benchmark2.exe 含 wstats / PASS CHECK，不含旧 REWARD_DEBUG；b2_debug.log 仍留有旧调试打印（历史二进制产物）"]])
            + '<p class="small">编译命令（复现）：<code>gcc -O2 -Wall -Wextra -c brain_b2/benchmark2_core.c</code> → '
            '<code>g++ -O2 -Wall -Wextra -c host_b2/benchmark2_host.cpp</code> → '
            '<code>g++ core.o host.o -o benchmark2.exe -lm</code>；DLL 用 <code>gcc -O2 -shared ... -lm</code>。'
            '仓库中没有任何 Makefile/CMake/批处理脚本，且 3 处输出路径硬编码为 E:/shen_jing_yuan。</p></section>')

body.append('<section id="s4"><h2>4　数据与日志检测</h2>'
            '<p class="lead">正式实验数据（B1 多种子、B2 近期 v10–v19）结构完整、无 NaN、无重复、'
            'ratio 与 reward 公式自洽；问题集中在历史调试文件与「数据—日志」配对缺失。</p>'
            '<div class="chart">' + figs["fig7"] + '</div>'
            '<h3>4.1　完整性问题（确认为真实缺陷）</h3>'
            + table(["问题", "数量", "明细"],
                    [["空 CSV", 1, "results_delta.csv（0 字节）"],
                     ["CSV 末行截断 / 列数异常", 4, "results_2k.csv、v3_2000.csv、v6_2000.csv、v6_500.csv（如 v6_500.csv 末行仅 5 列）"],
                     ["日志无 FINAL RESULTS（运行中断）", 8, ", ".join(Q["logs_without_final_results"])],
                     ["CSV 无配对日志", len(R["orphan_csvs"]), "54 个，多为 results_*/mini*/v1x 短跑，超参只能靠文件名猜测"],
                     ["日志无配对 CSV", len(R["orphan_logs"]), ", ".join(R["orphan_logs"])],
                     ["CSV 与日志 episode 数不符", 5, "quick500.csv、tracefix2_500.csv、v3_2000.csv、v6_2000.csv、v6_500.csv"],
                     ["CSV 与日志中位数不符", 1, "v3_500.csv（中断/版本差异）"],
                     ["B1 历史失败 CSV", 3, "abl_noreward.csv、abl_notrace.csv、full_results.csv（predicted_vote 全 0，准确率 0%）"]])
            + '<h3>4.2　「疑似异常」的澄清（避免误报）</h3>'
            '<ul>'
            f'<li><b>{Q["b2_ratio_mismatch_rows"]} 行 ratio 与 final/init 偏差：</b>全部 init_dist &lt; 1，其中 45 行 &lt; 0.1。'
            'CSV 只保留 4 位小数，小分母把舍入误差放大（例：init=0.0239、final=0.1377、ratio=5.7727 而 5.7615）。'
            '属于指标设计缺陷，不是数据损坏。</li>'
            f'<li><b>{Q["b2_reward_mismatch_rows"]} 行 reward 公式偏差：</b>其中 92 行是旧版本把负奖励裁剪为 0'
            '（当前公式为 1 − final_dist/2）；另有旧版随机目标文件（mini*.csv、full*.csv）含多个 target。'
            '属于版本演化，跨版本 CSV 不宜直接混合统计。</li>'
            '<li><b>b2_debug.log 的 PowerShell NativeCommandError：</b>是 stderr 被 PowerShell 转录为错误记录，非程序崩溃；'
            '其中 [REWARD_DEBUG] 打印来自旧二进制，当前源码已无该代码。</li>'
            '<li><b>B1 3 个 0% CSV：</b>输出发放率恒为 1.0000、predicted_vote 全 0，是早期失败/消融遗留，'
            '不应与 data/multi_seed/ 的 30 个正式种子实验合并计算。</li></ul>'
            + '<h3>4.3　结构一致性（全部通过）</h3>'
            + table(["检查", "结果"],
                    [["B2 CSV 列头", "103/103 有效文件为 7 列标准头（另 1 个空文件）"],
                     ["B1 CSV 列头", "38/38 为 9 列标准头"],
                     ["B1 correct 逻辑", "19,060 行全部满足 correct = (predicted == (stim==0 ? 1 : 2))，0 行不一致"],
                     ["NaN / Inf / 负值 / 重复 trial", "0（B1 与 B2 有效行）"],
                     ["reward 公式（当前版本）", "近期 v10–v19 全部满足 1 − final_dist/2，误差 < 0.002"],
                     ["目标与初始距离", "近期版本均为固定目标 (1.5, 0.5)、init_dist=0.7071，与 TARGET_MODE 1 一致"]])
            + '</section>')

body.append('<section id="s5"><h2>5　实验结果检测</h2>'
            '<p class="lead">B1 达标但判据过松；B2 全面未达标，且近期版本没有形成持续改善趋势。'
            '不同 episode 数、不同目标模式的 median 不可直接比较，因此下面按 500 / 2000 episode 分层分析。</p>'
            '<h3>5.1　B2 全量：103 次有效运行，0 次达标</h3>'
            '<div class="chart">' + figs["fig1"] + '</div>'
            f'<p>累计 {Q["b2_total_episodes"]:,} 个 episode；最佳中位数 {best.median_ratio:.4f}（{esc(best.file)}），'
            f'最新长跑 v16_2000 为 0.5035，最新短跑 v19_decay 为 {latest.median_ratio:.4f}。'
            f'500 ep 子集中位数 {med500:.4f}，2000 ep 子集中位数 {med2000:.4f}。'
            '另有 25 个根目录日志中 22 个 FAIL、3 个中断无结论（b2_run*.log）。</p>'
            '<h3>5.2　可比子集：近期调参连续回退</h3>'
            '<div class="chart">' + figs["fig2"] + '</div>'
            '<p>在同样 500 episode 的子集中，v17_noise0（0.6390）→ v18_psp35（0.6496）→ v19_decay（0.6776）'
            '连续三次变差；v13_prop、v14_in144 也在 0.68 附近。说明最近一组「噪声/PSP/衰减」改动没有带来收益。</p>'
            '<h3>5.3　长训练子集：中位数改善，但没有稳定掌握</h3>'
            '<div class="chart">' + figs["fig3"] + '</div>'
            + table(["运行", "episode", "中位数", "均值", "≤0.20 的 episode 占比", "首尾十分位改善"],
                    [[esc(r.file), f'{int(r.n):,}', f'{r.median_ratio:.4f}', f'{r.mean_ratio:.4f}',
                      f'{r.frac_le_target:.2%}', f'{r.learning_delta:+.4f}']
                     for _, r in c2000.sort_values("mtime").iterrows()])
            + '<p>早期 <code>results_best_2000.csv</code> 的中位数 0.4398 仍是长训练（≥1500 episode）子集的最佳，且 16.4% 的 episode 达到阈值；'
            '最新的 <code>v16_2000.csv</code> 虽然中位数 0.5035 优于若干中期版本，但只有 0.55% 的 episode 达标。'
            '也就是说「中位数变好」主要来自整体分布左移，而不是学会稳定到达。</p>'
            '<h3>5.4　学习曲线与末期分布</h3>'
            '<div class="chart">' + figs["fig4"] + '</div>'
            '<div class="chart">' + figs["fig5"] + '</div>'
            '<p>v16_2000 的首个十分位中位数 0.6659 降到末个十分位 0.4653（改善 0.2006），证明学习信号有效；'
            '但曲线在 0.45 附近进入平台，且末期中位数仍高于早期 best_2000 的 0.3951。'
            '结合权重饱和日志（v16_2000.log:2043 的 max=9.958、satHI=366，WEIGHT_MAX=10.0），'
            '提示权重上限与学习率阶梯可能是平台期的来源之一（相关推断，非因果结论）。</p>'
            '<h3>5.5　B1：全部通过，但消融对照失效</h3>'
            '<div class="chart">' + figs["fig6"] + '</div>'
            '<p>30 个种子（完整机制 / 关闭资格迹 / 关闭奖励调制各 10 个）测试准确率均为 100.00%，零方差。'
            'brain/benchmark1_core.c:387-453 确认运行时开关确实生效，因此更可能是任务本身太容易或 80% 阈值过低。'
            '按 data-validation 的红旗规则，「结果完美一致」应触发复核，而不是直接作为机制有效的证据。</p>'
            '</section>')

body.append('<section id="s6"><h2>6　处置建议（按优先级）</h2>'
            + table(["时机", "动作", "具体做法", "预期收益"],
                    [[esc(a), f'<b>{esc(b)}</b>', esc(c), esc(d)] for a, b, c, d in RECOMMENDATIONS])
            + '</section>')

body.append('<section id="s7"><h2>7　限制、假设与置信度</h2>'
            '<ul>'
            f'<li><b>快照时间：</b>所有结论基于 {esc(R["generated_at"])} 的工作区状态；此后新增运行不在范围内。</li>'
            '<li><b>可比性：</b>104 个 B2 CSV 来自不同源码版本、超参、episode 数与目标模式；'
            '中位数只能在同一 episode 分层内作趋势参考，不能当作严格对照实验。</li>'
            '<li><b>超参缺失：</b>CSV/日志不记录 lr 阶梯、wmax、sigma、noise、TARGET_MODE 与二进制哈希，'
            '版本归因依赖文件名与源码时间推断，存在不确定性。</li>'
            '<li><b>误报排除：</b>47 行 ratio 偏差、94 行 reward 偏差已定位为小分母舍入与旧版裁剪规则，未计入「数据损坏」。</li>'
            '<li><b>未复现项：</b>P0-3（DLL 落后）由时间戳 + 导出符号 + 神经元数三方交叉验证；'
            '其余 P1/P2 静态风险为代码审阅结论，未逐项动态触发。</li>'
            '<li><b>副作用：</b>检测全程只读实验数据；新建文件仅位于 <code>Experiment/audit/</code>，'
            '临时编译与探针输出位于 <code>%TEMP%\\snn_audit</code>。</li>'
            '<li><b>置信度：</b>数据质量与实验结果——高（全量逐行扫描 + 日志交叉核对）；'
            '构建与 ABI 结论——高（实际编译/链接/ctypes 实测）；静态风险归因——中。</li></ul>'
            '<h3>复现方式</h3>'
            '<p><code>py audit/scan_data.py</code> → <code>py audit/inspect_results.py</code> → '
            '<code>py audit/build_report.py</code>；接口实测脚本为 '
            '<code>check_ctypes.py</code>、<code>check_config.py</code>、<code>check_snapshot.py</code>。'
            '注意本机 <code>python</code> 是 0 字节的 Windows 应用别名，需使用 <code>py</code> 启动器。</p>'
            '</section>')

body.append('<footer>本报告由只读自动检测生成 · 图表使用 Plotly 交互渲染 · 数据/结论可依第 7 节复现</footer></div>')

html_doc = ("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>SNN 实验项目全面体检报告</title><style>" + CSS + "</style>"
            "<script>" + get_plotlyjs() + "</script></head><body>" + "".join(body) + "</body></html>")
(HERE / "full_report.html").write_text(html_doc, encoding="utf-8")

# ---------------------------------------------------------------- Markdown
def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |")
    return "\n".join(out)


md = []
md.append("# SNN 实验项目全面体检报告\n")
md.append(f"> 检测时间：{R['generated_at']}　|　范围：代码静态 · 构建状态 · 数据/日志 · 实验结果　|　"
          "只读检测，未修改实验数据\n")
md.append("> 交互式可视化完整版：`audit/full_report.html`（7 张 Plotly 图表，含下钻与悬停明细）\n")
md.append("## 0　执行摘要\n")
md.append("**结论：项目可编译、B1 基准全部通过，但 B2 到达任务 103 次有效运行从未达标；"
          "当前源码有 2 处已动态复现的内存破坏，且可视化 DLL 与源码 ABI 已脱节。"
          "建议先修 P0 内存安全与构建一致性，再继续调参。**\n")
md.append(md_table(["检测线", "结论", "要点"],
                   [["代码静态", "不通过", "11 项风险（P0 3 / P1 5 / P2 3），2 项已复现为堆损坏与访问冲突"],
                    ["构建状态", "不通过", "可编译可运行；DLL 落后约 19.7 小时、exe 落后 host 源码 4 分钟、无构建脚本"],
                    ["数据与日志", "部分通过", "主数据完整；1 空文件、4 截断、8 中断日志、54 孤立 CSV"],
                    ["实验结果", "B2 未达标 / B1 达标（有保留）", f"B2 0 PASS / {S['b2_fail']} FAIL；B1 30/30 全 100% 且消融组同为 100%"]]))
md.append("\n五个关键发现：\n")
md.append(f"1. B2 全部失败：{Q['b2_csv_count']} 个 CSV 中 {S['b2_fail']} FAIL、1 空文件、0 PASS；"
          f"最佳中位数 {best.median_ratio:.4f}（{best.file}），为阈值 0.20 的 {best.median_ratio / THRESH:.1f} 倍。\n"
          f"2. 同为 500 episode 的可比子集，v17=0.6390 → v18=0.6496 → v19=0.6776 连续回退；"
          f"2000 episode 子集最佳 {best2000.file} {best2000.median_ratio:.4f}。\n"
          "3. 当前源码 DLL 在 vis 小配置（36/200/10/10, avg_conn=3）触发 0xC0000374 堆损坏；"
          "按旧神经元数分配快照缓冲区触发 0xC0000005 访问冲突。\n"
          "4. 仓库 `b2_core.dll` 缺 `net2_inject_reward2` 导出、不含 56 个本体感觉神经元，与当前源码 ABI 不一致。\n"
          "5. B1 的 30 个种子（含两组消融）全部 100% 通过，80% 判据缺少区分度。\n")

md.append("## 1　检测范围与方法\n")
md.append(f"- 数据：{Q['b2_csv_count']} 个 B2 CSV + {Q['b2_log_count']} 个日志、{Q['b1_csv_count']} 个 B1 CSV + "
          f"3 个多种子文本、{Q['root_log_count']} 个根目录/其他日志；"
          f"共 {Q['b2_total_episodes']:,} episode / {Q['b1_total_trials']:,} trial。\n"
          "- 代码：`brain*/`、`host*/` 的 C/C++ 源码与头文件，`vis/` 的 Python 后端与测试脚本。\n"
          "- 构建：MinGW-W64 gcc/g++ 16.1.0，`-O2 -Wall -Wextra` 分别编译 C 核心与 C++ host 后链接，另编译 `-shared` DLL；输出到 `%TEMP%\\snn_audit`。\n"
          "- 运行：B2 冒烟（3 ep × 50 tick）、ctypes 6 组配置 + 缓冲区探针、B1 `--help`。\n"
          "- 判据：B2 `median(final_dist/init_dist) ≤ 0.20`；B1 后一半试次正确率 `≥ 80%`。\n"
          "- 复现：`py audit/scan_data.py` → `py audit/inspect_results.py` → `py audit/build_report.py`。"
          "本机 `python` 是 0 字节 Windows 应用别名，必须用 `py`。\n")

md.append("## 2　代码静态检测\n")
md.append(md_table(["等级", "问题", "证据 / 复现结果", "修复方向"],
                   [[lv, title, ev + "；" + impact, action] for lv, title, ev, impact, action in FINDINGS]))
md.append("\n编译告警：B2 host 4 处（:83 `nextd` 未用、:124 `L1/L2` 未用、:213 缩进误导）、"
          "B1 host 1 处（:77）、B1 core 1 处（:477）；B2 core 无告警。\n")

md.append("## 3　构建状态检测\n")
md.append(md_table(["产物 / 源码", "修改时间", "说明"],
                   [[k, v["mtime"], "源码最新（晚于 exe）" if k.endswith("host.cpp") else
                     ("DLL 落后约 19.7 小时" if k.endswith(".dll") else
                      ("exe 早于最新 host 改动" if k == "benchmark2.exe" else ""))]
                    for k, v in art.items() if k.startswith(("brain_b2", "host_b2", "benchmark2", "b2_core"))]))
md.append("\n实测：\n")
md.append("- B1/B2 均可编译链接；B2 冒烟运行正常（2256 neurons / 44300 synapses，median 0.9540 FAIL，短训练符合预期）。\n"
          "- 重建 DLL 默认配置 `net2_init(100,2000,50,50,8,42)` → 2256 neurons / 44401 synapses，"
          "与 `v16_2000.log` 完全一致（冒烟 44300 的差异仅因 seed 不同，不是版本不同）。\n"
          "- 仓库 DLL 缺 `net2_inject_reward2`；旧 DLL 小配置 256 neurons，重建 DLL 312（+PROP_N 56）。\n"
          "- `36/200/10/10 + avg_conn=3` → 0xC0000374；`avg_conn=8/20` → 正常（6168 / 8590 synapses）。\n"
          "- `vis/backend_cpu.py` 按 N=256 分配而 C 写 312 个 double ×3 → 0xC0000005。\n"
          "- 无 Makefile/CMake/批处理；3 处硬编码 `E:/shen_jing_yuan` 输出路径。\n")

md.append("## 4　数据与日志检测\n")
md.append("正式实验数据结构完整、无 NaN / 重复 / 负值，近期 CSV 的 ratio 与 reward 公式自洽；问题集中在历史调试文件。\n")
md.append(md_table(["问题", "数量", "明细"],
                   [["空 CSV", 1, "results_delta.csv"],
                    ["CSV 末行截断 / 列数异常", 4, "results_2k.csv、v3_2000.csv、v6_2000.csv、v6_500.csv"],
                    ["日志无 FINAL RESULTS", 8, ", ".join(Q["logs_without_final_results"])],
                    ["CSV 无配对日志", len(R["orphan_csvs"]), "多为 results_*/mini*/v1x 短跑，超参只能靠文件名猜测"],
                    ["日志无配对 CSV", len(R["orphan_logs"]), ", ".join(R["orphan_logs"])],
                    ["CSV 与日志 episode 数不符", 5, "quick500.csv、tracefix2_500.csv、v3_2000.csv、v6_2000.csv、v6_500.csv"],
                    ["CSV 与日志中位数不符", 1, "v3_500.csv"],
                    ["B1 历史失败 CSV", 3, "abl_noreward.csv、abl_notrace.csv、full_results.csv（准确率 0%）"]]))
md.append("\n**误报澄清（重要）**：\n")
md.append(f"- {Q['b2_ratio_mismatch_rows']} 行 ratio 偏差全部发生在 init_dist < 1（45 行 < 0.1），"
          "是 4 位小数舍入被小分母放大，属指标缺陷而非数据损坏。\n"
          f"- {Q['b2_reward_mismatch_rows']} 行 reward 偏差中 92 行是旧版把负奖励裁剪为 0；"
          "旧版 `mini*.csv`/`full*.csv` 使用随机目标。属版本演化，跨版本不宜混合统计。\n"
          "- `b2_debug.log` 的 PowerShell `NativeCommandError` 是 stderr 转录，非崩溃；"
          "其中 `[REWARD_DEBUG]` 打印来自旧二进制。\n"
          "- B1 三个 0% CSV 是早期失败遗留，不应与 30 个正式种子实验合并。\n")
md.append("\n结构一致性检查全部通过：B2 103/103 有效文件标准 7 列（另 1 空文件）、B1 38/38 标准 9 列；"
          "19,060 行 B1 `correct` 逻辑 0 不一致；近期 v10–v19 的 reward 公式误差 < 0.002；"
          "目标与 init_dist 与 TARGET_MODE 1 一致。\n")

md.append("## 5　实验结果检测\n")
md.append(f"B2 累计 {Q['b2_total_episodes']:,} episode，0 PASS / {S['b2_fail']} FAIL；"
          f"最佳 {best.median_ratio:.4f}（{best.file}），最新长跑 v16_2000 0.5035，"
          f"最新短跑 v19_decay {latest.median_ratio:.4f}；500 ep 子集中位数 {med500:.4f}，"
          f"2000 ep 子集中位数 {med2000:.4f}。根目录 25 个日志：22 FAIL、3 中断。\n")
md.append(md_table(["2000-episode 运行", "中位数", "均值", "≤0.20 占比", "首尾十分位改善"],
                   [[r.file, f"{r.median_ratio:.4f}", f"{r.mean_ratio:.4f}",
                     f"{r.frac_le_target:.2%}", f"{r.learning_delta:+.4f}"]
                    for _, r in c2000.sort_values("mtime").iterrows()]))
md.append("\n- **近期回退**：同为 500 ep，v17 0.6390 → v18 0.6496 → v19 0.6776；v13/v14 也在 0.68 附近。\n"
          "- **平台期**：v16_2000 首十分位 0.6659 → 末十分位 0.4653（改善 0.2006），但只有 0.55% 的 episode 达标；"
          "早期 results_best_2000 中位数 0.4398、16.4% 达标、末十分位 0.3951，仍优于近期长跑。\n"
          "- **权重饱和线索**：v16_2000.log:2043 记录 `max=9.958 satHI=366`（WEIGHT_MAX=10.0）；"
          "与平台期相关但不足以证明因果。\n"
          "- **B1**：30 个种子（完整/关闭资格迹/关闭奖励调制各 10）测试准确率全部 100.00%，零方差；"
          "core 中运行时开关确实生效，说明任务难度或 80% 阈值不足以区分机制。\n")
md.append("可比性说明：不同 episode 数、超参、目标模式的 median 不可直接比较，"
          "报告按 500 ep / 2000 ep 分层；超参未入日志，版本归因依赖文件名与源码时间。\n")

md.append("## 6　处置建议\n")
md.append(md_table(["时机", "动作", "具体做法", "预期收益"],
                   [[a, b, c, d] for a, b, c, d in RECOMMENDATIONS]))

md.append("\n## 7　限制、假设与置信度\n")
md.append(f"- 快照时间 {R['generated_at']}；此后运行不在范围内。\n"
          "- 104 个 B2 CSV 来自不同版本/超参/目标模式，趋势仅作参考，不是严格对照实验。\n"
          "- 未记录 lr 阶梯、wmax、sigma、noise、TARGET_MODE 与二进制哈希，版本归因存在不确定性。\n"
          "- 47 行 ratio 偏差、94 行 reward 偏差已定位为舍入/旧版裁剪，未计为数据损坏。\n"
          "- P0-1 / P0-2 已动态复现；P0-3 由时间戳 + 导出符号 + 神经元数交叉验证；其余 P1/P2 为静态审阅结论。\n"
          "- 未做：完整 2000 ep 重跑、B1 完整运行（诊断路径硬编码会覆盖 logs/）、GUI 端到端。\n"
          "- 置信度：数据质量与实验结果 高；构建与 ABI 结论 高；静态风险归因 中。\n"
          "- 副作用：仅新建 `Experiment/audit/` 下报告与脚本，临时编译产物在 `%TEMP%\\snn_audit`。\n")
md.append("\n---\n\n*图表与逐项明细请查看交互式报告 `audit/full_report.html`；"
          "原始数据见 `audit/scan_results.json`，复核输出见 `audit/inspect.txt`。*\n")

(HERE / "检测报告.md").write_text("\n".join(md), encoding="utf-8")

print("WROTE", HERE / "full_report.html", (HERE / "full_report.html").stat().st_size, "bytes")
print("WROTE", HERE / "检测报告.md", (HERE / "检测报告.md").stat().st_size, "bytes")
print("CHARTS", len(figs), sorted(figs))
print("CHECK", {"b2_pass": S["b2_pass"], "b2_fail": S["b2_fail"], "best": best.median_ratio,
                "best_file": best.file, "b1_seed_pass": S["b1_seed_pass"],
                "median_500": round(float(med500), 4), "median_2000": round(float(med2000), 4)})
