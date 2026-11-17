# -*- coding: utf-8 -*-
"""AI 开发进度仪表盘生成器。

把 `audit/ai_dev_progress_plan_2026-11-13.md` 的表格、真实 git 时间戳与
冻结判据里可核对的数字，装配成一个自包含、可离线打开的交互式 HTML 仪表盘。

运行： py -3 Experiment/tools/gen_ai_dev_dashboard.py
输出： Experiment/audit/ai_dev_progress_plan_2026-11-13.html

设计约束：单一强调色、无外部资源、无占位图、无 emoji、只动 transform/opacity、
尊重 prefers-reduced-motion、表格有筛选与空状态、图表由 plotly 内嵌。
"""
import html
import re
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PLAN_MD = ROOT / "audit" / "ai_dev_progress_plan_2026-11-13.md"
OUT = ROOT / "audit" / "ai_dev_progress_plan_2026-11-13.html"

C = {
    "bg": "#0b1220", "card": "#131f33", "line": "#24344f",
    "text": "#e6edf7", "muted": "#94a8c4", "accent": "#4cc2f1",
    "ok": "#3ddc97", "warn": "#ffc857", "bad": "#ff6b6b",
    "grid": "#1e2c44",
}

# 来自 progress_review_2026-11-12.md §9.3–§9.7 与 judgments.md §6.4–§6.8。
# 每项：判据版本、章节、主实验 g=0 正确率、同 seed 配对「完整 − 无迹」pp。
ROUNDS = [
    ("v1.4 · §9.3", "严格对称初始读出", 0.00, 0.00),
    ("v1.5 · §9.4", "两必要条件析因", 0.00, 0.00),
    ("v1.6 · §9.5", "通路选择性第三因子", 0.00, 0.00),
    ("v1.7 · §9.6", "可学习内部前向模型", 0.00, 0.00),
    ("v1.8 · §9.7", "时序差分自预测", 30.00, 30.00),
]


def md_tables(text):
    """按 ### 标题收集 Markdown 表格：{标题: [[表头, 行...], ...]}。"""
    out, cur, lines, i = {}, None, text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            cur = line[4:].strip()
        if line.startswith("|") and cur:
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = []
            for ln in block:
                cells = [x.strip() for x in ln.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells):
                    continue
                rows.append(cells)
            if rows:
                out.setdefault(cur, []).append(rows)
            continue
        i += 1
    return out


def first_table(tables, key):
    for heading, blocks in tables.items():
        if key in heading:
            return blocks[0][0], blocks[0][1:]
    raise RuntimeError("未在计划文档中找到表格：" + key)


def clean(s):
    return html.escape(s.replace("**", "").replace("`", "").strip())


def status_of(s):
    if any(k in s for k in ("FAIL", "不通过", "阻塞", "未开始")):
        return "bad"
    if any(k in s for k in ("PASS", "已完成", "达成", "通过")):
        return "ok"
    if any(k in s for k in ("待执行", "部分", "PARTIAL", "建议", "条件")):
        return "warn"
    return "neutral"


def html_table(headers, rows, cls="", status_col=None, filterable=False):
    th = "".join("<th>%s</th>" % clean(h) for h in headers)
    body = []
    for r in rows:
        cells = []
        for idx, cell in enumerate(r):
            if status_col is not None and idx == status_col:
                cls_s = status_of(cell)
                cells.append('<td><span class="tag %s">%s</span></td>' % (cls_s, clean(cell)))
            else:
                cells.append("<td>%s</td>" % clean(cell))
        attrs = ""
        if filterable and status_col is not None and status_col < len(r):
            attrs = ' data-status="%s"' % status_of(r[status_col])
        body.append("<tr%s>%s</tr>" % (attrs, "".join(cells)))
    return ('<table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table>'
            % (cls, th, "".join(body)))


def fig_layout(fig, height=330, title=""):
    fig.update_layout(
        template="plotly_dark", title=dict(text=title, font=dict(size=15)),
        paper_bgcolor=C["card"], plot_bgcolor=C["card"], height=height,
        font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=12, color=C["text"]),
        margin=dict(l=70, r=24, t=52, b=48),
        legend=dict(orientation="h", y=1.02, x=1, yanchor="bottom", xanchor="right",
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        hoverlabel=dict(bgcolor="#0b1220", font_size=12),
    )
    return fig


def build_figures(tables, dates):
    figs = {}

    # 1) 里程碑完成度
    hdr, rows = first_table(tables, "1.2 里程碑完成度")
    names = ["%s %s" % (r[0], r[1]) for r in rows]
    vals = [float(re.sub(r"[^\d.]", "", r[6])) for r in rows]
    detail = ["DONE %s / PARTIAL %s / MISSING %s" % (r[3], r[4], r[5]) for r in rows]
    colors = [C["ok"] if v >= 70 else C["warn"] if v >= 30 else C["bad"] for v in vals]
    f = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker_color=colors,
        text=["%.0f%%" % v for v in vals], textposition="outside",
        customdata=detail,
        hovertemplate="%{y}<br>完成度 %{x:.0f}%<br>%{customdata}<extra></extra>",
    ))
    f.add_vline(x=100, line=dict(color=C["muted"], dash="dot"),
                annotation_text="全部交付物齐备", annotation_position="top")
    f.update_xaxes(range=[0, 112], title="交付物覆盖率（DONE=1 / PARTIAL=0.5 / MISSING=0）")
    f.update_yaxes(autorange="reversed")
    figs["milestones"] = fig_layout(f, 360, "里程碑完成度 · 覆盖率，不是通过率")

    # 2) T1 五轮：g=0 正确率与配对归因
    labels = [r[0].split(" · ")[0] for r in ROUNDS]
    desc = ["%s · %s" % (r[0], r[1]) for r in ROUNDS]
    acc = [r[2] for r in ROUNDS]
    pp = [r[3] for r in ROUNDS]
    f = go.Figure()
    f.add_trace(go.Bar(x=labels, y=acc, name="主实验 g=0 正确率 (%)",
                       marker_color=C["accent"], text=["%.0f" % v for v in acc],
                       textposition="outside", customdata=desc,
                       hovertemplate="%{x}<br>%{customdata}<br>g=0 %{y:.2f}%<extra></extra>"))
    f.add_trace(go.Scatter(x=labels, y=pp, name="配对「完整 − 无迹」(pp)",
                           mode="lines+markers", line=dict(color=C["warn"], width=2.5),
                           marker=dict(size=9),
                           hovertemplate="%{x}<br>配对 %{y:.2f}pp<extra></extra>"))
    f.add_hline(y=80, line=dict(color=C["bad"], dash="dash"),
                annotation_text="B1 正确率门槛 80%", annotation_position="top left")
    f.add_hline(y=30, line=dict(color=C["ok"], dash="dot"),
                annotation_text="归因门槛 30pp", annotation_position="bottom left")
    f.update_yaxes(range=[0, 105], title="百分数 / 百分点")
    figs["rounds"] = fig_layout(f, 350, "T1 连续五轮 · v1.8 首次出现可归因提升，但仍未达 80%")
    figs["rounds"].update_layout(
        margin=dict(l=70, r=24, t=52, b=88),
        legend=dict(orientation="h", y=-0.24, x=0.5, xanchor="center", yanchor="top",
                    bgcolor="rgba(0,0,0,0)"))

    # 3) 提交时间线（真实 git 时间戳）
    days = sorted(d for d in dates if d >= "2026-11-09")
    counts = [dates[d] for d in days]
    f = go.Figure(go.Bar(x=days, y=counts, marker_color=C["accent"],
                         text=[str(v) for v in counts], textposition="outside",
                         hovertemplate="%{x}<br>提交 %{y} 次<extra></extra>"))
    f.update_yaxes(title="提交数", rangemode="tozero")
    f.update_xaxes(title="日期（git log --date=short）")
    figs["timeline"] = fig_layout(f, 300, "提交密度 · AI 开发的真实时间线")

    # 4) 近期计划甘特（只画有日期的项；阻塞项单独列表，不伪造日期）
    tasks = [
        ("时序差分自预测（已完成）", 0, 1, C["ok"]),
        ("改进 actor 方向校准", 1, 1, C["warn"]),
        ("回到 host 原口径重做消融", 2, 1, C["warn"]),
        ("B1 正式验收重做（条件）", 3, 3, C["bad"]),
    ]
    f = go.Figure()
    for name, start, dur, color in tasks:
        f.add_trace(go.Bar(y=[name], x=[dur], base=[start], orientation="h",
                           marker_color=color, width=0.5, showlegend=False,
                           hovertemplate=f"%{{y}}<br>11-{13 + start:02d} 起 · {dur} 天"
                                         "<extra></extra>"))
    f.update_xaxes(title="2026-11 日期偏移",
                   tickvals=[0, 1, 2, 3, 4, 5, 6],
                   ticktext=["11-13", "11-14", "11-15", "11-16", "11-17", "11-18", "11-19"])
    f.update_yaxes(autorange="reversed")
    figs["plan"] = fig_layout(f, 300, "近期排期 · 每个日期对应一次用户发起的会话")
    return figs


def div(fig, key):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart-" + key,
                       config={"displayModeBar": False, "responsive": True})


CSS = """
:root{--bg:%(bg)s;--card:%(card)s;--line:%(line)s;--text:%(text)s;--muted:%(muted)s;
      --accent:%(accent)s;--ok:%(ok)s;--warn:%(warn)s;--bad:%(bad)s}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);
     font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif;line-height:1.75;
     -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.wrap{max-width:1280px;margin:0 auto;padding:0 28px 100px}
.wrap.top{padding-bottom:0}
header{border-bottom:1px solid var(--line);padding:46px 0 30px;
       display:grid;grid-template-columns:minmax(0,1.6fr) minmax(260px,1fr);gap:40px;align-items:end}
h1{font-size:34px;letter-spacing:-.6px;margin:0 0 10px;font-weight:700}
h2{font-size:22px;margin:54px 0 8px;padding-left:13px;border-left:3px solid var(--accent)}
h3{font-size:15px;margin:30px 0 10px;color:#bcd4f0;font-weight:600}
p,li{color:#cbd8ea;font-size:14px}
.sub{color:var(--muted);font-size:13px}
.eyebrow{color:var(--accent);font-size:12px;letter-spacing:1.6px;text-transform:uppercase;margin:0 0 12px}
.status-panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
.status-panel .big{font-size:26px;font-weight:700;color:var(--bad);margin:2px 0 6px}
.status-panel dl{margin:12px 0 0;display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:12px}
.status-panel dt{color:var(--muted)}.status-panel dd{margin:0;color:#dbe7f6}
nav{position:sticky;top:0;z-index:20;background:rgba(11,18,32,.94);
    backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
nav .inner{max-width:1280px;margin:0 auto;padding:0 28px;display:flex;gap:4px;overflow-x:auto}
nav a{display:block;padding:13px 15px;color:var(--muted);font-size:13px;white-space:nowrap;
      border-bottom:2px solid transparent;transition:color .2s ease,border-color .2s ease}
nav a:hover{color:var(--text);text-decoration:none}
nav a.active{color:var(--accent);border-bottom-color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:14px;margin:28px 0 10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:17px 18px}
.kpi .v{font-size:25px;font-weight:700;margin:5px 0 2px;letter-spacing:-.4px}
.kpi .l{color:var(--muted);font-size:12px}
.kpi .n{color:var(--muted);font-size:11px;margin-top:5px}
.callout{border-left:3px solid var(--warn);background:#1c2233;padding:15px 20px;
         border-radius:0 10px 10px 0;margin:20px 0;font-size:14px;color:#d6e2f2}
.callout.bad{border-left-color:var(--bad)}
.callout.ok{border-left-color:var(--ok)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.grid2 .chart{min-width:0}
.chart{background:var(--card);border:1px solid var(--line);border-radius:14px;
       padding:10px 8px;margin:16px 0;overflow:hidden}
.note{color:var(--muted);font-size:13px;margin:2px 4px 22px}
table{width:100%%;border-collapse:collapse;font-size:13px;margin:12px 0 22px}
th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);
   padding:10px 9px;white-space:nowrap}
td{border-bottom:1px solid rgba(255,255,255,.05);padding:9px;color:#d3e0f2;vertical-align:top}
tbody tr:hover td{background:rgba(76,194,241,.05)}
tbody tr.is-hidden{display:none}
code{background:#0b1526;color:#8fd8f7;padding:1px 6px;border-radius:4px;font-size:12px}
.tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap}
.tag.ok{background:rgba(61,220,151,.15);color:var(--ok)}
.tag.warn{background:rgba(255,200,87,.15);color:var(--warn)}
.tag.bad{background:rgba(255,107,107,.15);color:var(--bad)}
.tag.neutral{background:rgba(148,168,196,.15);color:var(--muted)}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 4px}
.filters button{background:transparent;border:1px solid var(--line);color:var(--muted);
                border-radius:20px;padding:5px 14px;font-size:12px;cursor:pointer;
                transition:transform .15s ease,color .15s ease,border-color .15s ease}
.filters button:hover{color:var(--text);border-color:var(--accent)}
.filters button:active{transform:scale(.98)}
.filters button[aria-pressed="true"]{color:#0b1220;background:var(--accent);border-color:var(--accent)}
.empty{padding:22px;text-align:center;color:var(--muted);font-size:13px;
       border:1px dashed var(--line);border-radius:12px;display:none}
.rules{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:0 26px}
.rules div{border-top:1px solid var(--line);padding:11px 0;color:#cbd8ea;font-size:13px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:26px}
footer{margin-top:64px;border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:12px}
pre{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;
    overflow-x:auto;color:#b9d4ec;font-size:12px;line-height:1.7}
#boot{position:fixed;inset:0;background:var(--bg);display:grid;place-items:center;
      color:var(--muted);font-size:14px;z-index:60;transition:opacity .35s ease}
#boot.done{opacity:0;pointer-events:none}
#plotly-error{display:none}
.reveal{opacity:0;transform:translateY(10px);transition:opacity .45s ease,transform .45s ease}
.reveal.in{opacity:1;transform:none}
@media(max-width:900px){
  header{grid-template-columns:1fr;gap:22px;padding-top:32px}
  .grid2,.split{grid-template-columns:1fr}
  .wrap{padding:0 16px 70px}
  nav .inner{padding:0 16px}
  h1{font-size:26px}
}
@media(prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  .reveal,#boot{opacity:1;transform:none;transition:none}
  *{animation-duration:.01ms !important}
}
""" % C


def build():
    text = PLAN_MD.read_text(encoding="utf-8")
    tables = md_tables(text)
    log = subprocess.run(["git", "log", "--date=short", "--pretty=format:%ad"],
                         cwd=str(REPO), capture_output=True, text=True,
                         encoding="utf-8", errors="ignore", timeout=30).stdout
    dates = Counter(x.strip() for x in log.splitlines() if x.strip())
    figs = build_figures(tables, dates)

    tbl = {}
    for key, label in [
        ("1.2 里程碑完成度", "milestones"), ("1.3 判据状态", "criteria"),
        ("1.4 已完成时间线", "timeline"), ("2.1 角色分工", "roles"),
        ("2.3 实测速率", "speed"), ("3.1 判据主线", "plan"),
        ("3.2 非判据线并行项", "parallel"), ("3.3 阻塞项", "blocked"),
    ]:
        tbl[label] = first_table(tables, key)

    def table(label, status_col=None, filterable=False):
        h, rows = tbl[label]
        return html_table(h, rows, status_col=status_col, filterable=filterable)

    parts = []
    parts.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    parts.append("<title>NeuralOS 开发进度与计划 · AI 开发口径</title>")
    parts.append("<style>" + CSS + "</style><script>" + get_plotlyjs()
                 + "</script></head><body>")
    parts.append("<div id='boot'>正在装配仪表盘…</div>")
    parts.append("<div class='wrap top'>")

    # header：非对称分栏
    parts.append("<header><div>")
    parts.append("<p class='eyebrow'>NeuralOS / AI-driven development</p>")
    parts.append("<h1>开发进度与后续计划</h1>")
    parts.append("<p class='sub'>AI 自主开发口径 · 快照 2026-11-13 15:55 · "
                 "数据源 <code>git log</code> / 冻结判据 v1.8 / 进度报告 §9.1–§9.7 / "
                 "<code>audit/ai_dev_progress_plan_2026-11-13.md</code></p>")
    parts.append("</div><div class='status-panel'>")
    parts.append("<div class='sub'>当前判据状态</div><div class='big'>B1 不通过</div>")
    parts.append("<div class='sub'>M1 停在 T1 机制研究；B2 不通过、B3 正式实现为 0</div>")
    parts.append("<dl><dt>冻结判据</dt><dd>v1.8 · 阈值零变更</dd>"
                 "<dt>HEAD</dt><dd>d76eba7 · 已推送 GitHub</dd>"
                 "<dt>下一步执行日</dt><dd>2026-11-14（需发起会话）</dd>"
                 "<dt>纪律</dt><dd>不进 B2 / 不写容器 / 不写 SNC</dd></dl>")
    parts.append("</div></header></div>")

    parts.append("<nav><div class='inner'>")
    for href, label in [("#overview", "概览"), ("#progress", "进度与判据"),
                        ("#timeline", "时间线"), ("#plan", "后续计划"),
                        ("#mode", "AI 开发模式"), ("#rules", "纪律与边界")]:
        parts.append("<a href='%s' data-nav='%s'>%s</a>" % (href, href[1:], label))
    parts.append("</div></nav>")

    parts.append("<div class='wrap'>")
    parts.append("<div id='plotly-error' class='callout bad'>图表库未能加载；"
                 "下方全部表格与结论仍可直接阅读。</div>")

    # 概览
    parts.append("<section id='overview' class='reveal'>")
    kpis = [
        ("当前阶段", "T1 研究", "var(--warn)", "按 §1 失败动作停在 T1/T2"),
        ("本轮 g=0", "30.00%", "var(--warn)", "门槛 80% · SD 42.16"),
        ("配对归因", "30.00pp", "var(--ok)", "首次 PASS（要求 ≥30pp）"),
        ("seed 覆盖", "4 / 10", "var(--bad)", "有提升；其中 2 个达 100%"),
        ("M0 / M1", "79% / 65%", "var(--accent)", "交付物覆盖率"),
        ("判据版本", "v1.8", "var(--accent)", "阈值零变更 · 附录 J 销项"),
    ]
    parts.append("<div class='kpis'>")
    for label, v, col, note in kpis:
        parts.append("<div class='kpi'><div class='l'>%s</div>"
                     "<div class='v' style='color:%s'>%s</div><div class='n'>%s</div></div>"
                     % (label, col, v, note))
    parts.append("</div>")
    parts.append("<div class='callout bad'><b>一句话结论：</b>时序差分自预测让局部 R-STDP "
                 "首次破缺对称并学出映射，同 seed 配对「完整 − 无迹」<b>30.00pp ≥ 30pp</b> 达标；"
                 "但 g=0 均值仅 30.00%（要求 ≥80%）、口径偏离 host、新增内部结构，"
                 "<b>不构成 B1 通过</b>。瓶颈已从「自监督读出」推进到 "
                 "<b>actor 在破缺初期把随机方向校准到正确符号的可靠性</b>。</div>")
    parts.append("<div class='callout'><b>AI 开发 ≠ 自动推进：</b>每个计划日期对应一次用户发起的会话轮次；"
                 "AI 完成实现、10 seed 验证、判据与文档同步、构建与提交，人只给方向与验收。"
                 "日期是排期建议，不承诺某日通过判据。</div>")
    parts.append("</section>")

    # 进度与判据
    parts.append("<section id='progress' class='reveal'>")
    parts.append("<h2>01 · 里程碑与判据</h2>")
    parts.append("<p class='note'>完成度 = 可核对交付物覆盖率（DONE=1 / PARTIAL=0.5 / MISSING=0），"
                 "不是工时估计、也不代表判据通过。M1 的 65% 来自强制报告交付物补齐。</p>")
    parts.append("<div class='grid2'><div class='chart'>%s</div><div class='chart'>%s</div></div>"
                 % (div(figs["milestones"], "milestones"), div(figs["rounds"], "rounds")))
    parts.append(table("milestones", status_col=None))
    parts.append("<h3>判据状态（冻结 v1.8，阈值零变更）</h3>")
    parts.append(table("criteria", status_col=3))
    parts.append("<div class='callout'><b>口径提醒：</b>§9.7 的 30pp 达标是 <b>T1 独立实验台的归因判据</b>，"
                 "不是 B1 正式验收；其任务定义偏离 host（平局视为失败）且新增内部结构，"
                 "判据 §6.8 已明确写「不构成 B1 通过」。</div>")
    parts.append("</section>")

    # 时间线
    parts.append("<section id='timeline' class='reveal'>")
    parts.append("<h2>02 · 真实时间线</h2>")
    parts.append("<p class='note'>按 <code>git log --date=short</code> 统计，共 %d 次提交；"
                 "下方表格为计划文档 §1.4 的关键节点摘录。</p>" % sum(dates.values()))
    parts.append("<div class='chart'>%s</div>" % div(figs["timeline"], "timeline"))
    parts.append(table("timeline"))
    parts.append("</section>")

    # 计划
    parts.append("<section id='plan' class='reveal'>")
    parts.append("<h2>03 · 后续计划（带日期）</h2>")
    parts.append("<div class='chart'>%s</div>" % div(figs["plan"], "plan"))
    parts.append("<p class='note'>只画有日历日期的项。B2 以上按冻结判据阻塞，"
                 "<b>不伪造日期</b>，只在下方给相对轮次与门槛。</p>")
    parts.append("<h3>判据主线（唯一合规方向，来自 §6.8「下一步」）</h3>")
    parts.append("<div class='filters' id='plan-filters'>")
    for key, label in [("all", "全部"), ("warn", "待执行"), ("bad", "阻塞"), ("ok", "已完成")]:
        parts.append("<button type='button' data-filter='%s' aria-pressed='%s'>%s</button>"
                     % (key, "true" if key == "all" else "false", label))
    parts.append("</div>")
    parts.append("<div id='plan-table'>%s</div>" % table("plan", status_col=5, filterable=True))
    parts.append("<div class='empty' id='plan-empty'>该状态下暂无计划项。</div>")
    parts.append("<h3>非判据线并行项（不触碰 B1 结论）</h3>")
    parts.append(table("parallel"))
    parts.append("<h3>阻塞项：B2 以上无确定日历日期</h3>")
    parts.append(table("blocked"))
    parts.append("<div class='callout bad'><b>为什么不给 M2–M6 排日期：</b>"
                 "<code>judgments.md:4</code> 规定「未通过 Benchmark-1 之前，不得启动 Benchmark-2 以上开发」，"
                 "大纲 §5.1 又写「M1 不过，不写容器、不写 SNC、不碰 B2 以上」。"
                 "工程实现速率可测，研究是否收敛不可预测，写死日期等于伪造承诺。</div>")
    parts.append("</section>")

    # AI 开发模式
    parts.append("<section id='mode' class='reveal'>")
    parts.append("<h2>04 · AI 开发模式</h2>")
    parts.append("<div class='split'><div>")
    parts.append("<h3>角色分工</h3>")
    parts.append(table("roles"))
    parts.append("</div><div>")
    parts.append("<h3>实测速率 vs 大纲人力估算</h3>")
    parts.append(table("speed"))
    parts.append("<div class='callout'><b>严禁线性外推：</b>v1.4–v1.8 连续五轮均为负结果或局部结果。"
                 "「代码交付速率」不等于「研究突破速率」；大纲 §5.3 的 "
                 "「48 周全职单人 / 2.5 年兼职」是人力工时假设，对 AI 开发已不适用。</div>")
    parts.append("</div></div>")
    parts.append("<h3>每一轮的固定交付物（不可裁剪）</h3>")
    parts.append("<div class='rules'>")
    for i, item in enumerate([
        "独立 C 实验台 + DLL，核心源码零改动",
        "Python 驱动：10 seed / 300 试次，quick + report-only",
        "可复算 JSONL 数据，并加入 .gitignore",
        "audit 实验报告，含完整 SHA1 证据链",
        "判据追加 §6.x + 版本记录，阈值零变更",
        "进度报告 §9.x（MD + HTML + 章节断言）",
        "桌面大纲与 audit 副本附录，两份 SHA1 校验",
        "工作日志 + build.ps1 清单 + 构建 + lint + 提交",
    ], 1):
        parts.append("<div><span class='sub'>%02d</span> &nbsp; %s</div>" % (i, item))
    parts.append("</div>")
    parts.append("<div class='callout ok'><b>本轮实测产出：</b>1884 行增 / 41 行删（12 个文件），"
                 "约 600 行 C + 700 行 Python + 184 条数据 + 5 处文档同步；"
                 "v1.6→v1.7 用 23 分钟，v1.7→v1.8 用 1 小时 57 分。</div>")
    parts.append("</section>")

    # 纪律
    parts.append("<section id='rules' class='reveal'>")
    parts.append("<h2>05 · 纪律与明确不做</h2>")
    parts.append("<div class='rules'>")
    for item in [
        "判据阈值零变更；不调 lr / psp / τ、不改刺激生成器、不挑最好一组刷分",
        "能力口径严格 g=0；冻结投票；评估前后权重漂移必须为 0",
        "同时报告 A/B 预测分布差；同 seed 配对「完整 − 无迹 ≥30pp」归因",
        "四流单变量覆盖；诊断组 / 作弊对照 / 扫描组零星提升不计入验收",
        "核心 benchmark1_core.c SHA1 恒定 c8de144f…；构建与报告哈希互可核",
        "不宣布「B1 已通过」「B3 已启动」；不把 T1 的 30pp 写成 B1 验收",
        "不进 B2 探针、不写容器、不写 SNC；消融表缺失时不上结构可塑",
        "不用 g&gt;0 到达率替代 g=0 能力；教学探针不计入 B3 验收",
    ]:
        parts.append("<div>%s</div>" % item)
    parts.append("</div>")
    parts.append("<h3>证据索引与复现</h3>")
    parts.append("<pre>git --no-pager log --date=iso --pretty=format:'%h %ad %s' -30\n"
                 "py -3 Experiment/tools/gen_progress_report.py\n"
                 "py -3 Experiment/tools/gen_ai_dev_dashboard.py\n"
                 "py -3 Experiment/tools/b1_self_supervised.py --report-only\n"
                 "powershell -ExecutionPolicy Bypass -File Experiment/build.ps1 -Backup -Smoke</pre>")
    parts.append("<p class='note'>判据 <code>Experiment/spec/judgments.md</code> §6.1–§6.8 · "
                 "进度 <code>Experiment/audit/progress_review_2026-11-12.md</code> §9.1–§9.7 · "
                 "本轮实验 <code>audit/b1_self_supervised_prediction_2026-11-13.md</code> · "
                 "计划 <code>audit/ai_dev_progress_plan_2026-11-13.md</code></p>")
    parts.append("</section>")

    parts.append("<footer>本页只陈述可核对事实与排期建议；完成度是交付物覆盖率，不代表判据通过。"
                 "自包含单文件，无外部资源 · 生成于 %s · "
                 "生成脚本 <code>Experiment/tools/gen_ai_dev_dashboard.py</code></footer>"
                 % date.today().isoformat())
    parts.append("</div>")

    js = """
<script>
(function(){
  var boot=document.getElementById('boot');
  function ready(){ boot.classList.add('done'); setTimeout(function(){boot.remove();},400); }
  if(document.readyState==='complete'){ready();}else{window.addEventListener('load',ready);}
  if(typeof Plotly==='undefined'){document.getElementById('plotly-error').style.display='block';}

  // 计划表筛选
  var fbox=document.getElementById('plan-filters');
  var table=document.querySelector('#plan-table table');
  var empty=document.getElementById('plan-empty');
  function applyFilter(key){
    var shown=0;
    if(table){ Array.prototype.forEach.call(table.tBodies[0].rows,function(tr){
      var hit=(key==='all')||(tr.getAttribute('data-status')===key);
      tr.classList.toggle('is-hidden',!hit); if(hit){shown++;}
    }); }
    empty.style.display=(shown===0)?'block':'none';
    Array.prototype.forEach.call(fbox.querySelectorAll('button'),function(b){
      b.setAttribute('aria-pressed', b.getAttribute('data-filter')===key ? 'true':'false');
    });
  }
  fbox.addEventListener('click',function(e){
    var b=e.target.closest('button'); if(b){applyFilter(b.getAttribute('data-filter'));}
  });
  applyFilter('all');

  // 滚动进入的轻量显现（只动 transform/opacity）
  var reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var els=document.querySelectorAll('.reveal');
  // 兜底：任何情况下都保证正文最终可见
  setTimeout(function(){
    Array.prototype.forEach.call(els,function(el){el.classList.add('in');});
  },2500);
  if(reduce||!('IntersectionObserver' in window)){
    Array.prototype.forEach.call(els,function(el){el.classList.add('in');});
  }else{
    try{
      var io=new IntersectionObserver(function(entries){
        entries.forEach(function(en){ if(en.isIntersecting){en.target.classList.add('in');io.unobserve(en.target);} });
      },{rootMargin:'0px 0px -60px 0px',threshold:.08});
      Array.prototype.forEach.call(els,function(el){io.observe(el);});
    }catch(err){
      Array.prototype.forEach.call(els,function(el){el.classList.add('in');});
    }
  }

  // 顶部导航滚动高亮
  var navs=document.querySelectorAll('nav a');
  var sections=['overview','progress','timeline','plan','mode','rules'].map(function(id){return document.getElementById(id);});
  function spy(){
    var best=0, y=window.scrollY+140;
    sections.forEach(function(sec,i){ if(sec && sec.offsetTop<=y){best=i;} });
    Array.prototype.forEach.call(navs,function(a,i){a.classList.toggle('active',i===best);});
  }
  window.addEventListener('scroll',spy,{passive:true}); spy();

  // 窗口尺寸变化时让 plotly 重排
  var t=null;
  window.addEventListener('resize',function(){
    clearTimeout(t); t=setTimeout(function(){
      if(typeof Plotly!=='undefined'){
        Array.prototype.forEach.call(document.querySelectorAll('.js-plotly-plot'),function(el){
          Plotly.Plots.resize(el);
        });
      }
    },180);
  });
})();
</script>
"""
    parts.append(js)
    parts.append("</body></html>")
    return "".join(parts)


def main():
    html_text = build()
    OUT.write_text(html_text, encoding="utf-8")
    assert "B1 不通过" in html_text and "30.00pp" in html_text
    assert "chart-milestones" in html_text and "chart-rounds" in html_text
    assert "chart-timeline" in html_text and "chart-plan" in html_text
    banned = ("unsplash", "picsum", "placehold.co", "via.placeholder",
              "placeholder.com", "dummyimage", "lorem.space")
    assert not any(b in html_text for b in banned), "发现占位图外链"
    print("HTML -> %s (%d bytes)" % (OUT, OUT.stat().st_size))
    print("AI DEV DASHBOARD OK")


if __name__ == "__main__":
    main()
