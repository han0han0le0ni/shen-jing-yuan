"""报告交付前 QA：结构、图表、关键数字与源数据交叉核对。"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
h = (HERE / "full_report.html").read_text(encoding="utf-8")
m = (HERE / "检测报告.md").read_text(encoding="utf-8")
body = h[h.find("</head>") + len("</head>"):]      # 排除内嵌 plotly.js 库源码
R = json.loads((HERE / "scan_results.json").read_text(encoding="utf-8"))
S, Q = R["summary"], R["quality"]

checks = []


def chk(name, cond, detail=""):
    checks.append((name, bool(cond), detail))


sections = re.findall(r"^## \d", m, re.M)
plot_calls = body.count("Plotly.newPlot")
graph_divs = body.count("plotly-graph-div")
todo_h = [x for x in ("__TITLE__", "TODO") if x in body]
todo_m = [x for x in ("__TITLE__", "TODO") if x in m]
summary_bits = {"103 次有效": "103 次有效运行" in h
                                and "103 次有效 B2 运行" in (HERE / "build_report.py").read_text(encoding="utf-8"),
                "1 个空文件": "1 个空文件" in h,
                "0 PASS(md)": "0 PASS" in m}

chk("HTML 结构完整", h.startswith("<!DOCTYPE html>") and h.rstrip().endswith("</html>"),
    f"{len(h):,} 字符")
chk("内嵌 Plotly.js（离线可用）", "plotly.js v" in h or "Plotly" in h[:200000], "无需 CDN")
chk("7 张交互图", plot_calls == 7 and graph_divs == 7,
    f"newPlot={plot_calls}, graph-div={graph_divs}")
chk("无占位符 / TODO（正文）", not todo_h and not todo_m, f"html={todo_h} md={todo_m}")
chk("Markdown 8 个章节", len(sections) == 8, f"实际 {len(sections)}")
chk("源数据 B2 计数", S["b2_fail"] == 103 and Q["b2_csv_count"] == 104 and S["b2_pass"] == 0,
    f"FAIL={S['b2_fail']} CSV={Q['b2_csv_count']} PASS={S['b2_pass']}")
chk("报告含 103 FAIL / 1 空文件", all(summary_bits.values()), str(summary_bits))
chk("关键实验数字", all(x in h for x in ("0.4164", "0.5035", "0.6776", "0.4398", "19.7 小时")),
    "最佳/最新/长训练/时间差")
chk("内存缺陷证据", all(x in h for x in ("0xC0000374", "0xC0000005", "net2_inject_reward2", "PROP_N"))
    and all(x in m for x in ("0xC0000374", "0xC0000005")))
chk("数据质量数字", all(x in h for x in ("54", "8", "23", "47", "94", "19,060")),
    "孤立 CSV/中断日志/标记文件/舍入行/公式行/B1 行数")
chk("episode 数不符修正为 5", "v6_2000.csv、v6_500.csv" in h and "v6_2000.csv、v6_500.csv" in m
    and '"CSV 与日志 episode 数不符", 3' not in (HERE / "build_report.py").read_text(encoding="utf-8"))
chk("可比性限制已声明", "不可直接比较" in h and "不可直接比较" in m)
chk("复现命令已给出", "scan_data.py" in m and "build_report.py" in m)
chk("Markdown 表格完整", m.count("|---") >= 6, f"{m.count('|---')} 张表")

for name, ok, detail in checks:
    print(("PASS " if ok else "FAIL "), name, "—", detail)
print("TOTAL", sum(ok for _, ok, _ in checks), "/", len(checks))
if not all(ok for _, ok, _ in checks):
    raise SystemExit(1)
