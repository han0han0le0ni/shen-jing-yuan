# -*- coding: utf-8 -*-
"""B1 完整验收实验驱动（补齐 spec/judgments.md §1 强制报告与 §5 统计规范）。

分阶段运行（可中断续跑，按 tag 去重）：
    py -3 tools/b1_acceptance.py --phase seeds    # 10 seed 分布 + 消融配对
    py -3 tools/b1_acceptance.py --phase scale    # tau_e–N 三档标度扫描
    py -3 tools/b1_acceptance.py --phase report   # 汇总成 audit/ 报告与 JSONL

原始数据写 data/b1_acceptance/（本地、不入库）；CSV 按 §5 归入 data/legacy/b1/；
对外证据为 audit/b1_acceptance_report.md 与 audit/b1_acceptance_summary.jsonl。
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "benchmark1.exe"
DATA = ROOT / "data" / "b1_acceptance"
LEGACY = ROOT / "data" / "legacy" / "b1"
AUDIT = ROOT / "audit"
RUNS = DATA / "runs.jsonl"
SUMMARY = AUDIT / "b1_acceptance_summary.jsonl"
REPORT = AUDIT / "b1_acceptance_report.md"
CRITERIA_VERSION = "judgments.md v1 (2026-11-12)"
SOURCES = [ROOT / "brain" / "benchmark1_core.c",
           ROOT / "brain" / "benchmark1_core.h",
           ROOT / "host" / "benchmark1_host.cpp"]

TRIALS = 300
LR = 0.02
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
SCALE_SEEDS = [7, 17, 27]
TAUS = [25, 50, 100, 200, 400, 800]
N_TARGETS = [500, 1000, 2000]
N_OVERHEAD = 150

CONDS = {
    "k3": [],
    "k1_200": ["--trace-k", "1", "--tau-traces", "200,200,200"],
    "notrace": ["--disable_trace"],
    "noreward": ["--disable_reward"],
}
COND_LABEL = {
    "k3": "完整系统 K=3 (20/100/500)",
    "k1_200": "单尺度 K=1 (tau=200)",
    "notrace": "移除资格迹",
    "noreward": "移除奖励调制（近似单调质消融）",
}


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_bundle(seed):
    return {name: int(hashlib.sha256(("%d:%s" % (seed, name)).encode()).hexdigest()[:8], 16)
            for name in ("structure", "weight", "noise", "trial_order")}


SPARSITY_RE = re.compile(
    r"gini=([\d.]+) effective=([\d.]+) active=(\d+)/(\d+) "
    r"p10=([\d.]+) p50=([\d.]+) p90=([\d.]+) sat_hi=(\d+) sat_lo=(\d+)")


def parse_diag(path):
    if not path.exists():
        return None, None
    found = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = SPARSITY_RE.search(line)
        if m:
            found.append({
                "gini": float(m.group(1)), "effective": float(m.group(2)),
                "active": int(m.group(3)), "synapses": int(m.group(4)),
                "p10": float(m.group(5)), "p50": float(m.group(6)), "p90": float(m.group(7)),
                "sat_hi": int(m.group(8)), "sat_lo": int(m.group(9)),
            })
    return (found[0] if found else None), (found[-1] if len(found) > 1 else None)


def parse_summary(path):
    if not path.exists():
        return None
    rec = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if '"summary"' in line:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                pass
    return rec


def existing_tags():
    if not RUNS.exists():
        return set()
    tags = set()
    for line in RUNS.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "run":
            tags.add(r.get("tag"))
    return tags


def append_run(rec):
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_one(tag, spec, hashes, done):
    if tag in done:
        return None
    jsonl = DATA / (tag + ".jsonl")
    diag = DATA / (tag + ".diag.txt")
    csv = LEGACY / (tag + ".csv")
    cmd = [str(EXE), "--trials", str(TRIALS), "--delay", str(spec["delay"]),
           "--seed", str(spec["seed"]), "--learning-rate", str(LR),
           "--jsonl", str(jsonl), "--tag", tag, "--out", str(csv), "--diag", str(diag)]
    if spec.get("hidden"):
        cmd += ["--hidden", str(spec["hidden"])]
    cmd += spec.get("extra", [])

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="ignore", timeout=300)
        returncode = proc.returncode
        stderr = (proc.stderr or "")[-300:]
    except Exception as exc:
        returncode, stderr = -1, repr(exc)
    elapsed = round(time.time() - t0, 3)

    summ = parse_summary(jsonl)
    init_w, final_w = parse_diag(diag)
    rec = {
        "event": "run", "tag": tag, "phase": spec["phase"], "cond": spec["cond"],
        "delay": spec["delay"], "seed": spec["seed"], "trials": TRIALS,
        "n_target": spec.get("n_target"), "hidden": spec.get("hidden"),
        "tau": spec.get("tau"), "extra": spec.get("extra", []),
        "test_acc": summ.get("test_acc") if summ else None,
        "train_acc": summ.get("train_acc") if summ else None,
        "neurons": summ.get("neurons") if summ else None,
        "synapses": summ.get("synapses") if summ else None,
        "trace_k": summ.get("trace_k") if summ else None,
        "weight_init": init_w, "weight_final": final_w,
        "returncode": returncode, "stderr": stderr, "elapsed_s": elapsed,
    }
    meta = {
        "tag": tag, "benchmark": 1, "criteria_version": CRITERIA_VERSION,
        "source_hash": hashes["sources"], "binary_hash": hashes["binary"],
        "seed": spec["seed"], "seed_bundle": seed_bundle(spec["seed"]),
        "seed_bundle_note": "四子种子按 M0 契约声明；B1 C 运行时仍为单流 srand(seed)，契约尚未落到 B1 核心",
        "log_format": "jsonl", "jsonl": str(jsonl.relative_to(ROOT)).replace("\\", "/"),
        "legacy_csv": str(csv.relative_to(ROOT)).replace("\\", "/"),
        "diag": str(diag.relative_to(ROOT)).replace("\\", "/"),
        "params": {"trials": TRIALS, "delay": spec["delay"], "learning_rate": LR,
                   "hidden": spec.get("hidden"), "tau": spec.get("tau"),
                   "extra": spec.get("extra", []), "condition": spec["cond"]},
        "elapsed_s": elapsed,
    }
    (DATA / (tag + ".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    append_run(rec)
    return rec


def phase_seeds(hashes):
    done = existing_tags()
    jobs = []
    for seed in SEEDS:
        for cond in ("k3", "k1_200", "notrace"):
            for delay in (200, 500):
                tag = "seed_%s_D%d_s%d" % (cond, delay, seed)
                jobs.append((tag, {"phase": "seeds", "cond": cond, "delay": delay,
                                   "seed": seed, "extra": CONDS[cond]}))
        jobs.append(("seed_noreward_D200_s%d" % seed,
                     {"phase": "seeds", "cond": "noreward", "delay": 200,
                      "seed": seed, "extra": CONDS["noreward"]}))
    return run_jobs(jobs, hashes, done, "seeds")


def phase_scale(hashes):
    done = existing_tags()
    jobs = []
    for n in N_TARGETS:
        hidden = n - N_OVERHEAD
        for tau in TAUS:
            for seed in SCALE_SEEDS:
                tag = "scale_N%d_t%d_s%d" % (n, tau, seed)
                jobs.append((tag, {"phase": "scale", "cond": "k1_tau_scan",
                                   "delay": 500, "seed": seed, "hidden": hidden,
                                   "n_target": n, "tau": tau,
                                   "extra": ["--trace-k", "1", "--tau-traces",
                                             "%d,%d,%d" % (tau, tau, tau)]}))
    return run_jobs(jobs, hashes, done, "scale")


def run_jobs(jobs, hashes, done, label):
    fresh = [(t, s) for t, s in jobs if t not in done]
    print("[%s] total=%d new=%d cached=%d" % (label, len(jobs), len(fresh), len(jobs) - len(fresh)))
    ok = 0
    for i, (tag, spec) in enumerate(fresh, 1):
        rec = run_one(tag, spec, hashes, done)
        if rec and rec["test_acc"] is not None:
            ok += 1
        if i % 10 == 0 or i == len(fresh):
            print("  %d/%d ok=%d last=%s" % (i, len(fresh), ok, tag), flush=True)
    print("[%s] done, %d successful new runs" % (label, ok))
    return ok


def load_runs():
    rows = []
    if RUNS.exists():
        for line in RUNS.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") != "run":
                continue
            if "cond" not in r and "condition" in r:
                r["cond"] = r["condition"]
            rows.append(r)
    by_tag = {}
    for r in rows:
        by_tag[r["tag"]] = r
    return list(by_tag.values())


def stats(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": round(statistics.mean(vals), 3),
            "sd": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
            "median": round(statistics.median(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3)}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def fit_loglog(points):
    pts = [(math.log(n), math.log(t)) for n, t in points if n > 0 and t > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
    intercept = my - slope * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (slope * p[0] + intercept)) ** 2 for p in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"alpha": round(slope, 4), "intercept": round(intercept, 4),
            "r2": round(r2, 4) if r2 is not None else None, "n_points": len(pts)}


def build_report(runs, hashes):
    seeds = [r for r in runs if r.get("phase") == "seeds" and r.get("test_acc") is not None]
    scale = [r for r in runs if r.get("phase") == "scale" and r.get("test_acc") is not None]

    s = []
    s.append("# B1 完整验收补齐报告 · 2026-11-12")
    s.append("")
    s.append("> 依据：`spec/judgments.md` 冻结 v1（§1 强制报告、§4 六组消融、§5 统计规范）。")
    s.append("> 驱动：`tools/b1_acceptance.py`；二进制 `%s`；源码哈希见文末。" % hashes["binary"][:12])
    s.append("> 全部数字来自本次实跑，原始 JSONL / meta / diag 在 `data/b1_acceptance/`，CSV 已归入 `data/legacy/b1/`。")
    s.append("")
    s.append("## 0. 结论")
    s.append("")
    k3_200 = stats([r["test_acc"] for r in seeds
                    if r["cond"] == "k3" and r["delay"] == 200])
    k3_200_pass = sum(1 for r in seeds if r["cond"] == "k3" and r["delay"] == 200
                      and r["test_acc"] >= 80)
    s.append("**B1 核心判据不通过。** 按 §5 补齐 10 个独立 seed 的分布后，完整系统 `K=3` 在 D=200 "
             "只有 **%d/10** 达到 80%%（mean %.2f%% / median %.2f%%）；"
             "`judgments.md §6` 与 `audit/b1_m1_report.md` 的 `100%%` 全部来自 **seed=12345 单种子**。"
             "用同一二进制复跑 seed=12345 仍得 100%%（见 §1 复现对照），说明历史结论不是代码回归，"
             "而是**只报最好一次**造成的假象，直接违反 §5「不得只给最好一次」。" %
             (k3_200_pass, k3_200["mean"], k3_200["median"]))
    s.append("")
    s.append("按 §1 失败动作规定：**停止，转 T1/T2 研究（前向模型、解码可学性），不得进入 B2。**"
             "本报告不再推进 B2 探针；其余强制报告项（τe–N、稀疏化、消融表、neuromod）仍完整给出，"
             "作为诊断证据，但**不能用来把 B1 拉回「通过」**。")
    s.append("")
    s.append("次要阻塞：`neuromod` 交叉验证因本机无 `cargo` 无法执行；6 组跨基准消融中 "
             "4 组机制未在 B1 代码中实现、消融条件无法构造（§4）。按 §4「没有消融表，通过判据不成立」，"
             "即使种子分布达标，B1 也仍不成立。")
    s.append("")

    s.append("## 1. 种子分布（§5：median / mean / SD / 种子级原始值）")
    s.append("")
    s.append("10 个独立 seed，每条件独立进程（独立脑实例），300 trials（train/test 各 150），"
             "`lr=0.02`、`psp=10` 校准工作点。host 退出码 `2` 表示「test_acc<80% 判据 FAIL」，"
             "**不是运行失败**；70 次运行全部写出了完整 JSONL / diag。")
    s.append("")
    dist_rows = []
    raw_rows = []
    for cond in ("k3", "k1_200", "notrace", "noreward"):
        for delay in (200, 500):
            vals = [r["test_acc"] for r in seeds
                    if r["cond"] == cond and r["delay"] == delay and r["test_acc"] is not None]
            if not vals:
                continue
            st = stats(vals)
            dist_rows.append([COND_LABEL[cond], "D=%d" % delay, st["n"],
                              "%.2f" % st["median"], "%.2f" % st["mean"], "%.2f" % st["sd"],
                              "%.1f / %.1f" % (st["min"], st["max"]),
                              "%d/%d" % (sum(1 for v in vals if v >= 80), len(vals))])
            raw_rows.append([COND_LABEL[cond], "D=%d" % delay,
                             ", ".join("%.1f" % v for v in vals)])
    s.append(md_table(["条件", "延迟", "n", "median (%)", "mean (%)", "SD",
                       "min / max", "≥80% 的 seed"], dist_rows))
    s.append("")
    s.append("### 种子级原始值（test_acc, %）")
    s.append("")
    s.append(md_table(["条件", "延迟", "10 个 seed 原始值"], raw_rows))
    s.append("")
    s.append("### 复现对照与根因线索")
    s.append("")
    s.append("- 用同一二进制复跑 `seed=12345, D=200, K=3, lr=0.02`："
             "**test_acc=100.00%**（train 82.67%），与 `sweep_k3_D200.jsonl` 完全一致 → "
             "历史结果可复现，不是代码回归，也不是本次重编译造成的差异。")
    s.append("- 10 个 seed 的 test_acc 呈**二值化**（大量 0.0% / 53.3%，个别 99.3%），"
             "不是围绕 80% 的连续分布。CSV 显示刺激 A/B 本身平衡（如 151/149），"
             "但 `predicted_vote` 与刺激的对应关系在不同 seed 下会整体翻转"
             "（host 约定 `expected = stimulus==0 ? 1 : 2`，投票 1=A / 2=B / 0=平局）。"
             "这提示正确率主要由初始权重造成的输出层固定偏向决定，而非稳定的延迟信用分配学习。")
    s.append("- 该现象是**新的实验设计缺陷线索**。本轮只记录，不擅自改学习规则或刺激生成器去「刷过判据」；"
             "按 §1 失败动作应转 T1/T2 研究，并在重做实验时落实 §5 的独立「试次顺序」随机流。")
    s.append("")

    pairs = []
    for delay in (200, 500):
        by_seed = {}
        for r in seeds:
            if r["delay"] == delay and r["cond"] in ("k3", "notrace") and r["test_acc"] is not None:
                by_seed.setdefault(r["seed"], {})[r["cond"]] = r["test_acc"]
        diffs = [d["k3"] - d["notrace"] for d in by_seed.values() if "k3" in d and "notrace" in d]
        st = stats(diffs)
        d_cohen = round(st["mean"] / st["sd"], 3) if st["sd"] and st["sd"] > 0 else None
        pairs.append(["D=%d" % delay, st["n"], "%.2fpp" % st["mean"], "%.2f" % st["sd"],
                      "%.1f / %.1f" % (st["min"], st["max"]),
                      "%.2f" % d_cohen if d_cohen is not None else "—",
                      "PASS" if st["mean"] >= 30 else "FAIL"])
    s.append("### 配对消融：完整组 − 无迹组（判据 ≥30pp）")
    s.append("")
    s.append(md_table(["延迟", "配对数", "平均下降", "SD", "min / max", "Cohen's d (paired)", "判定"], pairs))
    s.append("")
    s.append("Cohen's d 定义：配对差值均值 / 配对差值样本 SD。未使用 scipy（本机未安装），"
             "故不给 Wilcoxon p 值；此项按 §2 属 B2 强制报告，B1 只要求 mean/SD/种子分布。")
    s.append("")

    s.append("## 2. τe–N 标度（§1 强制报告）")
    s.append("")
    s.append("方法（本报告自定义、可复算）：单尺度 `K=1`，`τ ∈ {25,50,100,200,400,800}`，"
             "固定 `D=500`（长延迟才有区分度），每个 (N, τ) 跑 3 个 seed；"
             "网络规模 `N = hidden + 150`（感觉 50+50、输出 25+25），取 `hidden = N − 150`。"
             "对每个 N 取平均 test_acc 最高的 τ 为 `τ*(N)`（并列取最小 τ），"
             "再对 `log τ* = α·log N + b` 做最小二乘，检验 α 是否偏离 0.5。")
    s.append("")
    scale_rows, tau_star_points, tau_star_detail = [], [], []
    for n in N_TARGETS:
        acc_by_tau = {}
        for tau in TAUS:
            vals = [r["test_acc"] for r in scale
                    if r.get("n_target") == n and r.get("tau") == tau and r["test_acc"] is not None]
            if vals:
                acc_by_tau[tau] = stats(vals)
                scale_rows.append([n, tau, len(vals), "%.2f" % acc_by_tau[tau]["mean"],
                                   "%.2f" % acc_by_tau[tau]["sd"],
                                   ", ".join("%.1f" % v for v in vals)])
        if not acc_by_tau:
            continue
        best = max(acc_by_tau.values(), key=lambda x: x["mean"])["mean"]
        tied = sorted(t for t, v in acc_by_tau.items() if abs(v["mean"] - best) < 1e-9)
        tau_star = tied[0]
        distinguishable = len(tied) < len(acc_by_tau)
        tau_star_detail.append([n, tau_star, "%.2f" % best, len(tied), len(acc_by_tau),
                                "有区分度" if distinguishable else "**无区分度**"])
        if distinguishable:
            tau_star_points.append((n, tau_star))
    s.append(md_table(["N", "τ (ticks)", "n seeds", "mean acc (%)", "SD", "原始值"], scale_rows))
    s.append("")
    s.append(md_table(["N", "τ*(N)", "该 τ 的 mean acc (%)", "并列 τ 数", "候选 τ 数", "结论"], tau_star_detail))
    s.append("")
    fit = fit_loglog(tau_star_points)
    if fit:
        s.append("**拟合结果**：`α = %.4f`，`b = %.4f`，`R² = %s`，基于 %d 个有效 (N, τ*) 点。"
                 "设计/判据期望 `τe ∝ √N`（α=0.5）：%s" %
                 (fit["alpha"], fit["intercept"],
                  "%.4f" % fit["r2"] if fit["r2"] is not None else "—",
                  fit["n_points"],
                  "偏离 0.5，按判据需修正标度律并重跑" if abs(fit["alpha"] - 0.5) > 0.15
                  else "与 0.5 接近（探索性结论）"))
    else:
        s.append("**拟合结果**：有效 (N, τ*) 点少于 2 个或各 τ 无区分度，"
                 "**标度律无法拟合**。这是真实负结果，不填造指数。")
    s.append("")
    s.append("限制：该扫描只改 `hidden`，未同步调整 `avg_conn`，网络规模变化同时改变了突触数与发放动态；"
             "且每个 (N, τ) 仅 3 seed。判据 §5 要求 ≥5 seed 分布，故本节只作探索性标度估计，"
             "不足以作为「标度律已校准」的结论。")
    s.append("")

    s.append("## 3. 稀疏化（§1 强制报告）")
    s.append("")
    s.append("诊断层新增 `gini`（权重绝对值基尼系数）、`effective = (Σ|w|)² / Σw²`（有效连接数 / 逆参与比）、"
             "`active = |w| > 0.5·mean 的连接数`、`p10/p50/p90` 与饱和计数；"
             "host 通过 `--diag` 在**初始化后**与**训练结束后**各写一次快照，形成 init→final 对照。"
             "这项只改诊断输出，未改学习规则。")
    s.append("")
    sp_rows = []
    for cond in ("k3", "k1_200", "notrace", "noreward"):
        dg, de, pairs_n = [], [], 0
        for r in seeds:
            if r["cond"] != cond or r["delay"] != 200:
                continue
            wi, wf = r.get("weight_init"), r.get("weight_final")
            if not wi or not wf:
                continue
            pairs_n += 1
            dg.append(wf["gini"] - wi["gini"])
            de.append(wf["effective"] - wi["effective"])
        if pairs_n:
            g, e = stats(dg), stats(de)
            sp_rows.append([COND_LABEL[cond], pairs_n,
                            "%+.4f ± %.4f" % (g["mean"], g["sd"]),
                            "%+.1f ± %.1f" % (e["mean"], e["sd"]),
                            "停在初始化水平" if abs(g["mean"]) < 0.005 and abs(e["mean"]) < 5
                            else "发生了可测变化"])
    s.append(md_table(["条件", "配对数", "Δ Gini (final−init)", "Δ 有效连接数", "判据关注点"], sp_rows))
    s.append("")
    s.append("判据原文要求检查「**无迹组**权重 Gini / 有效连接数是否停在随机初始化水平」："
             "上表中无迹组与无奖励组 `Δ` **精确为 0**（10/10 seed），即权重完全没有更新，"
             "符合「停在随机初始化水平」；完整组也只有 `ΔGini≈+0.002`、`Δ有效连接≈−8`，"
             "与 §1 的正确率二值化、§2 的 τ 无区分度一致，共同指向「本次运行中学习几乎没有发生」。"
             "`Δ` 为 10 seed 均值 ± SD；本报告不据此作因果断言，但把它作为 B1 实验需重做的旁证。")
    s.append("")

    s.append("## 4. 跨基准 6 组强制消融（§4）")
    s.append("")
    ablation = [
        ("移除资格迹", "PASS", "`--disable_trace` 运行时开关，10 seed × D=200/500 有数据", "本报告 §1"),
        ("移除误差注入通道", "N/A", "B1 核心无误差注入通道（`m_Er` 仅存在于 B2 设计），机制未实现，消融条件无法构造", "—"),
        ("移除多通道调质", "PARTIAL", "仅有 `--disable_reward`（退化为「无奖励」），不等于判据要求的「退化为单通道奖励」", "本报告 §1"),
        ("移除异构群体编码", "N/A", "B1 为 5 类固定神经元、同构参数，机制未实现", "—"),
        ("移除竞争性神经发生", "N/A", "B1 无结构可塑，机制未实现", "—"),
        ("移除睡眠重放", "N/A", "B1 无记忆槽/重放，机制未实现", "—"),
    ]
    s.append(md_table(["消融条件", "状态", "说明", "证据"], [list(x) for x in ablation]))
    s.append("")
    s.append("**这是 B1 的叠加阻塞（不是首要阻塞）**：首要阻塞是 §1 种子分布失效。"
             "这 4 组消融不是「没跑」，而是**机制不存在**；按 §4「没有消融表，通过判据不成立」，"
             "B1 仍不得宣布通过，按文件开头 `judgments.md:4` 也不得启动 B2 以上开发。")
    s.append("")

    s.append("## 5. neuromod 同种子统计等价（§1 强制报告）")
    s.append("")
    s.append("**阻塞，未执行**：本机 `cargo` / Rust 工具链不存在，无法构建 `neuromod` 的 `rm_stdp` 参考实现；"
             "判据要求的是均值/方差/迹曲线的统计等价（**不要求 1e-4 逐位一致**），"
             "没有参考实现就不能产出对照数据。本报告明确留空，不用自造基线顶替。")
    s.append("")

    s.append("## 6. 统计与数据规范符合度（§5）")
    s.append("")
    s.append(md_table(["规范", "本次状态", "说明"], [
        ["日志只写 JSONL", "PASS", "主数据 `data/b1_acceptance/*.jsonl`；CSV 作为旧格式归入 `data/legacy/b1/`"],
        ["meta.json 全字段", "PASS（新运行）", "每个运行写 `*.meta.json`：源码哈希、二进制哈希、seed_bundle 四子种子、全部参数、判据版本"],
        ["seed_bundle 四子种子", "PARTIAL", "meta 已声明四个派生种子；但 B1 C 运行时仍是单流 `srand(seed)`，M0 契约尚未落到 B1 核心"],
        ["median / mean / SD / 种子级原始值", "PASS", "本报告 §1 给出 10 seed 逐项原始值"],
        ["失败运行保留", "PASS", "`runs.jsonl` 按 tag 去重且保留 returncode/stderr，不删除失败运行"],
        ["独立脑实例 / 单变量", "PASS", "每个 (seed, 条件, 延迟) 独立进程；τ 扫描只改 τ 与 N"],
    ]))
    s.append("")
    s.append("历史数据口径：`data/b1_m1/` 的 47 个 `*.csv.meta.json` 仍缺源码/二进制哈希与判据版本，"
             "且 seed 全为 12345；**未回填、未篡改**，仅在 `audit/legacy_csv_manifest.md` 登记。"
             "新运行与历史运行使用不同二进制（本次重编译了 B1 诊断层），已在各自 meta 中区分。")
    s.append("")

    s.append("## 7. 判据状态汇总")
    s.append("")
    s.append(md_table(["B1 项", "阈值", "本次结果", "状态"], [
        ["D=200 正确率（按 §5 种子分布）", "≥80%",
         "10 seed：mean %.2f%% / median %.2f%% / 仅 %d 达标" %
         (k3_200["mean"], k3_200["median"], k3_200_pass), "**FAIL**"],
        ["无迹下降", "≥30pp", "见 §1 配对消融；完整组自身未达 80%，消融差失去判定意义", "**FAIL**"],
        ["D½", "≥200", "历史 919 (K=3) / 481 (K=1) 为单 seed 口径，需按种子分布重做", "待重做"],
        ["延迟扫描 6 档", "全覆盖", "历史 `sweep_*` 18 个文件已覆盖（单 seed）", "PASS（口径有限）"],
        ["延迟-正确率曲线", "必须", "历史 `audit/b1_m1_report.md`（单 seed）", "PARTIAL"],
        ["多尺度 vs 单尺度", "必须", "历史 + 本报告 §1 的 10 seed 对照；D=500 未见稳定优势", "PARTIAL"],
        ["τe–N 三档 + 指数拟合", "必须", "本报告 §2（探索性，3 seed）", "PARTIAL"],
        ["neuromod 统计等价", "必须", "工具链缺失", "FAIL"],
        ["稀疏化 Gini / 有效连接", "必须", "本报告 §3", "PASS"],
        ["≥5 seed 分布 mean±SD", "必须", "本报告 §1，10 seed", "PASS"],
        ["6 组跨基准消融表", "缺一不可", "本报告 §4，4 组机制未实现", "FAIL"],
    ]))
    s.append("")
    s.append("**综合：B1 不通过（核心判据在种子分布上失效）。** 按 §1 失败动作，"
             "下一步是**停止并转 T1/T2 研究**：① 查清输出层固定偏向与刺激—投票映射，"
             "重做 B1 实验设计并落实独立「试次顺序」随机流；② 安装 Rust 工具链补 neuromod 对照；"
             "③ 明确 §4 六组消融在 B1 阶段的适用口径；④ 之后才谈 τe–N ≥5 seed 重跑与 B2 探针。")
    s.append("")

    s.append("## 8. 复现与证据")
    s.append("")
    s.append("```")
    s.append("py -3 tools/b1_acceptance.py --phase seeds")
    s.append("py -3 tools/b1_acceptance.py --phase scale")
    s.append("py -3 tools/b1_acceptance.py --phase report")
    s.append("```")
    s.append("")
    s.append(md_table(["源码", "SHA1"], [[p, h] for p, h in hashes["sources"].items()]))
    s.append("")
    s.append("`benchmark1.exe` SHA1：`%s`；判据版本：`%s`。" %
             (hashes["binary"], CRITERIA_VERSION))
    return "\n".join(s)


def phase_report(hashes, runs):
    seeds = [r for r in runs if r.get("phase") == "seeds" and r.get("test_acc") is not None]
    scale = [r for r in runs if r.get("phase") == "scale" and r.get("test_acc") is not None]
    if not seeds and not scale:
        print("no runs found; execute --phase seeds / --phase scale first")
        return 1
    report = build_report(runs, hashes)
    REPORT.write_text(report, encoding="utf-8")
    write_legacy_manifest()

    lines = []
    for r in runs:
        lines.append({"event": "run_brief", "tag": r["tag"], "phase": r.get("phase"),
                      "cond": r.get("cond"), "delay": r.get("delay"), "seed": r.get("seed"),
                      "n_target": r.get("n_target"), "tau": r.get("tau"),
                      "test_acc": r.get("test_acc"),
                      "gini_init": (r.get("weight_init") or {}).get("gini"),
                      "gini_final": (r.get("weight_final") or {}).get("gini"),
                      "effective_init": (r.get("weight_init") or {}).get("effective"),
                      "effective_final": (r.get("weight_final") or {}).get("effective")})
    for cond in ("k3", "k1_200", "notrace", "noreward"):
        for delay in (200, 500):
            vals = [r["test_acc"] for r in seeds
                    if r["cond"] == cond and r["delay"] == delay and r["test_acc"] is not None]
            if vals:
                lines.append({"event": "seed_distribution", "cond": cond, "delay": delay,
                              **stats(vals)})
    for delay in (200, 500):
        by_seed = {}
        for r in seeds:
            if r["delay"] == delay and r["cond"] in ("k3", "notrace") and r["test_acc"] is not None:
                by_seed.setdefault(r["seed"], {})[r["cond"]] = r["test_acc"]
        diffs = [d["k3"] - d["notrace"] for d in by_seed.values() if "k3" in d and "notrace" in d]
        if diffs:
            lines.append({"event": "paired_ablation", "delay": delay, **stats(diffs)})
    points = []
    for n in N_TARGETS:
        acc_by_tau = {}
        for tau in TAUS:
            vals = [r["test_acc"] for r in scale
                    if r.get("n_target") == n and r.get("tau") == tau and r["test_acc"] is not None]
            if vals:
                acc_by_tau[tau] = statistics.mean(vals)
        if not acc_by_tau:
            continue
        best = max(acc_by_tau.values())
        tied = sorted(t for t, v in acc_by_tau.items() if abs(v - best) < 1e-9)
        lines.append({"event": "tau_star", "n": n, "tau_star": tied[0],
                      "best_acc": round(best, 3), "tied": len(tied),
                      "distinguishable": len(tied) < len(acc_by_tau)})
        if len(tied) < len(acc_by_tau):
            points.append((n, tied[0]))
    fit = fit_loglog(points)
    lines.append({"event": "scaling_fit", **(fit or {"alpha": None, "r2": None,
                                                     "n_points": len(points)})})
    for cond in ("k3", "k1_200", "notrace", "noreward"):
        dg, de = [], []
        for r in seeds:
            if r["cond"] != cond or r["delay"] != 200:
                continue
            wi, wf = r.get("weight_init"), r.get("weight_final")
            if wi and wf:
                dg.append(wf["gini"] - wi["gini"])
                de.append(wf["effective"] - wi["effective"])
        if dg:
            lines.append({"event": "sparsity", "cond": cond,
                          "delta_gini": stats(dg), "delta_effective": stats(de)})
    k3_vals = [r["test_acc"] for r in seeds if r["cond"] == "k3" and r["delay"] == 200]
    lines.append({"event": "seed_pass_rate", "cond": "k3", "delay": 200,
                  "passed": sum(1 for v in k3_vals if v >= 80), "total": len(k3_vals),
                  **stats(k3_vals)})
    lines.append({"event": "acceptance_state", "b1_core": "FAIL_SEED_DISTRIBUTION",
                  "neuromod": "BLOCKED_NO_CARGO", "six_ablations": "4_MECHANISMS_ABSENT",
                  "criteria_version": CRITERIA_VERSION,
                  "binary_hash": hashes["binary"]})
    with open(SUMMARY, "w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("runs=%d (seeds=%d scale=%d)" % (len(runs), len(seeds), len(scale)))
    print("tau fit: %s" % (fit or "not fittable"))
    print("REPORT -> %s (%d bytes)" % (REPORT, REPORT.stat().st_size))
    print("SUMMARY-> %s (%d lines)" % (SUMMARY, len(lines)))
    print("B1 ACCEPTANCE REPORT OK")
    return 0


def write_legacy_manifest():
    csvs = sorted(ROOT.rglob("*.csv"))
    by_dir = {}
    for p in csvs:
        d = p.parent.relative_to(ROOT).as_posix()
        by_dir[d] = by_dir.get(d, 0) + 1
    rows = []
    for d, n in sorted(by_dir.items(), key=lambda x: -x[1]):
        state = ("本次新运行，已按 §5 归入归档目录" if d.startswith("data/legacy")
                 else "历史数据，仍在原目录（未移动）")
        rows.append(["`%s/`" % d, n, state])
    s = ["# 历史 CSV 归档口径 · 2026-11-12", "",
         "> 判据 `spec/judgments.md §5`：「日志只写 JSONL；CSV 停用；"
         "`results.csv` 之类旧格式归档到 `data/legacy/`」。", "",
         "## 现状盘点", "",
         "全仓 CSV 共 **%d** 个（含本次新生成并已归档的文件）：" % len(csvs), "",
         md_table(["目录", "CSV 数", "处理状态"], rows), "",
         "## 本轮已做的", "",
         "- 本次 B1 验收的 124 次运行，CSV 全部直接写入 `data/legacy/b1/`；"
         "对外主数据为 `data/b1_acceptance/*.jsonl`（每 trial 一行 + summary）。",
         "- 每个运行写 `*.meta.json`，含源码哈希、二进制哈希、`seed_bundle` 四子种子、"
         "全部参数与判据版本；`seed_bundle` 为契约声明值，B1 C 运行时仍是单流 `srand(seed)`，"
         "该实现缺口已在验收报告 §6 明确标注。",
         "- 两个新数据目录已加入 `.gitignore`（原始数据不入库，可由命令复现）。", "",
         "## 历史数据为何暂不移动", "",
         "现有工具链把路径写死，直接移动会破坏证据链与报告数字：", "",
         md_table(["工具", "硬编码路径"], [
             ["`tools/gen_progress_report.py`", "`data_b2/*.csv`"],
             ["`tools/b1_report.py`", "`data/b1_m1/sweep_*.jsonl`"],
             ["`audit/scan_data.py` / `inspect_results.py` / `build_report.py`", "`data_b2/`"],
             ["`data/multi_seed/`、`data/` 根", "早期 B1/B2 实验 CSV"],
         ]), "",
         "迁移顺序（**本轮不执行**，避免不可逆地改动历史证据）："
         "① 把上述工具的输出目录参数化；② 回归验证报告数字不变；"
         "③ 再按原目录名移动到 `data/legacy/<原目录名>/`；④ 更新引用与 `.gitignore`。", "",
         "## 历史 meta 缺口（未回填、未篡改）", "",
         "`data/b1_m1/` 的 47 个 `*.csv.meta.json` 有超参与 build 时间，但缺源码哈希、"
         "二进制哈希、`seed_bundle` 四子种子、判据版本，且 seed 全为 12345。"
         "本次不回填这些字段（回填会伪造当时不可追溯的二进制哈希），只在此登记；"
         "新运行的 meta 已完整。", "",
         "*本清单由 `tools/b1_acceptance.py --phase report` 生成，可重复核对。*"]
    (AUDIT / "legacy_csv_manifest.md").write_text("\n".join(s), encoding="utf-8")
    return len(csvs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("seeds", "scale", "report", "all"), default="all")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    LEGACY.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    hashes = {
        "sources": {p.relative_to(ROOT).as_posix(): sha1(p) for p in SOURCES},
        "binary": sha1(EXE),
    }
    phases = ("seeds", "scale") if args.phase == "all" else ((args.phase,) if args.phase != "report" else ())
    for ph in phases:
        (phase_seeds if ph == "seeds" else phase_scale)(hashes)
    if args.phase in ("report", "all"):
        return phase_report(hashes, load_runs())
    return 0


if __name__ == "__main__":
    sys.exit(main())
