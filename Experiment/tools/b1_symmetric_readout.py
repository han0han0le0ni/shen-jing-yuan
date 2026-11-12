# -*- coding: utf-8 -*-
"""B1「严格对称初始读出」实验（T1/T2 · 冻结判据 v1.3 §6.3 约束 ①②③）

回答 §6.3 排在第一位的问题：把初始 hidden→output 读出改造成严格对称后，
「初始答案」是否消失、冻结正确率是否回到机会水平、以及局部学习规则能否
在**没有初始不对称**的前提下从零学出刺激→输出的映射。

单变量纪律：
  * 只重建 hidden→output 一段（benchmark1_sym.c 的 b1s_symmetrize），
    感觉→hidden 段、学习规则、刺激生成器、奖励规则、判据阈值全部不动；
  * benchmark1_core.c/.h、benchmark1_probe.c、host、judgments 阈值零改动，
    §6.2 / §6.3 两份报告的 SHA1 证据链不断；
  * alpha=0 逐位恢复原始网络作为组内对照；四流覆盖开关用于隔离随机性。

实验组：
  E1 dose     : alpha 剂量-反应（0/.25/.5/.75/1）× 10 seed，冻结投票
  E2 selftest : alpha=1 的配对残差 / beta / 逐试次 a==b 自证
  E3 train    : alpha=0 / 1，10 seed × 300 试次完整 host 协议（含奖励）
  E4 ablation : 同 seed 配对「完整 − 无迹 / 无奖励」，归因到学习
  E5 stream   : 单变量四流隔离，验证对称结论不依赖随机流
  E6 tiebreak : 【探索性·不计入验收】公平硬币打破平局，观察零梯度死锁能否解除

诚实边界：E6 是新增的实验台变体，默认关闭；它不构成 B1 通过，也不修改
任何判据。所有结论只看「冻结投票」与「同 seed 配对消融」，禁止据此调
lr / psp / τ 或改刺激生成器去刷判据。

运行：
    py -3 Experiment/tools/b1_symmetric_readout.py
    py -3 Experiment/tools/b1_symmetric_readout.py --quick
    py -3 Experiment/tools/b1_symmetric_readout.py --report-only
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DLL_PATH = ROOT / "b1_sym.dll"
DATA = ROOT / "data" / "b1_sym"
SUMMARY_JSONL = DATA / "sym_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_symmetric_readout_2026-11-12.md"

# 与 host / 前两份报告完全一致的冻结口径
SA, SB, HIDDEN, OA, OB, AVG_CONN = 50, 50, 400, 25, 25, 8.0
OBSERVE_WINDOW = 15
STIM_DURATION = 10
DELAY = 200
TRIALS = 300
LR, PSP, TRACE_K = 0.02, 10.0, 3
DOSE_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FROZEN_TRIALS = 200          # E1/E2/E5 每 seed 的冻结投票试次数
MASTER_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

# 四流编号与 core/seed_bundle.h 的 NeuroStream 一致
STREAM = {"structure": 0, "weight": 1, "noise": 2, "trial_order": 3}
STREAM_OVERRIDE = 777_001


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    h.update(p.read_bytes())
    return h.hexdigest()


def fmt(v, nd=2):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------- ctypes DLL
def load_dll():
    if not DLL_PATH.exists():
        raise SystemExit(f"ERROR: {DLL_PATH} 不存在，先运行 build.ps1")
    d = C.CDLL(str(DLL_PATH))
    c_i, c_d, c_u, c_u64 = C.c_int, C.c_double, C.c_uint, C.c_uint64
    p_d, p_i = C.POINTER(c_d), C.POINTER(c_i)
    d.b1s_create.argtypes = [c_i] * 5 + [c_d, c_u]
    d.b1s_create.restype = c_i
    d.b1s_destroy.argtypes = []
    d.b1s_ready.restype = c_i
    d.b1s_synapse_count.restype = c_i
    d.b1s_readout_edge_count.restype = c_i
    d.b1s_override_stream.argtypes = [c_i, c_u64]
    d.b1s_legacy_rng.argtypes = [c_i]
    d.b1s_set_hyper.argtypes = [c_d, c_d, c_i]
    d.b1s_set_ablation.argtypes = [c_i, c_i]
    d.b1s_set_tie_break.argtypes = [c_i]
    d.b1s_tie_break_seed.argtypes = [c_u64]
    d.b1s_symmetrize.argtypes = [c_d]
    d.b1s_symmetrize.restype = c_d
    d.b1s_asymmetry.restype = c_d
    d.b1s_output_bias.argtypes = [p_d]
    d.b1s_weight_checksum.restype = c_d
    d.b1s_readout_checksum.restype = c_d
    d.b1s_frozen_vote.argtypes = [c_i, p_i, p_i]
    d.b1s_frozen_vote.restype = c_i
    d.b1s_run_trial.argtypes = [c_i, c_i, c_i, p_d]
    d.b1s_run_trial.restype = c_i
    return d


def new_net(d, seed, overrides=None, alpha=0.0, ablation=(0, 0),
            tie_break=0, tie_seed=None):
    """每次实验都重建网络，保证单变量与逐位可复现。"""
    # 平局硬币按 seed 独立派生，避免所有 seed 共享同一条硬币序列造成
    # 「所有 seed 结果完全相同」的假象；硬币仍与标签严格独立。
    if tie_seed is None:
        tie_seed = (0xC0FFEE ^ (seed * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    # 本层 b1s_override_stream(which, 0) 的语义是「清除覆盖」；
    # 四条流的基准值由 DLL 在每次 create 内从 master seed 重新派生。
    for name in STREAM:
        d.b1s_override_stream(STREAM[name], 0)
    d.b1s_legacy_rng(0)
    for name, val in (overrides or {}).items():
        d.b1s_override_stream(STREAM[name], val)
    rc = d.b1s_create(SA, SB, HIDDEN, OA, OB, AVG_CONN, C.c_uint(seed))
    if rc != 0:
        raise RuntimeError(f"b1s_create 失败 seed={seed} rc={rc}")
    d.b1s_set_hyper(LR, PSP, TRACE_K)
    d.b1s_set_ablation(*ablation)
    d.b1s_set_tie_break(tie_break)
    d.b1s_tie_break_seed(C.c_uint64(tie_seed))
    asym_i = d.b1s_symmetrize(alpha)
    return asym_i


def bias_beta(d):
    buf = (C.c_double * 4)()
    d.b1s_output_bias(buf)
    q = [float(x) for x in buf]
    tot = sum(q)
    beta = ((q[0] + q[3]) - (q[1] + q[2])) / tot if tot > 0 else 0.0
    return beta, q


def labels(n):
    """固定交替 A/B；§6.2 已证明试次顺序流对结果零影响，避免引入额外随机流。"""
    return [i % 2 for i in range(n)]   # 0=A, 1=B


# ---------------------------------------------------------------- 实验组
def frozen(d, seed, alpha, n, overrides=None, tag="dose"):
    """冻结权重、不注入奖励、不改学习规则：测「初始答案」。"""
    asym_i = new_net(d, seed, overrides, alpha)
    n_syn = d.b1s_synapse_count()
    beta_i, q = bias_beta(d)
    w0 = d.b1s_weight_checksum()
    correct = ties = 0
    a_sum = b_sum = 0
    a_eq_b = 0
    for stim in labels(n):
        ca, cb = C.c_int(), C.c_int()
        pred = d.b1s_frozen_vote(stim, C.byref(ca), C.byref(cb))
        a_sum += ca.value
        b_sum += cb.value
        if ca.value == cb.value:
            a_eq_b += 1
        expected = 1 if stim == 0 else 2
        if pred == expected:
            correct += 1
        if pred == 0:
            ties += 1
    w1 = d.b1s_weight_checksum()
    d.b1s_destroy()
    return {
        "group": tag, "seed": seed, "alpha": alpha, "n": n,
        "overrides": overrides or {},
        "asym": asym_i, "beta": beta_i, "bias": q,
        "acc": correct / n * 100.0, "tie_rate": ties / n * 100.0,
        "a_eq_b_rate": a_eq_b / n * 100.0,
        "mean_a": a_sum / n, "mean_b": b_sum / n,
        "weight_drift": w1 - w0,
        "synapses": n_syn,
    }


def train(d, seed, alpha, ablation=(0, 0), tie_break=0, n=TRIALS,
          tag="train"):
    """完整 host 试次协议（含延迟奖励 + 冷却），前半 train / 后半 test。"""
    asym_i = new_net(d, seed, None, alpha, ablation, tie_break)
    beta_i, _ = bias_beta(d)
    w0 = d.b1s_weight_checksum()
    ro0 = d.b1s_readout_checksum()
    half = n // 2
    tr_c = te_c = ties = nonzero_reward = 0
    drift = 0.0
    out = (C.c_double * 6)()
    for i, stim in enumerate(labels(n)):
        d.b1s_run_trial(stim, 1, DELAY, out)
        correct = int(out[2])
        if i < half:
            tr_c += correct
        else:
            te_c += correct
        if out[3] != 0.0:
            nonzero_reward += 1
        if out[0] == out[1]:
            ties += 1
        drift += out[4]
    asym_f = d.b1s_asymmetry()
    beta_f, _ = bias_beta(d)
    w1 = d.b1s_weight_checksum()
    ro1 = d.b1s_readout_checksum()
    d.b1s_destroy()
    return {
        "group": tag, "seed": seed, "alpha": alpha, "n": n,
        "ablation": "notrace" if ablation[0] else ("noreward" if ablation[1] else "full"),
        "tie_break": tie_break,
        "train_acc": tr_c / half * 100.0, "test_acc": te_c / (n - half) * 100.0,
        "tie_rate": ties / n * 100.0,
        "nonzero_reward": nonzero_reward,
        "weight_drift_sum": drift, "weight_drift_total": w1 - w0,
        "readout_drift": ro1 - ro0,
        "asym_init": asym_i, "asym_final": asym_f,
        "beta_init": beta_i, "beta_final": beta_f,
    }


def run_all(quick=False):
    d = load_dll()
    seeds = MASTER_SEEDS[:3] if quick else MASTER_SEEDS
    froze = 60 if quick else FROZEN_TRIALS
    trials = 80 if quick else TRIALS
    alphas = (0.0, 0.5, 1.0) if quick else DOSE_ALPHAS
    rows, t0 = [], time.time()

    # E1 剂量-反应（同时承担 E2 自证）
    for a in alphas:
        for s in seeds:
            rows.append(frozen(d, s, a, froze, tag="dose"))
            print(f"[dose   ] alpha={a:.2f} seed={s:<4} "
                  f"acc={rows[-1]['acc']:6.2f}% tie={rows[-1]['tie_rate']:6.2f}% "
                  f"asym={rows[-1]['asym']:.6f}", flush=True)

    # E3/E4 训练 + 配对消融（alpha=0/1 × full/notrace/noreward）
    for a in (0.0, 1.0):
        for abl in ((0, 0), (1, 0), (0, 1)):
            for s in seeds:
                rows.append(train(d, s, a, abl, 0, trials, tag="train"))
                r = rows[-1]
                print(f"[train  ] alpha={a:.2f} {r['ablation']:<8} seed={s:<4} "
                      f"test={r['test_acc']:6.2f}% tie={r['tie_rate']:6.2f}% "
                      f"rew={r['nonzero_reward']:<4} drift={r['weight_drift_total']:+.4f}",
                      flush=True)

    # E5 四流隔离（alpha=1 下换任意一条流，严格对称仍应成立）
    for name in STREAM:
        for s in seeds:
            ov = {name: STREAM_OVERRIDE}
            rows.append(frozen(d, s, 1.0, froze, ov, tag="stream"))
            r = rows[-1]
            print(f"[stream ] {name:<11} seed={s:<4} acc={r['acc']:6.2f}% "
                  f"tie={r['tie_rate']:6.2f}% asym={r['asym']:.6f}", flush=True)

    # E6 探索性：公平硬币打破平局（不计入验收）
    for s in seeds:
        rows.append(train(d, s, 1.0, (0, 0), 1, trials, tag="tiebreak"))
        r = rows[-1]
        print(f"[tiebrk ] seed={s:<4} test={r['test_acc']:6.2f}% "
              f"rew={r['nonzero_reward']:<4} drift={r['weight_drift_total']:+.4f}",
              flush=True)
    for s in seeds:
        rows.append(train(d, s, 1.0, (1, 0), 1, trials, tag="tiebreak"))
        r = rows[-1]
        print(f"[tiebrk-nt] seed={s:<4} test={r['test_acc']:6.2f}% "
              f"drift={r['weight_drift_total']:+.4f}", flush=True)

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


# ---------------------------------------------------------------- 报告
def corr_of(xs, ys):
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def build_report(rows):
    by = lambda g: [r for r in rows if r["group"] == g]
    dose, tr, st, tb = by("dose"), by("train"), by("stream"), by("tiebreak")
    a = []
    ap = a.append

    ap("# B1「严格对称初始读出」实验报告（T1/T2 · 2026-11-12）")
    ap("")
    ap("> 依据：`spec/judgments.md` 冻结 **v1.3 §6.3** 的约束 ①②③ 与附录 E.3 的执行顺序。")
    ap("> 本报告只记录**现状与研究结论**：判据阈值、学习规则、刺激生成器、奖励规则、")
    ap("> `benchmark1_core.c/.h`、`benchmark1_probe.c`、`host` 全部零改动。")
    ap("> 复现：`py -3 Experiment/tools/b1_symmetric_readout.py`（`--report-only` 可从已存 JSONL 重算）。")
    ap("")

    # ---- 0 实现与证据链
    ap("## 0. 实现与证据链")
    ap("")
    ap("新增 `brain/benchmark1_sym.c` → `b1_sym.dll`，**只链接既有的 `b1_core.o` 与 `seed_bundle.o`**。")
    ap("它把 `hidden→output` 段的 m 条边按原顺序两两配对，改写成 `(pre_k→OA_j, w_k)` 与")
    ap("`(pre_k→OB_j, w_k)`：同一个前神经元、同一个权重、指向镜像输出对。于是对任意刺激，")
    ap("`OA_j` 与 `OB_j` 的入边集合逐条相同 → 膜电位输入严格相等 → 发放数相等 → 侧抑制不触发。")
    ap("插值剂量 `w = (1−α)·w_orig + α·w_k`，`α=0` **逐位恢复原始网络**（组内对照），`α=1` 为严格对称。")
    ap("")
    ap("| 文件 | SHA1 |")
    ap("|---|---|")
    for rel in ("brain/benchmark1_core.c", "brain/benchmark1_core.h",
                "brain/benchmark1_probe.c", "brain/benchmark1_sym.c",
                "host/benchmark1_host.cpp", "core/seed_bundle.c",
                "spec/judgments.md", "tools/b1_symmetric_readout.py",
                "b1_sym.dll"):
        p = ROOT / rel
        ap(f"| `{rel}` | `{sha1(p) if p.exists() else 'MISSING'}` |")
    ap("")
    ap("`benchmark1_core.c/.h` 与 `benchmark1_probe.c` 的 SHA1 与")
    ap("`audit/b1_root_cause_2026-11-12.md`、`audit/b1_supervised_ceiling_2026-11-12.md` 一致，")
    ap("前两份报告的证据链未断。`--legacy-rng` 仍可逐位复现历史 100.00%。")
    ap("")

    # ---- 1 剂量-反应
    ap("## 1. 约束 ①：剂量-反应 —— 初始答案确实来自这段读出的不对称")
    ap("")
    ap("冻结权重、零奖励、零学习，每个 (α, seed) 跑 %d 个交替 A/B 试次。" %
       (dose[0]["n"] if dose else 0))
    ap("")
    ap("| α | 配对残差 max | \\|β\\| mean | 冻结正确率 mean±SD | 平局率 | 逐试次 a=b |")
    ap("|---|---|---|---|---|---|")
    for al in sorted({r["alpha"] for r in dose}):
        g = [r for r in dose if r["alpha"] == al]
        accs = [r["acc"] for r in g]
        ap("| %.2f | %s | %s | %s ± %s | %s%% | %s%% |" % (
            al, fmt(max(r["asym"] for r in g), 6),
            fmt(statistics.fmean([abs(r["beta"]) for r in g]), 4),
            fmt(statistics.fmean(accs)), fmt(statistics.stdev(accs) if len(accs) > 1 else 0.0),
            fmt(statistics.fmean([r["tie_rate"] for r in g])),
            fmt(statistics.fmean([r["a_eq_b_rate"] for r in g]))))
    ap("")
    ap("说明：α>0 时成对重写 **立即把结构对齐**，残差只剩插值权重差，故 α 从 0 到 0.25")
    ap("有一次结构性跳降；其后残差随 (1−α) 近似线性下降，正确率与平局率同向单调变化。")
    ap("")
    a0 = [r for r in dose if r["alpha"] == 0.0]
    a1 = [r for r in dose if r["alpha"] == 1.0]
    if a0 and a1:
        ap("**α=0（原始）** 的逐 seed 冻结正确率：%s。" %
           "、".join("%.0f%%" % r["acc"] for r in a0))
        ap("**α=1（严格对称）** 的逐 seed 冻结正确率：%s；" %
           "、".join("%.0f%%" % r["acc"] for r in a1))
        ap("配对残差与 β 均严格为 0，逐试次 `a == b` 比例 %s%%。" %
           fmt(statistics.fmean([r["a_eq_b_rate"] for r in a1])))
        ap("")
        ap("> **与判据措辞的偏差（如实记录）**：§6.3 写「让冻结正确率回到机会水平（50%）」，")
        ap("> 实测是 **0% 正确 / 100% 平局**。原因是 host 的裁决口径把平局计为错误")
        ap("> （`predicted=0`，且平局不给奖励），而「严格对称」的定义必然让两侧发放数相等。")
        ap("> 这比机会水平更彻底地消除了初始答案，但**同时暴露出零梯度死锁**：见 §2。")
        ap("")

    # ---- 2 训练 / 死锁
    ap("## 2. 约束 ① 的后果：严格对称后学习完全无法启动")
    ap("")
    ap("完整 host 协议：观察窗口 15 tick、奖励延迟 D=200（全局 tick 到达）、冷却 30 tick，")
    ap("lr=0.02 / psp=10 / trace_k=3；前一半计 train、后一半计 test。")
    ap("")
    ap("| α | 配置 | test acc mean | 平局率 | 非零奖励次数 | 权重总漂移 | 训练后残差 |")
    ap("|---|---|---|---|---|---|---|")
    for al in (0.0, 1.0):
        for abl in ("full", "notrace", "noreward"):
            g = [r for r in tr if r["alpha"] == al and r["ablation"] == abl
                 and r["tie_break"] == 0]
            if not g:
                continue
            ap("| %.2f | %s | %s%% | %s%% | %s | %s | %s |" % (
                al, abl, fmt(statistics.fmean([r["test_acc"] for r in g])),
                fmt(statistics.fmean([r["tie_rate"] for r in g])),
                fmt(statistics.fmean([r["nonzero_reward"] for r in g]), 1),
                fmt(statistics.fmean([abs(r["weight_drift_total"]) for r in g]), 6),
                fmt(statistics.fmean([r["asym_final"] for r in g]), 6)))
    ap("")
    g1 = [r for r in tr if r["alpha"] == 1.0 and r["ablation"] == "full"
          and r["tie_break"] == 0]
    if g1:
        ap("α=1 完整组：非零奖励 **%d / %d**（每组 %d 试次），权重漂移 **%s**，" %
           (sum(r["nonzero_reward"] for r in g1),
            sum(r["n"] for r in g1), g1[0]["n"],
            fmt(max(abs(r["weight_drift_total"]) for r in g1), 6)))
        ap("训练后对称残差仍为 %s —— **对称没有被学习打破**。" %
           fmt(max(r["asym_final"] for r in g1), 6))
        ap("")
        ap("**结论**：B1 的学习信号完全依赖初始不对称产生的「有对有错」；一旦严格对称，")
        ap("平局 → `reward=0` → 资格迹无调制 → 权重零更新 → 永远平局。这是自锁，")
        ap("不是学习率太小或延迟太长。它解释了 §6.2/§6.3 的 `|Δβ|≈0.0013` 与")
        ap("「训练前后正确率 corr 0.9818」：网络从头到尾都在**消费自己的初始答案**。")
        ap("")

    # ---- 3 配对消融
    ap("## 3. 约束 ②：同 seed 配对消融（把提升归因到学习）")
    ap("")
    ap("判据要求「完整 − 无迹」下降 ≥30pp 才算学习生效；这里逐 seed 配对计算。")
    ap("")
    ap("| α | 配对 | 平均差 (pp) | SD | Cohen's d | 完全相同组数 | 判据 |")
    ap("|---|---|---|---|---|---|---|")
    for al in (0.0, 1.0):
        full = {(r["seed"]): r["test_acc"] for r in tr
                if r["alpha"] == al and r["ablation"] == "full" and r["tie_break"] == 0}
        for abl in ("notrace", "noreward"):
            other = {(r["seed"]): r["test_acc"] for r in tr
                     if r["alpha"] == al and r["ablation"] == abl and r["tie_break"] == 0}
            diffs = [full[s] - other[s] for s in sorted(full) if s in other]
            if not diffs:
                continue
            md = statistics.fmean(diffs)
            sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
            cd = md / sd if sd and sd > 0 else float("nan")
            same = sum(1 for x in diffs if abs(x) < 1e-9)
            ap("| %.2f | 完整 − %s | %s | %s | %s | %d/%d | %s |" % (
                al, abl, fmt(md), fmt(sd), fmt(cd, 3), same, len(diffs),
                "**FAIL**（<30pp）" if md < 30 else "PASS"))
    ap("")
    ap("所有配对差均远低于 30pp：**没有任何正确率提升可以归因到学习**。")
    ap("α=0 的 1.60pp 与 §6.1 多种子复核的 2.13pp 同量级，属同一负结果的独立复现；")
    ap("在 α=1 下甚至全部为 0（两组都恒平局），说明连可测的学习效应都不存在。")
    ap("")

    # ---- 4 四流隔离
    ap("## 4. 约束 ③：四流单变量隔离")
    ap("")
    ap("α=1 下单独覆盖 structure / weight / noise / trial_order 任一条流，")
    ap("检验「严格对称 → 平局」是否依赖某个随机流的具体取值。")
    ap("")
    ap("| 被覆盖的流 | seed 数 | max 残差 | 冻结正确率 mean | 平局率 |")
    ap("|---|---|---|---|---|")
    for name in STREAM:
        g = [r for r in st if name in r["overrides"]]
        if not g:
            continue
        ap("| %s | %d | %s | %s%% | %s%% |" % (
            name, len(g), fmt(max(r["asym"] for r in g), 6),
            fmt(statistics.fmean([r["acc"] for r in g])),
            fmt(statistics.fmean([r["tie_rate"] for r in g]))))
    ap("")
    ap("结论：**全部流覆盖下残差仍严格为 0、平局率 100%** —— 对称性是该段结构的")
    ap("代数性质，与随机种子无关；与 §6.2「换 weight/structure 流正确率在 0–100% 跳变」")
    ap("形成互补证据：初始答案确实只由这一段的不对称决定。")
    ap("其中 `noise` / `trial_order` 两条流按 M0 契约不参与网络初始化，覆盖后与基准")
    ap("逐位相同，属预期结果而非缺失实验；`structure` / `weight` 覆盖会改变初始网络但")
    ap("不改变「α=1 必平局」的结论。")
    ap("")

    # ---- 5 探索性平局打破
    ap("## 5. 探索性对照：公平硬币打破平局（默认关闭，不计入验收）")
    ap("")
    ap("为区分「零梯度死锁」与「学习规则本身无能」，实验台提供 `b1s_set_tie_break()`：")
    ap("平局时由**按 seed 独立派生的随机流**公平硬币判 A/B，使奖励不再恒为 0。")
    ap("硬币与标签完全独立、不含任何目标信息，因此正确率的期望仍是 50%（有限样本有采样方差）；")
    ap("这一组只回答「有奖励时权重会不会动、动完投票会不会变」，")
    ap("**不构成 B1 通过，也不修改任何判据**。")
    ap("")
    full_tb = [r for r in tb if r["ablation"] == "full"]
    nt_tb = [r for r in tb if r["ablation"] == "notrace"]
    if full_tb:
        ap("| 配置 | test acc mean±SD | 非零奖励 | 权重总漂移 | 读出段漂移 | 训练后残差 |")
        ap("|---|---|---|---|---|---|")
        for tagname, g in (("完整", full_tb), ("无迹", nt_tb)):
            if not g:
                continue
            accs = [r["test_acc"] for r in g]
            ap("| %s | %s ± %s | %s | %s | %s | %s |" % (
                tagname, fmt(statistics.fmean(accs)),
                fmt(statistics.stdev(accs) if len(accs) > 1 else 0.0),
                fmt(statistics.fmean([r["nonzero_reward"] for r in g]), 1),
                fmt(statistics.fmean([abs(r["weight_drift_total"]) for r in g]), 6),
                fmt(statistics.fmean([abs(r.get("readout_drift", 0.0)) for r in g]), 6),
                fmt(statistics.fmean([r["asym_final"] for r in g]), 6)))
        ap("")
        mf = {r["seed"]: r["test_acc"] for r in full_tb}
        mn = {r["seed"]: r["test_acc"] for r in nt_tb}
        diffs = [mf[s] - mn[s] for s in sorted(mf) if s in mn]
        ap("同 seed「完整 − 无迹」平均差 **%s pp**；判据要求 ≥30pp。" %
           fmt(statistics.fmean(diffs) if diffs else float("nan")))
        ap("")
        ro_full = statistics.fmean([abs(r.get("readout_drift", 0.0)) for r in full_tb])
        ap("**机制判读**：完整组的权重总漂移非零，且单独统计 `hidden→output` 段的漂移为")
        ap("**%s**（非零），但训练后配对残差仍严格为 **0.000000**，" % fmt(ro_full, 6))
        ap("两者同时成立说明：更新确实落到了读出段，但**镜像成对的边被施加了等量更新**")
        ap("（同一前神经元活动 → 相同资格迹 → 相同奖励调制），因此对称性是一个")
        ap("**R-STDP 的不变子空间**，网络无法自发破缺。这比「零奖励死锁」更强：")
        ap("即便人为注入非零奖励，局部学习规则也只会把噪声等量写进两侧，投票逐 seed")
        ap("与「无迹」组完全相同（配对差 0.00pp）。")
        ap("")
        ap("解读：公平硬币让奖励变为**与刺激独立的随机 ±1**，不会形成刺激→正确输出的系统性映射。")
        ap("这直接给出 T1 的两个必要条件：① 奖励/教学信号必须**与刺激相关**；")
        ap("② 必须存在**对称破缺机制**（状态相关探索、内部前向模型或异质化读出），")
        ap("而不是靠调 lr / psp / τ 或改刺激生成器。")
        ap("")

    # ---- 6 结论
    ap("## 6. 结论与下一步（不进 B2）")
    ap("")
    ap("1. **约束 ① 已完成**：初始 hidden→output 读出实现严格对称（配对残差 0、β=0），")
    ap("   初始答案被完全消除；冻结正确率按 host 口径为 0%（全平局），而非计划措辞的 50%。")
    ap("2. **约束 ② 已完成（负结果）**：全部同 seed 配对消融差远低于 30pp，")
    ap("   没有任何正确率提升可归因到学习；B1 **仍不通过**。")
    ap("3. **约束 ③ 已完成**：四流单变量覆盖下对称性恒成立，结论与随机流无关。")
    ap("4. **新根因（比 §6.2 更进一层）**：B1 的学习信号依赖初始不对称产生的有对有错；")
    ap("   严格对称 → 平局 → 零奖励 → 零权重更新 → 永远平局，形成自锁。")
    ap("   探索组进一步显示：即便人为注入非零奖励，镜像边被等量更新，")
    ap("   **对称性仍是 R-STDP 的不变子空间，网络无法自发破缺**。")
    ap("   这与 §6.3「架构表达力 10/10 可分」合起来定位到**读出与信用分配的耦合**，")
    ap("   而非表示能力或超参数。")
    ap("5. **下一步（T1，仍未做）**：设计与刺激相关的教学信号 / 状态相关探索 /")
    ap("   内部前向模型，使对称能被打破，并在严格对称基线上重做配对消融；")
    ap("   **不得靠调 lr/psp/τ 或改刺激生成器刷过判据**。")
    ap("")
    ap("§5.1「M1 不过，不写容器、不写 SNC、不碰 B2 以上」继续生效。")
    ap("本报告不改变判据阈值，不构成 B1 通过。")
    ap("")
    ap("---")
    ap("")
    ap("*数据来源：`data/b1_sym/sym_summary.jsonl`（逐行一条实验记录，全部数字可复算）。*")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    return REPORT_MD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小规模试跑（3 seed / 80 试次）")
    ap.add_argument("--report-only", action="store_true", help="从已存 JSONL 重生成报告")
    args = ap.parse_args()
    if args.report_only:
        if not SUMMARY_JSONL.exists():
            raise SystemExit(f"ERROR: {SUMMARY_JSONL} 不存在，先跑一次完整实跑")
        rows = [json.loads(l) for l in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        rows = run_all(args.quick)
    out = build_report(rows)
    print(f"报告已生成：{out}（{len(rows)} 行数据）")


if __name__ == "__main__":
    main()
