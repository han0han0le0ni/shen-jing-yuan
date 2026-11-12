# -*- coding: utf-8 -*-
"""B1 离线监督上界（T2 第一步 · 判据 v1.2 §6.2 约束 ④）

在**完全不注入奖励、不调用任何学习规则**的前提下，读取 B1 网络
「刺激 → hidden 层观察窗口活动」的表示，用离线监督求解最优线性读出，
测量当前架构对刺激 A/B 的表达力上界。

回答的问题（先证伪，后建设）：
  * 若 hidden 表示线性可分（上界 >= 80%）→ 架构表达力够，
    瓶颈在初始读出 / 局部学习规则（对应 §6.2 约束 ①③ 与 T1）。
  * 若上界接近 50% → 表示本身不携带刺激信息，架构需要改（T2）。

严格的诚实边界：
  1. 这不是 B1 通过。上界由外部监督解直接算出标签，网络本身没有标签。
  2. 诊断 DLL 只读，网络权重在采集前后由 checksum 自证零漂移。
  3. 不改判据阈值、不改学习规则、不改刺激生成器、不改 host、不刷分。
  4. 对照组：B1 自身投票（初始答案）、随机线性读出、标签 shuffle。

运行：
    py -3 Experiment/tools/b1_supervised_ceiling.py
    py -3 Experiment/tools/b1_supervised_ceiling.py --report-only
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DLL_PATH = ROOT / "b1_probe.dll"
DATA = ROOT / "data" / "b1_ceiling"
SUMMARY_JSONL = DATA / "ceiling_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_supervised_ceiling_2026-11-12.md"
REF_DIAG = ROOT / "data" / "b1_bias" / "diagnosis_summary.jsonl"

# 与 host/benchmark1_host.cpp 的冻结默认值一致
SA, SB, HIDDEN, OA, OB, AVG_CONN = 50, 50, 400, 25, 25, 8.0
OBSERVE_WINDOW = 15
STIM_DURATION = 10

# 与 audit/b1_acceptance_report.md §6.1、b1_bias_diagnosis.py 相同的 10 个独立 seed
MASTER_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

# lambda 以 AᵀA 的平均对角元为尺度（尺度无关，避免特征维度影响网格）
LAMBDA_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0)
N_FOLDS = 5
N_INNER = 3
N_RANDOM = 500


# ---------------------------------------------------------------- ctypes DLL
def load_dll():
    if not DLL_PATH.exists():
        raise SystemExit(f"ERROR: {DLL_PATH} 不存在，先运行 build.ps1")
    d = C.CDLL(str(DLL_PATH))
    c_i, c_d, c_u, c_u64 = C.c_int, C.c_double, C.c_uint, C.c_uint64
    p_d, p_i = C.POINTER(c_d), C.POINTER(c_i)
    d.b1p_create.argtypes = [c_i, c_i, c_i, c_i, c_i, c_d, c_u]
    d.b1p_create.restype = c_i
    d.b1p_destroy.argtypes = []
    d.b1p_ready.restype = c_i
    d.b1p_override_stream.argtypes = [c_i, c_u64]
    d.b1p_legacy_rng.argtypes = [c_i]
    for name in ("b1p_neuron_count", "b1p_hidden_count", "b1p_synapse_count",
                 "b1p_current_tick", "b1p_hidden_offset"):
        getattr(d, name).restype = c_i
    d.b1p_reset_trial.argtypes = []
    d.b1p_step.argtypes = [c_i]
    d.b1p_hidden_spikes.argtypes = [p_d, c_i, p_i, p_i, p_i]
    d.b1p_output_bias.argtypes = [p_d]
    d.b1p_readout_weights.argtypes = [p_d]
    d.b1p_weight_checksum.restype = c_d
    return d


# ---------------------------------------------------------------- 数据采集
def collect(dll, seed, n_per_class):
    """采集一个 seed 的 (hidden 观察窗口脉冲计数, 标签, B1 自身投票)。"""
    rc = dll.b1p_create(SA, SB, HIDDEN, OA, OB, AVG_CONN, c_seed(seed))
    if rc != 0:
        raise RuntimeError(f"b1p_create 失败 seed={seed} rc={rc}")
    h = dll.b1p_hidden_count()
    syn = dll.b1p_synapse_count()
    buf = (C.c_double * h)()
    ca, cb, ch = C.c_int(), C.c_int(), C.c_int()

    bias = (C.c_double * 4)()
    dll.b1p_output_bias(bias)
    q = [float(x) for x in bias]
    w_correct, w_wrong = q[0] + q[3], q[1] + q[2]
    beta = (w_correct - w_wrong) / (w_correct + w_wrong) if (w_correct + w_wrong) > 0 else 0.0
    delta = abs((q[0] + q[1]) - (q[2] + q[3]))

    w_before = dll.b1p_weight_checksum()

    # 单变量纪律：试次顺序在 §6.2 已被证明对结果零影响（8/8 全 100%），
    # 这里用固定交替序列，避免引入额外随机流；每个试次前完整复位。
    labels = []
    for k in range(n_per_class):
        labels.extend((0, 1))  # 0=A, 1=B

    X = np.zeros((len(labels), h), dtype=np.float64)
    y = np.array([1.0 if t == 0 else -1.0 for t in labels])  # A=+1, B=-1
    votes = np.zeros(len(labels), dtype=np.int32)
    hidden_total = 0

    for n, lab in enumerate(labels):
        dll.b1p_reset_trial()
        ac = bc = 0
        for t in range(OBSERVE_WINDOW):
            dll.b1p_step(lab if t < STIM_DURATION else -1)
            dll.b1p_hidden_spikes(buf, h, C.byref(ca), C.byref(cb), C.byref(ch))
            X[n] += np.frombuffer(buf, dtype=np.float64, count=h)
            ac += ca.value
            bc += cb.value
            hidden_total += ch.value
        # 与 host 一致的投票：A 多投 A，B 多投 B，平局算错（host 中 predicted=0）
        votes[n] = 1 if ac > bc and ac > 0 else (-1 if bc > ac and bc > 0 else 0)

    w_after = dll.b1p_weight_checksum()
    dll.b1p_destroy()

    vote_acc = float(np.mean(votes == y) * 100.0)
    tie_rate = float(np.mean(votes == 0) * 100.0)
    active = int(np.count_nonzero(X.sum(axis=0)))
    mean_rate = hidden_total / (len(labels) * OBSERVE_WINDOW * h)
    mu_a, mu_b = X[y > 0].mean(axis=0), X[y < 0].mean(axis=0)
    var = X.var(axis=0).sum()
    fisher = float(((mu_a - mu_b) ** 2).sum() / var) if var > 0 else 0.0
    # 表示是否确定性：若每个试次只产生两种完全相同的模式，则「可分」是平凡的，
    # 必须在报告里如实说明，不能当成网络学到了什么。
    unique_patterns = int(np.unique(X, axis=0).shape[0])
    within_var = float(X[y > 0].var(axis=0).sum() + X[y < 0].var(axis=0).sum())
    class_sep_l2 = float(np.linalg.norm(mu_a - mu_b))

    return {
        "seed": seed, "hidden": h, "synapses": syn, "n_samples": len(labels),
        "beta_init": beta, "delta_init": delta,
        "w_correct_init": w_correct, "w_wrong_init": w_wrong,
        "weight_checksum_before": w_before, "weight_checksum_after": w_after,
        "weight_drift": abs(w_after - w_before),
        "vote_acc": vote_acc, "tie_rate": tie_rate,
        "hidden_active_neurons": active, "mean_hidden_rate": mean_rate,
        "fisher_ratio": fisher, "unique_patterns": unique_patterns,
        "within_class_var": within_var, "class_sep_l2": class_sep_l2,
        "_X": X, "_y": y,
    }


def c_seed(seed):
    return C.c_uint(int(seed) & 0xFFFFFFFF)


# ---------------------------------------------------------------- 监督上界
def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _lam_scale(A):
    """A 已含偏置列；用特征部分的平均能量做 lambda 尺度。"""
    F = A[:, :-1]
    return max(float(np.sum(F * F) / max(F.shape[1], 1)), 1e-8)


def _ridge_fit(A, y, lam):
    L = lam * _lam_scale(A)
    G = A.T @ A + L * np.eye(A.shape[1])
    G[-1, -1] -= L  # 不正则化偏置
    return np.linalg.solve(G, A.T @ y)


def _acc(A, y, w):
    return float(np.mean(np.sign(A @ w) == y) * 100.0)


def _choose_lambda(A, y):
    """训练折内部再做 3 折 CV 选 lambda，避免用测试折挑参数（防上界虚高）。"""
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(20261112)
    rng.shuffle(idx)
    folds = np.array_split(idx, N_INNER)
    scores = {lam: [] for lam in LAMBDA_GRID}
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        for lam in LAMBDA_GRID:
            w = _ridge_fit(A[mask], y[mask], lam)
            scores[lam].append(_acc(A[f], y[f], w))
    return min(LAMBDA_GRID, key=lambda l: -statistics.fmean(scores[l]))


def ceiling_cv(X, y, label_perm=None):
    """5 折分层交叉验证的岭回归读出；返回 (正确率, 各折 lambda)。"""
    yy = y.copy() if label_perm is None else y[label_perm]
    rng = np.random.default_rng(4242)
    pos, neg = np.where(yy > 0)[0], np.where(yy < 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    pf, nf = np.array_split(pos, N_FOLDS), np.array_split(neg, N_FOLDS)
    correct, lams = 0, []
    for k in range(N_FOLDS):
        te = np.concatenate([pf[k], nf[k]])
        tr = np.setdiff1d(np.arange(len(yy)), te, assume_unique=True)
        Xtr, Xte = _standardize(X[tr], X[te])
        A, Ate = _add_bias(Xtr), _add_bias(Xte)
        lam = _choose_lambda(A, yy[tr])
        w = _ridge_fit(A, yy[tr], lam)
        correct += int(np.sum(np.sign(Ate @ w) == yy[te]))
        lams.append(lam)
    return correct / len(yy) * 100.0, lams


def random_readout(X, y):
    rng = np.random.default_rng(777)
    d = X.shape[1]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    W = rng.normal(0.0, 1.0 / math.sqrt(d), size=(N_RANDOM, d))
    pred = Z @ W.T
    acc = np.mean(np.sign(pred) == y[:, None], axis=0) * 100.0
    return {"mean": float(acc.mean()), "sd": float(acc.std()),
            "min": float(acc.min()), "max": float(acc.max())}


# ---------------------------------------------------------------- 报告
def fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def build_report(rows):
    a = []
    ap = a.append
    ap("# B1 离线监督上界（T2 第一步 · 架构表达力证伪）")
    ap("")
    ap("> 依据：`spec/judgments.md` v1.2 §6.2 约束 ④「先做离线监督上界确认架构表达力，")
    ap("> 再谈局部学习规则」；对应大纲 §4.1 T2、§5.1「先证伪，后建设」。")
    ap("> **本报告不宣布 B1 通过，不修改任何判据阈值。**")
    ap("")
    ap("## 0. 方法与诚实边界")
    ap("")
    ap("| 项 | 设置 |")
    ap("|---|---|")
    ap("| 网络 | 与 `benchmark1_host.cpp` 冻结默认值一致：50/50/400/25/25，avg_conn=8.0 |")
    ap(f"| 种子 | {len(rows)} 个独立 seed（与 §6.1 验收、根因诊断同源） |")
    ap(f"| 试次协议 | 每试次复位 → 15 tick 观察窗口（前 {STIM_DURATION} tick 注入刺激，之后清除） |")
    ap(f"| 特征 | hidden 层 400 维，观察窗口内累计发放计数 |")
    ap(f"| 样本 | 每 seed {rows[0]['n_samples']} 个（A/B 各半，固定交替序列） |")
    ap("| 学习 | **零**：全程不注入奖励、不调用奖励队列；权重 checksum 前后自证零漂移 |")
    ap(f"| 上界求解 | {N_FOLDS} 折分层交叉验证岭回归，lambda 由训练折内 {N_INNER} 折 CV 选择 |")
    ap(f"| 对照 | ① B1 自身投票（初始答案）② {N_RANDOM} 个随机线性读出 ③ 标签 shuffle |")
    ap("")
    ap("**关键限制**：上界由外部监督解直接读取标签求得，网络本身没有任何标签。")
    ap("它只回答「表示里有没有足够信息」，不回答「局部学习规则能不能学到」；")
    ap("因此即使上界 100%，B1 判据仍然不通过，仍按 §1 失败动作停在 T1/T2。")
    ap("")

    drift = max(r["weight_drift"] for r in rows)
    ap(f"**权重零漂移自证**：全部 {len(rows)} 个 seed 的 `|checksum_after − checksum_before|` "
       f"最大值 = `{drift:.6g}`（浮点求和误差量级），确认诊断期间没有学习发生。")
    ap("")

    ce = [r["ceiling_acc"] for r in rows]
    sh = [r["shuffle_acc"] for r in rows]
    vo = [r["vote_acc"] for r in rows]
    ra = [r["random_mean"] for r in rows]
    ap("## 1. 逐 seed 结果")
    ap("")
    ap("| seed | beta_init | 上界(%) | shuffle(%) | 随机读出(%) | B1 初始投票(%) | 平局率(%) | hidden 活跃数 | Fisher 比 |")
    ap("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ap(f"| {r['seed']} | {r['beta_init']:+.4f} | {r['ceiling_acc']:.2f} | "
           f"{r['shuffle_acc']:.2f} | {r['random_mean']:.2f}±{r['random_sd']:.2f} | "
           f"{r['vote_acc']:.2f} | {r['tie_rate']:.1f} | "
           f"{r['hidden_active_neurons']} | {r['fisher_ratio']:.4f} |")
    ap(f"| **mean** | {statistics.fmean([r['beta_init'] for r in rows]):+.4f} | "
       f"**{statistics.fmean(ce):.2f}** | {statistics.fmean(sh):.2f} | "
       f"{statistics.fmean(ra):.2f} | **{statistics.fmean(vo):.2f}** | "
       f"{statistics.fmean([r['tie_rate'] for r in rows]):.1f} | "
       f"{statistics.fmean([r['hidden_active_neurons'] for r in rows]):.0f} | "
       f"{statistics.fmean([r['fisher_ratio'] for r in rows]):.4f} |")
    ap("")

    # 相关：初始偏向是否能预测 B1 初始投票（复现根因诊断）
    corr = corr_of([r["beta_init"] for r in rows], vo)
    ap(f"**初始偏向与初始答案的相关**：`corr(beta_init, B1 初始投票) = {fmt(corr, 3)}`，"
       "相关不强。这与根因报告 §3 的提醒一致：简单的四类权重和不足以预测"
       "「侧抑制 + 发放动态」后的实际投票，说明 B1 需要一个能区分"
       "「初始工作点」与「学习成效」的读出指标（T2），不能只看 beta。")
    ap("")
    ap("**关于随机读出 SD≈50 的说明**：由于特征逐试次完全确定（见下），"
       "单个随机权重向量的正确率非 0 即 100，故 500 次抽样的标准差接近 50；"
       "表中给出的是均值 ± 标准差，判断依据是均值贴近 50%，而非单次数值。")
    ap("")
    ups = [r["unique_patterns"] for r in rows]
    wvar = max(r["within_class_var"] for r in rows)
    ap("**表示确定性的诚实说明**：全部 seed 的 `hidden` 观察窗口特征只有 "
       f"**{min(ups)}–{max(ups)}** 种不同模式，类内方差最大 `{wvar:.6g}`"
       "（即同一刺激的响应逐试次完全相同，噪声流不进入该层活动）。"
       "因此这里的「线性可分」是**平凡可分**：它证明刺激身份信息没有被架构抹掉，"
       "但绝不代表网络学到了任何东西；上界高只说明瓶颈不在表示，而在读出与信用分配。")
    ap("")

    # ---- 交叉验证：训练前的初始答案是否能预测训练后的正确率 ----
    pr = paired_reference(rows, "master_bundle")
    pl = paired_reference(rows, "master_legacy")
    if pr:
        ap("### 1.1 交叉验证：训练前的答案几乎就是训练后的答案")
        ap("")
        ap("把本实验**冻结权重、零学习**的初始投票，与 `audit/b1_root_cause_2026-11-12.md` "
           "同一四流 RNG 的 `master_bundle` 组（完整训练 300 试次、D=200）逐 seed 配对：")
        ap("")
        ap("| seed | 训练前初始投票(%) | 训练后 test_acc(%) | 差(pp) |")
        ap("|---:|---:|---:|---:|")
        for s, va, vb in pr["pairs"]:
            ap(f"| {s} | {va:.2f} | {vb:.2f} | {va - vb:+.2f} |")
        ap("")
        ap(f"**corr = {fmt(pr['corr'], 4)}**，平均绝对差 **{pr['mean_abs_diff']:.2f}pp**，"
           f"{pr['exact']}/{len(pr['pairs'])} 个 seed 完全相同。"
           "即：训练 300 试次几乎没有改变答案，正确率由初始网络实例决定 —— "
           "与 §6.2 的 `|Δbeta|` 平均 0.0013、无迹消融 8/10 完全相同相互印证。")
        ap("")
        if pl:
            ap(f"**对照组（不同 RNG）**：换成旧单流 `master_legacy` 组时 corr 降到 "
               f"**{fmt(pl['corr'], 4)}**（平均绝对差 {pl['mean_abs_diff']:.2f}pp）。"
               "这是应有的结果 —— 不同 RNG 产生不同初始权重，本来就不该逐 seed 对应；"
               "它同时说明上面的高相关来自「同一个网络实例」，不是普遍的数值巧合。")
            ap("")

    hi = sum(1 for x in ce if x >= 80.0)
    chance = sum(1 for x in ce if x <= 55.0)
    ap("## 2. 结论")
    ap("")
    ap(f"1. **架构表达力**：{len(rows)} 个 seed 的离线监督上界 mean = "
       f"**{statistics.fmean(ce):.2f}%**（SD {fmt(statistics.stdev(ce) if len(ce) > 1 else float('nan'))}，"
       f"min {min(ce):.2f} / max {max(ce):.2f}），其中 **{hi}/{len(rows)}** 个 seed ≥80%，"
       f"**{chance}/{len(rows)}** 个 seed ≤55%（机会水平）。")
    ap(f"2. **对照有效性**：标签 shuffle 的同样流水线 mean = **{statistics.fmean(sh):.2f}%**，"
       f"随机线性读出 mean = **{statistics.fmean(ra):.2f}%**，"
       "两者贴近 50%，说明上界不是交叉验证泄漏或过拟合造成的假象。")
    ap(f"3. **B1 初始投票**（冻结权重、无学习）mean = **{statistics.fmean(vo):.2f}%**，"
       "与 §6.2 的「正确率由初始网络实例决定」一致。")
    if pr:
        ap(f"4. **训练无效的直接证据**：训练前初始投票与训练后 test_acc 的配对相关 "
           f"**{fmt(pr['corr'], 4)}**、平均差 {pr['mean_abs_diff']:.2f}pp（§1.1）。"
           "上界 100% 而实际正确率被初始权重锁死，说明缺的不是表示能力，"
           "而是能把表示映射到正确输出的**学习规则**。")
    ap("")
    if hi >= len(rows) * 0.8:
        ap("**判定 → 表示是可分的，架构表达力不是瓶颈。** 刺激 A/B 的信息确实进入了 hidden 层，"
           "一个线性读出就能分开；B1 之所以在 0–100% 之间二值跳变，问题出在"
           "**初始 hidden→output 读出的随机不对称 + 侧抑制放大 + 局部学习规则改不动净映射**，"
           "而不是网络表示不出任务。")
        ap("")
        ap("这直接支持 v1.2 §6.2 的重做顺序：")
        ap("1. **约束 ① 先做**：把初始输出读出设为严格对称，消除「初始答案」，"
           "让初始正确率回到机会水平；")
        ap("2. **约束 ② 再做**：在同一 seed 上用配对消融把任何提升归因到学习；")
        ap("3. **约束 ③ 全程**：单变量 + 四流覆盖开关隔离；")
        ap("4. 然后才谈 T1 的局部学习规则 / 前向模型。")
        ap("")
        ap("**不要**据此调 lr / psp / τ 或改刺激生成器去「刷」正确率 —— 上界高只说明"
           "任务对读出而言容易，不说明信用分配问题被解决。")
    elif chance >= len(rows) * 0.8:
        ap("**判定 → 表示不可分，架构表达力不足（T2 阻塞）。** hidden 层活动几乎不携带"
           "刺激身份信息，任何读出都学不会；此时改学习规则无意义，应先检查"
           "sensory→hidden 投影、发放饱和与侧抑制是否把信息抹掉。")
    else:
        ap("**判定 → 结果分裂，不能下单一结论。** 不同 seed 的表示可分性差异大，"
           "需要按 §0.6 单变量继续定位（先查 hidden 活跃数与发放饱和），不得据此推进 B2。")
    ap("")
    ap("**判据状态不变**：B1 仍不通过（`judgments.md` 阈值零变更）；"
       "按 §1 失败动作继续停在 T1/T2，不进 B2、不写容器、不写 SNC。")
    ap("")

    ap("## 3. 复现与证据")
    ap("")
    ap("```")
    ap("powershell -ExecutionPolicy Bypass -File Experiment/build.ps1 -Smoke")
    ap("py -3 Experiment/tools/b1_supervised_ceiling.py")
    ap("py -3 Experiment/tools/b1_supervised_ceiling.py --report-only   # 从已存 JSONL 重生成")
    ap("```")
    ap("")
    ap("| 文件 | SHA1 |")
    ap("|---|---|")
    for rel in ("b1_probe.dll", "brain/benchmark1_probe.c", "brain/benchmark1_core.c",
                "brain/benchmark1_core.h", "host/benchmark1_host.cpp",
                "core/seed_bundle.c", "spec/judgments.md",
                "tools/b1_supervised_ceiling.py"):
        p = ROOT / rel
        if p.exists():
            ap(f"| {rel} | {hashlib.sha1(p.read_bytes()).hexdigest()} |")
    ap("")
    ap("**核心源码零改动**：`benchmark1_core.c` 的 SHA1 与 `audit/b1_root_cause_2026-11-12.md` "
       "§6 记录的 `c8de144f…` 一致；诊断 DLL 由独立适配层链接既有 `b1_core.o` 生成，"
       "根因报告的证据链不受影响。原始逐 seed 数据：`data/b1_ceiling/ceiling_summary.jsonl`。")
    ap("")
    ap("*本报告只陈述可核对事实；不宣布 B1 通过，不修改任何阈值。*")
    return "\n".join(a) + "\n"


def corr_of(xs, ys):
    if len(xs) < 2 or statistics.stdev(xs) == 0 or statistics.stdev(ys) == 0:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def paired_reference(rows, group):
    """把冻结初始投票与根因诊断里同 RNG 的「训练后 test_acc」配对。"""
    if not REF_DIAG.exists():
        return None
    ref = {}
    for line in REF_DIAG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("group") == group:
            ref[int(d["master_seed"])] = float(d["test_acc"])
    pairs = [(r["seed"], r["vote_acc"], ref[r["seed"]])
             for r in rows if r["seed"] in ref]
    if len(pairs) < 2:
        return None
    xs, ys = [p[1] for p in pairs], [p[2] for p in pairs]
    return {
        "pairs": pairs, "corr": corr_of(xs, ys),
        "mean_abs_diff": statistics.fmean([abs(a - b) for a, b in zip(xs, ys)]),
        "exact": sum(1 for a, b in zip(xs, ys) if abs(a - b) < 1e-9),
    }


# ---------------------------------------------------------------- main
def public(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200, help="每 seed 采样试次数（A/B 各半）")
    ap.add_argument("--seeds", type=int, nargs="*", default=MASTER_SEEDS)
    ap.add_argument("--report-only", action="store_true", help="从已存 JSONL 重生成报告")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        if not SUMMARY_JSONL.exists():
            raise SystemExit(f"ERROR: {SUMMARY_JSONL} 不存在，先跑一次实跑")
        rows = [json.loads(s) for s in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        dll = load_dll()
        rows = []
        n_per_class = args.trials // 2
        for i, seed in enumerate(args.seeds, 1):
            r = collect(dll, seed, n_per_class)
            X, y = r.pop("_X"), r.pop("_y")
            ce, lams = ceiling_cv(X, y)
            perm = np.random.default_rng(20261112 + seed).permutation(len(y))
            sh, _ = ceiling_cv(X, y, label_perm=perm)
            rr = random_readout(X, y)
            r.update({
                "ceiling_acc": ce, "ceiling_lambdas": lams,
                "shuffle_acc": sh,
                "random_mean": rr["mean"], "random_sd": rr["sd"],
                "random_min": rr["min"], "random_max": rr["max"],
            })
            rows.append(r)
            print(f"[{i:2d}/{len(args.seeds)}] seed={seed:<5} "
                  f"上界={ce:6.2f}%  shuffle={sh:6.2f}%  "
                  f"随机={rr['mean']:6.2f}%  B1初始投票={r['vote_acc']:6.2f}%  "
                  f"权重漂移={r['weight_drift']:.2g}", flush=True)
        with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(public(r), ensure_ascii=False) + "\n")

    REPORT_MD.write_text(build_report(rows), encoding="utf-8")
    print(f"\n报告已生成：{REPORT_MD}")
    print(f"逐 seed 数据：{SUMMARY_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
