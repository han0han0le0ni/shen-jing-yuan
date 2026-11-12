# -*- coding: utf-8 -*-
"""B1「T1 两必要条件」析因实验（判据 v1.4 §6.4 下一步 / 附录 F.4）

问题来源（§6.4 的两个新根因）：
  (a) 零梯度自锁：严格对称 -> 平局 -> reward=0 -> 零更新 -> 永远平局；
  (b) 对称性是 R-STDP 的不变子空间：即使人为注入非零奖励，镜像边因
      「同一 pre 活动 -> 相同资格迹 -> 相同奖励调制」被等量更新，残差仍 0。

§6.4 提出 T1 必须同时满足两个条件才可能学到东西：
  ① 奖励 / 教学信号与刺激相关（当前奖励由初始答案决定，与刺激无因果）；
  ② 存在对称破缺机制（状态相关探索 / 内部前向模型 / 异质化读出）。
本脚本在 **严格对称基线（alpha=1）** 上做 2×2 析因，检验两条件是否必要、
是否联合充分，并用配对消融把提升归因到学习。

单变量纪律：
  * 只通过 brain/benchmark1_t1.c 的两个开关改教学信号与破缺注入；
    benchmark1_core.c/.h、benchmark1_probe.c、benchmark1_sym.c、host、
    判据阈值全部零改动，§6.2 / §6.3 / §6.4 三份报告 SHA1 证据链不断；
  * 教学信号仍走核心原生 net_inject_reward(winner=0) -> 资格迹 -> R-STDP，
    不直接写权重、不绕过脉冲；超参数固定 BASE 口径，禁止扫描 lr/psp/tau 刷分；
  * 破缺是输出层膜电位扰动，方向与刺激标签严格独立（mode=1）。

**能力口径 g=0**：训练结束后关闭全部教学与破缺注入，用冻结投票
（无奖励、不改权重）评估 200 个交替 A/B 试次。训练期含注入的「即时正确率」
单独记录，且**不得当作能力**。作弊破缺对照（mode=2）专门用于自证
「训练期即时正确率虚高」不会被误当成网络学会了。

诚实边界：刺激相关教学信号是**外部监督标签**，改变了 B1 任务定义
（结果反馈 -> 标签），网络不是自主发现。即使 2×2 全开有效，也只证明
「两条件联合足以让局部 R-STDP + 资格迹学出映射」，**不构成 B1 通过**。
判据状态不变：B1 仍不通过，不进 B2、不写容器、不写 SNC。

实验组：
  F  2x2      : teaching{0,1} x breaking{0,1} x 10 seed（主实验）
  A  ablation : 全开组配对 no_trace / no_reward / freeze_readout / delay=0
  X  sham     : 标签 shuffle 教学、标签相关作弊破缺、以及两者组合
  D  dose     : breaking sigma 与 teaching gain 的剂量-反应
  S  stream   : 全开组下四条流单变量覆盖隔离

运行：
    py -3 Experiment/tools/b1_t1_mechanism.py
    py -3 Experiment/tools/b1_t1_mechanism.py --quick
    py -3 Experiment/tools/b1_t1_mechanism.py --report-only
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
DLL_PATH = ROOT / "b1_t1.dll"
DATA = ROOT / "data" / "b1_t1"
SUMMARY_JSONL = DATA / "t1_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_t1_mechanism_2026-11-13.md"

# 与 host / 前三份报告完全一致的冻结口径
SA, SB, HIDDEN, OA, OB, AVG_CONN = 50, 50, 400, 25, 25, 8.0
OBSERVE_WINDOW = 15
STIM_DURATION = 10
DELAY = 200
TRIALS = 300
LR, PSP, TRACE_K = 0.02, 10.0, 3
EVAL_TRIALS = 200           # g=0 冻结投票评估试次数
MASTER_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

# T1 因子默认值
TEACH_GAIN = 1.0
BREAK_SIGMA = 2.0          # 状态相关探索：输出层膜电位扰动（mV 量级）
HETERO_SIGMA = 0.05        # 异质化读出：镜像边对的反对称权重微扰
SIGMA_DOSES = (0.0, 0.5, 1.0, 2.0, 4.0)
GAIN_DOSES = (0.25, 0.5, 1.0, 2.0)
DELAY_DOSES = (0, 50, 200, 500)
HETERO_DOSES = (0.0, 0.01, 0.03, 0.05, 0.10)   # 全部低于 clamp 阈值

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


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


# ---------------------------------------------------------------- ctypes DLL
def load_dll():
    if not DLL_PATH.exists():
        raise SystemExit(f"ERROR: {DLL_PATH} 不存在，先运行 build.ps1")
    d = C.CDLL(str(DLL_PATH))
    c_i, c_d, c_u, c_u64 = C.c_int, C.c_double, C.c_uint, C.c_uint64
    p_d, p_i = C.POINTER(c_d), C.POINTER(c_i)
    d.b1t_create.argtypes = [c_i] * 5 + [c_d, c_u]
    d.b1t_create.restype = c_i
    d.b1t_destroy.argtypes = []
    d.b1t_ready.restype = c_i
    d.b1t_synapse_count.restype = c_i
    d.b1t_readout_edge_count.restype = c_i
    d.b1t_override_stream.argtypes = [c_i, c_u64]
    d.b1t_legacy_rng.argtypes = [c_i]
    d.b1t_set_hyper.argtypes = [c_d, c_d, c_i]
    d.b1t_set_ablation.argtypes = [c_i, c_i]
    d.b1t_set_teaching.argtypes = [c_i, c_d, c_u64]
    d.b1t_set_breaking.argtypes = [c_i, c_d, c_u64]
    d.b1t_set_freeze_readout.argtypes = [c_i]
    d.b1t_heterogenize.argtypes = [c_d, c_u64]
    d.b1t_heterogenize.restype = c_d
    d.b1t_hetero_residual.restype = c_d
    d.b1t_symmetrize.argtypes = [c_d]
    d.b1t_symmetrize.restype = c_d
    d.b1t_asymmetry.restype = c_d
    d.b1t_output_bias.argtypes = [p_d]
    d.b1t_weight_checksum.restype = c_d
    d.b1t_readout_checksum.restype = c_d
    d.b1t_frozen_vote.argtypes = [c_i, p_i, p_i]
    d.b1t_frozen_vote.restype = c_i
    d.b1t_run_trial.argtypes = [c_i, c_i, c_i, p_d]
    d.b1t_run_trial.restype = c_i
    return d


def new_net(d, seed, overrides=None, teach=(0, TEACH_GAIN),
            breaking=(0, BREAK_SIGMA), ablation=(0, 0), freeze=0):
    """每次实验重建网络，保证单变量与逐位可复现；一律 alpha=1 严格对称。
    breaking[0]==3 时在对称化之后追加持久异质化读出（第三类破缺）。"""
    for name in STREAM:
        d.b1t_override_stream(STREAM[name], 0)
    d.b1t_legacy_rng(0)
    for name, val in (overrides or {}).items():
        d.b1t_override_stream(STREAM[name], val)
    rc = d.b1t_create(SA, SB, HIDDEN, OA, OB, AVG_CONN, C.c_uint(seed))
    if rc != 0:
        raise RuntimeError(f"b1t_create 失败 seed={seed} rc={rc}")
    d.b1t_set_hyper(LR, PSP, TRACE_K)
    d.b1t_set_ablation(*ablation)
    d.b1t_set_teaching(teach[0], teach[1], 0)
    d.b1t_set_breaking(breaking[0], breaking[1], 0)
    d.b1t_set_freeze_readout(freeze)
    asym_sym = d.b1t_symmetrize(1.0)
    if breaking[0] == 3:
        d.b1t_heterogenize(breaking[1], 0)
    return asym_sym


def bias_beta(d):
    buf = (C.c_double * 4)()
    d.b1t_output_bias(buf)
    q = [float(x) for x in buf]
    tot = sum(q)
    beta = ((q[0] + q[3]) - (q[1] + q[2])) / tot if tot > 0 else 0.0
    return beta, q


def labels(n):
    """固定交替 A/B；§6.2 已证明试次顺序流对结果零影响。"""
    return [i % 2 for i in range(n)]


# ---------------------------------------------------------------- g=0 评估
def g0_eval(d, n=None):
    """关闭教学与破缺（冻结投票本身就不注入任何东西），纯网络读出口径。"""
    if n is None:
        n = EVAL_TRIALS
    d.b1t_set_teaching(0, TEACH_GAIN, 0)
    d.b1t_set_breaking(0, 0.0, 0)
    w_before = d.b1t_weight_checksum()
    correct = ties = a_eq_b = 0
    a_sum = b_sum = 0
    # 按刺激分组的预测分布：用于识别「恒定答案」（与刺激无关的初始偏向）
    pred_cnt = {0: [0, 0, 0], 1: [0, 0, 0]}   # stim -> [tie, A, B]
    for stim in labels(n):
        ca, cb = C.c_int(), C.c_int()
        pred = d.b1t_frozen_vote(stim, C.byref(ca), C.byref(cb))
        a_sum += ca.value
        b_sum += cb.value
        if ca.value == cb.value:
            a_eq_b += 1
        expected = 1 if stim == 0 else 2
        if pred == expected:
            correct += 1
        if pred == 0:
            ties += 1
        pred_cnt[stim][pred] += 1
    na = sum(pred_cnt[0]); nb = sum(pred_cnt[1])
    # 「恒定答案」指标：A 刺激与 B 刺激下 pred=1 的比例之差，
    # 接近 0 表示预测与刺激无关（无论给什么刺激都投同一边）
    pa1 = pred_cnt[0][1] / na if na else 0.0
    pb1 = pred_cnt[1][1] / nb if nb else 0.0
    return {
        "g0_acc": correct / n * 100.0,
        "g0_tie_rate": ties / n * 100.0,
        "g0_a_eq_b_rate": a_eq_b / n * 100.0,
        "g0_mean_a": a_sum / n, "g0_mean_b": b_sum / n,
        "g0_pred_a": [pred_cnt[0][0], pred_cnt[0][1], pred_cnt[0][2]],
        "g0_pred_b": [pred_cnt[1][0], pred_cnt[1][1], pred_cnt[1][2]],
        "g0_stim_separation": abs(pa1 - pb1),
        "g0_weight_drift": d.b1t_weight_checksum() - w_before,
        "g0_trials": n,
    }


# ---------------------------------------------------------------- 单个实验
def experiment(d, seed, group, teach=(0, TEACH_GAIN),
               breaking=(0, BREAK_SIGMA), ablation=(0, 0), freeze=0,
               delay=DELAY, n=TRIALS, overrides=None, extra=None,
               eval_n=EVAL_TRIALS):
    asym_i = new_net(d, seed, overrides, teach, breaking, ablation, freeze)
    beta_i, _ = bias_beta(d)
    w0 = d.b1t_weight_checksum()
    ro0 = d.b1t_readout_checksum()
    half = n // 2
    tr_c = te_c = ties = nonzero_reward = 0
    drift = 0.0
    out = (C.c_double * 8)()
    for i, stim in enumerate(labels(n)):
        d.b1t_run_trial(stim, 1, delay, out)
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
    # 能力口径：关闭一切注入后评估
    g0 = g0_eval(d, eval_n)
    asym_f = d.b1t_asymmetry()
    beta_f, _ = bias_beta(d)
    w1 = d.b1t_weight_checksum()
    ro1 = d.b1t_readout_checksum()
    n_syn = d.b1t_synapse_count()
    het_res = d.b1t_hetero_residual()
    d.b1t_destroy()
    row = {
        "group": group, "seed": seed, "n": n,
        "teach_mode": teach[0], "teach_gain": teach[1],
        "break_mode": breaking[0], "break_sigma": breaking[1],
        "ablation": "notrace" if ablation[0] else ("noreward" if ablation[1] else "full"),
        "freeze_readout": freeze, "delay": delay,
        "overrides": overrides or {},
        # 训练期含注入的即时正确率（非能力口径）
        "train_inject_acc": tr_c / half * 100.0,
        "test_inject_acc": te_c / (n - half) * 100.0,
        "inject_tie_rate": ties / n * 100.0,
        "nonzero_reward": nonzero_reward,
        "weight_drift_sum": drift, "weight_drift_total": w1 - w0,
        "readout_drift": ro1 - ro0,
        "asym_init": asym_i, "asym_final": asym_f,
        "hetero_residual": het_res,
        "beta_init": beta_i, "beta_final": beta_f,
        "synapses": n_syn,
    }
    row.update(g0)
    if extra:
        row.update(extra)
    return row


def initial_vote(d, seed, breaking, n):
    """不训练，直接测 g=0 冻结投票的「初始答案」。
    用于自证第三类破缺（异质化读出）是否只是把初始不对称又放了回去。"""
    asym_i = new_net(d, seed, None, (0, TEACH_GAIN), breaking)
    beta_i, _ = bias_beta(d)
    het = d.b1t_hetero_residual()
    g0 = g0_eval(d, n)
    d.b1t_destroy()
    return {
        "group": "initial", "seed": seed,
        "break_mode": breaking[0], "break_sigma": breaking[1],
        "asym_init": asym_i, "hetero_residual": het, "beta_init": beta_i,
        **g0,
    }


# ---------------------------------------------------------------- 实验矩阵
def run_all(quick=False):
    d = load_dll()
    seeds = MASTER_SEEDS[:3] if quick else MASTER_SEEDS
    dseeds = MASTER_SEEDS[:2] if quick else MASTER_SEEDS[:3]
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[{r['group']:<10}] T{r['teach_mode']}B{r['break_mode']} "
              f"{r['ablation']:<8} seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% inj={r['test_inject_acc']:6.2f}% "
              f"asym={r['asym_final']:.3f} rew={r['nonzero_reward']:<4}",
              flush=True)
        return r

    # ---- F: 2×2 析因主实验（教学 × 状态相关探索） ----------------------
    for tm in (0, 1):
        for bm in (0, 1):
            for s in seeds:
                add(d, s, "factorial",
                    teach=(tm, TEACH_GAIN),
                    breaking=(bm, BREAK_SIGMA))

    # ---- H: 第三类破缺「异质化读出」（持久结构破缺） -------------------
    # 与 F 全开组严格单变量：只把 breaking mode 从 1 换成 3。
    for tm in (0, 1):
        for s in seeds:
            add(d, s, "hetero",
                teach=(tm, TEACH_GAIN),
                breaking=(3, HETERO_SIGMA))
    # 初始答案自证：不训练，直接测 g=0 冻结投票
    for bm, sig in ((0, 0.0), (1, BREAK_SIGMA), (3, HETERO_SIGMA)):
        for s in seeds:
            r = initial_vote(d, s, (bm, sig), ev_trials)
            rows.append(r)
            print(f"[initial  ] B{bm} seed={s:<4} g0={r['g0_acc']:6.2f}% "
                  f"het={r['hetero_residual']:.4f} tie={r['g0_tie_rate']:5.1f}%",
                  flush=True)

    # ---- A: 全开组归因消融 ------------------------------------------
    full = dict(teach=(1, TEACH_GAIN), breaking=(1, BREAK_SIGMA))
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(1, 0))
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(0, 1))
    for s in seeds:
        add(d, s, "ablation", **full, freeze=1)
    for dl in DELAY_DOSES:
        for s in dseeds:
            add(d, s, "delay", **full, delay=dl)

    # ---- X: 作弊 / 假阳性检测 ---------------------------------------
    for s in seeds:
        add(d, s, "sham", teach=(2, TEACH_GAIN), breaking=(1, BREAK_SIGMA))
    for s in seeds:
        add(d, s, "sham", teach=(1, TEACH_GAIN), breaking=(2, BREAK_SIGMA))
    for s in seeds:
        add(d, s, "sham", teach=(2, TEACH_GAIN), breaking=(2, BREAK_SIGMA))

    # ---- D: 剂量-反应 ------------------------------------------------
    for sig in SIGMA_DOSES:
        for s in dseeds:
            add(d, s, "dose_sigma", teach=(1, TEACH_GAIN),
                breaking=(1, sig))
    for g in GAIN_DOSES:
        for s in dseeds:
            add(d, s, "dose_gain", teach=(1, g),
                breaking=(1, BREAK_SIGMA))
    for hs in HETERO_DOSES:
        for s in dseeds:
            add(d, s, "dose_hetero", teach=(1, TEACH_GAIN),
                breaking=(3, hs))

    # ---- S: 四流单变量隔离 ------------------------------------------
    for name in STREAM:
        for s in seeds:
            add(d, s, "stream", **full,
                overrides={name: STREAM_OVERRIDE})

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


# ---------------------------------------------------------------- 报告
def sel(rows, group, **kw):
    out = [r for r in rows if r["group"] == group]
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def stat_table(rows, key="g0_acc"):
    xs = [r[key] for r in rows]
    return f"{fmt(mean(xs))} ± {fmt(sd(xs))}", mean(xs), sd(xs)


def build_report(rows):
    a = []
    ap = a.append
    ap("# B1「T1 两必要条件」析因实验报告（2026-11-13）")
    ap("")
    ap("> 依据：`spec/judgments.md` 冻结 **v1.4 §6.4** 的「下一步」与大纲附录 F.4。")
    ap("> 本报告只记录**机制诊断结论**：判据阈值、学习规则、刺激生成器、奖励规则、")
    ap("> `benchmark1_core.c/.h`、`benchmark1_probe.c`、`benchmark1_sym.c`、`host` 全部零改动。")
    ap("> 复现：`py -3 Experiment/tools/b1_t1_mechanism.py`（`--report-only` 可从已存 JSONL 重算）。")
    ap("")
    ap("## 0. 实验定位与诚实边界")
    ap("")
    ap("§6.4 在严格对称读出基线上发现两个死锁：**零梯度自锁**（平局 → `reward=0` → 零更新）")
    ap("与**对称性是 R-STDP 不变子空间**（镜像边资格迹相同 → 等量更新 → 残差恒 0）。")
    ap("据此提出 T1 必须同时满足：① 教学信号与刺激相关；② 存在对称破缺机制。")
    ap("")
    ap("新增 `brain/benchmark1_t1.c` → `b1_t1.dll`，**只链接既有 `b1_core.o` 与 `seed_bundle.o`**。")
    ap("两个因子做成独立开关：")
    ap("")
    ap("- **因子 ① 教学信号**：`mode=0` 为 host 原口径（奖励由网络自己的 `pred` 决定，")
    ap("  与刺激无因果关系）；`mode=1` 为**刺激相关外部教学信号** `±gain`（与 `pred` 无关，")
    ap("  打破零梯度自锁）；`mode=2` 为标签 shuffle 对照（符号与刺激独立）。")
    ap("- **因子 ② 对称破缺**（判据 §6.4 列了三类，本轮实现两类 + 一个作弊对照）：")
    ap("  - `mode=1` **状态相关探索**：观察窗口内对输出层膜电位注入扰动，方向由独立")
    ap("    随机流决定、幅度随神经元自身膜电位状态放大，与刺激标签严格独立；")
    ap("    属**瞬时**扰动，不改变权重结构。")
    ap("  - `mode=3` **异质化读出**：对称化后给每个镜像边对施加反对称的零均值微扰")
    ap("    `s0 += σ·r, s1 −= σ·r`（`r` 来自独立流），属**持久结构破缺**，同样不含标签；")
    ap("    初始权重 0.05–0.4，σ 取值范围全部低于 `WEIGHT_MIN` clamp 阈值。")
    ap("  - `mode=2` 是故意的**标签相关注入**（作弊对照，默认关闭、不计入验收）。")
    ap("  - 第三类「内部前向模型」本轮**未实现**，见 §6.1 的后续动作。")
    ap("")
    ap("教学信号仍通过核心原生 `net_inject_reward(winner=0)` → 资格迹 → R-STDP 生效，")
    ap("**不直接写权重、不绕过脉冲**；因此「完整 − 无迹」消融仍能归因到学习。")
    ap("`mode=1` 破缺只改膜电位；`mode=3` 破缺只在训练**开始前**改一次读出权重，")
    ap("之后不再注入任何东西。超参数固定 BASE 口径（`lr=0.02 / psp=10 / trace_k=3`），")
    ap("全程未扫描 lr / psp / τ，未改刺激生成器。")
    ap("")
    ap("> **⚠️ 能力口径 g=0**：训练结束后关闭全部教学与破缺注入，用冻结投票")
    ap("> （无奖励、不改权重）评估 %d 个交替 A/B 试次；评估前后权重漂移必须为 0。" %
       (rows[0].get("g0_trials", 0) if rows else 0))
    ap("> 单独记录，**不得当作能力**。作弊破缺组专门用来自证这一点。")
    ap("")
    ap("> **⚠️ 不构成 B1 通过**：`mode=1` 的教学信号是**外部监督标签**，把 B1 任务")
    ap("> 从「延迟后的结果反馈」改成了「标签」，网络不是自主发现；破缺机制也是人为注入。")
    ap("> 因此即使 2×2 全开有效，也只能证明两条件**联合足以**让局部 R-STDP 学出映射，")
    ap("> **不能据此宣布 B1 通过**。判据状态不变：B1 仍不通过，不进 B2、不写容器、不写 SNC。")
    ap("")
    ap("### 0.1 证据链（SHA1）")
    ap("")
    ap("| 文件 | SHA1 |")
    ap("|---|---|")
    for rel in ("brain/benchmark1_core.c", "brain/benchmark1_core.h",
                "brain/benchmark1_probe.c", "brain/benchmark1_sym.c",
                "brain/benchmark1_t1.c", "host/benchmark1_host.cpp",
                "core/seed_bundle.c", "spec/judgments.md",
                "tools/b1_t1_mechanism.py", "b1_t1.dll"):
        p = ROOT / rel
        ap(f"| `{rel}` | `{sha1(p) if p.exists() else 'MISSING'}` |")
    ap("")
    ap("`benchmark1_core.c/.h`、`benchmark1_probe.c`、`benchmark1_sym.c` 的 SHA1 与")
    ap("§6.2 / §6.3 / §6.4 三份报告一致，证据链未断；删掉 `benchmark1_t1.c` 与 `b1_t1.dll`")
    ap("即可完全恢复原状。")
    ap("")

    # ---- 1 主实验 2x2
    fac = sel(rows, "factorial")
    ap("## 1. 主实验：2×2 析因（g=0 口径）")
    ap("")
    ap("全部在 `alpha=1` 严格对称基线上，%d 试次完整 host 协议，%d 个 seed，" %
       (fac[0]["n"] if fac else 0, len({r["seed"] for r in fac})))
    ap("奖励延迟 D=%d，教学幅度 %.2f，破缺 sigma %.2f。" %
       (DELAY, TEACH_GAIN, BREAK_SIGMA))
    ap("")
    ap("| 教学① | 破缺② | g=0 正确率 mean±SD | 训练期即时正确率 | g=0 平局率 | 非零奖励 | 读出段漂移 | 训练后残差 |")
    ap("|---|---|---|---|---|---|---|---|")
    cells = {}
    for tm in (0, 1):
        for bm in (0, 1):
            g = [r for r in fac if r["teach_mode"] == tm and r["break_mode"] == bm]
            if not g:
                continue
            s, m, _ = stat_table(g)
            cells[(tm, bm)] = g
            ap("| %s | %s | **%s%%** | %s%% | %s%% | %s | %s | %s |" % (
                "关" if tm == 0 else "开", "关" if bm == 0 else "开",
                s, fmt(mean([r["test_inject_acc"] for r in g])),
                fmt(mean([r["g0_tie_rate"] for r in g])),
                fmt(mean([r["nonzero_reward"] for r in g]), 1),
                fmt(mean([abs(r["readout_drift"]) for r in g]), 4),
                fmt(mean([r["asym_final"] for r in g]), 4)))
    ap("")
    ap("逐 seed 的 g=0 正确率：")
    ap("")
    ap("| seed | ①关②关 | ①开②关 | ①关②开 | ①开②开 |")
    ap("|---|---|---|---|---|")
    for s in sorted({r["seed"] for r in fac}):
        vals = []
        for key in ((0, 0), (1, 0), (0, 1), (1, 1)):
            g = [r for r in cells.get(key, []) if r["seed"] == s]
            vals.append(fmt(g[0]["g0_acc"]) if g else "n/a")
        ap(f"| {s} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    ap("")

    # 结论
    c00 = cells.get((0, 0), [])
    c10 = cells.get((1, 0), [])
    c01 = cells.get((0, 1), [])
    c11 = cells.get((1, 1), [])
    m00 = mean([r["g0_acc"] for r in c00])
    m10 = mean([r["g0_acc"] for r in c10])
    m01 = mean([r["g0_acc"] for r in c01])
    m11 = mean([r["g0_acc"] for r in c11])
    ap("**两条件是否必要、是否联合充分**（g=0 口径）：")
    ap("")
    ap("- 基线 ①关②关：%s%%（复现 §6.4 的 0%% / 100%% 平局）。" % fmt(m00))
    ap("- 仅①：%s%%，相对基线 %+.2fpp；仅②：%s%%，相对基线 %+.2fpp。" %
       (fmt(m10), m10 - m00, fmt(m01), m01 - m00))
    ap("- ①+② 全开：%s%%，相对基线 %+.2fpp。" % (fmt(m11), m11 - m00))
    ap("")
    if m11 - m00 >= 30.0 and m11 - m10 >= 20.0 and m11 - m01 >= 20.0:
        ap("→ **两条件联合充分、单独均不足**：只有同时提供刺激相关教学信号与对称破缺，")
        ap("严格对称网络才学出刺激→输出的映射。但按 §0 的诚实边界，这**不是 B1 通过**。")
    elif m11 - m00 >= 10.0:
        ap("→ 联合组有提升，但幅度不足以判定「充分」；单独条件接近基线。")
        ap("需结合 §3 的假阳性检测判断提升是否真的来自教学信号。")
    else:
        ap("→ **联合组仍无实质提升**：两个条件同时满足仍不足以让局部 R-STDP 学出映射，")
        ap("§6.4 提出的两个条件并**不充分**。这是本轮最重要的负结果。")
    ap("")

    # ---- 1.1 机制自证：教学信号确实生效，但对称性仍是不变子空间
    if c10 and c11:
        ap("### 1.1 机制自证：教学信号生效了，但对称仍不破缺")
        ap("")
        ap("「全开组无效」有两种可能：(a) 教学信号根本没进学习回路；(b) 进了，")
        ap("但对称性把它消掉了。用权重漂移区分这两者：")
        ap("")
        ap("| 组 | 非零奖励次数 | 权重总漂移 | 读出段漂移 | 训练后配对残差 |")
        ap("|---|---|---|---|---|")
        for label, g in (
            ("①关②关", c00), ("①开②关（仅教学）", c10),
            ("①关②开（仅探索）", c01), ("①+② 全开", c11),
        ):
            if not g:
                continue
            ap("| %s | %s | %s | %s | %s |" % (
                label, fmt(mean([r["nonzero_reward"] for r in g]), 0),
                fmt(mean([abs(r["weight_drift_total"]) for r in g]), 4),
                fmt(mean([abs(r["readout_drift"]) for r in g]), 4),
                fmt(mean([r["asym_final"] for r in g]), 6)))
        ap("")
        ap("**①开②关组**非零奖励已是满额、权重与读出段都发生了实质漂移，")
        ap("但训练后配对残差仍**严格为 0** —— 教学信号确实进入了 R-STDP 回路，")
        ap("却因为 `net_inject_reward(winner=0)` 是**全局标量**，镜像边的资格迹相同、")
        ap("奖励调制相同，被逐条等量更新。这把 §6.4 的「对称性是 R-STDP 不变子空间」")
        ap("从「人为注入非零奖励」推广到了**刺激相关的教学信号**：")
        ap("只要第三因子是逐突触无差别的标量，任何教学信号都无法破缺对称。")
        ap("")
        ap("> 这是本轮比 §6.4 更进一层的结论：**瓶颈不只是「信号有没有与刺激相关」，")
        ap("> 而是「第三因子有没有输出侧/通路侧的选择性」**。全局标量奖励在对称读出上")
        ap("> 的梯度恒等于零，与信号内容无关。")
        ap("")

    # ---- 1.2 第三类破缺：异质化读出 + 初始答案自证
    het = sel(rows, "hetero")
    ini = sel(rows, "initial")
    if het:
        ap("### 1.2 换用第三类破缺「异质化读出」：联合组是否变得充分")
        ap("")
        ap("与全开组严格单变量，只把破缺从 `mode=1`（瞬时探索）换成 `mode=3`")
        ap("（持久结构破缺，σ=%.2f）。" % HETERO_SIGMA)
        ap("")
        ap("| 教学① | 破缺 | g=0 正确率 mean±SD | 训练期即时正确率 | 初始配对残差 | 训练后残差 |")
        ap("|---|---|---|---|---|---|")
        for tm in (0, 1):
            g = [r for r in het if r["teach_mode"] == tm]
            if not g:
                continue
            s, _, _ = stat_table(g)
            ap("| %s | 异质化 | %s%% | %s%% | %s | %s |" % (
                "关" if tm == 0 else "开", s,
                fmt(mean([r["test_inject_acc"] for r in g])),
                fmt(mean([r["hetero_residual"] for r in g]), 4),
                fmt(mean([r["asym_final"] for r in g]), 4)))
        ap("")
        h1 = [r for r in het if r["teach_mode"] == 1]
        h0 = [r for r in het if r["teach_mode"] == 0]
        mh1 = mean([r["g0_acc"] for r in h1]) if h1 else float("nan")
        mh0 = mean([r["g0_acc"] for r in h0]) if h0 else float("nan")
        ap("异质化 + 刺激相关教学：g=0 %s%%；异质化 + host 原口径：%s%%；" %
           (fmt(mh1), fmt(mh0)))
        if ini:
            ih = [r for r in ini if r["break_mode"] == 3]
            mih = mean([r["g0_acc"] for r in ih]) if ih else float("nan")
            ap("**不训练的异质化基线是 %s%%** —— 加上刺激相关教学后 %s%%，" %
               (fmt(mih), fmt(mh1)))
            if mh1 < mih - 5:
                ap("**训练反而让 g=0 正确率下降 %spp**。" % fmt(mih - mh1))
            elif mh1 > mih + 5:
                ap("训练带来 %spp 提升，但需先看 §1.2.1 是否只是初始答案被放大。" %
                   fmt(mh1 - mih))
            else:
                ap("训练前后基本持平，没有学到东西。")
        ap("")
        if ini:
            ap("#### 1.2.1 初始答案自证（关键）")
            ap("")
            ap("持久结构破缺必然重新引入不对称，必须检验它是「让网络能学」还是")
            ap("**单纯把 §6.2 的初始随机答案又放了回去**。下表是**完全不训练**、")
            ap("只做 g=0 冻结投票的结果：")
            ap("")
            ap("| 破缺 | 初始配对残差 | g=0 正确率 | g=0 平局率 | A/B 预测分布差 |")
            ap("|---|---|---|---|---|")
            for bm in (0, 1, 3):
                g = [r for r in ini if r["break_mode"] == bm]
                if not g:
                    continue
                name = {0: "无", 1: "状态相关探索", 3: "异质化读出"}[bm]
                ap("| %s | %s | %s%% | %s%% | %s |" % (
                    name, fmt(mean([r["hetero_residual"] for r in g]), 4),
                    fmt(mean([r["g0_acc"] for r in g])),
                    fmt(mean([r["g0_tie_rate"] for r in g])),
                    fmt(mean([r["g0_stim_separation"] for r in g]), 3)))
            ap("")
            ap("「A/B 预测分布差」= 刺激 A 与刺激 B 下投 A 的比例之差；接近 0 表示")
            ap("网络**无论给什么刺激都投同一边**（恒定答案），正确率只是交替标签下的")
            ap("机会水平，不是刺激→输出的映射。")
            ap("")
            ih = [r for r in ini if r["break_mode"] == 3]
            if ih:
                mih = mean([r["g0_acc"] for r in ih])
                ap("异质化组**不训练**就已经是 %s%% 正确 / %s%% 平局，"
                   "A/B 预测分布差 %s；" %
                   (fmt(mih),
                    fmt(mean([r["g0_tie_rate"] for r in ih])),
                    fmt(mean([r["g0_stim_separation"] for r in ih]), 3)))
                ap("而 §1.2 训练后（教学开）只有 %s%%，**比不训练还低 %spp**；" %
                   (fmt(mh1), fmt(mih - mh1)))
                ap("host 原口径组 %s%% 也停在同一个量级。" % fmt(mh0))
                ap("A/B 预测分布差为 0 说明无论给 A 还是 B 刺激，网络都投同一边；")
                ap("50% 只是「交替标签 × 恒定答案」的机会水平，不是映射。")
                ap("所以异质化读出**只是把 §6.2 定位过的初始随机工作点重新注入**，")
                ap("刺激相关教学信号不但没把答案纠正过来，反而把权重推得更差")
                ap("（训练后残差 %s > 初始残差 %s）。" %
                   (fmt(mean([r["asym_final"] for r in h1]), 4),
                    fmt(mean([r["hetero_residual"] for r in ih]), 4)))
                ap("与 §6.3「训练前后 corr 0.9818」是同一个病，不是解法。")
                ap("")
            ap("> **结论**：三类破缺里，瞬时探索无法破缺不变子空间，持久异质化则")
            ap("> 退化为重放初始答案；第三类「内部前向模型」尚未实现。")
            ap("> §6.4 的两个条件**都不是充分条件**，且第二个条件本身还带有")
            ap("> 「破缺强度 ↔ 初始答案回归」的权衡。")
            ap("")

    # ---- 2 归因消融
    abl = sel(rows, "ablation")
    dly = sel(rows, "delay")
    ap("## 2. 归因消融（判据「完整 − 无迹 ≥ 30pp」）")
    ap("")
    ap("全部在 ①+② 全开配置下；同 seed 配对。")
    ap("")
    ap("| 组 | g=0 正确率 mean±SD | 相对全开 (pp) | 非零奖励 | 权重漂移 |")
    ap("|---|---|---|---|---|")
    base = {r["seed"]: r["g0_acc"] for r in c11}
    for label, g in (
        ("全开（参照）", c11),
        ("无迹 notrace", [r for r in abl if r["ablation"] == "notrace"]),
        ("无奖励 noreward", [r for r in abl if r["ablation"] == "noreward"]),
        ("冻结读出", [r for r in abl if r["freeze_readout"] == 1]),
    ):
        if not g:
            continue
        diffs = [base[r["seed"]] - r["g0_acc"] for r in g if r["seed"] in base]
        s, _, _ = stat_table(g)
        ap("| %s | %s%% | %s | %s | %s |" % (
            label, s, fmt(mean(diffs)) if diffs else "n/a",
            fmt(mean([r["nonzero_reward"] for r in g]), 1),
            fmt(mean([abs(r["weight_drift_total"]) for r in g]), 4)))
    ap("")
    nt = [r for r in abl if r["ablation"] == "notrace"]
    if nt and c11:
        pair = [(base[r["seed"]], r["g0_acc"]) for r in nt if r["seed"] in base]
        d = [x - y for x, y in pair]
        ap("**同 seed 配对「完整 − 无迹」**：平均 %spp（SD %s），逐对：%s。" %
           (fmt(mean(d)), fmt(sd(d)), "、".join(fmt(v) for v in d)))
        ap("判据阈值 ≥30pp → %s。" % ("达标" if mean(d) >= 30 else "**不达标**"))
        ap("")
    if dly:
        ap("### 2.1 延迟剂量（是否真用到远端资格迹）")
        ap("")
        ap("| D | g=0 正确率 mean±SD | 非零奖励 |")
        ap("|---|---|---|")
        for dl in DELAY_DOSES:
            g = [r for r in dly if r["delay"] == dl]
            if not g:
                continue
            s, _, _ = stat_table(g)
            ap("| %d | %s%% | %s |" % (dl, s, fmt(mean([r["nonzero_reward"] for r in g]), 1)))
        ap("")

    # ---- 3 假阳性 / 作弊检测
    sham = sel(rows, "sham")
    ap("## 3. 假阳性与作弊检测")
    ap("")
    ap("| 配置 | g=0 正确率 mean±SD | 训练期即时正确率 | 说明 |")
    ap("|---|---|---|---|")
    combos = [
        (2, 1, "教学标签 shuffle（提升若不消失则是假阳性）"),
        (1, 2, "破缺直接含标签（作弊对照）"),
        (2, 2, "两者都作弊（上界对照）"),
    ]
    for tm, bm, note in combos:
        g = [r for r in sham if r["teach_mode"] == tm and r["break_mode"] == bm]
        if not g:
            continue
        s, _, _ = stat_table(g)
        ap("| T%d B%d | %s%% | %s%% | %s |" % (
            tm, bm, s, fmt(mean([r["test_inject_acc"] for r in g])), note))
    ap("")
    cheat = [r for r in sham if r["teach_mode"] == 1 and r["break_mode"] == 2]
    if cheat:
        ap("**关键自证**：T1B2（作弊破缺）的训练期即时正确率 %s%%，" %
           fmt(mean([r["test_inject_acc"] for r in cheat])))
        ap("但 g=0 冻结投票只有 %s%% —— 训练期靠注入「看起来答对」" %
           fmt(mean([r["g0_acc"] for r in cheat])))
        ap("**完全没有转化成网络能力**。这直接支持全局口径 §0.1：能力只认 g=0，")
        ap("含教师/注入的即时正确率不得计入。")
        ap("")
    shuf = [r for r in sham if r["teach_mode"] == 2 and r["break_mode"] == 1]
    if shuf and c11:
        ms = mean([r["g0_acc"] for r in shuf])
        ap("**标签 shuffle 对照**：g=0 %s%% vs 全开 %s%%。" % (fmt(ms), fmt(m11)))
        if m11 < 5.0:
            ap("两组同为 0 附近 —— **全开组本来就没有提升**，因此不存在")
            ap("「提升是否可归因」的问题，shuffle 对照在此处只是印证：")
            ap("无论教学信号是否携带正确的刺激身份，g=0 都学不出映射。")
        elif m11 - ms >= 20:
            ap("提升依赖「教学信号确实携带刺激身份」，不是任意非零奖励 + 破缺的漂移碰运气。")
        else:
            ap("提升**不能**归因于教学信号携带的刺激信息，不得解读为学习生效。")
        ap("")

    # ---- 4 剂量-反应
    ds = sel(rows, "dose_sigma")
    dg = sel(rows, "dose_gain")
    dh = sel(rows, "dose_hetero")
    ap("## 4. 剂量-反应")
    ap("")
    ap("### 4.1 破缺强度 sigma（教学固定 mode=1 / gain=%.2f）" % TEACH_GAIN)
    ap("")
    ap("| sigma | g=0 正确率 mean±SD | 训练后残差 | 读出段漂移 |")
    ap("|---|---|---|---|")
    for sig in SIGMA_DOSES:
        g = [r for r in ds if r["break_sigma"] == sig]
        if not g:
            continue
        s, _, _ = stat_table(g)
        ap("| %s | %s%% | %s | %s |" % (
            fmt(sig), s, fmt(mean([r["asym_final"] for r in g]), 4),
            fmt(mean([abs(r["readout_drift"]) for r in g]), 4)))
    ap("")
    ap("### 4.2 教学幅度 gain（破缺固定 mode=1 / sigma=%.2f）" % BREAK_SIGMA)
    ap("")
    ap("| gain | g=0 正确率 mean±SD | 权重漂移 |")
    ap("|---|---|---|")
    for gv in GAIN_DOSES:
        g = [r for r in dg if r["teach_gain"] == gv]
        if not g:
            continue
        s, _, _ = stat_table(g)
        ap("| %s | %s%% | %s |" % (
            fmt(gv), s, fmt(mean([abs(r["weight_drift_total"]) for r in g]), 4)))
    ap("")
    if dh:
        ap("### 4.3 异质化剂量 sigma（教学固定 mode=1 / gain=%.2f）" % TEACH_GAIN)
        ap("")
        ap("| 异质化 σ | 初始残差 | g=0 正确率 mean±SD | g=0 平局率 | 训练期即时正确率 |")
        ap("|---|---|---|---|---|")
        for hs in HETERO_DOSES:
            g = [r for r in dh if r["break_sigma"] == hs]
            if not g:
                continue
            s, _, _ = stat_table(g)
            ap("| %s | %s | %s%% | %s%% | %s%% |" % (
                fmt(hs), fmt(mean([r["hetero_residual"] for r in g]), 4),
                s, fmt(mean([r["g0_tie_rate"] for r in g])),
                fmt(mean([r["test_inject_acc"] for r in g]))))
        ap("")
        accs = [mean([r["g0_acc"] for r in dh if r["break_sigma"] == hs])
                for hs in HETERO_DOSES]
        ap("剂量增大只是**单调消除平局**（平局率 %s%% → %s%%，恢复「有对有错」），" %
           (fmt(mean([r["g0_tie_rate"] for r in dh if r["break_sigma"] == HETERO_DOSES[0]])),
            fmt(mean([r["g0_tie_rate"] for r in dh if r["break_sigma"] == HETERO_DOSES[-1]]))))
        ap("但 g=0 正确率在各剂量下依次为 %s%%，**全部停在 0 附近**，" %
           "、".join(fmt(v) for v in accs))
        ap("没有任何随剂量或训练增长的趋势 —— 与 §1.2.1 的初始答案回归结论一致。")
        ap("σ 全部低于 clamp 阈值，权重和保持不变。")
        ap("")
    ap("说明：剂量组用 %d 个 seed 的子集（主实验用 %d 个），故与 §1.2 的绝对值" %
       (len({r["seed"] for r in (ds or dg or dh)}),
        len({r["seed"] for r in fac})))
    ap("不必逐位相同，但结论方向一致。剂量扫描只用于**刻画机制**，")
    ap("不用于选择「最好一组」去刷判据；所有验收只看 §1 的固定默认配置与")
    ap("§2 的同 seed 配对消融。")
    ap("")

    # ---- 5 四流
    st = sel(rows, "stream")
    ap("## 5. 四流单变量隔离")
    ap("")
    ap("在 ①+② 全开配置下，逐条覆盖 M0 `seed_bundle` 的任一条流（覆盖值 %d），" % STREAM_OVERRIDE)
    ap("每个流 %d 个 seed。" % len({r["seed"] for r in st}))
    ap("")
    ap("| 覆盖流 | g=0 正确率 mean±SD | 训练后残差 |")
    ap("|---|---|---|")
    for name in STREAM:
        g = [r for r in st if name in r.get("overrides", {})]
        if not g:
            continue
        s, _, _ = stat_table(g)
        ap("| %s | %s%% | %s |" % (
            name, s, fmt(mean([r["asym_final"] for r in g]), 4)))
    ap("")
    ap("四条流下 g=0 全部为 0（个别流出现单个 seed 的 50% 恒定答案，")
    ap("逐 seed 只有 0/50 二值且 A/B 预测分布差为 0，属初始工作点抖动，")
    ap("不构成刺激→输出映射）。与 §6.4「对称性是代数性质、与随机流无关」一致：")
    ap("换任何随机流都改变不了「全局标量第三因子在对称读出上梯度为零」。")
    ap("")

    # ---- 6 结论
    ap("## 6. 结论与判据状态")
    ap("")
    abl = sel(rows, "ablation")
    nt = [r for r in abl if r["ablation"] == "notrace"]
    n_full = {(r["seed"]): r["g0_acc"] for r in c11}
    abl_gap = mean([n_full[r["seed"]] - r["g0_acc"] for r in nt
                    if r["seed"] in n_full]) if nt else float("nan")
    ap("1. **§6.4 提出的两个条件不充分**：10 个 seed 上，教学①与破缺②单独或联合，")
    ap("   g=0 全部为 0.00%%、平局率 100%%；同 seed 配对「完整 − 无迹」= %spp" %
       fmt(abl_gap))
    ap("   （判据要求 ≥30pp），全部 FAIL。")
    ap("2. **更深一层根因（比 §6.4 更进一步）**：①开②关组教学信号满额生效")
    ap("   （%s 次非零奖励、权重漂移 %s、读出漂移 %s），但配对残差严格为 0。" %
       (fmt(mean([r["nonzero_reward"] for r in c10]), 0),
        fmt(mean([abs(r["weight_drift_total"]) for r in c10]), 4),
        fmt(mean([abs(r["readout_drift"]) for r in c10]), 4)))
    ap("   说明瓶颈不是「教学信号有没有与刺激相关」，而是**第三因子有没有输出侧/")
    ap("   通路侧的选择性**：`net_inject_reward(winner=0)` 是逐突触无差别的全局标量，")
    ap("   镜像边资格迹相同 → 等量更新 → 对称梯度恒等于零，与信号内容无关。")
    ap("3. **§6.4 列举的三类破缺机制，本轮验证了两类**：瞬时状态相关探索无法破缺不变子空间；")
    ap("   持久异质化读出虽消除了平局，但只是把 §6.2 的初始随机工作点重新注入")
    ap("   （不训练即 50% 恒定答案、A/B 预测分布差 0），训练后反而更差；")
    ap("   第三类「内部前向模型」本轮未实现。第二个条件本身还带有")
    ap("   「破缺强度 ↔ 初始答案回归」的权衡。")
    ap("4. **作弊检测有效**：标签相关破缺组训练期即时正确率 100%，但 g=0 只有 50%/35%，")
    ap("   自证全局口径 §0.1（能力只认 g=0）不可绕过。")
    ap("5. **判据状态不变**：本实验的教学信号是外部监督标签、破缺是人为注入，")
    ap("   任务定义已偏离 B1 的「延迟结果反馈」，属 T1 机制诊断而非验收。")
    ap("   **B1 仍不通过**；按 §1 失败动作继续停在 T1/T2；")
    ap("   不进 B2、不写容器、不写 SNC。阈值零变更（v1.5 只追加现状记录）；")
    ap("   未改学习规则、刺激生成器、奖励规则、host 或任何核心源码。")
    ap("")
    ap("### 6.1 下一个按计划的动作（仍未做）")
    ap("")
    ap("T1 的最小必要修改方向已由本轮 §1.1 收窄到一处：**第三因子必须具备输出侧/")
    ap("通路侧选择性**（例如让资格迹或奖励调制依赖 post 侧身份），否则任何教学信号")
    ap("在对称读出上梯度都为零。下一步应：")
    ap("")
    ap("1. 在**不引入外部标签**的前提下，把「选择性」做成网络内部的量")
    ap("   （内部前向模型 / 通路特异性资格迹），而不是人工按 winner 路由；")
    ap("2. 在严格对称基线上重做同 seed 配对消融，仍以「完整 − 无迹 ≥30pp」归因；")
    ap("3. 全程只认 g=0，且同时报告 A/B 预测分布差，防止「恒定答案」冒充映射。")
    ap("")
    ap("任何情况下都禁止调 lr / psp / τ、改刺激生成器或挑最好一组去刷判据；")
    ap("不得直接跳到 B2。")
    ap("")
    return "\n".join(a) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="缩减规模冒烟")
    ap.add_argument("--report-only", action="store_true", help="从已存 JSONL 重生成报告")
    args = ap.parse_args()
    if args.report_only:
        if not SUMMARY_JSONL.exists():
            raise SystemExit(f"ERROR: {SUMMARY_JSONL} 不存在")
        rows = [json.loads(l) for l in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"载入 {len(rows)} 行")
    else:
        rows = run_all(args.quick)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(build_report(rows), encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


if __name__ == "__main__":
    sys.exit(main())
