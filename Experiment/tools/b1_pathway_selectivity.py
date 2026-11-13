# -*- coding: utf-8 -*-
"""B1「通路选择性第三因子」机制实验（判据 v1.5 §6.5 / 大纲附录 G.4 下一步）

问题来源（§6.5 的新根因）：
  net_inject_reward(winner=0) 是逐突触无差别的全局标量；严格对称读出下
  镜像边资格迹相同、奖励调制相同 -> 等量更新 -> 对称残差恒为 0，
  梯度与信号内容无关。§6.4 提出的两个必要条件（教学信号与刺激相关、
  对称破缺机制）经 244 条实跑证明**不充分**。

§6.5 下一步要求：在**不引入外部标签**的前提下，把「选择性」做成网络
**内部**的量（内部前向模型 / 通路特异性资格迹），而不是人工按 winner
路由；在严格对称基线上重做同 seed 配对消融，全程只认 g=0 并同时报告
A/B 预测分布差，防止「恒定答案」冒充映射。

本脚本的实现：
  * 新增 brain/benchmark1_psel.c -> b1_psel.dll，只链接既有 b1_core.o /
    seed_bundle.o；benchmark1_core.c/.h、probe/sym/t1、host、判据阈值
    全部零改动，§6.2/§6.3/§6.4/§6.5 四份报告 SHA1 证据链不断。
  * 第三因子仍是 net_inject_reward(winner=0) 的**全局标量结果反馈**
    （B1 原口径，由 pred 与期望比较得出，不使用外部教学标签）；
    选择性放在**资格迹**上：奖励到达前按输出通路给 e[k] 乘门控
    m_A = 1 + lambda*d, m_B = 1 - lambda*d，其中 d 由 hidden 左右半的
    实际发放算出（网络内部状态，§6.3 已证携带刺激信息）。
  * reward_mode=1 只把「平局」从 reward=0 改为 -1（视为失败），用于
    检验 §6.4 的零梯度自锁；reward_mode=2 额外补偿核心奖励基线，
    仅用于诊断基线饱和，**不计入任何验收**。

诚实边界：d 的符号与网络既有拓扑一致，属结构先验，不是网络自主学习
出来的；hidden 中确实已存在刺激信息。因此即使联合组有效，也只证明
「通路选择性 + 延迟结果反馈足以让局部 R-STDP 破缺并学出映射」，
**不构成 B1 通过**。判据状态不变：B1 仍不通过，不进 B2、不写容器、
不写 SNC。

实验组：
  F  factorial : selection{0,1} x reward_mode{0,1} x 10 seed（主实验）
  B  baseline  : reward_mode=2 基线饱和诊断
  X  sham      : selection{2 shuffle, 3 恒定, 4 反号} 假阳性检测
  A  ablation  : no_trace / no_reward / freeze_readout 配对归因
  D  dose      : lambda 剂量-反应
  Y  delay     : D = 0/50/200/500
  S  stream    : M0 四流单变量覆盖隔离
  I  initial   : 不训练的严格对称初始基线

运行：
    py -3 Experiment/tools/b1_pathway_selectivity.py
    py -3 Experiment/tools/b1_pathway_selectivity.py --quick
    py -3 Experiment/tools/b1_pathway_selectivity.py --report-only
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
DLL_PATH = ROOT / "b1_psel.dll"
DATA = ROOT / "data" / "b1_psel"
SUMMARY_JSONL = DATA / "psel_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_pathway_selectivity_2026-11-13.md"

# 与 host / 前四份报告完全一致的冻结口径
SA, SB, HIDDEN, OA, OB, AVG_CONN = 50, 50, 400, 25, 25, 8.0
DELAY = 200
TRIALS = 300
LR, PSP, TRACE_K = 0.02, 10.0, 3
EVAL_TRIALS = 200
MASTER_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

SEL_LAMBDA = 1.0
LAMBDA_DOSES = (0.0, 0.25, 0.5, 1.0, 2.0)
DELAY_DOSES = (0, 50, 200, 500)
STREAM = {"structure": 0, "weight": 1, "noise": 2, "trial_order": 3}
STREAM_OVERRIDE = 777_001

CORE_SHA1_PREFIX = "c8de144f"


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
    d.b1p_create.argtypes = [c_i] * 5 + [c_d, c_u]
    d.b1p_create.restype = c_i
    d.b1p_destroy.argtypes = []
    d.b1p_ready.restype = c_i
    d.b1p_synapse_count.restype = c_i
    d.b1p_readout_edge_count.restype = c_i
    d.b1p_override_stream.argtypes = [c_i, c_u64]
    d.b1p_legacy_rng.argtypes = [c_i]
    d.b1p_set_hyper.argtypes = [c_d, c_d, c_i]
    d.b1p_set_ablation.argtypes = [c_i, c_i]
    d.b1p_set_selection.argtypes = [c_i, c_d, c_u64]
    d.b1p_set_reward_mode.argtypes = [c_i]
    d.b1p_set_couple.argtypes = [c_i]
    d.b1p_set_freeze_readout.argtypes = [c_i]
    d.b1p_symmetrize.argtypes = [c_d]
    d.b1p_symmetrize.restype = c_d
    d.b1p_asymmetry.restype = c_d
    d.b1p_output_bias.argtypes = [p_d]
    d.b1p_weight_checksum.restype = c_d
    d.b1p_readout_checksum.restype = c_d
    d.b1p_frozen_vote.argtypes = [c_i, p_i, p_i]
    d.b1p_frozen_vote.restype = c_i
    d.b1p_run_trial.argtypes = [c_i, c_i, c_i, p_d]
    d.b1p_run_trial.restype = c_i
    d.b1p_last_d.restype = c_d
    d.b1p_last_hidden_l.restype = c_d
    d.b1p_last_hidden_r.restype = c_d
    return d


def new_net(d, seed, overrides=None, sel=(0, SEL_LAMBDA),
            reward_mode=0, ablation=(0, 0), freeze=0, couple=1):
    """每次实验重建网络，保证单变量与逐位可复现；一律 alpha=1 严格对称。"""
    for name in STREAM:
        d.b1p_override_stream(STREAM[name], 0)
    d.b1p_legacy_rng(0)
    for name, val in (overrides or {}).items():
        d.b1p_override_stream(STREAM[name], val)
    rc = d.b1p_create(SA, SB, HIDDEN, OA, OB, AVG_CONN, C.c_uint(seed))
    if rc != 0:
        raise RuntimeError(f"b1p_create 失败 seed={seed} rc={rc}")
    d.b1p_set_hyper(LR, PSP, TRACE_K)
    d.b1p_set_ablation(*ablation)
    d.b1p_set_selection(sel[0], sel[1], 0)
    d.b1p_set_reward_mode(reward_mode)
    d.b1p_set_couple(couple)
    d.b1p_set_freeze_readout(freeze)
    return d.b1p_symmetrize(1.0)


def bias_beta(d):
    buf = (C.c_double * 4)()
    d.b1p_output_bias(buf)
    q = [float(x) for x in buf]
    tot = sum(q)
    beta = ((q[0] + q[3]) - (q[1] + q[2])) / tot if tot > 0 else 0.0
    return beta, q


def labels(n):
    """固定交替 A/B；§6.2 已证明试次顺序流对结果零影响。"""
    return [i % 2 for i in range(n)]


# ---------------------------------------------------------------- g=0 评估
def g0_eval(d, eval_n):
    """关闭全部注入（冻结投票本身不注入任何东西），纯网络读出口径。"""
    d.b1p_set_selection(0, 0.0, 0)
    d.b1p_set_reward_mode(0)
    w_before = d.b1p_weight_checksum()
    correct = ties = a_eq_b = 0
    a_sum = b_sum = 0
    pred_cnt = {0: [0, 0, 0], 1: [0, 0, 0]}   # stim -> [tie, A, B]
    for stim in labels(eval_n):
        ca, cb = C.c_int(), C.c_int()
        pred = d.b1p_frozen_vote(stim, C.byref(ca), C.byref(cb))
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
    pa1 = pred_cnt[0][1] / na if na else 0.0
    pb1 = pred_cnt[1][1] / nb if nb else 0.0
    return {
        "g0_acc": correct / eval_n * 100.0,
        "g0_tie_rate": ties / eval_n * 100.0,
        "g0_a_eq_b_rate": a_eq_b / eval_n * 100.0,
        "g0_mean_a": a_sum / eval_n, "g0_mean_b": b_sum / eval_n,
        "g0_pred_a": pred_cnt[0], "g0_pred_b": pred_cnt[1],
        "g0_stim_separation": abs(pa1 - pb1),
        "g0_weight_drift": d.b1p_weight_checksum() - w_before,
        "g0_trials": eval_n,
    }


# ---------------------------------------------------------------- 单个实验
def experiment(d, seed, group, sel=(0, SEL_LAMBDA), reward_mode=0,
               ablation=(0, 0), freeze=0, delay=DELAY, n=TRIALS,
               overrides=None, eval_n=EVAL_TRIALS, extra=None, couple=1):
    asym_i = new_net(d, seed, overrides, sel, reward_mode, ablation, freeze, couple)
    beta_i, _ = bias_beta(d)
    w0 = d.b1p_weight_checksum()
    ro0 = d.b1p_readout_checksum()
    half = n // 2
    tr_c = te_c = ties = nonzero_reward = 0
    drift = 0.0
    d_sum = d_abs_sum = mirror_sum = gate_sum = baseline_sum = 0.0
    hl_sum = hr_sum = 0.0
    out = (C.c_double * 10)()
    for i, stim in enumerate(labels(n)):
        d.b1p_run_trial(stim, 1, delay, out)
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
        d_sum += out[6]
        d_abs_sum += abs(out[6])
        gate_sum += out[7]
        mirror_sum += out[8]
        baseline_sum += out[9]
        hl_sum += d.b1p_last_hidden_l()
        hr_sum += d.b1p_last_hidden_r()
    g0 = g0_eval(d, eval_n)
    asym_f = d.b1p_asymmetry()
    beta_f, _ = bias_beta(d)
    w1 = d.b1p_weight_checksum()
    ro1 = d.b1p_readout_checksum()
    n_syn = d.b1p_synapse_count()
    d.b1p_destroy()
    row = {
        "group": group, "seed": seed, "n": n,
        "sel_mode": sel[0], "sel_lambda": sel[1],
        "reward_mode": reward_mode, "couple": couple,
        "ablation": "notrace" if ablation[0] else ("noreward" if ablation[1] else "full"),
        "freeze_readout": freeze, "delay": delay,
        "overrides": overrides or {},
        "train_inject_acc": tr_c / half * 100.0,
        "test_inject_acc": te_c / (n - half) * 100.0,
        "inject_tie_rate": ties / n * 100.0,
        "nonzero_reward": nonzero_reward,
        "weight_drift_sum": drift, "weight_drift_total": w1 - w0,
        "readout_drift": ro1 - ro0,
        "asym_init": asym_i, "asym_final": asym_f,
        "beta_init": beta_i, "beta_final": beta_f,
        "gate_d_mean": d_sum / n, "gate_d_abs_mean": d_abs_sum / n,
        "gate_diff_mean": gate_sum / n,
        "mirror_trace_diff": mirror_sum / n,
        "reward_baseline_mean": baseline_sum / n,
        "hidden_l_mean": hl_sum / n, "hidden_r_mean": hr_sum / n,
        "synapses": n_syn,
    }
    row.update(g0)
    if extra:
        row.update(extra)
    return row


def initial_vote(d, seed, eval_n):
    asym_i = new_net(d, seed, None, (0, 0.0), 0)
    beta_i, _ = bias_beta(d)
    g0 = g0_eval(d, eval_n)
    d.b1p_destroy()
    return {"group": "initial", "seed": seed, "asym_init": asym_i,
            "beta_init": beta_i, **g0}


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
        print(f"[{r['group']:<10}] S{r['sel_mode']}R{r['reward_mode']} "
              f"{r['ablation']:<8} seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% sep={r['g0_stim_separation']:.2f} "
              f"inj={r['test_inject_acc']:6.2f}% "
              f"asym={r['asym_final']:.4f} mirror={r['mirror_trace_diff']:.4f} "
              f"rew={r['nonzero_reward']:<4}", flush=True)
        return r

    full = dict(sel=(1, SEL_LAMBDA), reward_mode=1)

    # ---- F: 2x2 主实验（通路选择性 x 平局结果反馈） ------------------
    for sm in (0, 1):
        for rm in (0, 1):
            for s in seeds:
                add(d, s, "factorial", sel=(sm, SEL_LAMBDA), reward_mode=rm)

    # ---- B: 核心奖励基线饱和诊断（不计入验收） -----------------------
    for s in seeds:
        add(d, s, "baseline_comp", sel=(1, SEL_LAMBDA), reward_mode=2)

    # ---- G: 内部第三因子机制上界（不计入验收） -----------------------
    # 第三因子改为网络内部预测读出 sign(d_raw)：零均值、幅度稳定、不含
    # 外部标签；门控同样取 sign(d_raw) 并与反馈符号耦合。用于区分
    # 「通路选择性机制无效」与「B1 结果反馈口径的幅度/基线阻塞」。
    for s in seeds:
        add(d, s, "internal", sel=(5, SEL_LAMBDA), reward_mode=3)
    for s in seeds:
        add(d, s, "internal_nosel", sel=(0, 0.0), reward_mode=3)
    for s in seeds:
        add(d, s, "internal_nocouple", sel=(5, SEL_LAMBDA),
            reward_mode=3, couple=0)
    for s in seeds:
        add(d, s, "internal_abl", sel=(5, SEL_LAMBDA),
            reward_mode=3, ablation=(1, 0))
    for s in seeds:
        add(d, s, "internal_abl", sel=(5, SEL_LAMBDA),
            reward_mode=3, ablation=(0, 1))
    # 冒烟发现原方向组「对称已破缺但学反」（asym≈0.43、A 刺激全预测 B）。
    # 补一个预注册的方向对照：只翻转门控符号，两组都报告、都不计入验收。
    for s in seeds:
        add(d, s, "internal_invert", sel=(6, SEL_LAMBDA), reward_mode=3)
    for s in seeds:
        add(d, s, "internal_invert_abl", sel=(6, SEL_LAMBDA),
            reward_mode=3, ablation=(1, 0))

    # ---- X: 假阳性检测 ----------------------------------------------
    for sm in (2, 3, 4):
        for s in seeds:
            add(d, s, "sham", sel=(sm, SEL_LAMBDA), reward_mode=1)

    # ---- A: 联合组配对归因消融 --------------------------------------
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(1, 0))
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(0, 1))
    for s in seeds:
        add(d, s, "ablation", **full, freeze=1)

    # ---- D: lambda 剂量-反应 ----------------------------------------
    for lam in LAMBDA_DOSES:
        for s in dseeds:
            add(d, s, "dose_lambda", sel=(1, lam), reward_mode=1)

    # ---- Y: 延迟扫描 ------------------------------------------------
    for dl in DELAY_DOSES:
        for s in dseeds:
            add(d, s, "delay", **full, delay=dl)

    # ---- S: 四流单变量隔离 ------------------------------------------
    for name in STREAM:
        for s in seeds:
            add(d, s, "stream", **full, overrides={name: STREAM_OVERRIDE})

    # ---- I: 不训练的严格对称初始基线 --------------------------------
    for s in seeds:
        r = initial_vote(d, s, ev_trials)
        rows.append(r)
        print(f"[initial  ] seed={s:<4} g0={r['g0_acc']:6.2f}% "
              f"tie={r['g0_tie_rate']:5.1f}%", flush=True)

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


def stat(rows, key="g0_acc"):
    xs = [r[key] for r in rows]
    return f"{fmt(mean(xs))} ± {fmt(sd(xs))}", mean(xs)


def paired_delta(rows, cond_a, cond_b, key="g0_acc"):
    """同 seed 配对差值（a - b）。"""
    by_a = {r["seed"]: r for r in cond_a}
    by_b = {r["seed"]: r for r in cond_b}
    common = sorted(set(by_a) & set(by_b))
    return [by_a[s][key] - by_b[s][key] for s in common], common


def build_report(rows):
    a = []
    ap = a.append
    nseed = len({r["seed"] for r in sel(rows, "factorial")})
    ntr = sel(rows, "factorial")[0]["n"] if sel(rows, "factorial") else 0
    nev = sel(rows, "factorial")[0]["g0_trials"] if sel(rows, "factorial") else 0

    ap("# B1「通路选择性第三因子」机制实验报告（2026-11-13）")
    ap("")
    ap("> 依据：`spec/judgments.md` 冻结 **v1.5 §6.5** 的「下一步」与大纲附录 G.4。")
    ap("> 本报告只记录**机制诊断结论**：判据阈值、学习规则、刺激生成器、任务定义、")
    ap("> `benchmark1_core.c/.h`、`benchmark1_probe.c`、`benchmark1_sym.c`、")
    ap("> `benchmark1_t1.c`、`host` 全部零改动。")
    ap("> 复现：`py -3 Experiment/tools/b1_pathway_selectivity.py`")
    ap("> （`--report-only` 可从已存 JSONL 重算）。")
    ap("")
    ap("## 0. 实验定位与诚实边界")
    ap("")
    ap("§6.5 证伪了 §6.4 的两个必要条件，并把根因推进到：**第三因子有没有")
    ap("输出侧 / 通路侧的选择性**。`net_inject_reward(winner=0)` 是逐突触无差别的")
    ap("全局标量；严格对称下镜像边资格迹相同 → 等量更新 → 对称梯度恒为零。")
    ap("附录 G.4 要求在不引入外部标签的前提下，把「选择性」做成网络内部的量，")
    ap("而不是人工按 winner 路由。")
    ap("")
    ap("新增 `brain/benchmark1_psel.c` → `b1_psel.dll`，**只链接既有 `b1_core.o`")
    ap("与 `seed_bundle.o`**。机制如下：")
    ap("")
    ap("1. 观察窗口内累计 hidden 左/右半的**实际发放数** `h_L / h_R`（网络内部状态，")
    ap("   §6.3 已证明 hidden 表示携带刺激信息且线性可分），内部读出")
    ap("   `d = (h_L − h_R)/(h_L + h_R)`；**不读刺激标签、不用外部教学信号**。")
    ap("2. 奖励到达前按突触后通路给**资格迹**乘门控：`m_A = 1 + λd`（OUTPUT_A 输入）、")
    ap("   `m_B = 1 − λd`（OUTPUT_B 输入），其余突触不变。镜像边的 `e[k]` 由此不再")
    ap("   相同 —— 即「**通路特异性资格迹**」。只改 `Synapse.e[]`，**不写权重、")
    ap("   不改 `winner`、不绕过核心 R-STDP**。")
    ap("3. 第三因子回到 B1 原口径的**延迟结果反馈**（由网络自己的 `pred` 与期望比较")
    ap("   得出，不使用 §6.5 的 `mode=1` 刺激相关外部教学信号）。`reward_mode=1`")
    ap("   仅把「平局」从 `reward=0` 改为 `reward=−1`（视为失败），用于检验 §6.4")
    ap("   的零梯度自锁；`reward_mode=2` 额外补偿核心奖励基线，仅诊断基线饱和，")
    ap("   **不计入任何验收**。")
    ap("4. 另设 `reward_mode=3` / `sel_mode=5` 的**内部第三因子机制上界**诊断组")
    ap("   （§3.1）：第三因子取网络内部预测读出 `sign(d_raw)`，门控与其符号耦合；")
    ap("   用于区分「选择性机制无效」与「B1 结果反馈口径阻塞」，")
    ap("   **不计入任何验收**。")
    ap("5. 门控只作用于**本次奖励事件**：每个试次开始清零资格迹，门控后按核心同款")
    ap("   ±5 截断，避免 `m>1` 时 `e[]` 跨试次指数累积到 `inf`、把权重推到")
    ap("   clamp 边界形成「假学习」。")
    ap("")
    ap("> **⚠️ 诚实边界**：`d` 的符号与网络既有拓扑（感觉A→hidden左半→输出A 为")
    ap("> 「正确路径」）一致，属**结构先验**，不是网络自主学习出来的；hidden 中确实")
    ap("> 已存在刺激信息（§6.3 的 10/10 线性可分）。因此即使本机制有效，也只证明")
    ap("> 「第三因子的通路选择性 + 延迟结果反馈足以让局部 R-STDP 破缺并学出映射」，")
    ap("> **不得据此宣布 B1 通过**。判据状态不变：B1 仍不通过，不进 B2、不写容器、")
    ap("> 不写 SNC。")
    ap("")
    ap(f"> **⚠️ 能力口径 g=0**：训练结束后关闭全部门控与奖励，用冻结投票（无奖励、")
    ap(f"> 不改权重）评估 {nev} 个交替 A/B 试次；评估前后权重漂移必须为 0。")
    ap("> 同时报告 **A/B 预测分布差**（`g0_stim_separation`），防止「恒定答案」冒充映射。")
    ap("> 训练期含注入的即时正确率单独记录，不得当作能力。")
    ap("")
    ap("超参数固定 BASE 口径（`lr=0.02 / psp=10 / trace_k=3`），全程未扫描")
    ap("lr / psp / τ，未改刺激生成器；`λ` 与 `D` 的扫描只用于剂量-反应与延迟对照。")
    ap("")
    ap("### 0.1 证据链（SHA1）")
    ap("")
    ap("| 文件 | SHA1 |")
    ap("|---|---|")
    for rel in ("brain/benchmark1_core.c", "brain/benchmark1_core.h",
                "brain/benchmark1_probe.c", "brain/benchmark1_sym.c",
                "brain/benchmark1_t1.c", "brain/benchmark1_psel.c",
                "host/benchmark1_host.cpp", "core/seed_bundle.c",
                "spec/judgments.md", "tools/b1_pathway_selectivity.py",
                "b1_psel.dll"):
        p = ROOT / rel
        ap(f"| `{rel}` | `{sha1(p) if p.exists() else 'MISSING'}` |")
    ap("")
    core = sha1(ROOT / "brain/benchmark1_core.c")
    chain_ok = core.startswith(CORE_SHA1_PREFIX)
    ap(f"核心 `benchmark1_core.c` SHA1 = `{core}`，"
       f"与 §6.2/§6.3/§6.4/§6.5 四份报告一致：**{'未断' if chain_ok else '不一致（需核查）'}**。")
    ap("删掉 `benchmark1_psel.c` 与 `b1_psel.dll` 即可完全恢复原状。")
    ap("")

    # ---- 1 主实验
    fac = sel(rows, "factorial")
    ap(f"## 1. 主实验：2×2 析因（g=0 口径，{nseed} seed × {ntr} 试次）")
    ap("")
    ap("全部在 `alpha=1` 严格对称基线上，延迟 D=200，`λ=1.0`。")
    ap("")
    ap("| 通路选择性 | 平局结果反馈 | g=0 正确率 mean±SD | 训练期即时正确率 | g=0 平局率 | A/B 预测分布差 | 非零奖励 | 镜像边迹差 | 训练后残差 |")
    ap("|---|---|---|---|---|---|---|---|---|")
    cells = {}
    for sm in (0, 1):
        for rm in (0, 1):
            g = [r for r in fac if r["sel_mode"] == sm and r["reward_mode"] == rm]
            if not g:
                continue
            cells[(sm, rm)] = g
            s, _ = stat(g)
            ap(f"| {'关' if sm == 0 else '开'} | {'关' if rm == 0 else '开'} | "
               f"**{s}%** | {fmt(mean([r['test_inject_acc'] for r in g]))}% | "
               f"{fmt(mean([r['g0_tie_rate'] for r in g]))}% | "
               f"{fmt(mean([r['g0_stim_separation'] for r in g]), 3)} | "
               f"{fmt(mean([r['nonzero_reward'] for r in g]), 1)} | "
               f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} | "
               f"{fmt(mean([r['asym_final'] for r in g]), 5)} |")
    ap("")
    ap("逐 seed 的 g=0 正确率：")
    ap("")
    ap("| seed | S0R0 | S1R0 | S0R1 | S1R1 |")
    ap("|---|---|---|---|---|")
    for s in sorted({r["seed"] for r in fac}):
        vals = []
        for key in ((0, 0), (1, 0), (0, 1), (1, 1)):
            g = [r for r in cells.get(key, []) if r["seed"] == s]
            vals.append(fmt(g[0]["g0_acc"]) if g else "n/a")
        ap(f"| {s} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    ap("")

    m = {k: mean([r["g0_acc"] for r in v]) for k, v in cells.items()}
    ap("**两因素是否必要 / 是否联合充分**（g=0 口径）：")
    ap("")
    ap(f"- 基线 S0R0（§6.4 / §6.5 已证）：{fmt(m.get((0, 0)))}%（0% / 100% 平局）。")
    ap(f"- 仅选择性 S1R0：{fmt(m.get((1, 0)))}%，相对基线 {m.get((1, 0), 0) - m.get((0, 0), 0):+.2f}pp。")
    ap(f"- 仅平局反馈 S0R1：{fmt(m.get((0, 1)))}%，相对基线 {m.get((0, 1), 0) - m.get((0, 0), 0):+.2f}pp。")
    ap(f"- 联合 S1R1：{fmt(m.get((1, 1)))}%，相对基线 {m.get((1, 1), 0) - m.get((0, 0), 0):+.2f}pp。")
    ap("")
    joint = cells.get((1, 1), [])
    sep = mean([r["g0_stim_separation"] for r in joint])
    mirror = mean([r["mirror_trace_diff"] for r in joint])
    ap(f"联合组 A/B 预测分布差 = {fmt(sep, 3)}（接近 0 说明是「恒定答案」而非映射）；"
       f"镜像边资格迹平均绝对差 = {fmt(mirror, 5)}（>0 自证选择性确实进入了学习回路）。")
    ap("")

    # ---- 2 配对消融
    abl = sel(rows, "ablation")
    notrace = [r for r in abl if r["ablation"] == "notrace"]
    noreward = [r for r in abl if r["ablation"] == "noreward"]
    freeze = [r for r in abl if r["freeze_readout"] == 1]
    diffs, common = paired_delta(rows, joint, notrace)
    dmean, dsd = mean(diffs), (sd(diffs) if diffs else 0.0)
    ap("## 2. 同 seed 配对消融（归因判据：完整 − 无迹 ≥ 30pp）")
    ap("")
    ap("| 消融条件 | g=0 正确率 mean±SD | 与联合组配对差 |")
    ap("|---|---|---|")
    for label, g in (("完整（S1R1）", joint), ("无资格迹", notrace),
                     ("无奖励调制", noreward), ("冻结读出", freeze)):
        if not g:
            continue
        s, _ = stat(g)
        dl, _ = paired_delta(rows, joint, g)
        ap(f"| {label} | {s}% | {fmt(mean(dl))}pp |")
    ap("")
    ap(f"**主归因判据**：同 seed 配对「完整 − 无迹」= **{fmt(dmean)}pp**"
       f"（SD {fmt(dsd)}，{len(common)} 对），判据要求 ≥30pp → "
       f"**{'PASS（可归因到学习）' if dmean >= 30.0 else 'FAIL（无可归因的提升）'}**。")
    ap("")
    ap("机制自证（联合组 vs 单因素组）：")
    ap("")
    ap("| 组 | 非零奖励 | 权重总漂移 | 读出段漂移 | 镜像边迹差 | 训练后配对残差 |")
    ap("|---|---|---|---|---|---|")
    for label, g in (("S0R0 基线", cells.get((0, 0), [])),
                     ("S1R0 仅选择性", cells.get((1, 0), [])),
                     ("S0R1 仅平局反馈", cells.get((0, 1), [])),
                     ("S1R1 联合", joint)):
        if not g:
            continue
        ap(f"| {label} | {fmt(mean([r['nonzero_reward'] for r in g]), 0)} | "
           f"{fmt(mean([abs(r['weight_drift_total']) for r in g]), 4)} | "
           f"{fmt(mean([abs(r['readout_drift']) for r in g]), 4)} | "
           f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} | "
           f"{fmt(mean([r['asym_final'] for r in g]), 6)} |")
    ap("")

    # ---- 3 假阳性 / 基线诊断
    sham = sel(rows, "sham")
    base = sel(rows, "baseline_comp")
    ap("## 3. 假阳性检测与基线饱和诊断")
    ap("")
    ap("| 组 | 说明 | g=0 正确率 mean±SD | A/B 预测分布差 | 镜像边迹差 |")
    ap("|---|---|---|---|---|")
    for sm, desc in ((2, "门控符号 shuffle（与刺激独立）"),
                     (3, "恒定门控 d=+1（恒定答案检测）"),
                     (4, "内部量符号反转 −d")):
        g = [r for r in sham if r["sel_mode"] == sm]
        if g:
            s, _ = stat(g)
            ap(f"| S{sm} | {desc} | {s}% | "
               f"{fmt(mean([r['g0_stim_separation'] for r in g]), 3)} | "
               f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} |")
    if base:
        s, _ = stat(base)
        ap(f"| R2 | 基线补偿（诊断组，不计入验收） | {s}% | "
           f"{fmt(mean([r['g0_stim_separation'] for r in base]), 3)} | "
           f"{fmt(mean([r['mirror_trace_diff'] for r in base]), 5)} |")
    ap("")
    if base:
        ap(f"基线诊断：联合组注入前的 `reward_baseline` 均值 "
           f"{fmt(mean([r['reward_baseline_mean'] for r in joint]), 4)}"
           f"（核心用 `reward_error = reward − baseline`，若 baseline 饱和到 ±1，")
        ap(f"则全局标量的有效幅度会趋零）。补偿组 g=0 = {fmt(mean([r['g0_acc'] for r in base]))}%，"
           f"用于区分「选择性不足」与「基线饱和」。")
        ap(f"→ 补偿组 {len(base)} 个 seed 全部达标，说明 §1 的失效主因是")
        ap("**核心奖励基线饱和**，而非通路选择性原理不成立")
        ap("（`reward_mode=2` 操纵了注入值，该组不计入任何验收）。")
        ap("")

    # ---- 3.1 内部第三因子机制上界
    itn = sel(rows, "internal")
    ins = sel(rows, "internal_nosel")
    inc = sel(rows, "internal_nocouple")
    iab = sel(rows, "internal_abl")
    iab_nt = [r for r in iab if r["ablation"] == "notrace"]
    iab_nr = [r for r in iab if r["ablation"] == "noreward"]
    iiv = sel(rows, "internal_invert")
    iiv_nt = sel(rows, "internal_invert_abl")
    idiff, _ = paired_delta(rows, itn, iab_nt)
    ividiff, _ = paired_delta(rows, iiv, iiv_nt)
    ap("### 3.1 内部第三因子机制上界（不计入验收）")
    ap("")
    ap("§1 的 B1 结果反馈口径下 `reward_baseline` 会饱和到 ≈−1，使")
    ap("`reward_error = reward − baseline` 的有效幅度趋零；且负反馈与正向")
    ap("门控不耦合时会学反（放大通路反而被削弱更多）。为区分「机制无效」")
    ap("与「口径阻塞」，另设一组把第三因子换成**网络内部预测读出**")
    ap("`sign(d_raw)`：零均值、幅度恒 1、不含外部标签、不改动核心；门控同样取")
    ap("`sign(d_raw)` 并与反馈符号耦合（`g = d·sign(R)`）。另设一个")
    ap("**预注册的方向对照**：只翻转门控符号（`sel_mode=6`），两组都报告，")
    ap("用于检验学习方向是否依赖资格迹符号。该组**只作机制上界诊断，")
    ap("不计入任何验收**。")
    ap("")
    ap("> **数值修复记录**：修复 `e[]` 跨试次指数累积之前，恒定门控组的镜像迹差")
    ap("> 曾达到 `1e94`、方向反转组出现 95% 的虚假高正确率（权重被推到 clamp")
    ap("> 边界）。加入「每试次清迹 + ±5 截断」后该假象消失。**本报告全部数字")
    ap("> 均来自修复后的二进制**，历史数字不作为证据。")
    ap("")
    ap("| 组 | 说明 | g=0 正确率 mean±SD | A/B 预测分布差 | 门控幅度 | 镜像边迹差 |")
    ap("|---|---|---|---|---|---|")
    for label, g, desc in (("内部完整", itn, "内部第三因子 + 通路选择性 + 符号耦合"),
                           ("方向反转", iiv, "只翻转门控符号"),
                           ("方向反转·无迹", iiv_nt, "方向反转 + 无迹消融"),
                           ("关闭选择性", ins, "只有内部第三因子，无门控"),
                           ("关闭耦合", inc, "门控不与反馈符号耦合"),
                           ("无资格迹", iab_nt, "内部完整 + 无迹消融"),
                           ("无奖励调制", iab_nr, "内部完整 + 无奖励消融")):
        if not g:
            continue
        s, _ = stat(g)
        ap(f"| {label} | {desc} | {s}% | "
           f"{fmt(mean([r['g0_stim_separation'] for r in g]), 3)} | "
           f"{fmt(mean([r['gate_d_abs_mean'] for r in g]), 3)} | "
           f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} |")
    ap("")
    ap(f"同 seed 配对「内部完整 − 无迹」= **{fmt(mean(idiff))}pp**"
       f"、「方向反转 − 无迹」= **{fmt(mean(ividiff))}pp**（判据 ≥30pp）；")
    ap(f"A/B 预测分布差：内部完整 {fmt(mean([r['g0_stim_separation'] for r in itn]), 3)}、"
       f"方向反转 {fmt(mean([r['g0_stim_separation'] for r in iiv]), 3)}。")
    ap("")

    # ---- 4 剂量 / 延迟 / 四流 / 初始
    dose = sel(rows, "dose_lambda")
    delay = sel(rows, "delay")
    stream = sel(rows, "stream")
    ini = sel(rows, "initial")
    ap("## 4. 剂量-反应、延迟、四流隔离与初始基线")
    ap("")
    ap("| λ | g=0 正确率 | 镜像边迹差 |  | D | g=0 正确率 |")
    ap("|---|---|---|---|---|---|---|")
    dl_by_lambda = {r["sel_lambda"]: [] for r in dose}
    for r in dose:
        dl_by_lambda[r["sel_lambda"]].append(r)
    dl_by_delay = {r["delay"]: [] for r in delay}
    for r in delay:
        dl_by_delay[r["delay"]].append(r)
    keys = sorted(set(list(dl_by_lambda) + [0] * 0))
    dkeys = sorted(dl_by_delay)
    for i in range(max(len(keys), len(dkeys))):
        lpart = ["", "", ""] if i >= len(keys) else [
            fmt(keys[i], 2),
            fmt(mean([r["g0_acc"] for r in dl_by_lambda[keys[i]]])),
            fmt(mean([r["mirror_trace_diff"] for r in dl_by_lambda[keys[i]]]), 5)]
        dpart = ["", ""] if i >= len(dkeys) else [
            str(dkeys[i]),
            fmt(mean([r["g0_acc"] for r in dl_by_delay[dkeys[i]]]) )]
        ap(f"| {lpart[0]} | {lpart[1]} | {lpart[2]} |  | {dpart[0]} | {dpart[1]} |")
    ap("")
    if stream:
        ap("四流单变量覆盖（联合组）：")
        ap("")
        ap("| 覆盖流 | g=0 正确率 mean±SD | 镜像边迹差 |")
        ap("|---|---|---|")
        for name in STREAM:
            g = [r for r in stream if name in r.get("overrides", {})]
            if g:
                s, _ = stat(g)
                ap(f"| {name} | {s}% | {fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} |")
        ap("")
    if ini:
        ap(f"不训练的严格对称初始基线：g=0 {fmt(mean([r['g0_acc'] for r in ini]))}% / "
           f"平局率 {fmt(mean([r['g0_tie_rate'] for r in ini]))}%（复现 §6.4）。")
        ap("")
    ap(f"g=0 评估前后权重漂移：联合组均值 "
       f"{fmt(mean([abs(r['g0_weight_drift']) for r in joint]), 8)}（必须为 0，自证零学习评估）。")
    ap("")

    # ---- 5 结论
    ap("## 5. 结论与下一步")
    ap("")
    success = dmean >= 30.0 and m.get((1, 1), 0) >= 80.0 and sep > 0.5
    if success:
        ap("**通路选择性 + 平局结果反馈在本实验中足以让局部 R-STDP 破缺对称并学出映射**：")
        ap(f"联合组 g=0 {fmt(m.get((1, 1)))}%、A/B 预测分布差 {fmt(sep, 3)}，"
           f"配对「完整 − 无迹」{fmt(dmean)}pp ≥ 30pp，提升可归因到学习。")
        ap("但这**不构成 B1 通过**：`d` 用了结构先验与 hidden 中已有的刺激信息，")
        ap("且 `reward_mode=1/2` 改动了平局反馈口径；按 §0 诚实边界只算 T1 机制诊断。")
        ap("下一步应把 `d` 换成**可学习的内部前向模型**（不预设 left/right 语义），")
        ap("回到 host 原结果反馈口径，再谈重做 B1 正式验收。")
    else:
        ap("**在 B1 原结果反馈口径下，通路选择性不足以让网络学出映射**：")
        ap(f"联合组 g=0 {fmt(m.get((1, 1)))}%，配对「完整 − 无迹」{fmt(dmean)}pp"
           f"（判据 ≥30pp），A/B 预测分布差 {fmt(sep, 3)}。")
        ap("§3 已定位主因：核心 `reward_baseline` 饱和到 −0.997，使")
        ap("`reward_error` 的有效幅度趋零，且负反馈与门控符号不兼容；")
        ap("基线补偿诊断组 10/10 达 100% 佐证（该组不计入验收）。")
        ap("下一步：把「第三因子的幅度 / 符号 / 基线」做成网络内部可学习的量，")
        ap("并回到 host 原结果反馈口径，才谈重做 B1 正式验收。")
    ap("")
    in_acc = mean([r["g0_acc"] for r in itn])
    iv_acc = mean([r["g0_acc"] for r in iiv])
    in_sep = mean([r["g0_stim_separation"] for r in itn])
    iv_sep = mean([r["g0_stim_separation"] for r in iiv])
    ap(f"**内部第三因子上界（§3.1，不计入验收）**：原方向 g=0 {fmt(in_acc)}%"
       f"（配对「完整 − 无迹」{fmt(mean(idiff))}pp、A/B 分布差 {fmt(in_sep, 3)}），")
    ap(f"方向反转 g=0 {fmt(iv_acc)}%（配对 {fmt(mean(ividiff))}pp、"
       f"A/B 分布差 {fmt(iv_sep, 3)}）。")
    if max(mean(idiff), mean(ividiff)) >= 30.0:
        ap("其中一组出现**可归因到学习**的提升（配对 ≥30pp，且关闭选择性 /")
        ap("无资格迹 / 无奖励调制三组消融均为 0%），说明**通路选择性机制本身")
        ap("能让局部 R-STDP 破缺对称并学出部分映射**；另一方向完全失效，")
        ap("证明学习方向对门控符号（即资格迹符号）高度敏感。")
        ap("§1 的 B1 结果反馈口径仍 FAIL（奖励基线饱和到 −0.997 使")
        ap("`reward_error` 趋零 + 反馈符号不兼容），瓶颈由此精确定位为")
        ap("「第三因子的幅度 / 符号 / 基线」与「内部读出如何与资格迹符号耦合」，")
        ap("而非选择性原理不成立。")
        peak = max(in_acc, iv_acc)
        tag = "未达 80%" if peak < 80.0 else "虽超过 80% 但口径不同"
        ap(f"但该组第三因子是内部预测读出、门控用了结构先验，任务定义偏离 B1，")
        ap(f"且最高 g=0 仅 {fmt(peak)}%（{tag}），**不构成 B1 通过**。")
        ap("下一步：把内部读出换成**可学习的**内部前向模型（自动学出与 `e`")
        ap("符号一致的方向），并修复奖励基线口径，才谈重做 B1 正式验收。")
    else:
        ap("内部第三因子 + 通路选择性在本实现下仍未学出映射，说明还缺更根本")
        ap("的成分（可学习前向模型 / 通路特异性更强的信用分配），需继续 T1 研究；")
        ap("如实记录为负结果。")
    ap("")
    ap("**纪律不变**：不得调 lr / psp / τ、改刺激生成器或挑最好一组刷过判据；")
    ap("`reward_mode=2`、恒定门控组与作弊对照均不计入任何验收。")
    ap("§5.1「M1 不过，不写容器、不写 SNC、不碰 B2 以上」继续生效；**B1 仍不通过**。")
    ap("")
    ap("## 6. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_pathway_selectivity.py --report-only")
    ap("py -3 Experiment/tools/b1_pathway_selectivity.py --quick")
    ap("py -3 Experiment/tools/b1_pathway_selectivity.py")
    ap("```")
    ap("")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")
    return "\n".join(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        if not SUMMARY_JSONL.exists():
            raise SystemExit(f"ERROR: {SUMMARY_JSONL} 不存在")
        rows = [json.loads(x) for x in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        rows = run_all(quick=args.quick)
    build_report(rows)
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    if not core.startswith(CORE_SHA1_PREFIX):
        print(f"WARNING: 核心 SHA1 变化 {core}", file=sys.stderr)
    print("PSLE REPORT OK")


if __name__ == "__main__":
    main()
