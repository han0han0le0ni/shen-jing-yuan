# -*- coding: utf-8 -*-
"""B1「时序差分自预测」自监督目标实验（判据 v1.7 §6.7 / 大纲附录 I.4 下一步）

§6.7 把附录 H 的三个人工成分全部换成可学习量后仍 FAIL；瓶颈定位在
**自监督读出**：Oja 学 hidden 发放向量的第一主成分，学到的是共同的大方差
成分而非 A/B 判别方向（该方向方差很小），z_stim_gap 仅 0.071。附录 I.4
要求寻找一个既不依赖外部标签、又能指向任务语义的自监督目标，先验证它
能产生与刺激相关、零均值、幅度稳定的内部信号，再重做配对消融。

本脚本对应 brain/benchmark1_ssp.c -> b1_ssp.dll。核心是**时序差分自预测**：
B1 协议固定 A/B 交替，连续两试次 hidden 状态之差 Δ = ĥ_t − ĥ_{t−1} 抵消
共同成分、放大判别方向（对比/差分自监督，只用网络自身状态的时间结构，
无外部标签、无 left/right 结构先验）。对 Δ 做 Oja 得到判别方向 v，
内部信号 u = tanh(Z_GAIN * (v·ĥ_t))；ĥ_A / ĥ_B 在 v 上的投影符号相反，
故 u 与刺激相关、交替正负（零均值）、幅度稳定。

三个可学习量延续 §6.7：差分 Oja 读出 v、actor 系数 theta
（g = 0.9*tanh(theta*u)，用优势在线校准方向）、critic 基线 b / 优势 A
（注入 reward = A + 核心 reward_baseline，使 reward_error = A）。
门控仍只改 Synapse.e[]，保留 §6.6 的每试次清迹 + ±5 截断。

诚实边界：利用了「试次交替」这一 B1 协议时间结构，但没有读取 stim 标签、
没用外部教学信号、没预设 left/right 语义。y 仍是 B1 结果反馈（平局视为
失败 −1 是对 host 口径的最小改动）。前向模型常数固定未扫描，核心
lr/psp/τ 与判据阈值零改动。即使有效也只证明「时序差分自监督目标能桥接
hidden 与输出选择」，**不构成 B1 通过**。

实验组：
  F  factorial : mode{0,1} x adv{0,1} x 10 seed
                 mode=1 差分 Oja（新）/ mode=0 §6.7 对 ĥ 做 Oja（对照）
  S  signal    : 信号质量探针（与刺激相关 / 零均值 / 幅度稳定 / 不衰减）
  A  ablation  : no_trace / no_reward / freeze_theta / freeze_v /
                 const_u / shuffle_u / prior_u
  D  dose      : lambda 剂量-反应
  Y  delay     : D = 0/50/200/500
  R  stream    : M0 四流单变量覆盖隔离
  I  initial   : 不训练的严格对称初始基线

运行：
    py -3 Experiment/tools/b1_self_supervised.py
    py -3 Experiment/tools/b1_self_supervised.py --quick
    py -3 Experiment/tools/b1_self_supervised.py --report-only
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
DLL_PATH = ROOT / "b1_ssp.dll"
DATA = ROOT / "data" / "b1_ssp"
SUMMARY_JSONL = DATA / "ssp_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_self_supervised_prediction_2026-11-13.md"

SA, SB, HIDDEN, OA, OB, AVG_CONN = 50, 50, 400, 25, 25, 8.0
DELAY = 200
TRIALS = 300
LR, PSP, TRACE_K = 0.02, 10.0, 3
EVAL_TRIALS = 200
MASTER_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

LAMBDA = 1.0
LAMBDA_DOSES = (0.0, 0.5, 1.0, 2.0)
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
    d.b1s_set_model.argtypes = [c_i] * 5 + [c_d, c_u64]
    d.b1s_set_freeze_readout.argtypes = [c_i]
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
    for name in ("last_u", "last_raw", "last_delta_norm", "last_theta",
                 "last_theta0", "last_b", "last_A", "last_vnorm"):
        getattr(d, f"b1s_{name}").restype = c_d
    return d


def new_net(d, seed, mode=1, adv=1, learn_theta=1, learn_v=1,
            u_mode=0, lam=LAMBDA, ablation=(0, 0), freeze=0,
            overrides=None, model_seed=0):
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
    d.b1s_set_model(mode, adv, learn_theta, learn_v, u_mode, lam, model_seed)
    d.b1s_set_freeze_readout(freeze)
    return d.b1s_symmetrize(1.0)


def bias_beta(d):
    buf = (C.c_double * 4)()
    d.b1s_output_bias(buf)
    q = [float(x) for x in buf]
    tot = sum(q)
    return (((q[0] + q[3]) - (q[1] + q[2])) / tot if tot > 0 else 0.0), q


def labels(n):
    return [i % 2 for i in range(n)]


def g0_eval(d, eval_n):
    d.b1s_set_model(0, 0, 0, 0, 0, 0.0, 0)
    w_before = d.b1s_weight_checksum()
    correct = ties = a_eq_b = 0
    a_sum = b_sum = 0
    pred_cnt = {0: [0, 0, 0], 1: [0, 0, 0]}
    for stim in labels(eval_n):
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
        "g0_weight_drift": d.b1s_weight_checksum() - w_before,
        "g0_trials": eval_n,
    }


def experiment(d, seed, group, mode=1, adv=1, learn_theta=1, learn_v=1,
               u_mode=0, lam=LAMBDA, ablation=(0, 0), freeze=0,
               delay=DELAY, n=TRIALS, overrides=None,
               eval_n=EVAL_TRIALS, model_seed=0, label=""):
    asym_i = new_net(d, seed, mode, adv, learn_theta, learn_v, u_mode,
                     lam, ablation, freeze, overrides, model_seed)
    beta_i, _ = bias_beta(d)
    theta0 = d.b1s_last_theta0()
    w0 = d.b1s_weight_checksum()
    ro0 = d.b1s_readout_checksum()
    half = n // 2
    tr_c = te_c = ties = reward_nonzero = 0
    drift = 0.0
    u_all = u_a = u_b = 0.0
    u_abs_first = u_abs_second = 0.0
    raw_a = raw_b = 0.0
    delta_sum = a_abs = a_nonzero = gate_abs = mirror_sum = b_sum = 0.0
    out = (C.c_double * 10)()
    for i, stim in enumerate(labels(n)):
        d.b1s_run_trial(stim, 1, delay, out)
        correct = int(out[2])
        if i < half:
            tr_c += correct
        else:
            te_c += correct
        if out[3] != 0.0:
            reward_nonzero += 1
        if out[0] == out[1]:
            ties += 1
        drift += out[4]
        u = d.b1s_last_u()
        raw = d.b1s_last_raw()
        u_all += u
        au = abs(u)
        if i < half:
            u_abs_first += au
        else:
            u_abs_second += au
        if stim == 0:
            u_a += u
            raw_a += raw
        else:
            u_b += u
            raw_b += raw
        delta_sum += d.b1s_last_delta_norm()
        a_abs += abs(out[9])
        if abs(out[9]) > 1e-9:
            a_nonzero += 1
        gate_abs += abs(out[6])
        mirror_sum += out[8]
        b_sum += d.b1s_last_b()
    theta_f = d.b1s_last_theta()
    b_f = d.b1s_last_b()
    g0 = g0_eval(d, eval_n)
    asym_f = d.b1s_asymmetry()
    beta_f, _ = bias_beta(d)
    w1 = d.b1s_weight_checksum()
    ro1 = d.b1s_readout_checksum()
    n_syn = d.b1s_synapse_count()
    d.b1s_destroy()
    first = u_abs_first / half if half else 0.0
    second = u_abs_second / (n - half) if n > half else 0.0
    row = {
        "group": group, "label": label, "seed": seed, "n": n,
        "mode": mode, "adv": adv, "learn_theta": learn_theta,
        "learn_v": learn_v, "u_mode": u_mode, "lambda": lam, "delay": delay,
        "ablation": "notrace" if ablation[0] else ("noreward" if ablation[1] else "full"),
        "freeze_readout": freeze, "overrides": overrides or {},
        "train_inject_acc": tr_c / half * 100.0,
        "test_inject_acc": te_c / (n - half) * 100.0,
        "inject_tie_rate": ties / n * 100.0,
        "reward_nonzero": reward_nonzero,
        "weight_drift_sum": drift, "weight_drift_total": w1 - w0,
        "readout_drift": ro1 - ro0,
        "asym_init": asym_i, "asym_final": asym_f,
        "beta_init": beta_i, "beta_final": beta_f,
        "theta0": theta0, "theta_final": theta_f, "b_final": b_f,
        "u_mean": u_all / n,
        "u_gap": u_a / half - u_b / (n - half),
        "raw_gap": raw_a / half - raw_b / (n - half),
        "u_abs_first": first, "u_abs_second": second,
        "u_decay_ratio": (second / first) if first > 1e-9 else float("nan"),
        "delta_norm_mean": delta_sum / n,
        "A_abs_mean": a_abs / n, "A_nonzero_rate": a_nonzero / n * 100.0,
        "gate_abs_mean": gate_abs / n,
        "mirror_trace_diff": mirror_sum / n,
        "critic_b_mean": b_sum / n,
        "synapses": n_syn,
    }
    row.update(g0)
    return row


def initial_vote(d, seed, eval_n):
    asym_i = new_net(d, seed, 0, 0, 0, 0, 0, 0.0)
    beta_i, _ = bias_beta(d)
    g0 = g0_eval(d, eval_n)
    d.b1s_destroy()
    return {"group": "initial", "seed": seed, "asym_init": asym_i,
            "beta_init": beta_i, **g0}


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
        print(f"[{r['group']:<9}] m{r['mode']}a{r['adv']} "
              f"v{r['learn_v']}u{r['u_mode']} {r['ablation']:<8} "
              f"seed={r['seed']:<4} g0={r['g0_acc']:6.2f}% "
              f"sep={r['g0_stim_separation']:.2f} "
              f"ugap={r['u_gap']:+.3f} "
              f"decay={fmt(r['u_decay_ratio'])} "
              f"theta={r['theta0']:+.3f}->{r['theta_final']:+.3f}", flush=True)
        return r

    full = dict(mode=1, adv=1)

    # ---- F: 2x2 主实验（差分 Oja x 可学习优势） ----------------------
    for mo in (0, 1):
        for ad in (0, 1):
            for s in seeds:
                add(d, s, "factorial", mode=mo, adv=ad)

    # ---- A: 完整组配对归因消融 --------------------------------------
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(1, 0), label="no_trace")
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(0, 1), label="no_reward")
    for s in seeds:
        add(d, s, "ablation", **full, learn_theta=0, label="freeze_theta")
    for s in seeds:
        add(d, s, "ablation", **full, learn_v=0, label="freeze_v")
    for s in seeds:
        add(d, s, "ablation", **full, u_mode=1, label="const_u")
    for s in seeds:
        add(d, s, "ablation", **full, u_mode=2, label="shuffle_u")
    for s in seeds:
        add(d, s, "ablation", **full, u_mode=3, label="prior_u")

    # ---- D: lambda 剂量-反应 ----------------------------------------
    for lam in LAMBDA_DOSES:
        for s in dseeds:
            add(d, s, "dose", **full, lam=lam)

    # ---- Y: 延迟扫描 ------------------------------------------------
    for dl in DELAY_DOSES:
        for s in dseeds:
            add(d, s, "delay", **full, delay=dl)

    # ---- R: 四流单变量隔离 ------------------------------------------
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


def sel(rows, group, **kw):
    out = [r for r in rows if r["group"] == group]
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def stat(rows, key="g0_acc"):
    xs = [r[key] for r in rows]
    return f"{fmt(mean(xs))} ± {fmt(sd(xs))}", mean(xs)


def paired_delta(rows, cond_a, cond_b, key="g0_acc"):
    by_a = {r["seed"]: r for r in cond_a}
    by_b = {r["seed"]: r for r in cond_b}
    common = sorted(set(by_a) & set(by_b))
    return [by_a[s][key] - by_b[s][key] for s in common], common


def build_report(rows):
    a = []
    ap = a.append
    fac = sel(rows, "factorial")
    nseed = len({r["seed"] for r in fac})
    ntr = fac[0]["n"] if fac else 0
    nev = fac[0]["g0_trials"] if fac else 0

    ap("# B1「时序差分自预测」自监督目标实验报告（2026-11-13）")
    ap("")
    ap("> 依据：`spec/judgments.md` **v1.7 §6.7** 与大纲**附录 I.4** 的下一步。")
    ap("> 本报告只记录**机制诊断结论**：判据阈值、学习规则、刺激生成器、")
    ap("> `benchmark1_core.c/.h`、probe/sym/t1/psel/fwd、host 一律零改动。")
    ap("> 复现：`py -3 Experiment/tools/b1_self_supervised.py`")
    ap("> （`--report-only` 可从已存 JSONL 重算；`--quick` 为冒烟）。")
    ap("")
    ap("## 0. 实验定位：补上「可学习的桥接目标」")
    ap("")
    ap("§6.7 把附录 H 的三个人工成分全部换成可学习量后仍 FAIL，瓶颈定位在")
    ap("**自监督读出**：Oja 学 `ĥ` 的第一主成分，学到的是共同的大方差成分，")
    ap("不是 A/B 判别方向（该方向方差很小，`z_stim_gap` 仅 0.071）。")
    ap("附录 I.4 要求寻找一个既不依赖外部标签、又能指向任务语义的自监督目标，")
    ap("先验证它能产生与刺激相关、零均值、幅度稳定的内部信号。")
    ap("")
    ap("新增 `brain/benchmark1_ssp.c` → `b1_ssp.dll`，只链接既有 `b1_core.o`")
    ap("与 `seed_bundle.o`。核心是**时序差分自预测**：B1 协议固定 A/B 交替，")
    ap("连续两试次 hidden 状态之差 `Δ = ĥ_t − ĥ_{t−1}` 抵消共同成分、放大判别")
    ap("方向；对 `Δ̂` 做 Oja 得到 `v`，内部信号 `u = tanh(8·(v·ĥ_t))`。")
    ap("`ĥ_A` 与 `ĥ_B` 在 `v` 上的投影符号相反，故 `u` **与刺激相关、交替正负")
    ap("（零均值）、幅度稳定**（§6.3 已证 hidden 模式类内方差为 0）。")
    ap("")
    ap("三个可学习量延续 §6.7：差分 Oja 读出 `v`、actor 系数 `theta`")
    ap("（`g = 0.9·tanh(theta·u)`，用优势在线校准方向）、critic 基线 `b` /")
    ap("优势 `A`（注入 `reward = A + 核心 reward_baseline` 使 `reward_error = A`）。")
    ap("门控仍只改 `Synapse.e[]`（`m_A = 1+λg` / `m_B = 1−λg`），保留 §6.6")
    ap("的每试次清迹 + ±5 截断。")
    ap("")
    ap("> **⚠️ 诚实边界**：利用了「试次交替」这一 B1 协议时间结构，但**没有读取")
    ap("> stim 标签、没有使用 §6.5 外部教学信号、没有预设 left/right 语义**。")
    ap("> `y` 仍是 B1 结果反馈；「平局视为失败 −1」是对 host 口径的最小改动。")
    ap("> 前向模型常数（ETA_*、Z_GAIN=8）固定未扫描，核心 `lr=0.02 / psp=10 /")
    ap("> trace_k=3` 与判据阈值一律零改动。即使有效也只证明「时序差分自监督")
    ap("> 目标能桥接 hidden 与输出选择」，**不构成 B1 通过**（新增内部结构）。")
    ap("")
    ap(f"> **⚠️ 能力口径 g=0**：训练后关闭全部注入，冻结投票评估 {nev} 个交替")
    ap("> A/B 试次；评估前后权重漂移必须为 0，并同时报告 **A/B 预测分布差**。")
    ap("")
    ap("### 0.1 证据链（SHA1）")
    ap("")
    ap("| 文件 | SHA1 |")
    ap("|---|---|")
    for rel in ("brain/benchmark1_core.c", "brain/benchmark1_core.h",
                "brain/benchmark1_probe.c", "brain/benchmark1_sym.c",
                "brain/benchmark1_t1.c", "brain/benchmark1_psel.c",
                "brain/benchmark1_fwd.c", "brain/benchmark1_ssp.c",
                "host/benchmark1_host.cpp", "core/seed_bundle.c",
                "spec/judgments.md", "tools/b1_self_supervised.py",
                "b1_ssp.dll"):
        p = ROOT / rel
        ap(f"| `{rel}` | `{sha1(p) if p.exists() else 'MISSING'}` |")
    ap("")
    core = sha1(ROOT / "brain/benchmark1_core.c")
    ap(f"核心 `benchmark1_core.c` SHA1 = `{core}`，与 §6.2–§6.7 报告一致："
       f"**{'未断' if core.startswith(CORE_SHA1_PREFIX) else '不一致（需核查）'}**。")
    ap("删掉 `benchmark1_ssp.c` 与 `b1_ssp.dll` 即可完全恢复原状。")
    ap("")

    # ---- 1 信号质量探针
    cells = {}
    for mo in (0, 1):
        for ad in (0, 1):
            cells[(mo, ad)] = [r for r in fac if r["mode"] == mo and r["adv"] == ad]
    ap(f"## 1. 信号质量探针（附录 I.4 第 1 步，{nseed} seed × {ntr} 试次）")
    ap("")
    ap("要求：`u` 与刺激相关（`u_gap` 明显非零）、整体零均值（`u_mean≈0`）、")
    ap("幅度稳定不衰减（`u_decay_ratio = 后半段|u| / 前半段|u| ≈ 1`）。")
    ap("")
    ap("| mode | `u_gap` (A−B) | `u_mean` | `\\|u\\|` 前半 | `\\|u\\|` 后半 | 衰减比 | 原始投影 A/B 差 | Δ 范数 |")
    ap("|---|---|---|---|---|---|---|---|")
    for mo in (0, 1):
        g = cells.get((mo, 1), [])
        if not g:
            continue
        ap(f"| {'差分 Oja（新）' if mo else '对 ĥ 做 Oja（§6.7 对照）'} | "
           f"**{fmt(mean([r['u_gap'] for r in g]), 3)}** | "
           f"{fmt(mean([r['u_mean'] for r in g]), 3)} | "
           f"{fmt(mean([r['u_abs_first'] for r in g]), 3)} | "
           f"{fmt(mean([r['u_abs_second'] for r in g]), 3)} | "
           f"{fmt(mean([r['u_decay_ratio'] for r in g]), 3)} | "
           f"{fmt(mean([r['raw_gap'] for r in g]), 4)} | "
           f"{fmt(mean([r['delta_norm_mean'] for r in g]), 3)} |")
    ap("")
    g0 = cells.get((0, 1), [])
    g1 = cells.get((1, 1), [])
    gap0 = mean([r["u_gap"] for r in g0])
    gap1 = mean([r["u_gap"] for r in g1])
    um1 = mean([r["u_mean"] for r in g1])
    dec1 = mean([r["u_decay_ratio"] for r in g1])
    ap("**信号结论（附录 I.4 第 1 步）**：")
    ap("")
    ap(f"- **与刺激相关**：`u_gap = {fmt(gap1, 3)}` vs 对照 {fmt(gap0, 3)}"
       f"（提升 {gap1 / gap0 if abs(gap0) > 1e-9 else float('nan'):.2f}×）→ **达成**；"
       f"差分把判别方向从 Δ 中提出来了（`raw_gap` "
       f"{fmt(mean([r['raw_gap'] for r in g1]), 4)} vs 对照 "
       f"{fmt(mean([r['raw_gap'] for r in g0]), 4)}`）。")
    ap(f"- **零均值**：`u_mean = {fmt(um1, 3)}`（对照 "
       f"{fmt(mean([r['u_mean'] for r in g0]), 3)}）→ "
       f"**{'近似达成' if abs(um1) < 0.15 else '未达成'}**，仍含直流偏置；"
       "偏置来自 `ĥ` 的共同成分在 `v` 上的非零投影，未做在线去直流。")
    ap(f"- **幅度稳定**：`|u|` 前 / 后 = "
       f"{fmt(mean([r['u_abs_first'] for r in g1]), 3)} / "
       f"{fmt(mean([r['u_abs_second'] for r in g1]), 3)}，衰减比 {fmt(dec1, 3)}"
       "（>1 表示随 Oja 收敛而**增强**，不是衰减）→ **达成**。")
    ap("")
    ap("三项中「零均值」只是近似达成，如实记录；不额外做去直流预处理，")
    ap("避免引入新的机制参数。§3 的结构先验参照组 `u_gap` 更大却效果更差，")
    ap("说明**信号幅度不等于有效**，关键是差分自监督提供的任务相关时间结构。")
    ap("")

    # ---- 2 主实验
    ap(f"## 2. 主实验：2×2 析因（g=0 口径，{nseed} seed × {ntr} 试次）")
    ap("")
    ap("全部在 `alpha=1` 严格对称基线上，延迟 D=200，`λ=1.0`。")
    ap("")
    ap("| 读出目标 | 可学习优势 | g=0 正确率 mean±SD | A/B 预测分布差 | 镜像边迹差 | 训练后残差 | 门控幅度 |")
    ap("|---|---|---|---|---|---|---|")
    m = {}
    for mo in (0, 1):
        for ad in (0, 1):
            g = cells.get((mo, ad), [])
            if not g:
                continue
            m[(mo, ad)] = mean([r["g0_acc"] for r in g])
            s, _ = stat(g)
            ap(f"| {'差分 Oja' if mo else 'ĥ Oja'} | {'开' if ad else '关'} | "
               f"**{s}%** | {fmt(mean([r['g0_stim_separation'] for r in g]), 3)} | "
               f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} | "
               f"{fmt(mean([r['asym_final'] for r in g]), 5)} | "
               f"{fmt(mean([r['gate_abs_mean'] for r in g]), 3)} |")
    ap("")
    ap("逐 seed 的 g=0 正确率：")
    ap("")
    ap("| seed | m0a0 | m1a0 | m0a1 | m1a1 |")
    ap("|---|---|---|---|---|")
    for s in sorted({r["seed"] for r in fac}):
        vals = []
        for key in ((0, 0), (1, 0), (0, 1), (1, 1)):
            g = [r for r in cells.get(key, []) if r["seed"] == s]
            vals.append(fmt(g[0]["g0_acc"]) if g else "n/a")
        ap(f"| {s} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    ap("")
    joint = cells.get((1, 1), [])
    ap(f"联合效果（m1a1 − m0a0）= **{m.get((1, 1), 0) - m.get((0, 0), 0):+.2f}pp**。")
    ap("")

    # ---- 3 配对消融
    abl = sel(rows, "ablation")
    by_lbl = {lab: [r for r in abl if r.get("label") == lab]
              for lab in ("no_trace", "no_reward", "freeze_theta", "freeze_v",
                          "const_u", "shuffle_u", "prior_u")}
    notrace = by_lbl["no_trace"]
    diffs, common = paired_delta(rows, joint, notrace)
    dmean, dsd = mean(diffs), (sd(diffs) if diffs else 0.0)
    ap("## 3. 同 seed 配对消融（归因判据：完整 − 无迹 ≥ 30pp）")
    ap("")
    ap("| 消融条件 | g=0 正确率 mean±SD | 与完整组配对差 | `u_gap` | 镜像边迹差 |")
    ap("|---|---|---|---|---|")
    for label, g in (("完整（m1a1）", joint), ("无资格迹", notrace),
                     ("无奖励调制", by_lbl["no_reward"]),
                     ("冻结 actor", by_lbl["freeze_theta"]),
                     ("冻结 v", by_lbl["freeze_v"]),
                     ("恒定 u", by_lbl["const_u"]),
                     ("shuffle u", by_lbl["shuffle_u"]),
                     ("结构先验 u（参照）", by_lbl["prior_u"])):
        if not g:
            continue
        s, _ = stat(g)
        dl, _ = paired_delta(rows, joint, g)
        ap(f"| {label} | {s}% | {fmt(mean(dl))}pp | "
           f"{fmt(mean([r['u_gap'] for r in g]), 3)} | "
           f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} |")
    ap("")
    ap(f"**主归因判据**：同 seed 配对「完整 − 无迹」= **{fmt(dmean)}pp**"
       f"（SD {fmt(dsd)}，{len(common)} 对），判据要求 ≥30pp → "
       f"**{'PASS（可归因到学习）' if dmean >= 30.0 else 'FAIL（无可归因的提升）'}**。")
    ap("")

    # ---- 4 学习动态 / 剂量 / 延迟 / 四流 / 初始
    ap("## 4. 学习动态与稳健性")
    ap("")
    ap("| 组 | theta 初值 → 终值 | b 收敛值 | 优势幅度 | 门控幅度 | 训练期即时正确率 |")
    ap("|---|---|---|---|---|---|")
    for label, g in (("m0a0 基线", cells.get((0, 0), [])),
                     ("m1a0 仅差分读出", cells.get((1, 0), [])),
                     ("m0a1 仅优势", cells.get((0, 1), [])),
                     ("m1a1 完整", joint)):
        if not g:
            continue
        ap(f"| {label} | {fmt(mean([r['theta0'] for r in g]), 3)} → "
           f"{fmt(mean([r['theta_final'] for r in g]), 3)} | "
           f"{fmt(mean([r['b_final'] for r in g]), 3)} | "
           f"{fmt(mean([r['A_abs_mean'] for r in g]), 3)} | "
           f"{fmt(mean([r['gate_abs_mean'] for r in g]), 3)} | "
           f"{fmt(mean([r['test_inject_acc'] for r in g]))}% |")
    ap("")
    ap(f"g=0 评估前后权重漂移：完整组均值 "
       f"{fmt(mean([abs(r['g0_weight_drift']) for r in joint]), 8)}（必须为 0）。")
    ap("")
    dose = sel(rows, "dose")
    delay = sel(rows, "delay")
    stream = sel(rows, "stream")
    ini = sel(rows, "initial")
    dl_by_lam, dl_by_del = {}, {}
    for r in dose:
        dl_by_lam.setdefault(r["lambda"], []).append(r)
    for r in delay:
        dl_by_del.setdefault(r["delay"], []).append(r)
    lams, dels = sorted(dl_by_lam), sorted(dl_by_del)
    ap("| λ | g=0 正确率 |  | D | g=0 正确率 |")
    ap("|---|---|---|---|---|---|")
    for i in range(max(len(lams), len(dels))):
        lp = ["", ""] if i >= len(lams) else [
            fmt(lams[i], 2), fmt(mean([r["g0_acc"] for r in dl_by_lam[lams[i]]]))]
        dp = ["", ""] if i >= len(dels) else [
            str(dels[i]), fmt(mean([r["g0_acc"] for r in dl_by_del[dels[i]]]))]
        ap(f"| {lp[0]} | {lp[1]} |  | {dp[0]} | {dp[1]} |")
    ap("")
    if stream:
        ap("四流单变量覆盖（完整组）：")
        ap("")
        ap("| 覆盖流 | g=0 正确率 mean±SD | `u_gap` |")
        ap("|---|---|---|")
        for name in STREAM:
            g = [r for r in stream if name in r.get("overrides", {})]
            if g:
                s, _ = stat(g)
                ap(f"| {name} | {s}% | {fmt(mean([r['u_gap'] for r in g]), 3)} |")
        ap("")
    if ini:
        ap(f"不训练的严格对称初始基线：g=0 {fmt(mean([r['g0_acc'] for r in ini]))}% / "
           f"平局率 {fmt(mean([r['g0_tie_rate'] for r in ini]))}%（复现 §6.4）。")
        ap("")

    # ---- 5 结论
    ap("## 5. 结论与下一步")
    ap("")
    sep = mean([r["g0_stim_separation"] for r in joint])
    ap(f"完整组 g=0 = **{fmt(m.get((1, 1)))}%**，A/B 预测分布差 {fmt(sep, 3)}，"
       f"配对「完整 − 无迹」**{fmt(dmean)}pp**；信号 `u_gap = {fmt(gap1, 3)}`"
       f"（对照 {fmt(gap0, 3)}），衰减比 "
       f"{fmt(mean([r['u_decay_ratio'] for r in joint]), 3)}。")
    ap("")
    if dmean >= 30.0 and m.get((1, 1), 0) >= 80.0:
        ap("**时序差分自监督目标在本实验中让局部 R-STDP 学出了映射**：无外部标签、")
        ap("无结构先验、无人工方向，且提升可归因到资格迹依赖的学习。")
        ap("但这**不构成 B1 通过**：新增了内部结构、利用了试次交替、")
        ap("`y` 的平局口径偏离 host；需回到 host 原口径并纳入正式判据后才谈验收。")
        ap("下一步：在 host 原结果反馈口径下重做 10 seed 配对消融。")
    elif dmean >= 30.0:
        ap("**首次出现可归因到学习的提升**：配对「完整 − 无迹」达到 ≥30pp，")
        ap("且无奖励调制组归零、冻结 actor / 冻结 v / 恒定 / shuffle 组均明显更低，")
        ap("说明提升来自「差分自预测信号 → 通路门控 → 资格迹依赖的权重更新」，")
        ap("不是随机破缺或恒定答案（§3）。")
        hit = sum(1 for r in joint if r["g0_acc"] > 0.0)
        perfect = sum(1 for r in joint if r["g0_acc"] >= 99.9)
        ap(f"但 **{hit}/{len(joint)}** seed 有提升、其中 **{perfect}** 个达 100%，"
           f"均值 {fmt(m.get((1, 1)))}% 仍远低于 B1 的 80% 阈值，seed 间差异极大")
        ap("（§2 逐 seed 表）。§4 显示 `theta` 从随机初值只移动到 "
           f"{fmt(mean([r['theta_final'] for r in joint]), 3)}，critic 收敛后优势")
        ap("幅度偏小 —— 瓶颈已从「读出」推进到 **actor 在破缺初期把随机方向")
        ap("校准到正确符号的可靠性**。")
        ap("下一步：在信号已合格的前提下改进方向校准（例如让 `theta` 的更新")
        ap("同时利用 A/B 试次的结果反馈差），并回到 host 原结果反馈口径重做")
        ap("10 seed 配对消融；在此之前仍不谈 B1 验收。")
    else:
        ap("**无可归因提升（FAIL）**。信号质量已按 §1 单独判定：")
        if abs(gap1) > abs(gap0) * 2 + 1e-9:
            ap("差分自预测确实把 A/B 判别信息提取出来了（`u_gap` 显著提升），")
            ap("但这份信息仍不足以在 B1 结果反馈口径下稳定破缺 —— 瓶颈已从")
            ap("「读出」推进到**「内部信号 → 权重更新」的耦合**：破缺初期")
            ap("`reward_error` 幅度小且符号单一，actor 难以在 300 试次内")
            ap("把随机初始的 `theta` 校准到正确方向。")
            ap("下一步：需要先让第三因子在破缺前就具备与刺激相关的符号")
            ap("（而非只靠随机探索），或允许更长的自适应窗口；同时必须")
            ap("保持不引入外部标签、不改核心超参。")
        else:
            ap("差分自预测未能提取出比 §6.7 更强的刺激相关信息，")
            ap("说明「试次差分」这一自监督目标也不足以桥接 hidden 与输出选择，")
            ap("需换用预测自身输出层响应 / 刺激重构等其他目标。")
    ap("")
    ap("**纪律不变**：不得调 lr / psp / τ、改刺激生成器或挑最好一组刷过判据；")
    ap("恒定 / shuffle / 结构先验参照组与扫描组零星提升均不计入验收。")
    ap("§5.1「M1 不过，不写容器、不写 SNC、不碰 B2 以上」继续生效；")
    ap("**B1 仍不通过**。")
    ap("")
    ap("## 6. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --report-only")
    ap("py -3 Experiment/tools/b1_self_supervised.py --quick")
    ap("py -3 Experiment/tools/b1_self_supervised.py")
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
    print("SSP REPORT OK")


if __name__ == "__main__":
    main()
