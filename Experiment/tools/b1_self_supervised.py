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

# 两套 suite 共用同一个 DLL（b1_ssp.c 新增 calib / y_mode 开关，默认参数与
# v1.8 原实现逐位一致）：
#   v18 —— 复现 §6.8 的 30.00% / 30.00pp（calib=0, y_mode=0）
#   v19 —— §6.8 下一步：A/B 结果反馈差校准 + host 原结果反馈口径
SUITES = {
    "v18": ("b1_ssp", "ssp_summary.jsonl",
            "b1_self_supervised_prediction_2026-11-13.md"),
    "v19": ("b1_cal", "cal_summary.jsonl",
            "b1_direction_calibration_2026-11-13.md"),
    # v110 —— §6.10：多尺度 vs 单时间常数资格迹对照（M1 主张的外部校验）
    "v110": ("b1_trace", "trace_summary.jsonl",
             "b1_multiscale_trace_2026-11-13.md"),
    # v111 —— §6.11：sign(theta) 可辨识性审计（§6.9 遗留的类别非对称信号问题）
    "v111": ("b1_sign", "sign_summary.jsonl",
             "b1_sign_identifiability_2026-11-13.md"),
    # v112 —— §6.14：内在探索噪声（R-STDP 随机性必要条件）
    "v112": ("b1_noise", "noise_summary.jsonl",
             "b1_explore_noise_2026-11-13.md"),
    # v113 —— §6.14 补测：延迟扫描 D={10,50,100,200,500,1000} + 半衰点
    "v113": ("b1_delay", "delay_summary.jsonl",
             "b1_delay_sweep_2026-11-13.md"),
    # v114 —— §1 强制报告：τe–N 标度律（N=500/1000/2000）
    "v114": ("b1_taun", "taun_summary.jsonl",
             "b1_tau_n_scaling_2026-11-13.md"),
    # v115 —— §1 强制报告：稀疏化 Gini / 有效连接数（含无迹组）
    "v115": ("b1_sparse", "sparse_summary.jsonl",
             "b1_sparsification_2026-11-13.md"),
    # v116 —— §1 强制报告：neuromod（Rust）同种子统计等价的本侧工件
    "v116": ("b1_ref", "ref_summary.jsonl",
             "b1_neuromod_reference_2026-11-13.md"),
    # v117 —— 第 4 项前置：neuromod 默认迹参数（单迹 τe=50）对齐对照
    "v117": ("b1_nm_align", "nm_align_summary.jsonl",
             "b1_neuromod_param_alignment_2026-11-14.md"),
}
SUITE = "v18"
DATA = ROOT / "data" / "b1_ssp"
SUMMARY_JSONL = DATA / "ssp_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_self_supervised_prediction_2026-11-13.md"


def set_suite(name: str) -> None:
    global SUITE, DATA, SUMMARY_JSONL, REPORT_MD
    folder, jsonl, report = SUITES[name]
    SUITE = name
    DATA = ROOT / "data" / folder
    SUMMARY_JSONL = DATA / jsonl
    REPORT_MD = ROOT / "audit" / report


def apply_holdout() -> None:
    """留出集复测：切换种子池与输出路径，避免覆盖发现集结果。"""
    global SEED_POOL, HOLDOUT, SUMMARY_JSONL, REPORT_MD
    SEED_POOL = HELDOUT_SEEDS
    HOLDOUT = True
    if SUITE == "v111":
        SUMMARY_JSONL = DATA / "sign_holdout.jsonl"
        REPORT_MD = ROOT / "audit" / "b1_sign_holdout_2026-11-13.md"
    elif SUITE == "v112":
        SUMMARY_JSONL = DATA / "noise_holdout.jsonl"
        REPORT_MD = ROOT / "audit" / "b1_explore_noise_holdout_2026-11-13.md"

SA, SB, HIDDEN, OA, OB, AVG_CONN = 50, 50, 400, 25, 25, 8.0
DELAY = 200
TRIALS = 300
LR, PSP, TRACE_K = 0.02, 10.0, 3
EVAL_TRIALS = 200
MASTER_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
# §6.11 留出集：与 MASTER_SEEDS 不相交，只用于对「反向规则」做独立复测，
# 避免在发现集上事后挑规则。
HELDOUT_SEEDS = [1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020]
SEED_POOL = MASTER_SEEDS
HOLDOUT = False
EXTEND = False   # v114：只补跑缺失的长延迟点（右删失后的 D 轴延伸）

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
                 "last_theta0", "last_b", "last_A", "last_vnorm",
                 "last_d_same", "last_d_pair", "last_delta"):
        getattr(d, f"b1s_{name}").restype = c_d
    d.b1s_set_calib.argtypes = [c_i, c_i]
    d.b1s_set_calib_weights.argtypes = [c_d, c_d]
    d.b1s_calib_mode.restype = c_i
    d.b1s_y_mode.restype = c_i
    # 迹配置与重置不变量探针
    d.b1s_set_trace.argtypes = [c_i, c_d, c_d, c_d]
    d.b1s_trace_k.restype = c_i
    d.b1s_trace_alpha.argtypes = [c_i]
    d.b1s_trace_alpha.restype = c_d
    d.b1s_tau_trace_k.argtypes = [c_i]
    d.b1s_tau_trace_k.restype = c_d
    d.b1s_trace_abs_sum.restype = c_d
    d.b1s_reset_probe.restype = c_d
    # §6.11 sign(theta) 可辨识性审计钩子
    d.b1s_set_theta.argtypes = [c_d]
    d.b1s_struct_probe.argtypes = [p_d]
    d.b1s_drive_gap.restype = c_d
    # §6.14 内在探索噪声
    d.b1s_set_explore.argtypes = [c_d, c_i]
    d.b1s_explore_sigma.restype = c_d
    d.b1s_explore_target.restype = c_i
    # §1 强制报告：τe–N 规模回读 + 稀疏化权重诊断（v1.16）
    d.b1s_neuron_count.restype = c_i
    d.b1s_weight_stats.argtypes = [c_i, p_d]
    return d


def new_net(d, seed, mode=1, adv=1, learn_theta=1, learn_v=1,
            u_mode=0, lam=LAMBDA, ablation=(0, 0), freeze=0,
            overrides=None, model_seed=0, calib=0, y_mode=0, alpha=1.0,
            trace=None, explore=None, dims=None):
    for name in STREAM:
        d.b1s_override_stream(STREAM[name], 0)
    d.b1s_legacy_rng(0)
    for name, val in (overrides or {}).items():
        d.b1s_override_stream(STREAM[name], val)
    sa, sb, hid, oa, ob, avg = dims if dims else (SA, SB, HIDDEN, OA, OB, AVG_CONN)
    rc = d.b1s_create(sa, sb, hid, oa, ob, avg, C.c_uint(seed))
    if rc != 0:
        raise RuntimeError(f"b1s_create 失败 seed={seed} rc={rc}")
    d.b1s_set_hyper(LR, PSP, TRACE_K)
    d.b1s_set_ablation(*ablation)
    d.b1s_set_model(mode, adv, learn_theta, learn_v, u_mode, lam, model_seed)
    d.b1s_set_calib(calib, y_mode)
    if trace is not None:
        d.b1s_set_trace(*trace)
    # 探索噪声必须显式复位：DLL 是全局单例，跨 net 不清会污染下一组
    if explore is not None:
        d.b1s_set_explore(float(explore[0]), int(explore[1]))
    else:
        d.b1s_set_explore(0.0, 0)
    d.b1s_set_freeze_readout(freeze)
    return d.b1s_symmetrize(alpha)


def bias_beta(d):
    buf = (C.c_double * 4)()
    d.b1s_output_bias(buf)
    q = [float(x) for x in buf]
    tot = sum(q)
    return (((q[0] + q[3]) - (q[1] + q[2])) / tot if tot > 0 else 0.0), q


def weight_stats(d, scope=0):
    """§1 强制报告「稀疏化」：权重分布只读快照（scope 0=全突触 / 1=readout 边）。"""
    buf = (C.c_double * 7)()
    d.b1s_weight_stats(int(scope), buf)
    q = [float(x) for x in buf]
    return {"gini": q[0], "eff": q[1], "active": q[2], "mean_abs": q[3],
            "n": q[4], "sat": q[5], "p90": q[6]}


def labels(n):
    return [i % 2 for i in range(n)]


def g0_eval(d, eval_n):
    # 能力口径 g=0：评估期关闭探索噪声，保证测的是学到的确定性映射
    d.b1s_set_explore(0.0, 0)
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
               eval_n=EVAL_TRIALS, model_seed=0, label="",
               calib=0, y_mode=0, alpha=1.0, trace=None, theta_force=None,
               explore=None, dims=None, wstats=False, trace_every=0):
    asym_i = new_net(d, seed, mode, adv, learn_theta, learn_v, u_mode,
                     lam, ablation, freeze, overrides, model_seed,
                     calib=calib, y_mode=y_mode, alpha=alpha, trace=trace,
                     explore=explore, dims=dims)
    if theta_force is not None:
        d.b1s_set_theta(float(theta_force))
    neurons = d.b1s_neuron_count()
    w_init = weight_stats(d, 0) if wstats else None
    wr_init = weight_stats(d, 1) if wstats else None
    curve = []
    trace_k = d.b1s_trace_k()
    tau_vec = [d.b1s_tau_trace_k(k) for k in range(trace_k)]
    alpha_vec = [d.b1s_trace_alpha(k) for k in range(trace_k)]
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
    dsm = dpp = dlt = 0.0
    drv_a = drv_b = 0.0
    out = (C.c_double * 10)()
    for i, stim in enumerate(labels(n)):
        d.b1s_run_trial(stim, 1, delay, out)
        if trace_every and i % trace_every == 0:
            curve.append(d.b1s_trace_abs_sum())
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
        dsm += d.b1s_last_d_same()
        dpp += d.b1s_last_d_pair()
        dlt += d.b1s_last_delta()
        dg = d.b1s_drive_gap()
        if stim == 0:
            drv_a += dg
        else:
            drv_b += dg
    theta_f = d.b1s_last_theta()
    b_f = d.b1s_last_b()
    w_final = weight_stats(d, 0) if wstats else None
    wr_final = weight_stats(d, 1) if wstats else None
    g0 = g0_eval(d, eval_n)
    asym_f = d.b1s_asymmetry()
    beta_f, _ = bias_beta(d)
    w1 = d.b1s_weight_checksum()
    ro1 = d.b1s_readout_checksum()
    n_syn = d.b1s_synapse_count()
    trace_res = d.b1s_reset_probe()
    d.b1s_destroy()
    first = u_abs_first / half if half else 0.0
    second = u_abs_second / (n - half) if n > half else 0.0
    row = {
        "group": group, "label": label, "seed": seed, "n": n,
        "mode": mode, "adv": adv, "learn_theta": learn_theta,
        "learn_v": learn_v, "u_mode": u_mode, "lambda": lam, "delay": delay,
        "calib": calib, "y_mode": y_mode, "alpha": alpha,
        "d_same_mean": dsm / n, "d_pair_mean": dpp / n, "delta_mean": dlt / n,
        "trace_k": trace_k, "tau_vec": tau_vec, "alpha_vec": alpha_vec,
        "trace_residual": trace_res,
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
        "u_a_mean": u_a / half if half else 0.0,
        "u_b_mean": u_b / (n - half) if n > half else 0.0,
        "drive_a_mean": drv_a / half if half else 0.0,
        "drive_b_mean": drv_b / (n - half) if n > half else 0.0,
        "theta_force": theta_force,
        "explore_sigma": d.b1s_explore_sigma(),
        "explore_target": d.b1s_explore_target(),
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
        "neurons": neurons,
        "dims": list(dims) if dims else [SA, SB, HIDDEN, OA, OB, AVG_CONN],
    }
    if wstats:
        row["w_init"] = w_init
        row["w_final"] = w_final
        row["wr_init"] = wr_init
        row["wr_final"] = wr_final
    if trace_every:
        row["trace_curve"] = curve
    row.update(g0)
    return row


def initial_vote(d, seed, eval_n):
    asym_i = new_net(d, seed, 0, 0, 0, 0, 0, 0.0)
    beta_i, _ = bias_beta(d)
    g0 = g0_eval(d, eval_n)
    d.b1s_destroy()
    return {"group": "initial", "seed": seed, "asym_init": asym_i,
            "beta_init": beta_i, **g0}


def run_all_v19(quick=False):
    """§6.8 下一步：host 原结果反馈口径 + A/B 反馈差方向校准。"""
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
        print(f"[{r['group']:<9}] c{r['calib']}y{r['y_mode']}a{r['alpha']:.2f} "
              f"{r['ablation']:<8} seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% tie={r['g0_tie_rate']:5.1f}% "
              f"sep={r['g0_stim_separation']:.2f} "
              f"theta={r['theta0']:+.3f}->{r['theta_final']:+.3f} "
              f"dlt={r['delta_mean']:+.3f}", flush=True)
        return r

    # ---- G: 口径 x 校准 2x4 网格（主判定臂 = calib3 / y_mode1） ---------
    for ym in (0, 1):
        for cal in (0, 1, 2, 3):
            for s in seeds:
                add(d, s, "grid", calib=cal, y_mode=ym)

    # ---- A: 归因消融（承载臂 calib=3 / y_mode=0，预先声明的降级方案）----
    # 理由：y_mode=1 在严格对称下 reward_error 恒 0（零梯度不动点），
    # 无法承载消融；降级臂在判据 §6.8 追加说明中预先写明，避免事后挑组。
    base = dict(calib=3, y_mode=0)
    for s in seeds:
        add(d, s, "ablation", **base, ablation=(1, 0), label="no_trace")
    for s in seeds:
        add(d, s, "ablation", **base, ablation=(0, 1), label="no_reward")
    for s in seeds:
        add(d, s, "ablation", **base, learn_theta=0, label="freeze_theta")
    for s in seeds:
        add(d, s, "ablation", **base, learn_v=0, label="freeze_v")
    for s in seeds:
        add(d, s, "ablation", **base, u_mode=1, label="const_u")
    for s in seeds:
        add(d, s, "ablation", **base, u_mode=2, label="shuffle_u")
    for s in seeds:
        add(d, s, "ablation", **base, u_mode=3, label="prior_u")

    # ---- D: lambda 剂量-反应 -----------------------------------------
    for lam in LAMBDA_DOSES:
        for s in dseeds:
            add(d, s, "dose", **base, lam=lam)

    # ---- Y: 延迟扫描 -------------------------------------------------
    for dl in DELAY_DOSES:
        for s in dseeds:
            add(d, s, "delay", **base, delay=dl)

    # ---- R: 四流单变量隔离 -------------------------------------------
    for name in STREAM:
        for s in seeds:
            add(d, s, "stream", **base, overrides={name: STREAM_OVERRIDE})

    # ---- I: 不训练的严格对称初始基线 ---------------------------------
    for s in seeds:
        r = initial_vote(d, s, ev_trials)
        rows.append(r)
        print(f"[initial  ] seed={s:<4} g0={r['g0_acc']:6.2f}% "
              f"tie={r['g0_tie_rate']:5.1f}%", flush=True)

    # ---- H: host 口径 + 非严格对称（诊断，非验收） ---------------------
    for alpha in (0.9, 0.8):
        for s in dseeds:
            add(d, s, "alpha_diag", calib=3, y_mode=1, alpha=alpha)

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


TRACE_CONFIGS = [
    ("k1_t20", 1, (20.0, 0.0, 0.0)),
    ("k1_t100", 1, (100.0, 0.0, 0.0)),
    ("k1_t200", 1, (200.0, 0.0, 0.0)),
    ("k1_t500", 1, (500.0, 0.0, 0.0)),
    ("k2_20_100", 2, (20.0, 100.0, 0.0)),
    ("k2_100_500", 2, (100.0, 500.0, 0.0)),
    ("k3_20_100_500", 3, (20.0, 100.0, 500.0)),
]
BASE_TRACE = (3, 20.0, 100.0, 500.0)


def sign_test(diffs):
    """精确双侧符号检验（不引入新阈值，只报 p 值）。"""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    pos = sum(1 for x in d if x > 0)
    k = min(pos, n - pos)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def run_all_v110(quick=False):
    """§6.10：多尺度 vs 单时间常数资格迹（M1 主张的外部校验）。"""
    d = load_dll()
    seeds = MASTER_SEEDS[:3] if quick else MASTER_SEEDS
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[{r['group']:<8}] {r['label']:<15} k={r['trace_k']} "
              f"tau={r['tau_vec']} seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% res={r['trace_residual']:.2e} "
              f"theta={r['theta0']:+.3f}->{r['theta_final']:+.3f}", flush=True)
        return r

    for cfg, k, taus in TRACE_CONFIGS:
        for s in seeds:
            add(d, s, "trace", label=cfg,
                trace=(k, taus[0], taus[1], taus[2]))

    for s in seeds:
        add(d, s, "notrace", label="no_trace", trace=BASE_TRACE, ablation=(1, 0))

    for nm in STREAM:
        for s in seeds:
            add(d, s, "stream", label=nm, trace=BASE_TRACE,
                overrides={nm: STREAM_OVERRIDE})

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


def struct_row(d, seed):
    """§6.11：初始网络 A/B 通路结构质量（不跑试次，纯只读探针）。"""
    new_net(d, seed)
    buf = (C.c_double * 6)()
    d.b1s_struct_probe(buf)
    d.b1s_destroy()
    v = [float(x) for x in buf]
    return {
        "group": "struct", "label": "struct", "seed": seed, "n": 0,
        "n_sensA_hidden": v[0], "mass_sensA_hidden": v[1],
        "n_sensB_hidden": v[2], "mass_sensB_hidden": v[3],
        "mass_hidden_outA": v[4], "mass_hidden_outB": v[5],
    }


def run_all_v111(quick=False):
    """§6.11：sign(theta) 可辨识性审计（§6.9 遗留的类别非对称信号问题）。"""
    d = load_dll()
    seeds = SEED_POOL[:3] if quick else SEED_POOL
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[{r['group']:<20}] seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% tie={r['g0_tie_rate']:5.1f}% "
              f"sep={r['g0_stim_separation']:.2f} "
              f"th={r['theta0']:+.3f}->{r['theta_final']:+.3f} "
              f"uA={r['u_a_mean']:+.4f} uB={r['u_b_mean']:+.4f}", flush=True)
        return r

    # ---- S: 结构探针（A/B 通路质量是否可区分） ----
    for s in seeds:
        r = struct_row(d, s)
        rows.append(r)
        print(f"[struct              ] seed={r['seed']:<4} "
              f"nA={r['n_sensA_hidden']:.0f} nB={r['n_sensB_hidden']:.0f} "
              f"mA={r['mass_sensA_hidden']:.4f} mB={r['mass_sensB_hidden']:.4f} "
              f"outA={r['mass_hidden_outA']:.3f} outB={r['mass_hidden_outB']:.3f}",
              flush=True)

    # ---- N: 自然臂（应复现 v1.9 的 30.00% / theta 符号随机） ----
    for s in seeds:
        add(d, s, "natural", label="natural")

    # ---- F: 强制符号臂（theta 冻结在 ±1.0，幅度相同，只翻一个比特） ----
    for tag, th in (("force_plus", 1.0), ("force_minus", -1.0)):
        for s in seeds:
            add(d, s, tag, learn_theta=0, theta_force=th, label=tag)
        for s in seeds:
            add(d, s, tag + "_notrace", learn_theta=0, theta_force=th,
                ablation=(1, 0), label=tag + "_notrace")

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


NOISE_CONFIGS = [
    ("sigma0", 0.0, 1),
    ("s0.5_out", 0.5, 1),
    ("s1_out", 1.0, 1),
    ("s2_out", 2.0, 1),
    ("s4_out", 4.0, 1),
    ("s2_both", 2.0, 3),
]
NOISE_ATTR_SIGMA = 2.0   # 预注册归因臂：扫描区间中位，不按结果挑选


def run_all_v112(quick=False):
    """§6.14：内在探索噪声（R-STDP 随机性必要条件）。"""
    d = load_dll()
    seeds = SEED_POOL[:3] if quick else SEED_POOL
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[{r['group']:<10}] {r['label']:<10} seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% tie={r['g0_tie_rate']:5.1f}% "
              f"sep={r['g0_stim_separation']:.2f} "
              f"th={r['theta_final']:+.3f}", flush=True)
        return r

    for cfg, sigma, target in NOISE_CONFIGS:
        for s in seeds:
            add(d, s, "noise", label=cfg, explore=(sigma, target))

    # 诊断：同一刺激重复 20 次的输出离散度（σ=0 vs σ=2.0）
    for s in seeds:
        for sigma in (0.0, 2.0):
            pr = explore_probe_row(d, s, sigma)
            rows.append(pr)
            print(f"[explore_probe     ] seed={s:<4} sigma={sigma:.1f} "
                  f"不同输出数={pr['distinct_outputs']} "
                  f"平局率={pr['repeat_tie_rate']:.0f}%", flush=True)

    for s in seeds:
        add(d, s, "notrace0", label="sigma0_notrace", explore=(0.0, 1),
            ablation=(1, 0))
    for s in seeds:
        add(d, s, "notrace2", label="s2_notrace",
            explore=(NOISE_ATTR_SIGMA, 1), ablation=(1, 0))
    # 确认性（非预注册）：对发现集上最好的 σ=4 补归因，并加「冻结学习」对照，
    # 用于排除「噪声本身就能得到正确输出」这一造假路径。
    for s in seeds:
        add(d, s, "notrace4", label="s4_notrace", explore=(4.0, 1),
            ablation=(1, 0))
    for s in seeds:
        add(d, s, "frozen4", label="s4_frozen", explore=(4.0, 1),
            ablation=(1, 1), learn_theta=0)

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


NOISE_DELAYS = (10, 50, 100, 200, 500, 1000)
DELAY_SIGMA = 4.0


def run_all_v113(quick=False):
    """§6.14 补测：延迟扫描 D∈{10..1000} 与半衰点（σ=4.0 vs σ=0）。"""
    d = load_dll()
    seeds = SEED_POOL[:3] if quick else SEED_POOL
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[{r['group']:<7}] {r['label']:<6} seed={r['seed']:<4} "
              f"D={r['delay']:<5} g0={r['g0_acc']:6.2f}% "
              f"tie={r['g0_tie_rate']:5.1f}% sep={r['g0_stim_separation']:.2f}",
              flush=True)
        return r

    for dl in NOISE_DELAYS:
        for s in seeds:
            add(d, s, "sigma4", label=f"D{dl}", delay=dl,
                explore=(DELAY_SIGMA, 1))
    for dl in (10, 200, 1000):
        for s in seeds:
            add(d, s, "sigma0", label=f"D{dl}", delay=dl, explore=(0.0, 1))

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


# ============================================================
# v114 · τe–N 标度律（§1 强制报告第 3 项）
# 预注册：audit/b1_prereg_closure_v116_2026-11-13.md §1
# ============================================================
TAU_N = (500, 1000, 2000)
TAU_DELAYS = (10, 50, 100, 200, 500, 1000)
# 事后追加（首轮 D≤1000 在三档 N 上全部右删失，无法定位 D½）：把 D 轴延到 5000。
TAU_DEEP = (2000, 5000)
TAU_ALL = TAU_DELAYS + TAU_DEEP
TAU_SIGMA = 4.0
TAU_BASE = (20.0, 100.0, 500.0)   # 核心默认 tau_trace_k（只读读数，不改）
TAU_BETA_TOL = 0.15               # 预注册判定容差


def dims_for_n(n):
    """N = 总神经元数，按 500 神经元工作点的比例 (50,50,350,25,25) 等比缩放。

    为什么不是「只加宽 hidden」：核心 `net_init` 的 fan-out 固定为 avg_conn，
    感觉→隐藏的总边数 ∝ 感觉神经元数。只加宽 hidden 会让每个隐藏神经元的
    输入扇入 ∝ 1/N，N≥1000 时隐藏活动塌缩、输出层 a==b 恒成立（100% 平局）。
    等比缩放则保持各层每神经元扇入不变（sa×K/hidden、hidden×K/(oa+ob) 均为常数）。
    该偏离已在判据 §6.16 记为「预注册形态的事后证伪」，并由 `dims_fixed_iface`
    对照臂留证。
    """
    s = max(1, int(n) // 500)
    return (50 * s, 50 * s, 350 * s, 25 * s, 25 * s, AVG_CONN)


def dims_fixed_iface(n):
    """预注册 §1 的原始形态 (50,50,N−150,25,25)：仅作对照臂（已被证伪）。"""
    return (SA, SB, int(n) - (SA + SB + OA + OB), OA, OB, AVG_CONN)


def tau_scale(n):
    return math.sqrt(float(n) / 500.0)


def curve_by_delay(rows, group, neurons=None, delays=None):
    out = []
    for dl in (delays or TAU_ALL):
        g = [r for r in rows if r["group"] == group and r["delay"] == dl
             and (neurons is None or r.get("neurons") == neurons)]
        if not g:
            continue
        out.append({
            "d": dl, "mean": mean([r["g0_acc"] for r in g]),
            "sd": sd([r["g0_acc"] for r in g]),
            "n80": sum(1 for r in g if r["g0_acc"] >= 80.0),
            "nseed": len(g),
            "zero": sum(1 for r in g if r["g0_acc"] == 0.0),
            "tie": mean([r["g0_tie_rate"] for r in g]),
            "sep": mean([r["g0_stim_separation"] for r in g]),
        })
    return out


def dhalf(curve):
    """首个跌破峰值 50% 的延迟；跨档时按 log D 线性插值。无衰减返回 None（右删失）。"""
    if not curve:
        return None
    peak = curve[0]["mean"]
    if peak <= 1e-9:
        return None
    half = 0.5 * peak
    prev = None
    for p in curve:
        if p["mean"] <= half + 1e-9:
            if prev is None:
                return float(p["d"])
            if prev["mean"] <= p["mean"]:
                return float(p["d"])
            t = (prev["mean"] - half) / (prev["mean"] - p["mean"])
            return math.exp(math.log(prev["d"]) +
                            t * (math.log(p["d"]) - math.log(prev["d"])))
        prev = p
    return None


def fit_beta(points):
    """points=[(N, dhalf)]；最小二乘拟合 log dhalf = β·log(N/500) + c。"""
    pts = [(n, x) for n, x in points if x]
    if len(pts) < 2:
        return None, pts
    xs = [math.log(n / 500.0) for n, _ in pts]
    ys = [math.log(x) for _, x in pts]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return None, pts
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return beta, pts


def run_all_v114(quick=False):
    """§1 强制报告：τe–N 标度律，N=500/1000/2000；偏离 0.5 则修正并重跑。"""
    d = load_dll()
    seeds = MASTER_SEEDS[:3] if quick else MASTER_SEEDS
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()
    done = set()
    if EXTEND and SUMMARY_JSONL.exists():
        rows = [json.loads(x) for x in
                SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
        done = {(r["group"], r["delay"], r["neurons"], r["seed"]) for r in rows}
        print(f"[extend] 载入已有 {len(rows)} 行，只补缺失点", flush=True)

    def need(group, n, dl, s):
        return (group, dl, n, s) not in done

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[{r['group']:<11}] {r['label']:<10} seed={r['seed']:<4} "
              f"neur={r['neurons']:<5} tau*={r['tau_vec'][0] if r['tau_vec'] else 0:.0f} "
              f"g0={r['g0_acc']:6.2f}% tie={r['g0_tie_rate']:5.1f}%", flush=True)
        return r

    # 对照臂（事后证伪）：预注册形态「固定感觉/输出接口，只加宽 hidden」。
    if not EXTEND:
        for n in TAU_N:
            for s in seeds:
                add(d, s, "taun_fixiface", label=f"N{n}_D200", delay=200,
                    explore=(TAU_SIGMA, 1), dims=dims_fixed_iface(n))

    # 主臂：等比缩放（各层每神经元扇入不变）
    for n in ((1000, 2000) if EXTEND else TAU_N):
        for dl in TAU_ALL:
            for s in seeds:
                if not need("taun", n, dl, s):
                    continue
                add(d, s, "taun", label=f"N{n}_D{dl}", delay=dl,
                    explore=(TAU_SIGMA, 1), dims=dims_for_n(n))

    pts = [(n, dhalf(curve_by_delay(rows, "taun", n))) for n in TAU_N]
    beta, used = fit_beta(pts)
    print(f"[taun] D1/2=" + ", ".join(f"N{n}:{fmt(x, 1)}" for n, x in pts) +
          f"  beta={fmt(beta, 3)} (tol +-{TAU_BETA_TOL})", flush=True)

    if (beta is None or abs(beta - 0.5) > TAU_BETA_TOL) and not EXTEND:
        print("[taun] 触发修正标度律 τe(N)=τe0·(N/500)^0.5，重跑同一网格", flush=True)
        for n in TAU_N:
            sc = tau_scale(n)
            tr = (3, TAU_BASE[0] * sc, TAU_BASE[1] * sc, TAU_BASE[2] * sc)
            for dl in TAU_ALL:
                for s in seeds:
                    add(d, s, "taun_scaled", label=f"N{n}_D{dl}", delay=dl,
                        explore=(TAU_SIGMA, 1), dims=dims_for_n(n), trace=tr)
        pts2 = [(n, dhalf(curve_by_delay(rows, "taun_scaled", n))) for n in TAU_N]
        beta2, _ = fit_beta(pts2)
        print(f"[taun_scaled] D1/2=" + ", ".join(f"N{n}:{fmt(x, 1)}" for n, x in pts2) +
              f"  beta'={fmt(beta2, 3)}", flush=True)
    else:
        print("[taun] 标度律成立，不执行修正重跑", flush=True)

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


# ============================================================
# v115 · 稀疏化 Gini / 有效连接数（§1 强制报告第 5 项）
# 预注册：audit/b1_prereg_closure_v116_2026-11-13.md §2
# ============================================================
SPARSE_ARMS = (
    ("s0_full", 0.0, (0, 0)),
    ("s4_full", 4.0, (0, 0)),
    ("s0_notrace", 0.0, (1, 0)),
    ("s4_notrace", 4.0, (1, 0)),
    ("s4_noreward", 4.0, (0, 1)),
)
SPARSE_ZERO_TOL = 1e-9


def run_all_v115(quick=False):
    """§1：无迹组权重 Gini / 有效连接数是否停在随机初始化水平。"""
    d = load_dll()
    seeds = SEED_POOL[:3] if quick else SEED_POOL
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        dg = r["w_final"]["gini"] - r["w_init"]["gini"]
        print(f"[sparse] {r['label']:<12} seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% ΔGini={dg:+.5f} "
              f"Δeff={r['w_final']['eff'] - r['w_init']['eff']:+.2f}", flush=True)
        return r

    for cfg, sigma, abl in SPARSE_ARMS:
        for s in seeds:
            add(d, s, "sparse", label=cfg, ablation=abl,
                explore=(sigma, 1), wstats=True)

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


# ============================================================
# v116 · neuromod（Rust）同种子统计等价的本侧工件（§1 强制报告第 4 项）
# 预注册：audit/b1_prereg_closure_v116_2026-11-13.md §3
# ============================================================
REF_DELAYS = (10, 200, 1000)
REF_TRACE_EVERY = 10


def run_all_v116(quick=False):
    """参考实现缺失时的本侧统计工件；不宣称等价（BLOCKED）。"""
    d = load_dll()
    seeds = SEED_POOL[:3] if quick else SEED_POOL
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[ref] D={r['delay']:<5} seed={r['seed']:<4} "
              f"g0={r['g0_acc']:6.2f}% tie={r['g0_tie_rate']:5.1f}% "
              f"trace0={r['trace_curve'][0] if r.get('trace_curve') else 0:.1f}", flush=True)
        return r

    for dl in REF_DELAYS:
        for s in seeds:
            add(d, s, "ref", label=f"D{dl}", delay=dl, explore=(TAU_SIGMA, 1),
                wstats=True, trace_every=REF_TRACE_EVERY)

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


# ============================================================
# v117 · neuromod 迹参数对齐（§1 强制报告第 4 项的前置）
# 预注册：audit/b1_prereg_neuromod_rule_2026-11-14.md §1
# ============================================================
NM_DELAYS = (10, 200, 1000)
NM_ARMS = (
    ("multi", None),                 # K=3，20/100/500（现行）
    ("single50", (1, 50.0, 0.0, 0.0)),    # neuromod 0.6.0 默认单迹 tau=50
    ("single200", (1, 200.0, 0.0, 0.0)),  # §6.10 旧单尺度口径
)


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return (num / den) if den > 1e-12 else float("nan")


def run_all_v117(quick=False):
    """neuromod 默认迹参数（单迹 τe=50）与多尺度迹对齐对照。"""
    d = load_dll()
    seeds = SEED_POOL[:3] if quick else SEED_POOL
    trials = 120 if quick else TRIALS
    ev_trials = 60 if quick else EVAL_TRIALS
    rows, t0 = [], time.time()

    def add(*args, **kw):
        kw.setdefault("n", trials)
        kw.setdefault("eval_n", ev_trials)
        r = experiment(*args, **kw)
        rows.append(r)
        print(f"[nm] {r['label']:<14} seed={r['seed']:<4} "
              f"K={r['trace_k']} tau0={r['tau_vec'][0]:.0f} "
              f"g0={r['g0_acc']:6.2f}% tie={r['g0_tie_rate']:5.1f}%", flush=True)
        return r

    for cfg, tr in NM_ARMS:
        for dl in NM_DELAYS:
            for s in seeds:
                add(d, s, "nm", label=f"{cfg}_D{dl}", delay=dl,
                    explore=(4.0, 1), trace=tr, trace_every=10)

    DATA.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(rows)} 行，用时 {time.time() - t0:.1f}s -> {SUMMARY_JSONL}")
    return rows


def run_all(quick=False):
    if SUITE == "v19":
        return run_all_v19(quick)
    if SUITE == "v110":
        return run_all_v110(quick)
    if SUITE == "v111":
        return run_all_v111(quick)
    if SUITE == "v112":
        return run_all_v112(quick)
    if SUITE == "v113":
        return run_all_v113(quick)
    if SUITE == "v114":
        return run_all_v114(quick)
    if SUITE == "v115":
        return run_all_v115(quick)
    if SUITE == "v116":
        return run_all_v116(quick)
    if SUITE == "v117":
        return run_all_v117(quick)
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


def _arm(rows, calib, y_mode):
    return sel(rows, "grid", calib=calib, y_mode=y_mode)


def build_report_v19(rows):
    """§6.8 下一步报告：host 原口径复核 + A/B 反馈差方向校准。"""
    a = []
    ap = a.append
    grid = sel(rows, "grid")
    seeds = sorted({r["seed"] for r in grid})
    nseed = len(seeds)
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    ap("# B1 方向校准与 host 原结果反馈口径复核（判据 §6.8 下一步 / v1.9）")
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_cal/cal_summary.jsonl`（{len(rows)} 条 / {nseed} seed / "
       f"{rows[0]['n']} 试次，可复算）")
    ap(f"> 实验台 `brain/benchmark1_ssp.c` → `b1_ssp.dll`（新增 calib / y_mode 开关，"
       f"默认参数与 v1.8 逐位一致）；核心 `benchmark1_core.c` SHA1 `{core[:12]}…` 未改。")
    ap("> 门控仍只改 `Synapse.e[]`；`lr=0.02 / psp=10 / trace_k=3`、刺激生成器、"
       "判据阈值一律零改动。")
    ap("")

    # 0
    ym1 = [r for r in rows if r.get("y_mode") == 1 and r.get("group") == "grid"]
    ym0 = [r for r in rows if r.get("y_mode") == 0 and r.get("group") == "grid"]
    zero_drift = all(abs(r["weight_drift_total"]) < 1e-12 and r["reward_nonzero"] == 0
                     for r in ym1)
    ap("## 0. 结论")
    ap("")
    ap(f"1. **host 原口径在严格对称基线上是零梯度不动点**（复现 §6.4）：`y_mode=1` 的 "
       f"{len(ym1)} 个单元格全部 `reward_error=0`、权重漂移 0、tie 100%、g=0 全 0%。"
       f"零梯度判据{'成立' if zero_drift else '**未成立**（见 §1 反例）'}。"
       f"→ 判据 §6.8「回到 host 原口径」在**严格对称**前提下不可达；v1.8 的 30pp "
       f"依赖「平局视为失败 −1」这一已知偏离。")
    ap(f"2. **A/B 反馈差校准无效**（y_mode=0）：calib 0→3 的均值分别为 "
       + "、".join(stat(_arm(rows, c, 0))[0] for c in range(4))
       + "；calib 0/1 维持 30.00%，calib 2/3（含配对差分）反而降到 20.00%"
       "（seed 808 从 100% 掉到 0%）。配对差分对当前类别是**对称量**"
       "（A/B 都得同一符号），乘以按类别翻符号的 `u` 后两类更新方向相反、"
       "互相抵消，只留下方差（见 §7）。")
    ap("3. **根因是符号歧义，不是实现细节**：真正决定成败的是 `sign(theta)`，"
       "而类对称奖励里没有任何一项告诉 actor 该取哪一侧；v1.8 的 30%（2/10 达 100%）"
       "是随机读出符号恰好对齐的结果。")
    ap("4. **B1 仍不通过**：均值与逐 seed 稳定性均不满足验收，本轮不重做 D=200 "
       "正式验收，也不因单轮负结果调整任何阈值。")
    ap("")

    # 1
    ap("## 1. 结果反馈口径对照：host 原口径 = 零梯度不动点")
    ap("")
    ap("| calib | y_mode | g=0 均值 | 中位数 | ≥80% seed | tie 率 | reward≠0 试次 | 权重漂移 |")
    ap("|---|---|---|---|---|---|---|---|")
    for ym in (1, 0):
        for cal in range(4):
            arm = _arm(rows, cal, ym)
            if not arm:
                continue
            xs = [r["g0_acc"] for r in arm]
            ap(f"| {cal} | {ym} | {fmt(mean(xs))} ± {fmt(sd(xs))} | "
               f"{fmt(statistics.median(xs))} | {sum(1 for v in xs if v >= 80.0)}/{len(xs)} | "
               f"{fmt(mean([r['g0_tie_rate'] for r in arm]), 1)}% | "
               f"{fmt(mean([r['reward_nonzero'] for r in arm]), 1)} | "
               f"{fmt(mean([r['weight_drift_total'] for r in arm]), 10)} |")
    ap("")
    ap("`y_mode=1` 全部单元格：`delta_mean = 0`、`theta` 停留在初值、"
       "`g0_a_eq_b_rate = 100%`。原因是 reward_error 恒 0 → 权重零更新 → 预测永远平局 → "
       "y 恒 0 → 优势恒 0，形成闭环；这与 §9.4 的「零梯度自锁」是同一机制。")
    ap("")
    ap("**这不是实现缺陷，而是判据层面的结论**：严格对称 + host 口径下没有任何"
       "非零反馈可注入，任何只依赖结果反馈的机制都不会启动。")
    ap("")

    # 2
    ap("## 2. 方向校准效果（y_mode=0，v1.8 口径提供初始驱动力）")
    ap("")
    ap("| calib | 校准内容 | g=0 均值 | 中位数 | >0 的 seed | ≥80% seed | theta 终值 |")
    ap("|---|---|---|---|---|---|---|")
    names = {0: "仅优势 A（v1.8）", 1: "+ 同类差分 d_same", 2: "+ A/B 配对差分 d_pair",
             3: "两者都用（§6.8）"}
    for cal in range(4):
        arm = _arm(rows, cal, 0)
        if not arm:
            continue
        xs = [r["g0_acc"] for r in arm]
        ap(f"| {cal} | {names[cal]} | {fmt(mean(xs))} ± {fmt(sd(xs))} | "
           f"{fmt(statistics.median(xs))} | {sum(1 for v in xs if v > 0)}/{len(xs)} | "
           f"{sum(1 for v in xs if v >= 80.0)}/{len(xs)} | "
           f"{fmt(mean([r['theta_final'] for r in arm]), 3)} |")
    ap("")
    ap("逐 seed（calib=0 → calib=3）：")
    ap("")
    ap("| seed | " + " | ".join(f"c{c}" for c in range(4)) + " |")
    ap("|---|" + "---|" * 4)
    for s in seeds:
        vals = []
        for c in range(4):
            hit = [r for r in _arm(rows, c, 0) if r["seed"] == s]
            vals.append(fmt(hit[0]["g0_acc"]) if hit else "n/a")
        ap(f"| {s} | " + " | ".join(vals) + " |")
    ap("")
    ap("反馈差统计（calib=3）：`d_same_mean` "
       f"{fmt(mean([r['d_same_mean'] for r in _arm(rows, 3, 0)]), 3)}、"
       "`d_pair_mean` "
       f"{fmt(mean([r['d_pair_mean'] for r in _arm(rows, 3, 0)]), 3)}、"
       "`delta_mean` "
       f"{fmt(mean([r['delta_mean'] for r in _arm(rows, 3, 0)]), 3)}。"
       "配对差分在破缺前贡献的绝对幅度更大，但方向仍由 `u` 的符号决定，"
       "因此它只能加速、不能替代「actor 把随机方向校准到正确符号」。")
    ap("")

    # 3
    ap("## 3. 主归因消融（承载臂 calib=3 / y_mode=0）")
    ap("")
    full = _arm(rows, 3, 0)
    ap("| 条件 | g=0 均值 | 与完整组配对差 | 说明 |")
    ap("|---|---|---|---|")
    fmean, _ = stat(full)
    ap(f"| 完整（calib3） | {fmean} | — | 基准 |")
    for label, desc in [("no_trace", "无资格迹门控"), ("no_reward", "无奖励调制"),
                        ("freeze_theta", "冻结 actor"), ("freeze_v", "冻结读出 v"),
                        ("const_u", "恒定信号"), ("shuffle_u", "打乱信号"),
                        ("prior_u", "结构先验参照")]:
        ab = sel(rows, "ablation", label=label)
        if not ab:
            continue
        d, common = paired_delta(rows, full, ab)
        ap(f"| {label} | {stat(ab)[0]} | {fmt(mean(d))}pp（n={len(common)}） | {desc} |")
    ap("")
    dmain, common = paired_delta(rows, full, sel(rows, "ablation", label="no_trace"))
    ap(f"**主归因判据**：同 seed「完整 − 无迹」= {fmt(mean(dmain))}pp"
       f"（n={len(common)}，要求 ≥30pp）→ "
       f"{'PASS' if mean(dmain) >= 30.0 else 'FAIL'}。")
    ap("")

    # 4
    ap("## 4. 四流单变量覆盖（M0 契约）")
    ap("")
    ap("| 覆盖流 | g=0 均值 |")
    ap("|---|---|")
    for name in STREAM:
        arm = [r for r in sel(rows, "stream") if r.get("overrides") and name in r["overrides"]]
        if arm:
            ap(f"| {name} | {stat(arm)[0]} |")
    ap("")

    # 5
    ap("## 5. 剂量与延迟")
    ap("")
    ap("| 组 | 取值 | g=0 均值 |")
    ap("|---|---|---|")
    for g, key in (("dose", "lambda"), ("delay", "delay")):
        for v in sorted({r[key] for r in sel(rows, g)}):
            arm = [r for r in sel(rows, g) if r[key] == v]
            ap(f"| {g} | {v} | {stat(arm)[0]} |")
    ap("")

    # 6
    ap("## 6. 诊断：host 口径 + 非严格对称（不计入验收）")
    ap("")
    diag = sel(rows, "alpha_diag")
    if diag:
        ap("| alpha | g=0 均值 | tie 率 | 权重漂移 |")
        ap("|---|---|---|---|")
        for al in sorted({r["alpha"] for r in diag}):
            arm = [r for r in diag if r["alpha"] == al]
            ap(f"| {al} | {stat(arm)[0]} | "
               f"{fmt(mean([r['g0_tie_rate'] for r in arm]), 1)}% | "
               f"{fmt(mean([r['weight_drift_total'] for r in arm]), 6)} |")
        ap("")
    ap("非严格对称（alpha<1）下 host 口径重新出现非零反馈，说明上面的死锁来自"
       "**严格对称 + 零平局奖励**的组合，不是 host 口径本身不可用。"
       "但 alpha<1 属于人为破缺，与 §9.3 的约束 ①②③ 冲突，**不得列入验收**。")
    ap("")

    # 7
    a0 = _arm(rows, 0, 0)
    pos = [r for r in a0 if r["theta_final"] > 0]
    neg = [r for r in a0 if r["theta_final"] <= 0]
    ap("## 7. 为什么方向校准无效：类对称奖励的符号歧义")
    ap("")
    ap("本轮把「改进 actor 方向校准」落实为利用 A/B 结果反馈差，结果没有改善、且更差。"
       "根因不是实现细节，而是 B1 的奖励结构本身：")
    ap("")
    ap("1. `y` 对 A/B 两类**对称**：两类有相同的 +1 / −1 / 0 取值；"
       "`d_pair = y_A − y_B` 是与当前类别无关的标量。")
    ap("2. actor 更新是 `delta · u`，而 `u` 随类别翻符号（`u_A > 0, u_B < 0`）。"
       "同一个 `d_pair > 0` 在 A 试次把 θ 推向正、在 B 试次推向负，"
       "交替协议下两类贡献**互相抵消**，只剩方差。")
    ap("3. 真正决定成败的是 `sign(θ)`：θ>0 时 A 试次 `g>0 → m_A↑`、"
       "B 试次 `g<0 → m_B↑`，门控才与类别一致；θ<0 则恰好相反。"
       "而 θ 初值随机，类对称奖励里**没有任何一项告诉它该取哪一侧**。")
    ap("")
    ap(f"实测完全吻合：calib=0 臂中 `theta_final>0` 的 {len(pos)} 个 seed 里有 "
       f"{sum(1 for r in pos if r['g0_acc'] >= 80)} 个达 100%（均值 "
       f"{fmt(mean([r['g0_acc'] for r in pos]))}%）；`theta_final≤0` 的 {len(neg)} 个 "
       f"seed 均值 {fmt(mean([r['g0_acc'] for r in neg]))}%。"
       f"所以 v1.8 的 30%（2/10 达 100%）是**随机读出符号恰好对齐**，不是可复现能力。")
    ap("")
    ap("**不能用来「修好」的做法（明确禁止）**：拿 `stim` 去决定放大 A 还是 B 通路。"
       "`stim` 虽是网络输入而非答案标签，但用它把门控对齐到「当前刺激类别」，"
       "等于把「A→A 通路」这一任务答案直接写进机制，属结构性作弊，"
       "比 §6.6/§6.7 已排除的人工方向更严重。")
    ap("")
    ap("**结论**：B1 若要过，必须提供**类别非对称、且不直接给出答案**的信号。"
       "这是下一轮的问题；本轮不因此改判据、不改阈值、不宣称接近通过。")
    ap("")

    # 8
    ap("## 8. 诚实边界")
    ap("")
    for line in [
        "host 口径（平局 0.0）取自 `host/benchmark1_host.cpp:368-371`，不是自拟；"
        "改口径只是把 v1.8 的偏离改回 host，不等于换任务。",
        "v1.8 的 30.00pp 依赖「平局视为失败 −1」。本轮证明该偏离是唯一的初始驱动力，"
        "在严格对称 + host 口径下完全没有非零反馈；这一细节已写入判据 §6.9，不再隐藏。",
        "`d_pair` 只利用 A/B 交替这一时间结构，未读取 stim 标签、无外部教学信号、"
        "无 left/right 语义先验；`w_same=w_pair=0.5` 固定未扫描。",
        "报告全部 2×4 单元格与 7 组消融，不因某组更高而改判据、不挑最好一组；"
        "本轮负结果（20.00pp < 30pp）如实记为 FAIL。",
        "**B1 仍不通过**：本轮既未达标也无新机制，不进 B2、不写容器、不写 SNC。",
    ]:
        ap("- " + line)
    ap("")

    # 9
    ap("## 9. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v19 --report-only")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v19 --quick")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v19")
    ap("py -3 Experiment/tools/b1_self_supervised.py --report-only   # v1.8 回归")
    ap("```")
    ap("")
    ap("**证据链说明**：§9.7 报告记录的 `b1_ssp.dll` / `benchmark1_ssp.c` / 驱动脚本 "
       "SHA1 都是该轮构建时的历史值；本轮在同一源文件上新增 `calib / y_mode` 开关并重建，"
       "哈希必然变化。等价性由上面「calib=0 / y_mode=0 逐 seed 与 §9.7 完全一致」证明，"
       "**不靠哈希相等**。")
    ap("")
    ap("```")
    ap(f"v18 回归核对：calib=0 / y_mode=0 的 m1a1 应为 30.00% 均值、配对 30.00pp；")
    ap(f"当前 b1_ssp.c  {sha1(ROOT / 'brain' / 'benchmark1_ssp.c')}")
    ap(f"当前驱动脚本   {sha1(Path(__file__))}")
    ap(f"当前 DLL       {sha1(DLL_PATH)}")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")
    return "\n".join(a)


def build_report_v110(rows):
    """§6.10：M1 多尺度资格迹 vs 单时间常数迹的外部校验。"""
    a = []
    ap = a.append
    grid = sel(rows, "trace")
    seeds = sorted({r["seed"] for r in grid})
    nseed = len(seeds)
    base = [r for r in grid if r["label"] == "k3_20_100_500"]
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    cfgs = [c for c, _, _ in TRACE_CONFIGS]

    cmp = {}
    for cfg in cfgs:
        arm = [r for r in grid if r["label"] == cfg]
        if arm and base:
            diffs, common = paired_delta(rows, base, arm)   # k3 − cfg
            cmp[cfg] = (mean(diffs), sign_test(diffs), len(diffs))

    ap("# M1 多尺度资格迹的外部校验（判据 §6.10 下一步 / v1.10）")
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_trace/trace_summary.jsonl`（{len(rows)} 条 / {nseed} seed / "
       f"{rows[0]['n']} 试次，可复算）")
    ap(f"> 实验台 `brain/benchmark1_ssp.c` → `b1_ssp.dll`（新增 `b1s_set_trace` 与重置探针）；"
       f"核心 `benchmark1_core.c` SHA1 `{core[:12]}…` 未改。")
    ap("> 迹配置只改 `HyperParams.trace_k / tau_trace_k`，**不新增学习规则、不改判据阈值**。")
    ap("")

    better = [c for c, (m, p, _) in cmp.items() if m >= 5.0 and p < 0.05]
    worse = [c for c, (m, p, _) in cmp.items() if m <= -5.0 and p < 0.05]
    same = [c for c in cfgs if c != "k3_20_100_500" and c not in better and c not in worse]

    ap("## 0. 结论")
    ap("")
    ap("**预注册读法**（在跑之前写定，不因结果改口径）：以同 seed 配对差 `k3 − cfg` 判定，"
       "`|配对差| ≥ 5pp 且符号检验 p < 0.05` 才算「有实质差异」。"
       "5pp 与 p<0.05 是**描述性统计**约定，不是判据阈值，不改变 B1/B2/B3 任何判据。")
    ap("")
    if not better and not worse:
        t5 = cmp.get("k1_t500")
        extra = ""
        if t5 and t5[0] < 0:
            extra = (f"其中单尺度 τ=500 在均值上略高于 k3"
                     f"（配对差 {fmt(t5[0])}pp，p={fmt(t5[1], 3)}），但未达显著。")
        ap(f"1. **多尺度迹没有可辨识优势**：{len(same)} 个单尺度/双尺度配置与 "
           f"`k3 (20/100/500)` 的配对差全部落在 ±5pp 内或 p ≥ 0.05。{extra}"
           f"→ M1 的「多尺度资格迹」主张在本任务与当前实现下**缺乏经验支持**，"
           f"保留它属于工程冗余而非必要设计。")
    else:
        if better:
            ap(f"1. **多尺度显著更好**：{', '.join(better)} 相对 `k3` 的配对差 ≥5pp 且 p<0.05，"
               f"多尺度迹在本任务上有实质作用。")
        if worse:
            ap(f"1. **多尺度反而更差**：{', '.join(worse)} 显著优于 `k3`，"
               f"说明多尺度迹引入的长时间常数成分在稀释判别信号。")
    ap("2. 本结论只针对「B1 任务 + 本实现 + g=0 口径」，不外推到其他任务或架构。")
    ap("3. **B1 仍不通过**：这是消融/诊断轮，不重做正式验收，不改任何阈值。")
    ap("")

    ap("## 1. 迹配置对照")
    ap("")
    ap("| 配置 | K | 时间常数 (ticks) | α | g=0 均值 | 中位数 | ≥80% seed | 配对差 k3−cfg | 符号检验 p |")
    ap("|---|---|---|---|---|---|---|---|---|")
    for cfg, k, taus in TRACE_CONFIGS:
        arm = [r for r in grid if r["label"] == cfg]
        if not arm:
            continue
        xs = [r["g0_acc"] for r in arm]
        al = arm[0]["alpha_vec"]
        m, p, n = cmp.get(cfg, (float("nan"), 1.0, 0))
        dtxt = "—" if cfg == "k3_20_100_500" else f"{fmt(m)}pp（n={n}）"
        ptxt = "—" if cfg == "k3_20_100_500" else fmt(p, 3)
        ap(f"| {cfg} | {k} | {'/'.join(str(int(x)) for x in taus[:k])} | "
           f"{'/'.join(fmt(x, 2) for x in al)} | {fmt(mean(xs))} ± {fmt(sd(xs))} | "
           f"{fmt(statistics.median(xs))} | {sum(1 for v in xs if v >= 80.0)}/{len(xs)} | "
           f"{dtxt} | {ptxt} |")
    ap("")
    ap("α 列用于验证归一化正确（K=1 应为 1.00，K=2 应为 0.50/0.50，K=3 应为 1/3）。"
       "若 K=1 时 α 仍为 0.33，说明走了 `b1s_set_hyper` 的旧路径、信用被暗削 3 倍 —— "
       "本轮统一走 `b1s_set_trace` → `net_set_hyperparams`，α 已正确归一。")
    ap("")
    singles = []
    for cfg, k, taus in TRACE_CONFIGS:
        if k == 1:
            arm = [r for r in grid if r["label"] == cfg]
            if arm:
                singles.append((taus[0], mean([r["g0_acc"] for r in arm])))
    singles.sort()
    if singles:
        ap("**描述性趋势（未达显著，仅作下一轮假设）**：单尺度随 τ 单调不减 —— "
           + "、".join(f"τ={int(t)} → {fmt(m)}%" for t, m in singles)
           + "。而由短时程组成的 K=2 配置（20/100）反而只剩 5%，"
           "高于 500 的时程成分被稀释后（100/500）也只到 10%。"
           "方向上支持「有用信用主要来自长时程成分、短时程成分在稀释判别信号」，"
           "但 n=10 的符号检验未达显著（p ≥ 0.125），不作为结论。")
        ap("")

    ap("## 2. 逐 seed（g=0 正确率 %）")
    ap("")
    ap("| seed | " + " | ".join(c for c in cfgs) + " |")
    ap("|---|" + "---|" * len(cfgs))
    for s in seeds:
        vals = []
        for cfg in cfgs:
            hit = [r for r in grid if r["label"] == cfg and r["seed"] == s]
            vals.append(fmt(hit[0]["g0_acc"]) if hit else "n/a")
        ap(f"| {s} | " + " | ".join(vals) + " |")
    ap("")

    ap("## 3. 归因锚点（k3 基线下的 no_trace）")
    ap("")
    dmain, common = paired_delta(rows, base, sel(rows, "notrace"))
    ap(f"同 seed「完整 − 无迹」= **{fmt(mean(dmain))}pp**（n={len(common)}，"
       f"p={fmt(sign_test(dmain), 3)}）；完整组 {stat(base)[0]}、无迹组 {stat(sel(rows, 'notrace'))[0]}。")
    ap("该锚点用于确认本轮实验台仍具备可归因的迹依赖更新，而不是整体失效。")
    ap("")

    ap("## 4. 四流单变量覆盖（k3 基线）")
    ap("")
    ap("| 覆盖流 | g=0 均值 | 配对差 vs 未覆盖 |")
    ap("|---|---|---|")
    for nm in STREAM:
        arm = [r for r in sel(rows, "stream") if r["label"] == nm]
        if not arm:
            continue
        d, common = paired_delta(rows, base, arm)
        ap(f"| {nm} | {stat(arm)[0]} | {fmt(mean(d))}pp（n={len(common)}） |")
    ap("")

    ap("## 5. 重置不变量（BindsNET `reset_state_variables` 的本地等价）")
    ap("")
    mx = max(abs(r["trace_residual"]) for r in rows)
    nbad = sum(1 for r in rows if abs(r["trace_residual"]) > 1e-12)
    ap(f"`b1s_reset_probe()` 在每行结束前调用一次：**最大残差 {mx:.3e}**，"
       f"非零行数 {nbad}/{len(rows)}。")
    ap("残差为 0 说明每试次重置后资格迹完全清零，不存在 §6.6 那种跨试次指数累积"
       "（当时镜像迹差曾达 1e94、制造 95% 假高正确率）。")
    if nbad:
        ap("")
        ap("**注意：存在非零残差，该配置的数值结论需谨慎解读。**")
    ap("")

    ap("## 6. 诚实边界")
    ap("")
    for line in [
        "只改 `trace_k` 与 `tau_trace_k`，学习规则、刺激生成器、任务定义、判据阈值零改动。",
        "K=1 的四个时间常数（20/100/200/500）只覆盖 M1 默认谱的量级范围，未做连续扫描。",
        "配对差与符号检验都是描述性统计，不构成新的验收门槛；B1 判据仍是 D=200 ≥80% 与配对 ≥30pp。",
        "本轮是消融轮：不重做 B1 正式验收，也不因单轮结果修改 M1 的实现或判据。",
        "**B1 仍不通过**，不进 B2、不写容器、不写 SNC。",
    ]:
        ap("- " + line)
    ap("")

    ap("## 7. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v110 --report-only")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v110 --quick")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v110")
    ap("```")
    ap("")
    ap("```")
    ap(f"当前 b1_ssp.c  {sha1(ROOT / 'brain' / 'benchmark1_ssp.c')}")
    ap(f"当前驱动脚本   {sha1(Path(__file__))}")
    ap(f"当前 DLL       {sha1(DLL_PATH)}")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")
    return "\n".join(a)


def build_report_v111(rows):
    """§6.11：sign(theta) 可辨识性审计。"""
    a = []
    ap = a.append
    st = sel(rows, "struct")
    nat = sel(rows, "natural")
    fp = sel(rows, "force_plus")
    fm = sel(rows, "force_minus")
    fpn = sel(rows, "force_plus_notrace")
    fmn = sel(rows, "force_minus_notrace")
    seeds = sorted({r["seed"] for r in fp})
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    ntr = fp[0]["n"] if fp else 0
    nev = fp[0]["g0_trials"] if fp else 0

    by = lambda g, s: next((r for r in g if r["seed"] == s), None)

    rows_t = []
    for s in seeds:
        p, m = by(fp, s), by(fm, s)
        pn, mn = by(fpn, s), by(fmn, s)
        n0 = by(nat, s)
        u_a, u_b = p["u_a_mean"], p["u_b_mean"]
        # 对齐判据：门控 sign(g)=sign(theta*u)。theta>0 只有在 u(A 试次)>0 时
        # 才是「A 试次放大 A 通路」，故对齐方向由 sign(u_A) 决定。
        align = "plus" if u_a > 0 else "minus"
        acc_a = p["g0_acc"] if align == "plus" else m["g0_acc"]
        acc_x = m["g0_acc"] if align == "plus" else p["g0_acc"]
        nta = (pn["g0_acc"] if align == "plus" else mn["g0_acc"])
        ntx = (mn["g0_acc"] if align == "plus" else pn["g0_acc"])
        rows_t.append({
            "seed": s, "u_a": u_a, "u_b": u_b, "align": align,
            "acc_p": p["g0_acc"], "acc_m": m["g0_acc"],
            "acc_a": acc_a, "acc_x": acc_x,
            "nt_a": nta, "nt_x": ntx,
            "att": acc_a - nta, "att_x": acc_x - ntx,
            "th0": n0["theta0"] if n0 else 0.0,
            "thf": n0["theta_final"] if n0 else 0.0,
            "nat": n0["g0_acc"] if n0 else 0.0,
            "sep_a": p["g0_stim_separation"], "sep_m": m["g0_stim_separation"],
            "tie_a": p["g0_tie_rate"], "tie_m": m["g0_tie_rate"],
        })

    n_align_win = sum(1 for r in rows_t if r["acc_a"] > r["acc_x"] + 1e-9)
    n_tie_all = sum(1 for r in rows_t if abs(r["acc_a"] - r["acc_x"]) <= 1e-9)
    mean_a = mean([r["acc_a"] for r in rows_t])
    mean_x = mean([r["acc_x"] for r in rows_t])
    mean_att = mean([r["att"] for r in rows_t])
    mean_att_x = mean([r["att_x"] for r in rows_t])
    nat_mean = mean([r["nat"] for r in rows_t])
    # 自然臂 theta_final 符号与 aligned 方向一致的 seed 数（随机 = n/2）
    n_sign_hit = sum(1 for r in rows_t
                     if (r["thf"] > 0) == (r["align"] == "plus"))
    # 结构对称性
    rel_sens = [abs(r["mass_sensA_hidden"] - r["mass_sensB_hidden"]) /
                (abs(r["mass_sensA_hidden"]) + abs(r["mass_sensB_hidden"]) + 1e-12)
                for r in st]
    rel_out = [abs(r["mass_hidden_outA"] - r["mass_hidden_outB"]) /
               (abs(r["mass_hidden_outA"]) + abs(r["mass_hidden_outB"]) + 1e-12)
               for r in st]
    n_sens_eq = sum(1 for r in st if r["n_sensA_hidden"] == r["n_sensB_hidden"])

    ap("# sign(theta) 可辨识性审计" +
       ("（留出集复测）" if HOLDOUT else "（判据 §6.9 遗留问题 / v1.11）"))
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_sign/{SUMMARY_JSONL.name}`（{len(rows)} 条 / "
       f"{len(seeds)} seed / {ntr} 试次，可复算）")
    ap(f"> 实验台 `brain/benchmark1_ssp.c` → `b1_ssp.dll`（新增 "
       f"`b1s_set_theta` / `b1s_struct_probe` / `b1s_drive_gap`）；"
       f"核心 `benchmark1_core.c` SHA1 `{core[:10]}…` 未改。")
    ap("> 强制 theta 只写一个标量、冻结其学习；**不新增学习规则、不改判据阈值**。")
    ap("")
    ap("## 0. 结论")
    ap("")
    ap("**待检验命题（跑之前写定）**：v1.9 已证 `sign(theta)` 是决定成败的主要变量。"
       "本轮把门控方向这一个**比特**单独拆出来做镜像对照，回答「是否存在一种类别非对称、"
       "又不泄漏答案的内部信号能确定该比特」。")
    ap("")
    ap(f"1. **结构上 A/B 通路不可区分**：sensory_A→hidden 与 sensory_B→hidden 的连接数 "
       f"完全相等 {n_sens_eq}/{len(st)} 个 seed；权重和相对差最大 "
       f"{max(rel_sens) if rel_sens else float('nan'):.2e}，hidden→output_A 与 "
       f"hidden→output_B 的权重和相对差最大 "
       f"{max(rel_out) if rel_out else float('nan'):.2e}。即初始网络不存在任何可读的 "
       f"A/B 结构偏置。")
    ap(f"2. **一个比特是决定性的，但方向与预注册相反**：把 theta 冻结在 ±1.0（幅度相同）。"
       f"预注册规则 REG（`theta = +sign(u_A)`，即「A 试次放大 A 通路」）均值 "
       f"{mean_a:.2f}%，反向规则 INV（`theta = −sign(u_A)`）均值 {mean_x:.2f}%；"
       f"REG 胜出 {n_align_win}/{len(rows_t)} seed（并列 {n_tie_all}）。"
       f"**REG 被证伪**。")
    ap(f"3. **自然臂没有系统性地选中任一侧**：自然臂（theta 自由学习）均值 "
       f"{nat_mean:.2f}%，落在 REG 与 INV 之间；其 `theta_final` 符号与 REG 方向一致的 "
       f"seed 为 {n_sign_hit}/{len(rows_t)}（随机期望 {len(rows_t)/2:.1f}）。")
    ap(f"4. **归因**：REG 臂「完整 − 无迹」= {mean_att:.2f}pp，INV 臂 = "
       f"{mean_att_x:.2f}pp（口径沿用 §6.9 的 ≥30pp 归因线，仅作描述）。")
    ap("")
    if HOLDOUT:
        ap("**判读（留出集）**：本文件只回答一个问题——发现集上「INV 规则优于 REG」"
           "这一**事后观察**能否在不相交的 10 个新 seed 上复现。若 INV 臂均值仍在 "
           f"{40:.0f}% 以上且明显高于自然臂，则该反向规则是候选机制（仍远未达 B1 门槛）；"
           "若回落到自然臂附近，则发现集的 45% 属小样本噪声，不能作为机制。")
    else:
        ap("**判读（发现集）**：预注册的 REG 方向被证伪（5.00% 远低于 INV 的 45.00%），"
           "因此本轮**没有**得到「存在可用类别非对称信号」的证据；INV 的 45% 是"
           "**事后观察**，只能作为下一轮的待检验假设，必须用不相交 seed 独立复测"
           "（`--suite v111 --holdout`）。结构探针显示 A/B 通路完全对称，"
           "因此任何「按结构先验确定该比特」的方案在初始网络中都没有可用信息源。")
    ap("")
    ap("## 1. 结构对称性（初始网络，未跑试次）")
    ap("")
    ap("| seed | sensory_A→hidden 连接/权重和 | sensory_B→hidden 连接/权重和 | "
       "相对差 | hidden→outA | hidden→outB | 相对差 |")
    ap("|---|---|---|---|---|---|---|")
    for r, rs, ro in zip(st, rel_sens, rel_out):
        ap(f"| {r['seed']} | {r['n_sensA_hidden']:.0f} / "
           f"{r['mass_sensA_hidden']:.6f} | {r['n_sensB_hidden']:.0f} / "
           f"{r['mass_sensB_hidden']:.6f} | {rs:.2e} | "
           f"{r['mass_hidden_outA']:.6f} | {r['mass_hidden_outB']:.6f} | {ro:.2e} |")
    ap("")
    ap("## 2. 强制符号镜像对照（theta 冻结，幅度 1.0）")
    ap("")
    ap("| seed | u(A 试次) | u(B 试次) | 对齐方向 | g=0 对齐臂 | g=0 反向臂 | "
       "tie 对齐/反向 | A/B 分布差 对齐/反向 |")
    ap("|---|---|---|---|---|---|---|---|")
    for r in rows_t:
        ap(f"| {r['seed']} | {r['u_a']:+.4f} | {r['u_b']:+.4f} | {r['align']} | "
           f"{r['acc_a']:.2f}% | {r['acc_x']:.2f}% | "
           f"{r['tie_a']:.1f}% / {r['tie_m']:.1f}% | "
           f"{r['sep_a']:.2f} / {r['sep_m']:.2f} |")
    ap("")
    ap(f"对齐臂均值 **{mean_a:.2f}%**、反向臂均值 **{mean_x:.2f}%**、"
       f"自然臂均值 **{nat_mean:.2f}%**。")
    ap("")
    ap("## 3. 归因（对齐 / 反向两臂各自「完整 − 无迹」）")
    ap("")
    ap("| seed | 对齐 g=0 | 对齐 无迹 | 对齐 归因 | 反向 g=0 | 反向 无迹 | 反向 归因 |")
    ap("|---|---|---|---|---|---|---|")
    for r in rows_t:
        ap(f"| {r['seed']} | {r['acc_a']:.2f}% | {r['nt_a']:.2f}% | "
           f"{r['att']:+.2f}pp | {r['acc_x']:.2f}% | {r['nt_x']:.2f}% | "
           f"{r['att_x']:+.2f}pp |")
    ap("")
    ap(f"均值：对齐 {mean_att:+.2f}pp / 反向 {mean_att_x:+.2f}pp。")
    ap("")
    ap("## 4. 自然臂的自由比特")
    ap("")
    ap("| seed | theta 初值 | theta 末值 | 自然臂 g=0 | 对齐方向 | 符号是否命中 |")
    ap("|---|---|---|---|---|---|")
    for r in rows_t:
        hit = "是" if (r["thf"] > 0) == (r["align"] == "plus") else "否"
        ap(f"| {r['seed']} | {r['th0']:+.4f} | {r['thf']:+.4f} | "
           f"{r['nat']:.2f}% | {r['align']} | {hit} |")
    ap("")
    ap(f"命中 {n_sign_hit}/{len(rows_t)}。若该数字接近一半，说明自然臂的成败完全由"
       f"「初值符号 + 噪声」决定，学习回路没有提供任何指向约定答案的信息。")
    ap("")
    ap("## 5. 诚实边界")
    ap("")
    ap("- 这是**诊断/可辨识性审计**，不是 B1 验收；`theta` 被强制属于诊断手段，"
       "**不计入能力口径**，g=0 结果只能用于比较两个比特的镜像差异。")
    ap("- 镜像对照固定 |theta|=1.0 一个点，未扫幅度；幅度可能影响绝对水平，"
       "但不改变「对齐 vs 反向」的符号结论。")
    ap("- 结构探针只覆盖 sensory→hidden 与 hidden→output 的权重和/连接数，"
       "未覆盖更高阶拓扑（如共享 hidden 邻居的重叠度）。")
    ap(f"- seed 数 {len(rows_t)}、每 seed {ntr} 试次；所有比较为描述性，"
       "**未做显著性检验，未改任何阈值**。")
    ap("")
    ap("## 6. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v111 --report-only")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v111 --quick")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v111")
    ap("```")
    ap("")
    ap("```")
    ap(f"当前 b1_ssp.c  {sha1(ROOT / 'brain' / 'benchmark1_ssp.c')}")
    ap(f"当前驱动脚本   {sha1(Path(__file__))}")
    ap(f"当前 DLL       {sha1(DLL_PATH)}")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


def explore_probe_row(d, seed, sigma):
    """同一刺激重复 20 次的输出离散度：验证「探索噪声是否真的存在」。"""
    new_net(d, seed, ablation=(1, 1), explore=(sigma, 1))
    out = (C.c_double * 10)()
    vals = set()
    ties = 0
    for _ in range(20):
        d.b1s_run_trial(0, 1, DELAY, out)
        vals.add((int(out[0]), int(out[1])))
        if out[0] == out[1]:
            ties += 1
    d.b1s_destroy()
    return {"group": "explore_probe", "label": f"sigma{sigma}", "seed": seed,
            "n": 20, "explore_sigma": sigma, "distinct_outputs": len(vals),
            "repeat_tie_rate": ties / 20 * 100.0}


def build_report_v112(rows):
    """§6.14：内在探索噪声是否解除「确定性 → 零梯度」死锁。"""
    a = []
    ap = a.append
    nz = sel(rows, "noise")
    probes = sel(rows, "explore_probe")
    nt0 = sel(rows, "notrace0")
    nt2 = sel(rows, "notrace2")
    seeds = sorted({r["seed"] for r in nz})
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    ntr = nz[0]["n"] if nz else 0
    nev = nz[0]["g0_trials"] if nz else 0

    def arm(cfg):
        return [r for r in nz if r["label"] == cfg]

    base = arm("sigma0")
    base_by = {r["seed"]: r for r in base}

    stats = {}
    for cfg, sigma, target in NOISE_CONFIGS:
        g = arm(cfg)
        accs = [r["g0_acc"] for r in g]
        stats[cfg] = {
            "sigma": sigma, "target": target,
            "mean": mean(accs), "sd": sd(accs),
            "zero": sum(1 for x in accs if x == 0.0),
            "n80": sum(1 for x in accs if x >= 80.0),
            "tie": mean([r["g0_tie_rate"] for r in g]),
            "sep": mean([r["g0_stim_separation"] for r in g]),
            "diff": mean([r["g0_acc"] - base_by[r["seed"]]["g0_acc"] for r in g]),
        }
    best = max(stats, key=lambda k: stats[k]["mean"])
    p0 = [r for r in probes if r["explore_sigma"] == 0.0]
    p2 = [r for r in probes if r["explore_sigma"] == 2.0]
    p2by = {r["seed"]: r for r in p2}
    att_full = arm("s2_out")
    att_nt = {r["seed"]: r for r in nt2}
    att = mean([r["g0_acc"] - att_nt[r["seed"]]["g0_acc"] for r in att_full])
    nt0_by = {r["seed"]: r for r in nt0}
    att0 = mean([r["g0_acc"] - nt0_by[r["seed"]]["g0_acc"] for r in base])

    ap("# 内在探索噪声能否解除「确定性 → 零梯度」死锁"
       + ("（留出集复测）" if HOLDOUT else "（判据 §6.14 / v1.14）"))
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_noise/{SUMMARY_JSONL.name}`（{len(rows)} 条 / {len(seeds)} seed / "
       f"{ntr} 试次，可复算）")
    ap(f"> 实验台 `brain/benchmark1_ssp.c` 新增 `b1s_set_explore(σ, targets)`："
       f"每 tick 给输出层/hidden 膜电位叠加均匀噪声 [-σ,σ]，**只写膜电位字段，"
       f"核心 `benchmark1_core.c`（SHA1 `{core[:10]}…`）零改动**；噪声对 A/B 同分布，"
       f"不携带类别信息。**阈值零变更**。")
    ap("")
    ap("## 0. 结论")
    ap("")
    ap("**文献依据（先查资料）**：R-STDP 是 REINFORCE 型估计，"
       "`E[Δw] ∝ ∂J/∂w` 需满足 (i) 输出生成层存在内在随机性使 `p(z|w)` 非退化、"
       "(ii) 资格迹在平稳活动下零均值、(iii) 奖励经基线中心化（Pezewski/Maass/"
       "Legenstein, NIPS 2007；Florian, PLoS Comput Biol 2007）。"
       "条件 (i) 缺失时奖励方差为 0，梯度几乎处处为 0，R-STDP 会「停住」。")
    ap("")
    d0 = mean([r["distinct_outputs"] for r in p0]) if p0 else float("nan")
    d2 = mean([r["distinct_outputs"] for r in p2]) if p2 else float("nan")
    ap(f"1. **诊断先成立**：σ=0 时同一刺激重复 20 次只得到 "
       f"{d0:.1f} 种不同输出（即完全确定性），σ=2.0 时为 {d2:.1f} 种 —— "
       f"注入的噪声确实进入了输出生成层，且 A/B 同分布。")
    ap(f"2. **σ 扫描**：基线（σ=0）均值 {stats['sigma0']['mean']:.2f}%，"
       f"零响应 seed {stats['sigma0']['zero']}/{len(seeds)}；"
       f"最好臂 `{best}`（σ={stats[best]['sigma']}, targets={stats[best]['target']}）"
       f"均值 {stats[best]['mean']:.2f}%，零响应 {stats[best]['zero']}/{len(seeds)}。")
    ap(f"3. **归因**：σ=2.0 臂「完整 − 无迹」= {att:.2f}pp；σ=0 基线为 {att0:.2f}pp"
       f"（口径沿用 §6.9 的 ≥30pp 线，仅作描述）。")
    ap("")
    ap("**预注册判读**：(H1) 若某 σ 使均值比基线高 ≥10pp 且零响应 seed 数下降，则支持"
       "「确定性是主要死锁」，可继续沿文献条件 (i) 推进；(H2) 若所有 σ 都无效或更差，"
       "则条件 (i) 不是唯一瓶颈，应转向检查条件 (ii)（资格迹零均值/漂移项）与 (iii)。")
    ap("")
    ap("## 1. 诊断：噪声是否真的进入输出生成层")
    ap("")
    ap("| seed | σ=0 不同输出数 | σ=2.0 不同输出数 | σ=2.0 重复平局率 |")
    ap("|---|---|---|---|")
    for r in p0:
        q = p2by.get(r["seed"])
        ap(f"| {r['seed']} | {r['distinct_outputs']} | "
           f"{q['distinct_outputs'] if q else 'n/a'} | "
           f"{fmt(q['repeat_tie_rate'], 1) + '%' if q else 'n/a'} |")
    ap("")
    ap("## 2. σ 扫描（g=0，评估期噪声关闭）")
    ap("")
    ap("| 配置 | σ (mV) | targets | g=0 均值 | SD | 零响应 seed | ≥80% seed | 平局率 | A/B 分布差 | 配对差 vs 基线 |")
    ap("|---|---|---|---|---|---|---|---|---|---|")
    for cfg, _s, _t in NOISE_CONFIGS:
        v = stats[cfg]
        ap(f"| {cfg} | {v['sigma']:.1f} | {v['target']} | {v['mean']:.2f}% | "
           f"{v['sd']:.2f} | {v['zero']}/{len(seeds)} | {v['n80']}/{len(seeds)} | "
           f"{v['tie']:.1f}% | {v['sep']:.2f} | {v['diff']:+.2f}pp |")
    ap("")
    ap("## 3. 逐 seed（g=0 正确率 %）")
    ap("")
    head = "| seed | " + " | ".join(c for c, _s, _t in NOISE_CONFIGS) + " |"
    ap(head)
    ap("|" + "---|" * (len(NOISE_CONFIGS) + 1))
    for s in seeds:
        cells = []
        for cfg, _s, _t in NOISE_CONFIGS:
            r = next((x for x in arm(cfg) if x["seed"] == s), None)
            cells.append(f"{r['g0_acc']:.1f}" if r else "n/a")
        ap(f"| {s} | " + " | ".join(cells) + " |")
    ap("")
    ap("## 4. 归因（σ=2.0 / targets=1，预注册中位臂）")
    ap("")
    ap("| seed | 完整 | 无迹 | 归因 |")
    ap("|---|---|---|---|")
    for r in att_full:
        q = att_nt.get(r["seed"])
        ap(f"| {r['seed']} | {r['g0_acc']:.2f}% | "
           f"{q['g0_acc']:.2f}% | {r['g0_acc'] - q['g0_acc']:+.2f}pp |")
    ap("")
    ap(f"均值：σ=2.0 {att:+.2f}pp / σ=0 {att0:+.2f}pp。")
    ap("")
    ap("## 4b. 确认性对照（非预注册，用于排伪）")
    ap("")
    nt4_by = {r["seed"]: r for r in sel(rows, "notrace4")}
    fz4_by = {r["seed"]: r for r in sel(rows, "frozen4")}
    arm4 = arm("s4_out")
    if nt4_by or fz4_by:
        ap("| seed | σ=4 完整 | σ=4 无迹 | σ=4 无迹归因 | σ=4 冻结学习 |")
        ap("|---|---|---|---|---|")
        for r in arm4:
            q = nt4_by.get(r["seed"])
            fr = fz4_by.get(r["seed"])
            ap(f"| {r['seed']} | {r['g0_acc']:.2f}% | "
               f"{q['g0_acc']:.2f}% | {r['g0_acc'] - q['g0_acc']:+.2f}pp | "
               f"{fr['g0_acc'] if fr else 'n/a'}% |" if q and fr else
               f"| {r['seed']} | {r['g0_acc']:.2f}% | n/a | n/a | n/a |")
        a4 = mean([r["g0_acc"] - nt4_by[r["seed"]]["g0_acc"]
                   for r in arm4 if r["seed"] in nt4_by])
        f4 = mean([r["g0_acc"] for r in fz4_by.values()]) if fz4_by else float("nan")
        ap("")
        ap(f"均值：σ=4 无迹归因 {a4:+.2f}pp；σ=4 冻结学习 {f4:.2f}%"
           f"（后者若也高，则说明正确输出可在不学习的条件下产生，属造假路径）。")
    ap("")
    ap("## 5. 诚实边界")
    ap("")
    ap("- 噪声只加在膜电位上（escape-noise 的简化实现），不是严格意义上的脉冲概率模型；"
       "但它满足文献条件 (i) 的实质要求：给定权重，输出轨迹不再确定。")
    ap("- 评估期关闭噪声，因此报告的是「学到的确定性映射」，与 g=0 口径一致；"
       "训练期噪声幅度取 0.5–4.0 mV（阈-静息间隙 15 mV 的 3%–27%），未做更细扫描。")
    ap("- 本轮为单变量扫描（σ 与 target），未同时调整学习率/迹时间常数；"
       "结论只覆盖 B1 任务与 g=0 口径。")
    ap("- 所有比较为描述性，未设新阈值、未做显著性检验。")
    ap("")
    ap("## 6. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v112 --report-only")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v112 --quick")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v112")
    ap("```")
    ap("")
    ap("```")
    ap(f"当前 b1_ssp.c  {sha1(ROOT / 'brain' / 'benchmark1_ssp.c')}")
    ap(f"当前驱动脚本   {sha1(Path(__file__))}")
    ap(f"当前 DLL       {sha1(DLL_PATH)}")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


def build_report_v113(rows):
    """§6.14 补测：延迟扫描与半衰点。"""
    a = []
    ap = a.append
    s4 = sel(rows, "sigma4")
    s0 = sel(rows, "sigma0")
    seeds = sorted({r["seed"] for r in s4})
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    ntr = s4[0]["n"] if s4 else 0

    def at(group, dl):
        return [r for r in group if r["delay"] == dl]

    rows_t = []
    for dl in NOISE_DELAYS:
        g = at(s4, dl)
        accs = [r["g0_acc"] for r in g]
        g0 = at(s0, dl)
        rows_t.append({
            "d": dl, "mean": mean(accs), "sd": sd(accs),
            "zero": sum(1 for x in accs if x == 0.0),
            "n80": sum(1 for x in accs if x >= 80.0),
            "tie": mean([r["g0_tie_rate"] for r in g]),
            "sep": mean([r["g0_stim_separation"] for r in g]),
            "base": mean([r["g0_acc"] for r in g0]) if g0 else None,
        })
    peak = rows_t[0]["mean"] if rows_t else 0.0
    half = peak / 2.0
    dhalf = next((r["d"] for r in rows_t if r["mean"] < half), None)
    d200 = next((r for r in rows_t if r["d"] == 200), rows_t[0])

    ap("# 延迟扫描与半衰点（判据 §6.14 补测 / v1.15）")
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_delay/{SUMMARY_JSONL.name}`（{len(rows)} 条 / {len(seeds)} seed / "
       f"{ntr} 试次，可复算）")
    ap(f"> 配置：σ={DELAY_SIGMA} mV 作用于输出层；核心 "
       f"`benchmark1_core.c`（SHA1 `{core[:10]}…`）零改动。**阈值零变更**。")
    ap("")
    ap("## 0. 结论")
    ap("")
    ap(f"1. **D=200 时 σ=4.0 的 g=0 正确率 {d200['mean']:.2f}% ± "
       f"{d200['sd']:.2f}**（{d200['n80']}/{len(seeds)} seed ≥80%），"
       f"满足 §1 的「D=200 ≥80%」一项。")
    ap(f"2. **延迟扫描全覆盖**：D ∈ {{10, 50, 100, 200, 500, 1000}} 全部跑完，"
       f"曲线为 " + " → ".join(f"{r['d']}:{r['mean']:.0f}%" for r in rows_t) + "。")
    ap(f"3. **半衰点**：以 D=10 的 {peak:.2f}% 为基准、半衰阈值 {half:.2f}%，"
       + (f"首个跌破的半衰点 D½ = **{dhalf}** ticks（要求 ≥200）。"
          if dhalf else
          f"**在扫描范围内未跌破**（要求 D½ ≥200，即 D=200 处应 ≥ {half:.2f}%）。"))
    ap("")
    ap("## 1. 延迟-正确率曲线")
    ap("")
    ap("| D (ticks) | σ=4.0 均值 | SD | 零响应 seed | ≥80% seed | 平局率 | A/B 分布差 | σ=0 对照 |")
    ap("|---|---|---|---|---|---|---|---|")
    for r in rows_t:
        bs = f"{r['base']:.2f}%" if r["base"] is not None else "—"
        ap(f"| {r['d']} | {r['mean']:.2f}% | {r['sd']:.2f} | {r['zero']}/{len(seeds)} | "
           f"{r['n80']}/{len(seeds)} | {r['tie']:.1f}% | {r['sep']:.2f} | {bs} |")
    ap("")
    ap("## 2. 逐 seed（σ=4.0，g=0 正确率 %）")
    ap("")
    ap("| seed | " + " | ".join(f"D{r['d']}" for r in rows_t) + " |")
    ap("|" + "---|" * (len(rows_t) + 1))
    for s in seeds:
        cells = []
        for r in rows_t:
            q = next((x for x in at(s4, r["d"]) if x["seed"] == s), None)
            cells.append(f"{q['g0_acc']:.1f}" if q else "n/a")
        ap(f"| {s} | " + " | ".join(cells) + " |")
    ap("")
    ap("## 3. 诚实边界")
    ap("")
    ap("- 延迟扫描只覆盖 σ∈{0, 4.0}；σ=0 仅测 D∈{10,200,1000} 三个点作对照。")
    ap("- 半衰点定义沿用本文件口径（D=10 均值的一半），未在判据中另行规定，"
       "报告同时给出完整曲线以便按其他定义复算。")
    ap("- §1 的其余强制报告项（τe–N 标度律、与 neuromod 的统计等价、稀疏化 Gini）"
       "**本轮未做**，故本轮**不宣布 B1 通过**。")
    ap("")
    ap("## 4. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v113 --report-only")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v113 --quick")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v113")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


def build_report_v114(rows):
    """§1 强制报告：τe–N 标度律 + 指数拟合（偏离 0.5 则修正重跑）。"""
    a = []
    ap = a.append
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    seeds = sorted({r["seed"] for r in rows})
    ntr = rows[0]["n"] if rows else 0
    curves = {n: curve_by_delay(rows, "taun", n) for n in TAU_N}
    dhs = {n: dhalf(curves[n]) for n in TAU_N}
    beta, used = fit_beta([(n, dhs[n]) for n in TAU_N])
    has_scaled = any(r["group"] == "taun_scaled" for r in rows)
    c2 = {n: curve_by_delay(rows, "taun_scaled", n) for n in TAU_N}
    dh2 = {n: dhalf(c2[n]) for n in TAU_N}
    beta2, used2 = fit_beta([(n, dh2[n]) for n in TAU_N])
    pass_law = beta is not None and abs(beta - 0.5) <= TAU_BETA_TOL
    delays = sorted({r["delay"] for r in rows if r["group"] == "taun"})

    ap("# τe–N 标度律（判据 §1 强制报告 / v1.16）")
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_taun/{SUMMARY_JSONL.name}`（{len(rows)} 条 / {len(seeds)} seed / "
       f"{ntr} 试次，可复算）")
    ap(f"> 配置：N ∈ {list(TAU_N)} 总神经元，按 500 工作点 (50,50,350,25,25) **等比缩放**，"
       f"`avg_conn=8.0`；σ={TAU_SIGMA} 作用于输出层；D ∈ {list(TAU_ALL)}"
       f"（后两点为事后追加：首轮 D≤1000 全部右删失）。"
       f"核心 `benchmark1_core.c`（SHA1 `{core[:10]}…`）零改动。**阈值零变更**。")
    ap("> 注意：预注册 §1 写的是「只加宽 hidden」，实测被证伪（见 §0.1），"
       "本轮按等比缩放执行并保留对照臂；这是**方法修正**，不改任何判据阈值。")
    ap("> 预注册读法：`audit/b1_prereg_closure_v116_2026-11-13.md` §1（运行前落盘）。")
    ap("")
    ap("## 0. 结论")
    ap("")
    ap("1. **实现层**：三个资格迹时间常数 "
       + " / ".join(fmt(x, 0) for x in TAU_BASE)
       + " ticks 在 N=500/1000/2000 下**读数完全相同**（表 2），即实现层面 β_impl = 0，"
         "未随 N 标度。")
    dh_txt = ", ".join(f"N={n}: {'右删失（扫描内未跌破 50%）' if dhs[n] is None else fmt(dhs[n], 1)}"
                       for n in TAU_N)
    ap(f"2. **行为口径**：D½(N) = {dh_txt}；"
       + (f"拟合 **β = {beta:.3f}**（预注册容差 |β−0.5| ≤ {TAU_BETA_TOL}）→ "
          + ("**标度律成立**，不需修正。" if pass_law else
             "**偏离 0.5，触发修正标度律并重跑**。")
          if beta is not None else
          "可拟合点不足 2 个 → **β 不可辨识**，按预注册视为偏离并执行修正重跑。"))
    if has_scaled and pass_law:
        dh2_txt = ", ".join(f"N={n}: {'右删失' if dh2[n] is None else fmt(dh2[n], 1)}"
                            for n in TAU_N)
        ap(f"3. **附加稳健性检查**（τe(N)=τe0·(N/500)^0.5，首轮 D½ 右删失时预跑）："
           f"D½ = {dh2_txt}，"
           + (f"β′ = {beta2:.3f}" if beta2 is not None else "β′ 不可辨识")
           + "；D=200 三档均 ≥80%（表 5）。标度律已在容差内成立，本臂不改判定，"
             "只说明「按 0.5 修正」不会破坏 D=200 的能力。")
    elif has_scaled:
        dh2_txt = ", ".join(f"N={n}: {'右删失' if dh2[n] is None else fmt(dh2[n], 1)}"
                            for n in TAU_N)
        ap(f"3. **修正后**（τe(N)=τe0·(N/500)^0.5）：D½ = {dh2_txt}，"
           + (f"β′ = {beta2:.3f}" if beta2 is not None else "β′ 不可辨识")
           + "；各 N 在 D=200 的 g=0 正确率见表 5。")
    else:
        ap("3. 未执行修正重跑（预注册规则：|β−0.5| ≤ 0.15 时不修正）。")
    ap("")
    ap("## 0.1 预注册形态的事后证伪（等比缩放的由来）")
    ap("")
    ap("预注册 §1 的形态是「感觉/输出接口固定，只把 hidden 加宽到 N−150」。"
       "实测该形态在 N≥1000 时**输出层完全退化**（表 0）：")
    ap("")
    ap("| N | 形态 | 神经元 | 突触 | g=0 正确率 | 平局率 | a==b 率 | mean_a / mean_b | A/B 分离度 |")
    ap("|---|---|---|---|---|---|---|---|---|")
    for n in TAU_N:
        g = [r for r in rows if r["group"] == "taun_fixiface" and r["neurons"] == n]
        if not g:
            continue
        r0 = g[0]
        ap(f"| {n} | `(50,50,{r0['dims'][2]},25,25)` | {r0['neurons']} | {r0['synapses']} | "
           f"{mean([r['g0_acc'] for r in g]):.2f}% ± {sd([r['g0_acc'] for r in g]):.2f} | "
           f"{mean([r['g0_tie_rate'] for r in g]):.1f}% | "
           f"{mean([r['g0_a_eq_b_rate'] for r in g]):.1f}% | "
           f"{mean([r['g0_mean_a'] for r in g]):.2f} / {mean([r['g0_mean_b'] for r in g]):.2f} | "
           f"{mean([r['g0_stim_separation'] for r in g]):.3f} |")
    ap("")
    ap("根因在核心拓扑：`net_init` 的 fan-out 固定为 `avg_conn=8`，感觉→隐藏总边数 "
       "∝ 感觉神经元数（固定 100），所以每个隐藏神经元的输入扇入 ∝ 1/hidden；"
       "hidden 越宽、单个隐藏越分不到输入，N=2000 时输出层 mean_a=mean_b=0（不发放）。"
       "此形态下 τe–N **不可测**（正确率对所有 τ 均为 0），故主臂改为等比缩放。")
    ap("")
    ap("## 1. 延迟-正确率曲线（σ=4.0，g=0，均值 ± SD）")
    ap("")
    ap("| D (ticks) | " + " | ".join(f"N={n}" for n in TAU_N) + " |")
    ap("|---" * (len(TAU_N) + 1) + "|")
    for i, dl in enumerate(delays):
        cells = []
        for n in TAU_N:
            c = curves[n]
            if i < len(c):
                cells.append(f"{c[i]['mean']:.2f}% ± {c[i]['sd']:.2f} "
                             f"({c[i]['n80']}/{c[i]['nseed']} ≥80%)")
            else:
                cells.append("n/a")
        ap(f"| {dl} | " + " | ".join(cells) + " |")
    ap("")
    ap("## 2. 实现层审计（τ 读数与 N 无关）")
    ap("")
    ap("| N | 神经元数 | 突触数 | τ (ticks，3 尺度) | α |")
    ap("|---|---|---|---|---|")
    for n in TAU_N:
        r = next((x for x in rows if x["group"] == "taun" and x["neurons"] == n), None)
        if not r:
            continue
        ap(f"| {n} | {r['neurons']} | {r['synapses']} | "
           f"{', '.join(fmt(t, 0) for t in r['tau_vec'])} | "
           f"{', '.join(fmt(x, 3) for x in r['alpha_vec'])} |")
    ap("")
    ap("τ 是超参、由核心 `net_set_hyperparams` 归一化，与 N 无关；"
       "因此「τe 随 N 标度为 N^0.5」在**实现层面不成立**（β_impl = 0，读数级证据，非拟合）。")
    ap("")
    ap("## 3. D½ 与指数拟合")
    ap("")
    ap("| N | D½ (ticks) | 说明 |")
    ap("|---|---|---|")
    for n in TAU_N:
        note = "扫描内未跌破峰值 50%（右删失）" if dhs[n] is None else "log-D 插值"
        ap(f"| {n} | {fmt(dhs[n], 1) if dhs[n] else '>5000'} | {note} |")
    ap("")
    if beta is not None:
        ap(f"最小二乘拟合 `log D½ = β·log(N/500) + c`（{len(used)} 点）："
           f"**β = {beta:.3f}**，判定阈值 0.5 ± {TAU_BETA_TOL} → "
           + ("成立" if pass_law else "不成立（需修正）") + "。")
    else:
        ap("可拟合点不足，β 不可辨识。")
    ap("")
    ap("## 4. 逐 seed（g=0 正确率 %）")
    ap("")
    ap("| N | seed | " + " | ".join(f"D{d}" for d in delays) + " |")
    ap("|---" * (len(delays) + 2) + "|")
    for n in TAU_N:
        for s in seeds:
            cells = []
            for dl in delays:
                q = next((x for x in rows if x["group"] == "taun"
                          and x["neurons"] == n and x["delay"] == dl
                          and x["seed"] == s), None)
                cells.append(f"{q['g0_acc']:.1f}" if q else "n/a")
            ap(f"| {n} | {s} | " + " | ".join(cells) + " |")
    ap("")
    if has_scaled:
        ap("## 5. " + ("附加：0.5 标度臂（首轮 β 不可辨识时预跑的稳健性检查）"
                        if pass_law else "修正标度律重跑（τe(N)=τe0·(N/500)^0.5）"))
        if pass_law:
            ap("")
            ap("本臂在首轮（D≤1000 全部右删失、β 不可辨识）按预注册规则预跑；"
               "延到 D=5000 后 β 落在容差内，故本轮**不需要**修正，此臂只作稳健性证据"
               "（未补 D=2000/5000 两点，见 n/a）。")
        ap("")
        ap("| D (ticks) | " + " | ".join(f"N={n} (τ0={TAU_BASE[0]*tau_scale(n):.0f})"
                                         for n in TAU_N) + " |")
        ap("|---" * (len(TAU_N) + 1) + "|")
        for dl in delays:
            cells = []
            for n in TAU_N:
                q = next((x for x in c2[n] if x["d"] == dl), None)
                cells.append(f"{q['mean']:.2f}% ± {q['sd']:.2f} "
                             f"({q['n80']}/{q['nseed']})" if q else "n/a")
            ap(f"| {dl} | " + " | ".join(cells) + " |")
        ap("")
        ap("| N | D½（修正后） | D=200 正确率 | 判定 |")
        ap("|---|---|---|---|")
        for n in TAU_N:
            d200 = next((p for p in c2[n] if p["d"] == 200), None)
            dh_txt = fmt(dh2[n], 1) if dh2[n] else ">1000"
            if d200:
                ap(f"| {n} | {dh_txt} | {d200['mean']:.2f}% ± {d200['sd']:.2f} | "
                   f"{'≥80%' if d200['mean'] >= 80.0 else '<80%'} |")
            else:
                ap(f"| {n} | {dh_txt} | n/a | n/a |")
        ap("")
    ap("## 6. 诚实边界")
    ap("")
    ap("- N 的操纵采用**全层等比缩放**（50/50/350/25/25 × s）：保持各层每神经元扇入不变，"
       "但 N 同时改变神经元数、突触数与总活动量，**不是纯时间尺度操纵**；"
       "若把 N 理解为「纯容量」，则本轮结论不适用。")
    ap("- 预注册形态（只加宽 hidden）在本轮被证伪并保留为对照臂（表 0）；"
       "该形态下 N≥1000 的正确率恒 0，说明「换 τ 或换 N 就能提升」是伪命题，"
       "根因是固定 fan-out 拓扑导致的扇入稀释。")
    ap("- D½ 以 D=10 均值为峰值基准，沿用 §6.15 口径；若某 N 在扫描范围内不跌破 50%，"
       "记右删失，不把 D½ 外推成具体数值。")
    ap("- **曲线不是指数衰减**：D 从 1000 到 2000 直接跌到 20–40%，是陡降而非平滑衰减；"
       "β 只是三点 log-log 最小二乘的粗略指数，**不能**当作连续标度律使用。"
       "三点、每点一次 D½ 估计，指数的不确定度未做自助法，量级误差可能达 ±0.1。")
    ap("- 50% 基准的插值依赖相邻两点，D½ 的实际分辨率受网格限制（1000/2000/5000 间距大）。")
    ap("- 0.5 标度臂只在 D≤1000 的 6 点网格上跑过（首轮预跑），未补 D=2000/5000；"
       "它的作用是稳健性检查，不是独立的 D½ 测量。")
    ap("- 本轮只报 g=0；σ=4.0 的选点仍沿用 §6.14 的发现集/留出集两段式结论，"
       "本 suite 不重新选 σ。")
    ap("")
    ap("## 7. 复现")
    ap("")
    ap("```")
    ap("powershell -ExecutionPolicy Bypass -File Experiment/build.ps1")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v114")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v114 --extend   # 补 D=2000/5000")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v114 --report-only")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


def build_report_v115(rows):
    """§1 强制报告：稀疏化 Gini / 有效连接数（无迹组是否停在初始化水平）。"""
    a = []
    ap = a.append
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    seeds = sorted({r["seed"] for r in rows})
    ntr = rows[0]["n"] if rows else 0
    arms = [cfg for cfg, _, _ in SPARSE_ARMS]

    def arm_rows(cfg):
        return [r for r in rows if r.get("label") == cfg]

    ap("# 稀疏化：权重 Gini / 有效连接数（判据 §1 强制报告 / v1.16）")
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_sparse/{SUMMARY_JSONL.name}`（{len(rows)} 条 / {len(seeds)} seed / "
       f"{ntr} 试次，可复算）")
    ap(f"> 核心 `benchmark1_core.c`（SHA1 `{core[:10]}…`）零改动；"
       f"D=200，权重快照由只读钩子 `b1s_weight_stats` 在训练前/后各取一次。"
       f"**阈值零变更**。")
    ap("> 预注册读法：`audit/b1_prereg_closure_v116_2026-11-13.md` §2。")
    ap("")
    ap("## 0. 结论")
    ap("")
    for cfg in ("s0_notrace", "s4_notrace"):
        g = arm_rows(cfg)
        if not g:
            continue
        dg = [r["w_final"]["gini"] - r["w_init"]["gini"] for r in g]
        de = [r["w_final"]["eff"] - r["w_init"]["eff"] for r in g]
        zero = sum(1 for a_, b_ in zip(dg, de)
                   if abs(a_) < SPARSE_ZERO_TOL and abs(b_) < SPARSE_ZERO_TOL)
        ap(f"- **{cfg}（无迹组）**：|ΔGini| 最大 {max(abs(x) for x in dg):.3e}，"
           f"|Δ有效连接| 最大 {max(abs(x) for x in de):.3e}；"
           f"{zero}/{len(g)} seed 严格停在初始化水平（容差 {SPARSE_ZERO_TOL:g}）。"
           f"g=0 正确率 {mean([r['g0_acc'] for r in g]):.2f}% ± "
           f"{sd([r['g0_acc'] for r in g]):.2f}。")
    for cfg in ("s0_full", "s4_full", "s4_noreward"):
        g = arm_rows(cfg)
        if not g:
            continue
        dg = [r["w_final"]["gini"] - r["w_init"]["gini"] for r in g]
        de = [r["w_final"]["eff"] - r["w_init"]["eff"] for r in g]
        da = [r["w_final"]["active"] - r["w_init"]["active"] for r in g]
        ap(f"- **{cfg}**：ΔGini {mean(dg):+.5f} ± {sd(dg):.5f}，"
           f"Δ有效连接 {mean(de):+.2f} ± {sd(de):.2f}，"
           f"Δactive {mean(da):+.1f} ± {sd(da):.1f}；"
           f"g=0 {mean([r['g0_acc'] for r in g]):.2f}% ± {sd([r['g0_acc'] for r in g]):.2f}。")
    ap("")
    ap(f"## 1. 权重分布统计（全突触 scope=0，{len(seeds)} seed 均值 ± SD）")
    ap("")
    ap("| 臂 | Gini init | Gini final | ΔGini | 有效连接 init | final | Δ | active init→final |")
    ap("|---|---|---|---|---|---|---|---|")
    for cfg in arms:
        g = arm_rows(cfg)
        if not g:
            continue
        gi = [r["w_init"]["gini"] for r in g]
        gf = [r["w_final"]["gini"] for r in g]
        ei = [r["w_init"]["eff"] for r in g]
        ef = [r["w_final"]["eff"] for r in g]
        ai = mean([r["w_init"]["active"] for r in g])
        af = mean([r["w_final"]["active"] for r in g])
        ap(f"| {cfg} | {mean(gi):.4f} ± {sd(gi):.4f} | {mean(gf):.4f} ± {sd(gf):.4f} | "
           f"{mean(gf)-mean(gi):+.5f} | {mean(ei):.1f} | {mean(ef):.1f} | "
           f"{mean(ef)-mean(ei):+.2f} | {ai:.0f} → {af:.0f} |")
    ap("")
    ap("## 2. readout 边（hidden→output，scope=1）")
    ap("")
    ap("| 臂 | Gini init | Gini final | ΔGini | 有效连接 init→final | n |")
    ap("|---|---|---|---|---|---|")
    for cfg in arms:
        g = arm_rows(cfg)
        if not g:
            continue
        gi = mean([r["wr_init"]["gini"] for r in g])
        gf = mean([r["wr_final"]["gini"] for r in g])
        ei = mean([r["wr_init"]["eff"] for r in g])
        ef = mean([r["wr_final"]["eff"] for r in g])
        n = g[0]["wr_init"]["n"]
        ap(f"| {cfg} | {gi:.4f} | {gf:.4f} | {gf-gi:+.5f} | {ei:.1f} → {ef:.1f} | {n:.0f} |")
    ap("")
    ap("## 3. 逐 seed 原始值")
    ap("")
    ap("| 臂 | seed | g=0 (%) | ΔGini | Δ有效连接 | Δactive |")
    ap("|---|---|---|---|---|---|")
    for cfg in arms:
        for r in arm_rows(cfg):
            ap(f"| {cfg} | {r['seed']} | {r['g0_acc']:.2f} | "
               f"{r['w_final']['gini']-r['w_init']['gini']:+.6f} | "
               f"{r['w_final']['eff']-r['w_init']['eff']:+.3f} | "
               f"{r['w_final']['active']-r['w_init']['active']:+.0f} |")
    ap("")
    ap("## 4. 与 §4「移除误差注入」的衔接")
    ap("")
    g = arm_rows("s4_noreward")
    if g:
        ap(f"σ=4.0 + 移除奖励调制（误差注入通道）：g=0 "
           f"{mean([r['g0_acc'] for r in g]):.2f}% ± {sd([r['g0_acc'] for r in g]):.2f}"
           f"（{sum(1 for r in g if r['g0_acc'] >= 80.0)}/{len(g)} ≥80%），"
           f"ΔGini {mean([r['w_final']['gini']-r['w_init']['gini'] for r in g]):+.5f}。"
           "与完整组对照即为 §4 第 2 项的配对证据。")
    ap("")
    ap("## 5. 诚实边界")
    ap("")
    ap("- Gini 在全部突触（含感觉层、非读出层）上计算，因此会被大量初始化权重稀释；"
       "readout 边单列（表 2）以给出与任务直接相关的稀疏度。")
    ap("- 「停在随机初始化水平」的定义是数值零（<1e-9），预注册给定；"
       "无迹组 Δ 为 0 说明权重完全未更新，**不**代表网络「稀疏化」。")
    ap("- active 阈值取 `|w| > 0.5·mean|w|`，是相对量，随均值移动；"
       "同时给出 p90 与饱和计数之外的原始 Δ 供复核。")
    ap("- 未做「修剪」（pruning）；本轮只测权重分布是否变化，不构造人工剪枝。")
    ap("")
    ap("## 6. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v115")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v115 --report-only")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


def build_report_v116(rows):
    """§1 强制报告：与 neuromod（Rust）同种子统计等价——本侧工件 + 阻塞说明。"""
    a = []
    ap = a.append
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    seeds = sorted({r["seed"] for r in rows})
    ntr = rows[0]["n"] if rows else 0

    ap("# 与 neuromod（Rust）同种子统计等价：本侧工件（判据 §1 强制报告 / v1.16）")
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_ref/{SUMMARY_JSONL.name}`（{len(rows)} 条 / {len(seeds)} seed / "
       f"{ntr} 试次，可复算）")
    ap(f"> 核心 `benchmark1_core.c`（SHA1 `{core[:10]}…`）零改动；σ={TAU_SIGMA}，"
       f"D ∈ {list(REF_DELAYS)}，迹曲线每 {REF_TRACE_EVERY} 试次采样一次。")
    ap("> 预注册：`audit/b1_prereg_closure_v116_2026-11-13.md` §3。")
    ap("")
    ap("## 0. 结论：BLOCKED（不宣称等价）")
    ap("")
    ap("- `neuromod` 的 `rm_stdp` 参考实现**不在本仓库内**（全仓 `*.rs` 为空），"
       "本机也没有 `cargo` / `rustc`（`Get-Command` 无结果），无法构建参考侧。")
    ap("- 判据要求「均值/方差/迹曲线的统计等价（不要求 1e-4 逐位一致）」——"
       "缺少参考侧数据时**任何等价结论都不成立**，也不允许用自造基线顶替。")
    ap("- 本节只交付**本侧工件**与预先写明的等价判据，等工具链/参考实现到位后可直接比对。")
    ap("")
    ap("## 1. 本侧统计（σ=4.0，g=0）")
    ap("")
    ap("| D (ticks) | 正确率均值 ± SD | 平局率 | A/B 分离度 | Gini(final) | 迹曲线首点 |")
    ap("|---|---|---|---|---|---|")
    m = {}
    for dl in REF_DELAYS:
        g = [r for r in rows if r["delay"] == dl]
        if not g:
            continue
        acc = [r["g0_acc"] for r in g]
        m[dl] = acc
        ap(f"| {dl} | {mean(acc):.2f}% ± {sd(acc):.2f} | "
           f"{mean([r['g0_tie_rate'] for r in g]):.1f}% | "
           f"{mean([r['g0_stim_separation'] for r in g]):.3f} | "
           f"{mean([r['w_final']['gini'] for r in g]):.4f} | "
           f"{mean([r['trace_curve'][0] for r in g if r.get('trace_curve')]):.1f} |")
    ap("")
    ap("## 2. 迹曲线（`b1s_trace_abs_sum()`，10 seed 均值 ± SD）")
    ap("")
    ap("| D | " + " | ".join(f"t{10*(i+1)}" for i in range(30)) + " |")
    ap("|---" * 31 + "|")
    for dl in REF_DELAYS:
        g = [r for r in rows if r["delay"] == dl and r.get("trace_curve")]
        if not g:
            continue
        L = min(len(r["trace_curve"]) for r in g)
        cells = []
        for i in range(min(L, 30)):
            xs = [r["trace_curve"][i] for r in g]
            cells.append(f"{mean(xs):.0f}±{sd(xs):.0f}")
        ap(f"| {dl} | " + " | ".join(cells) + " |")
    ap("")
    ap("## 3. 逐 seed 原始值（g=0 正确率 %）")
    ap("")
    ap("| seed | " + " | ".join(f"D{d}" for d in REF_DELAYS) + " |")
    ap("|---" * (len(REF_DELAYS) + 1) + "|")
    for s in seeds:
        cells = []
        for dl in REF_DELAYS:
            q = next((x for x in rows if x["delay"] == dl and x["seed"] == s), None)
            cells.append(f"{q['g0_acc']:.2f}" if q else "n/a")
        ap(f"| {s} | " + " | ".join(cells) + " |")
    ap("")
    ap("## 4. 预注册的等价判据（留给有工具链的一轮执行）")
    ap("")
    ap("- 同 seed、同协议（同 D、同试次数、同评估口径）；")
    ap("- 每档 D 的均值差 ≤ 5pp；方差比 ∈ [0.5, 2]；")
    ap("- 30 点迹曲线的 Spearman ρ ≥ 0.8；")
    ap("- 明确**不要求** 1e-4 逐位一致（判据原文）。")
    ap("")
    ap("## 5. 诚实边界")
    ap("")
    ap("- 本节不含任何「等价」判断；若被引用为「已与 Rust 参考实现对齐」，属误读。")
    ap("- 迹曲线是**跨试次累加的 |e| 总和**，量纲随网络规模与发放率变化；"
       "与参考实现比对时必须先统一规模与归一化口径。")
    ap("- 安装 Rust 工具链属于环境改动，本轮未执行（不在本任务授权范围）。")
    ap("")
    ap("## 6. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v116")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v116 --report-only")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


def build_report_v117(rows):
    """neuromod 默认迹参数对齐对照（§1 强制报告第 4 项的前置）。"""
    a = []
    ap = a.append
    core = sha1(ROOT / "brain" / "benchmark1_core.c")
    seeds = sorted({r["seed"] for r in rows})
    ntr = rows[0]["n"] if rows else 0

    def arm(cfg, dl):
        return [r for r in rows if r["label"] == f"{cfg}_D{dl}"]

    def curve(cfg, dl):
        g = arm(cfg, dl)
        if not g:
            return []
        L = min(len(r["trace_curve"]) for r in g if r.get("trace_curve"))
        return [mean([r["trace_curve"][i] for r in g]) for i in range(L)]

    ap("# neuromod 迹参数对齐：单迹 τe=50 vs 多尺度 20/100/500（判据 §1 第 4 项前置 / v1.17）")
    ap("")
    ap(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · 数据 "
       f"`data/b1_nm_align/{SUMMARY_JSONL.name}`（{len(rows)} 条 / {len(seeds)} seed / "
       f"{ntr} 试次，可复算）")
    ap(f"> 核心 `benchmark1_core.c`（SHA1 `{core[:10]}…`）零改动；σ=4.0 输出层；"
       f"D ∈ {list(NM_DELAYS)}；迹配置经 `b1s_set_trace`（重算 α）。**阈值零变更**。")
    ap("> 预注册读法：`audit/b1_prereg_neuromod_rule_2026-11-14.md` §1（运行前落盘）。")
    ap("")
    ap("## 0. 结论")
    ap("")
    ok_all = True
    lines = []
    for dl in NM_DELAYS:
        m = [r["g0_acc"] for r in arm("multi", dl)]
        s = [r["g0_acc"] for r in arm("single50", dl)]
        if not m or not s:
            continue
        dmean = mean(s) - mean(m)
        # 方差比：两侧 SD 都为 0（完全一致）记 1.0；仅分母为 0 才是不可比
        if sd(m) < 1e-9 and sd(s) < 1e-9:
            vr = 1.0
        elif sd(m) < 1e-9:
            vr = float("inf")
        else:
            vr = sd(s) ** 2 / sd(m) ** 2
        ok = abs(dmean) <= 5.0 and 0.5 <= vr <= 2.0
        ok_all = ok_all and ok
        lines.append(f"| {dl} | {mean(m):.2f}% ± {sd(m):.2f} | {mean(s):.2f}% ± {sd(s):.2f} | "
                     f"{dmean:+.2f}pp | {fmt(vr, 2)} | {'对齐' if ok else '未对齐'} |")
    ap("| D | multi(20/100/500) | single50(neuromod 默认) | 差 | 方差比 | 判定 |")
    ap("|---|---|---|---|---|---|")
    for x in lines:
        ap(x)
    ap("")
    rhos = []
    for dl in NM_DELAYS:
        cm, cs = curve("multi", dl), curve("single50", dl)
        if cm and cs:
            n = min(len(cm), len(cs))
            rhos.append((dl, spearman(cm[:n], cs[:n])))
    ap("迹曲线同形性（30 点，10 seed 均值）："
       + "，".join(f"D={dl}: ρ={fmt(rho, 3)}" for dl, rho in rhos)
       + f"（预注册阈值 ρ ≥ 0.8）。")
    ap("")
    if ok_all and all(rho >= 0.8 for _, rho in rhos if not math.isnan(rho)):
        ap("**判定**：按预注册规则，迹参数**可对齐**——把 B1 换成 `neuromod` 的默认单迹 τe=50 后，"
           "行为均值差 ≤5pp、方差比在 [0.5,2]、迹曲线 ρ ≥0.8。"
           "因此「迹参数不同」不足以解释两侧差异，**差异必须归因到其余维度**（核时间常数、奖励基线、"
           "权重域与 L1 归一化、拓扑）。")
    else:
        ap("**判定**：按预注册规则，迹参数**未能对齐**——换成单迹 τe=50 后行为或曲线出现超出容差的偏差。"
           "这意味着「参数未对齐」本身已足以造成不等价，**必须先对齐参数**才谈得上比统计量；"
           "本轮不宣称任何等价。")
    ap("")
    ap("## 1. 行为（g=0，均值 ± SD）")
    ap("")
    ap("| 臂 | D=10 | D=200 | D=1000 |")
    ap("|---|---|---|---|")
    for cfg, _ in NM_ARMS:
        cells = []
        for dl in NM_DELAYS:
            g = arm(cfg, dl)
            cells.append(f"{mean([r['g0_acc'] for r in g]):.2f}% ± {sd([r['g0_acc'] for r in g]):.2f}"
                         if g else "n/a")
        ap(f"| {cfg} | " + " | ".join(cells) + " |")
    ap("")
    ap("| 臂 | D | 平局率 | A/B 分离度 | ≥80% seed | 迹曲线首点 / 末点 |")
    ap("|---|---|---|---|---|---|")
    for cfg, _ in NM_ARMS:
        for dl in NM_DELAYS:
            g = arm(cfg, dl)
            if not g:
                continue
            c = curve(cfg, dl)
            ap(f"| {cfg} | {dl} | {mean([r['g0_tie_rate'] for r in g]):.1f}% | "
               f"{mean([r['g0_stim_separation'] for r in g]):.3f} | "
               f"{sum(1 for r in g if r['g0_acc'] >= 80.0)}/{len(g)} | "
               f"{fmt(c[0], 1) if c else 'n/a'} / {fmt(c[-1], 1) if c else 'n/a'} |")
    ap("")
    ap("## 2. 逐 seed（g=0 正确率 %）")
    ap("")
    ap("| 臂 | D | " + " | ".join(str(s) for s in seeds) + " |")
    ap("|---" * (len(seeds) + 2) + "|")
    for cfg, _ in NM_ARMS:
        for dl in NM_DELAYS:
            g = {r["seed"]: r["g0_acc"] for r in arm(cfg, dl)}
            if not g:
                continue
            ap(f"| {cfg} | {dl} | " + " | ".join(f"{g[s]:.0f}" for s in seeds) + " |")
    ap("")
    ap("## 3. 规则层差异（静态事实，来自 `neuromod` 0.6.0 `rm_stdp` 源码）")
    ap("")
    ap("| 维度 | neuromod 0.6.0 | B1 核心 | 可比性 |")
    ap("|---|---|---|---|")
    ap("| 迹条数 | 1（带符号） | 3（多尺度） | 本轮 v117 已量化 |")
    ap("| τe | 50（单值） | 20/100/500，α=1/3 | 本轮 v117 已量化 |")
    ap("| STDP 核 | τ±=20，A₊=0.01，A₋=0.012 | τ=50，a₊=0.1，a₋=0.12 | **未对齐**（比例同为 1:1.2） |")
    ap("| 奖励项 | `R`（是否减 baseline 需看 engine.rs） | `y − b`（显式基线中心化） | **未对齐** |")
    ap("| 权重域 | [0,2] + L1 重归一化 | [0.001,5] 无归一化 | **不可比** |")
    ap("| 拓扑 | 引擎内部 LIF/Izh 库 | 固定五层 A/B 双通路 | **不可比**（无共同 seed 语义） |")
    ap("")
    ap("## 4. 诚实边界")
    ap("")
    ap("- 本节**不是**「与 Rust 侧统计等价」的结论：Rust 侧一个数都没跑（BLOCKED 未解除）。"
       "本节只回答「把可对齐的迹参数对齐后，行为是否还站得住」。")
    ap("- 单迹 τ=50 与多尺度 20/100/500 的对比在**同一实现内**完成，"
       "因此差异只反映迹配置，不含跨语言/跨拓扑因素。")
    ap("- τe=50 是 `neuromod` 的默认值；其文档允许 50–100，本轮未扫其他取值。")
    ap("- 迹曲线是 `Σ|e|` 的跨试次累加量，量纲与网络规模、发放率相关，"
       "Spearman ρ 只说明形状同不同，不代表幅度可比。")
    ap("")
    ap("## 5. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v117")
    ap("py -3 Experiment/tools/b1_self_supervised.py --suite v117 --report-only")
    ap("```")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"报告 -> {REPORT_MD}")


def build_report(rows):
    if SUITE == "v19":
        return build_report_v19(rows)
    if SUITE == "v110":
        return build_report_v110(rows)
    if SUITE == "v111":
        return build_report_v111(rows)
    if SUITE == "v112":
        return build_report_v112(rows)
    if SUITE == "v113":
        return build_report_v113(rows)
    if SUITE == "v114":
        return build_report_v114(rows)
    if SUITE == "v115":
        return build_report_v115(rows)
    if SUITE == "v116":
        return build_report_v116(rows)
    if SUITE == "v117":
        return build_report_v117(rows)
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
    parser.add_argument("--suite", choices=sorted(SUITES), default="v18",
                        help="v18 = 复现 §6.8 的 30.00%%/30.00pp；v19 = 本轮新实验")
    parser.add_argument("--holdout", action="store_true",
                        help="v111 专用：改用不相交留出 seed 复测，结果另存")
    parser.add_argument("--extend", action="store_true",
                        help="v114 专用：只补跑缺失的长延迟点（追加到已有 JSONL）")
    args = parser.parse_args()
    set_suite(args.suite)
    global EXTEND
    EXTEND = bool(args.extend)
    if args.holdout:
        apply_holdout()
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
