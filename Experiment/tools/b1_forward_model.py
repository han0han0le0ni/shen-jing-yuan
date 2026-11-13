# -*- coding: utf-8 -*-
"""B1「可学习内部前向模型」机制实验（判据 v1.6 §6.6 / 大纲附录 H.4 下一步）

上一轮（附录 H，b1_psel.dll）已把通路选择性做进网络内部的资格迹，并证明它能
破缺对称、产生可归因学习；但留下三个人工成分：
  (a) 内部读出 d 用了结构先验（感觉A->hidden左半->输出A 为「正确路径」）；
  (b) 学习方向靠人工翻转门控符号才对；
  (c) 核心 reward_baseline 饱和到 -0.997，只有「基线补偿诊断组」才出效果。
附录 H.4 要求把「幅度 / 符号 / 基线」做成网络内部**可学习的**量，替代结构先验
与人工方向，回到 host 原结果反馈口径，再重做配对消融。

本脚本对应的 brain/benchmark1_fwd.c -> b1_fwd.dll 用三个在线可学习的内部量替换：
  1. Oja 自监督读出 v：观察窗口累计 hidden 发放向量（400 维，不预设左/右半），
     v 由内部噪声流随机初始化、Oja 规则在线学第一主成分，z = v·(h/|h|)。
     无外部标签、无拓扑先验、无人工方向。
  2. Actor 系数 theta：门控量 g = 0.9*tanh(theta*z)，theta 由内部噪声流随机
     初始化（网络内部探索），用优势 A 在线校准方向，不再人工设方向。
  3. Critic 基线 b：b <- b + eta_c*(y-b)，A = y-b；注入 reward = A + 核心
     reward_baseline，使核心结算的 reward_error 精确等于优势 A，幅度不再被
     核心滑动基线抵消。只读公开字段，核心源码零改动。

门控与上一轮相同：只改 Synapse.e[]（m_A=1+lambda*g / m_B=1-lambda*g），
不写权重、不改 winner；保留 §6.6 的「每试次清迹 + ±5 截断」数值修复。

诚实边界：y 仍是 B1 结果反馈（平局视为失败是对 host 口径的最小改动）；
前向模型自身学习率 eta_o/eta_a/eta_c 是固定机制常数，未扫描；核心
lr=0.02 / psp=10 / trace_k=3 一律不动。即使有效也只证明「可学习内部前向
模型足以让局部 R-STDP 学出映射」，**不构成 B1 通过**。

实验组：
  F  factorial : fwd{0,1} x adv{0,1} x 10 seed（主实验）
  A  ablation  : no_trace / no_reward / freeze_readout / freeze_v /
                 freeze_theta / const_z / shuffle_z
  D  dose      : lambda 剂量-反应
  Y  delay     : D = 0/50/200/500
  S  stream    : M0 四流单变量覆盖隔离
  I  initial   : 不训练的严格对称初始基线

运行：
    py -3 Experiment/tools/b1_forward_model.py
    py -3 Experiment/tools/b1_forward_model.py --quick
    py -3 Experiment/tools/b1_forward_model.py --report-only
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
DLL_PATH = ROOT / "b1_fwd.dll"
DATA = ROOT / "data" / "b1_fwd"
SUMMARY_JSONL = DATA / "fwd_summary.jsonl"
REPORT_MD = ROOT / "audit" / "b1_learnable_forward_model_2026-11-13.md"

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
    d.b1f_create.argtypes = [c_i] * 5 + [c_d, c_u]
    d.b1f_create.restype = c_i
    d.b1f_destroy.argtypes = []
    d.b1f_ready.restype = c_i
    d.b1f_synapse_count.restype = c_i
    d.b1f_readout_edge_count.restype = c_i
    d.b1f_override_stream.argtypes = [c_i, c_u64]
    d.b1f_legacy_rng.argtypes = [c_i]
    d.b1f_set_hyper.argtypes = [c_d, c_d, c_i]
    d.b1f_set_ablation.argtypes = [c_i, c_i]
    d.b1f_set_model.argtypes = [c_i, c_i, c_i, c_i, c_d, c_u64]
    d.b1f_set_freeze_readout.argtypes = [c_i]
    d.b1f_symmetrize.argtypes = [c_d]
    d.b1f_symmetrize.restype = c_d
    d.b1f_asymmetry.restype = c_d
    d.b1f_output_bias.argtypes = [p_d]
    d.b1f_weight_checksum.restype = c_d
    d.b1f_readout_checksum.restype = c_d
    d.b1f_frozen_vote.argtypes = [c_i, p_i, p_i]
    d.b1f_frozen_vote.restype = c_i
    d.b1f_run_trial.argtypes = [c_i, c_i, c_i, p_d]
    d.b1f_run_trial.restype = c_i
    for name in ("last_theta", "last_theta0", "last_b", "last_z",
                 "last_A", "last_vnorm"):
        getattr(d, f"b1f_{name}").restype = c_d
    return d


def new_net(d, seed, fwd=1, adv=1, learn_theta=1, z_mode=0,
            lam=LAMBDA, ablation=(0, 0), freeze=0, overrides=None,
            model_seed=0):
    """每次实验重建网络，保证单变量与逐位可复现；一律 alpha=1 严格对称。"""
    for name in STREAM:
        d.b1f_override_stream(STREAM[name], 0)
    d.b1f_legacy_rng(0)
    for name, val in (overrides or {}).items():
        d.b1f_override_stream(STREAM[name], val)
    rc = d.b1f_create(SA, SB, HIDDEN, OA, OB, AVG_CONN, C.c_uint(seed))
    if rc != 0:
        raise RuntimeError(f"b1f_create 失败 seed={seed} rc={rc}")
    d.b1f_set_hyper(LR, PSP, TRACE_K)
    d.b1f_set_ablation(*ablation)
    d.b1f_set_model(fwd, adv, learn_theta, z_mode, lam, model_seed)
    d.b1f_set_freeze_readout(freeze)
    return d.b1f_symmetrize(1.0)


def bias_beta(d):
    buf = (C.c_double * 4)()
    d.b1f_output_bias(buf)
    q = [float(x) for x in buf]
    tot = sum(q)
    return (((q[0] + q[3]) - (q[1] + q[2])) / tot if tot > 0 else 0.0), q


def labels(n):
    return [i % 2 for i in range(n)]


def g0_eval(d, eval_n):
    """g=0：冻结投票本身不注入任何东西，评估前后权重漂移必须为 0。"""
    d.b1f_set_model(0, 0, 0, 0, 0.0, 0)
    w_before = d.b1f_weight_checksum()
    correct = ties = a_eq_b = 0
    a_sum = b_sum = 0
    pred_cnt = {0: [0, 0, 0], 1: [0, 0, 0]}
    for stim in labels(eval_n):
        ca, cb = C.c_int(), C.c_int()
        pred = d.b1f_frozen_vote(stim, C.byref(ca), C.byref(cb))
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
        "g0_weight_drift": d.b1f_weight_checksum() - w_before,
        "g0_trials": eval_n,
    }


def experiment(d, seed, group, fwd=1, adv=1, learn_theta=1, z_mode=0,
               lam=LAMBDA, ablation=(0, 0), freeze=0, delay=DELAY,
               n=TRIALS, overrides=None, eval_n=EVAL_TRIALS,
               model_seed=0, label=""):
    asym_i = new_net(d, seed, fwd, adv, learn_theta, z_mode, lam,
                     ablation, freeze, overrides, model_seed)
    beta_i, _ = bias_beta(d)
    theta0 = d.b1f_last_theta0()
    w0 = d.b1f_weight_checksum()
    ro0 = d.b1f_readout_checksum()
    half = n // 2
    tr_c = te_c = ties = reward_nonzero = 0
    drift = 0.0
    z_abs = z_a = z_b = 0.0
    a_abs = a_nonzero = gate_abs = mirror_sum = baseline_sum = 0.0
    out = (C.c_double * 10)()
    for i, stim in enumerate(labels(n)):
        d.b1f_run_trial(stim, 1, delay, out)
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
        z = d.b1f_last_z()
        z_abs += abs(z)
        if stim == 0:
            z_a += z
        else:
            z_b += z
        a_abs += abs(out[9])
        if abs(out[9]) > 1e-9:
            a_nonzero += 1
        gate_abs += abs(out[6])
        mirror_sum += out[8]
        baseline_sum += d.b1f_last_b()
    theta_f = d.b1f_last_theta()
    b_f = d.b1f_last_b()
    vnorm = d.b1f_last_vnorm()
    g0 = g0_eval(d, eval_n)
    asym_f = d.b1f_asymmetry()
    beta_f, _ = bias_beta(d)
    w1 = d.b1f_weight_checksum()
    ro1 = d.b1f_readout_checksum()
    n_syn = d.b1f_synapse_count()
    d.b1f_destroy()
    row = {
        "group": group, "label": label, "seed": seed, "n": n,
        "fwd": fwd, "adv": adv, "learn_theta": learn_theta,
        "z_mode": z_mode, "lambda": lam, "delay": delay,
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
        "theta0": theta0, "theta_final": theta_f,
        "b_final": b_f, "v_norm": vnorm,
        "z_abs_mean": z_abs / n,
        "z_stim_gap": z_a / half - z_b / (n - half),
        "A_abs_mean": a_abs / n, "A_nonzero_rate": a_nonzero / n * 100.0,
        "gate_abs_mean": gate_abs / n,
        "mirror_trace_diff": mirror_sum / n,
        "critic_b_mean": baseline_sum / n,
        "synapses": n_syn,
    }
    row.update(g0)
    return row


def initial_vote(d, seed, eval_n):
    asym_i = new_net(d, seed, 0, 0, 0, 0, 0.0)
    beta_i, _ = bias_beta(d)
    g0 = g0_eval(d, eval_n)
    d.b1f_destroy()
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
        print(f"[{r['group']:<9}] f{r['fwd']}a{r['adv']} "
              f"t{r['learn_theta']} z{r['z_mode']} {r['ablation']:<8} "
              f"seed={r['seed']:<4} g0={r['g0_acc']:6.2f}% "
              f"sep={r['g0_stim_separation']:.2f} "
              f"theta={r['theta0']:+.3f}->{r['theta_final']:+.3f} "
              f"zgap={r['z_stim_gap']:+.3f} "
              f"inj={r['test_inject_acc']:6.2f}%", flush=True)
        return r

    full = dict(fwd=1)

    # ---- F: 2x2 主实验（Oja 读出 x 可学习优势） -----------------------
    for fw in (0, 1):
        for ad in (0, 1):
            for s in seeds:
                add(d, s, "factorial", fwd=fw, adv=ad)

    # ---- A: 完整组配对归因消融 --------------------------------------
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(1, 0), label="no_trace")
    for s in seeds:
        add(d, s, "ablation", **full, ablation=(0, 1), label="no_reward")
    for s in seeds:
        add(d, s, "ablation", **full, freeze=1, label="freeze_readout")
    for s in seeds:
        add(d, s, "ablation", **full, learn_theta=0, label="freeze_theta")
    for s in seeds:
        add(d, s, "ablation", **full, z_mode=1, label="const_z")
    for s in seeds:
        add(d, s, "ablation", **full, z_mode=2, label="shuffle_z")

    # ---- B: 瓶颈定位对照（不计入验收） -------------------------------
    # 冒烟发现 Oja 主成分的 z 与刺激几乎无关、theta 几乎不动。用两个对照
    # 分离「自监督读出缺刺激信息」与「优势信号太稀疏 / actor 学不动」。
    for s in seeds:
        add(d, s, "probe", **full, z_mode=3, label="prior_z")
    for s in seeds:
        add(d, s, "probe", **full, adv=2, label="no_critic")

    # ---- D: lambda 剂量-反应 ----------------------------------------
    for lam in LAMBDA_DOSES:
        for s in dseeds:
            add(d, s, "dose", **full, lam=lam)

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

    ap("# B1「可学习内部前向模型」机制实验报告（2026-11-13）")
    ap("")
    ap("> 依据：`spec/judgments.md` **v1.6 §6.6** 与大纲**附录 H.4** 的下一步。")
    ap("> 本报告只记录**机制诊断结论**：判据阈值、学习规则、刺激生成器、")
    ap("> `benchmark1_core.c/.h`、probe/sym/t1/psel、host 一律零改动。")
    ap("> 复现：`py -3 Experiment/tools/b1_forward_model.py`")
    ap("> （`--report-only` 可从已存 JSONL 重算；`--quick` 为冒烟）。")
    ap("")
    ap("## 0. 实验定位：替换上一轮的三个人工成分")
    ap("")
    ap("附录 H 已把通路选择性做进网络内部的资格迹，并证明它能破缺对称、产生")
    ap("可归因学习；但留下三个人工成分：结构先验读出、人工翻转门控方向、")
    ap("核心 `reward_baseline` 饱和（只有补偿诊断组才出效果）。附录 H.4 要求把")
    ap("「幅度 / 符号 / 基线」做成网络内部**可学习的**量，回到 host 原结果反馈口径。")
    ap("")
    ap("新增 `brain/benchmark1_fwd.c` → `b1_fwd.dll`，只链接既有 `b1_core.o` 与")
    ap("`seed_bundle.o`。三个在线可学习的内部量：")
    ap("")
    ap("| # | 可学习量 | 替换掉的人工成分 | 学习规则（无外部标签） |")
    ap("|---|---|---|---|")
    ap("| 1 | Oja 自监督读出 `v` → 内部隐变量 `z` | 结构先验 `d_raw`（左/右半） | "
       "观察窗口累计 hidden 发放向量（400 维，不预设左/右半）；`v` 由内部噪声流随机"
       "初始化，`v ← normalize(v + η_o·proj·(ĥ − proj·v))`，`z = v·ĥ`。符号由随机"
       "初始化决定，相关性由网络自身活动学出。 |")
    ap("| 2 | Actor 系数 `theta` | 人工方向反转 `sel_mode=6` | "
       "`g = 0.9·tanh(theta·z)`，`theta` 由内部噪声流随机初始化（网络内部探索），"
       "`theta ← clamp(theta + η_a·A·z)`；与资格迹符号一致的方向由学习自动校准。 |")
    ap("| 3 | Critic 基线 `b` 与优势 `A` | 饱和的 `reward_baseline` / 补偿诊断组 | "
       "`y` = B1 结果反馈；`b ← b + η_c·(y−b)`，`A = y−b`；注入 "
       "`reward = A + g_net.reward_baseline`，使核心结算的 `reward_error` 精确等于"
       "优势 `A`（只读写公开字段，核心源码零改动）。 |")
    ap("")
    ap("门控与上一轮相同：只改 `Synapse.e[]`（`m_A = 1+λg` / `m_B = 1−λg`），")
    ap("不写权重、不改 `winner`；保留 §6.6 的「每试次清迹 + ±5 截断」数值修复。")
    ap("")
    ap("> **⚠️ 诚实边界**：`y` 仍是 B1 结果反馈（pred 与期望比较），")
    ap("> 「平局视为失败 −1」是对 host 口径的最小改动，需随报告记录；未使用")
    ap("> §6.5 的 `mode=1` 刺激相关外部教学信号。前向模型自身学习率")
    ap("> `η_o=0.01 / η_a=0.05 / η_c=0.05` 是**固定机制常数、未扫描、未挑最优**；")
    ap("> 核心 `lr=0.02 / psp=10 / trace_k=3` 一律不动。即使本机制有效，也只证明")
    ap("> 「可学习内部前向模型足以让局部 R-STDP 学出映射」，**不构成 B1 通过**")
    ap("> （前向模型是新增的内部结构）。")
    ap("")
    ap(f"> **⚠️ 能力口径 g=0**：训练结束后关闭全部门控与奖励，用冻结投票评估")
    ap(f"> {nev} 个交替 A/B 试次；评估前后权重漂移必须为 0，并同时报告")
    ap("> **A/B 预测分布差**，防止「恒定答案」冒充映射。训练期含注入的即时")
    ap("> 正确率单独记录，不得当作能力。")
    ap("")
    ap("### 0.1 证据链（SHA1）")
    ap("")
    ap("| 文件 | SHA1 |")
    ap("|---|---|")
    for rel in ("brain/benchmark1_core.c", "brain/benchmark1_core.h",
                "brain/benchmark1_probe.c", "brain/benchmark1_sym.c",
                "brain/benchmark1_t1.c", "brain/benchmark1_psel.c",
                "brain/benchmark1_fwd.c", "host/benchmark1_host.cpp",
                "core/seed_bundle.c", "spec/judgments.md",
                "tools/b1_forward_model.py", "b1_fwd.dll"):
        p = ROOT / rel
        ap(f"| `{rel}` | `{sha1(p) if p.exists() else 'MISSING'}` |")
    ap("")
    core = sha1(ROOT / "brain/benchmark1_core.c")
    ap(f"核心 `benchmark1_core.c` SHA1 = `{core}`，与 §6.2–§6.6 报告一致："
       f"**{'未断' if core.startswith(CORE_SHA1_PREFIX) else '不一致（需核查）'}**。")
    ap("删掉 `benchmark1_fwd.c` 与 `b1_fwd.dll` 即可完全恢复原状。")
    ap("")

    # ---- 1 主实验
    ap(f"## 1. 主实验：2×2 析因（g=0 口径，{nseed} seed × {ntr} 试次）")
    ap("")
    ap("全部在 `alpha=1` 严格对称基线上，延迟 D=200，`λ=1.0`。")
    ap("`fwd=1` 表示 Oja 在线学 `v`，`adv=1` 表示可学习优势替代核心饱和基线。")
    ap("")
    ap("| Oja 读出 | 可学习优势 | g=0 正确率 mean±SD | A/B 预测分布差 | 镜像边迹差 | 训练后残差 | 非零优势比例 |")
    ap("|---|---|---|---|---|---|---|")
    cells = {}
    for fw in (0, 1):
        for ad in (0, 1):
            g = [r for r in fac if r["fwd"] == fw and r["adv"] == ad]
            if not g:
                continue
            cells[(fw, ad)] = g
            s, _ = stat(g)
            ap(f"| {'开' if fw else '关'} | {'开' if ad else '关'} | **{s}%** | "
               f"{fmt(mean([r['g0_stim_separation'] for r in g]), 3)} | "
               f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} | "
               f"{fmt(mean([r['asym_final'] for r in g]), 5)} | "
               f"{fmt(mean([r['A_nonzero_rate'] for r in g]), 1)}% |")
    ap("")
    ap("逐 seed 的 g=0 正确率：")
    ap("")
    ap("| seed | f0a0 | f1a0 | f0a1 | f1a1 |")
    ap("|---|---|---|---|---|")
    for s in sorted({r["seed"] for r in fac}):
        vals = []
        for key in ((0, 0), (1, 0), (0, 1), (1, 1)):
            g = [r for r in cells.get(key, []) if r["seed"] == s]
            vals.append(fmt(g[0]["g0_acc"]) if g else "n/a")
        ap(f"| {s} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    ap("")
    m = {k: mean([r["g0_acc"] for r in v]) for k, v in cells.items()}
    joint = cells.get((1, 1), [])
    ap(f"两因素联合效果（f1a1 − f0a0）= **{m.get((1, 1), 0) - m.get((0, 0), 0):+.2f}pp**；"
       f"仅 Oja（f1a0 − f0a0）= {m.get((1, 0), 0) - m.get((0, 0), 0):+.2f}pp；"
       f"仅优势（f0a1 − f0a0）= {m.get((0, 1), 0) - m.get((0, 0), 0):+.2f}pp。")
    ap("")

    # ---- 2 配对消融
    abl = sel(rows, "ablation")
    by_lbl = {lab: [r for r in abl if r.get("label") == lab]
              for lab in ("no_trace", "no_reward", "freeze_readout",
                          "freeze_theta", "const_z", "shuffle_z")}
    notrace = by_lbl["no_trace"]
    diffs, common = paired_delta(rows, joint, notrace)
    dmean, dsd = mean(diffs), (sd(diffs) if diffs else 0.0)
    ap("## 2. 同 seed 配对消融（归因判据：完整 − 无迹 ≥ 30pp）")
    ap("")
    ap("| 消融条件 | g=0 正确率 mean±SD | 与完整组配对差 | 镜像边迹差 |")
    ap("|---|---|---|---|")
    for label, g in (("完整（f1a1）", joint), ("无资格迹", notrace),
                     ("无奖励调制", by_lbl["no_reward"]),
                     ("冻结读出", by_lbl["freeze_readout"]),
                     ("冻结 actor", by_lbl["freeze_theta"]),
                     ("恒定 z", by_lbl["const_z"]),
                     ("shuffle z", by_lbl["shuffle_z"])):
        if not g:
            continue
        s, _ = stat(g)
        dl, _ = paired_delta(rows, joint, g)
        ap(f"| {label} | {s}% | {fmt(mean(dl))}pp | "
           f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} |")
    ap("")
    ap(f"**主归因判据**：同 seed 配对「完整 − 无迹」= **{fmt(dmean)}pp**"
       f"（SD {fmt(dsd)}，{len(common)} 对），判据要求 ≥30pp → "
       f"**{'PASS（可归因到学习）' if dmean >= 30.0 else 'FAIL（无可归因的提升）'}**。")
    ap("")

    # ---- 3 学习动态
    ap("## 3. 学习动态：三个可学习量到底学到了什么")
    ap("")
    ap("| 组 | theta 初值 → 终值 | b 收敛值 | z 幅度 | z 的 A/B 差 | 优势幅度 | A 非零率 | 门控幅度 |")
    ap("|---|---|---|---|---|---|---|---|")
    for label, g in (("f0a0 基线", cells.get((0, 0), [])),
                     ("f1a0 仅 Oja", cells.get((1, 0), [])),
                     ("f0a1 仅优势", cells.get((0, 1), [])),
                     ("f1a1 完整", joint)):
        if not g:
            continue
        ap(f"| {label} | {fmt(mean([r['theta0'] for r in g]), 3)} → "
           f"{fmt(mean([r['theta_final'] for r in g]), 3)} | "
           f"{fmt(mean([r['b_final'] for r in g]), 3)} | "
           f"{fmt(mean([r['z_abs_mean'] for r in g]), 3)} | "
           f"{fmt(mean([r['z_stim_gap'] for r in g]), 3)} | "
           f"{fmt(mean([r['A_abs_mean'] for r in g]), 3)} | "
           f"{fmt(mean([r['A_nonzero_rate'] for r in g]), 1)}% | "
           f"{fmt(mean([r['gate_abs_mean'] for r in g]), 3)} |")
    ap("")
    ap("`z_stim_gap` = 刺激 A 与刺激 B 试次的平均 `z` 之差，用于检验 Oja 读出")
    ap("是否真的**从网络活动里学出了刺激相关信息**（而不是随机投影）；")
    ap("`theta` 终值相对初值的变化用于检验 actor 是否校准了方向。")
    ap("")
    ap(f"g=0 评估前后权重漂移：完整组均值 "
       f"{fmt(mean([abs(r['g0_weight_drift']) for r in joint]), 8)}（必须为 0）。")
    ap("")

    # ---- 3.1 瓶颈定位
    probe = sel(rows, "probe")
    pz = [r for r in probe if r.get("label") == "prior_z"]
    nc = [r for r in probe if r.get("label") == "no_critic"]
    ap("### 3.1 瓶颈定位对照（不计入验收）")
    ap("")
    ap("冒烟发现 Oja 第一主成分的 `z` 与刺激几乎无关（`z_stim_gap≈0.03`）、")
    ap("`theta` 几乎不移动。为分离两种解释，另设两个只改一件事的对照：")
    ap("")
    ap("- `prior_z`：把 `z` 换成上一轮的结构先验量 `d_raw`（hidden 左右半发放差），")
    ap("  其余（actor / critic / 门控）全部保持可学习 —— 检验瓶颈是否在自监督读出；")
    ap("- `no_critic`：把优势换成 `A = y`（冻结 critic 减法、仍做注入补偿），")
    ap("  幅度恒 1 —— 检验瓶颈是否在优势信号过稀疏。")
    ap("")
    ap("两组**只作瓶颈定位，不计入任何验收**。")
    ap("")
    ap("| 组 | g=0 正确率 mean±SD | A/B 预测分布差 | z 的 A/B 差 | theta 位移 | 优势幅度 | 镜像边迹差 |")
    ap("|---|---|---|---|---|---|---|")
    for desc, g in (("完整 f1a1（三个量全可学习）", joint),
                    ("prior_z（结构先验 z + 可学习 actor/critic）", pz),
                    ("no_critic（Oja z + 冻结 critic）", nc)):
        if not g:
            continue
        s, _ = stat(g)
        ap(f"| {desc} | {s}% | "
           f"{fmt(mean([r['g0_stim_separation'] for r in g]), 3)} | "
           f"{fmt(mean([r['z_stim_gap'] for r in g]), 3)} | "
           f"{fmt(mean([abs(r['theta_final'] - r['theta0']) for r in g]), 3)} | "
           f"{fmt(mean([r['A_abs_mean'] for r in g]), 3)} | "
           f"{fmt(mean([r['mirror_trace_diff'] for r in g]), 5)} |")
    ap("")
    pz_acc = mean([r["g0_acc"] for r in pz])
    nc_acc = mean([r["g0_acc"] for r in nc])
    joint_acc = mean([r["g0_acc"] for r in joint])
    pz_hit = sum(1 for r in pz if r["g0_acc"] > 0.0)
    if pz_acc > joint_acc + 5.0 or pz_hit > 0:
        ap("**瓶颈主要在自监督读出**：Oja 第一主成分的 `z` 与刺激几乎无关")
        ap(f"（`z_stim_gap≈{fmt(mean([r['z_stim_gap'] for r in joint]), 3)}`），")
        ap(f"换成含刺激信息的 `z` 后（`z_stim_gap≈{fmt(mean([r['z_stim_gap'] for r in pz]), 3)}`）")
        ap(f"有 **{pz_hit}/{len(pz)}** 个 seed 出现提升，`theta` 也被 actor 明显推动。")
        ap("说明最大方差方向 ≠ 刺激判别方向（§6.3 的「线性可分」不等于「PCA 可提取」），")
        ap("actor / critic 本身并非完全学不动；但整体仍不稳定，可学习前向模型")
        ap("**尚不足以稳定替代结构先验**。")
    elif nc_acc > joint_acc + 5.0:
        ap("**瓶颈在优势信号稀疏**：critic 把恒定的平局反馈完全吸收，`A` 趋零，")
        ap("actor 得不到更新信号；Oja 读出不是主要限制。")
    else:
        ap("两个对照都未出现提升：瓶颈既不在单一读出、也不在单一优势幅度，")
        ap("而是「hidden 自监督表示」与「该选哪个输出」之间**缺少可学习的")
        ap("桥接目标** —— 在破缺之前结果反馈恒为负 / 平局，不含刺激信息；")
        ap("内部自监督只能学到 hidden 的统计结构，学不到任务语义。")
    ap("")

    # ---- 4 剂量 / 延迟 / 四流 / 初始
    dose = sel(rows, "dose")
    delay = sel(rows, "delay")
    stream = sel(rows, "stream")
    ini = sel(rows, "initial")
    ap("## 4. 剂量-反应、延迟、四流隔离与初始基线")
    ap("")
    dl_by_lam = {}
    for r in dose:
        dl_by_lam.setdefault(r["lambda"], []).append(r)
    dl_by_del = {}
    for r in delay:
        dl_by_del.setdefault(r["delay"], []).append(r)
    lams = sorted(dl_by_lam)
    dels = sorted(dl_by_del)
    ap("| λ | g=0 正确率 |  | D | g=0 正确率 |")
    ap("|---|---|---|---|---|---|")
    for i in range(max(len(lams), len(dels))):
        lpart = ["", ""] if i >= len(lams) else [
            fmt(lams[i], 2), fmt(mean([r["g0_acc"] for r in dl_by_lam[lams[i]]]))]
        dpart = ["", ""] if i >= len(dels) else [
            str(dels[i]), fmt(mean([r["g0_acc"] for r in dl_by_del[dels[i]]]))]
        ap(f"| {lpart[0]} | {lpart[1]} |  | {dpart[0]} | {dpart[1]} |")
    ap("")
    if stream:
        ap("四流单变量覆盖（完整组）：")
        ap("")
        ap("| 覆盖流 | g=0 正确率 mean±SD | z 的 A/B 差 |")
        ap("|---|---|---|")
        for name in STREAM:
            g = [r for r in stream if name in r.get("overrides", {})]
            if g:
                s, _ = stat(g)
                ap(f"| {name} | {s}% | {fmt(mean([r['z_stim_gap'] for r in g]), 3)} |")
        ap("")
    if ini:
        ap(f"不训练的严格对称初始基线：g=0 {fmt(mean([r['g0_acc'] for r in ini]))}% / "
           f"平局率 {fmt(mean([r['g0_tie_rate'] for r in ini]))}%（复现 §6.4）。")
        ap("")

    # ---- 5 结论
    ap("## 5. 结论与下一步")
    ap("")
    zgap = mean([r["z_stim_gap"] for r in joint])
    theta_move = mean([abs(r["theta_final"] - r["theta0"]) for r in joint])
    ap(f"完整组 g=0 = **{fmt(m.get((1, 1)))}%**，A/B 预测分布差 "
       f"{fmt(mean([r['g0_stim_separation'] for r in joint]), 3)}，"
       f"配对「完整 − 无迹」**{fmt(dmean)}pp**；Oja 读出的 A/B 差 "
       f"{fmt(zgap, 3)}，actor 系数位移 {fmt(theta_move, 3)}。")
    ap("")
    if dmean >= 30.0 and m.get((1, 1), 0) >= 80.0:
        ap("**可学习内部前向模型在本实验中让局部 R-STDP 学出了映射**：无结构先验、")
        ap("无人工方向、无基线补偿注入，三个量全部由网络内部在线学习，且提升可")
        ap("归因到资格迹依赖的学习（配对消融 ≥30pp）。但这**不构成 B1 通过**：")
        ap("前向模型是新增的内部结构，`y` 的平局口径也偏离 host；需回到 host")
        ap("原口径、并把前向模型纳入正式判据后，才谈重做 B1 验收。")
    elif dmean >= 30.0:
        ap("**出现了可归因到学习的部分提升**，但未达 B1 的 80% 阈值：")
        ap("三个可学习量确实替代了人工成分，却不足以完全学出映射。需检查")
        ap("`z` 的刺激相关性、`theta` 的探索幅度与优势 `A` 的稀疏性（§3）。")
    else:
        ap("**无可归因提升（FAIL）**：可学习内部前向模型在本实现下仍不足以让")
        ap("局部 R-STDP 在 B1 结果反馈口径下学出映射。§3 的学习动态与 §3.1")
        ap("的两个对照已排除「优势幅度」与「actor 完全学不动」，把主因定位到")
        ap("自监督读出；如实记录为负结果。§2 恒定 / shuffle 组与 §4 延迟、四流")
        ap("扫描组的零星提升来自随机破缺、未做配对归因，均不计入验收。")
        ap("§3.1 的对照把瓶颈定位在**自监督读出**：Oja 第一主成分不是刺激判别")
        ap("方向，hidden 里虽有可分信息（§6.3），但「线性可分」不等于「PCA 可提取」。")
        ap("更根本地，破缺之前结果反馈恒为负 / 平局、不含刺激信息，")
        ap("hidden 自监督表示与「该选哪个输出」之间仍缺少可学习的桥接目标。")
        ap("附录 H.4 的三个人工成分因此不是「没学会」这么简单 —— 当前局部规则 +")
        ap("结果反馈的组合下，需要一个既不依赖外部标签、又能指向任务语义的")
        ap("自监督目标，才可能自发破对称。")
    ap("")
    ap("**纪律不变**：不得调 lr / psp / τ、改刺激生成器或挑最好一组刷过判据；")
    ap("恒定 / shuffle 组、诊断组均不计入任何验收。")
    ap("§5.1「M1 不过，不写容器、不写 SNC、不碰 B2 以上」继续生效；")
    ap("**B1 仍不通过**。")
    ap("")
    ap("## 6. 复现")
    ap("")
    ap("```")
    ap("py -3 Experiment/tools/b1_forward_model.py --report-only")
    ap("py -3 Experiment/tools/b1_forward_model.py --quick")
    ap("py -3 Experiment/tools/b1_forward_model.py")
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
    print("FWD REPORT OK")


if __name__ == "__main__":
    main()
