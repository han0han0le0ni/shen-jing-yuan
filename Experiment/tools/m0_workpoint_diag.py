"""M0 工作点诊断（纯只读）。

目的：把 snn_quant_survey_2026-11-16.md §3 的"饱和/超临界"推断变成实测数字。
做法：用 ctypes 直接调 b2_core.dll，取真实拓扑 + 真实发放，全程不训练、不改核心、不改判据。

产出（stdout JSON + 人读摘要）：
  1. 规模/度数/权重统计（I->H、H->H、H->M、I->M）
  2. H->H 谱半径（Perron 幂迭代，全部正权重）
  3. 平均场临界权重 w_crit 与当前 w/w_crit
  4. 各层发放密度、隐藏层不应期上限占比、ISI-CV
  5. 群体状态有效维度（参与比 PR）
  6. 群体发放率自相关（滞后 1..6）→ 衰落记忆代理指标

用法：
  py tools/m0_workpoint_diag.py --seed 101 --ticks 1200
  py tools/m0_workpoint_diag.py --seed 101 --psp 12      # 仅诊断用，需负责人签字后才可作实验臂
"""
import argparse
import ctypes
import json
import math
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
DLL = EXP / "b2_core.dll"

MEMBRANE_REST = -65.0
SPIKE_THRESHOLD = -55.0
GAP = SPIKE_THRESHOLD - MEMBRANE_REST          # 10 mV
REFRACTORY_TICKS = 3
DEN_CEIL = 1.0 / (1.0 + REFRACTORY_TICKS)      # 0.25 每 tick 发放密度上限（非输入层）

NT_INPUT_X, NT_HIDDEN, NT_M1, NT_M2, NT_INPUT_PROP = 0, 1, 2, 3, 4


def load_lib():
    lib = ctypes.CDLL(str(DLL))
    lib.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_double, ctypes.c_uint]
    lib.net2_init.restype = ctypes.c_int
    lib.net2_destroy.argtypes = [ctypes.c_void_p]
    lib.net2_neuron_count.argtypes = [ctypes.c_void_p]
    lib.net2_neuron_count.restype = ctypes.c_int
    lib.net2_synapse_count.argtypes = [ctypes.c_void_p]
    lib.net2_synapse_count.restype = ctypes.c_int
    lib.net2_tick.argtypes = [ctypes.c_void_p]
    lib.net2_set_target.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    lib.net2_get_topology.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.net2_get_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.net2_export_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.net2_export_weights.restype = ctypes.c_int
    lib.net2_import_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.net2_import_weights.restype = ctypes.c_int
    lib.net2_set_param.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    lib.net2_set_param.restype = ctypes.c_int
    lib.net2_get_param.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.net2_get_param.restype = ctypes.c_double
    return lib


def power_iteration(M, iters=500, tol=1e-10, seed=0):
    """最大特征值模（Perron 根，权重全正）。"""
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.random(n) + 0.5
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        u = M @ v
        nrm = np.linalg.norm(u)
        if nrm == 0.0:
            return 0.0, 0
        v = u / nrm
        new = nrm
        if abs(new - lam) < tol * max(1.0, abs(new)):
            lam = new
            break
        lam = new
    return float(lam), iters


def isi_cv(spike_mat):
    """每个神经元的发放间隔 CV（只统计 ≥4 次发放的神经元），再取中位数。"""
    cvs = []
    for j in range(spike_mat.shape[1]):
        idx = np.flatnonzero(spike_mat[:, j])
        if idx.size < 5:
            continue
        d = np.diff(idx).astype(np.float64)
        m = d.mean()
        if m <= 0:
            continue
        cvs.append(float(d.std() / m))
    if not cvs:
        return float("nan"), 0
    return float(np.median(cvs)), len(cvs)


def participation_ratio(X):
    """X: T×N，返回参与比 (Σλ)²/Σλ²（λ 为协方差特征值 = 奇异值平方）。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = s.astype(np.float64) ** 2
    tot = lam.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    pr = float(tot * tot / (lam ** 2).sum())
    # 有效秩（99% 方差）
    c = np.cumsum(lam) / tot
    k99 = int(np.searchsorted(c, 0.99) + 1)
    return pr, k99


def autocorr_rate(rate, max_lag=6):
    x = rate - rate.mean()
    den = float((x * x).sum())
    out = {}
    if den <= 0:
        return out
    for lag in range(1, max_lag + 1):
        out[lag] = float((x[:-lag] * x[lag:]).sum() / den)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--input", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=3500)
    ap.add_argument("--motor", type=int, default=250)
    ap.add_argument("--avg-conn", type=float, default=8.0)
    ap.add_argument("--ticks", type=int, default=1200)
    ap.add_argument("--psp", type=float, default=25.0, help="仅诊断；实验臂需签字")
    ap.add_argument("--hidden-base", type=float, default=2.0, help="隐藏层内在基电流 mV/tick")
    ap.add_argument("--motor-base", type=float, default=3.0, help="运动层内在基电流 mV/tick")
    ap.add_argument("--tx", type=float, default=1.0)
    ap.add_argument("--ty", type=float, default=1.0)
    ap.add_argument("--target-every", type=int, default=0,
                    help="0=固定目标(默认，静态输入)；K>0=每 K tick 换一次随机目标（非静态输入，ISI-CV 才有意义）")
    ap.add_argument("--expect-synapses", type=int, default=115063,
                    help="§6.22 记录的 m2 拓扑突触数，用于确认 DLL 与源码同步")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--load-weights", type=str, default="",
                    help="初始化后导入权重（用于 import 回环校验）")
    ap.add_argument("--export-weights", type=str, default="",
                    help="在初始化后、任何 tick 之前导出权重（用于 load-weights 回环校验）")
    args = ap.parse_args()

    lib = load_lib()
    net = ctypes.create_string_buffer(1 << 20)  # 远大于 sizeof(NetworkState2)，C 侧自会 memset
    rc = lib.net2_init(net, args.input, args.hidden, args.motor, args.motor,
                       args.avg_conn, args.seed)
    if rc != 0:
        print(json.dumps({"error": f"net2_init rc={rc}"}, ensure_ascii=False))
        return 1

    psp_rc = lib.net2_set_param(net, b"psp_factor", args.psp)
    hb_rc = lib.net2_set_param(net, b"hidden_base_current", args.hidden_base)
    mb_rc = lib.net2_set_param(net, b"motor_base_current", args.motor_base)
    psp_now = lib.net2_get_param(net, b"psp_factor")
    hb_now = lib.net2_get_param(net, b"hidden_base_current")
    mb_now = lib.net2_get_param(net, b"motor_base_current")
    imp_rc = None
    if args.load_weights:
        lp = Path(args.load_weights)
        if not lp.is_absolute():
            lp = EXP / lp
        imp_rc = lib.net2_import_weights(net, str(lp).encode("utf-8"))
    exp_rc = None
    if args.export_weights:
        p = Path(args.export_weights)
        if not p.is_absolute():
            p = EXP / p
        p.parent.mkdir(parents=True, exist_ok=True)
        exp_rc = lib.net2_export_weights(net, str(p).encode("utf-8"))

    N = int(lib.net2_neuron_count(net))
    S = int(lib.net2_synapse_count(net))

    types = (ctypes.c_int * N)()
    px = (ctypes.c_double * N)()
    py = (ctypes.c_double * N)()
    po = (ctypes.c_double * N)()
    pre = (ctypes.c_int * S)()
    post = (ctypes.c_int * S)()
    wt = (ctypes.c_double * S)()
    lib.net2_get_topology(net, types, px, py, po, pre, post, wt)

    tp = np.frombuffer(types, dtype=np.int32).copy()
    pre_a = np.frombuffer(pre, dtype=np.int32).copy()
    post_a = np.frombuffer(post, dtype=np.int32).copy()
    w_a = np.frombuffer(wt, dtype=np.float64).copy()

    is_in = np.isin(tp, [NT_INPUT_X, NT_INPUT_PROP])
    is_h = tp == NT_HIDDEN
    is_m = np.isin(tp, [NT_M1, NT_M2])
    h_idx = np.flatnonzero(is_h)
    in_idx = np.flatnonzero(is_in)
    m_idx = np.flatnonzero(is_m)

    pre_in, post_in = is_in[pre_a], is_in[post_a]
    pre_h, post_h = is_h[pre_a], is_h[post_a]
    pre_m, post_m = is_m[pre_a], is_m[post_a]

    sel = {
        "I->H": pre_in & post_h,
        "H->H": pre_h & post_h,
        "H->M": pre_h & post_m,
        "I->M": pre_in & post_m,
    }
    deg = {}
    for k, m in sel.items():
        deg[k] = int(m.sum())

    # 隐藏层每个神经元的输入入度（I->H）
    indeg_ih = np.bincount(post_a[sel["I->H"]] - h_idx[0], minlength=len(h_idx)) \
        if h_idx.size and h_idx[0] == h_idx[-1] - h_idx.size + 1 else None
    if indeg_ih is None:  # 隐藏层索引不连续时的通用写法
        pos = {int(v): i for i, v in enumerate(h_idx)}
        indeg_ih = np.zeros(len(h_idx), dtype=np.int64)
        for p in post_a[sel["I->H"]]:
            indeg_ih[pos[int(p)]] += 1

    hh = sel["H->H"]
    hmap = {int(v): i for i, v in enumerate(h_idx)}
    rows = np.array([hmap[int(v)] for v in pre_a[hh]], dtype=np.int64)
    cols = np.array([hmap[int(v)] for v in post_a[hh]], dtype=np.int64)
    w_hh = w_a[hh]
    H = len(h_idx)
    Mhh = np.zeros((H, H), dtype=np.float64)
    np.add.at(Mhh, (rows, cols), w_hh)
    # 注意：C 侧 m2 的 PSP 作用于全部突触电流，谱半径按"原始权重"报（口径见报告 §3）
    rho_raw, _ = power_iteration(Mhh)
    rho_eff, _ = power_iteration(Mhh * (psp_now / GAP))

    # --- 平均场临界权重（Freddi et al. 2026, w_crit=(θ-2·I·T_ref)/(β·N)）---
    mean_indeg_h = (deg["I->H"] + deg["H->H"]) / float(H)
    betaN = mean_indeg_h / 2.0          # 平均度 = 2βN
    w_mean_hh_actual = float(w_hh.mean()) if w_hh.size else 0.0
    w_mean_hh_norm = w_mean_hh_actual * psp_now / GAP
    # 忽略外源 I（阶跃输入在公式里是弱项），w_crit 以阈值单位给出
    w_crit_norm = 1.0 / betaN
    w_crit_actual = w_crit_norm * GAP / psp_now

    # --- 闭环运行（无奖励注入，纯动力学）---
    mem = (ctypes.c_double * N)()
    spk = (ctypes.c_double * N)()
    last = (ctypes.c_double * N)()
    arm = (ctypes.c_double * 10)()
    meta = (ctypes.c_int * 3)()
    lib.net2_set_target(net, args.tx, args.ty)

    T = args.ticks
    Xh = np.zeros((T, H), dtype=np.float32)
    Xm = np.zeros((T, len(m_idx)), dtype=np.float32)
    Xin = np.zeros((T, len(in_idx)), dtype=np.float32)
    trng = np.random.default_rng(args.seed * 7919 + 13)
    for t in range(T):
        if args.target_every > 0 and t > 0 and t % args.target_every == 0:
            lib.net2_set_target(net, float(trng.uniform(-1.5, 1.5)),
                                float(trng.uniform(-1.5, 1.5)))
        lib.net2_tick(net)
        lib.net2_get_snapshot(net, mem, spk, last, arm, meta)
        s = np.frombuffer(spk, dtype=np.float64)
        Xh[t] = s[h_idx]
        Xm[t] = s[m_idx]
        Xin[t] = s[in_idx]

    den_h = float(Xh.mean())
    den_m = float(Xm.mean())
    den_in = float(Xin.mean())
    cv_h, n_cv = isi_cv(Xh)
    pr_h, k99_h = participation_ratio(Xh)
    rate_h = Xh.mean(axis=1)
    ac = autocorr_rate(rate_h)

    in_density = den_in
    drive_hidden = float(indeg_ih.mean() * in_density *
                         (float(w_a[sel["I->H"]].mean()) * psp_now) / GAP) if deg["I->H"] else float("nan")

    res = {
        "meta": {
            "seed": args.seed, "ticks": T, "dll": str(DLL),
            "neurons": N, "synapses": S,
            "expected_synapses_m2": args.expect_synapses,
            "reproduces_m2_topology": (S == args.expect_synapses),
            "psp_factor_requested": args.psp, "psp_factor_effective": psp_now,
            "set_param_rc": psp_rc,
            "hidden_base_requested": args.hidden_base, "hidden_base_effective": hb_now,
            "motor_base_requested": args.motor_base, "motor_base_effective": mb_now,
            "base_set_param_rc": [hb_rc, mb_rc],
            "target_every": args.target_every, "target": [args.tx, args.ty],
            "export_weights": args.export_weights, "export_rc": exp_rc,
            "layer_sizes": {"input": int(is_in.sum()), "hidden": int(H), "motor": int(is_m.sum())},
        },
        "topology": {
            "synapses_by_path": deg,
            "hidden_mean_indeg_from_input": float(indeg_ih.mean()),
            "hidden_mean_indeg_total": float(mean_indeg_h),
            "w_ItoH_mean": float(w_a[sel["I->H"]].mean()),
            "w_HtoH_mean": w_mean_hh_actual,
            "w_HtoH_min": float(w_hh.min()), "w_HtoH_max": float(w_hh.max()),
            "w_HtoM_mean": float(w_a[sel["H->M"]].mean()) if deg["H->M"] else None,
            "w_ItoM_mean": float(w_a[sel["I->M"]].mean()) if deg["I->M"] else None,
        },
        "criticality": {
            "spectral_radius_raw_weights": rho_raw,
            "spectral_radius_with_psp_over_gap": rho_eff,
            "betaN_used": betaN,
            "w_mean_hidden_norm_threshold_units": w_mean_hh_norm,
            "w_crit_norm_threshold_units": w_crit_norm,
            "w_ratio_over_wcrit": (w_mean_hh_norm / w_crit_norm) if w_crit_norm else None,
            "w_crit_actual_weight_units": w_crit_actual,
            "note": "w_crit=(theta-2*I*Tref)/(beta*N)，忽略外源 I；betaN=平均入度/2",
        },
        "activity": {
            "density_hidden": den_h,
            "density_hidden_over_ceiling": den_h / DEN_CEIL,
            "density_motor": den_m,
            "density_input": den_in,
            "isi_cv_hidden_median": cv_h,
            "isi_cv_neurons_used": n_cv,
            "participation_ratio_hidden": pr_h,
            "effective_rank_99pct": k99_h,
            "rate_autocorr_hidden": ac,
            "est_drive_hidden_per_tick_threshold_units": drive_hidden,
        },
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        p = (EXP / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[saved] {p}", file=sys.stderr)

    lib.net2_destroy(net)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
