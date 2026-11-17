"""T2g · hidden 侧结构消融：同一批状态上比较 input / hidden / 拼接 / 同维 hidden 子集 / RFF。

预注册：audit/t2_struct_prereg_2026-11-14.md
数据：T2e 的 input/hidden dump 来自同一条确定性轨迹（meta 逐位相等，已校验）。
主判定用与 A（input）的配对差，不用 0.40 绝对线。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_decode_upper import (ALPHAS, ROOT, cv_alpha, read_dump,  # noqa: E402
                            ridge_dual, run_eval)

BASE = ROOT / "data_b2" / "t2bal"
D = ROOT / "data_b2" / "t2str"
N_TRAIN_EP = 14
THRESH = 0.40
SUB_COLS = 1056          # D：与 input 同维
RFF_DIM = 3500
RFF_SCALE = 2.5          # 让 cos 相位有足够跨度（x 为二值，||x||≈12.6）
RFF_SEED = 0
MED_TOL = 0.05           # 配对差阈值
DIM_GAP = 0.30           # D 与 A 的维度对照阈值


def fit(X, Y, eps, tr):
    a, cv = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
    b = ridge_dual(X[tr], Y[tr], a)
    mae = float(np.mean(np.abs(X[~tr] @ b - Y[~tr])))
    base = float(np.mean(np.abs(Y[~tr] - Y[tr].mean(axis=0))))
    return a, cv, b, mae, base


def main():
    D.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Xi, mi, Wi, _ = read_dump(BASE / "dump_input.bin")
    Xh, mh, Wh, _ = read_dump(BASE / "dump_hidden.bin")
    if not np.array_equal(mi, mh):
        raise SystemExit("input/hidden dump 轨迹未对齐，拼接实验无效")
    Y = mi[:, 1:3].astype(np.float64)
    eps = mi[:, 7].astype(int)
    tr, te = eps < N_TRAIN_EP, eps >= N_TRAIN_EP
    print(f"样本 {Xi.shape[0]}｜input {Wi} 维｜hidden {Wh} 维｜轨迹对齐 OK", flush=True)

    rows = []

    def record(tag, desc, layer, med, arr, n, a, cv, mae, base, extra=None):
        rows.append({"tag": tag, "desc": desc, "layer": layer, "alpha": a, "cv_mse": cv,
                     "mae": mae, "const_mae": base, "mae_ratio": mae / base,
                     "median": med, "arrived": arr, "n_eval": n,
                     "pass_abs": bool(med <= THRESH), **(extra or {})})
        print(f"  {tag} {desc:<26} α={a:g} CV={cv:.4f} MAE={mae:.4f}/{base:.4f}="
              f"{mae/base:.2f} median={med:.4f} 到达 {arr}/{n}", flush=True)

    # A: input -----------------------------------------------------------
    a, cv, b, mae, base = fit(Xi, Y, eps, tr)
    np.ascontiguousarray(b.T.reshape(-1)).tofile(D / "beta_A_input.bin")
    med, arr, n, _ = run_eval(["--load-readout", str(D / "beta_A_input.bin"),
                               "--readout-layer", "input", "--avg-conn", "8"], D / "eval_A.csv")
    record("A", "input 层", "input", med, arr, n, a, cv, mae, base)

    # B: hidden ----------------------------------------------------------
    a, cv, b, mae, base = fit(Xh, Y, eps, tr)
    np.ascontiguousarray(b.T.reshape(-1)).tofile(D / "beta_B_hidden.bin")
    med, arr, n, _ = run_eval(["--load-readout", str(D / "beta_B_hidden.bin"),
                               "--readout-layer", "hidden", "--avg-conn", "8"], D / "eval_B.csv")
    record("B", "hidden 层", "hidden", med, arr, n, a, cv, mae, base)

    # C: input ⊕ hidden --------------------------------------------------
    Xc = np.hstack([Xi, Xh])
    a, cv, bc, mae, base = fit(Xc, Y, eps, tr)
    np.ascontiguousarray(bc.T.reshape(-1)).tofile(D / "beta_C_both.bin")
    n_in = float(np.linalg.norm(bc[:Wi]))
    n_hid = float(np.linalg.norm(bc[Wi:]))
    med, arr, n, _ = run_eval(["--load-readout", str(D / "beta_C_both.bin"),
                               "--readout-layer", "both", "--avg-conn", "8"], D / "eval_C.csv")
    record("C", "input⊕hidden（4556）", "both", med, arr, n, a, cv, mae, base,
           {"input_norm": n_in, "hidden_norm": n_hid,
            "hidden_norm_share": n_hid / (n_in + n_hid)})

    # D: hidden 随机同维子集（其余 β 补 0，仍走 hidden 层）------------------
    rng = np.random.default_rng(RFF_SEED)
    cols = np.sort(rng.choice(Wh, SUB_COLS, replace=False))
    a, cv, bsub, mae, base = fit(Xh[:, cols], Y, eps, tr)
    bfull = np.zeros((Wh, 2), dtype=np.float64)
    bfull[cols, :] = bsub
    np.ascontiguousarray(bfull.T.reshape(-1)).tofile(D / "beta_D_hidden_sub.bin")
    med, arr, n, _ = run_eval(["--load-readout", str(D / "beta_D_hidden_sub.bin"),
                               "--readout-layer", "hidden", "--avg-conn", "8"], D / "eval_D.csv")
    record("D", f"hidden 随机 {SUB_COLS} 列", "hidden", med, arr, n, a, cv, mae, base)

    # E: RFF(input)（只开环）--------------------------------------------
    W = rng.normal(0.0, RFF_SCALE / np.sqrt(Wi), size=(Wi, RFF_DIM))
    ph = rng.uniform(0.0, 2.0 * np.pi, size=RFF_DIM)
    Xr = np.cos(Xi @ W + ph)
    a, cv, b, mae, base = fit(Xr, Y, eps, tr)
    rows.append({"tag": "E", "desc": f"RFF(input) {RFF_DIM} 维（仅开环）", "layer": "—",
                 "alpha": a, "cv_mse": cv, "mae": mae, "const_mae": base,
                 "mae_ratio": mae / base, "median": None, "arrived": None, "n_eval": None,
                 "pass_abs": None})
    print(f"  E  RFF(input) 仅开环{'':<14} α={a:g} CV={cv:.4f} MAE={mae:.4f}/{base:.4f}="
          f"{mae/base:.2f}", flush=True)

    # 判定 ---------------------------------------------------------------
    A, C, Dd, E = rows[0], rows[2], rows[3], rows[4]
    dCA = C["median"] - A["median"]
    if abs(dCA) <= MED_TOL:
        v1 = f"hidden 无边际信息（C-A={dCA:+.4f}）→ 直通/残差（input→motor）是正确修法"
    elif dCA > MED_TOL:
        v1 = f"hidden 是有害特征（C-A={dCA:+.4f}）→ 必须先重构 hidden，不能只加直通"
    else:
        v1 = f"hidden 有互补信息（C-A={dCA:+.4f}）→ 改为联合解码方向"
    dDA = Dd["median"] - A["median"]
    v2 = (f"同维对照 D-A={dDA:+.4f}（{'不是维度问题，是内容/结构问题' if dDA > DIM_GAP else '维度也占一部分'}）")
    v3 = (f"RFF(input) 开环 MAE/基线={E['mae_ratio']:.2f}（"
          + ("随机非线性特征可行 → 问题在 hidden 的递归慢变量/共线" if E["mae_ratio"] <= 0.30
             else "随机非线性特征也不够 → 连输入侧都需要更结构化的编码") + "）")

    print("\n" + f"{'#':>3}{'median':>9}{'MAE/base':>10}{'C-A':>9}{'pass':>6}  说明")
    for r in rows:
        med = f"{r['median']:.4f}" if r["median"] is not None else "  —  "
        pr = "Y" if r["pass_abs"] else ("N" if r["pass_abs"] is not None else "—")
        print(f"{r['tag']:>3}{med:>9}{r['mae_ratio']:>10.2f}{dCA if r['tag'] == 'C' else 0:>9.4f}"
              f"{pr:>6}  {r['desc']}")
    print(f"\n① {v1}\n② {v2}\n③ {v3}\n用时 {time.time() - t0:.1f}s")

    (D / "t2str_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"), "thresh": THRESH,
        "data": "data_b2/t2bal（同一确定性轨迹，meta 逐位相等）",
        "rows": rows, "dCA": dCA, "dDA": dDA,
        "verdicts": [v1, v2, v3],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
