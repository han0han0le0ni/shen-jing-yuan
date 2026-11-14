"""T2c · input 层读出上界（判据 §2 门槛作参考线，不改变 B2 状态）。

预注册：audit/t2_input_prereg_2026-11-14.md
流程与 T2 一致，只把读出源层从 hidden 换成 input（1056 维：1000 位置编码 + 56 本体感觉）。
判定：input 上界 median ≤0.40 → 信息在输入、hidden 破坏（修拓扑）；否则 → 输入编码不足（换编码）。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_decode_upper import (ALPHAS, EXE, ROOT, cv_alpha, read_dump,  # noqa: E402
                            ridge_dual, run_eval)

D = ROOT / "data_b2" / "t2in"
N_DUMP_EP = 20
N_TRAIN_EP = 14
STRIDE = 10
THRESH = 0.40
REF_HIDDEN = 0.9314   # §6.23 / §6.24 的 hidden 上界（avg_conn=8），只作参照，不重跑


def load_targets(csv):
    """episode -> (tx, ty)，用于解析状态上界诊断。"""
    out = {}
    for ln in csv.read_text(encoding="utf-8").splitlines()[1:]:
        p = ln.split(",")
        if len(p) < 3:
            continue
        out[int(p[0])] = (float(p[1]), float(p[2]))
    return out


def main():
    D.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dump = D / "dump_input.bin"
    beta = D / "beta_input.bin"
    ev = D / "eval_input.csv"
    dump_csv = D / "dump_input.csv"

    print("[1/5] 采集 input 层发放（随机初始化，禁用奖励）…", flush=True)
    if not dump.exists():
        subprocess.run([str(EXE), "--preset", "m2", "--disable_reward",
                        "--avg-conn", "8", "--dump-layer", "input",
                        "--episodes", str(N_DUMP_EP), "--eval-episodes", "0",
                        "--seed", "11", "--stride", str(STRIDE),
                        "--dump-hidden", str(dump), "--out", str(dump_csv)],
                       stdout=subprocess.DEVNULL)

    X, meta, H, stride = read_dump(dump)
    Y = meta[:, 1:3].astype(np.float64)
    eps = meta[:, 7].astype(int)
    live = X.max(axis=0)                        # 每列是否曾发放
    print(f"      samples={X.shape[0]} width={H} stride={stride} "
          f"episodes={len(np.unique(eps))} 密度={X.mean():.4f} 活跃列={int(live.sum())}/{H}")

    tr = eps < N_TRAIN_EP
    te = ~tr
    a, cv_mse = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
    b = ridge_dual(X[tr], Y[tr], a)
    pred = X[te] @ b
    mae = float(np.mean(np.abs(pred - Y[te])))
    base = float(np.mean(np.abs(Y[te] - Y[tr].mean(axis=0))))
    print(f"[2/5] ridge α={a:g}（CV MSE={cv_mse:.4f}）→ β {b.shape}；"
          f"留出 MAE={mae:.4f}（基线 {base:.4f}）")

    np.ascontiguousarray(b.astype(np.float64).T.reshape(-1)).tofile(beta)

    # 附带诊断：6 维精确状态 [θ1, θ2, end_x, end_y, tx, ty] 的线性天花板
    diag = {}
    try:
        tg = load_targets(dump_csv)
        tgt = np.array([tg.get(int(e), (np.nan, np.nan)) for e in eps], dtype=np.float64)
        S = np.column_stack([meta[:, 3], meta[:, 4], meta[:, 5], meta[:, 6], tgt])
        ok = np.isfinite(S).all(axis=1)
        if ok.sum() > 100:
            Sa, Sb = S[ok][tr[ok]], Y[ok][tr[ok]]
            ea = eps[ok][tr[ok]]
            a2, cv2 = cv_alpha(Sa, Sb, ea, ALPHAS)
            b2 = ridge_dual(Sa, Sb, a2)
            p2 = S[ok][~tr[ok]] @ b2
            y2 = Y[ok][~tr[ok]]
            diag = {"state_alpha": a2, "state_cv_mse": cv2,
                    "state_mae": float(np.mean(np.abs(p2 - y2))),
                    "state_base": float(np.mean(np.abs(y2 - Sb.mean(axis=0))))}
            print(f"      诊断·精确状态 6 维线性上界：MAE={diag['state_mae']:.4f}"
                  f"（基线 {diag['state_base']:.4f}）")
    except Exception as e:      # 诊断失败不影响主判定
        print(f"      诊断跳过：{e}")

    print("[3/5] 写回 β …", flush=True)
    med, arr, n, _ = run_eval(["--load-readout", str(beta), "--readout-layer", "input",
                               "--avg-conn", "8"], ev)
    print(f"[4/5] input 上界（线性读出，闭环）median={med:.4f} 到达 {arr}/{n}")

    ok = med <= THRESH
    worse_than_hidden = med >= REF_HIDDEN
    if ok:
        verdict = "信息在输入、hidden 破坏：下一步只动拓扑（直通/残差、换 hidden 编码），不换编码"
    else:
        verdict = "输入编码不足：下一步换编码（更高维/更密/时间编码），不再动 hidden 扇入"
        if worse_than_hidden:
            verdict += "；且读出层信息量不增（input ≥ hidden），指向编码维度/分辨率"
    print(f"[5/5] 判定：input 上界 {'PASS' if ok else 'FAIL'}（门槛 {THRESH}，hidden 参照 {REF_HIDDEN}）"
          f" → {verdict}")
    print(f"用时 {time.time() - t0:.1f}s")

    (D / "t2in_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "width": int(H), "stride": int(stride), "samples": int(X.shape[0]),
        "density": float(X.mean()), "live_cols": int(live.sum()),
        "alpha": a, "cv_mse": cv_mse, "mae": mae, "const_mae": base,
        "median": med, "arrived": arr, "n_eval": n, "pass": bool(ok),
        "ref_hidden_median": REF_HIDDEN, "worse_than_hidden": bool(worse_than_hidden),
        "verdict": verdict, **diag,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
