"""T2b · 扇入密度扫描（avg_conn ∈ {8,16,32,64}），复用 T2 的上界流水线。

预注册：audit/t2_fanin_prereg_2026-11-14.md
判定：存在 K 使上界 median ≤ 0.40 → 扇入密度可解决；K=64 仍失败 → 架构表达力不足。
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

D = ROOT / "data_b2" / "t2"
KS = (8, 16, 32, 64)
N_DUMP_EP = 20
N_TRAIN_EP = 14
STRIDE = 10
THRESH = 0.40
REPRO_TOL = 0.02


def main():
    D.mkdir(parents=True, exist_ok=True)
    rows = []
    for K in KS:
        t0 = time.time()
        dump = D / f"dump_k{K}.bin"
        beta = D / f"beta_k{K}.bin"
        ev = D / f"eval_k{K}.csv"
        print(f"\n=== avg_conn={K} ===", flush=True)
        if not dump.exists():
            subprocess.run([str(EXE), "--preset", "m2", "--disable_reward",
                            "--avg-conn", str(K),
                            "--episodes", str(N_DUMP_EP), "--eval-episodes", "0",
                            "--seed", "11", "--stride", str(STRIDE),
                            "--dump-hidden", str(dump),
                            "--out", str(D / f"dump_k{K}.csv")],
                           stdout=subprocess.DEVNULL)

        X, meta, H, stride = read_dump(dump)
        Y = meta[:, 1:3].astype(np.float64)
        eps = meta[:, 7].astype(int)
        tr = eps < N_TRAIN_EP
        a, cv_mse = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
        b = ridge_dual(X[tr], Y[tr], a)
        pred = X[~tr] @ b
        mae = float(np.mean(np.abs(pred - Y[~tr])))
        base = float(np.mean(np.abs(Y[~tr] - Y[tr].mean(axis=0))))
        np.ascontiguousarray(b.astype(np.float64).T.reshape(-1)).tofile(beta)

        med, arr, n, _ = run_eval(["--load-readout", str(beta), "--avg-conn", str(K)], ev)
        rows.append({
            "avg_conn": K, "samples": int(X.shape[0]), "alpha": a, "cv_mse": cv_mse,
            "mae": mae, "const_mae": base, "median": med, "arrived": arr, "n_eval": n,
            "pass": med <= THRESH, "sec": round(time.time() - t0, 1),
        })
        print(f"  median={med:.4f} 到达 {arr}/{n}  MAE={mae:.4f} (base {base:.4f}) "
              f"α={a:g}  {rows[-1]['sec']}s", flush=True)

    hdr = f"{'avg_conn':>9}{'median':>9}{'arrived':>9}{'MAE':>9}{'base':>9}{'pass':>6}{'sec':>7}"
    print("\n" + hdr)
    for r in rows:
        print(f"{r['avg_conn']:>9}{r['median']:>9.4f}{r['arrived']:>9}{r['mae']:>9.4f}"
              f"{r['const_mae']:>9.4f}{'Y' if r['pass'] else 'N':>6}{r['sec']:>7}")

    base_k = next((r for r in rows if r["avg_conn"] == 8), None)
    repro = bool(base_k) and abs(base_k["median"] - 0.9314) <= REPRO_TOL
    win = next((r for r in rows if r["pass"]), None)
    if not repro:
        verdict = "作废：avg_conn=8 未复现 T2 的 0.9314"
    elif win:
        verdict = f"扇入密度可解决：最小达标 avg_conn={win['avg_conn']}（median {win['median']:.4f}）"
    else:
        verdict = "架构表达力不足：avg_conn=64 仍 >0.40，需换输入编码/拓扑"
    print(f"\n复现(±{REPRO_TOL})={repro}  → {verdict}")

    (D / "t2_fanin_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "thresh": THRESH, "repro": repro, "verdict": verdict, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
