"""T2j · 速率码读出（开+闭环）：τ_rate ∈ {20,100} × 层 {input, hidden}。

预注册：audit/t2_rate_prereg_2026-11-14.md
采集与 T2e 相同（--drive-direct --retarget 60），唯一变量是读出特征：瞬时 0/1 → 指数滑动速率迹。
判定：任一层（尤其 hidden）闭环 median ≤0.40 → 时序编码成立；hidden ≤0.80 → 部分成立；<10% 变化 → 不成立。
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

D = ROOT / "data_b2" / "t2rate"
TAUS = (20, 100)
LAYERS = ("input", "hidden")
N_DUMP_EP, N_TRAIN_EP, STRIDE = 20, 14, 10
THRESH = 0.40
# T2e（瞬时 0/1 读出）的参照值，用于「复现前提」与相对变化
REF_SPIKE = {"input": 0.5637, "hidden": 1.1180}
Q = 255.0   # 速率导出量化刻度（uint8 ×255）


def collect(layer, tau, dump):
    subprocess.run([str(EXE), "--preset", "m2", "--disable_reward",
                    "--drive-direct", "--retarget", "60", "--avg-conn", "8",
                    "--dump-layer", layer, "--rate-tau", str(tau), "--dump-values", "rate",
                    "--episodes", str(N_DUMP_EP), "--eval-episodes", "0",
                    "--seed", "11", "--stride", str(STRIDE),
                    "--dump-hidden", str(dump), "--out", str(dump.with_suffix(".csv"))],
                   stdout=subprocess.DEVNULL)


def main():
    D.mkdir(parents=True, exist_ok=True)
    rows = []
    for tau in TAUS:
        for layer in LAYERS:
            t0 = time.time()
            dump = D / f"dump_{layer}_tau{tau}.bin"
            beta = D / f"beta_{layer}_tau{tau}.bin"
            print(f"\n=== tau={tau} layer={layer} ===", flush=True)
            if not dump.exists():
                collect(layer, tau, dump)

            Xq, meta, W, stride = read_dump(dump)
            X = Xq.astype(np.float32) / Q
            Y = meta[:, 1:3].astype(np.float64)
            eps = meta[:, 7].astype(int)
            tr, te = eps < N_TRAIN_EP, eps >= N_TRAIN_EP
            a, cvm = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
            b = ridge_dual(X[tr], Y[tr], a)
            mae = float(np.mean(np.abs(X[te] @ b - Y[te])))
            base = float(np.mean(np.abs(Y[te] - Y[tr].mean(axis=0))))
            np.ascontiguousarray(b.T.reshape(-1)).tofile(beta)

            med, arr, n, _ = run_eval(["--load-readout", str(beta),
                                       "--readout-layer", layer,
                                       "--rate-tau", str(tau), "--readout-values", "rate",
                                       "--avg-conn", "8"], D / f"eval_{layer}_tau{tau}.csv")
            ref = REF_SPIKE[layer]
            rows.append({"tau": tau, "layer": layer, "width": int(W),
                         "density": float(X.mean()), "nonzero_cols": int((X.max(0) > 0).sum()),
                         "alpha": a, "cv_mse": cvm, "mae": mae, "const_mae": base,
                         "mae_ratio": mae / base, "median": med, "arrived": arr, "n_eval": n,
                         "ref_spike": ref, "delta_vs_spike": med - ref,
                         "rel_change": (med - ref) / ref, "pass": bool(med <= THRESH),
                         "sec": round(time.time() - t0, 1)})
            print(f"  α={a:g} CV={cvm:.4f} MAE={mae:.4f}/{base:.4f}={mae/base:.2f} "
                  f"密度={X.mean():.4f} 非零列={rows[-1]['nonzero_cols']}/{W} "
                  f"→ median={med:.4f}（瞬时 {ref:.4f}, Δ={med-ref:+.4f}）到达 {arr}/{n}", flush=True)

    hdr = f"{'tau':>4}{'layer':>8}{'median':>9}{'Δvs瞬时':>10}{'rel':>8}{'MAE/base':>10}{'pass':>6}"
    print("\n" + hdr)
    for r in rows:
        print(f"{r['tau']:>4}{r['layer']:>8}{r['median']:>9.4f}{r['delta_vs_spike']:>+10.4f}"
              f"{r['rel_change']*100:>7.0f}%{r['mae_ratio']:>10.2f}{'Y' if r['pass'] else 'N':>6}")

    best = min(rows, key=lambda r: r["median"])
    hid = [r for r in rows if r["layer"] == "hidden"]
    best_hid = min(hid, key=lambda r: r["median"])
    if best["median"] <= THRESH:
        verdict = (f"时序/速率编码成立：{best['layer']}/τ={best['tau']} median {best['median']:.4f} ≤0.40"
                   + ("（hidden 也达标）" if best_hid["median"] <= THRESH else "（仅 input 达标）"))
    elif best_hid["median"] <= 0.80:
        verdict = (f"部分成立：hidden 由 {REF_SPIKE['hidden']:.4f} 降到 {best_hid['median']:.4f}"
                   f"（τ={best_hid['tau']}），仍 >0.40")
    elif abs(best_hid["rel_change"]) < 0.10:
        verdict = (f"不成立：hidden 变化 {best_hid['rel_change']*100:+.0f}% <10%，"
                   "单 tick 二值化不是瓶颈 → 该拓扑在 g=0 下不可行")
    else:
        verdict = (f"未达判定线：hidden 最佳 {best_hid['median']:.4f}（τ={best_hid['tau']}），"
                   f"相对改进 {best_hid['rel_change']*100:+.0f}%")
    print("\n判定 → " + verdict)

    (D / "t2rate_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"), "thresh": THRESH,
        "ref_spike": REF_SPIKE, "rows": rows, "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
