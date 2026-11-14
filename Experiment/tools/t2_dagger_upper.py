"""T2f · DAgger 式 on-policy 聚合上界（input / hidden 各一遍，2 轮）。

预注册：audit/t2_dagger_prereg_2026-11-14.md
起点：T2e 的均衡采集（data_b2/t2bal）。每轮用当前读出闭环采集自己的状态，并入训练集重拟合。
判定：input 在 ≤2 轮内 median ≤0.40 → 瓶颈在闭环控制；否则 → 线性读出族不足。
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

BASE = ROOT / "data_b2" / "t2bal"
D = ROOT / "data_b2" / "t2dag"
LAYERS = ("input", "hidden")
ROUNDS = (1, 2)
N_EP = 20
N_TRAIN_EP = 14
STRIDE = 10
THRESH = 0.40


def collect(layer, beta, dump, csv):
    subprocess.run([str(EXE), "--preset", "m2", "--disable_reward",
                    "--avg-conn", "8", "--dump-layer", layer,
                    "--load-readout", str(beta), "--readout-layer", layer,
                    "--episodes", str(N_EP), "--eval-episodes", "0",
                    "--seed", "11", "--stride", str(STRIDE),
                    "--dump-hidden", str(dump), "--out", str(csv)],
                   stdout=subprocess.DEVNULL)


def stats(meta):
    th2 = meta[:, 4]
    return {"th2_std": float(th2.std()),
            "th1": [float(meta[:, 3].min()), float(meta[:, 3].max())],
            "th2": [float(th2.min()), float(th2.max())],
            "wmag": float(np.abs(meta[:, 1:3]).mean())}


def main():
    D.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in LAYERS:
        X0, m0, W, _ = read_dump(BASE / f"dump_{layer}.bin")
        beta = BASE / f"beta_{layer}.bin"
        print(f"\n=== layer={layer}（起点 {X0.shape[0]} 样本 × {W} 维）===", flush=True)

        for r in ROUNDS:
            t0 = time.time()
            dump = D / f"dump_{layer}_r{r}.bin"
            csv = D / f"dump_{layer}_r{r}.csv"
            print(f"  -- round {r}: 用 r{r-1} 读出闭环采集 …", flush=True)
            if not dump.exists():
                collect(layer, beta, dump, csv)
            Xr, mr, Wr, _ = read_dump(dump)
            sr = stats(mr)

            X = np.vstack([X0, Xr])
            meta = np.vstack([m0, mr])
            Y = meta[:, 1:3].astype(np.float64)
            eps = meta[:, 7].astype(int)
            tr, te = eps < N_TRAIN_EP, eps >= N_TRAIN_EP
            a, cv_mse = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
            b = ridge_dual(X[tr], Y[tr], a)
            mae = float(np.mean(np.abs(X[te] @ b - Y[te])))
            base = float(np.mean(np.abs(Y[te] - Y[tr].mean(axis=0))))
            beta = D / f"beta_{layer}_r{r}.bin"
            np.ascontiguousarray(b.astype(np.float64).T.reshape(-1)).tofile(beta)

            med, arr, n, _ = run_eval(["--load-readout", str(beta),
                                       "--readout-layer", layer, "--avg-conn", "8"],
                                      D / f"eval_{layer}_r{r}.csv")
            rows.append({"layer": layer, "round": r, "samples": int(X.shape[0]),
                         "new_samples": int(Xr.shape[0]), "alpha": a, "cv_mse": cv_mse,
                         "mae": mae, "const_mae": base, "mae_ratio": mae / base,
                         "median": med, "arrived": arr, "n_eval": n,
                         "pass": bool(med <= THRESH), "new_dump": sr,
                         "sec": round(time.time() - t0, 1)})
            print(f"     r{r}: 新增 {Xr.shape[0]}（θ2 std {sr['th2_std']:.3f}, |ω*| {sr['wmag']:.3f}）"
                  f" 聚合 {X.shape[0]} | α={a:g} MAE={mae:.4f}/{base:.4f}={mae/base:.2f}"
                  f" | median={med:.4f} 到达 {arr}/{n} {'PASS' if med <= THRESH else 'FAIL'}"
                  f" | {rows[-1]['sec']}s", flush=True)

    hdr = (f"{'layer':>8}{'round':>7}{'samples':>9}{'median':>9}{'arrived':>9}"
           f"{'MAE':>9}{'base':>9}{'ratio':>7}{'pass':>6}")
    print("\n" + hdr)
    for r in rows:
        print(f"{r['layer']:>8}{r['round']:>7}{r['samples']:>9}{r['median']:>9.4f}"
              f"{r['arrived']:>9}{r['mae']:>9.4f}{r['const_mae']:>9.4f}"
              f"{r['mae_ratio']:>7.2f}{'Y' if r['pass'] else 'N':>6}")

    inp = [r for r in rows if r["layer"] == "input"]
    hid = [r for r in rows if r["layer"] == "hidden"]
    best = min(inp, key=lambda r: r["median"])
    if best["pass"]:
        verdict = (f"分布漂移解释成立：input 第 {best['round']} 轮 median {best['median']:.4f} ≤0.40，"
                   "瓶颈在闭环控制（无积分项/量化/1 tick 延迟），不是表征")
    else:
        verdict = ("线性读出族不足：input 2 轮聚合后仍 >0.40，"
                   "下一步上非线性读出（把「读出族」与「表征」分开）")
    gap0 = inp[0]["median"] - hid[0]["median"]
    gap2 = inp[-1]["median"] - hid[-1]["median"]
    if abs(gap2) < 0.15 < abs(gap0):
        verdict += "；且 input-hidden 差距收窄 → hidden 缺的只是分布"
    elif abs(gap2) >= 0.15:
        verdict += "；input-hidden 差距未收窄 → hidden 的信息损失是结构性的"
    print(f"\n判定 → {verdict}")

    (D / "t2dag_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"), "thresh": THRESH,
        "base": "data_b2/t2bal (drive-direct + retarget 60)",
        "rounds": len(ROUNDS), "verdict": verdict, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if best["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
