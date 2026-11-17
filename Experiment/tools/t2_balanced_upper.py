"""T2e · 均衡覆盖采集后的读出上界（input / hidden）。

预注册：audit/t2_balanced_prereg_2026-11-14.md
相对 T2d 只改采集：--drive-direct --retarget 60（每 60 tick 换新目标），
避免手臂到点后长期停住导致「绝大多数样本 ω*≈0」的退化标签分布。
先做前置有效性检查（基线 MAE、|ω*|、覆盖率），不达标即作废、不做判定。
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

D = ROOT / "data_b2" / "t2bal"
LAYERS = (("input", 1), ("hidden", 0))
N_DUMP_EP = 20
N_TRAIN_EP = 14
STRIDE = 10
RETARGET = 60
THRESH = 0.40
GRID, GRID_HALF = 8, 2.2
MIN_BASE, MIN_WMAG, MIN_TH2_STD, MIN_GRID = 0.20, 0.15, 0.10, 0.30


def coverage(meta):
    th1, th2, ex, ey = meta[:, 3], meta[:, 4], meta[:, 5], meta[:, 6]
    cells = set()
    for x, y in zip(ex, ey):
        gx = int((x + GRID_HALF) / (2 * GRID_HALF) * GRID)
        gy = int((y + GRID_HALF) / (2 * GRID_HALF) * GRID)
        if 0 <= gx < GRID and 0 <= gy < GRID:
            cells.add((gx, gy))
    return {"th1": [float(th1.min()), float(th1.max())],
            "th2": [float(th2.min()), float(th2.max())], "th2_std": float(th2.std()),
            "ex": [float(ex.min()), float(ex.max())], "ey": [float(ey.min()), float(ey.max())],
            "grid_cover": len(cells) / (GRID * GRID)}


def run_layer(name):
    t0 = time.time()
    dump = D / f"dump_{name}.bin"
    beta = D / f"beta_{name}.bin"
    ev = D / f"eval_{name}.csv"
    dump_csv = D / f"dump_{name}.csv"

    print(f"\n=== layer={name} ===", flush=True)
    if not dump.exists():
        subprocess.run([str(EXE), "--preset", "m2", "--disable_reward", "--drive-direct",
                        "--retarget", str(RETARGET), "--avg-conn", "8", "--dump-layer", name,
                        "--episodes", str(N_DUMP_EP), "--eval-episodes", "0",
                        "--seed", "11", "--stride", str(STRIDE),
                        "--dump-hidden", str(dump), "--out", str(dump_csv)],
                       stdout=subprocess.DEVNULL)

    X, meta, H, stride = read_dump(dump)
    cov = coverage(meta)
    Y = meta[:, 1:3].astype(np.float64)
    eps = meta[:, 7].astype(int)
    tr, te = eps < N_TRAIN_EP, eps >= N_TRAIN_EP
    base = float(np.mean(np.abs(Y[te] - Y[tr].mean(axis=0))))
    wmag = float(np.abs(Y).mean())
    print(f"  samples={X.shape[0]} width={H} 密度={X.mean():.4f} |ω*|={wmag:.3f} 基线 MAE={base:.4f}")
    print(f"  覆盖 θ1[{cov['th1'][0]:.2f},{cov['th1'][1]:.2f}] θ2[{cov['th2'][0]:.2f},{cov['th2'][1]:.2f}]"
          f" (std {cov['th2_std']:.2f}) 网格 {cov['grid_cover']*100:.0f}%", flush=True)

    a, cv_mse = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
    b = ridge_dual(X[tr], Y[tr], a)
    mae = float(np.mean(np.abs(X[te] @ b - Y[te])))
    np.ascontiguousarray(b.astype(np.float64).T.reshape(-1)).tofile(beta)
    med, arr, n, _ = run_eval(["--load-readout", str(beta), "--readout-layer", name,
                               "--avg-conn", "8"], ev)
    print(f"  α={a:g} CV={cv_mse:.4f} 留出 MAE={mae:.4f}（基线 {base:.4f}，比值 {mae/base:.2f}）"
          f" → median={med:.4f} 到达 {arr}/{n} {'PASS' if med <= THRESH else 'FAIL'}", flush=True)
    return {"layer": name, "width": int(H), "samples": int(X.shape[0]),
            "density": float(X.mean()), "wmag": wmag, "alpha": a, "cv_mse": cv_mse,
            "mae": mae, "const_mae": base, "mae_ratio": mae / base,
            "median": med, "arrived": arr, "n_eval": n,
            "pass": bool(med <= THRESH), "coverage": cov, "sec": round(time.time() - t0, 1)}


def main():
    D.mkdir(parents=True, exist_ok=True)
    rows = [run_layer(n) for n, _ in LAYERS]
    rin, rhid = rows[0], rows[1]
    checks = {
        "base_mae": rin["const_mae"] >= MIN_BASE,
        "wmag": rin["wmag"] >= MIN_WMAG,
        "th2_std": rin["coverage"]["th2_std"] >= MIN_TH2_STD,
        "grid": rin["coverage"]["grid_cover"] >= MIN_GRID,
    }
    valid = all(checks.values())
    if not valid:
        verdict = "作废：均衡采集后前置有效性检查未通过（" + \
                  ", ".join(k for k, v in checks.items() if not v) + "）"
    elif rin["pass"] and rhid["pass"]:
        verdict = ("信息与线性可读性都够：瓶颈在闭环控制/学习信号；"
                   "§6.23/§6.24 的「表征不足」归因撤回（M2 仍 FAIL，根因改写）")
    elif rin["pass"] and not rhid["pass"]:
        verdict = "hidden 破坏了输入里已有的信息：修拓扑（直通/残差、换 hidden 编码），不换输入编码"
    elif not rin["pass"] and rhid["pass"]:
        verdict = "矛盾组合（input 失败而 hidden 通过）：先复核数据，不下结论"
    else:
        verdict = "编码/线性读出族都不够：换编码（更高维/更密/时间编码）或上非线性读出"

    print("\n" + f"{'layer':>8}{'median':>9}{'arrived':>9}{'MAE':>9}{'base':>9}"
                 f"{'MAE/base':>10}{'grid%':>7}{'pass':>6}")
    for r in rows:
        print(f"{r['layer']:>8}{r['median']:>9.4f}{r['arrived']:>9}{r['mae']:>9.4f}"
              f"{r['const_mae']:>9.4f}{r['mae_ratio']:>10.2f}"
              f"{r['coverage']['grid_cover']*100:>7.0f}{'Y' if r['pass'] else 'N':>6}")
    print(f"\n前置检查 {checks}  有效={valid}")
    print(f"判定 → {verdict}")

    (D / "t2bal_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"), "thresh": THRESH,
        "collection": {"drive": "direct", "retarget": RETARGET},
        "checks": checks, "valid": bool(valid), "verdict": verdict, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if valid else 2


if __name__ == "__main__":
    sys.exit(main())
