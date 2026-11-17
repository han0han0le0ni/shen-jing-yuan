"""T2d · 覆盖修正后的读出上界（input / hidden 各一遍）。

预注册：audit/t2_coverage_prereg_2026-11-14.md
背景：T2/T2b/T2c 的采集用「随机初始化、不训练」的网络驱动手臂，轨迹退化成一维窄带
（θ2≈0、θ1 单侧），导致上界被「采集分布退化 + 闭环漂移」混杂。
本脚本把采集改为 --drive-direct（解析 ω* 驱动，覆盖任务空间），其余协议不变。
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

D = ROOT / "data_b2" / "t2cov"
LAYERS = (("input", 1), ("hidden", 0))
N_DUMP_EP = 20
N_TRAIN_EP = 14
STRIDE = 10
THRESH = 0.40
GRID = 8
GRID_HALF = 2.2


def coverage(meta):
    th1, th2, ex, ey = meta[:, 3], meta[:, 4], meta[:, 5], meta[:, 6]
    cells = set()
    for x, y in zip(ex, ey):
        gx = int((x + GRID_HALF) / (2 * GRID_HALF) * GRID)
        gy = int((y + GRID_HALF) / (2 * GRID_HALF) * GRID)
        if 0 <= gx < GRID and 0 <= gy < GRID:
            cells.add((gx, gy))
    return {
        "th1": [float(th1.min()), float(th1.max())],
        "th2": [float(th2.min()), float(th2.max())],
        "th2_std": float(th2.std()),
        "ex": [float(ex.min()), float(ex.max())],
        "ey": [float(ey.min()), float(ey.max())],
        "grid_cover": len(cells) / (GRID * GRID),
    }


def run_layer(name, src):
    t0 = time.time()
    dump = D / f"dump_{name}.bin"
    beta = D / f"beta_{name}.bin"
    ev = D / f"eval_{name}.csv"
    dump_csv = D / f"dump_{name}.csv"

    print(f"\n=== layer={name} ===", flush=True)
    if not dump.exists():
        subprocess.run([str(EXE), "--preset", "m2", "--disable_reward", "--drive-direct",
                        "--avg-conn", "8", "--dump-layer", name,
                        "--episodes", str(N_DUMP_EP), "--eval-episodes", "0",
                        "--seed", "11", "--stride", str(STRIDE),
                        "--dump-hidden", str(dump), "--out", str(dump_csv)],
                       stdout=subprocess.DEVNULL)

    X, meta, H, stride = read_dump(dump)
    cov = coverage(meta)
    Y = meta[:, 1:3].astype(np.float64)
    eps = meta[:, 7].astype(int)
    print(f"  samples={X.shape[0]} width={H} 密度={X.mean():.4f} "
          f"活跃列={int(X.max(axis=0).sum())}/{H}", flush=True)
    print(f"  覆盖 θ1[{cov['th1'][0]:.3f},{cov['th1'][1]:.3f}] "
          f"θ2[{cov['th2'][0]:.3f},{cov['th2'][1]:.3f}] (std {cov['th2_std']:.3f}) "
          f"网格 {cov['grid_cover']*100:.0f}%", flush=True)

    tr = eps < N_TRAIN_EP
    te = ~tr
    a, cv_mse = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
    b = ridge_dual(X[tr], Y[tr], a)
    mae = float(np.mean(np.abs(X[te] @ b - Y[te])))
    base = float(np.mean(np.abs(Y[te] - Y[tr].mean(axis=0))))
    np.ascontiguousarray(b.astype(np.float64).T.reshape(-1)).tofile(beta)

    med, arr, n, _ = run_eval(["--load-readout", str(beta), "--readout-layer", name,
                               "--avg-conn", "8"], ev)
    row = {"layer": name, "width": int(H), "samples": int(X.shape[0]),
           "density": float(X.mean()), "alpha": a, "cv_mse": cv_mse,
           "mae": mae, "const_mae": base, "median": med, "arrived": arr,
           "n_eval": n, "pass": bool(med <= THRESH), "coverage": cov,
           "sec": round(time.time() - t0, 1)}
    print(f"  ridge α={a:g} CV={cv_mse:.4f} 留出 MAE={mae:.4f}（基线 {base:.4f}）"
          f" → 闭环 median={med:.4f} 到达 {arr}/{n} {'PASS' if row['pass'] else 'FAIL'}",
          flush=True)
    return row


def main():
    D.mkdir(parents=True, exist_ok=True)
    rows = [run_layer(n, s) for n, s in LAYERS]
    rin, rhid = rows[0], rows[1]

    invalid = (rin["coverage"]["th2_std"] < 0.10 or rhid["coverage"]["grid_cover"] < 0.30)
    if invalid:
        verdict = "作废：修正采集后覆盖仍不足（θ2 std<0.10 或网格覆盖<30%）"
    elif rin["pass"] and rhid["pass"]:
        verdict = ("信息与线性可读性都够：瓶颈在闭环控制/学习信号；"
                   "§6.23/§6.24 的「表征不足」归因撤回（M2 仍 FAIL，根因改写）")
    elif rin["pass"] and not rhid["pass"]:
        verdict = "hidden 破坏了输入里已有的信息：修拓扑（直通/残差、换 hidden 编码），不换输入编码"
    elif not rin["pass"] and rhid["pass"]:
        verdict = "矛盾组合（input 失败而 hidden 通过）：先复核数据，不下结论"
    else:
        verdict = "编码/线性读出族都不够：换编码（更高维/更密/时间编码）或上非线性读出"

    hdr = f"{'layer':>8}{'median':>9}{'arrived':>9}{'MAE':>9}{'base':>9}{'grid%':>7}{'pass':>6}{'sec':>7}"
    print("\n" + hdr)
    for r in rows:
        print(f"{r['layer']:>8}{r['median']:>9.4f}{r['arrived']:>9}{r['mae']:>9.4f}"
              f"{r['const_mae']:>9.4f}{r['coverage']['grid_cover']*100:>7.0f}"
              f"{'Y' if r['pass'] else 'N':>6}{r['sec']:>7}")
    print(f"\n判定 → {verdict}")

    (D / "t2cov_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "thresh": THRESH, "collection": "drive-direct", "invalid": bool(invalid),
        "verdict": verdict, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
