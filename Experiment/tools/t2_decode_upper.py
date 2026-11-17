"""T2 · 解码可学性上界（判据 §2 门槛作参考线，不改变 B2 状态）。

流程：
1) 用 benchmark2.exe --preset m2 --dump-hidden 采集随机初始化网络的 hidden 发放 + 解析 ω*；
2) 按 episode 划分训练/测试，ridge 回归 ω* ← hidden（α 由训练集 3 折 CV 选）；
3) 把 β 写回核心作为外部线性读出，闭环跑测试目标，统计 median(dist_ratio)；
4) 与参照 B（--drive-direct，直接用 ω*）对照。

预注册：audit/t2_decode_prereg_2026-11-14.md
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "benchmark2.exe"
D = ROOT / "data_b2" / "t2"
DUMP = D / "dump.bin"
BETA = D / "beta.bin"
EVAL_CSV = D / "eval.csv"
REF_CSV = D / "ref.csv"
N_DUMP_EP = 20
N_TRAIN_EP = 14
STRIDE = 10
ALPHAS = (1e-2, 1e-1, 1.0, 10.0)


def read_dump(path):
    raw = np.fromfile(path, dtype=np.uint8)
    hdr = np.frombuffer(raw[:12].tobytes(), dtype=np.int32)
    magic, H, stride = int(hdr[0]), int(hdr[1]), int(hdr[2])
    if magic != 0x42325432:
        raise SystemExit(f"bad magic {magic:#x}")
    rec = H + 64
    n = (len(raw) - 12) // rec
    body = raw[12:12 + n * rec].reshape(n, rec)
    X = np.ascontiguousarray(body[:, :H]).astype(np.float32)
    meta = np.frombuffer(np.ascontiguousarray(body[:, H:]).tobytes(),
                         dtype=np.float64).reshape(n, 8)
    return X, meta, H, stride


def ridge_dual(X, Y, alpha):
    """β = Xᵀ(XXᵀ+αI)⁻¹Y，返回 (d, k)。"""
    K = X @ X.T
    K[np.diag_indices_from(K)] += alpha
    A = np.linalg.solve(K.astype(np.float64), Y.astype(np.float64))
    return (X.T @ A).astype(np.float64)


def cv_alpha(X, Y, eps, alphas):
    """按 episode 分 3 折，选测试 MSE 最小的 α。"""
    ue = np.unique(eps)
    folds = [ue[i::3] for i in range(3)]
    best, best_a = float("inf"), alphas[0]
    for a in alphas:
        mse = []
        for f in folds:
            te = np.isin(eps, f)
            tr = ~te
            if te.sum() < 10 or tr.sum() < 10:
                continue
            b = ridge_dual(X[tr], Y[tr], a)
            pred = X[te] @ b
            mse.append(float(np.mean((pred - Y[te]) ** 2)))
        if mse and np.mean(mse) < best:
            best, best_a = float(np.mean(mse)), a
    return best_a, best


def run_eval(extra, out_csv, episodes=30):
    cmd = [str(EXE), "--preset", "m2", "--seed", "999",
           "--episodes", str(episodes), "--eval-episodes", str(episodes),
           "--out", str(out_csv)] + extra
    # 注意：benchmark2.exe 用退出码回报达标与否（0=PASS / 2=FAIL），不是错误码
    subprocess.run(cmd, capture_output=True)   # 不按文本解码：exe 输出是 GBK
    if not out_csv.exists():
        raise SystemExit(f"run produced no csv: {' '.join(cmd)}")
    rows = []
    for ln in out_csv.read_text(encoding="utf-8").splitlines()[1:]:
        p = ln.split(",")
        if len(p) < 10:
            continue
        rows.append({"init": float(p[3]), "ratio": float(p[5]),
                     "arrived": int(p[8]), "eval": int(p[10]) if len(p) > 10 else 0})
    ev = [x for x in rows if x["eval"] == 1 and x["init"] >= 0.10]
    ratios = sorted(x["ratio"] for x in ev)
    n = len(ratios)
    med = ratios[n // 2] if n % 2 else 0.5 * (ratios[n // 2 - 1] + ratios[n // 2])
    return med, sum(x["arrived"] for x in ev), n, ratios


def main():
    D.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/4] 采集 hidden 发放（随机初始化，禁用奖励）…", flush=True)
    if not DUMP.exists():
        subprocess.run([str(EXE), "--preset", "m2", "--disable_reward",
                        "--episodes", str(N_DUMP_EP), "--eval-episodes", "0",
                        "--seed", "11", "--stride", str(STRIDE),
                        "--dump-hidden", str(DUMP),
                        "--out", str(D / "dump.csv")],
                       stdout=subprocess.DEVNULL)

    X, meta, H, stride = read_dump(DUMP)
    Y = meta[:, 1:3].astype(np.float64)
    eps = meta[:, 7].astype(int)
    print(f"      samples={X.shape[0]} hidden={H} stride={stride} "
          f"episodes={len(np.unique(eps))} 密度={X.mean():.4f}")

    tr = eps < N_TRAIN_EP
    a, cv_mse = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
    beta = ridge_dual(X[tr], Y[tr], a)
    print(f"[2/4] ridge α={a:g}（CV MSE={cv_mse:.4f}）→ β {beta.shape}")

    # 测试集（留出 episode）上的单步动作误差，仅作诊断
    te = ~tr
    pred = X[te] @ beta
    mae = float(np.mean(np.abs(pred - Y[te])))
    base_mae = float(np.mean(np.abs(Y[te] - Y[tr].mean(axis=0))))
    print(f"      留出 episode 单步 MAE={mae:.4f}（常数基线 {base_mae:.4f}）")

    buf = np.ascontiguousarray(beta.astype(np.float64).T.reshape(-1))
    buf.tofile(BETA)
    print(f"[3/4] β 写回 -> {BETA} ({buf.size} doubles)")

    med_a, arr_a, n_a, _ = run_eval(["--load-readout", str(BETA)], EVAL_CSV)
    print(f"[4/4] 上界 A（线性读出，闭环）median={med_a:.4f} "
          f"到达 {arr_a}/{n_a}", flush=True)

    med_b, arr_b, n_b, _ = run_eval(["--drive-direct"], REF_CSV, episodes=10)
    print(f"      参照 B（直接 ω*）median={med_b:.4f} 到达 {arr_b}/{n_b}")

    ok_link = med_b <= 0.10
    ok_a = med_a <= 0.40
    verdict = ("作废（链路未通过）" if not ok_link else
               ("表征够用：瓶颈在学习信号" if ok_a else "表征不足：需重设计拓扑/输入"))
    print(f"\n判定：链路={'OK' if ok_link else 'FAIL'}  上界A={'PASS' if ok_a else 'FAIL'} → {verdict}")
    print(f"用时 {time.time() - t0:.1f}s")

    (D / "t2_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "samples": int(X.shape[0]), "hidden": int(H), "stride": int(stride),
        "alpha": a, "cv_mse": cv_mse, "test_mae": mae, "const_baseline_mae": base_mae,
        "upper_median": med_a, "upper_arrived": arr_a, "upper_n": n_a,
        "ref_median": med_b, "ref_arrived": arr_b, "ref_n": n_b,
        "link_ok": ok_link, "upper_ok": ok_a, "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok_a else 1


if __name__ == "__main__":
    sys.exit(main())
