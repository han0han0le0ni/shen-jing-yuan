"""T2q · K=1 闭环分布上的开环单步上界（预注册 audit/t2q_prereg_2026-11-16.md）。

做法：
1) 用 K=1 + error_gain=1 训练 320 ep，同时按 stride 导出 input 层发放 + 解析 ω*；
2) 训练 ep 280–299、测试 ep 300–319，ridge 回归 ω* ← input 特征（α 由训练集 3 折 CV）；
3) 报测试单步 MAE / 常数基线 MAE / MAE-基线比 / R²。

注意：这是**开环单步**上界，不是闭环上界；不做读出写回（训练后的权重没有落盘接口）。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_decode_upper import EXE, ROOT, cv_alpha, read_dump, ridge_dual  # noqa: E402

D = ROOT / "data_b2" / "t2q"
DUMP = D / "k1_g1_s101.bin"
CSV = D / "k1_g1_s101.csv"
STRIDE = 10
EPISODES = 320
TRAIN_EPS = (280, 300)   # [280, 300)
TEST_EPS = (300, 320)
ALPHAS = (1e-2, 1e-1, 1.0, 10.0)


def collect(seed=101):
    D.mkdir(parents=True, exist_ok=True)
    if DUMP.exists() and CSV.exists():
        print("[1/1] 复用已有采集", flush=True)
        return
    cmd = [str(EXE), "--preset", "m2", "--decode-mode", "wta", "--wta-k", "1",
           "--error-gain", "1", "--episodes", str(EPISODES), "--eval-episodes", "0",
           "--seed", str(seed), "--stride", str(STRIDE),
           "--dump-layer", "input", "--dump-hidden", str(DUMP), "--out", str(CSV)]
    print("[1/1] 采集：K=1,g=1 训练 320 ep 并导出 input 特征…", flush=True)
    subprocess.run(cmd, stdout=subprocess.DEVNULL)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(DUMP), help="要分析的 dump（默认 input 层采集）")
    ap.add_argument("--tag", default="input", help="标记，写入 json 文件名")
    args = ap.parse_args()
    if args.dump == str(DUMP):
        collect()
    X, meta, H, stride = read_dump(Path(args.dump))
    Y = meta[:, 1:3].astype(np.float64)
    eps = meta[:, 7].astype(int)
    print(f"      samples={X.shape[0]} H={H} stride={stride} 密度={X.mean():.4f} "
          f"episodes={len(np.unique(eps))}")

    tr = (eps >= TRAIN_EPS[0]) & (eps < TRAIN_EPS[1])
    te = (eps >= TEST_EPS[0]) & (eps < TEST_EPS[1])
    if tr.sum() < 50 or te.sum() < 50:
        raise SystemExit(f"样本不足: train={tr.sum()} test={te.sum()}")

    a, cv_mse = cv_alpha(X[tr], Y[tr], eps[tr], ALPHAS)
    beta = ridge_dual(X[tr], Y[tr], a)
    pred = X[te] @ beta
    mae = float(np.mean(np.abs(pred - Y[te])))
    mu = Y[tr].mean(axis=0)
    base = float(np.mean(np.abs(Y[te] - mu)))
    ratio = mae / base if base > 0 else float("inf")
    ss_tot = float(np.sum((Y[te] - mu) ** 2))
    ss_res = float(np.sum((Y[te] - pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 逐 episode 的 MAE，看分布是否集中在少数轮（注意 pred 只在 te 子集上）
    Yte = Y[te]
    eps_te = eps[te]
    per_ep = []
    for e in range(TEST_EPS[0], TEST_EPS[1]):
        m = eps_te == e
        if m.sum():
            per_ep.append(float(np.mean(np.abs(pred[m] - Yte[m]))))

    print(f"[2/4] ridge α={a:g}（CV MSE={cv_mse:.4f}）")
    print(f"[3/4] 测试单步 MAE={mae:.4f}  常数基线={base:.4f}  MAE/基线={ratio:.3f}  R²={r2:.4f}")
    print(f"      逐 episode MAE: min={min(per_ep):.4f} med={np.median(per_ep):.4f} max={max(per_ep):.4f}")

    flag = "表征够用（瓶颈在解码/学习）" if ratio <= 0.25 else (
        "部分（闭环分布更难，表征有损）" if ratio < 0.75 else "表征不足（需动输入/隐藏层）")
    print(f"[4/4] 判定：{flag}")

    (D / f"t2q_summary_{args.tag}.json").write_text(json.dumps({
        "samples": int(X.shape[0]), "H": int(H), "stride": int(stride),
        "train_eps": TRAIN_EPS, "test_eps": TEST_EPS, "alpha": a, "cv_mse": cv_mse,
        "test_mae": mae, "const_baseline_mae": base, "ratio": ratio, "r2": r2,
        "verdict": flag,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"用时 {time.time() - t0:.1f}s")
