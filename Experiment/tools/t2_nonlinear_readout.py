"""T2i · 非线性读出（开环）：线性 ridge / RFF-ReLU ridge / 小型 MLP（numpy）。

预注册：audit/t2_nonlinear_prereg_2026-11-14.md
数据：T2e 均衡采集（2400）与 T2f 第 1 轮聚合（4800）；层 input 1056 / hidden 3500。
判定：hidden 的非线性 MAE/基线 ≤0.50 → 有信息只是线性不可读；≥0.90 → 几乎无可用信息。
限制：核心外部读出只接受线性系数，故本实验**只有开环**指标（预注册 §2）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_decode_upper import ROOT, read_dump  # noqa: E402

BAL = ROOT / "data_b2" / "t2bal"
DAG = ROOT / "data_b2" / "t2dag"
D = ROOT / "data_b2" / "t2nl"
N_TRAIN_EP = 14
ALPHAS = (1e-2, 1e-1, 1.0, 10.0)      # 与 §6.23–§6.27 完全一致（保证线性基线可对齐）
ALPHAS_RF = (0.1, 1.0, 10.0, 100.0)    # RFF 特征尺度不同，单独给网格
RF_M = 512
RF_GAMMAS = (1.0, 3.0)
MLP_H = 32
MLP_EPOCH = 300
MLP_LR = 3e-3
MLP_BATCH = 256
PATIENCE = 30
SEED = 0


def load_set(name, layer):
    base = BAL / f"dump_{layer}.bin"
    X0, m0, W, _ = read_dump(base)
    if name == "A":
        return X0, m0, W
    r1 = DAG / f"dump_{layer}_r1.bin"
    X1, m1, _, _ = read_dump(r1)
    return np.vstack([X0, X1]), np.vstack([m0, m1]), W


# ---------------- 线性 ridge（对偶，α 由训练集子样本 3 折选） ----------------
def ridge_dual(X, Y, alpha):
    K = X @ X.T
    K[np.diag_indices_from(K)] += alpha
    return X.T @ np.linalg.solve(K, Y)


def cv_alpha(X, Y, eps, alphas):
    """在训练集内用 3 折（按 episode）选 α；为控制成本只用至多 900 个训练样本。"""
    ue = np.unique(eps)
    folds = [ue[i::3] for i in range(3)]
    best_a, best_m = alphas[0], float("inf")
    for a in alphas:
        mses = []
        for f in folds:
            te = np.isin(eps, f)
            tr = ~te
            if te.sum() < 20 or tr.sum() < 20:
                continue
            b = ridge_dual(X[tr], Y[tr], a)
            mses.append(float(np.mean((X[te] @ b - Y[te]) ** 2)))
        if mses and np.mean(mses) < best_m:
            best_a, best_m = a, float(np.mean(mses))
    return best_a, best_m


def evaluate(pred, Y):
    return float(np.mean(np.abs(pred - Y)))


# ---------------- RFF-ReLU + 岭回归（primal） ----------------
def rff_features(X, gamma, m, rng):
    d = X.shape[1]
    Wm = rng.normal(0.0, gamma / np.sqrt(d), size=(d, m))
    b = rng.normal(0.0, 0.1, size=m)
    return np.maximum(X @ Wm + b, 0.0)


def ridge_primal(H, Y, alpha):
    m = H.shape[1]
    A = H.T @ H + alpha * np.eye(m)
    return np.linalg.solve(A, H.T @ Y)


# ---------------- 小型 MLP（numpy 手写反向传播，Adam） ----------------
def train_mlp(Xtr, Ytr, Xva, Yva, rng):
    d = Xtr.shape[1]
    # 标准化：稀疏列（std 很小）必须给下限并截断，否则 (x-mu)/1e-6 会把 MLP 打爆
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 0.05)
    Xt = np.clip((Xtr - mu) / sd, -8.0, 8.0)
    Xv = np.clip((Xva - mu) / sd, -8.0, 8.0)
    # 目标也标准化（只影响优化稳定性，预测时再还原）
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Yt = (Ytr - ymu) / ysd
    Yv = (Yva - ymu) / ysd
    W1 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, MLP_H))
    b1 = np.zeros(MLP_H)
    W2 = rng.normal(0.0, 1.0 / np.sqrt(MLP_H), size=(MLP_H, 2))
    b2 = np.zeros(2)
    ps = [W1, b1, W2, b2]
    ms = [np.zeros_like(p) for p in ps]
    vs = [np.zeros_like(p) for p in ps]
    t = 0
    best = (float("inf"), [p.copy() for p in ps])
    bad = 0
    n = Xt.shape[0]
    for ep in range(MLP_EPOCH):
        order = rng.permutation(n)
        for i in range(0, n, MLP_BATCH):
            idx = order[i:i + MLP_BATCH]
            xb, yb = Xt[idx], Yt[idx]
            h = np.maximum(xb @ W1 + b1, 0.0)
            pred = h @ W2 + b2
            g = (pred - yb) * (2.0 / xb.shape[0])
            gW2 = h.T @ g
            gb2 = g.sum(0)
            gh = (g @ W2.T) * (h > 0)
            gW1 = xb.T @ gh
            gb1 = gh.sum(0)
            grads = [gW1, gb1, gW2, gb2]
            t += 1
            for k in range(4):
                ms[k] = 0.9 * ms[k] + 0.1 * grads[k]
                vs[k] = 0.999 * vs[k] + 0.001 * grads[k] ** 2
                mhat = ms[k] / (1 - 0.9 ** t)
                vhat = vs[k] / (1 - 0.999 ** t)
                ps[k] -= MLP_LR * mhat / (np.sqrt(vhat) + 1e-8)
            W1, b1, W2, b2 = ps
        hv = np.maximum(Xv @ W1 + b1, 0.0)
        vloss = float(np.mean((hv @ W2 + b2 - Yv) ** 2))
        if vloss < best[0] - 1e-9:
            best = (vloss, [p.copy() for p in ps])
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    W1, b1, W2, b2 = best[1]
    def predict(X):
        H = np.maximum(np.clip((X - mu) / sd, -8.0, 8.0) @ W1 + b1, 0.0)
        return (H @ W2 + b2) * ysd + ymu
    return predict, best[0]


def main():
    D.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for setname in ("A", "B"):
        for layer in ("input", "hidden"):
            t0 = time.time()
            X, meta, W = load_set(setname, layer)
            Y = meta[:, 1:3].astype(np.float64)
            eps = meta[:, 7].astype(int)
            tr, te = eps < N_TRAIN_EP, eps >= N_TRAIN_EP
            Xtr, Ytr = X[tr], Y[tr]
            base = float(np.mean(np.abs(Y[te] - Ytr.mean(axis=0))))
            tag = f"{setname}/{layer}"

            # 1) 线性
            sub = np.where(tr)[0][::4][:900]
            a, cvm = cv_alpha(X[sub], Y[sub], eps[sub], ALPHAS)
            bl = ridge_dual(Xtr, Ytr, a)
            mae_lin = evaluate(X[te] @ bl, Y[te])

            # 2) RFF-ReLU + 岭（γ、α 由训练集子样本 CV 选）
            best_rf = (float("inf"), None, None)
            for g in RF_GAMMAS:
                Hsub = rff_features(X[sub], g, RF_M, np.random.default_rng(SEED + 1))
                a2, _ = cv_alpha(Hsub, Y[sub], eps[sub], ALPHAS_RF)
                # 为保证子样本与全量特征一致，用同一随机矩阵重建特征
                d = X.shape[1]
                Wm = np.random.default_rng(SEED + 1).normal(0.0, g / np.sqrt(d), size=(d, RF_M))
                bb = np.random.default_rng(SEED + 2).normal(0.0, 0.1, size=RF_M)
                f = lambda Z, Wm=Wm, bb=bb: np.maximum(Z @ Wm + bb, 0.0)
                br = ridge_primal(f(Xtr), Ytr, a2)
                mae_rf = evaluate(f(X[te]) @ br, Y[te])
                if mae_rf < best_rf[0]:
                    best_rf = (mae_rf, g, a2)

            # 3) MLP（训练集内按 episode 分 80/20 早停）
            ue = np.unique(eps[tr])
            va_eps = ue[-max(2, len(ue) // 5):]
            mva = np.isin(eps[tr], va_eps)
            predf, vloss = train_mlp(Xtr[~mva], Ytr[~mva], Xtr[mva], Ytr[mva], rng)
            mae_nn = evaluate(predf(X[te]), Y[te])

            rows.append({"set": setname, "layer": layer, "width": int(W),
                         "n": int(X.shape[0]), "alpha_lin": a, "mae_lin": mae_lin,
                         "rff_gamma": best_rf[1], "rff_alpha": best_rf[2], "mae_rff": best_rf[0],
                         "mlp_val_mse": vloss, "mae_mlp": mae_nn,
                         "base": base,
                         "lin_ratio": mae_lin / base, "rff_ratio": best_rf[0] / base,
                         "mlp_ratio": mae_nn / base,
                         "sec": round(time.time() - t0, 1)})
            print(f"{tag:<14} n={X.shape[0]:<5} base={base:.4f} | lin {mae_lin:.4f} "
                  f"({mae_lin/base:.2f}) | rff {best_rf[0]:.4f} ({best_rf[0]/base:.2f}, "
                  f"γ={best_rf[1]:g}) | mlp {mae_nn:.4f} ({mae_nn/base:.2f}) "
                  f"| {rows[-1]['sec']}s", flush=True)

    # 判定（hidden 的非线性比值为准）
    hid = [r for r in rows if r["layer"] == "hidden"]
    best_hid = min(hid, key=lambda r: min(r["rff_ratio"], r["mlp_ratio"]))
    best_ratio = min(best_hid["rff_ratio"], best_hid["mlp_ratio"])
    if best_ratio <= 0.50:
        verdict = f"hidden 有信息、只是线性不可读（最佳非线性 MAE/基线 {best_ratio:.2f} ≤0.50）→ 修法改为可学习非线性解码头"
    elif best_ratio < 0.90:
        verdict = f"hidden 有限可读（最佳 {best_ratio:.2f}，0.50–0.80/0.90 区间）→ 部分信息，不足以支撑改架构"
    else:
        verdict = f"hidden 几乎无可用信息（最佳非线性 {best_ratio:.2f} ≥0.90）→ 转时序/速率编码"
    print("\n判定 → " + verdict)

    (D / "t2nl_summary.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "config": {"rf_m": RF_M, "rf_gammas": RF_GAMMAS, "mlp_h": MLP_H,
                   "mlp_epoch": MLP_EPOCH, "seed": SEED},
        "rows": rows, "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
