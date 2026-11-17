"""回答 §6.42 的未解问题：命令与 ω* 全程不相关，为什么 medR 还从 1.0 降到 ~0.5？

做法（纯离线，读 --trace-cmd 的 JSONL）：
  1. 由 (θ1,θ2) 解析求末端位置与 ∂dist/∂θ（L1=L2=1.0）；
  2. 计算「按命令走，距离的预期变化率」 pred = −∇dist·ω；
  3. 与实测 Δdist（相邻采样）比：
       - corr(Δdist, pred)  > 0  ⇒ 命令确实在把末端推向目标（存在方向性控制，只是与 ω* 无关）
       - corr(Δdist, |ω|)    ≈ 0 且 mean(Δdist) < 0 ⇒ 距离下降来自系统性偏置/漂移，而非指令方向
  4. 同时给 corr(ω, ω*) 作对照。

用法：py tools/cmd_regress.py --dir data_b2/v16_8s [--tail 20]
"""
import argparse
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
L1 = L2 = 1.0


def fk(t1, t2):
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return x, y


def grad_dist(t1, t2, ex, ey, tx, ty):
    """∂dist/∂θ1, ∂dist/∂θ2（解析雅可比 + 链式）"""
    d = np.hypot(ex - tx, ey - ty)
    d = np.maximum(d, 1e-9)
    dx1 = -L1 * np.sin(t1) - L2 * np.sin(t1 + t2)
    dy1 = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    dx2 = -L2 * np.sin(t1 + t2)
    dy2 = L2 * np.cos(t1 + t2)
    return ((ex - tx) * dx1 + (ey - ty) * dy1) / d, ((ex - tx) * dx2 + (ey - ty) * dy2) / d


def load(path):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('{"header"'):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def per_episode(rows):
    by = {}
    for r in rows:
        by.setdefault(r['ep'], []).append(r)
    out = {}
    for ep, rs in by.items():
        rs.sort(key=lambda r: r['t'])
        t1 = np.array([r['th1'] for r in rs])
        t2 = np.array([r['th2'] for r in rs])
        ex = np.array([r['ex'] for r in rs])
        ey = np.array([r['ey'] for r in rs])
        tx = np.array([r['tx'] for r in rs])
        ty = np.array([r['ty'] for r in rs])
        w1 = np.array([r['om1'] for r in rs])
        w2 = np.array([r['om2'] for r in rs])
        s1 = np.array([r['w1s'] for r in rs])
        s2 = np.array([r['w2s'] for r in rs])

        dist = np.hypot(ex - tx, ey - ty)
        g1, g2 = grad_dist(t1, t2, ex, ey, tx, ty)
        pred = -(g1 * w1 + g2 * w2)          # 预期 d(dist)/dt（负=靠近）
        ddist = np.diff(dist)
        pred_mid = pred[:-1]

        # 用解析 FK 校核 trace 里的 ex/ey
        fx, fy = fk(t1, t2)
        fk_err = float(np.max(np.abs(fx - ex)) + np.max(np.abs(fy - ey)))

        def corr(a, b):
            if a.std() <= 0 or b.std() <= 0:
                return float('nan')
            return float(np.corrcoef(a, b)[0, 1])

        out[ep] = {
            'n': len(rs),
            'fk_err': fk_err,
            'corr_ddist_pred': corr(ddist, pred_mid),
            'corr_ddist_absom': corr(ddist, np.abs(w1)[:-1]),
            'mean_ddist': float(ddist.mean()),
            'mean_pred': float(pred.mean()),
            'corr_om_star': corr(w1, s1),
            'dist_start': float(dist[0]),
            'dist_end': float(dist[-1]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data_b2/v16_8s')
    ap.add_argument('--tail', type=int, default=20)
    args = ap.parse_args()
    d = EXP / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)

    print(f"{'seed':>5}{'fk_err':>9}{'corr(dd,pred)':>15}{'corr(dd,|w|)':>14}"
          f"{'mean Δd':>10}{'mean pred':>11}{'corr(w,w*)':>12}{'Δd 降/升':>11}")
    agg = []
    for f in sorted(d.glob('cmd_s*.jsonl')):
        seed = int(f.stem.split('_s')[1])
        pe = per_episode(load(f))
        eps = sorted(pe)
        tail = [pe[e] for e in eps if e >= eps[-args.tail]]
        g = lambda k: float(np.nanmean([x[k] for x in tail]))
        down = sum(1 for x in tail if x['dist_end'] < x['dist_start'])
        print(f"{seed:>5}{max(x['fk_err'] for x in tail):>9.3f}{g('corr_ddist_pred'):>15.3f}"
              f"{g('corr_ddist_absom'):>14.3f}{g('mean_ddist'):>10.4f}{g('mean_pred'):>11.4f}"
              f"{g('corr_om_star'):>12.3f}{down:>6}/{len(tail):<4}")
        agg.append({k: g(k) for k in ('corr_ddist_pred', 'corr_ddist_absom', 'mean_ddist',
                                      'mean_pred', 'corr_om_star')})

    print('\n== 跨 seed 均值 ± SD ==')
    for k in agg[0]:
        v = [a[k] for a in agg]
        print(f"  {k:<18} {np.mean(v):+.4f} ± {np.std(v, ddof=1):.4f}")


if __name__ == '__main__':
    main()
