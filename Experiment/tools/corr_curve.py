"""从 v16 命令轨迹里提取 corr(ω,ω*) 随训练的变化：逐 seed 峰值位置 + 分块均值曲线。

用法：py tools/corr_curve.py --dir data_b2/v16_8s [--block 20]
目的：检验 §6.41 的推论「corr 峰值在 ep 40–80」，并与早停实验结果（最优 e120）对照。
"""
import argparse
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent


def load_trace(path, want_stats=False):
    eps = {}
    stats = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('{"header"'):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        eps.setdefault(r['ep'], []).append(r)
    out = {}
    for ep, rows in eps.items():
        c1 = np.array([r['om1'] for r in rows])
        w1 = np.array([r['w1s'] for r in rows])
        if c1.std() > 0 and w1.std() > 0:
            out[ep] = float(np.corrcoef(c1, w1)[0, 1])
        else:
            out[ep] = float('nan')
        if want_stats:
            stats[ep] = (float(w1.std()), float(c1.std()))
    return (out, stats) if want_stats else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data_b2/v16_8s')
    ap.add_argument('--block', type=int, default=20)
    args = ap.parse_args()
    d = EXP / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)

    curves = {}
    stt = {}
    for f in sorted(d.glob('cmd_s*.jsonl')):
        seed = int(f.stem.split('_s')[1])
        curves[seed], stt[seed] = load_trace(f, want_stats=True)

    if not curves:
        print('no trace')
        return

    print(f"{'seed':>5}{'argmax_ep':>10}{'peak':>8}{'ep40-80均值':>12}{'末20轮':>9}")
    peaks = []
    for seed, c in curves.items():
        eps = sorted(c)
        valid = [(e, v) for e, v in c.items() if not np.isnan(v)]
        if not valid:
            continue
        am = max(valid, key=lambda x: x[1])
        mid = [v for e, v in valid if 40 <= e <= 80]
        tail = [v for e, v in valid if e >= eps[-20]]
        peaks.append((seed, am[0], am[1]))
        print(f"{seed:>5}{am[0]:>10}{am[1]:>8.3f}{np.mean(mid):>12.3f}{np.mean(tail):>9.3f}")

    print(f"\nargmax 分布：{sorted(e for _, e, _ in peaks)}")
    print(f"峰值中位数 ep = {sorted(e for _, e, _ in peaks)[len(peaks)//2]}")

    eps_all = sorted({e for c in curves.values() for e in c})
    print(f"\n{'ep':>5}{'mean corr':>11}{'std_w*':>9}{'std_om':>8}{'n':>4}")
    for e in eps_all:
        if (e + 1) % args.block != 0:
            continue
        vals = [c[e] for c in curves.values() if e in c and not np.isnan(c[e])]
        sw = [stt[s][e][0] for s in stt if e in stt[s]]
        so = [stt[s][e][1] for s in stt if e in stt[s]]
        if vals:
            print(f"{e:>5}{np.mean(vals):>11.3f}{np.mean(sw):>9.3f}{np.mean(so):>8.3f}{len(vals):>4}")


if __name__ == '__main__':
    main()
