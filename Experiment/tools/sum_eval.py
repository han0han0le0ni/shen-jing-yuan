"""汇总冻结权重评测（eval_weights.ps1 的输出）。

用法：
  py tools/sum_eval.py --dir data_b2/m4/ho_eprop
  py tools/sum_eval.py --dir data_b2/m4/ho_eprop --label "e-prop"
口径：只用 is_eval==1 行（eval_weights.ps1 里 episodes==eval_episodes ⇒ 全部是 eval）。
medD = median(final_dist)，medR = median(dist_ratio)，arr = 到达轮数。
纪律：只读 CSV。
"""
import argparse
import csv
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    d = Path(args.dir) if Path(args.dir).is_absolute() else EXP / args.dir
    files = sorted(d.glob('*.csv'))
    if not files:
        print('no csv')
        return

    rows = []
    print(f"{'file':<40}{'n':>4}{'medD':>9}{'medR':>9}{'arr':>7}{'meanD':>9}")
    for f in files:
        with open(f, newline='', encoding='utf-8') as fh:
            rs = [r for r in csv.DictReader(fh) if r.get('is_eval') == '1']
        if not rs:
            continue
        dist = [float(r['final_dist']) for r in rs]
        rat = [float(r['dist_ratio']) for r in rs]
        arr = sum(1 for r in rs if int(r['arrived']) == 1)
        rows.append({'file': f.name, 'n': len(rs), 'medD': median(dist),
                     'medR': median(rat), 'arr': arr, 'meanD': float(np.mean(dist))})
        print(f"{f.name:<40}{len(rs):>4}{median(dist):>9.4f}{median(rat):>9.4f}"
              f"{arr:>4}/{len(rs):<2}{np.mean(dist):>9.4f}")

    md = np.array([r['medD'] for r in rows])
    ar = np.array([r['arr'] / r['n'] for r in rows])
    tag = args.label or d.name
    print(f"\n[{tag}] n_files={len(rows)}  medD={md.mean():.4f} ± "
          f"{(md.std(ddof=1) if len(md) > 1 else 0.0):.4f}  "
          f"≤0.40 = {(md <= 0.40).sum()}/{len(md)}  arr_rate={ar.mean():.3f}")

    if args.json:
        import json
        p = Path(args.json) if Path(args.json).is_absolute() else EXP / args.json
        p.write_text(json.dumps({'label': tag, 'rows': rows}, ensure_ascii=False, indent=2),
                     encoding='utf-8')
        print(f"[saved] {p}")


if __name__ == '__main__':
    main()
