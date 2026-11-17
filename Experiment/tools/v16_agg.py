"""v16 8-seed 汇总：训练后权重 Δ（分通路）+ 命令轨迹 corr(ω,ω*) + 闭环 median。

输入：data_b2/v16_8s/{w_init_s*.txt, w_final_s*.txt, cmd_s*.jsonl, s*.csv}
输出：stdout 表 + --json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wdump_stats import weights_report, trace_report  # noqa: E402

EXP = Path(__file__).resolve().parent.parent


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def read_csv_median(path):
    """与 tools/t2_arm_summary.py 完全同口径：
    medD = is_eval==1 行的 median(final_dist)，medR = median(dist_ratio)，arr = 到达数。"""
    import csv
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r.get('is_eval') == '1']
    if not rows:
        return float('nan'), float('nan'), 0, 0
    d = [float(r['final_dist']) for r in rows]
    rr = [float(r['dist_ratio']) for r in rows]
    arr = sum(1 for r in rows if int(r['arrived']) == 1)
    return median(d), median(rr), arr, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data_b2/v16_8s')
    ap.add_argument('--json', default='')
    args = ap.parse_args()
    d = EXP / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)

    rows = []
    for wi in sorted(d.glob('w_init_s*.txt')):
        seed = int(wi.stem.split('_s')[1])
        wf = d / f'w_final_s{seed}.txt'
        cmd = d / f'cmd_s{seed}.jsonl'
        csv = d / f's{seed}.csv'
        if not wf.exists():
            continue
        rep = weights_report(str(wf), str(wi))
        if not rep.get('ref_comparable'):
            print(f'[{seed}] 权重不可比（拓扑条数不同），跳过')
            continue
        r = {'seed': seed}
        if csv.exists():
            md, mr, na, nt = read_csv_median(csv)
            r.update({'medD': md, 'medR': mr, 'arr': na, 'n_eval': nt})
        else:
            r.update({'medD': float('nan'), 'medR': float('nan'), 'arr': 0, 'n_eval': 0})
        for p in rep['paths']:
            k = p['path']
            r[k + '_init_mean'] = None
            r[k + '_final_mean'] = p['mean']
            r[k + '_final_satHi'] = p['sat_hi']
            r[k + '_final_gini'] = p['gini']
            r[k + '_final_p5'] = p['p5']
        for k, v in rep['delta']['paths'].items():
            r[k + '_dmean'] = v['dmean']
        if cmd.exists():
            tr = trace_report(str(cmd))
            tail = [x for x in tr if x['ep'] >= max(0, len(tr) - 20)]
            r['corr1_tail'] = float(np.nanmean([x['corr_om_w1'] for x in tail]))
            r['corr2_tail'] = float(np.nanmean([x['corr_om_w2'] for x in tail]))
            r['corr1_peak'] = float(np.nanmax([x['corr_om_w1'] for x in tr]))
            r['std_om1_tail'] = float(np.nanmean([x['std_om1'] for x in tail]))
            r['abs_err1_tail'] = float(np.nanmean([x['mean_abs_cmd_err1'] for x in tail]))
        rows.append(r)

    if not rows:
        print('no data')
        return

    paths = ['I->H', 'H->H', 'H->M', 'I->M']
    print(f"{'seed':>5}{'medD':>8}{'medR':>8}{'arr':>5}" + ''.join(f'{p+" Δ":>11}' for p in paths)
          + ''.join(f'{p+" satHi":>12}' for p in ['H->H', 'I->M'])
          + f"{'corr1_tail':>12}{'corr1_peak':>12}")
    for r in rows:
        print(f"{r['seed']:>5}{r['medD']:>8.4f}{r['medR']:>8.4f}{r['arr']:>4}/{r['n_eval']:<2}"
              + ''.join(f"{r[p+'_dmean']:>11.4f}" for p in paths)
              + f"{r['H->H_final_satHi']*100:>11.2f}%{r['I->M_final_satHi']*100:>11.2f}%"
              + f"{r.get('corr1_tail', float('nan')):>12.3f}{r.get('corr1_peak', float('nan')):>12.3f}")

    print('\n== 跨 seed 均值 ± SD ==')
    keys = [p + '_dmean' for p in paths] + ['medD', 'medR', 'arr', 'corr1_tail', 'corr1_peak',
                                            'H->H_final_gini', 'H->M_final_p5', 'std_om1_tail']
    agg = {}
    for k in keys:
        v = [r[k] for r in rows if k in r and r[k] is not None]
        if not v:
            continue
        agg[k] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  'n': len(v), 'values': [float(x) for x in v]}
        print(f"  {k:<18} {agg[k]['mean']:+.4f} ± {agg[k]['sd']:.4f}  (n={len(v)})")

    print('\n== H->M 是否被压低（逐 seed 判据：Δmean < 0）==')
    neg = sum(1 for r in rows if r['H->M_dmean'] < 0)
    print(f"  {neg}/{len(rows)} 个 seed 的 H->M Δmean < 0")
    neg2 = sum(1 for r in rows if r['I->M_dmean'] > 0)
    print(f"  {neg2}/{len(rows)} 个 seed 的 I->M Δmean > 0")

    if args.json:
        Path(args.json).write_text(json.dumps({'rows': rows, 'agg': agg}, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
        print(f"\n[saved] {args.json}")


if __name__ == '__main__':
    main()
