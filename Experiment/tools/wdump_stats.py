"""权重 / 命令轨迹统计（v16 工具缺口补丁，2026-11-17）。

用法：
  py tools/wdump_stats.py --weights data_b2/m4/w_ep320.txt
  py tools/wdump_stats.py --weights new.txt --ref data_b2/m2/w_ep0.txt   # 对比两次
  py tools/wdump_stats.py --trace data_b2/m4/cmd_s101.jsonl

权重视图：按层掩码分 I→H / H→H / H→M / I→M 四路，报 count/mean/sd/分位数/
WEIGHT_MAX 饱和比/Gini。纪律：只读，不训练、不改任何东西。
"""
import argparse
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
WEIGHT_MAX = 10.0
WEIGHT_MIN = 0.001


def load_weights(path):
    n = None
    pre, post, w = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        head = f.readline().split()
        # "# B2W <ver> <neurons> <synapses>"
        if len(head) >= 5 and head[1] == 'B2W':
            n = int(head[3])
        for line in f:
            p = line.split()
            if len(p) != 3:
                continue
            pre.append(int(p[0]))
            post.append(int(p[1]))
            w.append(float(p[2]))
    return n, np.array(pre, dtype=np.int64), np.array(post, dtype=np.int64), np.array(w, dtype=np.float64)


def gini(x):
    x = np.sort(x)
    n = x.size
    if n == 0 or x.sum() <= 0:
        return float('nan')
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def describe(name, w):
    if w.size == 0:
        return None
    q = np.percentile(w, [5, 50, 95])
    return {
        'path': name, 'n': int(w.size), 'mean': float(w.mean()), 'sd': float(w.std()),
        'min': float(w.min()), 'max': float(w.max()),
        'p5': float(q[0]), 'p50': float(q[1]), 'p95': float(q[2]),
        'sat_hi': float((w >= WEIGHT_MAX - 1e-9).mean()),
        'sat_lo': float((w <= WEIGHT_MIN + 1e-9).mean()),
        'gini': gini(w),
    }


def weights_report(path, ref=None, sizes=(1056, 3500, 500)):
    nin, nh, nm = sizes
    n, pre, post, w = load_weights(path)
    b_in, b_h, b_m = nin, nin + nh, nin + nh + nm
    is_in = (pre < b_in)
    is_h = (pre >= b_in) & (pre < b_h)
    is_m = (pre >= b_h)
    post_in = post < b_in
    post_h = (post >= b_in) & (post < b_h)
    post_m = post >= b_h

    paths = {
        'I->H': is_in & post_h,
        'H->H': is_h & post_h,
        'H->M': is_h & post_m,
        'I->M': is_in & post_m,
    }
    res = {'file': str(path), 'neurons': n, 'synapses': int(w.size), 'sizes': list(sizes)}
    rows = []
    for k, m in paths.items():
        rows.append(describe(k, w[m]))
    res['paths'] = rows
    res['all'] = describe('ALL', w)

    if ref:
        n2, pre2, post2, w2 = load_weights(ref)
        same = (w2.size == w.size) and np.array_equal(pre, pre2) and np.array_equal(post, post2)
        res['ref'] = str(ref)
        res['ref_comparable'] = bool(same)
        if same:
            d = w - w2
            res['delta'] = {
                'mean_abs': float(np.abs(d).mean()),
                'max_abs': float(np.abs(d).max()),
                'changed_frac': float((np.abs(d) > 1e-12).mean()),
                'paths': {k: {'dmean': float(d[m].mean()), 'dmax': float(np.abs(d[m]).max())}
                          for k, m in paths.items()},
            }
    return res


def print_weights(res):
    print(f"file: {res['file']}  neurons={res['neurons']} synapses={res['synapses']}  sizes={res['sizes']}")
    print(f"{'path':<6}{'n':>8}{'mean':>9}{'sd':>8}{'p5':>8}{'p50':>8}{'p95':>8}{'max':>8}{'satHi':>8}{'Gini':>7}")
    for r in res['paths']:
        if not r:
            continue
        print(f"{r['path']:<6}{r['n']:>8}{r['mean']:>9.4f}{r['sd']:>8.4f}{r['p5']:>8.4f}"
              f"{r['p50']:>8.4f}{r['p95']:>8.4f}{r['max']:>8.3f}{r['sat_hi']*100:>7.2f}%{r['gini']:>7.3f}")
    if res.get('ref_comparable'):
        d = res['delta']
        print(f"\nvs ref: mean|Δw|={d['mean_abs']:.5f} max|Δw|={d['max_abs']:.4f} "
              f"changed={d['changed_frac']*100:.1f}%")
        for k, v in d['paths'].items():
            print(f"  {k:<6} Δmean={v['dmean']:+.5f}  Δmax={v['dmax']:.4f}")


def trace_report(path):
    eps = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('{"header"'):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            eps.setdefault(r['ep'], []).append(r)
    out = []
    for ep in sorted(eps):
        rows = eps[ep]
        om1 = np.array([r['om1'] for r in rows])
        om2 = np.array([r['om2'] for r in rows])
        w1 = np.array([r['w1s'] for r in rows])
        w2 = np.array([r['w2s'] for r in rows])
        e1 = np.array([r['e1'] for r in rows])
        e2 = np.array([r['e2'] for r in rows])
        out.append({
            'ep': ep, 'samples': len(rows),
            'mean_abs_cmd_err1': float(np.abs(om1 - w1).mean()),
            'mean_abs_cmd_err2': float(np.abs(om2 - w2).mean()),
            'corr_om_w1': float(np.corrcoef(om1, w1)[0, 1]) if om1.std() > 0 and w1.std() > 0 else float('nan'),
            'corr_om_w2': float(np.corrcoef(om2, w2)[0, 1]) if om2.std() > 0 and w2.std() > 0 else float('nan'),
            'mean_abs_cf_err1': float(np.abs(e1).mean()),
            'mean_abs_cf_err2': float(np.abs(e2).mean()),
            'std_om1': float(om1.std()), 'std_om2': float(om2.std()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='')
    ap.add_argument('--ref', default='')
    ap.add_argument('--trace', default='')
    ap.add_argument('--sizes', default='1056,3500,500')
    ap.add_argument('--json', default='')
    ap.add_argument('--ep-filter', default='', help='只打印这些 episode，逗号分隔；留空=全部')
    args = ap.parse_args()

    payload = {}
    if args.weights:
        sizes = tuple(int(x) for x in args.sizes.split(','))
        r = weights_report(args.weights, args.ref or None, sizes)
        print_weights(r)
        payload['weights'] = r
    if args.trace:
        tr = trace_report(args.trace)
        print(f"\ntrace: {args.trace}  episodes={len(tr)}")
        print(f"{'ep':>4}{'|om-w*|1':>10}{'|om-w*|2':>10}{'corr1':>8}{'corr2':>8}{'|cf_e1|':>9}{'std_om1':>9}")
        keep = None
        if args.ep_filter:
            keep = {int(x) for x in args.ep_filter.split(',') if x.strip()}
        for r in tr:
            if keep is not None and r['ep'] not in keep:
                continue
            print(f"{r['ep']:>4}{r['mean_abs_cmd_err1']:>10.4f}{r['mean_abs_cmd_err2']:>10.4f}"
                  f"{r['corr_om_w1']:>8.3f}{r['corr_om_w2']:>8.3f}"
                  f"{r['mean_abs_cf_err1']:>9.4f}{r['std_om1']:>9.4f}")
        payload['trace'] = tr
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n[saved] {args.json}")


if __name__ == '__main__':
    main()
