"""M4 三臂 + 消融 + held-out 的一表汇总（只读，不训练）。

用法：
  py tools/m4_agg.py
  py tools/m4_agg.py --json data_b2/m4/m4_agg.json

数据来源（全部已落盘）：
  训练内闭环：data_b2/v16_8s/s*.csv            [A]  R-STDP, error_gain=1
              data_b2/m4/rstdp_eg0/base_*.csv  [ctrl] R-STDP, error_gain=0
              data_b2/m4/ep_8s/ep_*.csv        [B]  e-prop 全通路, lr=1e-5
              data_b2/m4/scope_im/ep_*.csv     [D]  e-prop 仅 I->M
  held-out ：data_b2/m4/ho_base, ho_eprop, ho_im（固定四目标，权重冻结）
  命令对齐：data_b2/m4/cmd_align_base8s.json, cmd_align_ep8s.json
口径与 tools/v16_agg.py 一致：medD = is_eval==1 行的 median(final_dist)。
纪律：只读；阈值与判据不在本脚本内做任何裁决。
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def arm_stats(pattern):
    """pattern 可为单个 glob，或 glob 列表（用于把补跑 seed 与预注册样本合并）。"""
    pats = [pattern] if isinstance(pattern, str) else list(pattern)
    meds, arrs, ns = [], [], []
    files = []
    for pat in pats:
        files.extend(sorted(glob.glob(str(EXP / pat))))
    for p in sorted(set(files)):
        with open(p, newline='', encoding='utf-8') as f:
            rs = [r for r in csv.DictReader(f) if r.get('is_eval') == '1']
        if not rs:
            continue
        meds.append(med([float(r['final_dist']) for r in rs]))
        arrs.append(sum(1 for r in rs if int(r['arrived']) == 1) / len(rs))
        ns.append(len(rs))
    if not meds:
        return None
    m = np.array(meds)
    return {
        'n_seed': len(m),
        'medD_mean': float(m.mean()),
        'medD_sd': float(m.std(ddof=1)) if len(m) > 1 else 0.0,
        'medD_per_seed': [float(x) for x in m],
        'pass_040': int((m <= 0.40).sum()),
        'arr_rate': float(np.mean(arrs)),
        'n_eval_per_seed': max(ns),
    }


ARMS = [
    ('A    R-STDP  eg=1', ['data_b2/v16_8s/s*.csv', 'data_b2/m4/seed_extra_base/base_*.csv'],
     'data_b2/m4/ho_base/*.csv', 'cmd_align_base8s.json'),
    ('ctrl R-STDP  eg=0', 'data_b2/m4/rstdp_eg0/base_*.csv', None, None),
    ('B    eprop all lr=1e-5', ['data_b2/m4/ep_8s/ep_*.csv', 'data_b2/m4/seed_extra/ep_*.csv'],
     'data_b2/m4/ho_eprop/*.csv', 'cmd_align_ep8s.json'),
    ('D    eprop I->M only', 'data_b2/m4/scope_im/ep_*.csv', 'data_b2/m4/ho_im/*.csv', None),
    # 预注册臂 C：第二个工作点 psp=1.5 / hidden-base=0（同拓扑，权重可比）
    ('C    R-STDP  @C1', 'data_b2/m4/c1_base/base_*.csv', None, None),
    ('C    eprop   @C1', 'data_b2/m4/c1_eprop/ep_*.csv', 'data_b2/m4/ho_c1_eprop/*.csv', None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='data_b2/m4/m4_agg.json')
    args = ap.parse_args()

    out = []
    # 表头一律 ASCII：Windows 控制台默认 GBK，中文列名会把输出打乱
    print(f"{'arm':<28}{'n':>3}{'medD_in':>12}{'<=0.40':>8}{'arr':>7}{'medD_heldout':>15}{'<=0.40':>8}")
    print('-' * 92)
    for label, train_pat, ho_pat, json_name in ARMS:
        tr = arm_stats(train_pat)
        if tr is None:
            print(f"{label:<28}  (missing: {train_pat})")
            continue
        ho = arm_stats(ho_pat) if ho_pat else None
        row = {'arm': label, 'train': tr, 'heldout': ho}
        if json_name:
            p = EXP / 'data_b2/m4' / json_name
            if p.exists():
                j = json.loads(p.read_text(encoding='utf-8'))
                agg = j.get('agg', {})
                row['align'] = {k: agg.get(k, {}).get('mean') for k in
                                ('corr_ep_tail', 'corr_main', 'sign_acc', 'err_norm')}
        out.append(row)
        ho_s = (f"{ho['medD_mean']:>16.4f}" if ho else f"{'-':>16}")
        ho_p = (f"{ho['pass_040']:>4}/{ho['n_seed']:<2}" if ho else f"{'-':>7}")
        print(f"{label:<28}{tr['n_seed']:>3}"
              f"{tr['medD_mean']:>10.4f} ±{tr['medD_sd']:>4.3f}{tr['pass_040']:>4}/{tr['n_seed']:<2}"
              f"{tr['arr_rate']:>7.2f}{ho_s}{ho_p}")

    print('\n== align (8-seed mean) ==')
    for r in out:
        a = r.get('align')
        if not a:
            continue
        print(f"  {r['arm']:<28} corr_ep={a['corr_ep_tail']:+.4f} corr_main={a['corr_main']:+.4f} "
              f"sign_acc={a['sign_acc']:.4f} err_norm={a['err_norm']:.4f}")

    p = Path(args.json) if Path(args.json).is_absolute() else EXP / args.json
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[saved] {p}")


if __name__ == '__main__':
    main()
