"""M4 探索臂汇总：把任意一批同口径 320 ep 的 CSV 与官方 B 臂基线逐 seed 对比。

用法：
    py tools/m4_arm_summary.py data_b2/m4/armL data_b2/m4/armH2
    py tools/m4_arm_summary.py data_b2/m4/struct_dose/s170 --json out.json

口径与 audit/m4_eprop_report_2026-11-17.md §3.7/§3.8 一致：
    medR = 末 20 轮 eval 的 dist_ratio 中位数（B2 判据口径，阈值 0.20）
    medD = 末 20 轮 final_dist 中位数（M2 门槛口径，阈值 0.40）
    arr  = 末 20 轮中 final_dist <= 0.15 的轮数
基线取 data_b2/m4/ep_8s 与 seed_extra 的官方 B 臂（e-prop、lr=1e-5、wta_k=1）。

判读约定（计划里写死的）：尾部 4 seed 中 ≥3 个 medR<=0.20 才算"压住右尾"；
参考 seed 101/202 劣化 >0.05 即记为负面信号。
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
TAIL = (404, 707, 808, 1010)      # 基线 medR > 0.20 的四个尾部 seed
REF = (101, 202)
BASELINE_DIRS = ('data_b2/m4/ep_8s', 'data_b2/m4/seed_extra')


def med(a):
    a = sorted(a)
    n = len(a)
    return (a[n // 2] + a[(n - 1) // 2]) / 2.0


def read_arm(path):
    with open(path, newline='', encoding='utf-8') as f:
        rs = list(csv.DictReader(f))
    ev = [r for r in rs if r.get('is_eval') == '1']
    if len(ev) < 10:
        return None
    return {
        'n_ep': len(rs),
        'n_eval': len(ev),
        'medR': med([float(r['dist_ratio']) for r in ev]),
        'medD': med([float(r['final_dist']) for r in ev]),
        'arr': sum(1 for r in ev if float(r['final_dist']) <= 0.15),
    }


def baseline():
    out = {}
    for d in BASELINE_DIRS:
        for p in glob.glob(str(EXP / d / 'ep_*.csv')):
            m = re.search(r'_s(\d+)\.csv$', p)
            if not m:
                continue
            st = read_arm(p)
            if st:
                out[int(m.group(1))] = st
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    base = baseline()
    report = {}
    for d in args.dirs:
        full = EXP / d if not Path(d).is_absolute() else Path(d)
        rows = {}
        for p in sorted(glob.glob(str(full / '*.csv'))):
            m = re.search(r'_s(\d+)\.csv$', p)
            if not m:
                continue
            seed = int(m.group(1))
            st = read_arm(p)
            if st is None:
                rows[seed] = {'status': 'incomplete'}
                continue
            st['status'] = 'ok'
            if seed in base:
                st['dr'] = st['medR'] - base[seed]['medR']
            rows[seed] = st
        report[d] = rows

        print(f'== {d} ==')
        print(f"  {'seed':>5}{'基线medR':>10}{'臂medR':>9}{'dR':>9}{'medD':>8}{'arr':>6}  标记")
        for s in sorted(rows):
            r = rows[s]
            if r['status'] != 'ok':
                print(f'  {s:>5}{base.get(s, {}).get("medR", float("nan")):>10.3f}'
                      f'  (未完成: {r["status"]})')
                continue
            tag = []
            if r['medR'] <= 0.20:
                tag.append('过0.20')
            if s in TAIL:
                tag.append('尾部')
            if 'dr' in r and r['dr'] > 0.05:
                tag.append('劣化')
            print(f'  {s:>5}{base.get(s, {}).get("medR", float("nan")):>10.3f}'
                  f'{r["medR"]:>9.4f}{r.get("dr", float("nan")):>+9.4f}'
                  f'{r["medD"]:>8.4f}{r["arr"]:>5}/20  {" ".join(tag)}')

        ok = {s: r for s, r in rows.items() if r['status'] == 'ok'}
        tail_ok = [s for s in TAIL if s in ok and ok[s]['medR'] <= 0.20]
        refs = [s for s in REF if s in ok]
        print(f'  尾部 {TAIL} 通过: {len(tail_ok)}/{len(TAIL)} -> {tail_ok}')
        if refs:
            print('  参考 seed dR: ' + ', '.join(
                f's{s} {ok[s]["dr"]:+.3f}' for s in refs if 'dr' in ok[s]))
        if ok:
            drs = [ok[s]['dr'] for s in ok if 'dr' in ok[s]]
            if drs:
                print(f'  全体 dR: mean={np.mean(drs):+.4f} min={np.min(drs):+.4f} '
                      f'max={np.max(drs):+.4f}  改善(<-0.05)={sum(1 for x in drs if x < -0.05)}/{len(drs)}')
        print()

    if args.json:
        p = Path(args.json) if Path(args.json).is_absolute() else EXP / args.json
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[saved] {p}')


if __name__ == '__main__':
    main()
