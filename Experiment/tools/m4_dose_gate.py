"""B2-struct 剂量预注册的 Gate 1 / Gate 2 判定器（只读）。

对应 audit/b2_struct_dose_prereg_2026-11-18.md §3：
  Gate 1（筛查集 4 seed）  ① >=3 个 medR<=0.20  ② 筛查集 medR 均值 <=0.19  ③ medD 全部 <=0.40
  Gate 2（确认集 6 seed）  ④ >=5 个 medR<=0.20  ⑤ 合并 10 seed >=8 个 medR<=0.20

用法：
    py tools/m4_dose_gate.py data_b2/m4/dose2/*          # 判定所有臂
    py tools/m4_dose_gate.py data_b2/m4/dose2/S1_200_20_m3 --gate2
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
SCREEN = (303, 404, 707, 808)          # 预注册写死的筛查集
CONFIRM = (101, 202, 505, 606, 909, 1010)   # 确认集（全程不参与选择）
TAU_MEDR = 0.20
TAU_MEDD = 0.40
TAU_MEAN = 0.19


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
        'medR': med([float(r['dist_ratio']) for r in ev]),
        'medD': med([float(r['final_dist']) for r in ev]),
    }


def load_dir(d):
    out = {}
    for p in sorted(glob.glob(str(Path(d) / '*.csv'))):
        m = re.search(r'_s(\d+)\.csv$', p)
        if not m:
            continue
        st = read_arm(p)
        if st:
            out[int(m.group(1))] = st
    return out


def gate1(rows):
    s = [rows[x] for x in SCREEN if x in rows]
    if len(s) < len(SCREEN):
        return None, f'筛查集缺 {[x for x in SCREEN if x not in rows]}'
    npass = sum(1 for r in s if r['medR'] <= TAU_MEDR)
    mean = float(np.mean([r['medR'] for r in s]))
    worst_d = max(r['medD'] for r in s)
    ok = (npass >= 3) and (mean <= TAU_MEAN) and (worst_d <= TAU_MEDD)
    return ok, f'过线 {npass}/4（需≥3） 均值 {mean:.4f}（需≤{TAU_MEAN}） 最差 medD {worst_d:.3f}（需≤{TAU_MEDD}）'


def gate2(rows):
    c = [rows[x] for x in CONFIRM if x in rows]
    if len(c) < len(CONFIRM):
        return None, f'确认集缺 {[x for x in CONFIRM if x not in rows]}'
    npass = sum(1 for r in c if r['medR'] <= TAU_MEDR)
    allv = [rows[x] for x in list(SCREEN) + list(CONFIRM) if x in rows]
    total = sum(1 for r in allv if r['medR'] <= TAU_MEDR)
    ok = (npass >= 5) and (total >= 8)
    return ok, f'确认集过线 {npass}/6（需≥5） 合并 {total}/{len(allv)}（需≥8）'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--gate2', action='store_true')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    if len(args.dirs) == 1 and args.dirs[0].endswith('*'):
        args.dirs = sorted(glob.glob(str(EXP / args.dirs[0].rstrip('*') + '*')))

    report = {}
    print(f'筛查集={SCREEN}  确认集={CONFIRM}')
    print(f"  {'臂':<18}" + ''.join(f'{str(s):>8}' for s in SCREEN) + f"{'Gate1':>8}")
    for d in args.dirs:
        name = Path(d).name
        rows = load_dir(d)
        if not rows:
            continue
        ok, msg = gate1(rows)
        cells = ''.join(f'{rows[s]["medR"]:>8.4f}' if s in rows else f'{"-":>8}' for s in SCREEN)
        flag = 'PASS' if ok else ('FAIL' if ok is False else 'INC')
        print(f'  {name:<18}{cells}{flag:>8}   {msg}')
        rec = {'rows': rows, 'gate1': ok, 'gate1_msg': msg}
        if args.gate2:
            ok2, msg2 = gate2(rows)
            rec.update({'gate2': ok2, 'gate2_msg': msg2})
            print(f'    Gate2: {"PASS" if ok2 else ("FAIL" if ok2 is False else "INC")}  {msg2}')
        report[name] = rec

    passed = [k for k, v in report.items() if v['gate1']]
    print(f'\nGate1 通过臂: {passed if passed else "无 ⇒ 按预注册判 H3 被反驳、结构轴关闭"}')

    if args.json:
        p = Path(args.json) if Path(args.json).is_absolute() else EXP / args.json
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[saved] {p}')


if __name__ == '__main__':
    main()
