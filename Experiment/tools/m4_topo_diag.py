"""M4 尾部归因 D-f / D-g：拓扑异质性与循环层临界性（只读，零训练）。

背景：`m4_tail_diag.py` 的 D-a…D-e 五个维度都区分不了 `medR` 快/慢 seed
（结论见 audit/m4_eprop_report_2026-11-17.md §9b）。本脚本补两个**不依赖轨迹**的维度：

D-f 入度异质性：I→H / H→H / I→M / H→M 的入度分布（均值、离散度、孤立神经元占比）。
    m2 预设按"每神经元固定扇出"生长，所以这一维在 seed 间**按构造应当一致**；
    跑它是为了把"拓扑随机性"从嫌疑名单里显式排除，不是期待有差。

D-g 临界性：H→H 权重矩阵的谱半径（Perron 根，幂迭代）。权重全 ≥ WEIGHT_MIN=0.001，
    即非负矩阵，Perron–Frobenius 保证主特征值为实正 ⇒ 幂迭代的 ||y|| 收敛到 ρ。
    文献（Noè et al., Neuromorph. Comput. Eng. 5, 044002 (2025)）报告 e-prop 训练后
    ρ 升到 ~1.25（edge of chaos）；本网络若明显低于该值，循环整合不足。

边界：m2 预设 input_count=1056（1000 X + 56 prop）/ hidden=3500 / motor 250+250，
      与 `m4_tail_diag.py` 共用同一组常量（2026-11-18 修正后的正确值）。

纪律：只读。不训练、不改任何参数、不写权重。
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent

IN_N = 1056                 # 1000 个 X 输入 + 56 个 prop 输入
H_N = 3500
M1_N = 250
M2_N = 250
H0 = IN_N                   # 隐藏层首位
H1 = IN_N + H_N             # 隐藏层末位（不含）
M0 = H1                     # 运动层首位
M1B = M0 + M1_N
NTOT = M0 + M1_N + M2_N


def load_wdump(path):
    """-> (pre, post, w)。头部 `# B2W <ver> <neurons> <synapses>`。"""
    pre, post, w = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.split()
            if len(p) != 3:
                continue
            pre.append(int(p[0]))
            post.append(int(p[1]))
            w.append(float(p[2]))
    return np.array(pre, np.int64), np.array(post, np.int64), np.array(w, np.float64)


def find_pair(seed):
    for d in ('data_b2/m4/ep_8s', 'data_b2/m4/seed_extra'):
        w0 = EXP / d / f'w0_s{seed}.txt'
        w1 = EXP / d / f'w1_ep_lr1e05_s{seed}.txt'
        if w0.exists() and w1.exists():
            return w0, w1
    return None, None


def in_degree(pre, post, pre_lo, pre_hi, post_lo, post_hi):
    m = (pre >= pre_lo) & (pre < pre_hi) & (post >= post_lo) & (post < post_hi)
    return np.bincount(post[m] - post_lo, minlength=post_hi - post_lo)


def deg_stats(deg):
    return {
        'mean': float(deg.mean()),
        'sd': float(deg.std()),
        'min': int(deg.min()),
        'max': int(deg.max()),
        'p_le1': float((deg <= 1).mean()),          # 入度 ≤1 的神经元占比
        'p_zero': float((deg == 0).mean()),
    }


def rho_power(pre, post, w, iters=1000, tol=1e-10):
    """H→H 子矩阵的谱半径（幂迭代；非负矩阵 ⇒ 主特征值实正）。"""
    m = (pre >= H0) & (pre < H1) & (post >= H0) & (post < H1)
    a = pre[m] - H0
    b = post[m] - H0
    c = w[m]
    x = np.random.default_rng(0).normal(size=H_N)
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(iters):
        y = np.zeros(H_N)
        np.add.at(y, b, c * x[a])
        nl = float(np.linalg.norm(y))
        if nl == 0.0:
            return 0.0
        x = y / nl
        if abs(nl - lam) < tol:
            return nl
        lam = nl
    return lam


def medr_map():
    """逐 seed 的 medR（末 20 轮 eval 的 dist_ratio 中位数），与 m4_tail_diag.diag_c 同口径。"""
    out = {}
    for pat in ('data_b2/m4/ep_8s/ep_*.csv', 'data_b2/m4/seed_extra/ep_*.csv'):
        for p in sorted(glob.glob(str(EXP / pat))):
            seed = re.search(r'_s(\d+)\.csv', p).group(1)
            with open(p, newline='', encoding='utf-8') as f:
                ev = [r for r in csv.DictReader(f) if r.get('is_eval') == '1']
            if ev:
                out[seed] = float(np.median([float(r['dist_ratio']) for r in ev]))
    return out


def diag_fg(seeds):
    f, g = {}, {}
    for seed in seeds:
        w0p, w1p = find_pair(seed)
        if w0p is None:
            continue
        p0, q0, w0 = load_wdump(w0p)
        p1, q1, w1 = load_wdump(w1p)
        if p0.shape != p1.shape or not np.array_equal(p0, p1) or not np.array_equal(q0, q1):
            print(f'  [warn] seed {seed}: 突触顺序不一致，跳过')
            continue
        f[seed] = {
            'deg_hh': deg_stats(in_degree(p0, q0, H0, H1, H0, H1)),
            'deg_ih': deg_stats(in_degree(p0, q0, 0, H0, H0, H1)),
            'deg_hm': deg_stats(in_degree(p0, q0, H0, H1, M0, NTOT)),
            'deg_im': deg_stats(in_degree(p0, q0, 0, H0, M0, NTOT)),
            'synapses': int(len(p0)),
        }
        # 隐藏层总输入权重（I→H + H→H，按突触前层分列）：反映"每个隐藏神经元被喂多粗"
        for lbl, pre, post, w in (('w0', p0, q0, w0), ('w1', p1, q1, w1)):
            mh = (post >= H0) & (post < H1) & (pre < H0)
            mrec = (post >= H0) & (post < H1) & (pre >= H0) & (pre < H1)
            drive_in = np.bincount(post[mh] - H0, weights=w[mh], minlength=H_N)
            drive_hh = np.bincount(post[mrec] - H0, weights=w[mrec], minlength=H_N)
            f[seed][f'drive_{lbl}'] = {
                'in_mean': float(drive_in.mean()), 'in_sd': float(drive_in.std()),
                'hh_mean': float(drive_hh.mean()), 'hh_sd': float(drive_hh.std()),
                'p_zero_in': float((drive_in <= 0).mean()),
            }
        g[seed] = {
            'rho0': rho_power(p0, q0, w0),
            'rho1': rho_power(p1, q1, w1),
        }
        g[seed]['drho'] = g[seed]['rho1'] - g[seed]['rho0']
    return f, g


def split_stats(vals, fast, slow, key, fmt):
    vf = np.mean([vals[s][key] for s in fast])
    vs = np.mean([vals[s][key] for s in slow])
    return f"  {key:<16} fast={fmt.format(vf)}  slow={fmt.format(vs)}  Δ={fmt.format(vf - vs)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    seeds = ['101', '202', '303', '404', '505', '606', '707', '808', '909', '1010']
    mr = medr_map()
    f, g = diag_fg(seeds)

    print('== D-f 入度异质性（w0；m2 预设按固定扇出生长）==')
    print(f"  {'seed':>5}{'medR':>8}{'H<-H':>7}{'sd':>6}{'I<-H':>7}{'H<-M':>8}{'I<-M':>7}"
          f"{'deg0_HH%':>10}{'driveI':>8}")
    for s in seeds:
        if s not in f:
            continue
        d = f[s]
        print(f"  {s:>5}{mr.get(s, float('nan')):>8.3f}"
              f"{d['deg_hh']['mean']:>7.2f}{d['deg_hh']['sd']:>6.2f}"
              f"{d['deg_ih']['mean']:>7.2f}{d['deg_hm']['mean']:>8.2f}{d['deg_im']['mean']:>7.2f}"
              f"{d['deg_hh']['p_zero'] * 100:>9.1f}%{d['drive_w0']['in_mean']:>8.2f}")

    print('\n== D-g 循环层临界性（H→H 谱半径，幂迭代）==')
    print(f"  {'seed':>5}{'medR':>8}{'rho0':>8}{'rho1':>8}{'drho':>8}")
    for s in seeds:
        if s not in g:
            continue
        print(f"  {s:>5}{mr.get(s, float('nan')):>8.3f}"
              f"{g[s]['rho0']:>8.3f}{g[s]['rho1']:>8.3f}{g[s]['drho']:>+8.3f}")

    common = [s for s in seeds if s in f and s in g and s in mr]
    if len(common) >= 4:
        common.sort(key=lambda s: mr[s])
        half = len(common) // 2
        fast, slow = common[:half], common[half:]
        print(f"\n== 快/慢分组（medR 中位切分）==  fast={fast}  slow={slow}")
        print(split_stats({s: f[s]['deg_hh'] for s in common}, fast, slow, 'mean', '{:.2f}'))
        print(split_stats({s: f[s]['deg_ih'] for s in common}, fast, slow, 'mean', '{:.2f}'))
        print(split_stats({s: f[s]['drive_w0'] for s in common}, fast, slow, 'in_mean', '{:.2f}'))
        print(split_stats({s: f[s]['drive_w0'] for s in common}, fast, slow, 'hh_mean', '{:.2f}'))
        print(split_stats(g, fast, slow, 'rho0', '{:.3f}'))
        print(split_stats(g, fast, slow, 'rho1', '{:.3f}'))
        print(split_stats(g, fast, slow, 'drho', '{:+.3f}'))

        y = np.array([mr[s] for s in common])
        for lbl, src, key in (('deg_HH', f, 'deg_hh'), ('drive_I_H(w0)', f, 'drive_w0'),
                              ('rho0', g, 'rho0'), ('rho1', g, 'rho1'), ('drho', g, 'drho')):
            if lbl.startswith('deg') or lbl.startswith('drive'):
                col = 'mean' if lbl.startswith('deg') else 'in_mean'
                x = np.array([src[s][key][col] for s in common])
            else:
                x = np.array([src[s][key] for s in common])
            if x.std() > 0:
                print(f"  r({lbl}, medR) = {np.corrcoef(x, y)[0, 1]:+.3f}")

    if args.json:
        p = Path(args.json) if Path(args.json).is_absolute() else EXP / args.json
        p.write_text(json.dumps({'D_f': f, 'D_g': g, 'medR': mr},
                                ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n[saved] {p}")


if __name__ == '__main__':
    main()
