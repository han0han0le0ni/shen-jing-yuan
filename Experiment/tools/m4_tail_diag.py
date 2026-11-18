"""M4 尾部失败归因（B2 medR 右尾过长，2026-11-18）。

用法：
  py tools/m4_tail_diag.py
  py tools/m4_tail_diag.py --json data_b2/m4/tail_diag.json

问题：B 臂 10 seed 的 `medR` 只有 5/10 ≤ 0.20，尾部（s606/s707/s808/s1010）卡在 0.22–0.43。
三条互相排斥的假设，用**已有产物**判，不训练：

  D-a 表征不可分辨？  读权重：看训练是否在运动层群体上建出**单调的 pref→驱动**梯度。
      per-motor drive_i = Σ_{pre} Δw(该运动神经元的输入突触)
      corr_drive   = corr(drive_i, pref_i)      —— 单调整度（正负池是否被区分）
      contrast     = mean(drive | pref>+0.2) − mean(drive | pref<−0.2)  —— 正负池落差
      push         = Σ pref_i·drive_i / Σ|pref_i|  —— 净指令推动方向（应随任务需求变化）
  D-b 解码噪声撞顶？   读轨迹：末段（|e| 小）与远段（|e| 大）的符号一致率、指令标准差。
  D-c 只是没训满？     读 CSV：medR 与训练末 40 轮 medRatio 的跨 seed 相关。

判定规则（写死在本文件里，避免事后解释）：
  · D-c 相关 r > 0.7 且尾部 seed 末段仍在下降 → 「没训满」
  · D-a |corr_drive| 在快/慢 seed 间差 >2 倍，且慢 seed 的 contrast ≈ 0 → 「表征不可分辨」
  · D-b 末段符号一致率快/慢 seed 无差、而远段有差 → 「控制律而非解码」
纪律：只读，不训练、不改任何参数。
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent

# m2 预设：1000 输入 + 3500 隐藏 + 278/278 运动 = 5056
M1_BASE = 4500
M1_N = 278
M2_N = 278


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


def pref_of(post):
    """按 net2_init 的确定性定义还原 pref_omega（motor1 层：-0.5..0.5 均匀）。"""
    i = post - M1_BASE
    return -0.5 + 1.0 * i / (M1_N - 1)


def find_pair(seed):
    for d in ('data_b2/m4/ep_8s', 'data_b2/m4/seed_extra'):
        w0 = EXP / d / f'w0_s{seed}.txt'
        w1 = EXP / d / f'w1_ep_lr1e05_s{seed}.txt'
        if w0.exists() and w1.exists():
            return w0, w1
    return None, None


def diag_a(seeds, cs_dirs):
    """权重侧：运动层群体上的 pref→驱动 梯度。"""
    out = {}
    for seed in seeds:
        w0p, w1p = find_pair(seed)
        if w0p is None:
            continue
        p0, q0, w0 = load_wdump(w0p)
        p1, q1, w1 = load_wdump(w1p)
        if p0.shape != p1.shape or not np.array_equal(p0, p1) or not np.array_equal(q0, q1):
            print(f'  [warn] seed {seed}: 突触顺序不一致，跳过 D-a')
            continue
        dw = w1 - w0
        m = (q0 >= M1_BASE) & (q0 < M1_BASE + M1_N)
        drive = np.bincount(q0[m] - M1_BASE, weights=dw[m], minlength=M1_N)
        base = np.bincount(q0[m] - M1_BASE, weights=w0[m], minlength=M1_N)
        pref = pref_of(np.arange(M1_BASE, M1_BASE + M1_N))
        pos = pref > 0.2
        neg = pref < -0.2
        r = float(np.corrcoef(drive, pref)[0, 1]) if drive.std() > 0 else float('nan')
        r_base = float(np.corrcoef(drive, base)[0, 1]) if drive.std() > 0 else float('nan')
        # 有效神经元占比：Δw 驱动超过全体 1 个 sd 的运动神经元比例（"被选中强化"的集中度）
        eff = float((np.abs(drive) > drive.std()).mean())
        out[seed] = {
            'corr_drive': r,
            'corr_drive_base': r_base,
            'contrast': float(drive[pos].mean() - drive[neg].mean()),
            'push': float((pref * drive).sum() / np.abs(pref).sum()),
            'drive_sd': float(drive.std()),
            'eff_frac': eff,
        }
    return out


def diag_b(seeds, last_ep=40):
    """轨迹侧：**只用最后 last_ep 轮**看远段/末段符号一致率、指令噪声、胜者熵。

    只取末段的理由：训练早期的"远段"大量是未训练行为，混进来会把 far_agree 压到 0.2，
    那是 ep0 的性质，不是机制差异。
    胜者熵：解码是 ω=(1/K)Σpref_winner，om1 的取值≈胜者的 pref ⇒ om1 的分布直方图
    直接反映"哪些运动神经元在主导"，熵低=少数神经元垄断，熵高=群体平均。
    """
    out = {}
    for seed in seeds:
        paths = glob.glob(str(EXP / f'data_b2/m4/**/cmd_*s{seed}.jsonl'), recursive=True)
        if not paths:
            continue
        eps = {}
        for line in open(paths[0], encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('{"header"'):
                continue
            r = json.loads(line)
            eps.setdefault(r['ep'], []).append(r)
        keys = sorted(eps)[-last_ep:]
        far_ok, far_n, near_ok, near_n, sd = 0, 0, 0, 0, []
        om_all = []
        for ep in keys:
            rows = eps[ep]
            om = np.array([r['om1'] for r in rows])
            ws = np.array([r['w1s'] for r in rows])
            e = np.array([r['e1'] for r in rows])
            sd.append(om.std())
            om_all.append(om)
            ag = (np.sign(om) == np.sign(ws))
            far = np.abs(e) >= 0.3
            near = np.abs(e) < 0.3
            far_ok += int(ag[far].sum()); far_n += int(far.sum())
            near_ok += int(ag[near].sum()); near_n += int(near.sum())
        om_all = np.concatenate(om_all) if om_all else np.array([0.0])
        # 胜者熵：按 motor1 的 278 个离散 pref 分箱
        bins = np.round((om_all + 0.5) * (M1_N - 1)).astype(int)
        bins = np.clip(bins, 0, M1_N - 1)
        cnt = np.bincount(bins, minlength=M1_N).astype(np.float64)
        p = cnt / max(1.0, cnt.sum())
        nz = p[p > 0]
        ent = float(-(nz * np.log2(nz)).sum())
        # D-e 低频分量：逐 tick 指令是量化噪声，真正驱动手臂的是它的低频部分。
        # coherence = sd(EMA)/sd(raw)；corr_lf = corr(EMA(ω), ω*) 逐 ep 平均。
        al = 1.0 / 50.0
        coh, clf = [], []
        for ep in keys:
            rows = eps[ep]
            om = np.array([r['om1'] for r in rows])
            ws = np.array([r['w1s'] for r in rows])
            if om.size < 10:
                continue
            ema = np.empty_like(om)
            acc = om[0]
            for i, v in enumerate(om):
                acc += al * (v - acc)
                ema[i] = acc
            if om.std() > 0:
                coh.append(ema.std() / om.std())
            if ema.std() > 0 and ws.std() > 0:
                clf.append(np.corrcoef(ema, ws)[0, 1])
        out[seed] = {
            'far_agree': far_ok / far_n if far_n else float('nan'),
            'near_agree': near_ok / near_n if near_n else float('nan'),
            'sd_om': float(np.mean(sd)),
            'winner_entropy': ent,
            'winner_entropy_norm': ent / np.log2(M1_N),
            'coherence': float(np.mean(coh)) if coh else float('nan'),
            'corr_lf': float(np.nanmean(clf)) if clf else float('nan'),
        }
    return out


def diag_c():
    """闭环侧：medR 与训练末段 medRatio 的跨 seed 相关。"""
    rows = []
    for pat in ('data_b2/m4/ep_8s/ep_*.csv', 'data_b2/m4/seed_extra/ep_*.csv'):
        for p in sorted(glob.glob(str(EXP / pat))):
            seed = re.search(r'_s(\d+)\.csv', p).group(1)
            with open(p, newline='', encoding='utf-8') as f:
                rs = list(csv.DictReader(f))
            ev = [r for r in rs if r.get('is_eval') == '1']
            tr = [r for r in rs if r.get('is_eval') == '0' and r.get('dist_ratio')]
            if not ev or not tr:
                continue
            medR = float(np.median([float(r['dist_ratio']) for r in ev]))
            tail = [float(r['dist_ratio']) for r in tr[-40:]]
            head = [float(r['dist_ratio']) for r in tr[:40]]
            rows.append({'seed': seed, 'medR': medR,
                         'tail_ratio': float(np.median(tail)),
                         'late_drop': float(np.median(head) - np.median(tail))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    seeds = ['101', '202', '303', '404', '505', '606', '707', '808', '909', '1010']

    print('== D-c 闭环：medR vs 训练末 40 轮 medRatio ==')
    c = diag_c()
    if c:
        mr = np.array([r['medR'] for r in c])
        tl = np.array([r['tail_ratio'] for r in c])
        ld = np.array([r['late_drop'] for r in c])
        print(f"  {'seed':>5}{'medR':>9}{'tail_ratio':>12}{'late_drop':>11}")
        for r in sorted(c, key=lambda x: x['medR']):
            print(f"  {r['seed']:>5}{r['medR']:>9.4f}{r['tail_ratio']:>12.4f}{r['late_drop']:>11.4f}")
        print(f"  r(medR, tail_ratio) = {np.corrcoef(mr, tl)[0, 1]:+.3f}   "
              f"r(medR, late_drop) = {np.corrcoef(mr, ld)[0, 1]:+.3f}")

    print('\n== D-a 权重：运动层 pref->驱动 梯度（I/H -> M 的 Δw 汇总）==')
    a = diag_a(seeds, None)
    print(f"  {'seed':>5}{'corr_pref':>11}{'corr_base':>11}{'contrast':>10}"
          f"{'push':>9}{'drive_sd':>10}{'eff%':>7}")
    for s in seeds:
        if s in a:
            v = a[s]
            print(f"  {s:>5}{v['corr_drive']:>11.3f}{v['corr_drive_base']:>11.3f}"
                  f"{v['contrast']:>10.3f}{v['push']:>9.3f}{v['drive_sd']:>10.3f}"
                  f"{v['eff_frac'] * 100:>6.1f}%")

    print('\n== D-b 轨迹（仅末 40 轮）：符号一致率 / 噪声 / 胜者熵 / 低频 ==')
    b = diag_b(seeds)
    print(f"  {'seed':>5}{'far_agree':>11}{'near_agree':>12}{'sd_om':>8}{'H_norm':>9}"
          f"{'coher':>8}{'corr_lf':>9}")
    for s in seeds:
        if s in b:
            v = b[s]
            print(f"  {s:>5}{v['far_agree']:>11.3f}{v['near_agree']:>12.3f}{v['sd_om']:>8.3f}"
                  f"{v['winner_entropy_norm']:>9.3f}{v['coherence']:>8.3f}{v['corr_lf']:>9.3f}")

    # 快/慢分组对比（medR 中位数切分）
    if c and a and b:
        mr = {r['seed']: r['medR'] for r in c}
        common = [s for s in seeds if s in a and s in b and s in mr]
        common.sort(key=lambda s: mr[s])
        half = len(common) // 2
        fast, slow = common[:half], common[half:]
        print(f"\n== 快/慢分组（medR 中位切分）==  fast={fast}  slow={slow}")
        for key, src, fmt in [('corr_drive', a, '{:+.3f}'), ('contrast', a, '{:+.3f}'),
                              ('push', a, '{:+.3f}'), ('eff_frac', a, '{:.3f}'),
                              ('far_agree', b, '{:.3f}'), ('near_agree', b, '{:.3f}'),
                              ('sd_om', b, '{:.3f}'), ('winner_entropy_norm', b, '{:.3f}'),
                              ('coherence', b, '{:.3f}'), ('corr_lf', b, '{:+.3f}')]:
            vf = np.mean([src[s][key] for s in fast])
            vs = np.mean([src[s][key] for s in slow])
            print(f"  {key:<20} fast={fmt.format(vf)}  slow={fmt.format(vs)}  "
                  f"Δ={fmt.format(vf - vs)}")

    if args.json:
        p = Path(args.json) if Path(args.json).is_absolute() else EXP / args.json
        p.write_text(json.dumps({'D_a': a, 'D_b': b, 'D_c': c}, ensure_ascii=False, indent=2),
                     encoding='utf-8')
        print(f"\n[saved] {p}")


if __name__ == '__main__':
    main()
