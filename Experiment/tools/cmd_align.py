"""命令对齐度量（M4 主判据的口径修正，2026-11-17）。

用法：
  py tools/cmd_align.py --trace data_b2/m4/corr_ep5/cmd_s202.jsonl
  py tools/cmd_align.py --dir data_b2/v16_8s --block 40
  py tools/cmd_align.py --dir data_b2/v16_8s --json data_b2/m4/cmd_align_base.json

为什么需要这个脚本（见 audit/m4_eprop_report_2026-11-17.md §6）：
  预注册 §4.1 的主判据是「逐 episode 的 corr(ω,ω*)，末 20 轮均值」。实测发现两个硬缺陷：
  1) 35/80 个 episode 的 ω* 被 clamp 顶在 ±0.5 而**整段恒定** ⇒ 逐 ep 的 Pearson 分母为 0 ⇒ NaN，
     `np.nanmean` 会把这些 episode **静默丢弃**，等于只统计了对结论最有利的那 56%；
  2) 即便不 NaN，逐 ep 的 ω 是**逐 tick 噪声**（k=1 WTA 解码，std≈0.29），而 ω* 逐 tick 只动
     0.02–0.05 ⇒ 相关系数被噪声衰减到 ±0.1 以内，"学没学到"与"噪声多大"无法区分。

  因此本脚本不改训练、不改数据，只把同一批 jsonl 换几个**无自由参数**的口径重算：

  corr_ep   旧口径（逐 ep Pearson，末 20 ep 的 nanmean），并**显式**报 NaN 与饱和占比；
  corr_main **主判据候选**：先用每个 ep 的时间均值 (ω̄, ω̄*) 把 tick 噪声平均掉，
            再在**跨 episode** 上算 Pearson。行为相关的量本来就是平均指令，
            而"不同目标是否产生不同指令"是这个任务唯一需要的能力。
  sign_acc  逐 tick sign(ω)==sign(ω*) 的 |ω*| 加权一致率（0.5 = 掷硬币）。
  err_norm  mean|ω−ω*| / mean|ω*|（0 = 完美跟随；≥1 = 比"指令恒为 0"更差）。
  frac_sat  |ω*| 顶在 0.5 的 tick 占比（诊断项，不进判据：它说明任务在该 episode 上
            要求超出可达速度，任何策略都跟不住）。

纪律：只读 jsonl，不训练、不写权重、不改判据；阈值需另行走签字流程。
"""
import argparse
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
OM_CLAMP = 0.5


def load_trace(path):
    """ep -> {'om1','w1s','om2','w2s'} 的 numpy 数组。"""
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
            d = eps.setdefault(r['ep'], {'om1': [], 'w1s': [], 'om2': [], 'w2s': []})
            d['om1'].append(r['om1'])
            d['w1s'].append(r['w1s'])
            d['om2'].append(r['om2'])
            d['w2s'].append(r['w2s'])
    out = {}
    for ep, d in eps.items():
        out[ep] = {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
    return out


def _pearson(a, b):
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def episode_means(trace, joint):
    """按 ep 顺序返回 (ω̄, ω̄*) 两个向量；丢不掉的时代用 mask 过滤。"""
    om_key, ws_key = ('om1', 'w1s') if joint == 1 else ('om2', 'w2s')
    mo, mw = [], []
    for ep in sorted(trace):
        om, ws = trace[ep][om_key], trace[ep][ws_key]
        if om.size == 0:
            continue
        mo.append(float(om.mean()))
        mw.append(float(ws.mean()))
    return np.asarray(mo), np.asarray(mw)


def report(trace, joint=1, tail=20):
    om_key, ws_key = ('om1', 'w1s') if joint == 1 else ('om2', 'w2s')
    eps = sorted(trace)

    corr_ep, n_nan, sat_ticks, n_ticks = [], 0, 0, 0
    all_om, all_ws = [], []
    for ep in eps:
        om, ws = trace[ep][om_key], trace[ep][ws_key]
        c = _pearson(om, ws)
        if np.isnan(c):
            n_nan += 1
        corr_ep.append(c)
        sat_ticks += int(np.sum(np.abs(np.abs(ws) - OM_CLAMP) < 1e-9))
        n_ticks += ws.size
        all_om.append(om)
        all_ws.append(ws)

    tail_vals = [c for c in corr_ep[-tail:] if not np.isnan(c)]
    tag = [c for c in corr_ep if not np.isnan(c)]

    mo, mw = episode_means(trace, joint)
    all_om = np.concatenate(all_om)
    all_ws = np.concatenate(all_ws)
    agree = (np.sign(all_om) == np.sign(all_ws))
    wgt = np.abs(all_ws)
    err_norm = float(np.abs(all_om - all_ws).mean() / (np.abs(all_ws).mean() + 1e-12))

    return {
        'n_ep': len(eps),
        'n_nan': n_nan,
        'corr_ep_tail': float(np.mean(tail_vals)) if tail_vals else float('nan'),
        'corr_ep_all': float(np.mean(tag)) if tag else float('nan'),
        'corr_main': _pearson(mo, mw),
        'sign_acc': float((agree * wgt).sum() / (wgt.sum() + 1e-12)),
        'err_norm': err_norm,
        'frac_sat': float(sat_ticks / max(1, n_ticks)),
        'std_om_mean': float(np.mean([trace[e][om_key].std() for e in eps])),
        'std_ws_mean': float(np.mean([trace[e][ws_key].std() for e in eps])),
        'sd_om_ep': float(mo.std()),
        'sd_ws_ep': float(mw.std()),
    }


def block_curve(trace, joint=1, block=40):
    """按 ep 分块重算 corr_main / sign_acc，用于判断"持续性"而非单点峰值。"""
    eps = sorted(trace)
    out = []
    for i in range(0, len(eps), block):
        sub = {e: trace[e] for e in eps[i:i + block]}
        r = report(sub, joint, tail=min(20, len(sub)))
        out.append((eps[i], eps[min(i + block, len(eps)) - 1], r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', default='', help='单个 cmd_s*.jsonl')
    ap.add_argument('--dir', default='', help='目录，glob cmd_s*.jsonl')
    ap.add_argument('--joint', type=int, default=1, choices=[1, 2])
    ap.add_argument('--block', type=int, default=40, help='分块曲线的块大小（ep），0=关闭')
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    files = []
    if args.trace:
        files = [Path(args.trace) if Path(args.trace).is_absolute() else EXP / args.trace]
    elif args.dir:
        d = Path(args.dir) if Path(args.dir).is_absolute() else EXP / args.dir
        files = sorted(d.glob('cmd_*.jsonl'))
    if not files:
        print('no trace')
        return

    payload = {'joint': args.joint, 'rows': []}
    hdr = (f"{'seed':>5}{'ep':>5}{'NaN':>5}{'corr_ep':>9}{'corr_main':>11}"
           f"{'sign_acc':>10}{'err_norm':>10}{'sat%':>7}{'sd_om_ep':>10}{'sd_ws_ep':>10}")
    print(hdr)
    print('-' * len(hdr))
    for f in files:
        seed = f.stem.split('_s')[-1]
        tr = load_trace(f)
        r = report(tr, args.joint)
        r['seed'] = seed
        r['file'] = str(f)
        payload['rows'].append(r)
        print(f"{seed:>5}{r['n_ep']:>5}{r['n_nan']:>5}{r['corr_ep_tail']:>+9.3f}"
              f"{r['corr_main']:>+11.3f}{r['sign_acc']:>10.3f}{r['err_norm']:>10.3f}"
              f"{r['frac_sat'] * 100:>6.1f}%{r['sd_om_ep']:>10.3f}{r['sd_ws_ep']:>10.3f}")

        if args.block and len(tr) > args.block:
            print(f"  {'-- 分块（每块 %d ep）' % args.block}")
            for a, b, rb in block_curve(tr, args.joint, args.block):
                print(f"  {a:>4}-{b:<4} corr_main={rb['corr_main']:+.3f} "
                      f"sign_acc={rb['sign_acc']:.3f} err_norm={rb['err_norm']:.3f} "
                      f"NaN={rb['n_nan']}/{rb['n_ep']}")

    if len(payload['rows']) > 1:
        print('\n== 跨 seed 均值 ± SD ==')
        for k in ['corr_ep_tail', 'corr_main', 'sign_acc', 'err_norm', 'frac_sat', 'n_nan']:
            v = np.array([r[k] for r in payload['rows']], dtype=np.float64)
            v = v[~np.isnan(v)]
            if v.size == 0:
                continue
            sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            payload.setdefault('agg', {})[k] = {'mean': float(v.mean()), 'sd': sd, 'n': int(v.size)}
            print(f"  {k:<14} {v.mean():+.4f} ± {sd:.4f}  (n={v.size})")

    if args.json:
        p = Path(args.json) if Path(args.json).is_absolute() else EXP / args.json
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n[saved] {p}")


if __name__ == '__main__':
    main()
