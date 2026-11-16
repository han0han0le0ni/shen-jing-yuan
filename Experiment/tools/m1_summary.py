"""M1 汇总：读 data_b2/m1/*.json，按预注册 §3 的四项机制判据出表并给判定。

判据（阈值零变更，机制口径）：
  密度 ∈ [0.05, 0.15]、ISI-CV ≥ 0.30、PR ≥ 50、|lag4| ≤ 0.50
用法：py tools/m1_summary.py [--dir data_b2/m1]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent

ARMS = [("A0", 2.0, 25.0), ("A1", 0.5, 25.0), ("A2", 0.0, 25.0),
        ("B0", 2.0, 6.0), ("B1", 0.5, 6.0), ("C1", 0.0, 1.5)]


def crit(d, cv, pr, ac):
    return (
        (0.05 <= d <= 0.15),
        (cv >= 0.30),
        (pr >= 50.0),
        (abs(ac) <= 0.50),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_b2/m1")
    args = ap.parse_args()
    root = EXP / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)

    data = defaultdict(list)
    for p in sorted(root.glob('*.json')):
        arm = p.name.split('_')[0]
        try:
            j = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        a = j.get('activity') or {}
        ac = (a.get('rate_autocorr_hidden') or {}).get('4')
        if ac is None:
            continue
        data[arm].append({
            'seed': j.get('meta', {}).get('seed'),
            'psp': j.get('meta', {}).get('psp_factor_effective'),
            'hb': j.get('meta', {}).get('hidden_base_effective'),
            'd': a.get('density_hidden'),
            'ceil': a.get('density_hidden_over_ceiling'),
            'cv': a.get('isi_cv_hidden_median'),
            'pr': a.get('participation_ratio_hidden'),
            'k99': a.get('effective_rank_99pct'),
            'ac4': float(ac),
            'rho': (j.get('criticality') or {}).get('spectral_radius_with_psp_over_gap'),
        })

    print(f"{'arm':<4}{'hb':>5}{'psp':>6} | {'density':>18}{'vs 0.25':>9}{'ISI-CV':>9}"
          f"{'PR':>9}{'ER99':>7}{'lag4':>8} | 命中")
    print('-' * 96)
    verdict = {}
    for name, hb, psp in ARMS:
        rows = data.get(name, [])
        if not rows:
            print(f"{name:<4}{hb:>5}{psp:>6} | (no data)")
            continue
        n = len(rows)
        m = lambda k: sum(r[k] for r in rows) / n
        ok = [crit(r['d'], r['cv'], r['pr'], r['ac4']) for r in rows]
        n_all = sum(1 for o in ok if all(o))
        hits = [sum(1 for o in ok if o[i]) for i in range(4)]
        verdict[name] = (n_all, n)
        print(f"{name:<4}{hb:>5.1f}{psp:>6.1f} | {m('d'):>18.4f}{m('ceil')*100:>8.1f}%"
              f"{m('cv'):>9.3f}{m('pr'):>9.2f}{m('k99'):>7.0f}{m('ac4'):>8.3f} | "
              f"四项全中 {n_all}/{n}  (密度{hits[0]} CV{hits[1]} PR{hits[2]} AC{hits[3]})")

    print()
    print('逐 seed 明细：')
    for name, _, _ in ARMS:
        for r in data.get(name, []):
            print(f"  {name} seed={r['seed']} hb={r['hb']} psp={r['psp']} "
                  f"d={r['d']:.4f} cv={r['cv']:.3f} pr={r['pr']:.2f} ac4={r['ac4']:.3f} rho={r['rho']:.3f}")

    print()
    best = [(v[0], k) for k, v in verdict.items() if v[0] == v[1] and v[1] > 0]
    any_hit = [k for k, v in verdict.items() if v[0] > 0]
    if best:
        print(f"判定：工作点修复成立（四项全中且全 seed 一致）→ {sorted(best, reverse=True)[0][1]}")
    elif any_hit:
        print(f"判定：部分 seed 命中 → {any_hit}；按预注册需全 seed 一致才算成立，作「不稳定命中」记录")
    else:
        print("判定：六臂均无 seed 四项全中 → 工作点不是这两个旋钮能修的（预注册 §3 第四行）")


if __name__ == '__main__':
    main()
