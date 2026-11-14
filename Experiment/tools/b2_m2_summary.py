"""B2 · M2 探针汇总（判据 §0.4 / §2 门槛）。

读 data_b2/m2/*.csv（benchmark2.exe 输出），只统计评估段（is_eval=1），
按 §0.4 剔除 init_dist < 0.10 的 invalid episode，用标准中位数定义，
输出每 seed 的 median(dist_ratio)、到达率，以及 §2 门槛判定（≥6/10 seed ≤0.40 且总到达率>0）。

用法：py -3 tools/b2_m2_summary.py [目录] [--report]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT_MIN = 0.10
ARRIVE_R = 0.15
THRESH = 0.40
NEED = 6


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load(path):
    rows = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return rows
    head = lines[0].split(",")
    idx = {k.strip(): i for i, k in enumerate(head)}
    for ln in lines[1:]:
        p = ln.split(",")
        if len(p) < 10:
            continue
        try:
            r = {
                "ep": int(p[idx["episode"]]),
                "init": float(p[idx["init_dist"]]),
                "final": float(p[idx["final_dist"]]),
                "ratio": float(p[idx["dist_ratio"]]),
                "arrived": int(p[idx["arrived"]]),
                "arrival_tick": int(p[idx["arrival_tick"]]),
                "is_eval": int(p[idx["is_eval"]]) if "is_eval" in idx else 0,
            }
        except (ValueError, IndexError):
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(ROOT / "data_b2" / "m2"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)
    files = sorted(d.glob("*.csv"))
    if not files:
        raise SystemExit(f"no csv in {d}")

    per_seed, skipped = [], []
    for f in files:
        rows = load(f)
        ev = [r for r in rows if r["is_eval"] == 1]
        if not ev:
            skipped.append((f.name, "no eval rows"))
            continue
        valid = [r for r in ev if r["init"] >= INIT_MIN]
        if not valid:
            skipped.append((f.name, "all eval invalid"))
            continue
        ratios = [r["ratio"] for r in valid]
        arr = sum(1 for r in valid if r["arrived"] == 1)
        per_seed.append({
            "file": f.name,
            "n_eval": len(ev),
            "n_valid": len(valid),
            "n_invalid": len(ev) - len(valid),
            "median": med(ratios),
            "mean": statistics.fmean(ratios),
            "sd": statistics.pstdev(ratios) if len(ratios) > 1 else 0.0,
            "arrived": arr,
            "arrival_rate": arr / len(valid),
            "arrive_tick_mean": statistics.fmean(
                [r["arrival_tick"] for r in valid if r["arrived"] == 1]) if arr else None,
            "pass": med(ratios) <= THRESH,
        })

    n_pass = sum(1 for s in per_seed if s["pass"])
    tot_arr = sum(s["arrived"] for s in per_seed)
    tot_valid = sum(s["n_valid"] for s in per_seed)
    ok = (n_pass >= NEED) and (tot_arr > 0)

    print(f"{'file':<28}{'valid':>6}{'invalid':>8}{'median':>9}{'mean':>9}{'arr':>5}{'rate':>8}{'pass':>6}")
    for s in per_seed:
        print(f"{s['file']:<28}{s['n_valid']:>6}{s['n_invalid']:>8}"
              f"{s['median']:>9.4f}{s['mean']:>9.4f}{s['arrived']:>5}"
              f"{s['arrival_rate']*100:>7.1f}%{'Y' if s['pass'] else 'N':>6}")
    print(f"\nseeds={len(per_seed)}  median<= {THRESH}: {n_pass}/{len(per_seed)} (需 ≥{NEED})"
          f"  到达率 {tot_arr}/{tot_valid} = {(tot_arr / tot_valid * 100) if tot_valid else 0:.1f}%")
    print(("PASS" if ok else "FAIL") + f"  （§2 M2 门槛：≥{NEED}/{len(per_seed) or NEED} 且到达率>0）")
    for name, why in skipped:
        print(f"  [skip] {name}: {why}")

    out = d / "m2_summary.json"
    out.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"), "dir": str(d),
        "thresh": THRESH, "need": NEED, "init_min": INIT_MIN,
        "seeds": per_seed, "n_pass": n_pass, "ok": ok,
        "total_arrived": tot_arr, "total_valid": tot_valid, "skipped": skipped,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
