"""T2 系列口径汇总：把若干臂的评测段 CSV 合并成一张表。

用法（注意：label 里不要用逗号，PowerShell 会把逗号当参数分隔符）：
  py -3 tools/t2_arm_summary.py --arm "K1|g0=data_b2/t2o/k1_g0/t2o_k1_g0_s*.csv" --arm "K1|g1=data_b2/t2o/k1_g1/t2o_k1_g1_s*.csv"

多个 glob 用分号分隔。输出：合并 median 距离 / median 比值 / 逐 seed median / 到达数。
口径与 §6.22 一致：只统计 is_eval==1 的行（末 N 轮冻结评测）。
"""
import argparse
import csv
import glob
import statistics as st


def load(paths):
    D, R, per, arr, n = [], [], [], 0, 0
    for p in paths:
        with open(p, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("is_eval") == "1"]
        if not rows:
            continue
        d = [float(r["final_dist"]) for r in rows]
        per.append(st.median(d))
        D += d
        R += [float(r["dist_ratio"]) for r in rows]
        arr += sum(1 for r in rows if int(r["arrived"]) == 1)
        n += len(rows)
    return D, R, per, arr, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], help="label=glob[;glob...]，可重复")
    a = ap.parse_args()

    print("%-10s %8s %8s %-26s %9s %4s" % ("arm", "medD", "medR", "perSeed", "arr", "csv"))
    for spec in a.arm:
        label, _, pats = spec.partition("=")
        pats = [p.strip() for p in pats.split(";") if p.strip()]
        files = sorted({f for p in pats for f in glob.glob(p)})
        if not files:
            print("%-10s %8s" % (label, "no-data"))
            continue
        D, R, per, arr, n = load(files)
        if not D:
            print("%-10s %8s" % (label, "no-eval"))
            continue
        print("%-10s %8.4f %8.4f %-26s %5d/%-3d %4d" % (
            label, st.median(D), st.median(R),
            "/".join("%.3f" % x for x in per), arr, n, len(files)))


if __name__ == "__main__":
    main()
