"""打印 scan_results.json 中的关键细节，供报告撰写与复核使用。"""
from __future__ import annotations

import builtins
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "scan_results.json").read_text(encoding="utf-8"))
_LOG = (HERE / "inspect.txt").open("w", encoding="utf-8")


def print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    builtins.print(*args, file=_LOG, **kwargs)

print("=== ARTIFACT TIMESTAMPS ===")
for k, v in R["artifacts"].items():
    print(f"  {v['mtime']}  {v['bytes']:>8}  {v['sha16']}  {k}")
print("=== BINARY MARKERS ===")
for k, v in R["binary_markers"].items():
    print(" ", k, v)

print("\n=== ALL B2 FLAGGED FILES ===")
for x in R["quality"]["b2_bad_files"]:
    print(" ", x["file"], x["flags"])

print("\n=== B1 FAIL / LOW ACCURACY RUNS ===")
for r in R["b1_runs"]:
    if r.get("csv_status") != "PASS":
        print(" ", r["file"], "n=", r.get("n"), "test_acc=", r.get("test_acc"),
              "overall=", r.get("overall_acc"), r["flags"])

b2 = [r for r in R["b2_runs"] if r.get("n")]
print("\n=== B2 RUNS SORTED BY MTIME (name, n, median, mean, min, max, status, version, mtime) ===")
for r in sorted(b2, key=lambda x: x["mtime"]):
    v = re.search(r"v(\d+)", r["file"])
    print(f"  {r['mtime']}  {r['file']:<34} n={r['n']:<5} med={r['median_ratio']:<7} "
          f"mean={r['mean_ratio']:<7} min={r['min_ratio']:<7} max={r['max_ratio']:<7} "
          f"{str(r['status']):<5} v={v.group(1) if v else '-':<3} log={r['paired_log']}")

print("\n=== B2 BEST 12 BY MEDIAN ===")
for r in sorted(b2, key=lambda x: x["median_ratio"])[:12]:
    print(f"  {r['median_ratio']:<7} {r['file']:<34} n={r['n']:<5} "
          f"first={r['first_decile_median']} last={r['last_decile_median']} delta={r['learning_delta']}")

print("\n=== B2 WORST 6 BY MEDIAN ===")
for r in sorted(b2, key=lambda x: -x["median_ratio"])[:6]:
    print(f"  {r['median_ratio']:<7} {r['file']:<34} n={r['n']}")

print("\n=== B2 2000-EPISODE RUNS ===")
for r in sorted([x for x in b2 if x["n"] >= 1500], key=lambda x: x["mtime"]):
    print(f"  {r['mtime']} {r['file']:<26} n={r['n']} med={r['median_ratio']} "
          f"logmed={r.get('log', {}).get('median')} p25={r['p25']} p75={r['p75']} "
          f"frac<=0.2={r['frac_le_target']} frac>1={r['frac_worse_than_start']}")

print("\n=== B1 SEED RESULTS BY SOURCE ===")
for src in sorted(set(x["source"] for x in R["b1_seeds"])):
    group = [x for x in R["b1_seeds"] if x["source"] == src]
    acc = [x["test_acc"] for x in group if x["test_acc"] is not None]
    print(f"  {src:<16} n={len(group)} mean_test={statistics.fmean(acc):.2f} "
          f"min={min(acc)} max={max(acc)} pass={sum(x['status'] == 'PASS' for x in group)}")

print("\n=== ROOT LOGS (name, median, status, ep lines, config episodes) ===")
for r in sorted(R["root_logs"], key=lambda x: x["mtime"]):
    cfg = r.get("config") or {}
    print(f"  {r['mtime']} {r['file']:<24} med={r.get('median')} {str(r['status']):<5} "
          f"ep_lines={r['episode_lines']:<5} declared={cfg.get('episodes')} flags={r['flags']}")

print("\n=== QUALITY LISTS ===")
for k, v in R["quality"].items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)}", (v[:12] if len(v) > 12 else v))

print("\n=== MISMATCH ROW DETAILS (rounding / clipping diagnosis) ===")
import csv as _csv
ROOT = HERE.parent
mis_ratio, mis_rew = [], []
for p in sorted((ROOT / "data_b2").glob("*.csv")):
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = _csv.reader(f)
        next(rows, None)
        for row in rows:
            if len(row) != 7:
                continue
            try:
                v = [float(x) for x in row]
            except ValueError:
                continue
            init, final, ratio, rew = v[3], v[4], v[5], v[6]
            if init > 0 and abs(ratio - final / init) > 0.0011:
                mis_ratio.append((p.name, init, final, ratio, round(final / init, 4)))
            if abs(rew - (1 - final / 2.0)) > 0.002:
                mis_rew.append((p.name, final, rew, round(1 - final / 2.0, 4)))
inits = [x[1] for x in mis_ratio]
print("  ratio mismatch rows:", len(mis_ratio), "init median=",
      round(statistics.median(inits), 4) if inits else None,
      "init<0.1:", sum(x < 0.1 for x in inits), "init<1:", sum(x < 1 for x in inits))
for x in mis_ratio[:8]:
    print("   ", x)
clipped = [x for x in mis_rew if x[2] == 0.0 and x[3] < 0]
print("  reward mismatch rows:", len(mis_rew), "of which clipped-negative-to-zero:", len(clipped))
for x in mis_rew[:8]:
    print("   ", x)

print("\n=== B1 ACCURACY BY GROUP ===")
groups = {}
for r in R["b1_runs"]:
    p = Path(r["path"].replace("\\", "/"))
    key = ("multi_seed/" if p.parent.name == "multi_seed" else "root/") + p.stem.split("_")[0]
    groups.setdefault(key, []).append(r.get("test_acc"))
for k, acc in sorted(groups.items()):
    acc = [x for x in acc if x is not None]
    print(f"  {k:<24} n={len(acc):<3} mean={statistics.fmean(acc):.2f} min={min(acc)} max={max(acc)}")
