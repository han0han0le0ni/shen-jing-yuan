"""小网格筛选：在新开关的参数空间里找是否存在优于 legacy 的组合。

同 seed、20 ep x 300 ticks 快速筛选；只记录相对趋势，不等于最终达标结论。
用法: py audit/param_grid.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(os.environ.get("TEMP", ".")) / "snn_audit"
TMP.mkdir(parents=True, exist_ok=True)
EXE = ROOT / "benchmark2.exe"
EPISODES, TICKS, SEED = 20, 300, 42

GRID = {
    "legacy": [],
    "psp10_k3": ["--psp", "10", "--wta-k", "3"],
    "psp10_k1": ["--psp", "10", "--wta-k", "1"],
    "psp25_k1": ["--psp", "25", "--wta-k", "1"],
    "psp4_k3": ["--psp", "4", "--wta-k", "3"],
    "weighted_psp4": ["--decode-mode", "weighted", "--psp", "4"],
    "tau200_clamp2": ["--tau-trace", "200", "--trace-clamp", "2"],
    "potential_baseline": ["--reward-mode", "potential"],
    "psp10_tau200": ["--psp", "10", "--tau-trace", "200", "--trace-clamp", "2"],
    "k2_psp10": ["--psp", "10", "--wta-k", "2"],
    "clamp0": ["--trace-clamp", "0"],
}


def summarize(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in list(csv.reader(f))[1:] if len(r) == 7]
    ratio = [float(r[5]) for r in rows]
    final = [float(r[4]) for r in rows]
    k = max(1, len(ratio) // 10)
    return {
        "median": round(statistics.median(ratio), 4),
        "first": round(statistics.median(ratio[:k]), 4),
        "last": round(statistics.median(ratio[-k:]), 4),
        "delta": round(statistics.median(ratio[:k]) - statistics.median(ratio[-k:]), 4),
        "final_min": round(min(final), 4),
        "hit_pct": round(100.0 * sum(x <= 0.2 for x in ratio) / len(ratio), 2),
    }


results = {}
for name, extra in GRID.items():
    out = TMP / f"grid_{name}.csv"
    t0 = time.time()
    p = subprocess.run([str(EXE)] + extra + ["--episodes", str(EPISODES),
                       "--ticks", str(TICKS), "--seed", str(SEED), "--out", str(out)],
                       cwd=TMP, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=180,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    results[name] = {"args": extra, "exit": p.returncode,
                     "seconds": round(time.time() - t0, 1),
                     "stats": summarize(out) if out.is_file() else {}}

base = results["legacy"]["stats"]["median"]
for r in results.values():
    if r["stats"]:
        r["delta_vs_legacy"] = round(r["stats"]["median"] - base, 4)
ranked = sorted(((n, r["stats"]["median"], r["delta_vs_legacy"], r["stats"]["delta"])
                 for n, r in results.items() if r["stats"]), key=lambda x: x[1])
(ROOT / "audit" / "param_grid.json").write_text(json.dumps({
    "config": {"episodes": EPISODES, "ticks": TICKS, "seed": SEED},
    "legacy_median": base, "ranking": ranked, "results": results,
    "note": "快速筛选，单种子，非统计显著结论"}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{'组合':<20} {'median':>8} {'vs legacy':>10} {'学习改善':>9} {'final_min':>10} {'hit%':>7}")
for n, m, d, ld in ranked:
    s = results[n]["stats"]
    print(f"{n:<20} {m:>8} {d:>+10} {ld:>+9} {s['final_min']:>10} {s['hit_pct']:>6.1f}%")
print("\n优于 legacy 的组合:", [n for n, m, d, _ in ranked if d < -0.01])
