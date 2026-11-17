"""三级修复 A/B：legacy / fix1(解码) / fix2(+奖励) / fix3(+资格迹)。

先做字节级回归对比：修复后 legacy 的 CSV 必须与修复前完全一致。
用法: py audit/preset_ab.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
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
EPISODES, TICKS, SEED = 50, 500, 42


def summarize(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    body = [r for r in rows[1:] if len(r) == 7]
    ratio = [float(r[5]) for r in body]
    final = [float(r[4]) for r in body]
    k = max(1, len(ratio) // 10)
    return {
        "n": len(ratio),
        "median_ratio": round(statistics.median(ratio), 4),
        "mean_ratio": round(statistics.fmean(ratio), 4),
        "final_dist_min": round(min(final), 4),
        "final_dist_mean": round(statistics.fmean(final), 4),
        "first_decile": round(statistics.median(ratio[:k]), 4),
        "last_decile": round(statistics.median(ratio[-k:]), 4),
        "learning_delta": round(statistics.median(ratio[:k]) - statistics.median(ratio[-k:]), 4),
        "frac_le_0_2": round(sum(x <= 0.2 for x in ratio) / len(ratio), 4),
        "sha16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    }


results = {}
for preset in ("legacy", "fix1", "fix2", "fix3"):
    out = TMP / f"preset_{preset}.csv"
    t0 = time.time()
    p = subprocess.run([str(EXE), "--preset", preset, "--episodes", str(EPISODES),
                        "--ticks", str(TICKS), "--seed", str(SEED), "--out", str(out)],
                       cwd=TMP, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=240,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    text = (p.stdout or "") + (p.stderr or "")
    mode = re.search(r"Mode:.*", text)
    verdict = re.search(r"\[(PASS|FAIL)\]", text)
    stats = summarize(out) if out.is_file() else {}
    meta = Path(str(out) + ".meta.json")
    cfg = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    results[preset] = {
        "exit": p.returncode, "seconds": round(time.time() - t0, 1),
        "mode": mode.group(0) if mode else None,
        "verdict": verdict.group(1) if verdict else None,
        "stats": stats,
        "sidecar": {k: cfg.get(k) for k in ("preset", "psp_factor", "wta_k", "decode_mode",
                                            "reward_mode", "use_reward_baseline",
                                            "tau_trace", "trace_clamp")},
    }

# 修复前基线（reward_ablation.py 生成的 50ep x 500ticks, seed=42）
before = TMP / "ab_default.csv"
after = TMP / "preset_legacy.csv"
byte_equal = (before.is_file() and after.is_file()
              and before.read_bytes() == after.read_bytes())
baseline = summarize(before) if before.is_file() else {}

delta = {p: round(results[p]["stats"]["median_ratio"] - results["legacy"]["stats"]["median_ratio"], 4)
         for p in ("fix1", "fix2", "fix3") if results[p]["stats"]}
payload = {
    "config": {"episodes": EPISODES, "ticks": TICKS, "seed": SEED, "threshold": 0.20},
    "regression": {"before_csv": str(before), "before_median": baseline.get("median_ratio"),
                   "after_median": results["legacy"]["stats"].get("median_ratio"),
                   "byte_identical": byte_equal,
                   "note": "byte_identical=True 表示新增开关未改变默认路径"},
    "median_delta_vs_legacy": delta,
    "results": results,
}
(ROOT / "audit" / "preset_ab.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("默认路径字节级回归:", "一致 ✓" if byte_equal else "不一致（需排查）",
      f"修复前 median={baseline.get('median_ratio')} 修复后={results['legacy']['stats'].get('median_ratio')}")
print()
print(f"{'preset':<9} {'median':>8} {'last10':>8} {'delta':>8} {'final_min':>10} {'hit%':>7} {'verdict':>8}")
for name, r in results.items():
    s = r["stats"]
    print(f"{name:<9} {s.get('median_ratio'):>8} {s.get('last_decile'):>8} "
          f"{s.get('learning_delta'):>8} {s.get('final_dist_min'):>10} "
          f"{s.get('frac_le_0_2', 0) * 100:>6.1f}% {str(r['verdict']):>8}  {r['seconds']}s")
print("\n相对 legacy 的 median 变化:", delta)
