"""最终确认：fix1(单 winner) 在完整规模与长训练下是否让杆更靠近目标。

用法: py audit/k1_confirm.py
"""
from __future__ import annotations

import csv
import hashlib
import json
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
TH = re.compile(r"th1=([-\d.]+) th2=([-\d.]+)")

# 确保产物包含最新预设与日志改动
subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(ROOT / "build.ps1")], capture_output=True, text=True, check=True)


def run(tag: str, args, episodes: int, ticks: int, seed: int = 42) -> dict:
    out = TMP / f"confirm_{tag}.csv"
    t0 = time.time()
    p = subprocess.run([str(EXE)] + args + ["--episodes", str(episodes),
                       "--ticks", str(ticks), "--seed", str(seed), "--out", str(out)],
                       cwd=TMP, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    text = (p.stdout or "") + (p.stderr or "")
    rows = [r for r in list(csv.reader(out.open("r", encoding="utf-8", newline="")))[1:]
            if len(r) == 7] if out.is_file() else []
    ratio = [float(r[5]) for r in rows]
    final = [float(r[4]) for r in rows]
    th = [(float(a), float(b)) for a, b in TH.findall(text)]
    k = max(1, len(ratio) // 10)
    return {
        "args": args, "episodes": episodes, "ticks": ticks,
        "exit": p.returncode, "seconds": round(time.time() - t0, 1),
        "median": round(statistics.median(ratio), 4) if ratio else None,
        "first": round(statistics.median(ratio[:k]), 4) if ratio else None,
        "last": round(statistics.median(ratio[-k:]), 4) if ratio else None,
        "learning_delta": round(statistics.median(ratio[:k]) - statistics.median(ratio[-k:]), 4) if ratio else None,
        "final_dist_min": round(min(final), 4) if final else None,
        "hit_pct": round(100.0 * sum(x <= 0.2 for x in ratio) / len(ratio), 2) if ratio else None,
        "theta1_min": round(min(x[0] for x in th), 3) if th else None,
        "theta2_max": round(max(x[1] for x in th), 3) if th else None,
        "theta2_last_decile_mean": round(statistics.fmean(x[1] for x in th[-k:]), 3) if th else None,
        "sha16": hashlib.sha256(out.read_bytes()).hexdigest()[:16] if out.is_file() else None,
    }


results = {
    "legacy_50x500": run("legacy_50x500", ["--preset", "legacy"], 50, 500),
    "fix1_50x500": run("fix1_50x500", ["--preset", "fix1"], 50, 500),
    "k1_psp10_50x500": run("k1_psp10", ["--wta-k", "1", "--psp", "10"], 50, 500),
    "fix1_100x1000": run("fix1_100x1000", ["--preset", "fix1"], 100, 1000),
}

before = TMP / "ab_default.csv"
after = TMP / "confirm_legacy_50x500.csv"
byte_equal = before.is_file() and after.is_file() and before.read_bytes() == after.read_bytes()

payload = {
    "config": {"seed": 42, "threshold_ratio": 0.20, "theta_needed": [-0.335, 1.319]},
    "regression": {"byte_identical_to_pre_fix": byte_equal,
                   "note": "True 表示默认路径未受任何改动影响"},
    "results": results,
    "note": "所有运行产物在系统 TEMP；仓库数据未改动",
}
(ROOT / "audit" / "k1_confirm.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

print("默认路径字节级回归:", "一致 ✓" if byte_equal else "不一致（需排查）")
print(f"\n{'run':<20} {'median':>8} {'last10':>8} {'final_min':>10} {'th1min':>8} {'th2max':>8} {'hit%':>7} {'time':>7}")
for name, r in results.items():
    print(f"{name:<20} {str(r['median']):>8} {str(r['last']):>8} {str(r['final_dist_min']):>10} "
          f"{str(r['theta1_min']):>8} {str(r['theta2_max']):>8} {str(r['hit_pct']):>7} {r['seconds']:>6.1f}s")
