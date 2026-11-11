"""奖赏机制 A/B 对照：默认 / 关闭奖励调制 / 关闭资格迹。

同 seed、同网络、同 episode 数，比较学习曲线与 I->M 权重变化，
判断 reward 通道是否真的驱动学习（而不只是打印了开关）。
用法: py audit/reward_ablation.py
"""
from __future__ import annotations

import csv
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

EPISODES, TICKS, SEED = 50, 500, 42
RUNS = {
    "default": [],
    "no_reward": ["--disable_reward"],
    "no_trace": ["--disable_trace"],
}
WSTAT = re.compile(r"\[wstats\].*?mean=([-\d.]+).*?max=([-\d.]+).*?satHI=(\d+)")


def summarize(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    ratios = [float(r[5]) for r in rows[1:] if len(r) == 7]
    k = max(1, len(ratios) // 10)
    return {
        "n": len(ratios),
        "median_ratio": round(statistics.median(ratios), 4),
        "mean_ratio": round(statistics.fmean(ratios), 4),
        "first_decile": round(statistics.median(ratios[:k]), 4),
        "last_decile": round(statistics.median(ratios[-k:]), 4),
        "learning_delta": round(statistics.median(ratios[:k]) - statistics.median(ratios[-k:]), 4),
        "frac_le_0_2": round(sum(x <= 0.2 for x in ratios) / len(ratios), 4),
    }


results = {}
for name, extra in RUNS.items():
    out = TMP / f"ab_{name}.csv"
    t0 = time.time()
    p = subprocess.run(
        [str(EXE), "--episodes", str(EPISODES), "--ticks", str(TICKS),
         "--seed", str(SEED), "--out", str(out)] + extra,
        cwd=TMP, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=240, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    text = (p.stdout or "") + (p.stderr or "")
    stats = summarize(out) if out.is_file() else {}
    wlines = WSTAT.findall(text)
    first_w = wlines[0] if wlines else None
    last_w = wlines[-1] if wlines else None
    final = re.search(r"Median:\s*([\d.]+)", text)
    verdict = re.search(r"\[(PASS|FAIL)\]", text)
    results[name] = {
        "exit": p.returncode, "seconds": round(time.time() - t0, 1),
        "log_median": float(final.group(1)) if final else None,
        "verdict": verdict.group(1) if verdict else None,
        "stats": stats,
        "wstats_first": {"mean": float(first_w[0]), "max": float(first_w[1]), "satHI": int(first_w[2])} if first_w else None,
        "wstats_last": {"mean": float(last_w[0]), "max": float(last_w[1]), "satHI": int(last_w[2])} if last_w else None,
        "csv_bytes": out.stat().st_size if out.is_file() else 0,
        "csv_sha16": __import__("hashlib").sha256(out.read_bytes()).hexdigest()[:16] if out.is_file() else None,
    }

d, nr, nt = results["default"], results["no_reward"], results["no_trace"]
comparison = {
    "default_vs_no_reward_median_delta": round(nr["stats"]["median_ratio"] - d["stats"]["median_ratio"], 4),
    "default_vs_no_trace_median_delta": round(nt["stats"]["median_ratio"] - d["stats"]["median_ratio"], 4),
    "reward_channel_changes_trajectory": d["csv_sha16"] != nr["csv_sha16"],
    "trace_channel_changes_trajectory": d["csv_sha16"] != nt["csv_sha16"],
    "default_learning_delta": d["stats"]["learning_delta"],
    "no_reward_learning_delta": nr["stats"]["learning_delta"],
    "no_trace_learning_delta": nt["stats"]["learning_delta"],
    "passes_threshold": d["stats"]["median_ratio"] <= 0.20,
}
payload = {
    "config": {"episodes": EPISODES, "ticks": TICKS, "seed": SEED,
               "note": "单种子探索性对照，非统计显著结论"},
    "runs": results, "comparison": comparison,
}
(ROOT / "audit" / "reward_ablation.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"配置: {EPISODES} ep x {TICKS} ticks, seed={SEED}\n")
for name, r in results.items():
    s = r["stats"]
    print(f"{name:<12} exit={r['exit']} {r['seconds']:>5.1f}s  median={s.get('median_ratio')} "
          f"first={s.get('first_decile')} last={s.get('last_decile')} "
          f"delta={s.get('learning_delta')} frac<=0.2={s.get('frac_le_0_2')} "
          f"verdict={r['verdict']}")
    print(f"             wstats first={r['wstats_first']} last={r['wstats_last']}")
print("\n对照:", json.dumps(comparison, ensure_ascii=False))
