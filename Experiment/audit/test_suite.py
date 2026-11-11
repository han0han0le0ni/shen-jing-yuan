"""修复后完整测试套件：构建/ABI、边界、消融、输出、复现性、内存、后端、数据完整性。

用法: py audit/test_suite.py
输出: audit/test_results.json
所有运行产物写入系统 TEMP，不污染实验数据；仅重跑数据扫描器（统计应保持不变）。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "audit"
TMP = Path(os.environ.get("TEMP", ".")) / "snn_audit"
TMP.mkdir(parents=True, exist_ok=True)
PY = sys.executable
B1 = ROOT / "benchmark1.exe"
B2 = ROOT / "benchmark2.exe"

checks = []


def chk(cat, name, cond, detail="", seconds=0.0):
    checks.append({"category": cat, "name": name, "passed": bool(cond),
                   "detail": str(detail).replace("\ufffd", "?")[:260],
                   "seconds": round(seconds, 2)})


def run(args, cwd=None, timeout=180):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired:
        return -99, "TIMEOUT", time.time() - t0


def b2_out(tag, episodes=3, ticks=50, extra=()):
    out = TMP / f"test_{tag}.csv"
    args = [str(B2), "--episodes", str(episodes), "--ticks", str(ticks),
            "--out", str(out)] + list(extra)
    rc, text, dur = run(args, cwd=TMP)
    return rc, text, out, dur


B2_HEADER = "episode,target_x,target_y,init_dist,final_dist,dist_ratio,final_reward"
B1_HEADER = ("trial,stimulus_type,predicted_vote,correct,output_a_rate,"
             "output_b_rate,reward_given,reward_baseline,elapsed_ticks")


def csv_rows(path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.reader(f))


def validate_b2(path, expected):
    rows = csv_rows(path)
    if not rows or ",".join(rows[0]) != B2_HEADER:
        return False, "header mismatch"
    body = rows[1:]
    if len(body) != expected:
        return False, f"rows={len(body)} expected={expected}"
    bad = 0
    for r in body:
        if len(r) != 7:
            bad += 1
            continue
        try:
            v = [float(x) for x in r]
        except ValueError:
            bad += 1
            continue
        if not all(math.isfinite(x) for x in v):
            bad += 1
            continue
        init, final, ratio, rew = v[3], v[4], v[5], v[6]
        if init <= 0 or abs(ratio - final / init) > 0.0011:
            bad += 1
        if final <= 2.0 and abs(rew - (1.0 - final / 2.0)) > 0.002:
            bad += 1
    return bad == 0, f"rows={len(body)} invalid={bad}"


def validate_b1(path, expected):
    rows = csv_rows(path)
    if not rows or ",".join(rows[0]) != B1_HEADER:
        return False, "header mismatch"
    body = rows[1:]
    if len(body) != expected:
        return False, f"rows={len(body)} expected={expected}"
    bad = 0
    for r in body:
        if len(r) != 9:
            bad += 1
            continue
        try:
            v = [float(x) for x in r]
        except ValueError:
            bad += 1
            continue
        stim, pred, cor = v[1], v[2], v[3]
        if int(cor) != int(pred == (1 if stim == 0 else 2)):
            bad += 1
    return bad == 0, f"rows={len(body)} invalid={bad}"


# ---------------------------------------------------------------- A 构建 / ABI
CAT = "A 构建/ABI"
rc, out, dur = run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(ROOT / "build.ps1")], timeout=180)
chk(CAT, "build.ps1 可重复构建", rc == 0 and "Build OK" in out,
    out.strip().replace("\n", " | ")[-150:], dur)

info_path = ROOT / "build_info.json"
info = json.loads(info_path.read_text(encoding="utf-8-sig")) if info_path.is_file() else {}
hash_ok = True
for rel, digest in (info.get("sources") or {}).items():
    p = ROOT / rel.replace("\\", "/")
    hash_ok &= p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
for name, digest in (info.get("artifacts") or {}).items():
    p = ROOT / name
    hash_ok &= p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
chk(CAT, "build_info.json 哈希一致", bool(info) and hash_ok,
    f"sources={len(info.get('sources') or {})} artifacts={len(info.get('artifacts') or {})}")

warn = []
for label, tool, src in (("B2 core", "gcc", "brain_b2/benchmark2_core.c"),
                         ("B2 host", "g++", "host_b2/benchmark2_host.cpp"),
                         ("B1 core", "gcc", "brain/benchmark1_core.c"),
                         ("B1 host", "g++", "host/benchmark1_host.cpp")):
    rc, out, _ = run([tool, "-O2", "-Wall", "-Wextra", "-fsyntax-only", str(ROOT / src)])
    if rc != 0 or "warning:" in out:
        warn.append(f"{label}: {out.strip()[:70]}")
chk(CAT, "4 个编译单元零告警", not warn, "; ".join(warn) or "gcc/g++ -Wall -Wextra clean")

rc, out, _ = run(["nm", "-g", "--defined-only", str(ROOT / "b2_core.dll")])
need = {"net2_inject_reward2", "net2_neuron_count", "net2_synapse_count",
        "net2_get_topology", "net2_get_snapshot", "net2_init"}
missing = sorted(need - set(out.split()))
chk(CAT, "DLL 导出符号齐全", rc == 0 and not missing, f"missing={missing}")

# ---------------------------------------------------------------- B 参数边界
CAT = "B 参数边界"
for label, args in (("episodes=0", ["--episodes", "0"]),
                    ("ticks=0", ["--ticks", "0"]),
                    ("input=1", ["--input", "1"]),
                    ("motor=1", ["--motor", "1"]),
                    ("hidden=0", ["--hidden", "0"])):
    rc, out, dur = run([str(B2)] + args + ["--out", str(TMP / "invalid.csv")], cwd=TMP)
    chk(CAT, f"B2 拒绝 {label}", rc == 1,
        out.strip().replace("\n", " | ")[-110:], dur)
rc, out, dur = run([str(B1), "--trials", "1"], cwd=TMP)
chk(CAT, "B1 拒绝 trials=1", rc == 1 and "Invalid config" in out, out.strip()[-90:], dur)
rc, out, dur = run([str(B2), "--help"], cwd=TMP)
chk(CAT, "B2 --help 正常", rc == 0 and "Usage:" in out and "--disable_trace" in out,
    out.strip().replace("\n", " | ")[:120], dur)
rc, out, dur = run([str(B1), "--help"], cwd=TMP)
chk(CAT, "B1 --help 正常", rc == 0 and "Usage:" in out, out.strip().replace("\n", " | ")[:110], dur)

# ---------------------------------------------------------------- C 消融开关
CAT = "C 消融开关"
rc, out, _, dur = b2_out("ablation", episodes=2, ticks=20,
                         extra=("--disable_trace", "--disable_reward"))
chk(CAT, "B2 双消融生效", "El Trace: 0, Reward Mod: 0" in out,
    [x for x in out.splitlines() if "El Trace" in x], dur)
b1_ab = TMP / "test_b1_ablation.csv"
rc, out, dur = run([str(B1), "--trials", "50", "--disable_trace",
                    "--out", str(b1_ab)], cwd=TMP)
chk(CAT, "B1 资格迹消融生效", "OFF (ABLATION)" in out,
    [x for x in out.splitlines() if "El Trace" in x], dur)

# ---------------------------------------------------------------- D 输出与元数据
CAT = "D 输出/元数据"
rc, out, csv2, dur = b2_out("short3")
ok, detail = validate_b2(csv2, 3) if csv2.is_file() else (False, "no CSV")
chk(CAT, "B2 短跑 CSV 结构与公式", ok, detail, dur)
meta2 = Path(str(csv2) + ".meta.json")
if meta2.is_file():
    m = json.loads(meta2.read_text(encoding="utf-8"))
    chk(CAT, "B2 sidecar 字段完整",
        m.get("benchmark") == 2 and m.get("episodes") == 3 and m.get("target_mode") == 1
        and m.get("el_trace") == 1 and m.get("neurons", 0) > 0,
        json.dumps(m, ensure_ascii=False)[:170])
else:
    chk(CAT, "B2 sidecar 字段完整", False, "missing")

b1_csv = TMP / "test_b1_500.csv"
rc, out, dur = run([str(B1), "--trials", "500", "--seed", "7", "--out", str(b1_csv)], cwd=TMP)
ok, detail = validate_b1(b1_csv, 500) if b1_csv.is_file() else (False, "no CSV")
chk(CAT, "B1 500 trials CSV 结构", ok, detail, dur)
chk(CAT, "B1 500 trials 通过判据", rc == 0 and "RESULT: PASS" in out,
    out.strip().replace("\n", " | ")[-120:], dur)
meta1 = Path(str(b1_csv) + ".meta.json")
chk(CAT, "B1 sidecar 字段完整",
    meta1.is_file() and json.loads(meta1.read_text(encoding="utf-8")).get("benchmark") == 1,
    "ok" if meta1.is_file() else "missing")

# ---------------------------------------------------------------- E 复现性
CAT = "E 随机复现性"
_, _, a, dur = b2_out("seed_a", extra=("--seed", "12345"))
_, _, b, _ = b2_out("seed_b", extra=("--seed", "12345"))
chk(CAT, "同 seed 两次运行字节一致",
    a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes(),
    f"{a.stat().st_size if a.is_file() else 0} bytes", dur)
_, _, c, _ = b2_out("seed_c", extra=("--seed", "999"))
chk(CAT, "不同 seed 结果不同",
    a.is_file() and c.is_file() and a.read_bytes() != c.read_bytes(), "different seeds differ")

# ---------------------------------------------------------------- F 内存压力
CAT = "F 内存压力"
pressure = [(2, 1, 2, 1.0), (10, 5, 2, 3.0), (36, 200, 10, 3.0),
            (36, 200, 10, 8.0), (50, 100, 5, 8.0), (100, 2000, 50, 8.0)]
for inp, hid, mot, k in pressure:
    rc, out, dur = run([PY, str(HERE / "check_config.py"), str(inp), str(hid), str(mot), str(k)],
                       timeout=120)
    syn = "?"
    for line in out.splitlines():
        if "synapses =" in line:
            syn = line.split("synapses =")[-1].strip()
    chk(CAT, f"net2_init({inp},{hid},{mot},{mot},k={k:g})", rc == 0 and "destroy ok" in out,
        f"synapses={syn}, exit={rc}", dur)

# ---------------------------------------------------------------- G 可视化后端
CAT = "G 可视化后端"
rc, out, dur = run([PY, str(HERE / "test_backend_cpu.py")], timeout=120)
chk(CAT, "CPUBackend 拓扑/快照/tick 端到端",
    rc == 0 and "BACKEND_OK" in out, out.strip().replace("\n", " | ")[-180:], dur)

# ---------------------------------------------------------------- H 数据完整性
CAT = "H 数据完整性"
old = json.loads((HERE / "scan_results.json").read_text(encoding="utf-8"))
keys = [("quality", "b2_csv_count"), ("quality", "b1_csv_count"),
        ("quality", "b2_total_episodes"), ("quality", "b1_total_trials"),
        ("summary", "b2_pass"), ("summary", "b2_fail"),
        ("summary", "b1_seed_pass"), ("summary", "root_log_fail")]
old_vals = {k: old[k[0]][k[1]] for k in keys}
rc, out, dur = run([PY, str(HERE / "scan_data.py")], timeout=180)
new = json.loads((HERE / "scan_results.json").read_text(encoding="utf-8"))
diffs = {str(k): (old_vals[k], new[k[0]][k[1]])
         for k in keys if old_vals[k] != new[k[0]][k[1]]}
chk(CAT, "全量数据重扫统计不变（未污染实验数据）", rc == 0 and not diffs,
    f"diffs={diffs}" if diffs else
    f"CSV={new['quality']['b2_csv_count'] + new['quality']['b1_csv_count']}, "
    f"episodes={new['quality']['b2_total_episodes']}", dur)

# ---------------------------------------------------------------- I 中等规模稳定性
CAT = "I 稳定性"
rc, out, big, dur = b2_out("mid50", episodes=50, ticks=500)
ok, detail = validate_b2(big, 50) if big.is_file() else (False, "no CSV")
chk(CAT, "50 ep x 500 ticks 长跑无崩溃/无 NaN", rc in (0, 2) and ok,
    f"{detail}, exit={rc}, {dur:.1f}s", dur)
chk(CAT, "长跑输出 FINAL RESULTS 与判据行",
    "FINAL RESULTS" in out and ("[PASS]" in out or "[FAIL]" in out),
    [x for x in out.splitlines() if "Median:" in x or "PASS CHECK" in x][:3], dur)

# ---------------------------------------------------------------- 汇总
by_cat = {}
for c in checks:
    d = by_cat.setdefault(c["category"], {"total": 0, "passed": 0})
    d["total"] += 1
    d["passed"] += int(c["passed"])
passed = sum(c["passed"] for c in checks)
payload = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "root": str(ROOT), "total": len(checks), "passed": passed,
    "failed": [c["name"] for c in checks if not c["passed"]],
    "categories": by_cat, "checks": checks,
}
(HERE / "test_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
print("=" * 82)
for c in checks:
    print(("PASS  " if c["passed"] else "FAIL  ") + f"[{c['category']}] {c['name']}"
          + ("  — " + c["detail"] if c["detail"] else ""))
print("=" * 82)
for cat, d in by_cat.items():
    print(f"  {cat}: {d['passed']}/{d['total']}")
print("TOTAL", passed, "/", len(checks))
if passed != len(checks):
    raise SystemExit(1)
