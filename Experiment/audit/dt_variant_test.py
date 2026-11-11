"""不改仓库源码：在 TEMP 编译 ARM_DT 变体 DLL，比较末端能否靠近目标。

假设：ARM_DT=0.02 使每 tick 角位移仅 0.01 rad，网络输出的零均值小幅度 omega
在有限 ticks 内无法把 theta2 推到目标所需的 ~1.32 rad（任务时间尺度失配）。
用法: py audit/dt_variant_test.py
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(os.environ.get("TEMP", ".")) / "snn_audit" / "variants"
TMP.mkdir(parents=True, exist_ok=True)
TX, TY = 1.5, 0.5
INIT_D = math.hypot(2.0 - TX, TY)
THRESH_D = 0.2 * INIT_D
N = 2256
TICKS = 2000


def make_variant(dt: float) -> Path:
    tag = str(dt).replace(".", "")
    v = TMP / f"dt{tag}"
    src_dir = v / "brain_b2"
    src_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "brain_b2" / "benchmark2_core.c", src_dir / "benchmark2_core.c")
    hdr = (ROOT / "brain_b2" / "benchmark2_core.h").read_text(encoding="utf-8")
    hdr2, n = re.subn(r"(#define ARM_DT\s+)0\.02", r"\g<1>" + str(dt), hdr)
    assert n == 1, f"ARM_DT macro not found ({n})"
    (src_dir / "benchmark2_core.h").write_text(hdr2, encoding="utf-8")
    dll = v / "b2_core.dll"
    subprocess.run(["gcc", "-O2", "-shared", str(src_dir / "benchmark2_core.c"),
                    "-o", str(dll), "-lm"], check=True, capture_output=True)
    return dll


def probe(dll: Path, dt: float) -> dict:
    L = ctypes.CDLL(str(dll))
    L.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint]
    L.net2_init.restype = ctypes.c_int
    L.net2_destroy.argtypes = [ctypes.c_void_p]
    L.net2_tick.argtypes = [ctypes.c_void_p]
    L.net2_set_target.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    L.net2_inject_reward.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    L.net2_get_snapshot.argtypes = [ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_int)]
    buf = ctypes.create_string_buffer(4_000_000)
    net = ctypes.cast(buf, ctypes.c_void_p)
    assert L.net2_init(net, 100, 2000, 50, 50, 8.0, 42) == 0
    L.net2_set_target(net, TX, TY)
    mem, spk, last = (ctypes.c_double * N)(), (ctypes.c_double * N)(), (ctypes.c_double * N)()
    arm, meta = (ctypes.c_double * 10)(), (ctypes.c_int * 3)()
    prev_ex, prev_ey = 2.0, 0.0
    dists, th1s, th2s, o1s, o2s = [], [], [], [], []
    for _ in range(TICKS):
        L.net2_tick(net)
        L.net2_get_snapshot(net, mem, spk, last, arm, meta)
        ex, ey = arm[4], arm[5]
        d = math.hypot(TX - ex, TY - ey)
        vx, vy, dx, dy = ex - prev_ex, ey - prev_ey, TX - ex, TY - ey
        vmag = math.hypot(vx, vy)
        cos_r = (vx * dx + vy * dy) / (vmag * d) if vmag > 1e-6 and d > 1e-6 else 0.0
        prev_d = math.hypot(TX - prev_ex, TY - prev_ey)
        delta = max(-1.0, min(1.0, (prev_d - d) * 100.0))
        L.net2_inject_reward(net, 0.5 * cos_r + 0.5 * delta, 0)
        dists.append(d); th1s.append(arm[0]); th2s.append(arm[1])
        o1s.append(arm[2]); o2s.append(arm[3])
        prev_ex, prev_ey = ex, ey
    L.net2_destroy(net)
    k = max(1, len(dists) // 10)
    return {
        "dt": dt, "dist_min": round(min(dists), 4), "dist_end": round(dists[-1], 4),
        "dist_mean": round(statistics.fmean(dists), 4),
        "first10": round(statistics.fmean(dists[:k]), 4),
        "last10": round(statistics.fmean(dists[-k:]), 4),
        "frac_within_threshold": round(sum(x <= THRESH_D for x in dists) / len(dists), 4),
        "theta1_min": round(min(th1s), 3), "theta1_max": round(max(th1s), 3),
        "theta2_max": round(max(th2s), 3),
        "omega1_mean": round(statistics.fmean(o1s), 4),
        "omega2_mean": round(statistics.fmean(o2s), 4),
    }


results = []
for dt in (0.02, 0.05, 0.10, 0.20):
    dll = ROOT / "b2_core.dll" if dt == 0.02 else make_variant(dt)
    results.append(probe(dll, dt))

payload = {"config": {"ticks": TICKS, "seed": 42, "reward": "combo1",
                      "threshold_dist": THRESH_D, "target": [TX, TY],
                      "theta_needed": [-0.335, 1.319]},
           "results": results,
           "note": "变体 DLL 仅编译于系统 TEMP，仓库源码/产物未改动"}
(ROOT / "audit" / "dt_variant.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{'ARM_DT':>8} {'min':>7} {'end':>7} {'last10':>8} {'th1max':>8} {'th2max':>8} {'hit%':>7}")
for r in results:
    print(f"{r['dt']:>8} {r['dist_min']:>7} {r['dist_end']:>7} {r['last10']:>8} "
          f"{r['theta1_max']:>8} {r['theta2_max']:>8} "
          f"{r['frac_within_threshold'] * 100:>6.1f}%")
