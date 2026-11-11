"""两关节臂轨迹诊断：理论可达性 vs 实际末端行为。

复刻 host 的 combo1 奖励注入，逐 tick 读取 arm 状态，判断杆是
「物理不可达」「卡在关节边界」「单向匀速转圈」还是「随机游走」。
用法: py audit/arm_trajectory.py
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(os.environ.get("TEMP", ".")) / "snn_audit"
LIB = ctypes.CDLL(str(ROOT / "b2_core.dll"))

LIB.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint]
LIB.net2_init.restype = ctypes.c_int
LIB.net2_destroy.argtypes = [ctypes.c_void_p]
LIB.net2_tick.argtypes = [ctypes.c_void_p]
LIB.net2_set_target.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
LIB.net2_inject_reward.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
LIB.net2_get_snapshot.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_int)]

# --- 理论可达性：关节约束内网格搜索 ---------------------------------------
L1 = L2 = 1.0
TX, TY = 1.5, 0.5
grid = 601
t1 = np.linspace(-math.pi, math.pi, grid)
t2 = np.linspace(-0.9 * math.pi, 0.9 * math.pi, grid)
T1, T2 = np.meshgrid(t1, t2, indexing="ij")
EX = L1 * np.cos(T1) + L2 * np.cos(T1 + T2)
EY = L1 * np.sin(T1) + L2 * np.sin(T1 + T2)
D = np.hypot(EX - TX, EY - TY)
idx = np.unravel_index(np.argmin(D), D.shape)
reachable = {"min_dist": float(D[idx]), "theta1": float(T1[idx]), "theta2": float(T2[idx]),
             "init_dist": float(math.hypot(2.0 - TX, 0.0 - TY)),
             "threshold_dist": float(0.2 * math.hypot(2.0 - TX, 0.0 - TY))}

# --- 实际轨迹 ---------------------------------------------------------------
N = 100 + 56 + 2000 + 50 + 50
buf = ctypes.create_string_buffer(4_000_000)
net = ctypes.cast(buf, ctypes.c_void_p)
rc = LIB.net2_init(net, 100, 2000, 50, 50, 8.0, 42)
if rc != 0:
    raise SystemExit(f"net2_init failed: {rc}")
LIB.net2_set_target(net, TX, TY)

mem = (ctypes.c_double * N)()
spk = (ctypes.c_double * N)()
last = (ctypes.c_double * N)()
arm = (ctypes.c_double * 10)()
meta = (ctypes.c_int * 3)()

TICKS = 2000
trace = []
prev_ex, prev_ey = 2.0, 0.0
for t in range(TICKS):
    LIB.net2_tick(net)
    LIB.net2_get_snapshot(net, mem, spk, last, arm, meta)
    ex, ey = arm[4], arm[5]
    vx, vy = ex - prev_ex, ey - prev_ey
    dx, dy = TX - ex, TY - ey
    vmag, dmag = math.hypot(vx, vy), math.hypot(dx, dy)
    cos_r = (vx * dx + vy * dy) / (vmag * dmag) if vmag > 1e-6 and dmag > 1e-6 else 0.0
    prev_d = math.hypot(TX - prev_ex, TY - prev_ey)
    delta = max(-1.0, min(1.0, (prev_d - dmag) * 100.0))
    LIB.net2_inject_reward(net, 0.5 * cos_r + 0.5 * delta, 0)
    trace.append({"t": t, "ex": ex, "ey": ey, "th1": arm[0], "th2": arm[1],
                  "o1": arm[2], "o2": arm[3], "d": dmag, "r": arm[8]})
    prev_ex, prev_ey = ex, ey
LIB.net2_destroy(net)

d = [x["d"] for x in trace]
o1 = [x["o1"] for x in trace]
o2 = [x["o2"] for x in trace]
th1 = [x["th1"] for x in trace]
th2 = [x["th2"] for x in trace]
k = max(1, len(d) // 10)
stats = {
    "ticks": TICKS,
    "dist_start": round(d[0], 4), "dist_end": round(d[-1], 4),
    "dist_min": round(min(d), 4), "dist_max": round(max(d), 4),
    "dist_mean": round(statistics.fmean(d), 4),
    "first_decile_mean": round(statistics.fmean(d[:k]), 4),
    "last_decile_mean": round(statistics.fmean(d[-k:]), 4),
    "frac_within_threshold": round(sum(x <= reachable["threshold_dist"] for x in d) / len(d), 4),
    "theta1_range": [round(min(th1), 3), round(max(th1), 3)],
    "theta2_range": [round(min(th2), 3), round(max(th2), 3)],
    "theta1_at_boundary_frac": round(sum(abs(x) >= math.pi - 1e-6 for x in th1) / len(th1), 4),
    "theta2_at_boundary_frac": round(sum(abs(x) >= 0.9 * math.pi - 1e-6 for x in th2) / len(th2), 4),
    "omega1_mean": round(statistics.fmean(o1), 4), "omega1_std": round(statistics.pstdev(o1), 4),
    "omega2_mean": round(statistics.fmean(o2), 4), "omega2_std": round(statistics.pstdev(o2), 4),
    "omega1_nonzero_frac": round(sum(abs(x) > 1e-6 for x in o1) / len(o1), 4),
    "omega2_nonzero_frac": round(sum(abs(x) > 1e-6 for x in o2) / len(o2), 4),
    "endpoint_x_range": [round(min(x["ex"] for x in trace), 3), round(max(x["ex"] for x in trace), 3)],
    "endpoint_y_range": [round(min(x["ey"] for x in trace), 3), round(max(x["ey"] for x in trace), 3)],
}
samples = [trace[i] for i in range(0, TICKS, 100)]
payload = {"reachability": reachable, "trajectory_stats": stats,
           "samples_every_100_ticks": [{kk: round(vv, 4) for kk, vv in s.items()} for s in samples],
           "note": "单 episode 2000 ticks，seed=42，combo1 奖励，默认网络"}
(ROOT / "audit" / "arm_trajectory.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("理论可达:", json.dumps(reachable, ensure_ascii=False))
print("实际轨迹:", json.dumps(stats, ensure_ascii=False, indent=2))
