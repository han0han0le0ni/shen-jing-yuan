"""奖励策略诊断探针（不改仓库代码）：验证能否打破「只抬肩、不弯肘」局部最优。

用当前 DLL 的 net2_inject_reward2 双通道接口，比较 4 种奖励设计。
每组独立初始化（同 seed）、单 episode 2000 ticks，记录关节角与距离。
用法: py audit/reward_strategy_probe.py
"""
from __future__ import annotations

import ctypes
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
L = ctypes.CDLL(str(ROOT / "b2_core.dll"))
L.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint]
L.net2_init.restype = ctypes.c_int
L.net2_destroy.argtypes = [ctypes.c_void_p]
L.net2_tick.argtypes = [ctypes.c_void_p]
L.net2_set_target.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
L.net2_inject_reward.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
L.net2_inject_reward2.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
L.net2_get_snapshot.argtypes = [ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_int)]

N = 2256
TX, TY = 1.5, 0.5
INIT_D = math.hypot(2.0 - TX, 0.0 - TY)
THRESH_D = 0.2 * INIT_D
TICKS = 2000
L1 = 1.0


def run(mode: str, seed: int = 42) -> dict:
    buf = ctypes.create_string_buffer(4_000_000)
    net = ctypes.cast(buf, ctypes.c_void_p)
    if L.net2_init(net, 100, 2000, 50, 50, 8.0, seed) != 0:
        raise RuntimeError("init failed")
    L.net2_set_target(net, TX, TY)
    mem = (ctypes.c_double * N)()
    spk = (ctypes.c_double * N)()
    last = (ctypes.c_double * N)()
    arm = (ctypes.c_double * 10)()
    meta = (ctypes.c_int * 3)()

    prev_ex, prev_ey = 2.0, 0.0
    dists, th1s, th2s = [], [], []
    for t in range(TICKS):
        L.net2_tick(net)
        L.net2_get_snapshot(net, mem, spk, last, arm, meta)
        ex, ey, th1, th2 = arm[4], arm[5], arm[0], arm[1]
        d = math.hypot(TX - ex, TY - ey)
        vx, vy = ex - prev_ex, ey - prev_ey
        dx, dy = TX - ex, TY - ey
        vmag = math.hypot(vx, vy)
        cos_r = (vx * dx + vy * dy) / (vmag * d) if vmag > 1e-6 and d > 1e-6 else 0.0
        prev_d = math.hypot(TX - prev_ex, TY - prev_ey)
        delta = max(-1.0, min(1.0, (prev_d - d) * 100.0))
        combo = 0.5 * cos_r + 0.5 * delta
        progress = 1.0 - d / INIT_D

        elb_x, elb_y = L1 * math.cos(th1), L1 * math.sin(th1)
        psi2 = math.atan2(TY - elb_y, TX - elb_x) - (th1 + th2)
        c2 = math.cos(psi2)

        if mode == "combo1":
            L.net2_inject_reward(net, combo, 0)
        elif mode == "combo1_abs":
            L.net2_inject_reward(net, combo + 0.5 * progress, 0)
        elif mode == "dual_psi2":
            L.net2_inject_reward2(net, delta + 0.3 * progress,
                                  c2 + 0.3 * progress, 0)
        elif mode == "dual_psi2_strong":
            L.net2_inject_reward2(net, delta, 2.0 * c2 + delta, 0)
        else:
            raise ValueError(mode)

        dists.append(d)
        th1s.append(th1)
        th2s.append(th2)
        prev_ex, prev_ey = ex, ey
    L.net2_destroy(net)
    k = max(1, len(dists) // 10)
    return {
        "mode": mode,
        "dist_start": round(dists[0], 4), "dist_end": round(dists[-1], 4),
        "dist_min": round(min(dists), 4),
        "dist_mean": round(statistics.fmean(dists), 4),
        "first_decile_mean": round(statistics.fmean(dists[:k]), 4),
        "last_decile_mean": round(statistics.fmean(dists[-k:]), 4),
        "frac_within_threshold": round(sum(x <= THRESH_D for x in dists) / len(dists), 4),
        "theta1_min": round(min(th1s), 3), "theta1_max": round(max(th1s), 3),
        "theta2_min": round(min(th2s), 3), "theta2_max": round(max(th2s), 3),
        "theta2_target": 1.319, "theta1_target": -0.335,
    }


results = [run(m) for m in ("combo1", "combo1_abs", "dual_psi2", "dual_psi2_strong")]
payload = {"config": {"ticks": TICKS, "seed": 42, "network": "100/2000/50/50 avg_conn=8",
                      "threshold_dist": THRESH_D, "lr": "core default 0.15 (host uses 0.35 for ep<500)"},
           "results": results,
           "note": "单 episode 探索性诊断，非最终达标结论"}
(ROOT / "audit" / "reward_strategy.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{'mode':<20} {'min':>7} {'end':>7} {'first10':>8} {'last10':>8} "
      f"{'th1min':>8} {'th2max':>8} {'hit%':>7}")
for r in results:
    print(f"{r['mode']:<20} {r['dist_min']:>7} {r['dist_end']:>7} "
          f"{r['first_decile_mean']:>8} {r['last_decile_mean']:>8} "
          f"{r['theta1_min']:>8} {r['theta2_max']:>8} "
          f"{r['frac_within_threshold'] * 100:>6.1f}%")
