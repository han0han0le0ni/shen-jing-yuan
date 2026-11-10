"""Benchmark-2 大脑综合诊断 - v2"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from backend_vec import VecBackend

L1, L2 = 1.0, 1.0

def verify_kinematics():
    print("=== 运动学验证 ===")
    tx, ty = 1.5, 0.5
    d = math.hypot(tx, ty)
    print(f"target ({tx}, {ty}), dist from origin = {d:.3f}, max_reach={L1+L2}")
    print(f"可达? {d <= L1+L2}")
    
    # 逆运动学
    c2 = (d*d - L1*L1 - L2*L2) / (2*L1*L2)
    c2 = np.clip(c2, -1, 1)
    t2 = np.arccos(c2) * np.sign(np.arctan2(ty, tx))
    t1 = np.arctan2(ty, tx) - np.arctan2(L2 * np.sqrt(1 - c2*c2), L1 + L2*c2)
    
    print(f"需要旋转: theta1 {np.degrees(t1):.1f}°, theta2 {np.degrees(t2):.1f}°")
    omega1_need = (t1 - 0) / (1000 * 0.02)
    omega2_need = (t2 - 0) / (1000 * 0.02)
    print(f"需要 omega: omega1={omega1_need:.4f}, omega2={omega2_need:.4f} rad/s")
    print(f"max_omega=0.5, 足够? {abs(omega1_need)<=0.5 and abs(omega2_need)<=0.5}")
    print()

def run_once(tx_val, ty_val, warmup=100, collect=200):
    be = VecBackend()
    be.init(100, 2000, 50, 50)
    be.set_target(tx_val, ty_val)
    N_in, N_h = 100, 2000
    all_hid = np.zeros(N_h, dtype=np.int32)
    m1_base = N_in + N_h
    m2_base = m1_base + 50
    all_m1 = np.zeros(50, dtype=np.int32)
    all_m2 = np.zeros(50, dtype=np.int32)
    
    for _ in range(warmup):
        be.tick()
    for _ in range(collect):
        be.tick()
        idx = be.tick_count - 1
        all_hid += (be.last_spike_tick[100:2100] == idx).astype(np.int32)
        all_m1 += (be.last_spike_tick[m1_base:m1_base+50] == idx).astype(np.int32)
        all_m2 += (be.last_spike_tick[m2_base:m2_base+50] == idx).astype(np.int32)
    return all_hid, all_m1, all_m2

def jaccard(a, b):
    s1, s2 = set(np.where(a > 0)[0]), set(np.where(b > 0)[0])
    if not s1 and not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def hidden_sensitivity():
    print("=== Hidden Target Sensitivity ===")
    
    h1, m1a, m2a = run_once(1.5, 0.5)
    h2, m1b, m2b = run_once(1.5, 0.5)
    h3, m1c, m2c = run_once(0.5, -1.0)
    h4, _, _ = run_once(-0.5, 0.8)
    
    j_same = jaccard(h1, h2)
    j_diff1 = jaccard(h1, h3)
    j_diff2 = jaccard(h1, h4)
    
    set1 = set(np.where(h1 > 0)[0])
    print(f"hidden 活跃数: run1={len(set1)}/2000, spike/tick={np.mean(h1):.1f}")
    print(f"motor1 活跃: {np.mean([np.sum(m1a>0), np.sum(m1b>0), np.sum(m1c>0)]):.1f}/50")
    print(f"motor2 活跃: {np.mean([np.sum(m2a>0), np.sum(m2b>0), np.sum(m2c>0)]):.1f}/50")
    print()
    print(f"同一 target 两次 Jaccard: {j_same:.4f}")
    print(f"不同 target 1 Jaccard:    {j_diff1:.4f}")
    print(f"不同 target 2 Jaccard:    {j_diff2:.4f}")
    print()
    
    if j_same < 0.05:
        print("❌ Hidden 极不稳定（同一 target 两次几乎不重叠）")
    elif j_same <= max(j_diff1, j_diff2) * 1.1:
        print("❌ Hidden 对 target 不敏感（不同 target 相似度 ≈ 同一 target）")
    else:
        print("✅ Hidden 稳定且对 target 敏感")

if __name__ == "__main__":
    verify_kinematics()
    hidden_sensitivity()
