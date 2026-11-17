"""诊断 omega_cmd 数值"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from backend_vec import VecBackend

b = VecBackend(device="gpu", seed=42)
b.init(100, 2000, 50, 50, seed=42)

m1_base = 100 + 2000
m2_base = m1_base + 50

# 跑 500 tick warmup
for _ in range(500):
    b.tick()

# 然后跑 200 tick 收集 omega_cmd
# 手动从 motor 层解码（和 C 端同样的逻辑）
omega1_list = []
omega2_list = []
for _ in range(200):
    b.tick()
    mem, spk, last, arm, meta = b.get_snapshot_arrays()
    
    # 解码逻辑（方向池分离平均，与 C 端 + VecBackend 一致）
    tick = meta[0]
    
    omega1_cmd = 0.0
    pos_c, neg_c, pos_s, neg_s = 0, 0, 0.0, 0.0
    for i in range(50):
        lt = last[m1_base + i]
        active = 1 if (lt >= 0 and tick - lt >= 0 and tick - lt <= 5) else 0
        pref = -1.0 + 2.0 * i / 49.0
        if active:
            if pref > 0:
                pos_c += 1; pos_s += pref
            elif pref < 0:
                neg_c += 1; neg_s += pref
    if pos_c > neg_c and pos_c > 0:
        omega1_cmd = pos_s / pos_c
    elif neg_c > pos_c and neg_c > 0:
        omega1_cmd = neg_s / neg_c
    
    omega2_cmd = 0.0
    pos_c, neg_c, pos_s, neg_s = 0, 0, 0.0, 0.0
    for i in range(50):
        lt = last[m2_base + i]
        active = 1 if (lt >= 0 and tick - lt >= 0 and tick - lt <= 5) else 0
        pref = -1.0 + 2.0 * i / 49.0
        if active:
            if pref > 0:
                pos_c += 1; pos_s += pref
            elif pref < 0:
                neg_c += 1; neg_s += pref
    if pos_c > neg_c and pos_c > 0:
        omega2_cmd = pos_s / pos_c
    elif neg_c > pos_c and neg_c > 0:
        omega2_cmd = neg_s / neg_c
    
    omega1_list.append(omega1_cmd)
    omega2_list.append(omega2_cmd)

o1 = np.array(omega1_list)
o2 = np.array(omega2_list)
print(f"omega1_cmd: mean={o1.mean():.4f}, std={o1.std():.4f}, range=[{o1.min():.3f}, {o1.max():.3f}]")
print(f"omega2_cmd: mean={o2.mean():.4f}, std={o2.std():.4f}, range=[{o2.min():.3f}, {o2.max():.3f}]")
print()
print(f"|omega1| > 0.1 的比例: {(np.abs(o1) > 0.1).mean()*100:.1f}%")
print(f"|omega2| > 0.1 的比例: {(np.abs(o2) > 0.1).mean()*100:.1f}%")
print(f"|omega1| > 0.05 的比例: {(np.abs(o1) > 0.05).mean()*100:.1f}%")
print(f"|omega2| > 0.05 的比例: {(np.abs(o2) > 0.05).mean()*100:.1f}%")
print()
# omega*ARM_DT*1000 tick = 臂的总关节转角
# 如果 omega=0.05, dt=0.02, 1000 tick → 转角 = 0.05*0.02*1000 = 1 rad = 57 度
# 这个角度其实不小了！但 max_omega=0.5 clamp 后 0.05*0.02=0.001 每 tick
# 让我算实际末端位移：theta 从 0 转到 0.001 → 末端 ≈ (2*cos(0.001) - 1, 2*sin(0.001)) ≈ (0.9999, 0.002)
# 完全看不出来

# 换个思路：看 motor 层最近 5 tick 发过多少个神经元
total_active = []
for _ in range(200):
    b.tick()
    mem, spk, last, arm, meta = b.get_snapshot_arrays()
    tick = meta[0]
    a1 = sum(1 for i in range(50) if last[m1_base+i] >= 0 and tick - last[m1_base+i] <= 5)
    a2 = sum(1 for i in range(50) if last[m2_base+i] >= 0 and tick - last[m2_base+i] <= 5)
    total_active.append(a1 + a2)

print(f"motor 层最近 5tick 总活跃神经元: mean={np.mean(total_active):.1f}, range=[{min(total_active)}, {max(total_active)}]")
print(f"  （每关节平均 {np.mean(total_active)/2:.1f} 个，总共 50 个神经元）")
