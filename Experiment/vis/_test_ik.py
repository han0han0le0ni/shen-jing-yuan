"""运动学 baseline：直接用逆运动学算 omega_cmd → 确认任务可解"""
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))

L1, L2 = 1.0, 1.0
max_omega = 0.5
dt = 0.02
max_reach = 2.0

def ik(x, y):
    """逆运动学 (x,y) → (theta1, theta2) 两解取 elbow-up"""
    d2 = x*x + y*y
    d = math.sqrt(d2)
    if d > L1 + L2 - 0.01:
        return None, None
    if d < abs(L1 - L2) + 0.01:
        return None, None
    c2 = (d2 - L1*L1 - L2*L2) / (2*L1*L2)
    c2 = max(-1, min(1, c2))
    s2 = math.sqrt(1 - c2*c2)
    # elbow-up 解（theta2 负）
    theta2 = math.atan2(-s2, c2)
    theta1 = math.atan2(y, x) - math.atan2(L2*math.sin(theta2), L1 + L2*math.cos(theta2))
    return theta1, theta2

def fk(theta1, theta2):
    """正运动学 → (x, y)"""
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return x, y

def run_episode(tx, ty, ticks=1000):
    """运动学控制：每 tick 用 IK 算需要的关节角度差 → omega"""
    t1, t2 = 0.0, 0.0
    init_dist = math.hypot(tx - fk(t1, t2)[0], ty - fk(t1, t2)[1])
    
    target_t1, target_t2 = ik(tx, ty)
    if target_t1 is None:
        return 999.9
    
    for tick in range(ticks):
        # 指数趋近：每 tick 向 target 靠近 5%（比线性快得多）
        # rate = 0.05 → 约 60 tick 达稳态
        rate = 0.1
        omega1 = (target_t1 - t1) * rate
        omega2 = (target_t2 - t2) * rate
        # clamp max_omega
        omega1 = max(-max_omega, min(max_omega, omega1))
        omega2 = max(-max_omega, min(max_omega, omega2))
        t1 += omega1 * dt
        t2 += omega2 * dt
    
    final_dist = math.hypot(tx - fk(t1, t2)[0], ty - fk(t1, t2)[1])
    return final_dist / init_dist

# 测试：和 host 一样的 target 分布
import random
random.seed(42)
median_ratios = []
for ep in range(500):
    # 和 gen_target 一样：在 (2,0) 周围 + origin 圆内
    for _ in range(100):
        r = 0.3 + random.random() * 1.5
        theta = random.random() * 2 * math.pi
        tx = 2.0 + r * math.cos(theta)
        ty = r * math.sin(theta)
        if math.hypot(tx, ty) <= max_reach - 0.01:
            break
    
    ratio = run_episode(tx, ty)
    median_ratios.append(ratio)
    if ep < 20 or ratio < 0.3:
        print(f"Ep {ep:3d} target=({tx:.2f},{ty:.2f}) ratio={ratio:.4f} {'✅' if ratio <= 0.20 else ''}")

median_ratios.sort()
print(f"\n=== 运动学 Baseline ===")
print(f"Median ratio: {median_ratios[250]:.4f}")
print(f"Mean ratio:   {sum(median_ratios)/len(median_ratios):.4f}")
print(f"Min ratio:    {median_ratios[0]:.4f}")
print(f"Max ratio:    {median_ratios[-1]:.4f}")
print(f"Target: ≤ 0.20")
print(f"PASS: {'YES' if median_ratios[250] <= 0.20 else 'NO'}")
