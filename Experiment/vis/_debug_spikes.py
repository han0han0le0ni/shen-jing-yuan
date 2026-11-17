"""诊断 hidden/motor spike 率"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from backend_vec import VecBackend

b = VecBackend(device="gpu", seed=42)
b.init(100, 2000, 50, 50, seed=42)

m1_base = 100 + 2000
m2_base = m1_base + 50

# 先跑 500 tick warmup
for _ in range(500):
    b.tick()

# 然后跑 500 tick 统计 spike 率
hidden_spikes = []
m1_spikes = []
m2_spikes = []
for _ in range(500):
    b.tick()
    mem, spk, last, arm, meta = b.get_snapshot_arrays()
    h_mask = spk[100:2100]
    hidden_spikes.append(h_mask.sum())
    m1_mask = spk[m1_base:m1_base+50]
    m1_spikes.append(m1_mask.sum())
    m2_mask = spk[m2_base:m2_base+50]
    m2_spikes.append(m2_mask.sum())

# 再看 omega_cmd 数值（通过 arm 状态反推不了，换个方式看 motor 层最近 spike）
print(f"hidden spike/tick: mean={np.mean(hidden_spikes):.1f}, min={np.min(hidden_spikes)}, max={np.max(hidden_spikes)}")
print(f"m1 spike/tick: mean={np.mean(m1_spikes):.1f}, min={np.min(m1_spikes)}, max={np.max(m1_spikes)}")
print(f"m2 spike/tick: mean={np.mean(m2_spikes):.1f}, min={np.min(m2_spikes)}, max={np.max(m2_spikes)}")
print()
print(f"hidden 层 spike 率: {np.mean(hidden_spikes)/2000*100:.2f}%")
print(f"m1 层 spike 率: {np.mean(m1_spikes)/50*100:.2f}%")
print(f"m2 层 spike 率: {np.mean(m2_spikes)/50*100:.2f}%")
