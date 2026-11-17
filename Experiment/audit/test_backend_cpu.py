"""CPU 可视化后端端到端探针（由 test_suite.py 在独立进程中调用）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vis"))
from backend_cpu import CPUBackend  # noqa: E402

b = CPUBackend(str(ROOT / "b2_core.dll"))
b.init(36, 200, 10, 10, seed=42, avg_conn=8.0)
nc = b.config["neuron_count"]
topo = b.get_topology()
assert nc == 312, f"neuron_count={nc}"
assert topo["neuron_count"] == 312 and topo["synapse_count"] > 0
assert len(topo["neurons"]) == 312 and len(topo["synapses"]) == topo["synapse_count"]
assert [topo["input"], topo["hidden"], topo["motor1"], topo["motor2"]] == [92, 200, 10, 10], \
    f"layers={[topo['input'], topo['hidden'], topo['motor1'], topo['motor2']]}"
for _ in range(50):
    b.tick()
snap = b.get_snapshot()
assert len(snap["neurons"]) == 312, f"snapshot neurons={len(snap['neurons'])}"
assert all(math.isfinite(float(n["m"])) for n in snap["neurons"])
print(f"BACKEND_OK nc={nc} sc={topo['synapse_count']} layers=92/200/10/10 tick={snap['tick']}")
b.destroy()
