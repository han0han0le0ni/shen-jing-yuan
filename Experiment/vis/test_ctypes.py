"""ctypes 接口快速验证 v3 —— 7 参签名 + C 侧真实规模查询。

与 benchmark2_core.h 的 net2_init 契约一致；测试产物写入系统 TEMP，不污染仓库。
"""
import ctypes
import json
import os
import sys

# Windows 控制台默认 GBK，脚本里的 ✓/✗ 会触发 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.chdir(ROOT)
lib = ctypes.CDLL("./b2_core.dll")

# opaque NetworkState2 缓冲区（结构体含 8192 个奖励事件，4MB 足够）
NET_SIZE = 4000000
_net_buf = ctypes.create_string_buffer(NET_SIZE)
NET_PTR = ctypes.cast(_net_buf, ctypes.c_void_p)

OUT = os.path.join(os.environ.get("TEMP", "."), "snn_audit")
os.makedirs(OUT, exist_ok=True)

# === 函数签名 ===
lib.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint]
lib.net2_init.restype = ctypes.c_int
lib.net2_neuron_count.argtypes = [ctypes.c_void_p]
lib.net2_neuron_count.restype = ctypes.c_int
lib.net2_synapse_count.argtypes = [ctypes.c_void_p]
lib.net2_synapse_count.restype = ctypes.c_int
lib.net2_destroy.argtypes = [ctypes.c_void_p]
lib.net2_set_target.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
lib.net2_tick.argtypes = [ctypes.c_void_p]
lib.net2_dump_topology_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.net2_dump_snapshot_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.net2_dump_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# 7 参：net, input, hidden, motor1, motor2, avg_conn, seed
rc = lib.net2_init(NET_PTR, 36, 200, 10, 10, 8.0, 42)
print(f"net2_init rc={rc}")
if rc != 0:
    raise SystemExit(1)

nc = lib.net2_neuron_count(NET_PTR)
sc = lib.net2_synapse_count(NET_PTR)
print(f"真实规模: {nc} neurons (含 PROP_N=56), {sc} synapses")
assert nc == 36 + 56 + 200 + 10 + 10, "neuron_count 与预期不符，检查 DLL 是否为最新构建"

topo_path = os.path.join(OUT, "vis_topo_test.json")
snap_path = os.path.join(OUT, "vis_snap_test.json")

lib.net2_dump_topology_json(NET_PTR, topo_path.encode())
with open(topo_path) as f:
    topo = json.load(f)
assert topo["neuron_count"] == nc, "topology JSON 与 count getter 不一致"
print(f"✓ topology JSON: {topo['neuron_count']} neurons, {topo['synapse_count']} synapses")
print(f"  layers: in={topo['input']} h={topo['hidden']} m1={topo['motor1']} m2={topo['motor2']}")

lib.net2_set_target(NET_PTR, 0.5, 0.5)
for _ in range(50):
    lib.net2_tick(NET_PTR)
lib.net2_dump_snapshot_json(NET_PTR, snap_path.encode())
with open(snap_path) as f:
    snap = json.load(f)
print(f"✓ snapshot tick={snap['tick']}:")
print(f"  arm: end=({snap['arm']['ex']:.3f},{snap['arm']['ey']:.3f}) "
      f"target=({snap['arm']['tx']:.3f},{snap['arm']['ty']:.3f}) reward={snap['arm']['r']:.4f}")
spiking = sum(1 for n in snap["neurons"] if n["s"])
print(f"  spiking: {spiking}/{len(snap['neurons'])}")

lib.net2_destroy(NET_PTR)
print("\nctypes 接口全通")
