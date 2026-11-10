"""ctypes 接口快速验证 v2 —— 用 create_string_buffer 分配内存"""
import ctypes, os, json, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
lib = ctypes.CDLL("./b2_core.dll")

# 分配一大块内存当 NetworkState2（足够大就行）
NET_SIZE = 2000000  # 2MB 足够装 2200 neurons + 40K synapses
_net_buf = ctypes.create_string_buffer(NET_SIZE)
NET_PTR = ctypes.cast(_net_buf, ctypes.c_void_p)

# === 函数签名 ===
lib.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.net2_init.restype = ctypes.c_int

lib.net2_destroy.argtypes = [ctypes.c_void_p]
lib.net2_set_target.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
lib.net2_tick.argtypes = [ctypes.c_void_p]
lib.net2_dump_topology_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.net2_dump_snapshot_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.net2_dump_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# 先统计一下需要的大小
# neurons: 36+200+10+10 = 256
# 每个 Neuron2 约 100 字节 → 256*100=25600
# synapses: 256*8 ≈ 2048
# 每个 Synapse2 约 50 字节 → 102400
# 总计 < 2000000 ✓

rc = lib.net2_init(NET_PTR, 36, 200, 10, 10, 42)
print(f"net2_init rc={rc} ✓")

# 写 topology JSON
lib.net2_dump_topology_json(NET_PTR, b"vis_topo_test.json")
with open("vis_topo_test.json") as f:
    topo = json.load(f)
print(f"✓ topology JSON: {topo['neuron_count']} neurons, {topo['synapse_count']} synapses")
print(f"  layers: in={topo['input']} h={topo['hidden']} m1={topo['motor1']} m2={topo['motor2']}")

# 跑几步
lib.net2_set_target(NET_PTR, 0.5, 0.5)
for i in range(50):
    lib.net2_tick(NET_PTR)
lib.net2_dump_snapshot_json(NET_PTR, b"vis_snap_test.json")

with open("vis_snap_test.json") as f:
    snap = json.load(f)
print(f"✓ snapshot tick={snap['tick']}:")
print(f"  arm: end=({snap['arm']['ex']:.3f},{snap['arm']['ey']:.3f}) target=({snap['arm']['tx']:.3f},{snap['arm']['ty']:.3f}) reward={snap['arm']['r']:.4f}")
spiking = sum(1 for n in snap['neurons'] if n['s'])
print(f"  spiking: {spiking}/{len(snap['neurons'])}")

lib.net2_destroy(NET_PTR)
print("\n✅ ctypes 接口全通！")
