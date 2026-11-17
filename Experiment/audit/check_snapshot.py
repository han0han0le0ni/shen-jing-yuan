"""snapshot 缓冲区回归验证：按 C 侧真实规模分配，确认越界写已修复。

修复前：backend_cpu.py 用 N = input+hidden+m1+m2 = 256 分配，而 C 写 312 个 double，
进程退出时报 0xC0000005。修复后通过 net2_neuron_count() 取得真实 N。
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

dll = Path(os.environ.get("TEMP", ".")) / "snn_audit" / "b2_core_audit.dll"
lib = ctypes.CDLL(str(dll))
lib.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint]
lib.net2_init.restype = ctypes.c_int
lib.net2_neuron_count.argtypes = [ctypes.c_void_p]
lib.net2_neuron_count.restype = ctypes.c_int
lib.net2_destroy.argtypes = [ctypes.c_void_p]
lib.net2_get_snapshot.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_int)]
buf = ctypes.create_string_buffer(4_000_000)
net = ctypes.cast(buf, ctypes.c_void_p)
rc = lib.net2_init(net, 36, 200, 10, 10, 8.0, 42)
N = int(lib.net2_neuron_count(net))
N_old = 36 + 200 + 10 + 10
print(f"rc = {rc}, true N = {N}, 旧估算 N = {N_old}（少 {N - N_old}）", flush=True)
assert rc == 0 and N == N_old + 56, "真实规模不符合预期，检查 DLL 是否为最新构建"
mem = (ctypes.c_double * N)()
spk = (ctypes.c_double * N)()
last = (ctypes.c_double * N)()
arm = (ctypes.c_double * 10)()
meta = (ctypes.c_int * 3)()
lib.net2_get_snapshot(net, mem, spk, last, arm, meta)
print(f"snapshot ok: 3 个数组长度均为 {N}，与 C 侧写入量一致", flush=True)
lib.net2_destroy(net)
print("destroy ok — 无内存破坏", flush=True)
