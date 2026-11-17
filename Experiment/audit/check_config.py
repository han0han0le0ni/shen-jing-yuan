"""单配置初始化探针：在独立进程里跑一次 net2_init + topology dump。

用法: py audit/check_config.py <input> <hidden> <motor> <avg_conn>
退出码非 0（如 -1073740940 / 0xC0000374 堆损坏）表示该配置触发内存错误。
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

inp, hid, mot, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
dll = Path(os.environ.get("TEMP", ".")) / "snn_audit" / "b2_core_audit.dll"
out = Path(os.environ.get("TEMP", ".")) / "snn_audit" / "_cfg_topo.json"
lib = ctypes.CDLL(str(dll))
lib.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint]
lib.net2_init.restype = ctypes.c_int
lib.net2_destroy.argtypes = [ctypes.c_void_p]
lib.net2_dump_topology_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
buf = ctypes.create_string_buffer(4_000_000)
net = ctypes.cast(buf, ctypes.c_void_p)
print("CFG", inp, hid, mot, k, "init...", flush=True)
rc = lib.net2_init(net, inp, hid, mot, mot, k, 42)
print("  rc =", rc, flush=True)
lib.net2_dump_topology_json(net, str(out).encode())
j = json.loads(out.read_text(encoding="utf-8"))
print("  neurons =", j.get("neuron_count"), "synapses =", j.get("synapse_count"), flush=True)
lib.net2_destroy(net)
print("  destroy ok", flush=True)
