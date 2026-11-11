"""ctypes 接口一致性检测：对比仓库 DLL 与按当前源码重建的 DLL。

只读，不写实验数据；topology JSON 输出到系统临时目录。
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(os.environ.get("TEMP", ".")) / "snn_audit"
TARGETS = {
    "repo_b2_core.dll": ROOT / "b2_core.dll",
    "rebuilt_b2_core.dll": TMP / "b2_core_audit.dll",
}
EXPORTS = ["net2_init", "net2_inject_reward2", "net2_get_topology",
           "net2_get_snapshot", "net2_dump_topology_json", "net2_tick"]

buf = ctypes.create_string_buffer(4_000_000)
NET = ctypes.cast(buf, ctypes.c_void_p)
out = TMP / "_audit_topo.json"

for label, path in TARGETS.items():
    if not path.is_file():
        print(label, "MISSING", path)
        continue
    lib = ctypes.CDLL(str(path))
    lib.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_uint]
    lib.net2_init.restype = ctypes.c_int
    lib.net2_destroy.argtypes = [ctypes.c_void_p]
    lib.net2_dump_topology_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    print(label, "exports:", {n: hasattr(lib, n) for n in EXPORTS})
    rc = lib.net2_init(NET, 36, 200, 10, 10, 3.0, 42)
    print(label, "net2_init(36,200,10,10,avg_conn=3) rc =", rc)
    try:
        lib.net2_dump_topology_json(NET, str(out).encode())
        j = json.loads(out.read_text(encoding="utf-8"))
        print(label, "topology:", {k: j.get(k) for k in
                                   ("neuron_count", "synapse_count", "input", "hidden", "motor1", "motor2")})
    except Exception as exc:
        print(label, "topology read failed:", repr(exc))
    lib.net2_destroy(NET)
