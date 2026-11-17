"""修复后回归验证：内存安全、ABI、构建一致性、元数据、Python 语法、冒烟运行。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "audit"
TMP = Path(os.environ.get("TEMP", ".")) / "snn_audit"
PY = sys.executable
checks = []


def chk(name, cond, detail=""):
    checks.append((name, bool(cond), str(detail).replace("\ufffd", "?")[:220]))


def run(args, cwd=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# 1. 六组初始化配置（修复前 36/200/10/3 会以 0xC0000374 退出）
configs = [(36, 200, 10, 3.0), (36, 200, 10, 8.0), (36, 200, 10, 20.0),
           (100, 2000, 50, 8.0), (100, 2000, 50, 3.0), (144, 2000, 50, 8.0)]
syn_counts = {}
cfg_ok = True
for inp, hid, mot, k in configs:
    rc, out = run([PY, str(HERE / "check_config.py"), str(inp), str(hid), str(mot), str(k)])
    cfg_ok &= rc == 0
    for line in out.splitlines():
        if "synapses =" in line:
            syn_counts[f"{inp}/{hid}/{mot}/{k:g}"] = line.strip()
chk("6 组 net2_init 全部无内存错误", cfg_ok,
    "; ".join(f"{k}:{v.split('synapses =')[-1].strip()}" for k, v in syn_counts.items()))
chk("默认配置突触数保持 44401（不扰动实验语义）",
    "44401" in syn_counts.get("100/2000/50/8", ""), syn_counts.get("100/2000/50/8", "missing"))

# 2. snapshot 缓冲区按真实规模分配
rc, out = run([PY, str(HERE / "check_snapshot.py")])
chk("snapshot 按 net2_neuron_count 分配后正常", rc == 0, out.strip().replace("\n", " | "))

# 3. ctypes 导出与 ABI
rc, out = run([PY, str(HERE / "check_ctypes.py")])
chk("ctypes 接口对比无崩溃", rc == 0, out.strip().replace("\n", " | ")[:200])
export_hits = out.count("'net2_inject_reward2': True")
chk("两个 DLL 均导出 net2_inject_reward2", export_hits == 2, f"count={export_hits}")

# 4. vis 测试脚本
rc, out = run([PY, str(ROOT / "vis" / "test_ctypes.py")])
chk("vis/test_ctypes.py 7 参签名通过", rc == 0, out.strip().replace("\n", " | ")[:200])
chk("vis 测试读取到 312 神经元", "312 neurons" in out, out.strip().replace("\n", " | ")[:140])

# 5. B1 短跑（在 TEMP 下运行，相对诊断路径不会污染仓库）
b1_csv = TMP / "b1_smoke.csv"
rc, out = run([str(ROOT / "benchmark1.exe"), "--trials", "500", "--seed", "7",
               "--out", str(b1_csv)], cwd=TMP)
chk("B1 500 trials 运行通过", rc == 0 and "RESULT: PASS" in out,
    out.strip().replace("\n", " | ")[-180:])
chk("B1 CSV 与 sidecar 生成", b1_csv.is_file() and len(b1_csv.read_text().splitlines()) == 501
    and (TMP / "b1_smoke.csv.meta.json").is_file(),
    f"csv_rows={len(b1_csv.read_text().splitlines()) if b1_csv.is_file() else 0}")

# 6. 构建信息、产物时间戳、sidecar
info_path = ROOT / "build_info.json"
info = json.loads(info_path.read_text(encoding="utf-8-sig")) if info_path.is_file() else {}
hash_ok = True
for rel, digest in (info.get("sources") or {}).items():
    p = ROOT / rel.replace("\\", "/")
    hash_ok &= p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
for name, digest in (info.get("artifacts") or {}).items():
    p = ROOT / name
    hash_ok &= p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
chk("build_info.json 源码/产物哈希一致", bool(info) and hash_ok,
    f"sources={len(info.get('sources') or {})} artifacts={len(info.get('artifacts') or {})}")

src_times = [(ROOT / r.replace("\\", "/")).stat().st_mtime
             for r in (info.get("sources") or {}) if (ROOT / r.replace("\\", "/")).is_file()]
art_times = {n: (ROOT / n).stat().st_mtime for n in ("benchmark1.exe", "benchmark2.exe", "b2_core.dll", "b2_core.o")}
chk("所有二进制晚于源码", all(t >= max(src_times) for t in art_times.values()),
    {k: round(v - max(src_times), 1) for k, v in art_times.items()})

meta = TMP / "smoke2.csv.meta.json"
if meta.is_file():
    m = json.loads(meta.read_text(encoding="utf-8"))
    chk("B2 sidecar 字段完整",
        m.get("benchmark") == 2 and m.get("episodes") == 3 and m.get("target_mode") == 1
        and m.get("neurons", 0) > 0, json.dumps(m, ensure_ascii=False)[:180])
else:
    chk("B2 sidecar 字段完整", False, "missing")

# 7. DLL 导出符号
rc, out = run(["nm", "-g", "--defined-only", str(ROOT / "b2_core.dll")])
exports = {"net2_neuron_count", "net2_synapse_count", "net2_inject_reward2",
           "net2_get_topology", "net2_get_snapshot"}
chk("DLL 导出新增 getter 与双通道奖励接口",
    rc == 0 and all(name in out for name in exports),
    f"missing={sorted(exports - set(out.split()))}")

# 8. 编译告警
for label, tool, src in (("B2 core", "gcc", "brain_b2/benchmark2_core.c"),
                         ("B2 host", "g++", "host_b2/benchmark2_host.cpp"),
                         ("B1 core", "gcc", "brain/benchmark1_core.c"),
                         ("B1 host", "g++", "host/benchmark1_host.cpp")):
    rc, out = run([tool, "-O2", "-Wall", "-Wextra", "-fsyntax-only", str(ROOT / src)])
    chk(f"{label} 编译无告警", rc == 0 and "warning:" not in out, out.strip()[:120])

# 9. Python 语法
py_files = [ROOT / "vis" / "backend_cpu.py", ROOT / "vis" / "backend_vec.py",
            ROOT / "vis" / "test_ctypes.py"] + sorted(HERE.glob("*.py"))
syntax_errors = []
for p in py_files:
    try:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
    except SyntaxError as exc:
        syntax_errors.append(f"{p.name}: {exc}")
chk(f"{len(py_files)} 个 Python 文件语法通过", not syntax_errors, "; ".join(syntax_errors))

print("=" * 78)
for name, ok, detail in checks:
    print(("PASS  " if ok else "FAIL  ") + name + ("  — " + detail if detail else ""))
print("=" * 78)
print("TOTAL", sum(ok for _, ok, _ in checks), "/", len(checks))
if not all(ok for _, ok, _ in checks):
    raise SystemExit(1)
