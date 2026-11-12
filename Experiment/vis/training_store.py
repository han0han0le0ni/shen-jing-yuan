"""训练快照：把学到的突触权重与训练进度落盘 / 回读。

背景：SimEngine.reset() 会重建后端并按 seed 重新随机初始化权重，进程退出也
不保存，所以「刷新一次和没训练过一样」。本模块补齐这一环。

保存内容（np.savez_compressed，无 pickle）：
    W                  权重矩阵 float32 [pre, post]（Wt 由 W 重建）
    eligibility_trace  资格迹 float32（学习中间状态）
    meta               训练进度 / G / 拓扑签名 / 权重指纹（UTF-8 JSON bytes）
    guidance           引导监控最近 360 tick 的记录（UTF-8 JSON bytes）

限制：只有 numpy / CUDA 向量化后端有可写的权重数组；C DLL 只有
net2_get_topology 读接口、没有写权重的 C API，因此明确拒绝并提示切换后端。
"""
import hashlib
import json
import os
import time
from collections import deque

import numpy as np

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VIS_DIR)
SNAP_DIR = os.path.join(ROOT_DIR, "data", "training")
DEFAULT_NAME = "autosave.npz"
FORMAT_VERSION = 1


class SnapshotError(Exception):
    """可读的快照错误（不把 traceback 抛给 GUI）。"""


def snap_dir():
    os.makedirs(SNAP_DIR, exist_ok=True)
    return SNAP_DIR


def default_path():
    return os.path.join(snap_dir(), DEFAULT_NAME)


def list_snapshots():
    d = snap_dir()
    out = []
    for name in os.listdir(d):
        if not name.endswith(".npz"):
            continue
        p = os.path.join(d, name)
        try:
            out.append((os.path.getmtime(p), p))
        except OSError:
            continue
    return [p for _, p in sorted(out, reverse=True)]


def _to_numpy(a):
    if a is None:
        return None
    if hasattr(a, "get"):          # CuPy
        return np.asarray(a.get())
    return np.asarray(a)


def _topo_sig(be):
    """用边表 + 规模做指纹，确保权重不会错灌到另一张网络里。"""
    pre = _to_numpy(be.syn_pre).astype(np.int32, copy=False)
    post = _to_numpy(be.syn_post).astype(np.int32, copy=False)
    h = hashlib.sha1(pre.tobytes() + post.tobytes()).hexdigest()[:16]
    return "%s-%d-%d" % (h, len(pre), int(be.config["neuron_count"]))


def _weight_fingerprint(W):
    b = np.ascontiguousarray(W, dtype=np.float32).tobytes()
    return hashlib.sha1(b).hexdigest()[:16]


def describe(path=None):
    """只读快照元信息，返回 dict；文件不存在/损坏返回 None。"""
    p = path or default_path()
    if not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(z["meta"].tobytes().decode("utf-8"))
        meta["path"] = p
        meta["size"] = os.path.getsize(p)
        meta["mtime"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(os.path.getmtime(p)))
        return meta
    except Exception:
        return None


def current_info(engine):
    """当前网络的简报，供窗口显示（不写文件）。"""
    be = getattr(engine, "be", None)
    if be is None or not hasattr(be, "W"):
        return {"supported": False, "reason": "当前后端 C DLL 不提供权重写入接口"}
    W = _to_numpy(be.W)
    return {
        "supported": True,
        "device": "CUDA" if getattr(be, "_on_gpu", False) else "向量化 · 12 核",
        "neuron_count": int(be.config["neuron_count"]),
        "synapse_count": int(be.topo["synapse_count"]),
        "topo_sig": _topo_sig(be),
        "w_hash": _weight_fingerprint(W),
        "w_sum": float(W.sum()),
        "episode": int(engine.episode),
        "arrivals": int(engine.arrivals),
        "teacher_g": float(engine._teacher_g),
    }


def save(engine, path=None, extra=None):
    """保存当前权重 + 训练进度，返回 meta dict。"""
    be = getattr(engine, "be", None)
    if be is None or not hasattr(be, "W"):
        raise SnapshotError(
            "当前后端（C DLL）不提供权重写入接口，无法保存训练成果。"
            "请切到「向量化 · 12 核」后再保存（选老师时会自动切过去）。")
    W = np.ascontiguousarray(_to_numpy(be.W), dtype=np.float32)
    trace = _to_numpy(getattr(be, "eligibility_trace", None))
    trace = (np.ascontiguousarray(trace, dtype=np.float32)
             if trace is not None else np.zeros(0, dtype=np.float32))
    guidance = json.dumps(list(getattr(engine, "guidance_hist", ()) or ()),
                          ensure_ascii=False).encode("utf-8")

    meta = {
        "version": FORMAT_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cuda" if getattr(be, "_on_gpu", False) else "numpy",
        "neuron_count": int(be.config["neuron_count"]),
        "synapse_count": int(be.topo["synapse_count"]),
        "topo_sig": _topo_sig(be),
        "w_hash": _weight_fingerprint(W),
        "w_sum": float(W.sum()),
        "episode": int(engine.episode),
        "arrivals": int(engine.arrivals),
        "trial_tick": int(engine.trial_tick),
        "total_reward": float(engine.total_reward),
        "last_reward": float(engine.last_reward),
        "last_episode_ticks": int(engine.last_episode_ticks),
        "tick_count": int(getattr(be, "tick_count", 0)),
        "teacher_g": float(engine._teacher_g),
        "teacher_anneal": bool(engine._teacher_anneal),
        "teacher_name": getattr(getattr(engine, "_teacher", None), "name", ""),
    }
    if extra:
        meta.update({k: v for k, v in extra.items()
                     if isinstance(k, str) and isinstance(v, (str, int, float, bool))})

    p = path or default_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    meta_bytes = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                               dtype=np.uint8)
    guide_bytes = np.frombuffer(guidance, dtype=np.uint8)
    with open(tmp, "wb") as f:   # 传文件对象，避免 np.savez 自动追加 .npz 后缀
        np.savez_compressed(f, W=W, trace=trace,
                            meta=meta_bytes, guidance=guide_bytes)
    os.replace(tmp, p)          # 原子替换，避免写一半掉电损坏存档
    return meta


def load(engine, path=None):
    """把快照写回当前后端；拓扑不匹配时直接拒绝，返回 meta dict。"""
    be = getattr(engine, "be", None)
    if be is None or not hasattr(be, "W"):
        raise SnapshotError(
            "当前后端（C DLL）不提供权重写入接口。请切到「向量化 · 12 核」再载入。")
    p = path or default_path()
    if not os.path.exists(p):
        raise SnapshotError("还没有训练存档（%s）" % p)
    try:
        z = np.load(p, allow_pickle=False)
    except Exception as exc:
        raise SnapshotError("存档损坏或格式不支持：%s" % exc) from exc
    try:
        meta = json.loads(z["meta"].tobytes().decode("utf-8"))
        W = np.asarray(z["W"], dtype=np.float32)
        trace = np.asarray(z["trace"], dtype=np.float32) if "trace" in z else None
        guidance = (json.loads(z["guidance"].tobytes().decode("utf-8"))
                    if "guidance" in z else [])
    except Exception as exc:
        raise SnapshotError("存档字段不完整：%s" % exc) from exc
    finally:
        try:
            z.close()
        except Exception:
            pass

    if int(meta.get("version", 0)) != FORMAT_VERSION:
        raise SnapshotError("存档版本 %s 与当前 %d 不兼容"
                            % (meta.get("version"), FORMAT_VERSION))
    sig = _topo_sig(be)
    if meta.get("topo_sig") != sig:
        raise SnapshotError(
            "网络拓扑不匹配（存档 %s / 当前 %s）。换网络规模、种子或设备后不能沿用旧权重，"
            "请重建到相同配置再载入，或删除该存档。" % (meta.get("topo_sig"), sig))
    if W.shape != tuple(_to_numpy(be.W).shape):
        raise SnapshotError("权重形状不匹配：%s vs %s"
                            % (W.shape, _to_numpy(be.W).shape))

    xp = getattr(be, "_xp", np)
    be.W[...] = xp.asarray(W, dtype=xp.float32)
    be.Wt[...] = xp.ascontiguousarray(be.W.T)
    if trace is not None and hasattr(be, "eligibility_trace") and \
            len(trace) == len(be.eligibility_trace):
        be.eligibility_trace[...] = xp.asarray(trace, dtype=xp.float32)
    if hasattr(be, "tick_count") and "tick_count" in meta:
        be.tick_count = int(meta["tick_count"])

    for key in ("episode", "arrivals", "trial_tick", "last_episode_ticks"):
        if key in meta:
            setattr(engine, key, int(meta[key]))
    for key in ("total_reward", "last_reward"):
        if key in meta:
            setattr(engine, key, float(meta[key]))
    engine._teacher_g = float(meta.get("teacher_g", engine._teacher_g))
    engine._teacher_anneal = bool(meta.get("teacher_anneal", engine._teacher_anneal))
    engine.guidance_hist = deque(guidance[-360:], maxlen=360)
    engine.guidance = engine.guidance_hist[-1] if engine.guidance_hist else {}
    engine._trail_buf = []
    return meta


def delete(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False
