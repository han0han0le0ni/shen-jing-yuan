"""CPU 后端 —— 零文件 IO，ctypes 直接 flat buffer"""
import ctypes

LAYER_INPUT_X = 0
LAYER_HIDDEN  = 1
LAYER_MOTOR1  = 2
LAYER_MOTOR2  = 3

class CPUBackend:
    """
    封装 benchmark2_core.dll
    - 拓扑：net2_get_topology  →  flat buffers → Python list
    - 实时：net2_get_snapshot  →  flat buffers → Python list
    - 零文件 IO，无 JSON 序列化
    """
    NET_SIZE = 4000000

    def __init__(self, dll_path):
        self.lib = ctypes.CDLL(dll_path)
        self._net_buf = ctypes.create_string_buffer(self.NET_SIZE)
        self.net_ptr = ctypes.cast(self._net_buf, ctypes.c_void_p)
        self._setup_sigs()
        self.config = None
        self._neuron_membrane = None
        self._neuron_spiking = None
        self._neuron_last_spike = None
        self._arm = None
        self._meta = None
        self._topo = None

    def _setup_sigs(self):
        L = self.lib
        # 真实签名：net, input, hidden, motor1, motor2, double avg_conn, uint seed
        # 旧版漏掉 avg_conn，把 seed 塞进了连接度槽位，导致 K 恒为 3 且种子不可控
        L.net2_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                ctypes.c_uint]
        L.net2_init.restype = ctypes.c_int
        L.net2_destroy.argtypes = [ctypes.c_void_p]
        # 真实规模必须由 C 侧提供，不能按 Python 端层数相加估算
        L.net2_neuron_count.argtypes = [ctypes.c_void_p]
        L.net2_neuron_count.restype = ctypes.c_int
        L.net2_synapse_count.argtypes = [ctypes.c_void_p]
        L.net2_synapse_count.restype = ctypes.c_int
        L.net2_tick.argtypes = [ctypes.c_void_p]
        L.net2_set_target.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        L.net2_get_topology.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),    # types[N]
            ctypes.POINTER(ctypes.c_double), # pref_x[N]
            ctypes.POINTER(ctypes.c_double), # pref_y[N]
            ctypes.POINTER(ctypes.c_double), # pref_omega[N]
            ctypes.POINTER(ctypes.c_int),    # pre[S]
            ctypes.POINTER(ctypes.c_int),    # post[S]
            ctypes.POINTER(ctypes.c_double), # w[S]
        ]
        L.net2_get_snapshot.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double), # mem[N]
            ctypes.POINTER(ctypes.c_double), # spk[N]
            ctypes.POINTER(ctypes.c_double), # last[N]
            ctypes.POINTER(ctypes.c_double), # arm[10]
            ctypes.POINTER(ctypes.c_int),    # meta[3]
        ]
        L.net2_dump_topology_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.net2_dump_snapshot_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        # 训练接口：旧版 DLL 可能没有这些导出，逐个探测而不是直接崩在加载阶段
        train_sigs = (
            ("net2_inject_reward",  [ctypes.c_void_p, ctypes.c_double, ctypes.c_int], None),
            ("net2_inject_reward2", [ctypes.c_void_p, ctypes.c_double,
                                     ctypes.c_double, ctypes.c_int], None),
            ("net2_reset_ema",      [], None),
            ("net2_set_param",      [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.c_double], ctypes.c_int),
            ("net2_get_param",      [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_double),
            ("net2_episode_reset",  [ctypes.c_void_p, ctypes.c_double,
                                     ctypes.c_double], None),
            ("net2_get_arm",        [ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_double)], None),
        )
        self.can_train = True
        for name, argtypes, restype in train_sigs:
            if not hasattr(L, name):
                self.can_train = False
                continue
            fn = getattr(L, name)
            fn.argtypes = argtypes
            if restype is not None:
                fn.restype = restype

    def init(self, input_n, hidden_n, motor1_n, motor2_n, seed=42, avg_conn=8.0):
        rc = self.lib.net2_init(self.net_ptr, input_n, hidden_n, motor1_n,
                                motor2_n, float(avg_conn), int(seed))
        if rc != 0: raise RuntimeError(f"net2_init rc={rc}")
        neuron_count = int(self.lib.net2_neuron_count(self.net_ptr))
        if neuron_count <= 0:
            raise RuntimeError("net2_neuron_count() returned an invalid value")
        self.config = {
            "input": input_n, "hidden": hidden_n,
            "motor1": motor1_n, "motor2": motor2_n,
            "neuron_count": neuron_count,  # 真实值：含 PROP_N 个本体感觉神经元
        }
        self._alloc_buffers()
        self._topo = None   # 重新初始化后必须让拓扑缓存失效
        self._topo = self.get_topology()

    def destroy(self): self.lib.net2_destroy(self.net_ptr)

    def _alloc_buffers(self):
        N = self.config["neuron_count"]
        self._neuron_membrane   = (ctypes.c_double * N)()
        self._neuron_spiking    = (ctypes.c_double * N)()
        self._neuron_last_spike = (ctypes.c_double * N)()
        self._arm  = (ctypes.c_double * 10)()
        self._meta = (ctypes.c_int * 3)()

    def tick(self): self.lib.net2_tick(self.net_ptr)

    def set_target(self, x, y):
        self.lib.net2_set_target(self.net_ptr, float(x), float(y))

    # ---------- 训练 / 调参 ----------
    def get_arm(self):
        """逐 tick 轻量读取 (end_x, end_y, target_x, target_y, max_reach)"""
        self.lib.net2_get_arm(self.net_ptr, self._arm)
        return (float(self._arm[4]), float(self._arm[5]),
                float(self._arm[6]), float(self._arm[7]), float(self._arm[9]))

    def inject_reward(self, r, delay=0):
        self.lib.net2_inject_reward(self.net_ptr, float(r), int(delay))

    def inject_reward2(self, r1, r2, delay=0):
        self.lib.net2_inject_reward2(self.net_ptr, float(r1), float(r2), int(delay))

    def set_param(self, name, value):
        return int(self.lib.net2_set_param(self.net_ptr, name.encode("utf-8"),
                                           float(value)))

    def get_param(self, name):
        return float(self.lib.net2_get_param(self.net_ptr, name.encode("utf-8")))

    def episode_reset(self, tx, ty):
        self.lib.net2_episode_reset(self.net_ptr, float(tx), float(ty))

    def reset_ema(self):
        self.lib.net2_reset_ema()

    def get_topology(self):
        if self._topo: return self._topo
        # 规模直接由 C 侧提供，无需临时 JSON 文件
        nc = int(self.lib.net2_neuron_count(self.net_ptr))
        sc = int(self.lib.net2_synapse_count(self.net_ptr))
        if nc != self.config["neuron_count"] or nc <= 0 or sc < 0:
            raise RuntimeError("DLL topology/count mismatch; rebuild b2_core.dll")
        types     = (ctypes.c_int * nc)()
        pref_x    = (ctypes.c_double * nc)()
        pref_y    = (ctypes.c_double * nc)()
        pref_om   = (ctypes.c_double * nc)()
        pre       = (ctypes.c_int * sc)()
        post      = (ctypes.c_int * sc)()
        w         = (ctypes.c_double * sc)()
        self.lib.net2_get_topology(
            self.net_ptr,
            types, pref_x, pref_y, pref_om,
            pre, post, w,
        )
        neurons = []
        layers = [0, 0, 0, 0]
        for i in range(nc):
            neurons.append({
                "id": i, "layer": types[i],
                "pref_x": pref_x[i], "pref_y": pref_y[i], "pref_omega": pref_om[i],
            })
            if 0 <= types[i] < 4:
                layers[types[i]] += 1
        syns = []
        for i in range(sc):
            syns.append({"pre": pre[i], "post": post[i], "w": w[i]})
        return {
            "neuron_count": nc, "synapse_count": sc,
            "input": layers[0], "hidden": layers[1],
            "motor1": layers[2], "motor2": layers[3],
            "neurons": neurons, "synapses": syns,
        }

    def get_snapshot(self):
        self.lib.net2_get_snapshot(
            self.net_ptr,
            self._neuron_membrane, self._neuron_spiking, self._neuron_last_spike,
            self._arm, self._meta,
        )
        return self._snapshot_from_buffers()

    def _snapshot_from_buffers(self):
        N = self.config["neuron_count"]
        neurons = [
            {"m": self._neuron_membrane[i],
             "s": int(self._neuron_spiking[i]),
             "lst": int(self._neuron_last_spike[i])}
            for i in range(N)
        ]
        return {
            "tick": self._meta[0],
            "neurons": neurons,
            "arm": {
                "t1": self._arm[0], "t2": self._arm[1],
                "o1": self._arm[2], "o2": self._arm[3],
                "ex": self._arm[4], "ey": self._arm[5],
                "tx": self._arm[6], "ty": self._arm[7],
                "r":  self._arm[8], "mr": self._arm[9],
            },
        }

    def get_snapshot_arrays(self):
        """快速路径：直接 numpy array，省掉 dict 构造开销"""
        import numpy as np
        self.lib.net2_get_snapshot(
            self.net_ptr,
            self._neuron_membrane, self._neuron_spiking, self._neuron_last_spike,
            self._arm, self._meta,
        )
        N = self.config["neuron_count"]
        mem = np.frombuffer(self._neuron_membrane, dtype=np.float64, count=N).astype(np.float32)
        spk = np.frombuffer(self._neuron_spiking, dtype=np.float64, count=N).astype(np.int32)
        last = np.frombuffer(self._neuron_last_spike, dtype=np.float64, count=N).astype(np.int32)
        arm = np.frombuffer(self._arm, dtype=np.float64, count=10)
        meta = np.frombuffer(self._meta, dtype=np.int32, count=3)
        return mem, spk, last, arm, meta
