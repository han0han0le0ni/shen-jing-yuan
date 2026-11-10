"""backend_vec.py —— 统一 numpy 向量化后端

设计：
  一套代码，xp = numpy → CPU 模式 / xp = cupy → GPU 模式
  tick 循环完全向量化（OpenBLAS 多线程 / cuBLAS）

与 C 端 benchmark2_core.c 逐行等价：
  Step 0: 2D 高斯编码（INPUT_X 膜电位注入）
  Step 1: 膜电位衰减 + W^T @ spiking → 突触电流
  Step 2: spike 阈值判定 + 运动层 k-WTA
  Step 3: R-STDP 资格迹（向量化）
  Step 4: 群体向量解码 + 两关节物理模拟
"""
import math, json, os, sys, warnings

import numpy as np


def _silent_devnull():
    """上下文：把 stderr 丢进 devnull（避免 CuPy 编译失败刷屏）"""
    return open(os.devnull, "w")


# ---------- CUDA 环境自动配置（pip 装的 CUDA toolkit） ----------
def _setup_cuda_env():
    """让 CuPy 找到 pip 安装的 CUDA 头文件与 DLL（Windows）"""
    if os.environ.get("CUDA_PATH"):
        return
    try:
        import shutil, sysconfig
        nv_dir = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
        if not os.path.isdir(nv_dir):
            return
        shim = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "cupy_cuda_shim"
        )
        inc = os.path.join(shim, "include")
        os.makedirs(inc, exist_ok=True)
        # 合并各子包的头文件到单一 include 目录（薄弱但一次性的拷贝）
        if not os.path.exists(os.path.join(inc, "cuda_runtime.h")):
            for sub in os.listdir(nv_dir):
                src = os.path.join(nv_dir, sub, "include")
                if os.path.isdir(src):
                    for f in os.listdir(src):
                        s = os.path.join(src, f)
                        if os.path.isfile(s):
                            try:
                                shutil.copy2(s, os.path.join(inc, f))
                            except Exception:
                                pass
        os.environ["CUDA_PATH"] = shim
        for sub in os.listdir(nv_dir):
            b = os.path.join(nv_dir, sub, "bin")
            if os.path.isdir(b):
                os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]
    except Exception:
        pass  # 失败就按原样走，后端会自动回退 numpy


_setup_cuda_env()


# ---------- CuPy 可选（全程静默探测） ----------
cp = None
_CUPY_OK = False
_CUPY_USABLE = False

try:
    _old_err = sys.stderr
    _null = _silent_devnull()
    sys.stderr = _null
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
        _CUPY_OK = True
    finally:
        sys.stderr = _old_err
        _null.close()
except Exception:
    cp = None
    _CUPY_OK = False


def _probe_cupy():
    """真实探测：跑一个小 kernel，失败就判定不可用（结果缓存）"""
    global _CUPY_USABLE
    if _CUPY_USABLE:
        return True
    if not _CUPY_OK or cp is None:
        return False

    _old_err = sys.stderr
    _null = _silent_devnull()
    sys.stderr = _null
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = cp.full(2200, 1.0, dtype=cp.float32)
            val = float((probe * 2.0).sum())
            del probe
        if abs(val - 4400.0) < 1e-3:
            _CUPY_USABLE = True
            return True
        return False
    except Exception:
        return False
    finally:
        sys.stderr = _old_err
        _null.close()

# ---------- 与 C 端一致的常量 ----------
MEMBRANE_REST = -65.0
SPIKE_THRESHOLD = -55.0
MEMBRANE_DECAY = 0.95
REFRACTORY_TICKS = 3


# ============================================================
# 融合 CUDA 内核（GPU 路径：每 tick 6 次启动 vs 纯 CuPy 的 ~70 次）
# ============================================================
_CUDA_SRC = r"""
extern "C" __global__
void neuron_tick(const float* __restrict__ current,
                 float* membrane, int* refr, int* spiking,
                 const float* __restrict__ pref_x, const float* __restrict__ pref_y,
                 long long N, long long N_in, double tx, double ty, double sigma,
                 double rest, double thr, double decay, long long refr_ticks)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float m;
    if (i < N_in) {
        float dx = pref_x[i] - (float)tx, dy = pref_y[i] - (float)ty;
        float g = expf(-(dx * dx + dy * dy) / (2.0f * (float)sigma * (float)sigma));
        m = (float)rest + g * (float)(thr - rest + 5.0);
        spiking[i] = (m >= (float)thr) ? 1 : 0;
        membrane[i] = m;
        return;
    }
    if (refr[i] > 0) {
        refr[i] -= 1;
        membrane[i] = (float)rest;
        spiking[i] = 0;
        return;
    }
    m = membrane[i] * (float)decay + (float)rest * (1.0f - (float)decay) + current[i];
    spiking[i] = (m >= (float)thr) ? 1 : 0;
    membrane[i] = m;
}

extern "C" __global__
void motor_wta(int* spiking, const float* __restrict__ membrane,
               long long base, long long count, long long K)
{
    if (blockIdx.x || threadIdx.x) return;
    long long cnt = 0;
    for (long long i = 0; i < count; ++i) cnt += spiking[base + i];
    if (cnt <= K) return;
    for (long long k = 0; k < K; ++k) {
        long long best = -1; float bestv = -1e30f;
        for (long long i = 0; i < count; ++i) {
            if (spiking[base + i] == 1) {
                float v = membrane[base + i];
                if (v > bestv) { bestv = v; best = i; }
            }
        }
        if (best < 0) break;
        spiking[base + best] = 2;
    }
    for (long long i = 0; i < count; ++i) {
        int v = spiking[base + i];
        spiking[base + i] = (v == 2) ? 1 : 0;
    }
}

extern "C" __global__
void finalize_tick(int* spiking, float* membrane, int* refr, int* last_spike,
                   long long N, long long N_in, long long tick,
                   double rest, long long refr_ticks)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (spiking[i]) {
        last_spike[i] = (int)tick;
        membrane[i] = (float)rest;
        if (i >= N_in) refr[i] = (int)refr_ticks;
    }
}

extern "C" __global__
void trace_update(float* e, const int* __restrict__ pre, const int* __restrict__ post,
                  const int* __restrict__ spiking, const int* __restrict__ last_spike,
                  long long S, long long tick, double lambda, double tau)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;
    int p = pre[i], q = post[i];
    float inc = 0.f, dec = 0.f;
    if (spiking[p] && last_spike[q] >= 0)
        inc = expf(-(float)(tick - last_spike[q]) / (float)tau);
    if (spiking[q] && last_spike[p] >= 0)
        dec = expf(-(float)(tick - last_spike[p]) / (float)tau);
    float v = e[i] * (float)lambda + inc - dec;
    if (v > 5.f) v = 5.f;
    if (v < -5.f) v = -5.f;
    e[i] = v;
}

extern "C" __global__
void decode_arm(const int* __restrict__ last_spike, const float* __restrict__ pref_omega,
                double* arm, long long m1s, long long m1n,
                long long m2s, long long m2n, long long tick,
                double max_omega, double dt, double L1, double L2)
{
    if (blockIdx.x || threadIdx.x) return;
    double o1 = 0.0, o2 = 0.0;
    // 方向池分离平均：正 pref_omega 和负 pref_omega 分池，谁活跃多就用谁的池内平均
    {
        int pos_c = 0, neg_c = 0; float pos_s = 0.f, neg_s = 0.f;
        for (long long i = m1s; i < m1s + m1n; ++i) {
            int ls = last_spike[i];
            if (ls >= 0 && (tick - ls) >= 0 && (tick - ls) <= 5) {
                float po = pref_omega[i];
                if (po > 0.f) { pos_c++; pos_s += po; }
                else if (po < 0.f) { neg_c++; neg_s += po; }
            }
        }
        if (pos_c > neg_c && pos_c > 0) o1 = (double)(pos_s / pos_c);
        else if (neg_c > pos_c && neg_c > 0) o1 = (double)(neg_s / neg_c);
    }
    {
        int pos_c = 0, neg_c = 0; float pos_s = 0.f, neg_s = 0.f;
        for (long long i = m2s; i < m2s + m2n; ++i) {
            int ls = last_spike[i];
            if (ls >= 0 && (tick - ls) >= 0 && (tick - ls) <= 5) {
                float po = pref_omega[i];
                if (po > 0.f) { pos_c++; pos_s += po; }
                else if (po < 0.f) { neg_c++; neg_s += po; }
            }
        }
        if (pos_c > neg_c && pos_c > 0) o2 = (double)(pos_s / pos_c);
        else if (neg_c > pos_c && neg_c > 0) o2 = (double)(neg_s / neg_c);
    }
    if (o1 > max_omega) o1 = max_omega;
    if (o1 < -max_omega) o1 = -max_omega;
    if (o2 > max_omega) o2 = max_omega;
    if (o2 < -max_omega) o2 = -max_omega;

    double pi = 3.14159265358979323846;
    double t1 = arm[0] + o1 * dt;
    double t2 = arm[1] + o2 * dt;
    if (t1 > pi) t1 = pi; if (t1 < -pi) t1 = -pi;
    if (t2 > pi * 0.9) t2 = pi * 0.9; if (t2 < -pi * 0.9) t2 = -pi * 0.9;
    arm[0] = t1; arm[1] = t2; arm[2] = o1; arm[3] = o2;
    arm[4] = L1 * cos(t1) + L2 * cos(t1 + t2);
    arm[5] = L1 * sin(t1) + L2 * sin(t1 + t2);
}
"""
PSP_FACTOR = 25.0
ENCODE_SIGMA_POS = 0.85
INPUT_GAIN = 15.0
TAU_STDP = 50.0
LAMBDA_TRACE = math.exp(-1.0 / 200.0)  # ~0.951
ARM_DT = 0.02
MAX_OMEGA = 0.5
L1 = L2 = 1.0

LAYER_INPUT_X = 0
LAYER_HIDDEN  = 1
LAYER_MOTOR1  = 2
LAYER_MOTOR2  = 3


class VecBackend:
    def __init__(self, device="cpu", seed=42):
        self.device = device
        self.seed = seed
        self.tick_count = 0
        self.config = None
        self.topo = None
        self._xp = np
        self._on_gpu = False

    def _select_backend(self):
        """默认 GPU 向量化；CuPy 真能用就走 CUDA，否则静默用 numpy/OpenBLAS"""
        if self.device == "gpu" and _probe_cupy():
            self._xp = cp
            self._on_gpu = True
            self.backend_name = "CUDA (CuPy)"
        else:
            self._xp = np
            self._on_gpu = False
            self.backend_name = "向量化 (numpy/OpenBLAS)"

    def init(self, input_n, hidden_n, motor1_n, motor2_n, seed=None):
        if seed is None: seed = self.seed
        self._select_backend()
        self.config = {
            "input": input_n, "hidden": hidden_n,
            "motor1": motor1_n, "motor2": motor2_n,
            "neuron_count": input_n + hidden_n + motor1_n + motor2_n,
        }
        self._build_topology(seed)
        self._build_arrays()

    # ========== 拓扑构建 ==========
    def _build_topology(self, seed):
        xp = self._xp
        N = self.config["neuron_count"]
        N_in = self.config["input"]
        N_h = self.config["hidden"]
        N_m1 = self.config["motor1"]
        N_m2 = self.config["motor2"]

        rng = np.random.default_rng(seed)

        # 偏好点
        side = int(math.sqrt(N_in))
        assert side * side == N_in
        xs = np.linspace(-2, 2, side)
        ys = np.linspace(-2, 2, side)
        pref_x = np.zeros(N, dtype=np.float32)
        pref_y = np.zeros(N, dtype=np.float32)
        k = 0
        for yi in range(side):
            for xi in range(side):
                pref_x[k] = xs[xi]; pref_y[k] = ys[yi]; k += 1
        pref_omega = np.zeros(N, dtype=np.float32)
        m1_start = N_in + N_h
        m2_start = m1_start + N_m1
        if N_m1 > 1:
            pref_omega[m1_start:m2_start] = np.linspace(-1, 1, N_m1).astype(np.float32)
        if N_m2 > 1:
            pref_omega[m2_start:N] = np.linspace(-1, 1, N_m2).astype(np.float32)

        layer = np.full(N, LAYER_HIDDEN, dtype=np.int32)
        layer[:N_in] = LAYER_INPUT_X
        layer[m1_start:m2_start] = LAYER_MOTOR1
        layer[m2_start:N] = LAYER_MOTOR2

        # 稀疏连接 → dense W[N,N]，W[pre,post] = w
        K = 4
        W = np.zeros((N, N), dtype=np.float32)  # dense, 预分配
        syn_pre, syn_post, syn_w = [], [], []

        # 与 C 端一致：按【突触前】神经元发散 K 条边，去重、隐层禁自连
        edge_set = set()

        def fan_out(src_start, src_end, dst_start, dst_end, k, w_lo, w_hi, no_self=False):
            for pre in range(src_start, src_end):
                for _ in range(k):
                    post = int(rng.integers(dst_start, dst_end))
                    if post >= dst_end:
                        continue
                    if no_self and post == pre:
                        continue
                    key = (pre, post)
                    if key in edge_set:
                        continue
                    edge_set.add(key)
                    w = float(rng.uniform(w_lo, w_hi))
                    W[pre, post] = w
                    syn_pre.append(pre); syn_post.append(post); syn_w.append(w)

        fan_out(0, N_in, N_in, N_in + N_h, K, 0.5, 1.5)                       # INPUT → HIDDEN
        fan_out(N_in, N_in + N_h, N_in, N_in + N_h, max(1, K // 2), 0.1, 0.4, no_self=True)  # HIDDEN → HIDDEN
        fan_out(N_in, N_in + N_h, m1_start, m2_start, K, 0.1, 0.3)            # HIDDEN → MOTOR1
        fan_out(N_in, N_in + N_h, m2_start, N, K, 0.1, 0.3)                   # HIDDEN → MOTOR2

        S = len(syn_pre)

        if self._on_gpu:
            self.W = cp.asarray(W)
            self.pref_x = cp.asarray(pref_x)
            self.pref_y = cp.asarray(pref_y)
            self.pref_omega = cp.asarray(pref_omega)
            self.layer = cp.asarray(layer)
            self.syn_pre = cp.asarray(np.array(syn_pre, dtype=np.int32))
            self.syn_post = cp.asarray(np.array(syn_post, dtype=np.int32))
        else:
            self.W = W
            self.pref_x = pref_x
            self.pref_y = pref_y
            self.pref_omega = pref_omega
            self.layer = layer
            self.syn_pre = np.array(syn_pre, dtype=np.int32)
            self.syn_post = np.array(syn_post, dtype=np.int32)

        self._cpu_W = W       # CPU 副本（用于 topology dump）
        self._cpu_pref_x = pref_x
        self._cpu_pref_y = pref_y
        self._cpu_pref_omega = pref_omega
        self._cpu_layer = layer
        self._cpu_syn_pre = np.array(syn_pre, dtype=np.int32)
        self._cpu_syn_post = np.array(syn_post, dtype=np.int32)
        self._cpu_syn_w = np.array(syn_w, dtype=np.float32)

        self.m1_start = m1_start
        self.m2_start = m2_start

        self.topo = {
            "neuron_count": N, "synapse_count": S,
            "input": N_in, "hidden": N_h, "motor1": N_m1, "motor2": N_m2,
            "neurons": [{"id": i, "layer": int(layer[i]),
                          "pref_x": float(pref_x[i]), "pref_y": float(pref_y[i]),
                          "pref_omega": float(pref_omega[i])} for i in range(N)],
            "synapses": [{"pre": int(syn_pre[i]), "post": int(syn_post[i]),
                          "w": float(syn_w[i])} for i in range(S)],
        }

    def _build_arrays(self):
        xp = self._xp
        N = self.config["neuron_count"]
        self.membrane = xp.full(N, MEMBRANE_REST, dtype=xp.float32)
        self.is_spiking = xp.zeros(N, dtype=xp.int32)
        self.prev_spiking = xp.zeros(N, dtype=xp.int32)
        self.last_spike_tick = xp.full(N, -1, dtype=xp.int32)
        self.prev_last_spike = xp.full(N, -1, dtype=xp.int32)
        # 不应期计数 + 输入层掩码
        self.refractory = xp.zeros(N, dtype=xp.int32)
        is_input = np.zeros(N, dtype=np.float32)
        is_input[:self.config["input"]] = 1.0
        self.is_input_mask = xp.asarray(is_input)
        S = self.topo["synapse_count"]
        # 转置权重矩阵预存为连续内存（避免每 tick 转置开销）
        self.Wt = xp.ascontiguousarray(self.W.T)
        self.eligibility_trace = xp.zeros(S, dtype=xp.float32)
        # 物理
        self.arm_theta1 = 0.0
        self.arm_theta2 = 0.0
        self.arm_omega1 = 0.0
        self.arm_omega2 = 0.0
        self.arm_end_x = 1.0
        self.arm_end_y = 0.0
        self.target_x = 0.5
        self.target_y = 0.5
        self.max_reach = 2.0
        # GPU 融合内核编译（失败则回落到纯 CuPy 逐算子路径）
        self._kern = None
        self.arm_dev = None
        if self._on_gpu:
            self.arm_dev = xp.zeros(6, dtype=xp.float64)
            try:
                self._k_tick = xp.RawKernel(_CUDA_SRC, "neuron_tick")
                self._k_wta = xp.RawKernel(_CUDA_SRC, "motor_wta")
                self._k_fin = xp.RawKernel(_CUDA_SRC, "finalize_tick")
                self._k_elig = xp.RawKernel(_CUDA_SRC, "trace_update")
                self._k_arm = xp.RawKernel(_CUDA_SRC, "decode_arm")
                self._kern = True
            except Exception:
                self._kern = None

    # ========== tick 循环 ==========
    def tick(self):
        self.tick_count += 1
        if self._on_gpu and self._kern is not None:
            self._tick_cuda()
        else:
            self._tick_ops()

    def _tick_cuda(self):
        """融合 CUDA 路径：每 tick 仅 6 次 kernel 启动 + 1 次 cuBLAS matvec"""
        cp = self._xp
        N = self.config["neuron_count"]
        N_in = self.config["input"]
        n1 = self.config["motor1"]
        n2 = self.config["motor2"]

        spk_vec = self.is_spiking.astype(cp.float32)
        current = (self.Wt @ spk_vec) * PSP_FACTOR

        g = ((N + 255) // 256,)
        self._k_tick(g, (256,), (
            current, self.membrane, self.refractory, self.is_spiking,
            self.pref_x, self.pref_y,
            N, N_in, float(self.target_x), float(self.target_y),
            float(ENCODE_SIGMA_POS), float(MEMBRANE_REST),
            float(SPIKE_THRESHOLD), float(MEMBRANE_DECAY), REFRACTORY_TICKS,
        ))
        self._k_wta((1,), (1,), (
            self.is_spiking, self.membrane, self.m1_start, n1, max(2, n1 // 3),
        ))
        self._k_wta((1,), (1,), (
            self.is_spiking, self.membrane, self.m2_start, n2, max(2, n2 // 3),
        ))
        self._k_fin(g, (256,), (
            self.is_spiking, self.membrane, self.refractory, self.last_spike_tick,
            N, N_in, self.tick_count, float(MEMBRANE_REST), REFRACTORY_TICKS,
        ))
        S = self.topo["synapse_count"]
        self._k_elig(((S + 255) // 256,), (256,), (
            self.eligibility_trace, self.syn_pre, self.syn_post,
            self.is_spiking, self.last_spike_tick, S, self.tick_count,
            float(LAMBDA_TRACE), float(TAU_STDP),
        ))
        self._k_arm((1,), (1,), (
            self.last_spike_tick, self.pref_omega, self.arm_dev,
            self.m1_start, n1, self.m2_start, n2, self.tick_count,
            float(MAX_OMEGA), float(ARM_DT), float(L1), float(L2),
        ))

    def _tick_ops(self):
        xp = self._xp
        N_in = self.config["input"]

        # Step 0: 编码 → INPUT_X 膜电位直接注入（与 C 端一致：+5.0 增益）
        dx = self.pref_x[:N_in] - self.target_x
        dy = self.pref_y[:N_in] - self.target_y
        sigma = ENCODE_SIGMA_POS
        gauss = xp.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        self.membrane[:N_in] = MEMBRANE_REST + gauss * (SPIKE_THRESHOLD - MEMBRANE_REST + 5.0)

        # Step 1: 突触电流累加（用上一 tick 的发放）
        self.prev_spiking[:] = self.is_spiking
        self.prev_last_spike[:] = self.last_spike_tick
        spk_vec = self.is_spiking.astype(xp.float32)
        current = (self.Wt @ spk_vec) * PSP_FACTOR

        # Step 2: 不应期 → 泄漏衰减 → 阈值判定 → 运动层 k-WTA
        is_input = self.is_input_mask > 0.5
        refr_active = (self.refractory > 0) & (~is_input)

        # 非输入层：泄漏衰减（m*decay + rest*(1-decay)）+ 电流；输入层：只加电流
        self.membrane = xp.where(
            is_input,
            self.membrane + current,
            self.membrane * MEMBRANE_DECAY + MEMBRANE_REST * (1.0 - MEMBRANE_DECAY) + current
        )
        # 不应期：钳在静息电位，且本 tick 不发放
        self.membrane = xp.where(refr_active, MEMBRANE_REST, self.membrane)
        self.refractory = xp.where(refr_active, self.refractory - 1, self.refractory)

        spiking = (self.membrane >= SPIKE_THRESHOLD) & (~refr_active)

        # 运动层 k-WTA：每关节保留膜电位最高的 K 个（无主机同步）
        spiking_f = xp.where(spiking, self.membrane, xp.float32(-1e9))
        for base, count in ((self.m1_start, self.config["motor1"]),
                            (self.m2_start, self.config["motor2"])):
            K = max(2, count // 3)
            seg = spiking_f[base:base + count]
            kth = xp.partition(seg, count - K)[count - K]
            # 与原始发放掩码取交，避免发放数 < K 时把整层误判为发放
            spiking[base:base + count] = (seg >= kth) & spiking[base:base + count]

        self.is_spiking = spiking.astype(xp.int32)

        # 收尾：last_spike_tick / 膜电位重置 / 不应期（全程设备端，无同步）
        non_input = self.is_input_mask < 0.5
        self.last_spike_tick = xp.where(spiking, self.tick_count, self.last_spike_tick)
        self.membrane = xp.where(spiking, xp.float32(MEMBRANE_REST), self.membrane)
        self.refractory = xp.where(spiking & non_input, REFRACTORY_TICKS, self.refractory)

        # Step 3: R-STDP 资格迹（向量化）
        S = self.topo["synapse_count"]
        self.eligibility_trace *= LAMBDA_TRACE
        pre_spk = self.is_spiking[self.syn_pre]
        post_spk = self.is_spiking[self.syn_post]
        post_last = self.last_spike_tick[self.syn_post]
        pre_last = self.last_spike_tick[self.syn_pre]
        delta_post = self.tick_count - post_last
        delta_pre = self.tick_count - pre_last
        tau = TAU_STDP
        stdp_inc = xp.where(
            (pre_spk > 0) & (post_last >= 0),
            xp.exp(-delta_post.astype(xp.float32) / tau),
            xp.float32(0.0)
        )
        stdp_dec = xp.where(
            (post_spk > 0) & (pre_last >= 0),
            xp.exp(-delta_pre.astype(xp.float32) / tau),
            xp.float32(0.0)
        )
        self.eligibility_trace += (stdp_inc - stdp_dec).astype(xp.float32)
        self.eligibility_trace = xp.clip(self.eligibility_trace, -5.0, 5.0)

        # Step 4: 解码 omega1, omega2 + 物理模拟（设备端，无主机同步）
        omega1_cmd = xp.clip(self._decode_omega(self.m1_start, self.m2_start), -MAX_OMEGA, MAX_OMEGA)
        omega2_cmd = xp.clip(self._decode_omega(self.m2_start, self.config["neuron_count"]), -MAX_OMEGA, MAX_OMEGA)

        self.arm_omega1 = omega1_cmd
        self.arm_omega2 = omega2_cmd
        self.arm_theta1 = xp.clip(self.arm_theta1 + omega1_cmd * ARM_DT, -math.pi, math.pi)
        self.arm_theta2 = xp.clip(self.arm_theta2 + omega2_cmd * ARM_DT, -math.pi * 0.9, math.pi * 0.9)
        c1 = xp.cos(self.arm_theta1)
        s1 = xp.sin(self.arm_theta1)
        c12 = xp.cos(self.arm_theta1 + self.arm_theta2)
        s12 = xp.sin(self.arm_theta1 + self.arm_theta2)
        self.arm_end_x = L1 * c1 + L2 * c12
        self.arm_end_y = L1 * s1 + L2 * s12

    def _decode_omega(self, start, end):
        """方向池分离平均：正/负 pref_omega 分池，谁活跃多就用谁的池内平均（与 C 端一致）"""
        xp = self._xp
        if end <= start:
            return xp.float32(0.0)
        last = self.last_spike_tick[start:end]
        since = self.tick_count - last
        active = ((last >= 0) & (since >= 0) & (since <= 5))
        prefs = self.pref_omega[start:end]
        pos_mask = active & (prefs > 0)
        neg_mask = active & (prefs < 0)
        pos_c = xp.sum(pos_mask)
        neg_c = xp.sum(neg_mask)
        pos_s = xp.sum(xp.where(pos_mask, prefs, xp.float32(0.0)))
        neg_s = xp.sum(xp.where(neg_mask, prefs, xp.float32(0.0)))
        return xp.where(
            pos_c > neg_c, pos_s / xp.maximum(pos_c, xp.float32(1.0)),
            xp.where(neg_c > pos_c, neg_s / xp.maximum(neg_c, xp.float32(1.0)), xp.float32(0.0))
        )

    def set_target(self, x, y):
        self.target_x = float(x)
        self.target_y = float(y)

    # ========== 快照 ==========
    def get_snapshot_arrays(self):
        """零 dict 快路径：直接返回 numpy 数组"""
        xp = self._xp
        if self._on_gpu:
            m = xp.asnumpy(self.membrane)
            s = xp.asnumpy(self.is_spiking)
            lst = xp.asnumpy(self.last_spike_tick)
            if self._kern is not None:
                a = xp.asnumpy(self.arm_dev)
                t1, t2, o1, o2, ex, ey = (float(a[0]), float(a[1]), float(a[2]),
                                        float(a[3]), float(a[4]), float(a[5]))
            else:
                t1 = float(self.arm_theta1); t2 = float(self.arm_theta2)
                o1 = float(self.arm_omega1); o2 = float(self.arm_omega2)
                ex = float(self.arm_end_x); ey = float(self.arm_end_y)
        else:
            m = self.membrane
            s = self.is_spiking
            lst = self.last_spike_tick
            t1 = float(self.arm_theta1); t2 = float(self.arm_theta2)
            o1 = float(self.arm_omega1); o2 = float(self.arm_omega2)
            ex = float(self.arm_end_x); ey = float(self.arm_end_y)
        arm = np.array([
            t1, t2, o1, o2, ex, ey,
            self.target_x, self.target_y,
            max(0.0, 1.0 - math.hypot(ex - self.target_x, ey - self.target_y) / self.max_reach),
            self.max_reach,
        ], dtype=np.float64)
        meta = np.array([self.tick_count], dtype=np.int32)
        return (m.astype(np.float32), s.astype(np.int32),
                lst.astype(np.int32), arm, meta)

    def get_snapshot(self):
        mem_a, spk_a, lst_a, arm_a, meta_a = self.get_snapshot_arrays()
        N = self.config["neuron_count"]
        mem = mem_a.tolist()
        spk = spk_a.tolist()
        lst2 = lst_a.tolist()
        neurons = [{"m": mem[i], "s": int(spk[i]), "lst": int(lst2[i])} for i in range(N)]
        return {
            "tick": int(meta_a[0]),
            "neurons": neurons,
            "arm": {
                "t1": float(arm_a[0]), "t2": float(arm_a[1]),
                "o1": float(arm_a[2]), "o2": float(arm_a[3]),
                "ex": float(arm_a[4]), "ey": float(arm_a[5]),
                "tx": float(arm_a[6]), "ty": float(arm_a[7]),
                "r": float(arm_a[8]),
                "mr": float(arm_a[9]),
            },
        }

    def get_topology(self):
        return self.topo
