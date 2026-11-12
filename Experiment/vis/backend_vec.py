"""backend_vec.py —— 统一 numpy / CUDA 向量化后端（v6 闭环版）

设计：
  一套代码，xp = numpy → CPU 模式 / xp = cupy → GPU 模式
  tick 循环完全向量化（OpenBLAS 多线程 / cuBLAS + 5 个融合 CUDA kernel）

v6 与 C 端 benchmark2_core.c 的动力学对齐（修复 GPU 下机械臂「乱走」）：
  1. 输入层由「目标绝对坐标」改为「target - end 相对位置」+ 56 个本体感觉
     神经元（θ1 / θ2 / ψ1 / ψ2 环形角编码）→ 建立闭环反馈，网络能感知
     「我现在在哪、臂形是什么」，而不是永远输出一个固定方向。
  2. 补上 INPUT → MOTOR 直连通路（C 端 K=12），感知信息可绕过隐层直接
     驱动运动层；运动层偏好角速度改为 [-0.5, 0.5]，与 MAX_OMEGA 匹配。
  3. 解码改为「当前 tick 的 k-WTA winner 的 pref_omega 均值」，
     去掉原 5 tick 历史窗口（历史窗口会把不同方向的 winner 混在一起，
     造成方向信号滞后与来回翻转）。
  4. k-WTA 与 C 端一致：按膜电位取前 K（严格大于 + 索引小优先），
     GPU 串行扫描与 numpy stable argsort 完全一致，不再有 tie-break 分叉。
  5. 资格迹钳位参数化（CUDA kernel 不再硬编码 ±5），STDP 双分支幅度
     与 I→M 跳过 LTD 规则对齐 C 端，杜绝权重单向漂移饱和。
  6. 加入与 C 端同构的隐层/运动层基础电流 + 膜电位噪声 + 解码探索噪声；
     噪声由同一份 CPU 随机数池提供，GPU / numpy 两条路径逐位可复现。

v7 补「肩关节 θ1 在 ±π 吸收边界卡死、末端绕小圈不靠近目标」：
  7. 肩关节 θ1 改为周期性回绕，不再钳死在 ±π 形成吸收边界（旧版一旦
     顶边界，末端自由度只剩 θ2，只能在以肘为圆心的单位圆上打转）。

所有状态存在一个 6 元 arm_dev 数组（θ1,θ2,ω1,ω2,ex,ey），
GPU 路径由 decode_arm kernel 直接更新，numpy 路径由 Python 更新，
get_arm() / snapshot() 共用一份按 tick 缓存的读回，避免每 tick 多次同步。
"""
import math, os, sys, warnings

import numpy as np


def _silent_devnull():
    return open(os.devnull, "w")


# ---------- CUDA 环境自动配置（pip 装的 CUDA toolkit） ----------
def _setup_cuda_env():
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
        pass


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

PSP_FACTOR = 25.0
ENCODE_SIGMA_POS = 2.0      # C: ENCODE_SIGMA_POS
ENCODE_SIGMA_PROP = 0.6     # C: ENCODE_SIGMA_PROP
PROP_N = 56                 # 12×θ1 + 12×θ2 + 16×ψ1 + 16×ψ2
TAU_STDP = 50.0
ARM_DT = 0.02
MAX_OMEGA = 0.5
L1 = L2 = 1.0
NOISE_POOL = 512            # 共享噪声池行数，tick % NOISE_POOL 循环取用

LAYER_INPUT_X = 0
LAYER_HIDDEN = 1
LAYER_MOTOR1 = 2
LAYER_MOTOR2 = 3


# ============================================================
# 融合 CUDA 内核（GPU 路径：每 tick 5 次启动 + 1 次 cuBLAS matvec）
# ============================================================
_CUDA_SRC = r"""
extern "C" __global__
void neuron_tick(const float* __restrict__ current,
                 float* membrane, int* refr, int* spiking,
                 const float* __restrict__ pref_x,
                 const float* __restrict__ pref_y,
                 const float* __restrict__ prop_ang,
                 const int*   __restrict__ prop_tag,
                 const float* __restrict__ bias,
                 const float* __restrict__ noise,
                 const double* __restrict__ arm,
                 long long N, long long N_pos, long long N_in,
                 long long noise_row,
                 double tx, double ty,
                 double sigma_pos, double sigma_prop,
                 double rest, double thr, double decay,
                 long long refr_ticks)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double PI = 3.14159265358979323846;
    float m;
    if (i < N_in) {
        float g;
        if (i < N_pos) {
            /* 相对位置编码：target - 当前末端（闭环，与 C Step 0 一致） */
            float dx = pref_x[i] - (float)(tx - arm[4]);
            float dy = pref_y[i] - (float)(ty - arm[5]);
            g = expf(-(dx * dx + dy * dy) /
                     (2.0f * (float)sigma_pos * (float)sigma_pos));
        } else {
            /* 本体感觉：关节角 / 相对目标方位角，环形距离 */
            long long j = i - N_pos;
            int tag = prop_tag[j];
            double th;
            if (tag == 0)      th = arm[0];
            else if (tag == 1) th = arm[1];
            else if (tag == 2) th = atan2(ty, tx) - arm[0];
            else {
                double ex = cos(arm[0]), ey = sin(arm[0]);
                th = atan2(ty - ey, tx - ex) - (arm[0] + arm[1]);
            }
            double d = th - (double)prop_ang[j];
            while (d >  PI) d -= 2.0 * PI;
            while (d < -PI) d += 2.0 * PI;
            g = (float)exp(-d * d / (2.0 * sigma_prop * sigma_prop));
        }
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
    m = membrane[i] * (float)decay
        + (float)rest * (1.0f - (float)decay) + current[i];
    m += bias[i] + noise[(long long)noise_row * N + i];
    spiking[i] = (m >= (float)thr) ? 1 : 0;
    membrane[i] = m;
}

/* k-WTA：按膜电位取前 K，严格大于保证索引小者优先（与 numpy stable argsort 一致） */
extern "C" __global__
void motor_wta(int* spiking, const float* __restrict__ membrane,
               long long base, long long count, long long K)
{
    if (blockIdx.x || threadIdx.x) return;
    for (long long i = 0; i < count; ++i) spiking[base + i] = 0;
    for (long long k = 0; k < K; ++k) {
        long long best = -1; float bestv = -1e30f;
        for (long long i = 0; i < count; ++i) {
            if (spiking[base + i] == 1) continue;
            float v = membrane[base + i];
            if (v > bestv) { bestv = v; best = i; }
        }
        if (best < 0) break;
        spiking[base + best] = 1;
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

/* 事件驱动资格迹：幅度 / 双分支 / I→M 跳过 LTD / 钳位，全部与 C 端一致 */
extern "C" __global__
void trace_update(float* e, const int* __restrict__ pre,
                  const int* __restrict__ post,
                  const int* __restrict__ spiking,
                  const int* __restrict__ last_spike,
                  const unsigned char* __restrict__ is_i_to_m,
                  long long S, long long tick,
                  double lambda, double tau,
                  double a_plus, double a_minus, double tclamp)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;
    int p = pre[i], q = post[i];
    float v = e[i] * (float)lambda;
    long long dq = tick - last_spike[q];
    long long dp = tick - last_spike[p];
    if (!is_i_to_m[i] && spiking[p] && last_spike[q] > 0 && dq > 0)
        v -= (float)(a_minus * exp(-(double)dq / tau));
    if (spiking[q] && last_spike[p] > 0 && dp > 0)
        v += (float)(a_plus * exp(-(double)dp / tau));
    if (tclamp > 0.0) {
        if (v >  (float)tclamp) v = (float)tclamp;
        if (v < -(float)tclamp) v = -(float)tclamp;
    }
    e[i] = v;
}

/* 解码：本 tick WTA winner 的 pref_omega 均值；肩关节 θ1 周期性回绕，
 * 避免 ±π 吸收边界。命令可由外部通过 motor_override 注入（如教师信号）。 */
extern "C" __global__
void decode_arm(const int* __restrict__ spiking,
                const float* __restrict__ pref_omega,
                double* arm,
                long long m1s, long long m1n,
                long long m2s, long long m2n,
                double max_omega, double dt, double L1, double L2,
                double nz1, double nz2)
{
    if (blockIdx.x || threadIdx.x) return;
    double s = 0.0; long long c = 0;
    for (long long i = m1s; i < m1s + m1n; ++i)
        if (spiking[i]) { s += (double)pref_omega[i]; ++c; }
    double o1 = (c > 0) ? s / (double)c : 0.0;
    s = 0.0; c = 0;
    for (long long i = m2s; i < m2s + m2n; ++i)
        if (spiking[i]) { s += (double)pref_omega[i]; ++c; }
    double o2 = (c > 0) ? s / (double)c : 0.0;
    o1 += nz1; o2 += nz2;
    if (o1 >  max_omega) o1 =  max_omega;
    if (o1 < -max_omega) o1 = -max_omega;
    if (o2 >  max_omega) o2 =  max_omega;
    if (o2 < -max_omega) o2 = -max_omega;

    const double PI = 3.14159265358979323846;
    double t1 = arm[0] + o1 * dt;
    double t2 = arm[1] + o2 * dt;
    if (t1 >  PI) t1 -= 2.0 * PI;      /* 肩关节 360°：回绕而非钳死 */
    if (t1 < -PI) t1 += 2.0 * PI;
    if (t2 >  PI * 0.9) t2 =  PI * 0.9;
    if (t2 < -PI * 0.9) t2 = -PI * 0.9;
    arm[0] = t1; arm[1] = t2; arm[2] = o1; arm[3] = o2;
    arm[4] = L1 * cos(t1) + L2 * cos(t1 + t2);
    arm[5] = L1 * sin(t1) + L2 * sin(t1 + t2);
}
"""


class VecBackend:
    def __init__(self, device="cpu", seed=42):
        self.device = device
        self.seed = seed
        self.tick_count = 0
        self.config = None
        self.topo = None
        self._xp = np
        self._on_gpu = False
        self.backend_name = "—"

    def _select_backend(self):
        if self.device == "gpu" and _probe_cupy():
            self._xp = cp
            self._on_gpu = True
            self.backend_name = "CUDA (CuPy)"
        else:
            self._xp = np
            self._on_gpu = False
            self.backend_name = "向量化 (numpy/OpenBLAS)"

    # ========== 初始化 ==========
    def init(self, input_n, hidden_n, motor1_n, motor2_n, seed=None):
        if seed is None:
            seed = self.seed
        self._select_backend()
        N_pos = int(input_n)
        N_in = N_pos + PROP_N
        N_h, N_m1, N_m2 = int(hidden_n), int(motor1_n), int(motor2_n)
        self.config = {
            "input": N_in,            # 输入层总数（含本体感觉），供 GUI 分区
            "input_pos": N_pos,
            "hidden": N_h, "motor1": N_m1, "motor2": N_m2,
            "neuron_count": N_in + N_h + N_m1 + N_m2,
        }
        self._build_topology(seed)
        self._build_arrays()
        self.episode_reset(0.5, 0.5)

    def _build_topology(self, seed):
        xp = self._xp
        cfg = self.config
        N = cfg["neuron_count"]
        N_pos = cfg["input_pos"]
        N_in = cfg["input"]
        N_h, N_m1, N_m2 = cfg["hidden"], cfg["motor1"], cfg["motor2"]
        m1_start = N_in + N_h
        m2_start = m1_start + N_m1

        rng = np.random.default_rng(seed)

        # --- 位置编码偏好点 ---
        side = max(2, int(round(math.sqrt(N_pos))))
        xs = np.linspace(-2.0, 2.0, side)
        ys = np.linspace(-2.0, 2.0, side)
        pref_x = np.zeros(N, dtype=np.float32)
        pref_y = np.zeros(N, dtype=np.float32)
        for k in range(N_pos):
            r, c = divmod(k, side)
            pref_x[k] = xs[c % side]
            pref_y[k] = ys[r % side]

        # --- 本体感觉偏好角（与 C 端 PROP_N 分组一致） ---
        prop_tag = np.zeros(PROP_N, dtype=np.int32)
        prop_ang = np.zeros(PROP_N, dtype=np.float32)
        groups = ((0, 12), (12, 12), (24, 16), (40, 16))
        for tag, (off, gn) in enumerate(groups):
            for j in range(gn):
                prop_tag[off + j] = tag
                prop_ang[off + j] = -math.pi + 2.0 * math.pi * j / max(1, gn - 1)

        pref_omega = np.zeros(N, dtype=np.float32)
        for i in range(N_pos, N_in):
            pref_omega[i] = prop_ang[i - N_pos]
        if N_m1 > 1:
            pref_omega[m1_start:m2_start] = np.linspace(-0.5, 0.5, N_m1, dtype=np.float32)
        if N_m2 > 1:
            pref_omega[m2_start:N] = np.linspace(-0.5, 0.5, N_m2, dtype=np.float32)

        layer = np.full(N, LAYER_HIDDEN, dtype=np.int32)
        layer[:N_in] = LAYER_INPUT_X
        layer[m1_start:m2_start] = LAYER_MOTOR1
        layer[m2_start:N] = LAYER_MOTOR2

        # --- 基础电流 / 噪声（与 C 端：hidden +2.0、motor +3.0、σ=0.5） ---
        bias = np.zeros(N, dtype=np.float32)
        bias[N_in:m1_start] = 2.0
        bias[m1_start:N] = 3.0
        noise_rng = np.random.default_rng(int(seed) + 101)
        noise = noise_rng.normal(0.0, 0.5, (NOISE_POOL, N)).astype(np.float32)
        exp_rng = np.random.default_rng(int(seed) + 202)
        exp_noise = exp_rng.normal(0.0, 0.02, (NOISE_POOL, 2)).astype(np.float64)

        # --- 拓扑（与 C 端同构：INPUT→HIDDEN 密、H→H 稀疏、H→M、INPUT→M 直连） ---
        K = 8
        W = np.zeros((N, N), dtype=np.float32)
        syn_pre, syn_post, syn_w = [], [], []
        edge_set = set()

        def fan_out(src_start, src_end, dst_start, dst_end, k, w_lo, w_hi,
                    no_self=False):
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
                    syn_pre.append(pre)
                    syn_post.append(post)
                    syn_w.append(w)

        fan_out(0, N_in, N_in, N_in + N_h, 20, 1.0, 3.0)                  # INPUT → HIDDEN
        fan_out(N_in, N_in + N_h, N_in, N_in + N_h,
                max(1, K // 2), 0.1, 0.4, no_self=True)                    # HIDDEN → HIDDEN
        fan_out(N_in, N_in + N_h, m1_start, m2_start, K, 0.3, 0.7)        # HIDDEN → MOTOR1
        fan_out(N_in, N_in + N_h, m2_start, N, K, 0.3, 0.7)               # HIDDEN → MOTOR2
        fan_out(0, N_in, m1_start, m2_start, 12, 1.5, 2.5)                # INPUT → MOTOR1（直连）
        fan_out(0, N_in, m2_start, N, 12, 1.5, 2.5)                       # INPUT → MOTOR2（直连）

        S = len(syn_pre)
        pre_arr = np.array(syn_pre, dtype=np.int32)
        post_arr = np.array(syn_post, dtype=np.int32)
        w_arr = np.array(syn_w, dtype=np.float32)
        is_i_to_m = ((pre_arr < N_in) & (post_arr >= m1_start)).astype(np.uint8)

        def dev(a):
            return xp.asarray(a) if self._on_gpu else a

        self.W = dev(W)
        self.Wt = xp.ascontiguousarray(self.W.T)
        self.pref_x = dev(pref_x)
        self.pref_y = dev(pref_y)
        self.pref_omega = dev(pref_omega)
        self.prop_ang = dev(prop_ang)
        self.prop_tag = dev(prop_tag.astype(np.int32))
        self.bias = dev(bias)
        self.noise = dev(noise)
        self.layer = dev(layer)
        self.syn_pre = dev(pre_arr)
        self.syn_post = dev(post_arr)
        self.is_i_to_m = dev(is_i_to_m)
        self._exp_noise = exp_noise            # 常驻 CPU，每 tick 取两个标量

        # 编码用的 CPU 副本（PROP_N 很小，放主机算再上传，GPU/numpy 同源）
        self._cpu_pref_x = pref_x[:N_pos].copy()
        self._cpu_pref_y = pref_y[:N_pos].copy()
        self._cpu_prop_tag = prop_tag.copy()
        self._cpu_prop_ang = prop_ang.copy()

        self.m1_start, self.m2_start = m1_start, m2_start
        self.N_pos, self.N_in = N_pos, N_in
        self.topo = {
            "neuron_count": N, "synapse_count": S,
            "input": N_in, "hidden": N_h, "motor1": N_m1, "motor2": N_m2,
            "neurons": [{"id": i, "layer": int(layer[i]),
                         "pref_x": float(pref_x[i]), "pref_y": float(pref_y[i]),
                         "pref_omega": float(pref_omega[i])} for i in range(N)],
            "synapses": [{"pre": int(pre_arr[i]), "post": int(post_arr[i]),
                          "w": float(w_arr[i])} for i in range(S)],
        }

    def _build_arrays(self):
        xp = self._xp
        N = self.config["neuron_count"]
        S = self.topo["synapse_count"]
        self.membrane = xp.full(N, MEMBRANE_REST, dtype=xp.float32)
        self.is_spiking = xp.zeros(N, dtype=xp.int32)
        self.last_spike_tick = xp.full(N, -1, dtype=xp.int32)
        self.refractory = xp.zeros(N, dtype=xp.int32)
        self.eligibility_trace = xp.zeros(S, dtype=xp.float32)
        self.arm_dev = xp.zeros(6, dtype=xp.float64)
        self._arm_cache = None
        self._arm_cache_tick = -1

        self.hp = {
            "learning_rate": 0.15,
            "tau_stdp": TAU_STDP,
            "tau_trace": 2000.0,
            "lambda_trace": math.exp(-1.0 / 2000.0),
            "stdp_a_plus": 0.05,
            "stdp_a_minus": 0.06,
            "weight_decay_lambda": 1e-5,
            "reward_baseline_window": 200,
            "psp_factor": PSP_FACTOR,
            "trace_clamp": 1.0,
            "decode_mode": 0,
            "wta_k": 3,
            "use_reward_baseline": 1,
            "el_trace": 1,
            "reward_mod": 1,
            "reward_baseline": 0.0,
            "current_tick": 0,
            "encode_sigma_pos": ENCODE_SIGMA_POS,
            "encode_sigma_prop": ENCODE_SIGMA_PROP,
            "explore_noise": 0.02,
            "membrane_noise": 0.5,
        }
        self._reward_queue = []
        self._baseline_hist = []
        self.target_x, self.target_y = 0.5, 0.5
        self.max_reach = L1 + L2
        self.arm_theta1 = self.arm_theta2 = 0.0
        self.arm_omega1 = self.arm_omega2 = 0.0
        self.motor_override = None   # 外部注入的运动指令 (o1, o2)；None=用网络解码
        self.arm_end_x, self.arm_end_y = 2.0, 0.0

        self._kern = None
        if self._on_gpu:
            try:
                self._k_tick = xp.RawKernel(_CUDA_SRC, "neuron_tick")
                self._k_wta = xp.RawKernel(_CUDA_SRC, "motor_wta")
                self._k_fin = xp.RawKernel(_CUDA_SRC, "finalize_tick")
                self._k_elig = xp.RawKernel(_CUDA_SRC, "trace_update")
                self._k_arm = xp.RawKernel(_CUDA_SRC, "decode_arm")
                self._kern = True
            except Exception:
                self._kern = None

    # ========== tick ==========
    def tick(self):
        self._apply_reward()
        self.tick_count += 1
        self.hp["current_tick"] = self.tick_count
        self._arm_cache = None
        if self._on_gpu and self._kern is not None:
            self._tick_cuda()
        else:
            self._tick_ops()

    def _kernel_args(self):
        hp = self.hp
        return (float(self.target_x), float(self.target_y),
                float(hp["encode_sigma_pos"]), float(hp["encode_sigma_prop"]),
                float(MEMBRANE_REST), float(SPIKE_THRESHOLD),
                float(MEMBRANE_DECAY), REFRACTORY_TICKS)

    def _tick_cuda(self):
        cp_ = self._xp
        N, N_pos, N_in = self.config["neuron_count"], self.N_pos, self.N_in
        n1, n2 = self.config["motor1"], self.config["motor2"]
        hp = self.hp

        spk_vec = self.is_spiking.astype(cp_.float32)
        current = (self.Wt @ spk_vec) * hp["psp_factor"]
        tx, ty, sp, spr, rest, thr, decay, refr = self._kernel_args()
        row = self.tick_count % NOISE_POOL
        nz = self._exp_noise[row]

        grid = ((N + 255) // 256,)
        self._k_tick(grid, (256,), (
            current, self.membrane, self.refractory, self.is_spiking,
            self.pref_x, self.pref_y, self.prop_ang, self.prop_tag,
            self.bias, self.noise, self.arm_dev,
            N, N_pos, N_in, row,
            tx, ty, sp, spr, rest, thr, decay, refr,
        ))
        K = max(1, min(64, int(hp["wta_k"])))
        self._k_wta((1,), (1,), (self.is_spiking, self.membrane,
                                 self.m1_start, n1, K))
        self._k_wta((1,), (1,), (self.is_spiking, self.membrane,
                                 self.m2_start, n2, K))
        self._k_fin(grid, (256,), (
            self.is_spiking, self.membrane, self.refractory,
            self.last_spike_tick, N, N_in, self.tick_count, rest, refr,
        ))
        S = self.topo["synapse_count"]
        self._k_elig(((S + 255) // 256,), (256,), (
            self.eligibility_trace, self.syn_pre, self.syn_post,
            self.is_spiking, self.last_spike_tick, self.is_i_to_m,
            S, self.tick_count,
            float(hp["lambda_trace"]), float(hp["tau_stdp"]),
            float(hp["stdp_a_plus"]), float(hp["stdp_a_minus"]),
            float(hp["trace_clamp"]),
        ))
        self._k_arm((1,), (1,), (
            self.is_spiking, self.pref_omega, self.arm_dev,
            self.m1_start, n1, self.m2_start, n2,
            float(MAX_OMEGA), float(ARM_DT), float(L1), float(L2),
            float(nz[0] * hp["explore_noise"] / 0.02),
            float(nz[1] * hp["explore_noise"] / 0.02),
        ))

    def _input_encoding(self, xp):
        """返回输入层膜电位（相对位置 + 本体感觉环形角）

        全部在主机侧用 numpy 算（只有 N_pos+56 个数），再一次性上传设备，
        保证 CUDA / numpy 两条路径的输入编码逐位一致。
        """
        hp = self.hp
        tx, ty = self.target_x, self.target_y
        a = xp.asnumpy(self.arm_dev) if self._on_gpu else self.arm_dev
        rel_x, rel_y = tx - a[4], ty - a[5]
        dx = self._cpu_pref_x - rel_x
        dy = self._cpu_pref_y - rel_y
        sig = hp["encode_sigma_pos"]
        pos = np.exp(-(dx * dx + dy * dy) / (2.0 * sig * sig))

        th1, th2 = float(a[0]), float(a[1])
        e1x, e1y = math.cos(th1), math.sin(th1)   # 肘部：L1 = 1
        ang = np.array([th1, th2,
                        math.atan2(ty, tx) - th1,
                        math.atan2(ty - e1y, tx - e1x) - (th1 + th2)],
                       dtype=np.float64)
        d = ang[self._cpu_prop_tag] - self._cpu_prop_ang
        d = (d + math.pi) % (2.0 * math.pi) - math.pi
        prop = np.exp(-(d * d) / (2.0 * hp["encode_sigma_prop"] ** 2))
        g = np.concatenate((pos, prop)).astype(np.float32)
        out = MEMBRANE_REST + g * (SPIKE_THRESHOLD - MEMBRANE_REST + 5.0)
        return xp.asarray(out, dtype=xp.float32)

    def _tick_ops(self):
        xp = self._xp
        N, N_in = self.config["neuron_count"], self.N_in
        n1, n2 = self.config["motor1"], self.config["motor2"]
        hp = self.hp

        # 突触电流用「上一 tick」的发放
        spk_vec = self.is_spiking.astype(xp.float32)
        current = (self.Wt @ spk_vec) * hp["psp_factor"]

        # Step 0+2: 输入层持续编码；非输入层泄漏积分 + 基础电流 + 噪声
        m_in = self._input_encoding(xp)
        self.membrane[:N_in] = m_in
        self.is_spiking[:N_in] = (m_in >= SPIKE_THRESHOLD).astype(xp.int32)

        row = self.tick_count % NOISE_POOL
        refr_active = self.refractory[N_in:] > 0
        base = (self.membrane[N_in:] * MEMBRANE_DECAY
                + MEMBRANE_REST * (1.0 - MEMBRANE_DECAY)
                + current[N_in:]
                + self.bias[N_in:]
                + self.noise[row][N_in:] * (hp["membrane_noise"] / 0.5))
        m = xp.where(refr_active, xp.float32(MEMBRANE_REST), base)
        self.membrane[N_in:] = m
        self.refractory[N_in:] = xp.where(refr_active,
                                          self.refractory[N_in:] - 1,
                                          self.refractory[N_in:])
        spiking = xp.concatenate((
            self.is_spiking[:N_in],
            (m >= SPIKE_THRESHOLD).astype(xp.int32),
        ))

        # Step 2b: k-WTA（膜电位降序 + 索引升序，与 CUDA/C 端一致）
        K = max(1, min(64, int(hp["wta_k"])))
        for b, c in ((self.m1_start, n1), (self.m2_start, n2)):
            seg = self.membrane[b:b + c]
            seg_h = xp.asnumpy(seg) if self._on_gpu else seg
            idx = np.argsort(-seg_h, kind="stable")[:K]
            mask = xp.zeros(c, dtype=xp.int32)
            mask[xp.asarray(idx, dtype=xp.int32)] = 1
            spiking[b:b + c] = mask
        self.is_spiking = spiking.astype(xp.int32)

        # Step 2c: 收尾
        fired = self.is_spiking > 0
        self.last_spike_tick = xp.where(fired, self.tick_count, self.last_spike_tick)
        self.membrane = xp.where(fired, xp.float32(MEMBRANE_REST), self.membrane)
        self.refractory[N_in:] = xp.where(fired[N_in:], REFRACTORY_TICKS,
                                          self.refractory[N_in:])

        # Step 3: 资格迹（与 CUDA kernel 同公式）
        hp_ = hp
        self.eligibility_trace *= hp_["lambda_trace"]
        pre_spk = self.is_spiking[self.syn_pre]
        post_spk = self.is_spiking[self.syn_post]
        post_last = self.last_spike_tick[self.syn_post]
        pre_last = self.last_spike_tick[self.syn_pre]
        dq = self.tick_count - post_last
        dp = self.tick_count - pre_last
        tau = hp_["tau_stdp"]
        ltd = xp.where((self.is_i_to_m == 0) & (pre_spk > 0)
                       & (post_last > 0) & (dq > 0),
                       xp.exp(-dq.astype(xp.float32) / tau), xp.float32(0.0))
        ltp = xp.where((post_spk > 0) & (pre_last > 0) & (dp > 0),
                       xp.exp(-dp.astype(xp.float32) / tau), xp.float32(0.0))
        self.eligibility_trace += (hp_["stdp_a_plus"] * ltp
                                   - hp_["stdp_a_minus"] * ltd).astype(xp.float32)
        cl = hp_["trace_clamp"]
        if cl > 0:
            self.eligibility_trace = xp.clip(self.eligibility_trace, -cl, cl)
        else:
            self.eligibility_trace = xp.clip(self.eligibility_trace, -5.0, 5.0)

        # Step 4: 解码 + 物理
        nz = self._exp_noise[row] * (hp["explore_noise"] / 0.02)
        o1 = self._decode_omega(self.m1_start, self.m2_start) + float(nz[0])
        o2 = self._decode_omega(self.m2_start, N) + float(nz[1])
        o1 = float(xp.clip(o1, -MAX_OMEGA, MAX_OMEGA))
        o2 = float(xp.clip(o2, -MAX_OMEGA, MAX_OMEGA))
        # 命令默认直接用网络解码；外部（如 transfer_lab 的教师信号）可通过
        # self.motor_override = (o1, o2) 注入，覆盖本 tick 的运动指令。
        if self.motor_override is not None:
            o1, o2 = self.motor_override
        t1 = float(self.arm_dev[0]) + o1 * ARM_DT
        if t1 > math.pi:
            t1 -= 2.0 * math.pi       # 肩关节 360°：回绕而非钳死
        elif t1 < -math.pi:
            t1 += 2.0 * math.pi
        t2 = float(xp.clip(float(self.arm_dev[1]) + o2 * ARM_DT,
                           -math.pi * 0.9, math.pi * 0.9))
        self.arm_dev[0] = t1
        self.arm_dev[1] = t2
        self.arm_dev[2] = o1
        self.arm_dev[3] = o2
        self.arm_dev[4] = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
        self.arm_dev[5] = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)

    def _decode_omega(self, start, end):
        """本 tick WTA winner 的 pref_omega 均值（无历史窗口）"""
        if end <= start:
            return 0.0
        xp = self._xp
        fired = self.is_spiking[start:end] > 0
        prefs = self.pref_omega[start:end]
        c = xp.sum(fired)
        if int(c) == 0:
            return 0.0
        return float(xp.sum(xp.where(fired, prefs, xp.float32(0.0))) / c)

    # ========== 目标 / 手臂 ==========
    def set_target(self, x, y):
        self.target_x = float(x)
        self.target_y = float(y)

    def _read_arm(self):
        """按 tick 缓存的 arm 读回：GPU 每 tick 最多一次同步"""
        if self._arm_cache is not None and self._arm_cache_tick == self.tick_count:
            return self._arm_cache
        a = self._xp.asnumpy(self.arm_dev) if self._on_gpu else self.arm_dev
        vals = tuple(float(v) for v in a[:6])
        (self.arm_theta1, self.arm_theta2, self.arm_omega1, self.arm_omega2,
         self.arm_end_x, self.arm_end_y) = vals
        self._arm_cache = vals
        self._arm_cache_tick = self.tick_count
        return vals

    def get_arm(self):
        a = self._read_arm()
        return (a[4], a[5], self.target_x, self.target_y, self.max_reach)

    # ========== 训练 / 调参 ==========
    def inject_reward(self, r, delay=0):
        self._reward_queue.append(float(r))

    def inject_reward2(self, r1, r2, delay=0):
        self._reward_queue.append(0.5 * (float(r1) + float(r2)))

    def _apply_reward(self):
        if not self._reward_queue:
            return
        r = self._reward_queue.pop(0)
        hp = self.hp
        hist = self._baseline_hist
        hist.append(r)
        win = max(1, int(hp["reward_baseline_window"]))
        if len(hist) > win:
            del hist[:len(hist) - win]
        base = (sum(hist) / len(hist)) if hp["use_reward_baseline"] else 0.0
        hp["reward_baseline"] = base
        err = r - base
        if not hp["reward_mod"] or not hp["el_trace"] or abs(err) < 0.001:
            return
        xp = self._xp
        dw = xp.asarray(hp["learning_rate"] * err * self.eligibility_trace,
                        dtype=xp.float32)
        pre, post = self.syn_pre, self.syn_post
        wv = xp.clip(self.Wt[post, pre] + dw, 0.001, 10.0)
        self.Wt[post, pre] = wv
        self.W[pre, post] = wv

    def set_param(self, name, value):
        hp = self.hp
        v = float(value)
        if name == "tau_trace":
            if v < 1.0:
                return -1
            hp["tau_trace"] = v
            hp["lambda_trace"] = math.exp(-1.0 / v)
        elif name in ("learning_rate", "tau_stdp", "psp_factor", "trace_clamp",
                      "weight_decay_lambda", "stdp_a_plus", "stdp_a_minus",
                      "encode_sigma_pos", "encode_sigma_prop", "explore_noise",
                      "membrane_noise"):
            if v < 0.0 or (name == "psp_factor" and v <= 0.0):
                return -1
            hp[name] = v
        elif name in ("reward_baseline_window", "wta_k", "decode_mode",
                      "use_reward_baseline", "el_trace", "reward_mod"):
            hp[name] = int(v)
        else:
            return -1
        return 0

    def get_param(self, name):
        return float(self.hp.get(name, float("nan")))

    def episode_reset(self, tx, ty):
        """一轮结束：关节/神经元状态归位，保留权重、资格迹与奖励基线"""
        self.arm_dev[:] = self._xp.asarray([0, 0, 0, 0, 2, 0],
                                           dtype=self._xp.float64)
        self._arm_cache = None
        self._arm_cache_tick = -1
        self.arm_theta1 = self.arm_theta2 = 0.0
        self.arm_omega1 = self.arm_omega2 = 0.0
        self.motor_override = None
        self.arm_end_x, self.arm_end_y = 2.0, 0.0
        self.set_target(float(tx), float(ty))
        self.membrane[:] = MEMBRANE_REST
        self.is_spiking[:] = 0
        self.last_spike_tick[:] = -1
        self.refractory[:] = 0
        self._reward_queue.clear()

    def reset_ema(self):
        pass

    def destroy(self):
        pass

    # ========== 快照 ==========
    def get_snapshot_arrays(self):
        xp = self._xp
        if self._on_gpu:
            m = xp.asnumpy(self.membrane)
            s = xp.asnumpy(self.is_spiking)
            lst = xp.asnumpy(self.last_spike_tick)
        else:
            m, s, lst = self.membrane, self.is_spiking, self.last_spike_tick
        a = self._read_arm()
        arm = np.array([
            a[0], a[1], a[2], a[3], a[4], a[5],
            self.target_x, self.target_y,
            max(0.0, 1.0 - math.hypot(a[4] - self.target_x,
                                      a[5] - self.target_y) / self.max_reach),
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
        neurons = [{"m": mem[i], "s": int(spk[i]), "lst": int(lst2[i])}
                   for i in range(N)]
        return {
            "tick": int(meta_a[0]),
            "neurons": neurons,
            "arm": {
                "t1": float(arm_a[0]), "t2": float(arm_a[1]),
                "o1": float(arm_a[2]), "o2": float(arm_a[3]),
                "ex": float(arm_a[4]), "ey": float(arm_a[5]),
                "tx": float(arm_a[6]), "ty": float(arm_a[7]),
                "r": float(arm_a[8]), "mr": float(arm_a[9]),
            },
        }

    def get_topology(self):
        return self.topo
