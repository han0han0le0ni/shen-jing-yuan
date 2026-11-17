"""GPU 后端 —— CuPy 加速神经元计算核心

架构：
  - 拓扑构建（突触稀疏矩阵）用 C 端一次性完成 → 传拓扑 JSON 给 Python
  - 每 tick 的计算：
      Step 0  编码 →  CuPy（矩阵乘，纯数学）
      Step 1  膜电位更新 →  CuPy 稀疏 SpMV（GPU 加速核心）
      Step 2  spike 判定 + k-WTA → CuPy（topk）
      Step 3  R-STDP 资格迹 → CuPy（向量+门控）
      Step 4  解码 + 物理 → NumPy（CPU，量小）
  - Snapshot：CuPy 数组 → cupy.asnumpy → Python dict

核心思路：把 C 端的 for 循环翻译成 CuPy 张量操作
"""
import sys, os, math, json
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

LAYER_INPUT_X = 0
LAYER_HIDDEN  = 1
LAYER_MOTOR1  = 2
LAYER_MOTOR2  = 3

# 超参数 —— 与 C 端 benchmark2_core.c 保持一致
MEMBRANE_REST = -65.0
SPIKE_THRESHOLD = -50.0
MEMBRANE_DECAY = 0.95
PSP_FACTOR = 25.0
ENCODE_SIGMA_POS = 0.85
INPUT_GAIN = 15.0


class GPUBackend:
    def __init__(self, seed=42):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")
        self.seed = seed
        self.config = None
        self.topo = None
        self.tick_count = 0

    def init(self, input_n, hidden_n, motor1_n, motor2_n, seed=None):
        if seed is None: seed = self.seed
        self.config = {
            "input": input_n, "hidden": hidden_n,
            "motor1": motor1_n, "motor2": motor2_n,
            "neuron_count": input_n + hidden_n + motor1_n + motor2_n,
        }
        rng = np.random.default_rng(seed)
        self._build_topology(rng)
        self._build_cupy_buffers()

    # ---- 拓扑构建（复现 C 端的随机稀疏连接逻辑） ----
    def _build_topology(self, rng):
        N = self.config["neuron_count"]
        N_in = self.config["input"]
        N_h = self.config["hidden"]
        N_m1 = self.config["motor1"]
        N_m2 = self.config["motor2"]

        # 偏好点（与 C 端一致）
        # INPUT_X: 2D 网格
        side = int(math.sqrt(N_in))
        assert side * side == N_in, f"N_in={N_in} must be perfect square"
        xs = np.linspace(-2, 2, side)
        ys = np.linspace(-2, 2, side)
        pref_x = np.zeros(N)
        pref_y = np.zeros(N)
        idx = 0
        for yi in range(side):
            for xi in range(side):
                pref_x[idx] = xs[xi]
                pref_y[idx] = ys[yi]
                idx += 1
        # MOTOR1/MOTOR2: omega 偏好点 -1..1 均匀
        m1_start = N_in + N_h
        m2_start = m1_start + N_m1
        pref_omega = np.zeros(N)
        for k in range(N_m1):
            pref_omega[m1_start + k] = -1.0 + 2.0 * k / max(1, N_m1 - 1)
        for k in range(N_m2):
            pref_omega[m2_start + k] = -1.0 + 2.0 * k / max(1, N_m2 - 1)

        # 层类型
        layer = np.zeros(N, dtype=np.int32)
        layer[N_in:N_in+N_h] = LAYER_HIDDEN
        layer[m1_start:m2_start] = LAYER_MOTOR1
        layer[m2_start:N] = LAYER_MOTOR2

        # 稀疏连接（K=4 per neuron，与 C 端一致）
        K = 4
        pre_list = []
        post_list = []
        w_list = []

        # INPUT → HIDDEN（每个 hidden 选 4 个 input）
        for i in range(N_h):
            post_id = N_in + i
            src_start = 0; src_end = N_in
            src = rng.integers(src_start, src_end, K)
            for s in src:
                pre_list.append(int(s))
                post_list.append(int(post_id))
                w_list.append(float(rng.uniform(0.2, 0.6)))

        # HIDDEN → HIDDEN（递归自连接）
        for i in range(N_h):
            post_id = N_in + i
            src_start = N_in; src_end = N_in + N_h
            src = rng.integers(src_start, src_end, K)
            for s in src:
                pre_list.append(int(s))
                post_list.append(int(post_id))
                w_list.append(float(rng.uniform(0.2, 0.6)))

        # HIDDEN → MOTOR1
        for i in range(N_m1):
            post_id = m1_start + i
            src = rng.integers(N_in, N_in + N_h, K)
            for s in src:
                pre_list.append(int(s))
                post_list.append(int(post_id))
                w_list.append(float(rng.uniform(0.2, 0.6)))

        # HIDDEN → MOTOR2
        for i in range(N_m2):
            post_id = m2_start + i
            src = rng.integers(N_in, N_in + N_h, K)
            for s in src:
                pre_list.append(int(s))
                post_list.append(int(post_id))
                w_list.append(float(rng.uniform(0.2, 0.6)))

        # 递归层（INPUT→INPUT 不需要，MOTOR 之间也不需要）
        # 保持简洁：与 C 端默认 K=4 一致

        S = len(pre_list)
        syn_pre = np.array(pre_list, dtype=np.int32)
        syn_post = np.array(post_list, dtype=np.int32)
        syn_w = np.array(w_list, dtype=np.float64)

        self.topo = {
            "neuron_count": N, "synapse_count": S,
            "input": N_in, "hidden": N_h, "motor1": N_m1, "motor2": N_m2,
            "layer": layer, "pref_x": pref_x, "pref_y": pref_y, "pref_omega": pref_omega,
            "syn_pre": syn_pre, "syn_post": syn_post, "syn_w": syn_w,
            "neurons": [{"id": i, "layer": int(layer[i]), "pref_x": float(pref_x[i]),
                          "pref_y": float(pref_y[i]), "pref_omega": float(pref_omega[i])}
                        for i in range(N)],
            "synapses": [{"pre": int(syn_pre[i]), "post": int(syn_post[i]), "w": float(syn_w[i])}
                         for i in range(S)],
        }

    def _build_cupy_buffers(self):
        N = self.config["neuron_count"]
        tc = self.topo
        xp = cp  # GPU

        # 稀疏权重矩阵 W: N x N, W[pre, post] = w
        self.W_sparse = xp.sparse.coo_matrix(
            (cp.asarray(tc["syn_w"], dtype=cp.float32),
             (cp.asarray(tc["syn_pre"], dtype=cp.int32),
              cp.asarray(tc["syn_post"], dtype=cp.int32))),
            shape=(N, N)
        ).tocsr()

        # 膜电位（所有神经元）
        self.membrane = cp.full(N, MEMBRANE_REST, dtype=cp.float32)
        # 突触电流累加器（每 tick 清零）
        self.synaptic_current = cp.zeros(N, dtype=cp.float32)
        # 上一次 tick 的 spiking（用于 R-STDP STDP kernel）
        self.is_spiking = cp.zeros(N, dtype=cp.int32)
        self.prev_spiking = cp.zeros(N, dtype=cp.int32)
        self.last_spike_tick = cp.full(N, -1, dtype=cp.int32)
        # 资格迹（每个突触）
        S = tc["synapse_count"]
        self.eligibility_trace = cp.zeros(S, dtype=cp.float32)
        # 偏好点
        self.pref_x = cp.asarray(tc["pref_x"], dtype=cp.float32)
        self.pref_y = cp.asarray(tc["pref_y"], dtype=cp.float32)
        self.pref_omega = cp.asarray(tc["pref_omega"], dtype=cp.float32)
        # 层类型 mask
        self.layer = cp.asarray(tc["layer"], dtype=cp.int32)

        # 物理状态（CPU numpy 管理，量小）
        self.arm_theta1 = 0.0
        self.arm_theta2 = 0.0
        self.arm_omega1 = 0.0
        self.arm_omega2 = 0.0
        self.arm_end_x = 1.0
        self.arm_end_y = 0.0
        self.target_x = 0.5
        self.target_y = 0.5
        self.max_reach = 2.0

    # ---- 编码（Step 0）：2D 高斯群体编码 ----
    def _encode_target(self, tx, ty):
        """INPUT_X 层的神经元响应（0..1）"""
        N_in = self.config["input"]
        resp = cp.zeros(self.config["neuron_count"], dtype=cp.float32)
        sigma = ENCODE_SIGMA_POS
        dx = self.pref_x[:N_in] - tx
        dy = self.pref_y[:N_in] - ty
        gauss = cp.exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
        resp[:N_in] = gauss
        return resp

    # ---- 解码（Step 4）：群体向量解码 omega1, omega2 ----
    def _decode_omega(self, layer_mask):
        """对 MOTOR 层做加权平均：omega = sum(pref_omega * spike_rate) / sum(spike_rate)"""
        xp = cp
        # 用最近 5 个 tick 的平均 firing rate（近似：当前 is_spiking + 之前累计）
        spk = self.is_spiking.astype(xp.float32)
        rates = spk  # 简化：用最近一个 tick
        # 只看该层
        mask = layer_mask.astype(xp.float32)
        numer = float(xp.sum(self.pref_omega * rates * mask))
        denom = float(xp.sum(rates * mask))
        if denom < 1e-6: return 0.0
        return numer / denom

    # ---- 单 tick 计算 ----
    def tick(self):
        xp = cp
        self.tick_count += 1

        # Step 0: 编码 → INPUT_X 层直接注入膜电位
        encode_resp = self._encode_target(self.target_x, self.target_y)
        # INPUT_X 直接设膜电位（与 C 端一致：MEMBRANE_REST + resp * GAIN）
        N_in = self.config["input"]
        input_target = MEMBRANE_REST + encode_resp[:N_in] * (MEMBRANE_THRESHOLD - MEMBRANE_REST + INPUT_GAIN)
        self.membrane[:N_in] = input_target

        # Step 1: 膜电位衰减 + 突触电流
        # synaptic_current = W^T @ is_spiking  (因为 W[pre,post] 表示 pre→post 的权重)
        self.prev_spiking[:] = self.is_spiking
        # spiking 向量：float，值为权重贡献
        spk_vec = self.is_spiking.astype(xp.float32)
        # SpMV: current[post] += sum(pre W[pre,post] * spk[pre])
        current = (self.W_sparse.T @ spk_vec) * PSP_FACTOR
        # 衰减 + 累加
        mask_non_input = cp.ones(self.config["neuron_count"], dtype=cp.float32)
        mask_non_input[:N_in] = 0.0  # INPUT_X 跳过（膜电位已由编码设置）
        self.membrane = cp.where(
            mask_non_input > 0.5,
            self.membrane * MEMBRANE_DECAY + current,
            self.membrane
        )

        # Step 2: spike 判定 + k-WTA（运动层）
        new_spiking = (self.membrane >= SPIKE_THRESHOLD).astype(xp.int32)
        # INPUT_X 不受不应期限制（与 C 端一致）
        # 这里简化：直接用阈值
        # 运动层 k-WTA 简化：如果超过阈值就发（量小，对称抵消在解码时处理）
        self.is_spiking[:] = new_spiking
        # 更新 last_spike_tick
        spk_ids = cp.nonzero(new_spiking)[0]
        self.last_spike_tick[spk_ids] = self.tick_count

        # Step 3: R-STDP 资格迹更新
        # eligibility_trace[i] += stdp_kernel2(pre, post)
        # 简化：用 pre/post spiking 门控
        tc = self.topo
        # 逐突触向量化
        S = tc["synapse_count"]
        # pre 和 post 是否 spiking
        pre_spk = self.is_spiking[cp.asarray(tc["syn_pre"], dtype=cp.int32)]
        post_spk = self.is_spiking[cp.asarray(tc["syn_post"], dtype=cp.int32)]
        post_last = self.last_spike_tick[cp.asarray(tc["syn_post"], dtype=cp.int32)]
        pre_last = self.last_spike_tick[cp.asarray(tc["syn_pre"], dtype=cp.int32)]
        delta_post = self.tick_count - post_last
        delta_pre = self.tick_count - pre_last
        # 简化 STDP kernel（与 C 端一致：exp(-dt/tau) * sign）
        tau = 50.0
        kernel = cp.where(
            (pre_spk > 0) & (post_last >= 0),
            cp.exp(-delta_post.astype(cp.float32) / tau),
            0.0
        ) - cp.where(
            (post_spk > 0) & (pre_last >= 0),
            cp.exp(-delta_pre.astype(cp.float32) / tau),
            0.0
        )
        # 衰减旧的 eligibility_trace（每 tick 乘 lambda）
        self.eligibility_trace *= 0.951
        self.eligibility_trace += kernel.astype(cp.float32)
        self.eligibility_trace = cp.clip(self.eligibility_trace, -5.0, 5.0)

        # Step 4: 解码 omega1, omega2
        m1_mask = (self.layer == LAYER_MOTOR1).astype(cp.float32)
        m2_mask = (self.layer == LAYER_MOTOR2).astype(cp.float32)
        omega1_cmd = self._decode_omega(m1_mask)
        omega2_cmd = self._decode_omega(m2_mask)
        omega1_cmd = max(-0.5, min(0.5, omega1_cmd))
        omega2_cmd = max(-0.5, min(0.5, omega2_cmd))

        # 物理模拟（CPU numpy）
        L1 = L2 = 1.0
        ARM_DT = 0.02
        self.arm_omega1 = omega1_cmd
        self.arm_omega2 = omega2_cmd
        self.arm_theta1 += omega1_cmd * ARM_DT
        self.arm_theta2 += omega2_cmd * ARM_DT
        # 正运动学
        e1x = math.cos(self.arm_theta1)
        e1y = math.sin(self.arm_theta1)
        self.arm_end_x = e1x + L2 * math.cos(self.arm_theta1 + self.arm_theta2)
        self.arm_end_y = e1y + L2 * math.sin(self.arm_theta1 + self.arm_theta2)

    def set_target(self, x, y):
        self.target_x = float(x)
        self.target_y = float(y)

    def get_snapshot(self):
        xp = cp
        mem_np = xp.asnumpy(self.membrane)
        spk_np = xp.asnumpy(self.is_spiking)
        last_np = xp.asnumpy(self.last_spike_tick)
        N = self.config["neuron_count"]
        neurons = [
            {"m": float(mem_np[i]), "s": int(spk_np[i]), "lst": int(last_np[i])}
            for i in range(N)
        ]
        return {
            "tick": self.tick_count,
            "neurons": neurons,
            "arm": {
                "t1": self.arm_theta1, "t2": self.arm_theta2,
                "o1": self.arm_omega1, "o2": self.arm_omega2,
                "ex": self.arm_end_x, "ey": self.arm_end_y,
                "tx": self.target_x, "ty": self.target_y,
                "r": max(0.0, 1.0 - math.hypot(self.arm_end_x - self.target_x,
                                              self.arm_end_y - self.target_y) / self.max_reach),
                "mr": self.max_reach,
            },
        }

    def get_topology(self):
        return self.topo
