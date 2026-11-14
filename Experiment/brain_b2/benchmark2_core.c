/*
 * benchmark2_core.c - Benchmark-2 核心实现
 * 两关节虚拟臂到达
 *
 * 架构：
 *   目标位置(x,y) → 2D 高斯群体编码(N个) → 隐藏层 → 运动层1(ω1,群体解码)
 *                                              → 运动层2(ω2,群体解码)
 *   ω1,ω2 → 两关节臂物理模拟 → 末端 → reward = 1 - dist/max_dist
 */

#include "benchmark2_core.h"

/* ============================================================
 * 工具
 * ============================================================ */
static unsigned int g_rand = 12345;
static inline double r_u(void) {
    g_rand = g_rand * 1103515245 + 12345;
    return (double)((g_rand >> 16) & 0x7FFF) / 32767.0;
}
static inline double r_gauss(double m, double s) {
    double u1 = r_u(); if (u1 < 1e-10) u1 = 1e-10;
    return m + s * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * r_u());
}
static inline double clamp2(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* 突触数组容量保障：任何写入前调用，容量不足时 realloc 扩容。
 * 返回 0 成功；-1 表示分配失败（调用者应清理并返回错误）。 */
#define SYN_CAP_HARD_LIMIT 20000000
static int ensure_synapse_capacity(NetworkState2* net, int needed) {
    if (needed < 0 || needed > SYN_CAP_HARD_LIMIT) return -1;
    if (needed <= net->synapse_capacity) return 0;
    int newcap = net->synapse_capacity > 0 ? net->synapse_capacity : 1024;
    while (newcap < needed) {
        if (newcap > SYN_CAP_HARD_LIMIT / 2) { newcap = needed; break; }
        newcap *= 2;
    }
    Synapse2* p = (Synapse2*)realloc(net->synapses, (size_t)newcap * sizeof(Synapse2));
    if (!p) return -1;
    net->synapses = p;
    net->synapse_capacity = newcap;
    return 0;
}

/* ============================================================
 * 两关节臂物理
 * ============================================================ */
/* EMA 平滑状态（全局，便于 episode 重置时清零） */
static double g_ema_o1 = 0.0, g_ema_o2 = 0.0;
void net2_reset_ema(void) { g_ema_o1 = 0.0; g_ema_o2 = 0.0; }

/* ============================================================
 * v6 异构群体编码：独立于主 RNG，避免打开 hetero_* 时
 * 消耗 g_rand 而污染 A/B 对照的随机序列。
 * ============================================================ */
static unsigned int g_het_rand = 20261111u;
static inline double het_u(void) {
    g_het_rand = g_het_rand * 1664525u + 1013904223u;
    return (double)((g_het_rand >> 8) & 0xFFFFFF) / 16777216.0;
}
static inline double het_gauss(void) {
    double u1 = het_u(), u2 = het_u();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* 只扰动运动层。三个 sigma 全为 0 时结果 = 旧版逐位一致。
 * 膜时间常数取对数正态、阈值取高斯、调谐宽度取对数正态，
 * 偏好角速度保持均匀覆盖并加微抖动。 */
static void apply_heterogeneity(NetworkState2* net) {
    if (!net || !net->neurons) return;
    HyperParams2* hp = &net->hp;
    g_het_rand = 20261111u;          /* 固定种子：每次调用得到同一组异质参数 */
    int m1_base = net->input_count + net->hidden_count;
    int m2_base = m1_base + net->motor1_count;
    for (int layer = 0; layer < 2; layer++) {
        int base = (layer == 0) ? m1_base : m2_base;
        int cnt  = (layer == 0) ? net->motor1_count : net->motor2_count;
        for (int i = 0; i < cnt; i++) {
            Neuron2* n = &net->neurons[base + i];
            double base_pref = -0.5 + 1.0 * (double)i / (double)(cnt - 1);
            double f = (hp->hetero_tau > 0.0) ? exp(het_gauss() * hp->hetero_tau) : 1.0;
            n->membrane_decay = clamp2(1.0 - (1.0 - MEMBRANE_DECAY) * f, 0.80, 0.995);
            n->spike_threshold = SPIKE_THRESHOLD
                + ((hp->hetero_thresh > 0.0) ? het_gauss() * hp->hetero_thresh : 0.0);
            n->tuning_width = (hp->hetero_tune > 0.0)
                ? clamp2(exp(het_gauss() * hp->hetero_tune), 0.4, 2.5) : 1.0;
            n->pref_omega = base_pref
                + ((hp->hetero_tune > 0.0) ? het_gauss() * 0.02 * hp->hetero_tune : 0.0);
        }
    }
}

void arm_step(ArmState* arm, double omega1_cmd, double omega2_cmd) {
    /* 直接用原始 omega，不加 EMA 平滑
     * 抖动是好事：让 arm 有机会朝各种方向试探，碰到正反馈就能启动学习 */
    double max_omega = 0.5;
    arm->omega1 = clamp2(omega1_cmd, -max_omega, max_omega);
    arm->omega2 = clamp2(omega2_cmd, -max_omega, max_omega);
    
    arm->theta1 += arm->omega1 * ARM_DT;
    arm->theta2 += arm->omega2 * ARM_DT;
    
    /* 肩关节可 360° 旋转：周期性回绕，不能硬裁剪。
     * 旧版把 θ1 钳在 ±π，一旦顶到边界就成了吸收态（ω 方向不变时永远
     * 出不来），末端自由度只剩 θ2，只能在以肘为圆心的单位圆上绕圈。 */
    if (arm->theta1 > M_PI)  arm->theta1 -= 2.0 * M_PI;
    if (arm->theta1 < -M_PI) arm->theta1 += 2.0 * M_PI;
    /* 肘关节保留生理限位 */
    if (arm->theta2 < ARM_THETA2_MIN) arm->theta2 = ARM_THETA2_MIN;
    if (arm->theta2 > ARM_THETA2_MAX) arm->theta2 = ARM_THETA2_MAX;
    
    /* 正运动学 */
    arm->end_x = ARM_L1 * cos(arm->theta1) + ARM_L2 * cos(arm->theta1 + arm->theta2);
    arm->end_y = ARM_L1 * sin(arm->theta1) + ARM_L2 * sin(arm->theta1 + arm->theta2);
    
    arm->max_reach = ARM_L1 + ARM_L2;
    arm->reward = arm_reward(arm);
}

void arm_set_target(ArmState* arm, double tx, double ty) {
    arm->target_x = tx;
    arm->target_y = ty;
}

double arm_reward(const ArmState* arm) {
    double dx = arm->end_x - arm->target_x;
    double dy = arm->end_y - arm->target_y;
    double dist = sqrt(dx*dx + dy*dy);
    /* 用 (1 - dist/max_reach) 作为绝对 reward，同时改进量后面处理 */
    double normalized = dist / arm->max_reach;
    return 1.0 - clamp2(normalized, 0.0, 1.0);
}

/* ============================================================
 * 网络初始化
 * ============================================================ */
int net2_init(NetworkState2* net, int input_n, int hidden_n,
              int motor1_n, int motor2_n, double avg_conn,
              unsigned int seed) {
    /* 参数下界：避免 grid_side-1 / motor_n-1 的除零，以及空网络 */
    if (!net || input_n < 2 || hidden_n < 1 || motor1_n < 2 || motor2_n < 2
        || avg_conn < 1.0) return -1;
    memset(net, 0, sizeof(NetworkState2));
    g_rand = seed;
    
    net->input_count = input_n + PROP_N;   /* 位置编码 + 本体感觉 */
    net->hidden_count = hidden_n;
    net->motor1_count = motor1_n;
    net->motor2_count = motor2_n;
    net->neuron_count = net->input_count + hidden_n + motor1_n + motor2_n;
    
    net->neurons = (Neuron2*)calloc(net->neuron_count, sizeof(Neuron2));
    if (!net->neurons) return -1;
    
    /* 1. 位置编码输入层：偏好点均匀分布在工作空间 */
    int idx = 0, i;
    /* v6：所有神经元先写入统一的膜动力学默认值（calloc 给的是 0，
     * 直接拿来当衰减系数会让膜电位每 tick 归零，必须显式初始化）。*/
    for (i = 0; i < net->neuron_count; i++) {
        net->neurons[i].membrane_decay = MEMBRANE_DECAY;
        net->neurons[i].tuning_width = 1.0;
    }
    for (i = 0; i < input_n; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_INPUT_X;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
        
        /* 2D 网格 + 抖动，覆盖相对位置范围 [-4, 4]²
         * 之前是可达空间 [-2, 2]，改成 [-4, 4] 以覆盖 target - arm_end 的相对位置 */
        int grid_side = (int)ceil(sqrt((double)input_n));
        int gx = i % grid_side;
        int gy = i / grid_side;
        double range = ARM_L1 + ARM_L2 + ARM_L1 + ARM_L2;  /* 相对位置范围 = target - arm ∈ [-4, 4] */
        double step = 2.0 * range / (grid_side - 1);     /* 8.0 / 9 ≈ 0.89 for 100 神经元 */
        net->neurons[idx].pref_x = -range + gx * step + r_u() * 0.1;
        net->neurons[idx].pref_y = -range + gy * step + r_u() * 0.1;
        /* 不再限制到可达空间——相对位置本来就在可达空间之外 */
    }
    
    /* 1b. 本体感觉+方位特征输入层：
     *   tag0 (12个): θ1 关节角；tag1 (12个): θ2 关节角 —— 修复状态混叠
     *   tag2 (16个): ψ1 = bearing(target) - θ1 —— 上臂相对目标方位差 → 直接驱动 ω1
     *   tag3 (16个): ψ2 = bearing(target-elbow) - (θ1+θ2) —— 前臂方位差 → 直接驱动 ω2
     * 没有方位特征时 (dx,dy,θ)→ω 是乘性交互，加性随机特征表达不了（2000ep 只降到 0.77）。
     * pref_x = tag，pref_omega = 偏好角度 ∈ [-π, π] */
    for (i = 0; i < PROP_N; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_INPUT_PROP;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
        int tag, j, gn;
        if      (i < 12) { tag = 0; j = i;      gn = 12; }
        else if (i < 24) { tag = 1; j = i - 12; gn = 12; }
        else if (i < 40) { tag = 2; j = i - 24; gn = 16; }
        else             { tag = 3; j = i - 40; gn = 16; }
        net->neurons[idx].pref_x = (double)tag;
        net->neurons[idx].pref_y = 0.0;
        net->neurons[idx].pref_omega = -M_PI + 2.0 * M_PI * (double)j / (double)(gn - 1);
    }
    
    /* 2. 隐藏层 */
    for (i = 0; i < hidden_n; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_HIDDEN;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
    }
    
    /* 3. 关节1运动层：偏好角速度 -1~1 均匀分布 */
    for (i = 0; i < motor1_n; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_MOTOR_OMEGA1;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
        /* motor1: pref_omega ∈ [-0.5, 0.5] 均匀分布 → 和 max_omega 匹配，解码更精确 */
        net->neurons[idx].pref_omega = -0.5 + 1.0 * (double)i / (double)(motor1_n - 1);
    }
    
    /* 4. 关节2运动层 */
    for (i = 0; i < motor2_n; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_MOTOR_OMEGA2;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
        net->neurons[idx].pref_omega = -0.5 + 1.0 * (double)i / (double)(motor2_n - 1);
    }
    
    /* 突触容量：按 6 个连接层循环次数求安全上界（去重只会更少），
     * 并保留 ensure_synapse_capacity 作为兜底，任何估算偏差都不会越界。 */
    int K = (int)avg_conn;
    if (K < 3) K = 3;
    long long upper = (long long)net->input_count * 20
                    + (long long)hidden_n * (K / 2)
                    + (long long)hidden_n * K * 2
                    + (long long)net->input_count * (motor1_n < 12 ? motor1_n : 12)
                    + (long long)net->input_count * (motor2_n < 12 ? motor2_n : 12);
    if (upper < 1 || upper > SYN_CAP_HARD_LIMIT) { free(net->neurons); return -1; }
    net->synapse_capacity = (int)upper + 64;
    net->synapses = (Synapse2*)calloc((size_t)net->synapse_capacity, sizeof(Synapse2));
    if (!net->synapses) { free(net->neurons); return -1; }
    
    net->synapse_count = 0;
    int s_idx;
    
    /* === 连接1: INPUT_X → HIDDEN（更密！让 target 信息充分传播） === */
    for (i = 0; i < net->input_count; i++) {
        int pre = i;
        int hidden_base = net->input_count;
        for (int k = 0; k < 20; k++) {  /* K=20（原 4），更密连接 */
            int post = hidden_base + (int)(r_u() * hidden_n);
            if (post >= hidden_base + hidden_n) continue;
            
            int dup = 0;
            for (int s = 0; s < net->synapse_count; s++) {
                if (net->synapses[s].pre_id == pre && net->synapses[s].post_id == post) { dup = 1; break; }
            }
            if (dup) continue;
            
            if (ensure_synapse_capacity(net, net->synapse_count + 1) != 0) {
                net2_destroy(net);
                return -1;
            }
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 1.0 + r_u() * 2.0;  /* 1.0-3.0, mean 2.0 — handoff PASS 配置 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接2: HIDDEN → HIDDEN（稀疏递归） === */
    for (i = 0; i < hidden_n; i++) {
        int pre = net->input_count + i;
        for (int k = 0; k < K / 2; k++) {
            int post = net->input_count + (int)(r_u() * hidden_n);
            if (post == pre) continue;
            
            int dup = 0;
            for (int s = 0; s < net->synapse_count; s++) {
                if (net->synapses[s].pre_id == pre && net->synapses[s].post_id == post) { dup = 1; break; }
            }
            if (dup) continue;
            
            if (ensure_synapse_capacity(net, net->synapse_count + 1) != 0) {
                net2_destroy(net);
                return -1;
            }
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.1 + r_u() * 0.3;
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接3: HIDDEN → MOTOR1（减弱！让 I→M 主导初期） === */
    int motor1_base = net->input_count + hidden_n;
    for (i = 0; i < hidden_n; i++) {
        int pre = net->input_count + i;
        for (int k = 0; k < K; k++) {
            int post = motor1_base + (int)(r_u() * motor1_n);
            if (post >= motor1_base + motor1_n) continue;
            
            int dup = 0;
            for (int s = 0; s < net->synapse_count; s++) {
                if (net->synapses[s].pre_id == pre && net->synapses[s].post_id == post) { dup = 1; break; }
            }
            if (dup) continue;
            
            if (ensure_synapse_capacity(net, net->synapse_count + 1) != 0) {
                net2_destroy(net);
                return -1;
            }
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.3 + r_u() * 0.4;  /* 恢复 0.3-0.7 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接4: HIDDEN → MOTOR2 === */
    int motor2_base = motor1_base + motor1_n;
    for (i = 0; i < hidden_n; i++) {
        int pre = net->input_count + i;
        for (int k = 0; k < K; k++) {
            int post = motor2_base + (int)(r_u() * motor2_n);
            if (post >= motor2_base + motor2_n) continue;
            
            int dup = 0;
            for (int s = 0; s < net->synapse_count; s++) {
                if (net->synapses[s].pre_id == pre && net->synapses[s].post_id == post) { dup = 1; break; }
            }
            if (dup) continue;
            
            if (ensure_synapse_capacity(net, net->synapse_count + 1) != 0) {
                net2_destroy(net);
                return -1;
            }
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.3 + r_u() * 0.4;  /* 恢复 0.3-0.7 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接5: INPUT → MOTOR1（直接感知相对位置+本体感觉！绕过 hidden 层！）===
     * 每个 INPUT 神经元连 motor1 层 K=4 个神经元，初始权重 1.5-2.5 */
    for (i = 0; i < net->input_count; i++) {
        int pre = i;
        for (int k = 0; k < 12; k++) {
            int post = motor1_base + (int)(r_u() * motor1_n);
            if (post >= motor1_base + motor1_n) continue;
            int dup = 0;
            for (int s = 0; s < net->synapse_count; s++) {
                if (net->synapses[s].pre_id == pre && net->synapses[s].post_id == post) { dup = 1; break; }
            }
            if (dup) continue;
            if (ensure_synapse_capacity(net, net->synapse_count + 1) != 0) {
                net2_destroy(net);
                return -1;
            }
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 1.5 + r_u() * 1.0;  /* 1.5-2.5, mean 2.0 — 恢复 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接6: INPUT → MOTOR2（直接感知相对位置+本体感觉！）=== */
    for (i = 0; i < net->input_count; i++) {
        int pre = i;
        for (int k = 0; k < 12; k++) {
            int post = motor2_base + (int)(r_u() * motor2_n);
            if (post >= motor2_base + motor2_n) continue;
            int dup = 0;
            for (int s = 0; s < net->synapse_count; s++) {
                if (net->synapses[s].pre_id == pre && net->synapses[s].post_id == post) { dup = 1; break; }
            }
            if (dup) continue;
            if (ensure_synapse_capacity(net, net->synapse_count + 1) != 0) {
                net2_destroy(net);
                return -1;
            }
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 1.5 + r_u() * 1.0;  /* 1.5-2.5, mean 2.0 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* 超参数：防止 trace 瞬间饱和 */
    net->hp.tau_stdp = 50.0;
    net->hp.tau_trace = 2000.0;          /* 2000 最佳 */
    net->hp.lambda_trace = exp(-1.0 / net->hp.tau_trace);
    net->hp.stdp_a_plus = 0.05;         /* 旧版最佳 0.05：配合 trace clamp [-1,1]，trace 快速饱和到 1 */
    net->hp.stdp_a_minus = 0.06;        /* 保持 1.2:1 比例 */
    net->hp.learning_rate = 0.15;       /* 恢复 0.15 */
    net->hp.weight_decay_lambda = 1e-5;
    net->hp.reward_baseline_window = 200;  /* 200 最佳，500 太慢 */
    /* 运行时可调参数：默认值 = 旧版硬编码值 */
    net->hp.psp_factor = 25.0;
    net->hp.trace_clamp = 1.0;
    net->hp.decode_mode = 0;
    net->hp.wta_k = 3;
    net->hp.use_reward_baseline = 1;
    /* v6 三因子通道 / 异构群体编码：默认全关，旧实验逐位可复现 */
    net->hp.error_gain     = 0.0;
    net->hp.novelty_gain   = 0.0;
    net->hp.cf_lambda      = 1.0;
    net->hp.hetero_tau     = 0.0;
    net->hp.hetero_thresh  = 0.0;
    net->hp.hetero_tune    = 0.0;
    net->hp.prop_mode      = 0;   /* full：与历史实验逐位一致 */

    #if ELIGIBILITY_TRACE_ENABLE
    net->use_eligibility_trace = 1;
    #else
    net->use_eligibility_trace = 0;
    #endif
    #if REWARD_MODULATION_ENABLE
    net->use_reward_modulation = 1;
    #else
    net->use_reward_modulation = 0;
    #endif
    
    net->reward_baseline = 0.0;
    net->reward_history = (double*)calloc(net->hp.reward_baseline_window, sizeof(double));
    if (!net->reward_history) { net2_destroy(net); return -1; }
    
    /* 关节臂初始状态 */
    net->arm.theta1 = 0.0;
    net->arm.theta2 = 0.0;
    net->arm.end_x = ARM_L1 + ARM_L2;
    net->arm.end_y = 0.0;
    net->arm.max_reach = ARM_L1 + ARM_L2;
    arm_set_target(&net->arm, net->arm.end_x, net->arm.end_y);
    net->arm.reward = 1.0;
    
    net->queue_head = 0;
    net->queue_tail = 0;
    net->current_tick = 0;
    
    return 0;
}

void net2_destroy(NetworkState2* net) {
    if (!net) return;
    if (net->neurons) free(net->neurons);
    if (net->synapses) free(net->synapses);
    if (net->reward_history) free(net->reward_history);
    memset(net, 0, sizeof(NetworkState2));
}

int net2_neuron_count(const NetworkState2* net) {
    return net ? net->neuron_count : 0;
}

int net2_synapse_count(const NetworkState2* net) {
    return net ? net->synapse_count : 0;
}

/* ============================================================
 * 目标设置 → 群体编码激活
 * ============================================================ */
void net2_set_target(NetworkState2* net, double tx, double ty) {
    arm_set_target(&net->arm, tx, ty);
    
    /* 用高斯核直接激活输入层神经元——**编码 target - arm_end 的相对位置**！
     * SNN 必须知道"我离 target 还有多远、什么方向"才能输出正确 omega！
     * 之前编码绝对 target 位置 → SNN 输出固定值 → 永远不能精确到达！*/
    int i;
    double sigma2 = ENCODE_SIGMA_POS * ENCODE_SIGMA_POS;
    double cur_end_x = net->arm.end_x;
    double cur_end_y = net->arm.end_y;
    for (i = 0; i < net->input_count; i++) {
        Neuron2* n = &net->neurons[i];
        double rel_tx = tx - cur_end_x;   /* 相对 target 位置 */
        double rel_ty = ty - cur_end_y;
        double dx = n->pref_x - rel_tx;
        double dy = n->pref_y - rel_ty;
        double dist2 = dx*dx + dy*dy;
        double response = exp(-dist2 / (2.0 * sigma2));
        n->membrane_potential = MEMBRANE_REST + response * (SPIKE_THRESHOLD - MEMBRANE_REST + 3.0);
        if (r_u() < response * 0.3) {
            n->membrane_potential = SPIKE_THRESHOLD + 5.0;
        }
    }
}

/* ============================================================
 * STDP 核
 * ============================================================ */
static double stdp_kernel2(NetworkState2* net, double dt) {
    if (dt > 0)  return net->hp.stdp_a_plus * exp(-dt / net->hp.tau_stdp);
    if (dt < 0)  return -net->hp.stdp_a_minus * exp(dt / net->hp.tau_stdp);
    return 0.0;
}

/* ============================================================
 * Tick 核心
 * ============================================================ */
void net2_tick(NetworkState2* net) {
    /* ═══════════════════════════════════════════════════════
     * REWARD QUEUE：tick **开头** 处理上一 tick host 注入的 reward
     * 关键时序对齐：
     *   tick N 的 spike → omega → arm_step → dist 变化 → host 注入 reward N
     *   reward N 在下一 tick (N+1) 的**开头**被处理
     *   此时 trace 已包含 tick N 的 spike（tick N 的 Step 3 做过了）
     *   → reward N credit 给 tick N 的 spike ✓！
     * 之前放在 Step 6（结尾）导致 reward N credit 给了 tick N+1 的 spike（差 1 tick！）
     * ═══════════════════════════════════════════════════════ */
    while (net->queue_head != net->queue_tail &&
           net->reward_queue[net->queue_head].delivery_tick <= net->current_tick) {
        RewardEvent2 ev = net->reward_queue[net->queue_head];
        net->queue_head = (net->queue_head + 1) % QUEUE_SIZE;
        
        /* 默认减滑动基线；新模式可关闭，让缓慢变化的势能奖励保留长期梯度 */
        double base = net->hp.use_reward_baseline ? net->reward_baseline : 0.0;
        double err = ev.reward_scalar - base;
        double err1 = ev.r1 - base;
        double err2 = ev.r2 - base;

        /* 第三因子之「惊讶/新颖性」通道（类乙酰胆碱）：奖励偏离预期越多，
         * 本步学习率越高。只做增益，不碰 STDP 核与资格迹。 */
        double nov = 1.0 + net->hp.novelty_gain * clamp2(ev.novelty, 0.0, 2.0);
        int has_reward = fabs(err) > 0.001;
        int has_error  = (net->hp.error_gain > 0.0) &&
                         (fabs(ev.e1) > 1e-6 || fabs(ev.e2) > 1e-6);

        #if REWARD_MODULATION_ENABLE
        if (net->use_reward_modulation && (has_reward || has_error)) {
            for (int s = 0; s < net->synapse_count; s++) {
                Synapse2* syn = &net->synapses[s];
                #if ELIGIBILITY_TRACE_ENABLE
                double credit = net->use_eligibility_trace ? syn->eligibility_trace : 0.0;
                #else
                double credit = 0.0;
                #endif
                if (fabs(credit) <= 1e-8) continue;

                Neuron2* post = &net->neurons[syn->post_id];
                NeuronType2 pt = post->type;
                double e = (pt == NT_MOTOR_OMEGA1) ? err1
                         : (pt == NT_MOTOR_OMEGA2) ? err2 : err;

                /* 奖励通道（类多巴胺）：原 R-STDP，nov=1 时与旧版逐位一致 */
                double dw = net->hp.learning_rate * nov * e * credit;

                /* 运动误差通道（类小脑攀爬纤维 → 浦肯野细胞定向 LTD）：
                 * e_j = omega*_j - omega_j 是逐 tick、连续、有符号的关节级
                 * 感觉预测误差；dir = tanh(pref/0.15) 是突触后神经元编码的
                 * 输出方向。同号 → 增强正确方向(LTP)，异号 → 抑制导致错误
                 * 方向的突触(LTD)。只作用于运动层输入突触，不改变隐藏层。 */
                if (has_error && (pt == NT_MOTOR_OMEGA1 || pt == NT_MOTOR_OMEGA2)) {
                    /* 攀爬纤维 × 平行纤维同步条件：只有突触前神经元在本步
                     * 发放过（平行纤维活跃）的输入突触才接受误差信号，
                     * 对应小脑 PF–PF/CF 的联合 LTD 条件。*/
                    Neuron2* pre = &net->neurons[syn->pre_id];
                    if (pre->is_spiking) {
                        double ej = (pt == NT_MOTOR_OMEGA1) ? ev.e1 : ev.e2;
                        double dir = tanh(post->pref_omega / 0.15);
                        dw += net->hp.learning_rate * net->hp.error_gain * ej * dir * credit;
                    }
                }
                syn->weight = clamp2(syn->weight + dw, WEIGHT_MIN, WEIGHT_MAX);
            }
        }
        #endif
        
        /* 更新基线（sliding window=200）—— v2 已经证明这是正确配置
         * EMA alpha=0.005 太慢导致 baseline 和 reward 脱节 → 权重不涨反跌 */
        net->reward_history[net->reward_hist_head] = ev.reward_scalar;
        net->reward_hist_head = (net->reward_hist_head + 1) % net->hp.reward_baseline_window;
        if (net->reward_hist_count < net->hp.reward_baseline_window) net->reward_hist_count++;
        
        double sum = 0.0;
        for (int k = 0; k < net->reward_hist_count; k++) sum += net->reward_history[k];
        net->reward_baseline = sum / net->reward_hist_count;
    }

    net->current_tick++;
    int s, i;
    
    /* Step 0: 重新设置输入层膜电位（**持续编码 target - arm_end 相对位置 + 关节角度**！）
     * SNN 每 tick 感知到 "我离 target 还有多远 + 臂形是什么"，才能输出正确 omega！*/
    {
        double tx = net->arm.target_x;
        double ty = net->arm.target_y;
        double sigma2 = ENCODE_SIGMA_POS * ENCODE_SIGMA_POS;
        double sig_p2 = ENCODE_SIGMA_PROP * ENCODE_SIGMA_PROP;
        double cur_end_x = net->arm.end_x;
        double cur_end_y = net->arm.end_y;
        double rel_tx = tx - cur_end_x;
        double rel_ty = ty - cur_end_y;
        for (i = 0; i < net->input_count; i++) {
            Neuron2* n = &net->neurons[i];
            double response;
            if (n->type == NT_INPUT_PROP) {
                /* 本体感觉/方位特征：偏好角度 vs 当前特征角（环形距离） */
                int tag = (int)(n->pref_x + 0.5);
                /* v7 prop_mode=1（raw）：tag2/tag3 是由解析几何算出的目标方位差，
                 * g=0 严格口径下不给，保持静息，只剩 θ1/θ2 纯本体感觉。 */
                if (net->hp.prop_mode == 1 && tag >= 2) {
                    n->membrane_potential = MEMBRANE_REST;
                    continue;
                }
                double th;
                if (tag == 0) {
                    th = net->arm.theta1;
                } else if (tag == 1) {
                    th = net->arm.theta2;
                } else if (tag == 2) {
                    /* ψ1 = 目标相对肩的方位 - θ1 */
                    th = atan2(ty, tx) - net->arm.theta1;
                } else {
                    /* ψ2 = 目标相对肘的方位 - (θ1+θ2) */
                    double ex = ARM_L1 * cos(net->arm.theta1);
                    double ey = ARM_L1 * sin(net->arm.theta1);
                    th = atan2(ty - ey, tx - ex) - (net->arm.theta1 + net->arm.theta2);
                }
                double d = th - n->pref_omega;
                while (d >  M_PI) d -= 2.0 * M_PI;
                while (d < -M_PI) d += 2.0 * M_PI;
                response = exp(-d * d / (2.0 * sig_p2));
            } else {
                double dx = n->pref_x - rel_tx;
                double dy = n->pref_y - rel_ty;
                double dist2 = dx*dx + dy*dy;
                response = exp(-dist2 / (2.0 * sigma2));
            }
            n->membrane_potential = MEMBRANE_REST + response * (SPIKE_THRESHOLD - MEMBRANE_REST + 5.0);
        }
    }
    
    /* Step 1: 突触电流累加（用上一tick的发放） */
    for (s = 0; s < net->synapse_count; s++) {
        Neuron2* pre = &net->neurons[net->synapses[s].pre_id];
        Neuron2* post = &net->neurons[net->synapses[s].post_id];
        if (pre->is_spiking) {
            post->membrane_potential += net->synapses[s].weight * net->hp.psp_factor;
        }
    }
    
    /* Step 2: 清零 is_spiking → 膜电位衰减 → 发放判定 → motor 层 k-WTA */
    for (i = 0; i < net->neuron_count; i++) {
        net->neurons[i].is_spiking = 0;
        
        Neuron2* n = &net->neurons[i];
        /* 输入层（位置+本体感觉）：不衰减（持续刺激），但必须做发放判定！ */
        if (n->type != NT_INPUT_X && n->type != NT_INPUT_PROP) {
            /* 不应期 */
            if (n->refractory_remaining > 0) {
                n->refractory_remaining--;
                n->membrane_potential = MEMBRANE_REST;
                continue;
            }
            /* 膜电位衰减（v6：每神经元独立时间常数，默认 = MEMBRANE_DECAY） */
            double md = n->membrane_decay;
            n->membrane_potential = n->membrane_potential * md + MEMBRANE_REST * (1.0 - md);
            
            /* base current + 膜电位噪声：
             * 噪声让 WTA winner 每次都略有不同 → 探索不同方向 → 碰到正反馈就能增强权重！
             * 噪声 σ=1.0mV 不大：不会淹没突触电流信号，但足够让 winner 抖动 */
            if (n->type == NT_HIDDEN) {
                n->membrane_potential += 2.0;
                n->membrane_potential += r_gauss(0.0, 0.5);   /* 高斯噪声 σ=0.5（降噪提速） */
            } else if (n->type == NT_MOTOR_OMEGA1 || n->type == NT_MOTOR_OMEGA2) {
                n->membrane_potential += 3.0;
                n->membrane_potential += r_gauss(0.0, 0.5);   /* σ=0.5：WTA 更稳，reward 信号更干净 */
            }
        }
        
        /* 发放判定（先标记，后统一处理 WTA 和膜电位重置） */
        if (n->membrane_potential >= n->spike_threshold) {
            n->is_spiking = 1;
        }
    }
    
    /* MOTOR 层 k-winner-take-all：每关节只让 K=3 个运动神经元 spike
     * K 小 → 方向更明确（正负池不会数量相当抵消），学习更快
     * 正确算法：遍历找膜电位最高的 K 个（不断替换最小的 winner）*/
    {
        enum { WTA_MAX = 64 };
        int m1_base = net->input_count + net->hidden_count;
        int m2_base = m1_base + net->motor1_count;
        int WTA_K = net->hp.wta_k;  /* 运行时可配，默认 3 */
        if (WTA_K < 1) WTA_K = 1;
        if (WTA_K > WTA_MAX) WTA_K = WTA_MAX;
        
        for (int m_layer = 0; m_layer < 2; m_layer++) {
            int base = (m_layer == 0) ? m1_base : m2_base;
            int count = (m_layer == 0) ? net->motor1_count : net->motor2_count;

            /* v7 decode_mode=3（PDc 群体解码）：不做 k-WTA 截断，
             * 保留本 tick 所有越阈的运动神经元；Step 4 的 else 分支按全部发放
             * 神经元对 pref_omega 取平均，即 Σ s_i·pref_i / Σ s_i。 */
            if (net->hp.decode_mode == 3) continue;
            
            /* 正确的 k-winner 算法：
             * 维护 K 个 winner，每次找到最小的 winner，若当前神经元更大则替换 */
            int winners[WTA_MAX];
            for (int w = 0; w < WTA_K; w++) winners[w] = -1;
            
            for (int j = 0; j < count; j++) {
                /* 找 winners 中膜电位最小的那个（包括空位置 -1）*/
                int min_w = 0;
                for (int w = 1; w < WTA_K; w++) {
                    if (winners[w] == -1) { min_w = w; break; }
                    if (winners[min_w] == -1) continue;
                    if (net->neurons[base + winners[w]].membrane_potential < 
                        net->neurons[base + winners[min_w]].membrane_potential) min_w = w;
                }
                
                double cur_v = net->neurons[base + j].membrane_potential;
                /* 如果 min_w 是空的，或者当前神经元比最小的 winner 更大 → 替换 */
                if (winners[min_w] == -1 || cur_v > net->neurons[base + winners[min_w]].membrane_potential) {
                    winners[min_w] = j;
                }
            }
            
            /* 先清零所有，再标记 winner */
            for (int j = 0; j < count; j++) net->neurons[base + j].is_spiking = 0;
            for (int w = 0; w < WTA_K; w++) {
                if (winners[w] >= 0) {
                    net->neurons[base + winners[w]].is_spiking = 1;
                }
            }
            
            /* 加权解码：膜电位尚未被 winner 重置，按相对阈值强度加权 pref_omega */
            if (net->hp.decode_mode == 2) {
                double wsum = 0.0, wtot = 0.0;
                for (int w = 0; w < WTA_K; w++) {
                    if (winners[w] < 0) continue;
                    Neuron2* wn = &net->neurons[base + winners[w]];
                    double v = (wn->membrane_potential - MEMBRANE_REST) * wn->tuning_width;
                    if (v < 1e-6) v = 1e-6;
                    wsum += v * wn->pref_omega;
                    wtot += v;
                }
                double cmd = (wtot > 0.0) ? wsum / wtot : 0.0;
                if (m_layer == 0) net->decoded_omega1 = cmd;
                else              net->decoded_omega2 = cmd;
            }
        }
    }
    
    /* 最后统一处理：设置 last_spike_tick + 不应期 + 重置膜电位 */
    for (i = 0; i < net->neuron_count; i++) {
        if (!net->neurons[i].is_spiking) continue;
        Neuron2* n = &net->neurons[i];
        n->last_spike_tick = net->current_tick;
        if (n->type != NT_INPUT_X && n->type != NT_INPUT_PROP) {
            n->refractory_remaining = REFRACTORY_TICKS;
        }
        n->membrane_potential = MEMBRANE_REST;
    }
    
    /* Step 3: 资格迹更新（事件驱动） */
    for (s = 0; s < net->synapse_count; s++) {
        Neuron2* pre = &net->neurons[net->synapses[s].pre_id];
        Neuron2* post = &net->neurons[net->synapses[s].post_id];
        
        if (!pre->is_spiking && !post->is_spiking) continue;
        
        int delta_tick = net->current_tick - net->synapses[s].last_stdp_tick;
        if (delta_tick > 1) {
            net->synapses[s].eligibility_trace *= pow(net->hp.lambda_trace, (double)delta_tick - 1.0);
        }
        net->synapses[s].last_stdp_tick = net->current_tick;
        
        #if ELIGIBILITY_TRACE_ENABLE
        if (net->use_eligibility_trace) {
            double stdp_val = 0.0;
            /* STDP：pre→post 时间差 → LTP，post→pre 时间差 → LTD
             * 关键修复：I→M 突触跳过分支1（pre-then-post LTD）！
             * 原因：INPUT 每 tick 都 spike（持续编码），MOTOR 也常 spike
             * 两个分支同时触发，a_minus(0.06)>a_plus(0.05) → net LTD
             * → R-STDP 的 err*trace 被反转！学习停滞！
             * 对于 I→M 突触，只保留分支2（post spike, pre 已 spike → LTP）
             * H→H 和 H→M 突触保持完整 STDP */
            int is_I_to_M = (pre->type == NT_INPUT_X || pre->type == NT_INPUT_PROP) &&
                            (post->type == NT_MOTOR_OMEGA1 || post->type == NT_MOTOR_OMEGA2);
            
            if (!is_I_to_M && pre->is_spiking && post->last_spike_tick > 0) {
                /* 分支1: pre 在 post 后 spike → LTD（只对非 I→M 生效） */
                stdp_val += stdp_kernel2(net, -(double)(net->current_tick - post->last_spike_tick));
            }
            if (post->is_spiking && pre->last_spike_tick > 0) {
                /* 分支2: pre 在 post 前 spike → LTP（对所有突触生效） */
                stdp_val += stdp_kernel2(net, (double)(net->current_tick - pre->last_spike_tick));
            }
            net->synapses[s].eligibility_trace += stdp_val;
            /* 截断幅度运行时可配：默认 1.0（旧版行为），<=0 表示不截断 */
            double tc = net->hp.trace_clamp;
            if (tc > 0.0) {
                if (net->synapses[s].eligibility_trace >  tc) net->synapses[s].eligibility_trace =  tc;
                if (net->synapses[s].eligibility_trace < -tc) net->synapses[s].eligibility_trace = -tc;
            }
        } else {
            net->synapses[s].eligibility_trace = 0.0;
        }
        #else
        net->synapses[s].eligibility_trace = 0.0;
        #endif
    }
    
    /* Step 4: SNN 解码 → 直接平均所有 spike winner 的 pref_omega
     * STDP 是 spike-based，解码也用 spike-based 方式，信号匹配 */
    double omega1_cmd = 0.0, omega2_cmd = 0.0;
    if (net->hp.decode_mode == 2) {
        /* 加权解码结果在 WTA 阶段算好（膜电位那时还未被重置） */
        omega1_cmd = net->decoded_omega1;
        omega2_cmd = net->decoded_omega2;
    } else {
        int m1_base = net->input_count + net->hidden_count;
        int m2_base = m1_base + net->motor1_count;
        
        for (int j = 0; j < 2; j++) {
            int m_base = (j == 0) ? m1_base : m2_base;
            int m_count = (j == 0) ? net->motor1_count : net->motor2_count;
            
            int w_count = 0; double w_sum = 0.0;
            for (int i = 0; i < m_count; i++) {
                Neuron2* n = &net->neurons[m_base + i];
                if (!n->is_spiking) continue;
                w_sum += n->pref_omega;
                w_count++;
            }
            double cmd = (w_count > 0) ? w_sum / (double)w_count : 0.0;
            
            if (j == 0) omega1_cmd = cmd; else omega2_cmd = cmd;
        }
    }
    
    /* 解码后加小探索噪声 σ=0.02：最优值 */
    omega1_cmd += r_gauss(0.0, 0.02);
    omega2_cmd += r_gauss(0.0, 0.02);
    omega1_cmd = fmax(-0.5, fmin(0.5, omega1_cmd));
    omega2_cmd = fmax(-0.5, fmin(0.5, omega2_cmd));
    
    net->decoded_omega1 = omega1_cmd;
    net->decoded_omega2 = omega2_cmd;
    
    /* Step 5: 物理模拟一步 */
    arm_step(&net->arm, omega1_cmd, omega2_cmd);

    /* Step 5b: 攀爬纤维误差信号 —— 感觉目标误差 → 关节级运动指令误差
     * 生物学对应：小脑攀爬纤维在运动进行中（而非试次结束）传递感觉预测
     * 误差，诱导平行纤维–浦肯野细胞突触的定向 LTD，校准内部逆模型。
     * 工程实现：用 Jacobian 转置（最速下降方向，天然避开求逆奇异）把末端
     * 指向目标的单位向量映射成期望关节角速度 omega*，与实际解码 omega 的
     * 差即为逐 tick、连续、有符号、方向性的误差。密度远高于稀疏到达奖励。
     * 结果由 net2_inject_reward2 在本 tick 末随奖励一起入队，下一 tick 开头生效。*/
    {
        double t1 = net->arm.theta1, t2 = net->arm.theta2;
        double s1 = sin(t1), c1 = cos(t1);
        double s12 = sin(t1 + t2), c12 = cos(t1 + t2);
        double J11 = -ARM_L1 * s1 - ARM_L2 * s12;
        double J21 =  ARM_L1 * c1 + ARM_L2 * c12;
        double J12 = -ARM_L2 * s12;
        double J22 =  ARM_L2 * c12;
        double dx = net->arm.target_x - net->arm.end_x;
        double dy = net->arm.target_y - net->arm.end_y;
        double d  = sqrt(dx * dx + dy * dy);
        double ux = 0.0, uy = 0.0;
        if (d > 1e-6) { ux = dx / d; uy = dy / d; }
        double g1 = J11 * ux + J21 * uy;   /* J^T · u */
        double g2 = J12 * ux + J22 * uy;
        /* 接近目标时把期望速度收敛到 0，避免到达点附近持续抖动 */
        double prox = (d < 0.15) ? d / 0.15 : 1.0;
        double lam = (net->hp.cf_lambda > 0.0) ? net->hp.cf_lambda : 1.0;
        double w1s = clamp2(0.5 * g1 / lam, -0.5, 0.5) * prox;
        double w2s = clamp2(0.5 * g2 / lam, -0.5, 0.5) * prox;
        net->cf_omega1_star = w1s;
        net->cf_omega2_star = w2s;
        net->cf_err1 = clamp2(w1s - omega1_cmd, -1.0, 1.0);
        net->cf_err2 = clamp2(w2s - omega2_cmd, -1.0, 1.0);
        /* 内部模型对误差的预期：EMA ~50 tick。注入的是 de = e - EMA(e)，
         * 即「超出预期的那部分误差」，攀爬纤维在误差已被预测时沉默。 */
        const double CF_EMA_ALPHA = 0.02;
        net->cf_ema1 += CF_EMA_ALPHA * (net->cf_err1 - net->cf_ema1);
        net->cf_ema2 += CF_EMA_ALPHA * (net->cf_err2 - net->cf_ema2);
    }

    /* reward queue 已移到 tick 开头处理（修正了 1 tick 的时序错位） */
}

void net2_get_cf_error(const NetworkState2* net, double* out4) {
    if (!net || !out4) return;
    out4[0] = net->cf_err1;
    out4[1] = net->cf_err2;
    out4[2] = net->cf_omega1_star;
    out4[3] = net->cf_omega2_star;
}

void net2_inject_modulation(NetworkState2* net, double r1, double r2,
                            double e1, double e2, double novelty,
                            int delay_ticks) {
    if (!net) return;
    int next = (net->queue_tail + 1) % QUEUE_SIZE;
    if (next == net->queue_head) return;  /* 队列满 */
    RewardEvent2* ev = &net->reward_queue[net->queue_tail];
    ev->reward_scalar = 0.5 * (r1 + r2);   /* 兼容字段，仅用于滑动基线 */
    ev->r1 = r1;
    ev->r2 = r2;
    ev->e1 = e1;
    ev->e2 = e2;
    ev->novelty = novelty;
    ev->delivery_tick = net->current_tick + delay_ticks;
    net->queue_tail = next;
}

void net2_inject_reward2(NetworkState2* net, double r1, double r2, int delay_ticks) {
    if (!net) return;
    /* 自动携带核心在本 tick 物理步之后算好的攀爬纤维误差；
     * 惊讶度 = 当前奖励与滑动基线的偏差（乙酰胆碱通道）。
     * 这样旧调用方（host / 可视化）无需改动即可获得多通道调制。 */
    double nov = fabs(0.5 * (r1 + r2) - net->reward_baseline);
    net2_inject_modulation(net, r1, r2,
                           clamp2(net->cf_err1 - net->cf_ema1, -1.0, 1.0),
                           clamp2(net->cf_err2 - net->cf_ema2, -1.0, 1.0),
                           clamp2(nov, 0.0, 2.0), delay_ticks);
}

void net2_inject_reward(NetworkState2* net, double reward, int delay_ticks) {
    net2_inject_reward2(net, reward, reward, delay_ticks);
}

/* ============================================================
 * 诊断
 * ============================================================ */
void net2_dump_stats(NetworkState2* net, FILE* fp) {
    if (!net || !fp) return;
    double w_sum = 0, w_min = 1e9, w_max = -1e9;
    int s;
    for (s = 0; s < net->synapse_count; s++) {
        w_sum += net->synapses[s].weight;
        if (net->synapses[s].weight < w_min) w_min = net->synapses[s].weight;
        if (net->synapses[s].weight > w_max) w_max = net->synapses[s].weight;
    }
    fprintf(fp, "Neurons: in=%d h=%d m1=%d m2=%d total=%d\n",
            net->input_count, net->hidden_count, net->motor1_count, net->motor2_count, net->neuron_count);
    fprintf(fp, "Synapses: %d, w_mean=%.4f min=%.4f max=%.4f\n",
            net->synapse_count,
            net->synapse_count > 0 ? w_sum / net->synapse_count : 0.0,
            net->synapse_count > 0 ? w_min : 0.0,
            net->synapse_count > 0 ? w_max : 0.0);
    fprintf(fp, "Arm: theta1=%.3f theta2=%.3f endpoint=(%.3f,%.3f) target=(%.3f,%.3f) reward=%.3f\n",
            net->arm.theta1, net->arm.theta2, net->arm.end_x, net->arm.end_y,
            net->arm.target_x, net->arm.target_y, net->arm.reward);
}

/* ============================================================
 * VIS 调试接口实现
 * ============================================================ */

/* 层类型常量 —— 与 Python 端保持一致 */
#define LAYER_INPUT_X   0
#define LAYER_HIDDEN    1
#define LAYER_MOTOR1    2
#define LAYER_MOTOR2    3

int net2_get_topology(NetworkState2* net,
                      int* neuron_types, double* neuron_pref_x, double* neuron_pref_y, double* neuron_pref_omega,
                      int* syn_pre, int* syn_post, double* syn_weight) {
    if (!net) return -1;
    int n = net->neuron_count;
    int s = net->synapse_count;
    int i, j;
    for (i = 0; i < n; i++) {
        Neuron2* nr = &net->neurons[i];
        int ltype;
        if      (i < net->input_count)                       ltype = LAYER_INPUT_X;
        else if (i < net->input_count + net->hidden_count)   ltype = LAYER_HIDDEN;
        else if (i < net->input_count + net->hidden_count + net->motor1_count) ltype = LAYER_MOTOR1;
        else                                                 ltype = LAYER_MOTOR2;
        neuron_types[i]     = ltype;
        neuron_pref_x[i]    = nr->pref_x;
        neuron_pref_y[i]    = nr->pref_y;
        neuron_pref_omega[i]= nr->pref_omega;
    }
    for (j = 0; j < s; j++) {
        syn_pre[j]   = net->synapses[j].pre_id;
        syn_post[j]  = net->synapses[j].post_id;
        syn_weight[j]= net->synapses[j].weight;
    }
    return 0;
}

void net2_get_snapshot(NetworkState2* net,
                       double* neuron_membrane, double* neuron_spiking, double* neuron_last_spike,
                       double* arm_state, int* meta) {
    if (!net) return;
    int n = net->neuron_count;
    int i;
    for (i = 0; i < n; i++) {
        neuron_membrane[i]    = net->neurons[i].membrane_potential;
        neuron_spiking[i]     = net->neurons[i].is_spiking ? 1.0 : 0.0;
        neuron_last_spike[i]  = (double)net->neurons[i].last_spike_tick;
    }
    arm_state[0] = net->arm.theta1;
    arm_state[1] = net->arm.theta2;
    arm_state[2] = net->arm.omega1;
    arm_state[3] = net->arm.omega2;
    arm_state[4] = net->arm.end_x;
    arm_state[5] = net->arm.end_y;
    arm_state[6] = net->arm.target_x;
    arm_state[7] = net->arm.target_y;
    arm_state[8] = net->arm.reward;
    arm_state[9] = net->arm.max_reach;
    meta[0] = net->current_tick;
    meta[1] = net->queue_head;
    meta[2] = net->queue_tail;
}

/* ============================================================
 * 运行时参数设置 / 轮次控制（供宿主与 Python 可视化调用）
 * ============================================================ */
static int b2_param_is(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

int net2_set_param(NetworkState2* net, const char* name, double value) {
    HyperParams2* hp;
    if (!net || !name) return -1;
    hp = &net->hp;

    if (b2_param_is(name, "learning_rate")) {
        if (value < 0.0 || value > 10.0) return -1;
        hp->learning_rate = value;
    } else if (b2_param_is(name, "tau_stdp")) {
        if (value < 1.0) return -1;
        hp->tau_stdp = value;
    } else if (b2_param_is(name, "tau_trace")) {
        if (value < 1.0) return -1;
        hp->tau_trace = value;
        hp->lambda_trace = exp(-1.0 / value);   /* 派生量必须同步 */
    } else if (b2_param_is(name, "stdp_a_plus")) {
        if (value < 0.0) return -1;
        hp->stdp_a_plus = value;
    } else if (b2_param_is(name, "stdp_a_minus")) {
        if (value < 0.0) return -1;
        hp->stdp_a_minus = value;
    } else if (b2_param_is(name, "weight_decay_lambda")) {
        if (value < 0.0) return -1;
        hp->weight_decay_lambda = value;
    } else if (b2_param_is(name, "psp_factor")) {
        if (value <= 0.0) return -1;
        hp->psp_factor = value;
    } else if (b2_param_is(name, "trace_clamp")) {
        if (value < 0.0) return -1;
        hp->trace_clamp = value;
    } else if (b2_param_is(name, "decode_mode")) {
        if (value != 0.0 && value != 2.0 && value != 3.0) return -1;
        hp->decode_mode = (int)value;
    } else if (b2_param_is(name, "prop_mode")) {
        if (value != 0.0 && value != 1.0) return -1;
        hp->prop_mode = (int)value;
    } else if (b2_param_is(name, "wta_k")) {
        if (value < 1.0 || value > 64.0) return -1;
        hp->wta_k = (int)value;
    } else if (b2_param_is(name, "use_reward_baseline")) {
        hp->use_reward_baseline = value >= 0.5 ? 1 : 0;
    } else if (b2_param_is(name, "el_trace")) {
        net->use_eligibility_trace = value >= 0.5 ? 1 : 0;
    } else if (b2_param_is(name, "reward_mod")) {
        net->use_reward_modulation = value >= 0.5 ? 1 : 0;
    } else if (b2_param_is(name, "reward_baseline_window")) {
        int w = (int)value;
        double* nh;
        if (w < 1 || w > 100000) return -1;
        nh = (double*)calloc((size_t)w, sizeof(double));
        if (!nh) return -1;
        free(net->reward_history);
        net->reward_history = nh;
        hp->reward_baseline_window = w;
        net->reward_baseline = 0.0;
        net->reward_hist_head = 0;
        net->reward_hist_count = 0;
    } else if (b2_param_is(name, "error_gain")) {
        if (value < 0.0 || value > 10.0) return -1;
        hp->error_gain = value;
    } else if (b2_param_is(name, "novelty_gain")) {
        if (value < 0.0 || value > 10.0) return -1;
        hp->novelty_gain = value;
    } else if (b2_param_is(name, "cf_lambda")) {
        /* v7：允许 0 = 明确关闭解析逆运动学期望速度（g=0 口径）。
         * 注意 core 仍会计算 omega* 供诊断，但 error_gain=0 时它没有任何通往权重的路径。 */
        if (value < 0.0 || value > 10.0) return -1;
        hp->cf_lambda = value;
    } else if (b2_param_is(name, "hetero_tau")) {
        if (value < 0.0 || value > 1.0) return -1;
        hp->hetero_tau = value;
        apply_heterogeneity(net);
    } else if (b2_param_is(name, "hetero_thresh")) {
        if (value < 0.0 || value > 10.0) return -1;
        hp->hetero_thresh = value;
        apply_heterogeneity(net);
    } else if (b2_param_is(name, "hetero_tune")) {
        if (value < 0.0 || value > 1.0) return -1;
        hp->hetero_tune = value;
        apply_heterogeneity(net);
    } else {
        return -1;
    }
    return 0;
}

double net2_get_param(const NetworkState2* net, const char* name) {
    const HyperParams2* hp;
    if (!net || !name) return NAN;
    hp = &net->hp;
#define B2_GET(n, f) if (b2_param_is(name, n)) return (double)(f)
    B2_GET("learning_rate", hp->learning_rate);
    B2_GET("tau_stdp", hp->tau_stdp);
    B2_GET("tau_trace", hp->tau_trace);
    B2_GET("lambda_trace", hp->lambda_trace);
    B2_GET("stdp_a_plus", hp->stdp_a_plus);
    B2_GET("stdp_a_minus", hp->stdp_a_minus);
    B2_GET("weight_decay_lambda", hp->weight_decay_lambda);
    B2_GET("reward_baseline_window", hp->reward_baseline_window);
    B2_GET("psp_factor", hp->psp_factor);
    B2_GET("trace_clamp", hp->trace_clamp);
    B2_GET("decode_mode", hp->decode_mode);
    B2_GET("wta_k", hp->wta_k);
    B2_GET("use_reward_baseline", hp->use_reward_baseline);
    B2_GET("el_trace", net->use_eligibility_trace);
    B2_GET("reward_mod", net->use_reward_modulation);
    B2_GET("reward_baseline", net->reward_baseline);
    B2_GET("current_tick", net->current_tick);
    B2_GET("error_gain", hp->error_gain);
    B2_GET("novelty_gain", hp->novelty_gain);
    B2_GET("cf_lambda", hp->cf_lambda);
    B2_GET("hetero_tau", hp->hetero_tau);
    B2_GET("prop_mode", hp->prop_mode);
    B2_GET("hetero_thresh", hp->hetero_thresh);
    B2_GET("hetero_tune", hp->hetero_tune);
    B2_GET("cf_err1", net->cf_err1);
    B2_GET("cf_err2", net->cf_err2);
#undef B2_GET
    return NAN;
}

void net2_episode_reset(NetworkState2* net, double tx, double ty) {
    int i;
    if (!net) return;
    /* 关节与末端归位；资格迹、奖励基线、权重全部保留，学习成果不丢 */
    net->arm.theta1 = 0.0;
    net->arm.theta2 = 0.0;
    net->arm.omega1 = 0.0;
    net->arm.omega2 = 0.0;
    net->arm.end_x = ARM_L1 + ARM_L2;
    net->arm.end_y = 0.0;
    net->arm.max_reach = ARM_L1 + ARM_L2;
    net->arm.reward = 1.0;
    net->decoded_omega1 = 0.0;
    net->decoded_omega2 = 0.0;
    net->cf_err1 = 0.0;
    net->cf_err2 = 0.0;
    net->cf_omega1_star = 0.0;
    net->cf_omega2_star = 0.0;
    net->cf_ema1 = 0.0;
    net->cf_ema2 = 0.0;
    net2_reset_ema();
    net2_set_target(net, tx, ty);
    for (i = 0; i < net->neuron_count; i++) {
        Neuron2* n = &net->neurons[i];
        if (n->type == NT_INPUT_X) continue;   /* 输入层保留目标编码 */
        n->membrane_potential = MEMBRANE_REST;
        n->is_spiking = 0;
        n->refractory_remaining = 0;
        n->last_spike_tick = -1;
    }
    net->queue_head = 0;
    net->queue_tail = 0;
}

void net2_get_arm(const NetworkState2* net, double* a) {
    if (!net || !a) return;
    a[0] = net->arm.theta1;
    a[1] = net->arm.theta2;
    a[2] = net->arm.omega1;
    a[3] = net->arm.omega2;
    a[4] = net->arm.end_x;
    a[5] = net->arm.end_y;
    a[6] = net->arm.target_x;
    a[7] = net->arm.target_y;
    a[8] = net->arm.reward;
    a[9] = net->arm.max_reach;
}

/* JSON 拓扑（一次性，初始化后调用） */
void net2_dump_topology_json(NetworkState2* net, const char* filepath) {
    if (!net || !filepath) return;
    FILE* f = fopen(filepath, "w");
    if (!f) return;
    int n = net->neuron_count;
    int s = net->synapse_count;
    int i;
    fprintf(f, "{\"neuron_count\":%d,\"synapse_count\":%d,\"input\":%d,\"hidden\":%d,\"motor1\":%d,\"motor2\":%d,\n",
            n, s, net->input_count, net->hidden_count, net->motor1_count, net->motor2_count);
    fprintf(f, "\"neurons\":[");
    for (i = 0; i < n; i++) {
        Neuron2* nr = &net->neurons[i];
        int ltype;
        if      (i < net->input_count)                       ltype = LAYER_INPUT_X;
        else if (i < net->input_count + net->hidden_count)   ltype = LAYER_HIDDEN;
        else if (i < net->input_count + net->hidden_count + net->motor1_count) ltype = LAYER_MOTOR1;
        else                                                 ltype = LAYER_MOTOR2;
        fprintf(f, "%s{\"id\":%d,\"layer\":%d,\"pref_x\":%.4f,\"pref_y\":%.4f,\"pref_omega\":%.4f}",
                i ? "," : "", i, ltype, nr->pref_x, nr->pref_y, nr->pref_omega);
    }
    fprintf(f, "],\n\"synapses\":[");
    for (i = 0; i < s; i++) {
        fprintf(f, "%s{\"pre\":%d,\"post\":%d,\"w\":%.5f}",
                i ? "," : "", net->synapses[i].pre_id, net->synapses[i].post_id, net->synapses[i].weight);
    }
    fprintf(f, "]}\n");
    fclose(f);
}

/* JSON 实时状态（每 tick 覆盖写） */
void net2_dump_snapshot_json(NetworkState2* net, const char* filepath) {
    if (!net || !filepath) return;
    FILE* f = fopen(filepath, "w");
    if (!f) return;
    int n = net->neuron_count;
    int i;
    fprintf(f, "{\"tick\":%d,\"neurons\":[", net->current_tick);
    for (i = 0; i < n; i++) {
        Neuron2* nr = &net->neurons[i];
        fprintf(f, "%s{\"m\":%.3f,\"s\":%d,\"lst\":%d}",
                i ? "," : "", nr->membrane_potential, nr->is_spiking ? 1 : 0, nr->last_spike_tick);
    }
    ArmState* a = &net->arm;
    fprintf(f, "],\"arm\":{");
    fprintf(f, "\"t1\":%.4f,\"t2\":%.4f,\"o1\":%.4f,\"o2\":%.4f,\"ex\":%.4f,\"ey\":%.4f,",
            a->theta1, a->theta2, a->omega1, a->omega2, a->end_x, a->end_y);
    fprintf(f, "\"tx\":%.4f,\"ty\":%.4f,\"r\":%.4f,\"mr\":%.4f}}\n",
            a->target_x, a->target_y, a->reward, a->max_reach);
    fclose(f);
}
