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

/* ============================================================
 * 两关节臂物理
 * ============================================================ */
/* EMA 平滑状态（全局，便于 episode 重置时清零） */
static double g_ema_o1 = 0.0, g_ema_o2 = 0.0;
void net2_reset_ema(void) { g_ema_o1 = 0.0; g_ema_o2 = 0.0; }

void arm_step(ArmState* arm, double omega1_cmd, double omega2_cmd) {
    /* 直接用原始 omega，不加 EMA 平滑
     * 抖动是好事：让 arm 有机会朝各种方向试探，碰到正反馈就能启动学习 */
    double max_omega = 0.5;
    arm->omega1 = clamp2(omega1_cmd, -max_omega, max_omega);
    arm->omega2 = clamp2(omega2_cmd, -max_omega, max_omega);
    
    arm->theta1 += arm->omega1 * ARM_DT;
    arm->theta2 += arm->omega2 * ARM_DT;
    
    /* 边界反弹 */
    if (arm->theta1 < ARM_THETA1_MIN) arm->theta1 = ARM_THETA1_MIN;
    if (arm->theta1 > ARM_THETA1_MAX) arm->theta1 = ARM_THETA1_MAX;
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
    
    /* 突触分配：预估算（4 个连接层分别分配） */
    int K = (int)avg_conn;
    if (K < 3) K = 3;
    /* 需要的总突触数 = input→hidden + hidden→hidden/2 + hidden→motor1 + hidden→motor2 */
    int needed = net->input_count * K + hidden_n * (K / 2) + hidden_n * K + hidden_n * K;
    int est = needed * 3 / 2;  /* 留 50% 余量 */
    net->synapses = (Synapse2*)calloc(est, sizeof(Synapse2));
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
            
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 1.0 + r_u() * 2.0;  /* 更大初始权重 1-3 */
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
            
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.1 + r_u() * 0.3;
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接3: HIDDEN → MOTOR1 === */
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
            
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.3 + r_u() * 0.4;  /* 0.3-0.7（原 0.1-0.3） */
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
            
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.3 + r_u() * 0.4;  /* 0.3-0.7 */
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
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 1.5 + r_u() * 1.0;  /* 1.5-2.5，更强直连 */
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
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 1.5 + r_u() * 1.0;  /* 1.5-2.5 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* 超参数：防止 trace 瞬间饱和 */
    net->hp.tau_stdp = 50.0;
    net->hp.tau_trace = 500.0;          /* 从 2000 → 500，让 trace 更快衰减 */
    net->hp.lambda_trace = exp(-1.0 / net->hp.tau_trace);
    net->hp.stdp_a_plus = 0.05;         /* 从 0.1 → 0.05，每次 STDP 事件给 trace 更小增量 */
    net->hp.stdp_a_minus = 0.06;        /* 从 0.12 → 0.06，保持 a_minus/a_plus = 1.2 */
    net->hp.learning_rate = 0.15;       /* 恢复 0.15 */
    net->hp.weight_decay_lambda = 1e-5;
    net->hp.reward_baseline_window = 50;  /* 恢复 50 */
    
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
    if (net->neurons) free(net->neurons);
    if (net->synapses) free(net->synapses);
    if (net->reward_history) free(net->reward_history);
    memset(net, 0, sizeof(NetworkState2));
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
            post->membrane_potential += net->synapses[s].weight * PSP_FACTOR;
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
            /* 膜电位衰减 */
            n->membrane_potential = n->membrane_potential * MEMBRANE_DECAY + MEMBRANE_REST * (1.0 - MEMBRANE_DECAY);
            
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
        int m1_base = net->input_count + net->hidden_count;
        int m2_base = m1_base + net->motor1_count;
        int WTA_K = 3;  /* K=3 是之前验证最好的 */
        
        for (int m_layer = 0; m_layer < 2; m_layer++) {
            int base = (m_layer == 0) ? m1_base : m2_base;
            int count = (m_layer == 0) ? net->motor1_count : net->motor2_count;
            
            /* 正确的 k-winner 算法：
             * 维护 K 个 winner，每次找到最小的 winner，若当前神经元更大则替换 */
            int winners[64];
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
             * 两个分支分别覆盖 pre 先 spike 和 post 先 spike 的情况 */
            if (pre->is_spiking && post->last_spike_tick > 0) {
                /* pre 在 post 后 spike → dt = pre_tick - post_tick > 0 → pre→post 时间差 < 0 → LTD */
                stdp_val += stdp_kernel2(net, -(double)(net->current_tick - post->last_spike_tick));
            }
            if (post->is_spiking && pre->last_spike_tick > 0) {
                /* pre 在 post 前 spike → dt = pre_tick - post_tick < 0 → pre→post 时间差 > 0 → LTP */
                stdp_val += stdp_kernel2(net, (double)(net->current_tick - pre->last_spike_tick));
            }
            net->synapses[s].eligibility_trace += stdp_val;
            if (net->synapses[s].eligibility_trace > 1.0)  net->synapses[s].eligibility_trace = 1.0;
            if (net->synapses[s].eligibility_trace < -1.0) net->synapses[s].eligibility_trace = -1.0;
        } else {
            net->synapses[s].eligibility_trace = 0.0;
        }
        #else
        net->synapses[s].eligibility_trace = 0.0;
        #endif
    }
    
    /* Step 4: SNN 解码 → 直接看 is_spiking（WTA winner）+ **直接平均所有 winner**！
     * 之前用方向池分离平均：2正1负→只取正的平均（丢掉负信息），太粗！
     * 现在直接平均所有 winner pref_omega → 2正1负(+0.5,+0.7,-0.1) → 平均+0.37，精细连续值！*/
    double omega1_cmd = 0.0, omega2_cmd = 0.0;
    {
        int m1_base = net->input_count + net->hidden_count;
        int m2_base = m1_base + net->motor1_count;
        
        for (int j = 0; j < 2; j++) {
            int m_base = (j == 0) ? m1_base : m2_base;
            int m_count = (j == 0) ? net->motor1_count : net->motor2_count;
            
            /* 直接平均所有 spike winner 的 pref_omega → 精细连续值！*/
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
    
    /* 解码后加小探索噪声：让 arm 轻微试探不同方向，不会淹没 WTA 信号 */
    omega1_cmd += r_gauss(0.0, 0.05);
    omega2_cmd += r_gauss(0.0, 0.05);
    omega1_cmd = fmax(-0.5, fmin(0.5, omega1_cmd));
    omega2_cmd = fmax(-0.5, fmin(0.5, omega2_cmd));
    
    net->decoded_omega1 = omega1_cmd;
    net->decoded_omega2 = omega2_cmd;
    
    /* Step 5: 物理模拟一步 */
    arm_step(&net->arm, omega1_cmd, omega2_cmd);
    
    /* Step 6: 处理 reward queue */
    while (net->queue_head != net->queue_tail &&
           net->reward_queue[net->queue_head].delivery_tick <= net->current_tick) {
        RewardEvent2 ev = net->reward_queue[net->queue_head];
        net->queue_head = (net->queue_head + 1) % QUEUE_SIZE;
        
        double err = ev.reward_scalar - net->reward_baseline;
        double err1 = ev.r1 - net->reward_baseline;   /* 关节1 专用 err */
        double err2 = ev.r2 - net->reward_baseline;   /* 关节2 专用 err */
        
        #if REWARD_MODULATION_ENABLE
        if (net->use_reward_modulation && fabs(err) > 0.001) {
            for (s = 0; s < net->synapse_count; s++) {
                Synapse2* syn = &net->synapses[s];
                #if ELIGIBILITY_TRACE_ENABLE
                double credit = net->use_eligibility_trace ? syn->eligibility_trace : 0.0;
                #else
                double credit = 0.0;
                #endif
                if (fabs(credit) > 1e-8) {
                    /* 分关节 credit：post 是 MOTOR1 用 err1，MOTOR2 用 err2，其余用全局 err
                     * ω1 好 ω2 差时不再互相连坐，credit 精度翻倍 */
                    NeuronType2 pt = net->neurons[syn->post_id].type;
                    double e = (pt == NT_MOTOR_OMEGA1) ? err1
                             : (pt == NT_MOTOR_OMEGA2) ? err2 : err;
                    double dw = net->hp.learning_rate * e * credit;
                    syn->weight = clamp2(syn->weight + dw, WEIGHT_MIN, WEIGHT_MAX);
                }
            }
        }
        #endif
        
        /* 更新基线 */
        net->reward_history[net->reward_hist_head] = ev.reward_scalar;
        net->reward_hist_head = (net->reward_hist_head + 1) % net->hp.reward_baseline_window;
        if (net->reward_hist_count < net->hp.reward_baseline_window) net->reward_hist_count++;
        
        double sum = 0.0;
        for (int k = 0; k < net->reward_hist_count; k++) sum += net->reward_history[k];
        net->reward_baseline = sum / net->reward_hist_count;
    }
}

void net2_inject_reward2(NetworkState2* net, double r1, double r2, int delay_ticks) {
    int next = (net->queue_tail + 1) % QUEUE_SIZE;
    if (next == net->queue_head) return;  /* 队列满 */
    net->reward_queue[net->queue_tail].reward_scalar = 0.5 * (r1 + r2);
    net->reward_queue[net->queue_tail].r1 = r1;
    net->reward_queue[net->queue_tail].r2 = r2;
    net->reward_queue[net->queue_tail].delivery_tick = net->current_tick + delay_ticks;
    net->queue_tail = next;
}

void net2_inject_reward(NetworkState2* net, double reward, int delay_ticks) {
    net2_inject_reward2(net, reward, reward, delay_ticks);
}

/* ============================================================
 * 诊断
 * ============================================================ */
void net2_dump_stats(NetworkState2* net, FILE* fp) {
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
            net->synapse_count, w_sum / net->synapse_count, w_min, w_max);
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
