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
void arm_step(ArmState* arm, double omega1_cmd, double omega2_cmd) {
    double max_omega = 0.5;  /* 限制角速度，防止飞远 */
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
    
    net->input_count = input_n;
    net->hidden_count = hidden_n;
    net->motor1_count = motor1_n;
    net->motor2_count = motor2_n;
    net->neuron_count = input_n + hidden_n + motor1_n + motor2_n;
    
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
        
        /* 2D 网格 + 抖动，覆盖整个可达空间圆 [-R, R]² */
        int grid_side = (int)ceil(sqrt((double)input_n));
        int gx = i % grid_side;
        int gy = i / grid_side;
        double range = ARM_L1 + ARM_L2;                  /* R = 2.0 */
        double step = 2.0 * range / (grid_side - 1);     /* 4.0 / 5 = 0.8 for 36 neurons */
        net->neurons[idx].pref_x = -range + gx * step + r_u() * 0.1;
        net->neurons[idx].pref_y = -range + gy * step + r_u() * 0.1;
        /* 限制到可达空间 */
        double pr = sqrt(net->neurons[idx].pref_x * net->neurons[idx].pref_x +
                         net->neurons[idx].pref_y * net->neurons[idx].pref_y);
        if (pr > ARM_L1 + ARM_L2) {
            net->neurons[idx].pref_x *= (ARM_L1 + ARM_L2) / pr;
            net->neurons[idx].pref_y *= (ARM_L1 + ARM_L2) / pr;
        }
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
        net->neurons[idx].pref_omega = -1.0 + 2.0 * (double)i / (double)(motor1_n - 1);
    }
    
    /* 4. 关节2运动层 */
    for (i = 0; i < motor2_n; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_MOTOR_OMEGA2;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
        net->neurons[idx].pref_omega = -1.0 + 2.0 * (double)i / (double)(motor2_n - 1);
    }
    
    /* 突触分配：预估算（4 个连接层分别分配） */
    int K = (int)avg_conn;
    if (K < 3) K = 3;
    /* 需要的总突触数 = input→hidden + hidden→hidden/2 + hidden→motor1 + hidden→motor2 */
    int needed = input_n * K + hidden_n * (K / 2) + hidden_n * K + hidden_n * K;
    int est = needed * 3 / 2;  /* 留 50% 余量 */
    net->synapses = (Synapse2*)calloc(est, sizeof(Synapse2));
    if (!net->synapses) { free(net->neurons); return -1; }
    
    net->synapse_count = 0;
    int s_idx;
    
    /* === 连接1: INPUT_X → HIDDEN（稀疏） === */
    for (i = 0; i < input_n; i++) {
        int pre = i;
        int hidden_base = input_n;
        for (int k = 0; k < K; k++) {
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
            net->synapses[s_idx].weight = 0.3 + r_u() * 0.7;
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接2: HIDDEN → HIDDEN（稀疏递归） === */
    for (i = 0; i < hidden_n; i++) {
        int pre = input_n + i;
        for (int k = 0; k < K / 2; k++) {
            int post = input_n + (int)(r_u() * hidden_n);
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
    int motor1_base = input_n + hidden_n;
    for (i = 0; i < hidden_n; i++) {
        int pre = input_n + i;
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
            net->synapses[s_idx].weight = 0.1 + r_u() * 0.2;  /* 较低初始权重，避免同时激活所有 motor */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 连接4: HIDDEN → MOTOR2 === */
    int motor2_base = motor1_base + motor1_n;
    for (i = 0; i < hidden_n; i++) {
        int pre = input_n + i;
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
            net->synapses[s_idx].weight = 0.1 + r_u() * 0.2;  /* 同上 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* 超参数（延续 B1 校准值，规模更大时保持） */
    net->hp.tau_stdp = 50.0;
    net->hp.tau_trace = 200.0;
    net->hp.lambda_trace = exp(-1.0 / net->hp.tau_trace);
    net->hp.stdp_a_plus = 0.1;
    net->hp.stdp_a_minus = 0.12;
    net->hp.learning_rate = 0.1;
    net->hp.weight_decay_lambda = 1e-5;
    net->hp.reward_baseline_window = 100;
    
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
    
    /* 用高斯核直接激活输入层神经元（连续强度 → 发放概率 / 直接膜电位注入） */
    int i;
    double sigma2 = ENCODE_SIGMA_POS * ENCODE_SIGMA_POS;
    for (i = 0; i < net->input_count; i++) {
        Neuron2* n = &net->neurons[i];
        double dx = n->pref_x - tx;
        double dy = n->pref_y - ty;
        double dist2 = dx*dx + dy*dy;
        double response = exp(-dist2 / (2.0 * sigma2));
        /* 直接注入膜电位：响应越强 → 越接近阈值 */
        n->membrane_potential = MEMBRANE_REST + response * (SPIKE_THRESHOLD - MEMBRANE_REST + 3.0);
        /* 发放概率调制（概率发放让编码更自然） */
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
    
    /* Step 0: 重新设置输入层膜电位（持续编码目标位置） */
    {
        double tx = net->arm.target_x;
        double ty = net->arm.target_y;
        double sigma2 = ENCODE_SIGMA_POS * ENCODE_SIGMA_POS;
        for (i = 0; i < net->input_count; i++) {
            Neuron2* n = &net->neurons[i];
            double dx = n->pref_x - tx;
            double dy = n->pref_y - ty;
            double dist2 = dx*dx + dy*dy;
            double response = exp(-dist2 / (2.0 * sigma2));
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
        /* 输入层：不衰减（目标激活是持续刺激），但必须做发放判定！ */
        if (n->type != NT_INPUT_X) {
            /* 不应期 */
            if (n->refractory_remaining > 0) {
                n->refractory_remaining--;
                n->membrane_potential = MEMBRANE_REST;
                continue;
            }
            /* 膜电位衰减 */
            n->membrane_potential = n->membrane_potential * MEMBRANE_DECAY + MEMBRANE_REST * (1.0 - MEMBRANE_DECAY);
        }
        
        /* 发放判定（先标记，后统一处理 WTA 和膜电位重置） */
        if (n->membrane_potential >= n->spike_threshold) {
            n->is_spiking = 1;
        }
    }
    
    /* MOTOR 层 k-winner-take-all：每关节最多 K 个运动神经元发放 */
    {
        int m1_base = net->input_count + net->hidden_count;
        int m2_base = m1_base + net->motor1_count;
        int WTA_K = net->motor1_count / 3;  /* 约 1/3 能发 */
        if (WTA_K < 2) WTA_K = 2;
        
        for (int m_layer = 0; m_layer < 2; m_layer++) {
            int base = (m_layer == 0) ? m1_base : m2_base;
            int count = (m_layer == 0) ? net->motor1_count : net->motor2_count;
            
            /* 找膜电位最高的 WTA_K 个 */
            int winners[64];
            for (int w = 0; w < WTA_K; w++) winners[w] = -1;
            
            for (int j = 0; j < count; j++) {
                Neuron2* n = &net->neurons[base + j];
                /* 插入排序找 top WTA_K */
                int k = n->membrane_potential;  /* 用膜电位作 key */
                /* 简化：遍历找最低的 winner，若当前更高则替换 */
                int min_w = 0;
                if (winners[0] == -1 || k > net->neurons[base + winners[0]].membrane_potential) {
                    /* 找最小的 */
                    for (int w = 1; w < WTA_K; w++) {
                        if (winners[w] == -1) { min_w = w; break; }
                        if (net->neurons[base + winners[w]].membrane_potential < 
                            net->neurons[base + winners[min_w]].membrane_potential) min_w = w;
                    }
                    if (winners[min_w] == -1 || k > net->neurons[base + winners[min_w]].membrane_potential) {
                        winners[min_w] = j;
                    }
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
        if (n->type != NT_INPUT_X) {
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
            if (pre->is_spiking && post->last_spike_tick > 0) {
                stdp_val += stdp_kernel2(net, (double)(net->current_tick - post->last_spike_tick));
            }
            if (post->is_spiking && pre->last_spike_tick > 0) {
                stdp_val += stdp_kernel2(net, (double)(net->current_tick - pre->last_spike_tick));
            }
            net->synapses[s].eligibility_trace += stdp_val;
            if (net->synapses[s].eligibility_trace > 5.0)  net->synapses[s].eligibility_trace = 5.0;
            if (net->synapses[s].eligibility_trace < -5.0) net->synapses[s].eligibility_trace = -5.0;
        } else {
            net->synapses[s].eligibility_trace = 0.0;
        }
        #else
        net->synapses[s].eligibility_trace = 0.0;
        #endif
    }
    
    /* Step 4: 运动层解码 → 关节角速度命令 */
    double omega1_cmd = 0.0, omega2_cmd = 0.0;
    int m1_base = net->input_count + net->hidden_count;
    int m2_base = m1_base + net->motor1_count;
    double rate_sum1 = 0.0, rate_sum2 = 0.0;
    
    /* 统计每个运动神经元的发放率（近5 ticks） */
    for (i = 0; i < net->motor1_count; i++) {
        Neuron2* n = &net->neurons[m1_base + i];
        int since = net->current_tick - n->last_spike_tick;
        /* 只有真正发放过（last_spike_tick >= 0）才算 */
        double rate = (n->last_spike_tick >= 0 && since >= 0 && since <= 5) ? 1.0 : 0.0;
        omega1_cmd += n->pref_omega * rate;
        rate_sum1 += rate;
    }
    for (i = 0; i < net->motor2_count; i++) {
        Neuron2* n = &net->neurons[m2_base + i];
        int since = net->current_tick - n->last_spike_tick;
        double rate = (n->last_spike_tick >= 0 && since >= 0 && since <= 5) ? 1.0 : 0.0;
        omega2_cmd += n->pref_omega * rate;
        rate_sum2 += rate;
    }
    
    /* 归一化（无发放则保持0） */
    if (rate_sum1 > 0) omega1_cmd /= rate_sum1;
    if (rate_sum2 > 0) omega2_cmd /= rate_sum2;
    
    /* Step 5: 保存 prev → 物理模拟一步 */
    double prev_arm_dist = sqrt(pow(net->arm.end_x - net->arm.target_x, 2) +
                                pow(net->arm.end_y - net->arm.target_y, 2));
    arm_step(&net->arm, omega1_cmd, omega2_cmd);
    
    /* Step 6: 绝对 reward 为主（越近越好），改进奖励辅助 */
    double curr_arm_dist = sqrt(pow(net->arm.end_x - net->arm.target_x, 2) +
                                pow(net->arm.end_y - net->arm.target_y, 2));
    double improvement = (prev_arm_dist - curr_arm_dist) / net->arm.max_reach;
    double abs_reward = net->arm.reward;
    /* 绝对 reward 为主 + 正改进奖励（不惩罚后退） */
    double reward = abs_reward + (improvement > 0 ? improvement * 5.0 : 0.0);
    net2_inject_reward(net, reward, 50);
    
    /* 处理队列 */
    while (net->queue_head != net->queue_tail &&
           net->reward_queue[net->queue_head].delivery_tick <= net->current_tick) {
        RewardEvent2 ev = net->reward_queue[net->queue_head];
        net->queue_head = (net->queue_head + 1) % QUEUE_SIZE;
        
        double err = ev.reward_scalar - net->reward_baseline;
        
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
                    double dw = net->hp.learning_rate * err * credit;
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

void net2_inject_reward(NetworkState2* net, double reward, int delay_ticks) {
    int next = (net->queue_tail + 1) % QUEUE_SIZE;
    if (next == net->queue_head) return;  /* 队列满 */
    net->reward_queue[net->queue_tail].reward_scalar = reward;
    net->reward_queue[net->queue_tail].delivery_tick = net->current_tick + delay_ticks;
    net->queue_tail = next;
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
