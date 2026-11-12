/*
 * benchmark1_core.c - Benchmark-1 核心实现
 * 拟生物神经元系统 · 单突触远端信用分配验证
 *
 * 核心逻辑：
 *   1. Tick循环 → 膜电位更新 → 发放判定 → 事件驱动STDP+资格迹 → 奖励队列处理
 *   2. 资格迹：只累积不直接改权重，奖励到达后与奖励误差相乘更新
 *   3. 惰性衰减：突触衰减仅在被访问时按时间差批量计算
 */

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "benchmark1_core.h"

/* ============================================================
 * 工具函数：随机数（固定种子可复现）
 * ============================================================ */
static unsigned int g_rand_state = 12345;

static inline double rand_uniform(void) {
    g_rand_state = g_rand_state * 1103515245 + 12345;
    return (double)((g_rand_state >> 16) & 0x7FFF) / 32767.0;
}

static inline double rand_gaussian(double mean, double std) {
    /* Box-Muller变换 */
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-10) u1 = 1e-10;
    return mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static inline double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* ============================================================
 * 网络初始化
 * ============================================================ */
int net_init(NetworkState* net, int sensory_a, int sensory_b,
             int hidden, int output_a, int output_b,
             double avg_conn_per_n, unsigned int seed) {
    memset(net, 0, sizeof(NetworkState));
    
    /* 设置随机种子 */
    g_rand_state = seed;
    
    /* 神经元数量记录 */
    net->sensory_a_count = sensory_a;
    net->sensory_b_count = sensory_b;
    net->hidden_count    = hidden;
    net->output_a_count  = output_a;
    net->output_b_count  = output_b;
    net->neuron_count    = sensory_a + sensory_b + hidden + output_a + output_b;
    
    /* 分配神经元数组 */
    net->neurons = (Neuron*)calloc(net->neuron_count, sizeof(Neuron));
    if (!net->neurons) return -1;
    
    /* 初始化神经元 */
    int idx = 0;
    int i;
    for (i = 0; i < sensory_a; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_SENSORY_A;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
    }
    for (i = 0; i < sensory_b; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_SENSORY_B;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
    }
    for (i = 0; i < hidden; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_HIDDEN;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
    }
    for (i = 0; i < output_a; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_OUTPUT_A;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
    }
    for (i = 0; i < output_b; i++, idx++) {
        net->neurons[idx].id = idx;
        net->neurons[idx].type = NT_OUTPUT_B;
        net->neurons[idx].membrane_potential = MEMBRANE_REST;
        net->neurons[idx].last_spike_tick = -1;
        net->neurons[idx].spike_threshold = SPIKE_THRESHOLD;
    }
    
    /* 估算突触数量并分配 */
    int est_synapses = (int)(net->neuron_count * avg_conn_per_n);
    net->synapses = (Synapse*)calloc(est_synapses, sizeof(Synapse));
    if (!net->synapses) { free(net->neurons); return -1; }
    
    /* 构建突触连接（分层拓扑 + 少量跨层连接）
     * 拓扑：感觉A → 隐藏左半 → 输出A
     *       感觉B → 隐藏右半 → 输出B
     *       + 少量跨层连接提供学习可塑性
     */
    net->synapse_count = 0;
    int s_idx;
    int K_feedforward = (int)avg_conn_per_n;
    if (K_feedforward < 3) K_feedforward = 3;
    
    int hidden_mid = hidden / 2;  /* 隐藏层中点 */
    
    /* === 感觉A → 隐藏左半（主路径） + 少量跨到右半 === */
    for (i = 0; i < sensory_a; i++) {
        int pre = i;
        int left_count = (int)(K_feedforward * 0.65);
        int right_count = K_feedforward - left_count;
        
        int hidden_base = sensory_a + sensory_b;
        
        /* 连接到左半（正确路径） */
        for (int k = 0; k < left_count && net->synapse_count < est_synapses; k++) {
            int post = hidden_base + (int)(rand_uniform() * hidden_mid);
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.15 + rand_uniform() * 0.25;  /* 主路径 0.1~0.3 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
        /* 少量跨到右半（错误路径，供学习抑制） */
        for (int k = 0; k < right_count && net->synapse_count < est_synapses; k++) {
            int post = hidden_base + hidden_mid + (int)(rand_uniform() * (hidden - hidden_mid));
            if (post >= hidden_base + hidden) continue;
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.05 + rand_uniform() * 0.1;  /* 跨层 0.05~0.15 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 感觉B → 隐藏右半（主路径） + 少量跨到左半 === */
    for (i = 0; i < sensory_b; i++) {
        int pre = sensory_a + i;
        int right_count = (int)(K_feedforward * 0.65);
        int left_count = K_feedforward - right_count;
        
        int hidden_base = sensory_a + sensory_b;
        
        /* 连接到右半（正确路径） */
        for (int k = 0; k < right_count && net->synapse_count < est_synapses; k++) {
            int post = hidden_base + hidden_mid + (int)(rand_uniform() * (hidden - hidden_mid));
            if (post >= hidden_base + hidden) continue;
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.15 + rand_uniform() * 0.25;
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
        /* 少量跨到左半 */
        for (int k = 0; k < left_count && net->synapse_count < est_synapses; k++) {
            int post = hidden_base + (int)(rand_uniform() * hidden_mid);
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.05 + rand_uniform() * 0.1;
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 隐藏左半 → 输出A/B（初始 50/50，无预埋正确映射） === */
    for (i = 0; i < hidden_mid; i++) {
        int pre = sensory_a + sensory_b + i;
        int out_base = sensory_a + sensory_b + hidden;
        
        for (int k = 0; k < K_feedforward && net->synapse_count < est_synapses; k++) {
            int post;
            if (rand_uniform() < 0.5) {
                post = out_base + (int)(rand_uniform() * output_a);
            } else {
                post = out_base + output_a + (int)(rand_uniform() * output_b);
            }
            if (post >= net->neuron_count) continue;
            
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.12 + rand_uniform() * 0.2;  /* 弱权重 */
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* === 隐藏右半 → 输出A/B（初始 50/50） === */
    for (i = hidden_mid; i < hidden; i++) {
        int pre = sensory_a + sensory_b + i;
        int out_base = sensory_a + sensory_b + hidden;
        
        for (int k = 0; k < K_feedforward && net->synapse_count < est_synapses; k++) {
            int post;
            if (rand_uniform() < 0.5) {
                post = out_base + (int)(rand_uniform() * output_a);
            } else {
                post = out_base + output_a + (int)(rand_uniform() * output_b);
            }
            if (post >= net->neuron_count) continue;
            
            s_idx = net->synapse_count++;
            net->synapses[s_idx].pre_id = pre;
            net->synapses[s_idx].post_id = post;
            net->synapses[s_idx].weight = 0.12 + rand_uniform() * 0.2;
            net->synapses[s_idx].eligibility_trace = 0.0;
            net->synapses[s_idx].last_stdp_tick = 0;
        }
    }
    
    /* 超参数默认值 */
    net->hp.tau_stdp = 50.0;    /* STDP时间常数 (ticks) */
    net->hp.tau_trace = 200.0;  /* 单尺度回退时间常数 */
    net->hp.lambda_trace = exp(-1.0 / net->hp.tau_trace);

    /* 多尺度资格迹（M1）：K=3，tau = 20/100/500，alpha = 1/K。
     * 单尺度消融用 trace_k=1 + tau_trace_k[0]=200 复现旧行为。 */
    net->hp.trace_k = 3;
    {
        const double taus[3] = { 20.0, 100.0, 500.0 };
        int k;
        for (k = 0; k < 3; k++) {
            net->hp.tau_trace_k[k] = taus[k];
            net->hp.lambda_trace_k[k] = exp(-1.0 / taus[k]);
            net->hp.trace_alpha[k] = 1.0 / 3.0;
        }
    }

    net->hp.stdp_a_plus = 0.1;
    net->hp.stdp_a_minus = 0.12;
    net->hp.learning_rate = 0.005;  /* 很小的学习率，慢学习放大微弱偏向 */
    net->hp.weight_decay_lambda = 1e-5;
    net->hp.reward_baseline_window = 100;
    /* PSP 因子必须让输出层处于亚阈值区，否则权重变化不影响发放，
     * 学习对投票无效（M1 诊断：PSP=25 时输出饱和，准确率恒 46.7%）。 */
    net->hp.psp_factor = 10.0;
    
    /* 消融开关初始化 */
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
    
    /* 奖励基线初始化 */
    net->reward_baseline = 0.0;
    net->reward_history = (double*)calloc(net->hp.reward_baseline_window, sizeof(double));
    net->reward_hist_head = 0;
    net->reward_hist_count = 0;
    
    net->queue_head = 0;
    net->queue_tail = 0;
    net->current_tick = 0;
    
    return 0;
}

/* ============================================================
 * 网络销毁
 * ============================================================ */
void net_destroy(NetworkState* net) {
    if (net->neurons) free(net->neurons);
    if (net->synapses) free(net->synapses);
    if (net->reward_history) free(net->reward_history);
    memset(net, 0, sizeof(NetworkState));
}

/* ============================================================
 * 超参数设置
 * ============================================================ */
void net_set_hyperparams(NetworkState* net, HyperParams* hp) {
    net->hp = *hp;

    /* 归一化多尺度参数，容忍外部只填部分字段 */
    if (net->hp.trace_k < 1) net->hp.trace_k = 1;
    if (net->hp.trace_k > 3) net->hp.trace_k = 3;
    net->hp.lambda_trace = exp(-1.0 / net->hp.tau_trace);
    {
        int k;
        for (k = 0; k < 3; k++) {
            if (net->hp.tau_trace_k[k] <= 0.0) net->hp.tau_trace_k[k] = net->hp.tau_trace;
            net->hp.lambda_trace_k[k] = exp(-1.0 / net->hp.tau_trace_k[k]);
            if (net->hp.trace_alpha[k] < 0.0) net->hp.trace_alpha[k] = 0.0;
        }
        /* alpha 必须归一到 1：否则 K=1 时沿用默认 1/3，信用被暗削 3 倍 */
        double asum = 0.0;
        for (k = 0; k < net->hp.trace_k; k++) asum += net->hp.trace_alpha[k];
        if (asum <= 0.0 || fabs(asum - 1.0) > 1e-6) {
            if (asum <= 0.0) {
                for (k = 0; k < net->hp.trace_k; k++)
                    net->hp.trace_alpha[k] = 1.0 / (double)net->hp.trace_k;
            } else {
                for (k = 0; k < net->hp.trace_k; k++)
                    net->hp.trace_alpha[k] /= asum;
            }
        }
    }

    /* 重新分配奖励历史缓冲（如果窗口变了） */
    if (net->reward_history) free(net->reward_history);
    net->reward_history = (double*)calloc(net->hp.reward_baseline_window, sizeof(double));
    net->reward_hist_head = 0;
    net->reward_hist_count = 0;
    net->reward_baseline = 0.0;
}

/* ============================================================
 * 膜电位更新 + 发放判定
 *
 * 正确的生物神经元时序：
 *   Tick N 设置 is_spiking=1 → Tick N+1 开始时用它累加突触电流 → 然后清零 → 衰减 → 判定新发放
 * ============================================================ */
static void update_neurons(NetworkState* net) {
    int i, s;
    
    /* Step1: 用上一个tick遗留的 is_spiking 累加突触电流（不要先清零！） */
    for (s = 0; s < net->synapse_count; s++) {
        Neuron* pre = &net->neurons[net->synapses[s].pre_id];
        Neuron* post = &net->neurons[net->synapses[s].post_id];
        
        if (pre->is_spiking) {
            post->membrane_potential += net->synapses[s].weight * net->hp.psp_factor;
        }
    }
    
    /* Step2: 现在清零 is_spiking，准备判定本tick发放 */
    for (i = 0; i < net->neuron_count; i++) {
        net->neurons[i].is_spiking = 0;
    }
    
    /* Step3: 神经元膜电位衰减 + 发放判定 */
    for (i = 0; i < net->neuron_count; i++) {
        Neuron* n = &net->neurons[i];
        
        /* 不应期处理 */
        if (n->refractory_remaining > 0) {
            n->refractory_remaining--;
            n->membrane_potential = MEMBRANE_REST;
            continue;
        }
        
        /* 膜电位衰减 */
        n->membrane_potential = n->membrane_potential * MEMBRANE_DECAY + MEMBRANE_REST * (1 - MEMBRANE_DECAY);
        
        /* 发放判定 */
        if (n->membrane_potential >= n->spike_threshold) {
            n->is_spiking = 1;
            n->last_spike_tick = net->current_tick;
            n->refractory_remaining = REFRACTORY_TICKS;
            n->membrane_potential = MEMBRANE_REST;
        }
    }
}

/* ============================================================
 * 输出层竞争：本 tick 发放多的一侧抑制另一侧。
 * 没有竞争时 A/B 会同时持续发放，投票只反映初始随机权重，
 * 奖励调制无法形成可测偏好（M1 诊断结论）。
 * ============================================================ */
static void apply_output_competition(NetworkState* net) {
    int i, a_cnt = 0, b_cnt = 0;
    const double inh = 6.0;
    const double floor_v = MEMBRANE_REST - 20.0;

    for (i = 0; i < net->neuron_count; i++) {
        if (!net->neurons[i].is_spiking) continue;
        if (net->neurons[i].type == NT_OUTPUT_A) a_cnt++;
        else if (net->neurons[i].type == NT_OUTPUT_B) b_cnt++;
    }
    if (a_cnt == b_cnt) return;

    for (i = 0; i < net->neuron_count; i++) {
        Neuron* n = &net->neurons[i];
        if (n->is_spiking) continue;
        if (a_cnt > b_cnt && n->type == NT_OUTPUT_B) {
            n->membrane_potential -= inh;
            if (n->membrane_potential < floor_v) n->membrane_potential = floor_v;
        } else if (b_cnt > a_cnt && n->type == NT_OUTPUT_A) {
            n->membrane_potential -= inh;
            if (n->membrane_potential < floor_v) n->membrane_potential = floor_v;
        }
    }
}

/* ============================================================
 * STDP核计算
 * ============================================================ */
static double compute_stdp_kernel(NetworkState* net, double delta_t) {
    if (delta_t > 0) {
        /* pre先发放 → post后发放：LTP倾向 */
        return net->hp.stdp_a_plus * exp(-delta_t / net->hp.tau_stdp);
    } else if (delta_t < 0) {
        /* post先发放 → pre后发放：LTD倾向 */
        return -net->hp.stdp_a_minus * exp(delta_t / net->hp.tau_stdp);
    }
    return 0.0;
}

/* ============================================================
 * 惰性结算：把总迹推进到当前 tick（奖励到达/诊断时调用）
 * ============================================================ */
#if ELIGIBILITY_TRACE_ENABLE
static double synapse_total_credit(NetworkState* net, Synapse* syn) {
    int k;
    int delta_tick = net->current_tick - syn->last_stdp_tick;
    if (delta_tick > 0) {
        for (k = 0; k < net->hp.trace_k; k++)
            syn->e[k] *= pow(net->hp.lambda_trace_k[k], (double)delta_tick);
        syn->last_stdp_tick = net->current_tick;
    }
    double total = 0.0;
    for (k = 0; k < net->hp.trace_k; k++)
        total += net->hp.trace_alpha[k] * syn->e[k];
    syn->eligibility_trace = total;
    return total;
}
#endif

/* ============================================================
 * 资格迹更新（事件驱动，仅处理有发放的突触）
 * ============================================================ */
static void update_eligibility_traces(NetworkState* net) {
    int s;
    
    for (s = 0; s < net->synapse_count; s++) {
        Synapse* syn = &net->synapses[s];
        Neuron* pre = &net->neurons[syn->pre_id];
        Neuron* post = &net->neurons[syn->post_id];
        
        /* 仅当突触前或突触后神经元当前tick发放时处理（事件驱动） */
        if (!pre->is_spiking && !post->is_spiking) continue;
        
        /* 多尺度惰性衰减：每个尺度按自己的 tau 批量结算 */
        int delta_tick = net->current_tick - syn->last_stdp_tick;
        if (delta_tick > 1) {
            int kk;
            for (kk = 0; kk < net->hp.trace_k; kk++)
                syn->e[kk] *= pow(net->hp.lambda_trace_k[kk], (double)delta_tick - 1.0);
        }
        syn->last_stdp_tick = net->current_tick;
        
        double stdp_val = 0.0;
        
        if (pre->is_spiking && post->last_spike_tick != -1 && post->last_spike_tick < net->current_tick) {
            /* pre刚发放，post之前发过 → Δt = post_last - now（负数，LTD或弱LTP视时序） */
            int delta_t = net->current_tick - post->last_spike_tick; /* 正数：pre在post之后 */
            stdp_val = compute_stdp_kernel(net, delta_t);
        }
        
        if (post->is_spiking && pre->last_spike_tick != -1 && pre->last_spike_tick < net->current_tick) {
            /* post刚发放，pre之前发过 → Δt = post_now - pre_last（正数，LTP倾向） */
            int delta_t = net->current_tick - pre->last_spike_tick;
            stdp_val += compute_stdp_kernel(net, delta_t);
        }
        
        #if ELIGIBILITY_TRACE_ENABLE
        if (net->use_eligibility_trace) {
            int kk;
            for (kk = 0; kk < net->hp.trace_k; kk++) {
                syn->e[kk] += stdp_val;
                if (syn->e[kk] > 5.0) syn->e[kk] = 5.0;
                if (syn->e[kk] < -5.0) syn->e[kk] = -5.0;
            }
            /* 切换 K 后清掉多余尺度，避免旧迹残留 */
            for (kk = net->hp.trace_k; kk < 3; kk++) syn->e[kk] = 0.0;

            double total = 0.0;
            for (kk = 0; kk < net->hp.trace_k; kk++)
                total += net->hp.trace_alpha[kk] * syn->e[kk];
            syn->eligibility_trace = total;
        } else {
            /* 消融：不累积资格迹 */
            int kk;
            for (kk = 0; kk < 3; kk++) syn->e[kk] = 0.0;
            syn->eligibility_trace = 0.0;
        }
        #else
        {
            int kk;
            for (kk = 0; kk < 3; kk++) syn->e[kk] = 0.0;
        }
        syn->eligibility_trace = 0.0;
        #endif
    }
}

/* ============================================================
 * 奖励基线更新（滑动平均）
 * ============================================================ */
static void update_reward_baseline(NetworkState* net, double reward) {
    net->reward_history[net->reward_hist_head] = reward;
    net->reward_hist_head = (net->reward_hist_head + 1) % net->hp.reward_baseline_window;
    
    if (net->reward_hist_count < net->hp.reward_baseline_window) {
        net->reward_hist_count++;
    }
    
    /* 计算滑动平均 */
    double sum = 0.0;
    int i;
    for (i = 0; i < net->reward_hist_count; i++) {
        sum += net->reward_history[i];
    }
    if (net->reward_hist_count > 0) {
        net->reward_baseline = sum / net->reward_hist_count;
    }
}

/* ============================================================
 * 奖励队列处理 → R-STDP权重更新
 * ============================================================ */
static void process_reward_queue(NetworkState* net) {
    /* 处理所有到达当前tick的奖励事件 */
    while (net->queue_head != net->queue_tail && 
           net->reward_queue[net->queue_head].delivery_tick <= net->current_tick) {
        
        RewardEvent event = net->reward_queue[net->queue_head];
        net->queue_head = (net->queue_head + 1) % QUEUE_SIZE;
        
        /* 跳过无奖励事件 */
        if (event.reward_scalar == 0.0) {
            update_reward_baseline(net, event.reward_scalar);
            continue;
        }
        
        /* 计算奖励预测误差 */
        double reward_error = event.reward_scalar - net->reward_baseline;
        
        #if REWARD_MODULATION_ENABLE
        if (net->use_reward_modulation && fabs(reward_error) > 0.01) {
            int s;
            for (s = 0; s < net->synapse_count; s++) {
                Synapse* syn = &net->synapses[s];

                /* 只把信用分配给实际被选中的输出输入突触。
                 * 全局奖励会同时增强正确与错误通路，导致权重整体缩放而非偏好形成。 */
                if (event.winner != 0) {
                    NeuronType pt = net->neurons[syn->post_id].type;
                    int match = (event.winner == 1 && pt == NT_OUTPUT_A) ||
                                (event.winner == 2 && pt == NT_OUTPUT_B);
                    if (!match) continue;
                }

                double credit = 0.0;
                
                #if ELIGIBILITY_TRACE_ENABLE
                if (net->use_eligibility_trace) {
                    /* 奖励到达时把多尺度迹结算到当前 tick，再取加权和 */
                    credit = synapse_total_credit(net, syn);
                } else {
                    /* 消融：资格迹关闭 → 无法进行延迟信用分配，权重不更新 */
                    credit = 0.0;
                }
                #endif
                
                if (fabs(credit) > 1e-8) {
                    double delta_w = net->hp.learning_rate * reward_error * credit;
                    syn->weight = clamp(syn->weight + delta_w, WEIGHT_MIN, WEIGHT_MAX);
                }
            }
        }
        #endif
        
        update_reward_baseline(net, event.reward_scalar);
    }
}

/* ============================================================
 * 权重软遗忘（惰性衰减：仅在被访问突触上批量计算）
 * ============================================================ */
static void lazy_decay_weights(NetworkState* net) {
    (void)net;  /* 当前实现不访问网络状态，显式标注以避免 -Wunused-parameter */
    /* Benchmark-1 简化：不在tick循环中做全局扫描
     * 遗忘机制主要由资格迹截断（±5.0）和权重边界 clamp 提供自然约束
     * 完整遗忘在后续基准中实现 */
}

/* ============================================================
 * 核心Tick循环
 * ============================================================ */
void net_tick(NetworkState* net) {
    net->current_tick++;
    
    /* Step1: 膜电位更新 + 发放判定 */
    update_neurons(net);

    /* Step1b: 输出层竞争（winner-take-all 侧抑制） */
    apply_output_competition(net);
    
    /* Step2: 资格迹更新（事件驱动STDP） */
    update_eligibility_traces(net);
    
    /* Step3: 奖励队列处理 → R-STDP权重更新 */
    process_reward_queue(net);
    
    /* Step4: 惰性衰减（Benchmark-1留空） */
    lazy_decay_weights(net);
}

/* ============================================================
 * 刺激注入
 * ============================================================ */
void net_inject_stimulus_a(NetworkState* net) {
    int i;
    int count = 0;
    for (i = 0; i < net->neuron_count && count < net->sensory_a_count; i++) {
        if (net->neurons[i].type == NT_SENSORY_A) {
            /* 强制提高膜电位使其在下一个tick发放 */
            net->neurons[i].membrane_potential = SPIKE_THRESHOLD + 10.0;
            count++;
        }
    }
}

void net_inject_stimulus_b(NetworkState* net) {
    int i;
    int count = 0;
    for (i = 0; i < net->neuron_count && count < net->sensory_b_count; i++) {
        if (net->neurons[i].type == NT_SENSORY_B) {
            net->neurons[i].membrane_potential = SPIKE_THRESHOLD + 10.0;
            count++;
        }
    }
}

void net_clear_stimulus(NetworkState* net) {
    int i;
    for (i = 0; i < net->neuron_count; i++) {
        Neuron* n = &net->neurons[i];
        if (n->type == NT_SENSORY_A || n->type == NT_SENSORY_B) {
            n->membrane_potential = MEMBRANE_REST;
            n->is_spiking = 0;
        }
    }
}

/* ============================================================
 * 奖励注入（写入队列）
 * ============================================================ */
void net_inject_reward_vote(NetworkState* net, double reward, int delay_ticks, int winner) {
    /* 检查队列空间 */
    int next_tail = (net->queue_tail + 1) % QUEUE_SIZE;
    if (next_tail == net->queue_head) {
        fprintf(stderr, "Warning: reward queue full, dropping reward at tick %d\n", net->current_tick);
        return;
    }
    
    net->reward_queue[net->queue_tail].reward_scalar = reward;
    net->reward_queue[net->queue_tail].delivery_tick = net->current_tick + delay_ticks;
    net->reward_queue[net->queue_tail].winner = winner;
    net->queue_tail = next_tail;
}

void net_inject_reward(NetworkState* net, double reward, int delay_ticks) {
    net_inject_reward_vote(net, reward, delay_ticks, 0);
}

/* ============================================================
 * 输出读取：投票判断偏好
 * ============================================================ */
int net_read_output_vote(NetworkState* net) {
    /* 统计最近一段窗口内的发放率（简化：当前tick+最近10 ticks的累积） */
    /* Benchmark-1 简化：直接看当前tick发放的输出神经元 */
    int a_count = 0, b_count = 0;
    int i;
    
    for (i = 0; i < net->neuron_count; i++) {
        Neuron* n = &net->neurons[i];
        if (n->type == NT_OUTPUT_A && n->is_spiking) a_count++;
        if (n->type == NT_OUTPUT_B && n->is_spiking) b_count++;
    }
    
    /* 如果当前tick没发放，用最近发放记录（简化：检查last_spike_tick在最近20 ticks内的） */
    if (a_count == 0 && b_count == 0) {
        for (i = 0; i < net->neuron_count; i++) {
            Neuron* n = &net->neurons[i];
            int since_last = net->current_tick - n->last_spike_tick;
            if (since_last >= 0 && since_last <= 20) {
                if (n->type == NT_OUTPUT_A) a_count++;
                if (n->type == NT_OUTPUT_B) b_count++;
            }
        }
    }
    
    if (a_count > b_count) return 1;  /* 偏好A */
    if (b_count > a_count) return 2;  /* 偏好B */
    return 0;  /* 平局 */
}

double net_read_output_a_rate(NetworkState* net) {
    int count = 0, total = 0;
    int i;
    for (i = 0; i < net->neuron_count; i++) {
        if (net->neurons[i].type == NT_OUTPUT_A) {
            total++;
            if (net->neurons[i].is_spiking) count++;
        }
    }
    return total > 0 ? (double)count / total : 0.0;
}

double net_read_output_b_rate(NetworkState* net) {
    int count = 0, total = 0;
    int i;
    for (i = 0; i < net->neuron_count; i++) {
        if (net->neurons[i].type == NT_OUTPUT_B) {
            total++;
            if (net->neurons[i].is_spiking) count++;
        }
    }
    return total > 0 ? (double)count / total : 0.0;
}

/* ============================================================
 * 诊断接口
 * ============================================================ */
static int _cmp_abs_weight(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

void net_dump_weight_stats(NetworkState* net, FILE* fp) {
    int n = net->synapse_count, s;
    double w_sum = 0, w_sq = 0, w_min = 1e9, w_max = -1e9;
    double* sorted = n > 0 ? (double*)malloc(sizeof(double) * (size_t)n) : NULL;
    for (s = 0; s < n; s++) {
        double w = net->synapses[s].weight;
        double a = fabs(w);
        w_sum += a;
        w_sq += a * a;
        if (w < w_min) w_min = w;
        if (w > w_max) w_max = w;
        if (sorted) sorted[s] = a;
    }
    fprintf(fp, "Weight stats: count=%d mean=%.4f min=%.4f max=%.4f\n",
            n, n > 0 ? w_sum / n : 0.0, w_min, w_max);

    double gini = 0.0, effective = 0.0, mean = n > 0 ? w_sum / n : 0.0;
    double p10 = 0.0, p50 = 0.0, p90 = 0.0;
    int active = 0, sat_hi = 0, sat_lo = 0;
    if (sorted && n > 0) {
        qsort(sorted, (size_t)n, sizeof(double), _cmp_abs_weight);
        double weighted = 0.0;
        for (s = 0; s < n; s++) {
            weighted += (double)(s + 1) * sorted[s];
            if (sorted[s] > 0.5 * mean) active++;
            if (sorted[s] >= 0.99 * WEIGHT_MAX) sat_hi++;
            if (sorted[s] <= 1.01 * WEIGHT_MIN) sat_lo++;
        }
        if (w_sum > 0.0) {
            gini = 2.0 * weighted / ((double)n * w_sum) - ((double)n + 1.0) / (double)n;
            effective = w_sum * w_sum / (w_sq > 0.0 ? w_sq : 1.0);
        }
        p10 = sorted[(n - 1) / 10];
        p50 = sorted[(n - 1) / 2];
        p90 = sorted[(n - 1) * 9 / 10];
    }
    fprintf(fp, "Weight sparsity: gini=%.4f effective=%.1f active=%d/%d "
                 "p10=%.4f p50=%.4f p90=%.4f sat_hi=%d sat_lo=%d\n",
            gini, effective, active, n, p10, p50, p90, sat_hi, sat_lo);
    if (sorted) free(sorted);
}

void net_dump_trace_stats(NetworkState* net, FILE* fp) {
    double t_sum = 0, t_min = 1e9, t_max = -1e9;
    int nonzero = 0;
    int s;
    for (s = 0; s < net->synapse_count; s++) {
        double t = net->synapses[s].eligibility_trace;
        if (fabs(t) > 1e-8) {
            nonzero++;
            t_sum += t;
            if (t < t_min) t_min = t;
            if (t > t_max) t_max = t;
        }
    }
    fprintf(fp, "Trace stats: K=%d nonzero=%d/%d mean=%.4f min=%.4f max=%.4f\n",
            net->hp.trace_k, nonzero, net->synapse_count,
            nonzero > 0 ? t_sum / nonzero : 0.0,
            nonzero > 0 ? t_min : 0.0,
            nonzero > 0 ? t_max : 0.0);

    /* 逐尺度统计：用于验证多尺度确实在各自时间窗内保留信用 */
    {
        int k;
        for (k = 0; k < net->hp.trace_k; k++) {
            int nz = 0;
            double sum = 0.0;
            for (s = 0; s < net->synapse_count; s++) {
                double v = net->synapses[s].e[k];
                if (fabs(v) > 1e-8) { nz++; sum += v; }
            }
            fprintf(fp, "  scale[%d]: tau=%.1f alpha=%.4f lambda=%.6f nonzero=%d mean=%.4f\n",
                    k, net->hp.tau_trace_k[k], net->hp.trace_alpha[k],
                    net->hp.lambda_trace_k[k], nz,
                    nz > 0 ? sum / nz : 0.0);
        }
    }
}

void net_dump_spiking_counts(NetworkState* net, FILE* fp) {
    int a_spikes = 0, b_spikes = 0, hidden_spikes = 0;
    int i;
    for (i = 0; i < net->neuron_count; i++) {
        if (net->neurons[i].is_spiking) {
            if (net->neurons[i].type == NT_OUTPUT_A) a_spikes++;
            else if (net->neurons[i].type == NT_OUTPUT_B) b_spikes++;
            else hidden_spikes++;
        }
    }
    fprintf(fp, "Tick %d spiking: out_a=%d out_b=%d other=%d\n",
            net->current_tick, a_spikes, b_spikes, hidden_spikes);
}
