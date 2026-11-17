/*
 * benchmark1_core.h - Benchmark-1 核心头文件
 * 拟生物神经元系统 · 单突触远端信用分配验证
 *
 * 冻结规则：所有结构体字段和宏开关一旦确定，不得修改
 * 消融开关通过宏定义控制：编译时 -D ELIGIBILITY_TRACE_ENABLE 或 -D REWARD_MODULATION_ENABLE
 */

#ifndef BENCHMARK1_CORE_H
#define BENCHMARK1_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ============================================================
 * 消融开关（编译时通过 -D 定义，默认全部启用）
 * ============================================================ */
#ifndef ELIGIBILITY_TRACE_ENABLE
#define ELIGIBILITY_TRACE_ENABLE 1   /* 资格迹信用分配：1=启用，0=消融 */
#endif

#ifndef REWARD_MODULATION_ENABLE
#define REWARD_MODULATION_ENABLE 1   /* 奖励调制学习：1=启用，0=消融 */
#endif

/* ============================================================
 * 常量定义（冻结默认值）
 * ============================================================ */
#define QUEUE_SIZE          4096      /* 奖励事件队列大小 */
#define WEIGHT_MIN          0.001     /* 权重下界 */
#define WEIGHT_MAX          5.0       /* 权重上界（放宽以允许大奖励调制） */
#define MEMBRANE_REST       -65.0     /* 静息膜电位 (mV) */
#define SPIKE_THRESHOLD     -50.0     /* 发放阈值 (mV) */
#define MEMBRANE_DECAY      0.95      /* 膜电位衰减因子 */
#define REFRACTORY_TICKS    3         /* 不应期持续tick数 */
#define PSP_FACTOR          25.0      /* 突触后电位幅值因子 */

/* ============================================================
 * 神经元类型（Benchmark-1 最小集合）
 * ============================================================ */
typedef enum {
    NT_SENSORY_A = 0,    /* 感觉神经元A（接收刺激A） */
    NT_SENSORY_B = 1,    /* 感觉神经元B（接收刺激B） */
    NT_HIDDEN    = 2,    /* 中间处理神经元 */
    NT_OUTPUT_A  = 3,    /* 输出神经元A（代表偏好A） */
    NT_OUTPUT_B  = 4,    /* 输出神经元B（代表偏好B） */
    NT_MODULATION = 5    /* 调制神经元（接收奖励信号） */
} NeuronType;

/* ============================================================
 * 神经元结构
 * ============================================================ */
typedef struct {
    int id;                     /* 神经元唯一ID */
    NeuronType type;            /* 类型 */
    double membrane_potential;  /* 膜电位 (mV) */
    int is_spiking;             /* 当前tick是否发放（1=是，0=否） */
    int last_spike_tick;        /* 上次发放的tick（-1=从未发放） */
    int refractory_remaining;   /* 剩余不应期tick数 */
    double spike_threshold;     /* 发放阈值（默认SPIKE_THRESHOLD） */
} Neuron;

/* ============================================================
 * 突触结构
 * ============================================================ */
typedef struct {
    int pre_id;                 /* 突触前神经元ID */
    int post_id;                /* 突触后神经元ID */
    double weight;              /* 突触权重 */
    double eligibility_trace;   /* 总迹 = Σ α_k e[k]（诊断/兼容用） */
    double e[3];                /* 多尺度资格迹，K<=3（M1） */
    int last_stdp_tick;        /* 上次STDP更新的tick（惰性衰减用） */
} Synapse;

/* ============================================================
 * 奖励事件队列（环形缓冲）
 * ============================================================ */
typedef struct {
    double reward_scalar;       /* 奖励标量（+1.0奖励/-1.0惩罚/0.0无奖励） */
    int delivery_tick;          /* 奖励到达tick */
    int winner;                 /* 0=全部突触, 1=只更新OUTPUT_A输入, 2=只更新OUTPUT_B输入 */
} RewardEvent;

/* ============================================================
 * 超参数配置（可在初始化时设置，运行时冻结）
 * ============================================================ */
typedef struct {
    double tau_stdp;            /* STDP时间常数 (ticks)，默认20 */
    double tau_trace;           /* 单尺度回退时间常数 (ticks)，默认200 */
    double lambda_trace;        /* = exp(-1/tau_trace) */
    double stdp_a_plus;         /* LTP幅值，默认0.1 */
    double stdp_a_minus;        /* LTD幅值，默认0.12 */
    double learning_rate;       /* 奖励调制学习率，默认0.05 */
    double psp_factor;          /* 突触后电位幅值因子（B1 校准项，默认3.0） */
    double weight_decay_lambda; /* 权重软遗忘衰减率，默认1e-5 */
    int    reward_baseline_window; /* 奖励预测误差基线窗口 (ticks)，默认100 */

    /* 多尺度资格迹（M1）：总迹 = Σ α_k e[k]，k < trace_k */
    int    trace_k;             /* 1..3，默认3；1=单尺度 */
    double tau_trace_k[3];      /* 各尺度时间常数，默认 20/100/500 */
    double lambda_trace_k[3];   /* 各尺度衰减因子 */
    double trace_alpha[3];      /* 各尺度权重，默认 1/K */
} HyperParams;

/* ============================================================
 * 网络全局状态
 * ============================================================ */
typedef struct {
    Neuron* neurons;
    int neuron_count;
    int sensory_a_count;        /* 感觉A神经元数量 */
    int sensory_b_count;        /* 感觉B神经元数量 */
    int hidden_count;           /* 中间神经元数量 */
    int output_a_count;         /* 输出A神经元数量 */
    int output_b_count;         /* 输出B神经元数量 */
    
    Synapse* synapses;
    int synapse_count;
    
    RewardEvent reward_queue[QUEUE_SIZE];
    int queue_head;
    int queue_tail;
    
    int current_tick;
    
    HyperParams hp;
    
    /* 奖励预测误差基线 */
    double reward_baseline;
    double* reward_history;     /* 奖励历史环形缓冲 */
    int reward_hist_head;
    int reward_hist_count;
    
    /* 消融开关（运行时读取宏定义） */
    int use_eligibility_trace;  /* ELIGIBILITY_TRACE_ENABLE */
    int use_reward_modulation;  /* REWARD_MODULATION_ENABLE */
} NetworkState;

/* ============================================================
 * 函数接口
 * ============================================================ */

/* 初始化与销毁 */
int  net_init(NetworkState* net, int sensory_a, int sensory_b, 
              int hidden, int output_a, int output_b,
              double avg_connections_per_neuron, unsigned int seed);
void net_destroy(NetworkState* net);

/* 超参数设置（可选，使用默认值时跳过） */
void net_set_hyperparams(NetworkState* net, HyperParams* hp);

/* 核心tick循环 */
void net_tick(NetworkState* net);

/* 刺激注入（宿主调用） */
void net_inject_stimulus_a(NetworkState* net);   /* 触发SENSORY_A发放 */
void net_inject_stimulus_b(NetworkState* net);   /* 触发SENSORY_B发放 */
void net_clear_stimulus(NetworkState* net);      /* 清除所有感觉神经元激活 */

/* 奖励注入（宿主调用） */
void net_inject_reward(NetworkState* net, double reward, int delay_ticks);
/* winner: 0=全部突触（旧行为）, 1=OUTPUT_A 输入, 2=OUTPUT_B 输入 */
void net_inject_reward_vote(NetworkState* net, double reward, int delay_ticks, int winner);

/* 输出读取（宿主调用） */
int  net_read_output_vote(NetworkState* net);     /* 返回: 1=偏好A, 2=偏好B, 0=平局 */
double net_read_output_a_rate(NetworkState* net); /* OUTPUT_A发放率 */
double net_read_output_b_rate(NetworkState* net); /* OUTPUT_B发放率 */

/* 诊断接口（宿主调用，用于日志） */
void net_dump_weight_stats(NetworkState* net, FILE* fp);
void net_dump_trace_stats(NetworkState* net, FILE* fp);
void net_dump_spiking_counts(NetworkState* net, FILE* fp);

/* M0 seed_bundle：旧单流复现开关（仅供历史数据证据链，默认关闭） */
void net_use_legacy_rng(int on);

/* 单变量诊断：在 net_init 之前覆盖某一条流，取值 0=structure,1=weight,
 * 2=noise,3=trial_order（与 core/seed_bundle.h 的 NeuroStream 一致）。 */
void net_override_seed_stream(int which, uint64_t value);

/* 输出层初始偏向诊断：out = [left->A, left->B, right->A, right->B] 权重和。
 * 只读，不改变网络状态，不参与判据计算。 */
void net_output_bias(NetworkState* net, double out[4]);

#endif /* BENCHMARK1_CORE_H */
