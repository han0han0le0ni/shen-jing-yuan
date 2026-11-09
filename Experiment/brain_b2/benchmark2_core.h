/*
 * benchmark2_core.h - Benchmark-2 核心头文件
 * 两关节虚拟臂到达 · 群体编码/解码
 *
 * 延续 B1 的：tick 循环 / STDP 核 / 资格迹 / 奖励队列 / R-STDP
 * 新增：2D 高斯群体编码 / 群体向量解码 / 关节物理模拟
 *
 * 编译：g++ -O2 -D ELIGIBILITY_TRACE_ENABLE=1 -D REWARD_MODULATION_ENABLE=1
 */

#ifndef BENCHMARK2_CORE_H
#define BENCHMARK2_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * 消融开关（与 B1 相同）
 * ============================================================ */
#ifndef ELIGIBILITY_TRACE_ENABLE
#define ELIGIBILITY_TRACE_ENABLE 1
#endif
#ifndef REWARD_MODULATION_ENABLE
#define REWARD_MODULATION_ENABLE 1
#endif

/* ============================================================
 * 常量
 * ============================================================ */
#define QUEUE_SIZE          8192
#define WEIGHT_MIN          0.001
#define WEIGHT_MAX          5.0
#define MEMBRANE_REST       -65.0
#define SPIKE_THRESHOLD     -50.0
#define MEMBRANE_DECAY      0.95
#define REFRACTORY_TICKS    3
#define PSP_FACTOR          25.0

/* 两关节臂参数 */
#define ARM_L1              1.0     /* 上臂长度 */
#define ARM_L2              1.0     /* 前臂长度 */
#define ARM_THETA1_MIN      (-M_PI)
#define ARM_THETA1_MAX      (M_PI)
#define ARM_THETA2_MIN      (-M_PI * 0.9)
#define ARM_THETA2_MAX      (M_PI * 0.9)
#define ARM_DT              0.02    /* 物理模拟时间步长（减小防止飞远） */

/* 群体编码参数 */
#define ENCODE_SIGMA_POS    0.85    /* 位置编码高斯 sigma（覆盖网格间距） */
#define DECODE_SIGMA_VEL    0.25    /* 速度解码高斯 sigma */

/* ============================================================
 * 神经元类型
 * ============================================================ */
typedef enum {
    NT_INPUT_X    = 0,   /* 位置编码神经元（2D 偏好点） */
    NT_HIDDEN     = 1,   /* 中间处理神经元 */
    NT_MOTOR_OMEGA1 = 2, /* 关节1角速度控制神经元 */
    NT_MOTOR_OMEGA2 = 3, /* 关节2角速度控制神经元 */
} NeuronType2;

/* ============================================================
 * 神经元（带偏好向量）
 * ============================================================ */
typedef struct {
    int id;
    NeuronType2 type;
    double membrane_potential;
    int is_spiking;
    int last_spike_tick;
    int refractory_remaining;
    double spike_threshold;
    
    /* 偏好参数（群体编码/解码用） */
    double pref_x;        /* 偏好位置 x（INPUT_X 用） */
    double pref_y;        /* 偏好位置 y（INPUT_X 用） */
    double pref_omega;    /* 偏好角速度（MOTOR 用，范围 -1~1） */
} Neuron2;

/* ============================================================
 * 突触
 * ============================================================ */
typedef struct {
    int pre_id;
    int post_id;
    double weight;
    double eligibility_trace;
    int last_stdp_tick;
} Synapse2;

/* ============================================================
 * 奖励事件队列
 * ============================================================ */
typedef struct {
    double reward_scalar;
    int delivery_tick;
} RewardEvent2;

/* ============================================================
 * 超参数
 * ============================================================ */
typedef struct {
    double tau_stdp;
    double tau_trace;
    double lambda_trace;
    double stdp_a_plus;
    double stdp_a_minus;
    double learning_rate;
    double weight_decay_lambda;
    int    reward_baseline_window;
} HyperParams2;

/* ============================================================
 * 两关节臂状态
 * ============================================================ */
typedef struct {
    double theta1;       /* 关节1角度 */
    double theta2;       /* 关节2角度 */
    double omega1;       /* 关节1角速度 */
    double omega2;       /* 关节2角速度 */
    double end_x;        /* 末端位置 x */
    double end_y;        /* 末端位置 y */
    double target_x;     /* 目标位置 x */
    double target_y;     /* 目标位置 y */
    double max_reach;    /* 最大伸展距离 */
    double reward;       /* 当前步奖励 */
} ArmState;

/* ============================================================
 * 网络全局状态
 * ============================================================ */
typedef struct {
    Neuron2* neurons;
    int neuron_count;
    int input_count;
    int hidden_count;
    int motor1_count;
    int motor2_count;
    
    Synapse2* synapses;
    int synapse_count;
    
    RewardEvent2 reward_queue[QUEUE_SIZE];
    int queue_head;
    int queue_tail;
    
    int current_tick;
    
    HyperParams2 hp;
    ArmState arm;
    
    double reward_baseline;
    double* reward_history;
    int reward_hist_head;
    int reward_hist_count;
    
    int use_eligibility_trace;
    int use_reward_modulation;
} NetworkState2;

/* ============================================================
 * 函数接口
 * ============================================================ */

/* 初始化与销毁 */
int  net2_init(NetworkState2* net, int input_n, int hidden_n,
               int motor1_n, int motor2_n, double avg_conn,
               unsigned int seed);
void net2_destroy(NetworkState2* net);

/* 核心 tick 循环（含物理模拟） */
void net2_tick(NetworkState2* net);

/* 目标控制（宿主调用） */
void net2_set_target(NetworkState2* net, double tx, double ty);

/* 奖励注入 */
void net2_inject_reward(NetworkState2* net, double reward, int delay_ticks);

/* 物理步进（宿主可选手动控制） */
void arm_step(ArmState* arm, double omega1_cmd, double omega2_cmd);
void arm_set_target(ArmState* arm, double tx, double ty);
double arm_reward(const ArmState* arm);

/* 诊断 */
void net2_dump_stats(NetworkState2* net, FILE* fp);

#endif /* BENCHMARK2_CORE_H */
