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

/* MSVC 默认不提供 M_PI；这里兜底定义，保证跨编译器可用 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
#define WEIGHT_MAX          10.0
#define MEMBRANE_REST       -65.0
#define SPIKE_THRESHOLD     -55.0
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
#define ENCODE_SIGMA_POS    2.0
#define DECODE_SIGMA_VEL    0.25    /* 速度解码高斯 sigma */
#define ENCODE_SIGMA_PROP   0.6     /* 本体感觉角度编码 sigma (rad) */
#define PROP_N              56      /* 12×θ1 + 12×θ2 + 16×ψ1 + 16×ψ2 (段位方位角特征) */

/* ============================================================
 * 神经元类型
 * ============================================================ */
typedef enum {
    NT_INPUT_X    = 0,   /* 位置编码神经元（2D 偏好点） */
    NT_HIDDEN     = 1,   /* 中间处理神经元 */
    NT_MOTOR_OMEGA1 = 2, /* 关节1角速度控制神经元 */
    NT_MOTOR_OMEGA2 = 3, /* 关节2角速度控制神经元 */
    NT_INPUT_PROP = 4,   /* 本体感觉神经元（编码关节角度 θ1/θ2） */
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

    /* v6 异构群体编码：每神经元独立的膜时间常数与调谐宽度。
     * net2_init 统一写入默认 = 全局常量，保证旧实验逐位可复现；
     * 只有显式打开 hetero_* 参数时运动层才会被扰动。 */
    double membrane_decay;   /* 默认 MEMBRANE_DECAY */
    double tuning_width;     /* 默认 1.0；decode_mode=2 时作为解码加权系数 */
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
 * 调制事件队列 —— v6 三因子消息格式
 *   第一因子：Hebbian 共激活（STDP 核，在资格迹里）
 *   第二因子：资格迹 eligibility_trace
 *   第三因子：全局调质，拆成三类语境通道
 *     r1/r2        奖励预测误差（类多巴胺），按关节分通道
 *     e1/e2        感觉/运动误差（类小脑攀爬纤维），逐 tick 连续有符号
 *     novelty      惊讶/新颖性（类乙酰胆碱），瞬时放大学习率
 * 旧版只有 r1/r2；新增字段默认 0，旧调用方行为完全不变。
 * ============================================================ */
typedef struct {
    double reward_scalar;   /* 兼容字段 = (r1+r2)/2，用于 baseline */
    double r1;              /* 关节1 专用奖励 */
    double r2;              /* 关节2 专用奖励 */
    double e1;              /* 关节1 感觉预测误差：(omega*_1 - omega_1) - EMA */
    double e2;              /* 关节2 感觉预测误差：(omega*_2 - omega_2) - EMA */
    double novelty;         /* |reward - baseline| 归一化后的惊讶度 */
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
    /* 运行时可调（默认值与旧版完全一致，保证历史实验可比） */
    double psp_factor;             /* 突触电流增益，默认 25.0 */
    double trace_clamp;            /* 资格迹截断幅度，默认 1.0；<=0 表示不截断 */
    int    decode_mode;            /* 0=WTA-K 均值(默认), 2=按 winner 膜电位加权 */
    int    wta_k;                  /* MOTOR 层 winner 数，默认 3 */
    int    use_reward_baseline;    /* 1=减滑动基线(默认), 0=使用原始奖励误差 */

    /* ---- v6：三因子调制 / 攀爬纤维误差 / 异构群体编码 ----
     * 全部默认 0（cf_lambda=1），即旧行为；A/B 时只打开需要的通道。 */
    double error_gain;       /* 运动误差通道增益（攀爬纤维 → 定向 LTD） */
    double novelty_gain;     /* 惊讶度通道增益：瞬时放大学习率 */
    double cf_lambda;        /* 逆运动学期望速度缩放，默认 1.0 */
    double hetero_tau;       /* 运动层膜衰减对数正态 sigma，0=关闭 */
    double hetero_thresh;    /* 运动层发放阈值高斯 sigma (mV)，0=关闭 */
    double hetero_tune;      /* 运动层调谐宽度对数正态 sigma，0=关闭 */
    /* v7（M2 探针）：本体感觉输入模式。
     * 0 = full（56 维：θ1/θ2 + 目标方位差 tag2/tag3，默认，与历史实验一致）
     * 1 = raw （只给 θ1/θ2，tag2/tag3 保持静息）——用于 g=0 严格口径，
     *     去掉由解析几何生成的目标方位特征，只保留纯状态输入。 */
    int    prop_mode;

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
    int synapse_capacity;   /* 已分配槽位数；写入前必须检查/扩容 */
    
    RewardEvent2 reward_queue[QUEUE_SIZE];
    int queue_head;
    int queue_tail;
    
    int current_tick;
    double decoded_omega1;    /* 当前 tick 解码的 omega1_cmd（供 reward 计算用） */
    double decoded_omega2;    /* 当前 tick 解码的 omega2_cmd */
    
    HyperParams2 hp;
    ArmState arm;
    
    double reward_baseline;
    double* reward_history;
    int reward_hist_head;
    int reward_hist_count;
    
    int use_eligibility_trace;
    int use_reward_modulation;

    /* 攀爬纤维：最近一个 tick 算出的关节级有符号运动误差与期望角速度。
     * net2_tick 的物理步之后写入，net2_inject_* 入队时读取，时序严格对齐。 */
    double cf_err1, cf_err2;
    double cf_omega1_star, cf_omega2_star;
    /* 误差的滑动预期：真正驱动学习的是 de = e - EMA(e)（预测误差/意外），
     * 而不是绝对误差 e。绝对误差是直流分量，会把权重单向推到饱和。 */
    double cf_ema1, cf_ema2;

    /* v8（T2 解码可学性上界）：外部线性读出与链路对照。
     * ext_beta 长度 = 2*hidden_count（前一半 joint1，后一半 joint2），
     * 由 net2_set_ext_readout 复制进来；ext_on=1 时绕过 motor 层解码，
     * ω = Σ_hidden s_i·β_i（不做任何权重修改，纯读）。 */
    double* ext_beta;
    int ext_on;
    /* v9（T2c）：外部读出的源层。0=hidden（默认，旧行为），1=input，2=input⊕hidden（拼接）。
     * 神经元数组里 input 段与 hidden 段相邻，故 2 可连续读。
     * ext_beta 长度恒为 2*src_count（前一半 joint1，后一半 joint2）。 */
    int ext_src;
    /* 0 = 正常解码；1 = 直接用解析 ω*（cf_omega*）驱动臂，仅作评估链路对照。 */
    int drive_mode;
} NetworkState2;

/* ============================================================
 * 函数接口
 * ============================================================ */

/* 初始化与销毁 */
int  net2_init(NetworkState2* net, int input_n, int hidden_n,
               int motor1_n, int motor2_n, double avg_conn,
               unsigned int seed);
/* v11（T2h 结构改动，负责人签字 2026-11-14）：
 * direct_fan = 每个输入神经元直连到每个运动层的扇出（12 = 旧拓扑）；
 * hm_scale   = HIDDEN→MOTOR 初始权重缩放，<=0 表示该通路不建（纯直通臂）；
 * im_scale   = INPUT→MOTOR 初始权重缩放（用于在加扇出时保持总驱动不变）。
 * net2_init(...) 等价于 net2_init_ex(..., 12, 1.0, 1.0)，旧路径逐位不变。 */
int  net2_init_ex(NetworkState2* net, int input_n, int hidden_n,
                  int motor1_n, int motor2_n, double avg_conn,
                  unsigned int seed, int direct_fan, double hm_scale, double im_scale);
void net2_destroy(NetworkState2* net);

/* 真实规模查询：VIS/ctypes 调用者必须用这两个返回值分配缓冲区。
 * 不可用 input+hidden+motor1+motor2 估算：input_count 已包含 PROP_N 个本体感觉神经元。 */
int  net2_neuron_count(const NetworkState2* net);
int  net2_synapse_count(const NetworkState2* net);

/* 核心 tick 循环（含物理模拟） */
void net2_tick(NetworkState2* net);

/* 目标控制（宿主调用） */
void net2_set_target(NetworkState2* net, double tx, double ty);

/* 奖励注入（旧接口，内部转调 net2_inject_modulation 并填 0 误差通道） */
void net2_inject_reward(NetworkState2* net, double reward, int delay_ticks);
/* 双通道奖励注入：r1→MOTOR1 突触，r2→MOTOR2 突触，其余突触用均值 */
void net2_inject_reward2(NetworkState2* net, double r1, double r2, int delay_ticks);

/* v6 多通道调制注入：奖励 + 关节级运动误差 + 惊讶度。
 * e1/e2 只作用于 MOTOR 层输入突触，按其偏好方向做定向 LTD/LTP；
 * novelty 只影响学习率增益，不改变 STDP 核。 */
void net2_inject_modulation(NetworkState2* net, double r1, double r2,
                            double e1, double e2, double novelty,
                            int delay_ticks);
/* 读取最近一 tick 的攀爬纤维状态：[err1, err2, omega1*, omega2*] */
void net2_get_cf_error(const NetworkState2* net, double* out4);

/* v8（T2 上界实验，不改判据状态）
 * net2_set_ext_readout：把 2*hidden_count 个系数复制进网络并启用外部线性读出；
 *   返回 0 成功。-1 = 参数非法（net 为空 / beta 为空）。
 * net2_set_drive_mode：0=正常解码，1=用解析 ω* 直接驱动（评估链路对照）。 */
int  net2_set_ext_readout(NetworkState2* net, const double* beta);
/* v9（T2c，不改判据状态）：同上，但显式指定源层。
 * src=0 → hidden（等价于 net2_set_ext_readout），src=1 → input（input_count 维）。 */
int  net2_set_ext_readout_src(NetworkState2* net, const double* beta, int src);
void net2_set_drive_mode(NetworkState2* net, int mode);

/* 物理步进（宿主可选手动控制） */
void arm_step(ArmState* arm, double omega1_cmd, double omega2_cmd);
void arm_set_target(ArmState* arm, double tx, double ty);
double arm_reward(const ArmState* arm);
void net2_reset_ema(void);

/* ============================================================
 * 运行时设置 / 轮次控制（宿主与 Python 可视化共用）
 * 调用方不应直接按内存偏移读写 NetworkState2，结构体变化会导致 ABI 错配。
 * ============================================================ */
/* 按名称设置运行时参数，支持的 name：
 *   learning_rate, tau_stdp, tau_trace, stdp_a_plus, stdp_a_minus,
 *   weight_decay_lambda, reward_baseline_window, psp_factor, trace_clamp,
 *   decode_mode(0|2), wta_k(1..64), use_reward_baseline(0|1),
 *   el_trace(0|1), reward_mod(0|1),
 *   error_gain(0..10), novelty_gain(0..10), cf_lambda(0.1..10),
 *   hetero_tau(0..1), hetero_thresh(0..10), hetero_tune(0..1)
 * 返回 0=成功，-1=未知参数或取值非法；tau_trace 会同步重算 lambda_trace，
 * reward_baseline_window 会重新分配奖励历史缓冲。 */
int    net2_set_param(NetworkState2* net, const char* name, double value);
double net2_get_param(const NetworkState2* net, const char* name);

/* 一轮结束后重置：关节/神经元状态/奖励队列归位，保留资格迹与奖励基线，
 * 并把新目标写入群体编码（不重建网络，权重保持已学到的结果）。 */
void   net2_episode_reset(NetworkState2* net, double tx, double ty);

/* 轻量臂状态（10 个 double，布局与 net2_get_snapshot 的 arm_state 相同），
 * 供训练循环逐 tick 读取末端位置，避免每 tick 复制整个神经元快照。 */
void   net2_get_arm(const NetworkState2* net, double* arm_state);

/* 诊断 */
void net2_dump_stats(NetworkState2* net, FILE* fp);

/* ============================================================
 * VIS 调试接口 —— 给可视化桌面用
 * 所有 buffer 由调用者分配，C 侧只写
 * ============================================================ */

/* 拓扑快照（一次性）：
 * 调用前先用 net2_neuron_count()/net2_synapse_count() 分配缓冲区：
 *   int[neuron_count] ×4（types + pref_x/pref_y/pref_omega 各 double[neuron_count]）
 *   int[synapse_count] ×2 + double[synapse_count]（pre/post/weight）
 * C 侧不检查缓冲区大小，调用者分配不足会造成越界写。
 * 返回 0 成功；net 为空返回 -1。 */
int net2_get_topology(NetworkState2* net,
                      int* neuron_types, double* neuron_pref_x, double* neuron_pref_y, double* neuron_pref_omega,
                      int* syn_pre, int* syn_post, double* syn_weight);

/* 实时快照（每 tick）：
 * out_neurons_buf: double[neuron_count * 3] → [membrane, is_spiking(0/1), last_spike_tick]
 * out_arm_buf: double[10] → [theta1, theta2, omega1, omega2, end_x, end_y, target_x, target_y, reward, max_reach]
 * out_meta_buf: int[3] → [current_tick, reward_queue_head, reward_queue_tail]
 * 注意：3 个神经元数组长度必须取 net2_neuron_count()（含 PROP_N），
 *       C 侧不检查长度，少分配会造成越界写。
 */
void net2_get_snapshot(NetworkState2* net,
                       double* neuron_membrane, double* neuron_spiking, double* neuron_last_spike,
                       double* arm_state, int* meta);

/* 写 topology JSON 文件（初始化后调用一次） */
void net2_dump_topology_json(NetworkState2* net, const char* filepath);

/* 写实时状态 JSON 文件（每 tick 调用，覆盖写） */
void net2_dump_snapshot_json(NetworkState2* net, const char* filepath);

#endif /* BENCHMARK2_CORE_H */
