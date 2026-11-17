/* ============================================================
 * benchmark1_t1.c - B1「T1 两必要条件」机制实验台（判据 v1.4 §6.4 下一步）
 *
 * 目的（只做机制可行性诊断，**不构成 B1 通过**）：
 *   §6.4 发现两个死锁：
 *     (a) 零梯度自锁：严格对称 -> 平局 -> reward=0 -> 零更新 -> 永远平局；
 *     (b) 对称性是 R-STDP 不变子空间：镜像边因「同一 pre 活动 -> 相同
 *         资格迹 -> 相同奖励调制」被等量更新，即使人为注入非零奖励，
 *         读出段权重漂移 0.589 但配对残差仍严格为 0。
 *   §6.4 提出 T1 必须同时满足两个条件才可能学到东西：
 *     ① 奖励 / 教学信号与刺激相关（当前奖励由初始答案决定，与刺激无因果）；
 *     ② 存在对称破缺机制（状态相关探索 / 内部前向模型 / 异质化读出）。
 *   本实验台把 ①② 做成两个**独立开关**，在严格对称读出基线上做 2×2
 *   析因 + 配对消融，验证「两条件是否必要、是否联合充分」。
 *
 * 诚实边界（务必随报告一起读）：
 *   - 「刺激相关教学信号」是**外部教师信号（监督标签）**，不是网络自主发现；
 *     它改变了 B1 任务定义（结果反馈 -> 标签），因此即使 2×2 全开有效，
 *     也只能证明机制可行性，**不得据此宣布 B1 通过**。
 *   - 能力口径一律 g=0：训练结束后关闭全部教学与破缺注入，用冻结投票评估。
 *     训练期含注入的即时正确率单独报告，且不得当作能力。
 *   - 破缺机制只注入输出层膜电位扰动，**与刺激标签严格独立**（mode=1）；
 *     mode=2 是故意的「标签相关作弊对照」，用于自证作弊检测手段有效。
 *   - 教学信号仍走核心原生 net_inject_reward(winner=0) -> 资格迹 -> R-STDP，
 *     不直接写权重、不绕过脉冲；故「完整 − 无迹」消融仍可归因学习。
 *
 * 设计约束（与前三个实验台一致）：
 *   1. benchmark1_core.c/.h、benchmark1_probe.c、benchmark1_sym.c、host、
 *      判据阈值一律零改动；核心源码 SHA1 与 §6.2/§6.3/§6.4 三份报告一致。
 *   2. 试次协议逐 tick 复刻 host（观察 15 / 刺激 10 / 延迟 D / 冷却 30），
 *      超参数固定用 BASE 口径 lr=0.02 / psp=10 / trace_k=3，禁止扫描刷分。
 *   3. 所有随机流沿用 M0 seed_bundle 四流；本层额外两条独立流
 *      （探索噪声 / 标签 shuffle）从 master seed 派生，保证逐位可复现。
 *   4. 本文件与 b1_t1.dll 可单独删除，完全恢复原状，不影响 benchmark1.exe。
 *
 * 编译：与 b1_core.o / seed_bundle.o 链接为 b1_t1.dll（见 build.ps1）。
 * ============================================================ */

#include "benchmark1_core.h"
#include "seed_bundle.h"

#include <stdint.h>

#define OBSERVE_WINDOW 15
#define STIM_DURATION  10
#define COOLDOWN_TICKS 30

static NetworkState g_net;
static int g_ready = 0;

/* hidden→output 段原始边快照（保证 symmetrize 幂等 + 冻结读出可恢复） */
typedef struct {
    int pre;
    int post;
    double w;
} Edge;

static Edge* g_orig = NULL;
static int*  g_idx  = NULL;
static int   g_m    = 0;

/* 实验台内部状态 */
static int      g_no_trace = 0;
static int      g_no_reward = 0;

/* 因子 ①：教学信号。0=host 口径(pred 决定，与刺激无因果)；
 * 1=刺激相关外部教学（监督标签，与 pred 无关）；2=标签 shuffle 对照 */
static int      g_teach_mode = 0;
static double   g_teach_gain = 1.0;
static uint64_t g_shuffle_state = 0;

/* 因子 ②：对称破缺。0=关闭；1=状态相关探索（瞬时膜电位噪声，与标签独立）；
 * 2=标签相关注入（故意的作弊对照）；3=异质化读出（持久结构破缺，与标签独立）。
 * 1 是试次内瞬时扰动、不改变权重结构；3 在 symmetrize 之后直接给镜像边对
 * 加零均值的确定性微扰，使不对称在权重里持久存在。两者都严格与刺激标签无关。 */
static int      g_break_mode = 0;
static double   g_break_sigma = 0.0;
static uint64_t g_explore_state = 0;
static double   g_hetero_residual = 0.0;   /* heterogenize 之后的配对残差 */

/* 冻结读出：每个试次后恢复 hidden→output 段，验证提升是否来自读出可塑 */
static int      g_freeze_readout = 0;

/* 本层四流覆盖状态（语义同 benchmark1_sym.c：value=0 表示清除覆盖） */
static uint64_t g_ov_val[NEURO_STREAM_COUNT];
static int      g_ov_on[NEURO_STREAM_COUNT];
static uint64_t g_master_seed = 0;

/* 最近一次试次的诊断量 */
static double   g_last_teach = 0.0;
static double   g_last_break_sum = 0.0;

/* ---------- 内部工具 ---------- */

double b1t_asymmetry(void);   /* 前置声明：供 b1t_heterogenize 调用 */

static uint64_t t1_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double t1_uniform11(uint64_t* state) {
    /* [-1, 1) */
    return (double)(t1_splitmix64(state) >> 11) / 9007199254740992.0 - 1.0;
}

static void free_snapshot(void) {
    free(g_orig); g_orig = NULL;
    free(g_idx);  g_idx  = NULL;
    g_m = 0;
}

static void clear_traces(Synapse* s) {
    int k;
    s->eligibility_trace = 0.0;
    for (k = 0; k < 3; k++) s->e[k] = 0.0;
    s->last_stdp_tick = g_net.current_tick;
}

static void reset_trial(void) {
    int i;
    for (i = 0; i < g_net.neuron_count; i++) {
        Neuron* n = &g_net.neurons[i];
        n->membrane_potential = MEMBRANE_REST;
        n->is_spiking = 0;
        n->refractory_remaining = 0;
        n->last_spike_tick = -1;
    }
}

static double checksum(void) {
    double s = 0.0;
    int i;
    for (i = 0; i < g_net.synapse_count; i++) s += g_net.synapses[i].weight;
    return s;
}

static int output_base(void) {
    return g_net.sensory_a_count + g_net.sensory_b_count + g_net.hidden_count;
}

/* ---------- 生命周期 ---------- */

void b1t_destroy(void);

int b1t_create(int sensory_a, int sensory_b, int hidden,
               int output_a, int output_b, double avg_conn,
               unsigned int seed) {
    int s, hbase, obase;
    NeuroSeedBundle bundle;
    uint64_t base[NEURO_STREAM_COUNT];
    if (g_ready) { b1t_destroy(); }

    g_master_seed = (uint64_t)seed;
    g_hetero_residual = 0.0;   /* 防止上一次实验的诊断量泄漏到本实例 */
    neuro_seed_bundle_init(&bundle, g_master_seed);
    base[NEURO_STREAM_STRUCTURE]   = bundle.structure;
    base[NEURO_STREAM_WEIGHT]      = bundle.weight;
    base[NEURO_STREAM_NOISE]       = bundle.noise;
    base[NEURO_STREAM_TRIAL_ORDER] = bundle.trial_order;
    for (s = 0; s < NEURO_STREAM_COUNT; s++) {
        net_override_seed_stream(s, g_ov_on[s] ? g_ov_val[s] : base[s]);
    }

    /* 两条独立实验流从 master seed 派生，避免所有 seed 共享同一序列 */
    g_explore_state = t1_splitmix64(&g_master_seed) ^ 0xA5A5A5A5A5A5A5A5ULL;
    g_shuffle_state = t1_splitmix64(&g_master_seed) ^ 0x5A5A5A5A5A5A5A5AULL;

    memset(&g_net, 0, sizeof(g_net));
    if (net_init(&g_net, sensory_a, sensory_b, hidden,
                 output_a, output_b, avg_conn, seed) != 0) return -1;
    g_ready = 1;

    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = hbase + g_net.hidden_count;
    g_orig = (Edge*)calloc((size_t)g_net.synapse_count, sizeof(Edge));
    g_idx  = (int*)calloc((size_t)g_net.synapse_count, sizeof(int));
    if (!g_orig || !g_idx) { b1t_destroy(); return -2; }
    g_m = 0;
    for (s = 0; s < g_net.synapse_count; s++) {
        int pre = g_net.synapses[s].pre_id;
        int post = g_net.synapses[s].post_id;
        if (pre < hbase || pre >= obase || post < obase) continue;
        g_orig[g_m].pre = pre;
        g_orig[g_m].post = post;
        g_orig[g_m].w = g_net.synapses[s].weight;
        g_idx[g_m] = s;
        g_m++;
    }
    return 0;
}

void b1t_destroy(void) {
    free_snapshot();
    if (g_ready) { net_destroy(&g_net); g_ready = 0; }
}

int b1t_ready(void) { return g_ready; }
int b1t_synapse_count(void) { return g_ready ? g_net.synapse_count : 0; }
int b1t_readout_edge_count(void) { return g_m; }

/* ---------- M0 四流契约（create 之前调用） ---------- */

void b1t_override_stream(int which, uint64_t value) {
    if (which < 0 || which >= NEURO_STREAM_COUNT) return;
    if (value == 0) {
        g_ov_on[which] = 0;
        g_ov_val[which] = 0;
    } else {
        g_ov_on[which] = 1;
        g_ov_val[which] = value;
    }
}
void b1t_legacy_rng(int on) { net_use_legacy_rng(on); }

/* ---------- 超参数 / 消融（BASE 口径固定，禁止扫描刷分） ---------- */

void b1t_set_hyper(double learning_rate, double psp_factor, int trace_k) {
    if (!g_ready) return;
    if (learning_rate > 0.0) g_net.hp.learning_rate = learning_rate;
    if (psp_factor > 0.0) g_net.hp.psp_factor = psp_factor;
    if (trace_k >= 1 && trace_k <= 3) g_net.hp.trace_k = trace_k;
}

void b1t_set_ablation(int no_trace, int no_reward) {
    g_no_trace = no_trace ? 1 : 0;
    g_no_reward = no_reward ? 1 : 0;
    if (g_ready) {
        g_net.use_eligibility_trace = g_no_trace ? 0 : 1;
        g_net.use_reward_modulation = g_no_reward ? 0 : 1;
    }
}

/* ---------- 两个 T1 因子 ---------- */

/* mode: 0=host 口径, 1=刺激相关教学（外部监督标签）, 2=标签 shuffle 对照 */
void b1t_set_teaching(int mode, double gain, uint64_t shuffle_seed) {
    g_teach_mode = (mode >= 0 && mode <= 2) ? mode : 0;
    g_teach_gain = (gain > 0.0) ? gain : 1.0;
    if (shuffle_seed != 0) g_shuffle_state = shuffle_seed;
}

/* mode: 0=关闭, 1=状态相关探索（标签独立）, 2=标签相关注入（作弊对照） */
void b1t_set_breaking(int mode, double sigma, uint64_t explore_seed) {
    g_break_mode = (mode >= 0 && mode <= 3) ? mode : 0;
    g_break_sigma = (sigma > 0.0) ? sigma : 0.0;
    if (explore_seed != 0) g_explore_state = explore_seed;
}

/* ---------- 第三类破缺：异质化读出（持久结构破缺） ---------- */

/* 在 b1t_symmetrize(1.0) 之后调用：对每个镜像边对 (2k, 2k+1) 施加
 * 反对称的零均值微扰  s0 += sigma*r_k, s1 -= sigma*r_k。
 * r_k 来自独立探索流，与刺激标签严格无关；总权重和不变，不预埋任何答案。
 * 返回施加后的配对残差（>0 表示对称确实被持久打破）。 */
double b1t_heterogenize(double sigma, uint64_t seed) {
    int k, n_pairs;
    double s = sigma;
    if (!g_ready) return -1.0;
    g_hetero_residual = 0.0;
    if (seed != 0) g_explore_state = seed;
    if (s <= 0.0) return b1t_asymmetry();
    n_pairs = g_m / 2;
    for (k = 0; k < n_pairs; k++) {
        int i0 = g_idx[2 * k], i1 = g_idx[2 * k + 1];
        double r = t1_uniform11(&g_explore_state);
        double d = s * r;
        double w0 = g_net.synapses[i0].weight + d;
        double w1 = g_net.synapses[i1].weight - d;
        if (w0 < WEIGHT_MIN) w0 = WEIGHT_MIN;
        if (w0 > WEIGHT_MAX) w0 = WEIGHT_MAX;
        if (w1 < WEIGHT_MIN) w1 = WEIGHT_MIN;
        if (w1 > WEIGHT_MAX) w1 = WEIGHT_MAX;
        g_net.synapses[i0].weight = w0;
        g_net.synapses[i1].weight = w1;
        clear_traces(&g_net.synapses[i0]);
        clear_traces(&g_net.synapses[i1]);
    }
    g_hetero_residual = b1t_asymmetry();
    return g_hetero_residual;
}

double b1t_hetero_residual(void) { return g_hetero_residual; }

void b1t_set_freeze_readout(int on) { g_freeze_readout = on ? 1 : 0; }

/* ---------- 严格对称基线（构造同 benchmark1_sym.c） ---------- */

double b1t_asymmetry(void) {
    double* acc;
    int hbase, obase, oa, pairs, i, s;
    double maxdiff = 0.0;
    if (!g_ready) return -1.0;
    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = output_base();
    oa = g_net.output_a_count;
    pairs = oa < g_net.output_b_count ? oa : g_net.output_b_count;
    if (pairs <= 0) return -1.0;
    acc = (double*)calloc((size_t)g_net.hidden_count * pairs * 2, sizeof(double));
    if (!acc) return -1.0;
    for (s = 0; s < g_net.synapse_count; s++) {
        int pre = g_net.synapses[s].pre_id;
        int post = g_net.synapses[s].post_id;
        int j, side, h;
        if (pre < hbase || pre >= obase || post < obase) continue;
        h = pre - hbase;
        if (post < obase + oa) { side = 0; j = post - obase; }
        else { side = 1; j = post - (obase + oa); }
        if (j < 0 || j >= pairs) continue;
        acc[(h * pairs + j) * 2 + side] += g_net.synapses[s].weight;
    }
    for (i = 0; i < g_net.hidden_count * pairs; i++) {
        double d = fabs(acc[i * 2] - acc[i * 2 + 1]);
        if (d > maxdiff) maxdiff = d;
    }
    free(acc);
    return maxdiff;
}

/* alpha=1 严格对称；alpha=0 逐位恢复原始。返回残差。 */
double b1t_symmetrize(double alpha) {
    int k, pairs, j, obase, oa, n_pairs;
    double a = alpha;
    if (!g_ready) return -1.0;
    if (a < 0.0) a = 0.0;
    if (a > 1.0) a = 1.0;
    obase = output_base();
    oa = g_net.output_a_count;
    pairs = oa < g_net.output_b_count ? oa : g_net.output_b_count;

    if (a <= 0.0) {
        for (k = 0; k < g_m; k++) {
            Synapse* s = &g_net.synapses[g_idx[k]];
            s->pre_id = g_orig[k].pre;
            s->post_id = g_orig[k].post;
            s->weight = g_orig[k].w;
            clear_traces(s);
        }
        return b1t_asymmetry();
    }
    n_pairs = g_m / 2;
    for (k = 0; k < n_pairs; k++) {
        int i0 = g_idx[2 * k], i1 = g_idx[2 * k + 1];
        Synapse* s0 = &g_net.synapses[i0];
        Synapse* s1 = &g_net.synapses[i1];
        int pre = g_orig[2 * k].pre;
        double target = g_orig[2 * k].w;
        j = pairs > 0 ? k % pairs : 0;
        s0->pre_id = pre;
        s0->post_id = obase + j;
        s0->weight = (1.0 - a) * g_orig[2 * k].w + a * target;
        s1->pre_id = pre;
        s1->post_id = obase + oa + j;
        s1->weight = (1.0 - a) * g_orig[2 * k + 1].w + a * target;
        clear_traces(s0);
        clear_traces(s1);
    }
    if (g_m % 2 == 1) {
        Synapse* s = &g_net.synapses[g_idx[g_m - 1]];
        s->pre_id = g_orig[g_m - 1].pre;
        s->post_id = g_orig[g_m - 1].post;
        s->weight = (1.0 - a) * g_orig[g_m - 1].w;
        clear_traces(s);
    }
    return b1t_asymmetry();
}

/* ---------- 只读诊断 ---------- */

void b1t_output_bias(double out[4]) {
    if (out) { out[0] = out[1] = out[2] = out[3] = 0.0; }
    if (g_ready) net_output_bias(&g_net, out);
}

double b1t_weight_checksum(void) { return g_ready ? checksum() : 0.0; }

double b1t_readout_checksum(void) {
    double s = 0.0;
    int k;
    if (!g_ready) return 0.0;
    for (k = 0; k < g_m; k++) s += g_net.synapses[g_idx[k]].weight;
    return s;
}

/* ---------- 破缺注入（只在训练试次观察窗口调用，g=0 评估绝不调用） ---------- */

/* 返回本次注入的绝对幅度累加（诊断用）。mode=1 与刺激标签严格独立。 */
static double inject_breaking(int stim) {
    int i, obase;
    double total = 0.0;
    /* mode=3 是持久结构破缺（已在权重里），不需要试次内注入 */
    if (!g_ready || g_break_mode == 0 || g_break_mode == 3 ||
        g_break_sigma <= 0.0) return 0.0;
    obase = output_base();
    for (i = obase; i < g_net.neuron_count; i++) {
        Neuron* n = &g_net.neurons[i];
        double d;
        if (n->type != NT_OUTPUT_A && n->type != NT_OUTPUT_B) continue;
        if (g_break_mode == 1) {
            /* 状态相关探索：方向由独立流决定，幅度随膜电位偏离静息的程度放大。
             * 与 stim 完全无关 —— 这是「对称破缺」而非「答案注入」。 */
            double r = t1_uniform11(&g_explore_state);
            double state = fabs(n->membrane_potential - MEMBRANE_REST) / 20.0;
            d = g_break_sigma * r * (1.0 + state);
        } else {
            /* 作弊对照：直接朝期望侧注入，用于自证检测手段有效。
             * 默认关闭、不计入任何验收。 */
            int want_a = (stim == 0);
            int is_a = (n->type == NT_OUTPUT_A);
            d = g_break_sigma * ((want_a == is_a) ? 1.0 : -1.0);
        }
        n->membrane_potential += d;
        total += fabs(d);
    }
    return total;
}

/* ---------- g=0 冻结投票（无教学、无破缺、无奖励、不改权重） ---------- */

int b1t_frozen_vote(int stim, int* a_cnt, int* b_cnt) {
    int t, i, a = 0, b = 0, pred;
    int saved_mod = 0;
    if (!g_ready) return -1;
    /* g=0 评估必须是纯前向、零学习：清空可能残留的奖励事件，
     * 并暂时关闭奖励调制（不改变权重，也不改变网络结构）。 */
    g_net.queue_head = g_net.queue_tail;
    saved_mod = g_net.use_reward_modulation;
    g_net.use_reward_modulation = 0;
    reset_trial();
    for (t = 0; t < OBSERVE_WINDOW; t++) {
        if (t < STIM_DURATION) {
            if (stim == 0) net_inject_stimulus_a(&g_net);
            else net_inject_stimulus_b(&g_net);
        } else {
            net_clear_stimulus(&g_net);
        }
        net_tick(&g_net);
        for (i = 0; i < g_net.neuron_count; i++) {
            if (!g_net.neurons[i].is_spiking) continue;
            if (g_net.neurons[i].type == NT_OUTPUT_A) a++;
            else if (g_net.neurons[i].type == NT_OUTPUT_B) b++;
        }
    }
    g_net.use_reward_modulation = saved_mod;
    if (a_cnt) *a_cnt = a;
    if (b_cnt) *b_cnt = b;
    if (a > b && a > 0) pred = 1;
    else if (b > a && b > 0) pred = 2;
    else pred = 0;
    return pred;
}

/* ---------- 完整试次（含 T1 因子，复刻 host 协议） ---------- */

/* out[0]=a, [1]=b, [2]=即时 correct(含注入，非能力口径),
 * [3]=注入的奖励标量, [4]=本试次权重 checksum 变化, [5]=tick 数,
 * [6]=教学信号值, [7]=破缺注入绝对幅度累加 */
int b1t_run_trial(int stim, int learning_on, int delay, double out[8]) {
    int t, i, a = 0, b = 0, pred, expected, correct;
    double reward, w0, teach;
    int ticks = 0, arrival;
    double break_sum = 0.0;
    double* ro_saved = NULL;
    if (!g_ready) return -1;
    w0 = checksum();
    if (g_freeze_readout) {
        ro_saved = (double*)calloc((size_t)g_m, sizeof(double));
        if (ro_saved) {
            for (i = 0; i < g_m; i++) ro_saved[i] = g_net.synapses[g_idx[i]].weight;
        }
    }
    reset_trial();

    for (t = 0; t < OBSERVE_WINDOW; t++) {
        if (t < STIM_DURATION) {
            if (stim == 0) net_inject_stimulus_a(&g_net);
            else net_inject_stimulus_b(&g_net);
        } else {
            net_clear_stimulus(&g_net);
        }
        break_sum += inject_breaking(stim);
        net_tick(&g_net);
        ticks++;
        for (i = 0; i < g_net.neuron_count; i++) {
            if (!g_net.neurons[i].is_spiking) continue;
            if (g_net.neurons[i].type == NT_OUTPUT_A) a++;
            else if (g_net.neurons[i].type == NT_OUTPUT_B) b++;
        }
    }

    if (a > b && a > 0) pred = 1;
    else if (b > a && b > 0) pred = 2;
    else pred = 0;
    expected = (stim == 0) ? 1 : 2;
    correct = (pred == expected) ? 1 : 0;

    /* 因子 ① 教学信号：仍走核心原生奖励队列 + 资格迹 + R-STDP */
    if (g_teach_mode == 0) {
        /* host 口径：由网络自己的答案决定，与刺激无因果关系 */
        teach = (pred == 0) ? 0.0 : (correct ? 1.0 : -1.0);
    } else if (g_teach_mode == 1) {
        /* 刺激相关外部教学（监督标签，与 pred 无关）：打破零梯度自锁 */
        teach = g_teach_gain * ((stim == 0) ? 1.0 : -1.0);
    } else {
        /* 标签 shuffle：符号与刺激独立，用于检测「提升是否真来自教学信息」 */
        teach = g_teach_gain * ((t1_splitmix64(&g_shuffle_state) >> 63) ? 1.0 : -1.0);
    }
    reward = teach;
    g_last_teach = teach;
    g_last_break_sum = break_sum;

    if (learning_on) net_inject_reward(&g_net, reward, delay);

    arrival = g_net.current_tick + delay + 1;
    while (g_net.current_tick < arrival) { net_tick(&g_net); ticks++; }
    for (t = 0; t < COOLDOWN_TICKS; t++) { net_tick(&g_net); ticks++; }

    /* 冻结读出：恢复权重与迹，使本试次学习成果不累积 */
    if (ro_saved) {
        for (i = 0; i < g_m; i++) {
            Synapse* s = &g_net.synapses[g_idx[i]];
            s->weight = ro_saved[i];
            clear_traces(s);
        }
        free(ro_saved);
    }

    if (out) {
        out[0] = (double)a;
        out[1] = (double)b;
        out[2] = (double)correct;
        out[3] = reward;
        out[4] = checksum() - w0;
        out[5] = (double)ticks;
        out[6] = teach;
        out[7] = break_sum;
    }
    return pred;
}
