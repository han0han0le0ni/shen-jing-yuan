/* ============================================================
 * benchmark1_sym.c - B1「严格对称初始读出」实验台（T1/T2 · 判据 v1.3 §6.3 约束 ①）
 *
 * 目的：把初始 hidden→output 读出改造成严格对称，验证「初始答案」是否
 * 消失，并在同一实验台上完成约束 ②（同 seed 配对消融）与 ③（四流隔离）。
 *
 * 与 benchmark1_probe.c（只读上界探针）的区别：
 *   - 本层会**改写 g_net.synapses[] 中 hidden→output 段的权重/拓扑**，
 *     并提供完整 host 试次协议（含奖励注入），因此可以跑带学习的训练。
 *   - 但**只改这一段**；感觉→hidden 段、学习规则、刺激生成器、奖励规则、
 *     判据阈值一律不动，保证单变量。
 *
 * 严格对称的构造（alpha=1）：
 *   对 hidden→output 段的 m 条边按原顺序两两配对，每对 (2k, 2k+1) 改写成
 *       (pre_k -> OA_j, w_k)  和  (pre_k -> OB_j, w_k)
 *   即同一个前神经元、同一个权重、指向镜像的输出对。于是对任意刺激，
 *   OA_j 与 OB_j 的入边集合逐条相同 -> 膜电位输入严格相等 -> 发放数相等
 *   -> 侧抑制不触发 -> 必然平局。alpha 是插值剂量：
 *       w = (1-alpha)*w_orig + alpha*w_k
 *   alpha=0 逐位恢复原始网络（对照组），alpha=1 为严格对称。
 *
 * 设计约束：
 *   1. benchmark1_core.c/.h、benchmark1_probe.c、host、判据一律零改动；
 *      核心源码 SHA1 与 audit/b1_root_cause_2026-11-12.md、
 *      audit/b1_supervised_ceiling_2026-11-12.md 保持一致，证据链不断。
 *   2. 试次协议逐 tick 复刻 host/benchmark1_host.cpp（观察窗口 15 /
 *      刺激 10 / 奖励延迟 D / 全局 tick 到达 / 冷却 30），默认参数一致。
 *   3. b1s_frozen_vote() 不注入奖励、不改权重，用于测「初始答案」。
 *   4. 所有随机流沿用 M0 seed_bundle 四流；四流覆盖开关在 create 前生效。
 *   5. b1s_set_tie_break() 是**探索性开关，默认关闭、不计入任何验收**：
 *      平局时由独立流公平硬币判 A/B，用于观察「没有初始答案是否还有梯度」。
 *
 * 编译：与 b1_core.o / seed_bundle.o 链接为 b1_sym.dll（见 build.ps1）。
 * 删除本文件与 b1_sym.dll 即可完全恢复原状，不影响 benchmark1.exe。
 * ============================================================ */

#include "benchmark1_core.h"
#include "seed_bundle.h"

#include <stdint.h>

#define OBSERVE_WINDOW 15
#define STIM_DURATION  10
#define COOLDOWN_TICKS 30

static NetworkState g_net;
static int g_ready = 0;

/* hidden→output 段原始边快照（保证 symmetrize 幂等 + alpha 剂量可重算） */
typedef struct {
    int pre;
    int post;
    double w;
} Edge;

static Edge* g_orig = NULL;   /* g_m 条原始边 */
static int*  g_idx  = NULL;   /* 每条边在 synapses[] 中的下标 */
static int   g_m    = 0;

/* 实验台内部状态：消融与平局打破（均默认关闭 = host 原口径） */
static int   g_no_trace = 0;
static int   g_no_reward = 0;
static int   g_tie_break = 0;
static uint64_t g_tie_state = 0x9E3779B97F4A7C15ULL;

/* 实验台自己的四流覆盖状态（注意与核心 net_override_seed_stream 的语义不同）：
 *   核心的覆盖开关是「一次设置、永久生效」的全局静态量，且 value=0 也会被
 *   当成有效覆盖值。若每次 create 前不显式重设全部四条流，先前的覆盖会泄漏
 *   到后续实验，甚至把不同 seed 钉死成同一网络。
 *   因此本层在每次 b1s_create 中：用 master seed 派生出四条基准流，
 *   逐条显式覆盖为基准值（与核心默认逐位等价），只把 g_ov_on=1 的那一条
 *   换成实验值。这样既保持严格单变量，又不必修改核心源码。
 *   b1s_override_stream(which, 0) 在本层的语义是「清除该流的覆盖」。 */
static uint64_t g_ov_val[NEURO_STREAM_COUNT];
static int      g_ov_on[NEURO_STREAM_COUNT];

static uint64_t sym_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---------- 内部工具 ---------- */

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

/* ---------- 生命周期 ---------- */

void b1s_destroy(void);   /* 供 b1s_create 在重建前调用 */

int b1s_create(int sensory_a, int sensory_b, int hidden,
               int output_a, int output_b, double avg_conn,
               unsigned int seed) {
    int s, hbase, obase;
    NeuroSeedBundle bundle;
    uint64_t base[NEURO_STREAM_COUNT];
    if (g_ready) { b1s_destroy(); }

    /* 每次重建都显式重设四条流：无覆盖时逐条写回 master seed 的派生值，
     * 与核心默认行为逐位一致；有覆盖时只替换目标流（严格单变量）。 */
    neuro_seed_bundle_init(&bundle, (uint64_t)seed);
    base[NEURO_STREAM_STRUCTURE]   = bundle.structure;
    base[NEURO_STREAM_WEIGHT]      = bundle.weight;
    base[NEURO_STREAM_NOISE]       = bundle.noise;
    base[NEURO_STREAM_TRIAL_ORDER] = bundle.trial_order;
    for (s = 0; s < NEURO_STREAM_COUNT; s++) {
        net_override_seed_stream(s, g_ov_on[s] ? g_ov_val[s] : base[s]);
    }

    memset(&g_net, 0, sizeof(g_net));
    if (net_init(&g_net, sensory_a, sensory_b, hidden,
                 output_a, output_b, avg_conn, seed) != 0) return -1;
    g_ready = 1;

    /* 快照 hidden→output 段 */
    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = hbase + g_net.hidden_count;
    g_orig = (Edge*)calloc((size_t)g_net.synapse_count, sizeof(Edge));
    g_idx  = (int*)calloc((size_t)g_net.synapse_count, sizeof(int));
    if (!g_orig || !g_idx) { b1s_destroy(); return -2; }
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

void b1s_destroy(void) {
    free_snapshot();
    if (g_ready) { net_destroy(&g_net); g_ready = 0; }
}

int b1s_ready(void) { return g_ready; }
int b1s_synapse_count(void) { return g_ready ? g_net.synapse_count : 0; }
int b1s_readout_edge_count(void) { return g_m; }

/* ---------- M0 四流契约（create 之前调用） ---------- */

/* value=0 表示「清除覆盖」；非 0 表示只覆盖这一条流。
 * 真正的 net_override_seed_stream 调用推迟到 b1s_create 内统一执行。 */
void b1s_override_stream(int which, uint64_t value) {
    if (which < 0 || which >= NEURO_STREAM_COUNT) return;
    if (value == 0) {
        g_ov_on[which] = 0;
        g_ov_val[which] = 0;
    } else {
        g_ov_on[which] = 1;
        g_ov_val[which] = value;
    }
}
void b1s_legacy_rng(int on) { net_use_legacy_rng(on); }

/* ---------- 超参数 / 消融 / 探索开关 ---------- */

/* 与 tools/b1_bias_diagnosis.py 的 BASE 口径一致：lr=0.02, psp=10, trace_k=3 */
void b1s_set_hyper(double learning_rate, double psp_factor, int trace_k) {
    if (!g_ready) return;
    if (learning_rate > 0.0) g_net.hp.learning_rate = learning_rate;
    if (psp_factor > 0.0) g_net.hp.psp_factor = psp_factor;
    if (trace_k >= 1 && trace_k <= 3) g_net.hp.trace_k = trace_k;
}

void b1s_set_ablation(int no_trace, int no_reward) {
    g_no_trace = no_trace ? 1 : 0;
    g_no_reward = no_reward ? 1 : 0;
    if (g_ready) {
        g_net.use_eligibility_trace = g_no_trace ? 0 : 1;
        g_net.use_reward_modulation = g_no_reward ? 0 : 1;
    }
}

void b1s_set_tie_break(int on) { g_tie_break = on ? 1 : 0; }
void b1s_tie_break_seed(uint64_t v) { g_tie_state = v; }

/* ---------- 对称化 ---------- */

/* 严格配对残差：对每个 (hidden 前神经元 h, 输出对 j)，比较 h→OA_j 与 h→OB_j
 * 的权重和，取最大绝对差。alpha=1 时应严格为 0。 */
double b1s_asymmetry(void) {
    double* acc;
    int hbase, obase, oa, pairs, i, s;
    double maxdiff = 0.0;
    if (!g_ready) return -1.0;
    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = hbase + g_net.hidden_count;
    oa = g_net.output_a_count;
    pairs = g_net.output_a_count < g_net.output_b_count
            ? g_net.output_a_count : g_net.output_b_count;
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

/* alpha ∈ [0,1]；0=逐位恢复原始，1=严格对称。返回对称化后的残差。 */
double b1s_symmetrize(double alpha) {
    int k, pairs, j, hbase, obase, oa, n_pairs;
    double a = alpha;
    if (!g_ready) return -1.0;
    if (a < 0.0) a = 0.0;
    if (a > 1.0) a = 1.0;
    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = hbase + g_net.hidden_count;
    oa = g_net.output_a_count;
    pairs = oa < g_net.output_b_count ? oa : g_net.output_b_count;

    if (a <= 0.0) {
        /* 逐位恢复原始（对照组），含资格迹状态 */
        for (k = 0; k < g_m; k++) {
            Synapse* s = &g_net.synapses[g_idx[k]];
            s->pre_id = g_orig[k].pre;
            s->post_id = g_orig[k].post;
            s->weight = g_orig[k].w;
            clear_traces(s);
        }
        return b1s_asymmetry();
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
    /* 奇数尾巴：alpha=1 时置 0（权重为 0 的边不破坏镜像对称） */
    if (g_m % 2 == 1) {
        Synapse* s = &g_net.synapses[g_idx[g_m - 1]];
        s->pre_id = g_orig[g_m - 1].pre;
        s->post_id = g_orig[g_m - 1].post;
        s->weight = (1.0 - a) * g_orig[g_m - 1].w;
        clear_traces(s);
    }
    return b1s_asymmetry();
}

/* ---------- 只读诊断 ---------- */

void b1s_output_bias(double out[4]) {
    if (out) { out[0] = out[1] = out[2] = out[3] = 0.0; }
    if (g_ready) net_output_bias(&g_net, out);
}

double b1s_weight_checksum(void) { return g_ready ? checksum() : 0.0; }

/* 只统计 hidden→output 段的权重和：用于区分「学习改了别的层」与
 * 「学习改了读出但镜像边被等量更新（对称不变子空间）」。 */
double b1s_readout_checksum(void) {
    double s = 0.0;
    int k;
    if (!g_ready) return 0.0;
    for (k = 0; k < g_m; k++) s += g_net.synapses[g_idx[k]].weight;
    return s;
}

/* ---------- 冻结投票（不注入奖励、不改权重） ---------- */

int b1s_frozen_vote(int stim, int* a_cnt, int* b_cnt) {
    int t, i, a = 0, b = 0, pred;
    if (!g_ready) return -1;
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
    if (a_cnt) *a_cnt = a;
    if (b_cnt) *b_cnt = b;
    if (a > b && a > 0) pred = 1;
    else if (b > a && b > 0) pred = 2;
    else pred = 0;
    return pred;
}

/* ---------- 完整试次（复刻 host） ---------- */

/* learning_on=0 仍完整推进 tick，只是不注入奖励（协议一致）。
 * out[0]=a 发放数, out[1]=b 发放数, out[2]=is_correct,
 * out[3]=reward, out[4]=本试次权重 checksum 变化, out[5]=本试次 tick 数 */
int b1s_run_trial(int stim, int learning_on, int delay, double out[6]) {
    int t, i, a = 0, b = 0, pred, expected, correct;
    double reward, w0;
    int ticks = 0, arrival;
    if (!g_ready) return -1;
    w0 = checksum();
    reset_trial();

    for (t = 0; t < OBSERVE_WINDOW; t++) {
        if (t < STIM_DURATION) {
            if (stim == 0) net_inject_stimulus_a(&g_net);
            else net_inject_stimulus_b(&g_net);
        } else {
            net_clear_stimulus(&g_net);
        }
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

    /* 探索性平局打破：与标签独立的公平硬币，默认关闭 */
    if (pred == 0 && g_tie_break) {
        pred = (int)(sym_splitmix64(&g_tie_state) >> 63) ? 2 : 1;
    }
    expected = (stim == 0) ? 1 : 2;
    correct = (pred == expected) ? 1 : 0;
    reward = (pred == 0) ? 0.0 : (correct ? 1.0 : -1.0);

    if (learning_on) net_inject_reward(&g_net, reward, delay);

    arrival = g_net.current_tick + delay + 1;
    while (g_net.current_tick < arrival) { net_tick(&g_net); ticks++; }
    for (t = 0; t < COOLDOWN_TICKS; t++) { net_tick(&g_net); ticks++; }

    if (out) {
        out[0] = (double)a;
        out[1] = (double)b;
        out[2] = (double)correct;
        out[3] = reward;
        out[4] = checksum() - w0;
        out[5] = (double)ticks;
    }
    return pred;
}
