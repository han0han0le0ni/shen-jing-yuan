/* ============================================================
 * benchmark1_psel.c - B1「通路选择性第三因子」机制实验台
 *                     判据 v1.5 §6.5 / 大纲附录 G.4 的下一步
 *
 * 目的（只做机制可行性诊断，**不构成 B1 通过**）：
 *   §6.5 证伪了 §6.4 的两个必要条件，并给出新根因：
 *     net_inject_reward(winner=0) 是逐突触无差别的**全局标量**；严格对称
 *     读出下镜像边的资格迹相同、奖励调制相同 -> 被等量更新 -> 对称残差
 *     恒为 0，梯度与信号内容无关。
 *   §6.5 的下一步要求：在**不引入外部标签**的前提下，把「选择性」做成
 *   网络**内部**的量（内部前向模型 / 通路特异性资格迹），而不是人工按
 *   winner 路由；在严格对称基线上重做同 seed 配对消融。
 *
 * 本实验台的实现（第三因子仍是全局标量，选择性放在资格迹上）：
 *   1. 观察窗口内累计 hidden 左/右半的实际发放数 h_L / h_R —— 这是网络
 *      **内部状态**，§6.3 已证明它携带刺激信息（hidden 表示线性可分）。
 *      内部读出 d = (h_L - h_R) / (h_L + h_R)，无外部标签、无人工 winner。
 *   2. 奖励到达前，按突触后神经元所属输出通路给**资格迹**乘门控：
 *        OUTPUT_A 输入边  m_A = 1 + lambda*d
 *        OUTPUT_B 输入边  m_B = 1 - lambda*d
 *      其余突触不变。这使镜像边的 e[k] 不再相同 —— 即「通路特异性
 *      资格迹」。第三因子（结果反馈）仍是 net_inject_reward(winner=0)
 *      的全局标量，核心原生 R-STDP 一行未改、不直接写权重。
 *   3. 奖励回到 B1 原口径的**延迟结果反馈**（由网络自己的 pred 与期望
 *      比较得出），不使用 §6.5 的 mode=1 刺激相关外部教学信号。
 *      reward_mode=1 只把「平局」从 reward=0 改为 reward=-1（视为失败），
 *      用于检验 §6.4 的零梯度自锁；reward_mode=2 额外补偿核心奖励基线，
 *      仅用于诊断基线饱和，**不计入任何验收**。
 *
 * 诚实边界（务必随报告一起读）：
 *   - d 的符号与网络既有拓扑（感觉A->hidden左半->输出A 为「正确路径」）
 *     一致，是结构先验，不是网络自主学习出来的；hidden 中确实已存在
 *     刺激信息（§6.3）。因此即使本机制有效，也只证明「第三因子的通路
 *     选择性 + 延迟结果反馈足以让局部 R-STDP 破缺并学出映射」，
 *     **不得据此宣布 B1 通过**。
 *   - 能力口径一律 g=0：训练结束后关闭全部门控与奖励，冻结投票评估，
 *     评估前后权重漂移必须为 0。训练期含注入的即时正确率单独报告。
 *   - 判据阈值、学习规则、刺激生成器、任务定义、benchmark1_core.c/.h、
 *     benchmark1_probe.c、benchmark1_sym.c、benchmark1_t1.c、host 一律
 *     零改动；核心 SHA1 与 §6.2/§6.3/§6.4/§6.5 四份报告一致。
 *
 * 设计约束（与前四个实验台一致）：
 *   1. 只链接既有 b1_core.o / seed_bundle.o；本文件与 b1_psel.dll 可单独
 *      删除，完全恢复原状，不影响 benchmark1.exe。
 *   2. 试次协议逐 tick 复刻 host（观察 15 / 刺激 10 / 延迟 D / 冷却 30），
 *      超参数固定 BASE 口径 lr=0.02 / psp=10 / trace_k=3，禁止扫描刷分。
 *   3. 随机流沿用 M0 seed_bundle 四流；额外的 shuffle 流从 master seed
 *      派生，保证逐位可复现。
 *
 * 编译：gcc -shared ... brain/benchmark1_psel.c b1_core.o seed_bundle.o
 *       -o b1_psel.dll -lm（见 build.ps1）。
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

/* 消融开关 */
static int      g_no_trace = 0;
static int      g_no_reward = 0;
static int      g_freeze_readout = 0;

/* 选择性因子：
 * mode 0=关闭（全局标量基线）；1=内部量 d（hidden 左右半发放差）；
 *      2=幅度同 d 但符号来自独立 shuffle 流（与刺激独立，对照）；
 *      3=恒定 d=+1（恒定答案检测）；4=符号反转 -d（方向对照）；
 *      5=sign(d_raw)（内部量幅度最大化，用于机制上界诊断）；
 *      6=-sign(d_raw)（门控方向反转，预注册的方向对照）。*/
static int      g_sel_mode = 0;
static double   g_sel_lambda = 0.0;
static uint64_t g_shuffle_state = 0;
/* 门控与第三因子符号耦合：g = d * sign(desired)。
 * 结果反馈为负时若不耦合，放大通路反而被削弱更多（学反）；耦合后
 * 「A 试次增强 A / B 试次削弱 A」在任意反馈符号下方向都正确。*/
static int      g_couple = 1;

/* 第三因子口径：0=host 原口径（平局 reward=0）；1=平局视为失败(-1)；
 * 2=平局视为失败 + 补偿核心奖励基线（仅诊断基线饱和，不计入验收）；
 * 3=内部预测读出 sign(d_raw)（零均值、幅度稳定、完全来自网络内部，
 *    用于检验「选择性」本身是否足以让局部 R-STDP 学出映射；机制诊断，
 *    不改变 B1 正式口径，不计入验收）。*/
static int      g_reward_mode = 0;

/* 本层四流覆盖状态（value=0 表示清除覆盖） */
static uint64_t g_ov_val[NEURO_STREAM_COUNT];
static int      g_ov_on[NEURO_STREAM_COUNT];
static uint64_t g_master_seed = 0;

/* 最近一次试次的诊断量 */
static double   g_last_d = 0.0;
static double   g_last_gate_diff = 0.0;
static double   g_last_mirror_diff = 0.0;
static double   g_last_baseline = 0.0;
static double   g_last_hidden_l = 0.0;
static double   g_last_hidden_r = 0.0;

/* ---------- 内部工具 ---------- */

double b1p_asymmetry(void);   /* 前置声明 */

static uint64_t psel_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
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
    /* 门控是「本次奖励事件」的一次性调制，不能让 e[] 跨试次累积，
     * 否则 m>1 时会指数增长到 inf，把权重推到 clamp 边界形成假学习。
     * 每个试次从零迹开始，与 host 的逐试次信用分配语义一致。*/
    for (i = 0; i < g_net.synapse_count; i++) {
        clear_traces(&g_net.synapses[i]);
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

void b1p_destroy(void);

int b1p_create(int sensory_a, int sensory_b, int hidden,
               int output_a, int output_b, double avg_conn,
               unsigned int seed) {
    int s, hbase, obase;
    NeuroSeedBundle bundle;
    uint64_t base[NEURO_STREAM_COUNT];
    if (g_ready) { b1p_destroy(); }

    g_master_seed = (uint64_t)seed;
    g_last_d = g_last_gate_diff = g_last_mirror_diff = 0.0;
    g_last_baseline = g_last_hidden_l = g_last_hidden_r = 0.0;
    neuro_seed_bundle_init(&bundle, g_master_seed);
    base[NEURO_STREAM_STRUCTURE]   = bundle.structure;
    base[NEURO_STREAM_WEIGHT]      = bundle.weight;
    base[NEURO_STREAM_NOISE]       = bundle.noise;
    base[NEURO_STREAM_TRIAL_ORDER] = bundle.trial_order;
    for (s = 0; s < NEURO_STREAM_COUNT; s++) {
        net_override_seed_stream(s, g_ov_on[s] ? g_ov_val[s] : base[s]);
    }
    g_shuffle_state = psel_splitmix64(&g_master_seed) ^ 0xC3D2E1F0A5B4C3D2ULL;

    memset(&g_net, 0, sizeof(g_net));
    if (net_init(&g_net, sensory_a, sensory_b, hidden,
                 output_a, output_b, avg_conn, seed) != 0) return -1;
    g_ready = 1;

    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = hbase + g_net.hidden_count;
    g_orig = (Edge*)calloc((size_t)g_net.synapse_count, sizeof(Edge));
    g_idx  = (int*)calloc((size_t)g_net.synapse_count, sizeof(int));
    if (!g_orig || !g_idx) { b1p_destroy(); return -2; }
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

void b1p_destroy(void) {
    free_snapshot();
    if (g_ready) { net_destroy(&g_net); g_ready = 0; }
}

int b1p_ready(void) { return g_ready; }
int b1p_synapse_count(void) { return g_ready ? g_net.synapse_count : 0; }
int b1p_readout_edge_count(void) { return g_m; }

/* ---------- M0 四流契约（create 之前调用） ---------- */

void b1p_override_stream(int which, uint64_t value) {
    if (which < 0 || which >= NEURO_STREAM_COUNT) return;
    if (value == 0) {
        g_ov_on[which] = 0;
        g_ov_val[which] = 0;
    } else {
        g_ov_on[which] = 1;
        g_ov_val[which] = value;
    }
}
void b1p_legacy_rng(int on) { net_use_legacy_rng(on); }

/* ---------- 超参数 / 消融（BASE 口径固定，禁止扫描刷分） ---------- */

void b1p_set_hyper(double learning_rate, double psp_factor, int trace_k) {
    if (!g_ready) return;
    if (learning_rate > 0.0) g_net.hp.learning_rate = learning_rate;
    if (psp_factor > 0.0) g_net.hp.psp_factor = psp_factor;
    if (trace_k >= 1 && trace_k <= 3) g_net.hp.trace_k = trace_k;
}

void b1p_set_ablation(int no_trace, int no_reward) {
    g_no_trace = no_trace ? 1 : 0;
    g_no_reward = no_reward ? 1 : 0;
    if (g_ready) {
        g_net.use_eligibility_trace = g_no_trace ? 0 : 1;
        g_net.use_reward_modulation = g_no_reward ? 0 : 1;
    }
}

void b1p_set_selection(int mode, double lambda, uint64_t shuffle_seed) {
    g_sel_mode = (mode >= 0 && mode <= 6) ? mode : 0;
    g_sel_lambda = (lambda >= 0.0) ? lambda : 0.0;
    if (shuffle_seed != 0) g_shuffle_state = shuffle_seed;
}

void b1p_set_reward_mode(int mode) {
    g_reward_mode = (mode >= 0 && mode <= 3) ? mode : 0;
}

void b1p_set_couple(int on) { g_couple = on ? 1 : 0; }

void b1p_set_freeze_readout(int on) { g_freeze_readout = on ? 1 : 0; }

double b1p_last_d(void) { return g_last_d; }
double b1p_last_hidden_l(void) { return g_last_hidden_l; }
double b1p_last_hidden_r(void) { return g_last_hidden_r; }

/* ---------- 严格对称基线（构造同 benchmark1_sym.c / t1.c） ---------- */

double b1p_asymmetry(void) {
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
double b1p_symmetrize(double alpha) {
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
        return b1p_asymmetry();
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
    return b1p_asymmetry();
}

/* ---------- 只读诊断 ---------- */

void b1p_output_bias(double out[4]) {
    if (out) { out[0] = out[1] = out[2] = out[3] = 0.0; }
    if (g_ready) net_output_bias(&g_net, out);
}

double b1p_weight_checksum(void) { return g_ready ? checksum() : 0.0; }

double b1p_readout_checksum(void) {
    double s = 0.0;
    int k;
    if (!g_ready) return 0.0;
    for (k = 0; k < g_m; k++) s += g_net.synapses[g_idx[k]].weight;
    return s;
}

/* ---------- 通路特异性资格迹：核心机制 ---------- */

/* 在奖励到达前一 tick 调用。只改 Synapse.e[] / eligibility_trace，
 * 不写权重、不改 winner、不绕过核心 R-STDP。返回 |m_A - m_B|。 */
static double apply_pathway_gate(double d) {
    double ma = 1.0 + g_sel_lambda * d;
    double mb = 1.0 - g_sel_lambda * d;
    int s, k, pairs;
    double mirror = 0.0;
    if (!g_ready) return 0.0;
    for (s = 0; s < g_net.synapse_count; s++) {
        Synapse* syn = &g_net.synapses[s];
        NeuronType pt = g_net.neurons[syn->post_id].type;
        double m;
        if (pt == NT_OUTPUT_A) m = ma;
        else if (pt == NT_OUTPUT_B) m = mb;
        else continue;
        for (k = 0; k < 3; k++) {
            syn->e[k] *= m;
            /* 与核心 update_eligibility_traces 相同的 ±5 截断，防止门控放大越界 */
            if (syn->e[k] > 5.0) syn->e[k] = 5.0;
            if (syn->e[k] < -5.0) syn->e[k] = -5.0;
        }
        syn->eligibility_trace *= m;
    }
    /* 自证：镜像边对的迹不再相同（严格对称 + 无门控时该值恒为 0） */
    pairs = g_m / 2;
    if (pairs > 0) {
        for (k = 0; k < pairs; k++) {
            double t0 = g_net.synapses[g_idx[2 * k]].eligibility_trace;
            double t1 = g_net.synapses[g_idx[2 * k + 1]].eligibility_trace;
            mirror += fabs(t0 - t1);
        }
        mirror /= (double)pairs;
    }
    g_last_gate_diff = fabs(ma - mb);
    g_last_mirror_diff = mirror;
    return g_last_gate_diff;
}

/* 观察窗口内累计 hidden 左/右半发放 —— 纯网络内部状态，不读 stim 标签 */
static void accum_hidden(int* hl, int* hr) {
    int i, hbase, mid;
    if (!g_ready) return;
    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    mid = g_net.hidden_count / 2;
    for (i = 0; i < g_net.hidden_count; i++) {
        if (!g_net.neurons[hbase + i].is_spiking) continue;
        if (i < mid) (*hl)++;
        else (*hr)++;
    }
}

/* ---------- g=0 冻结投票（无门控、无奖励、不改权重） ---------- */

int b1p_frozen_vote(int stim, int* a_cnt, int* b_cnt) {
    int t, i, a = 0, b = 0, pred;
    int saved_mod = 0;
    if (!g_ready) return -1;
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

/* ---------- 完整试次（复刻 host 协议 + 通路门控） ---------- */

/* out[0]=a, [1]=b, [2]=即时 correct(含注入，非能力口径),
 * [3]=注入的奖励标量, [4]=本试次权重 checksum 变化, [5]=tick 数,
 * [6]=实际门控量 g（含 sign(reward) 耦合）, [7]=|m_A-m_B|, [8]=镜像边迹平均绝对差,
 * [9]=注入前 reward_baseline */
int b1p_run_trial(int stim, int learning_on, int delay, double out[10]) {
    int t, i, a = 0, b = 0, pred, expected, correct;
    int hl = 0, hr = 0, ticks = 0, arrival, delivery, gate_tick, pairs;
    double reward, desired, w0, d, d_raw, sgn, gate_d, denom;
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
        net_tick(&g_net);
        ticks++;
        accum_hidden(&hl, &hr);
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

    /* 内部读出 d_raw：只用 hidden 层的实际发放，与 stim 标签无直接依赖 */
    denom = (double)(hl + hr);
    d_raw = (denom > 0.0) ? ((double)(hl - hr) / denom) : 0.0;
    g_last_hidden_l = (double)hl;
    g_last_hidden_r = (double)hr;

    /* 第三因子：B1 原口径结果反馈 / 内部预测读出（mode=3，机制诊断） */
    if (g_reward_mode == 3) {
        desired = (fabs(d_raw) < 1e-9) ? 0.0 : ((d_raw > 0.0) ? 1.0 : -1.0);
    } else if (pred == 0) {
        desired = (g_reward_mode == 0) ? 0.0 : -1.0;
    } else {
        desired = correct ? 1.0 : -1.0;
    }
    g_last_baseline = g_net.reward_baseline;
    reward = desired;
    if (g_reward_mode == 2) reward = desired + g_net.reward_baseline;

    /* 门控量：内部量经模式变换后，与第三因子符号耦合。
     * 严格对称下 Δw_A − Δw_B ∝ reward_error * (m_A − m_B)；
     * 耦合 sign(reward) 后该差值恒正比于 |d|，在奖惩两种反馈下方向都正确。*/
    d = d_raw;
    if (g_sel_mode == 2) {
        /* 幅度不变、符号来自独立流 -> 与刺激独立，用于检测信息是否真来自内部状态 */
        double sign = (psel_splitmix64(&g_shuffle_state) >> 63) ? 1.0 : -1.0;
        d = sign * fabs(d);
    } else if (g_sel_mode == 3) {
        d = 1.0;
    } else if (g_sel_mode == 4) {
        d = -d;
    } else if (g_sel_mode == 5) {
        d = (fabs(d_raw) < 1e-9) ? 0.0 : ((d_raw > 0.0) ? 1.0 : -1.0);
    } else if (g_sel_mode == 6) {
        d = (fabs(d_raw) < 1e-9) ? 0.0 : ((d_raw > 0.0) ? -1.0 : 1.0);
    }
    g_last_d = d;
    sgn = (desired > 0.0) ? 1.0 : ((desired < 0.0) ? -1.0 : 0.0);
    gate_d = g_couple ? (d * sgn) : d;
    g_last_gate_diff = 0.0;
    g_last_mirror_diff = 0.0;

    if (learning_on) net_inject_reward(&g_net, reward, delay);

    delivery = g_net.current_tick + delay;
    gate_tick = (delay > 0) ? delivery - 1 : -1;
    if (learning_on && g_sel_mode != 0 && g_sel_lambda > 0.0 && delay == 0) {
        apply_pathway_gate(gate_d);
    }
    arrival = delivery + 1;
    while (g_net.current_tick < arrival) {
        if (learning_on && g_sel_mode != 0 && g_sel_lambda > 0.0 &&
            gate_tick >= 0 && g_net.current_tick == gate_tick) {
            apply_pathway_gate(gate_d);
        }
        net_tick(&g_net);
        ticks++;
    }
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

    pairs = g_m / 2;
    if (out) {
        out[0] = (double)a;
        out[1] = (double)b;
        out[2] = (double)correct;
        out[3] = reward;
        out[4] = checksum() - w0;
        out[5] = (double)ticks;
        out[6] = gate_d;
        out[7] = g_last_gate_diff;
        out[8] = g_last_mirror_diff;
        out[9] = g_last_baseline;
    }
    (void)pairs;
    return pred;
}
