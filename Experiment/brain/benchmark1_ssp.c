/* ============================================================
 * benchmark1_ssp.c - B1「时序差分自预测」自监督目标实验台
 *                    判据 v1.7 §6.7 / 大纲附录 I.4 的下一步
 *
 * §6.7（可学习内部前向模型）把附录 H 的三个人工成分全部换成了可学习量，
 * 仍 FAIL；瓶颈定位在**自监督读出**：Oja 学 hidden 发放向量的第一主成分，
 * 学到的是「共同的大方差成分」，不是「A/B 判别方向」（该方向方差很小），
 * 所以 z_stim_gap 只有 0.071。附录 I.4 要求寻找一个既不依赖外部标签、
 * 又能指向任务语义的自监督目标，先验证它能产生与刺激相关、零均值、
 * 幅度稳定的内部信号，再重做配对消融。
 *
 * 本实验台的核心是**时序差分自预测**：B1 协议固定 A/B 交替，连续两个
 * 试次的 hidden 状态之差 Δh = ĥ_t − ĥ_{t−1} 会抵消共同成分、放大判别
 * 方向（对比/差分自监督，只用网络自身状态的时间结构，无外部标签、
 * 无 left/right 结构先验）。对 Δh 做 Oja 得到判别方向 v，再用
 *   u = tanh(Z_GAIN * (v·ĥ_t))
 * 作为内部信号：ĥ_A 与 ĥ_B 在 v 上的投影符号相反 -> 与刺激相关、
 * 交替正负 -> 零均值、幅度稳定（§6.3 已证 hidden 模式类内方差为 0）。
 *
 * 三个可学习量（延续 §6.7 的架构，只替换读出目标）：
 *   1. 差分 Oja 读出 v（mode=1）；mode=0 为 §6.7 的「对 ĥ 做 Oja」对照。
 *   2. Actor 系数 theta：g = 0.9*tanh(theta*u)，由内部噪声流随机初始化，
 *      用优势 A 在线校准方向（无人工方向）。
 *   3. Critic 基线 b / 优势 A：b <- b + eta_c*(y-b), A = y-b；
 *      注入 reward = A + g_net.reward_baseline，使核心 reward_error = A。
 * 门控仍只改 Synapse.e[]（m_A=1+λg / m_B=1−λg），保留 §6.6 的
 * 「每试次清迹 + ±5 截断」数值修复。
 *
 * 诚实边界：
 *   - 利用了「试次交替」这一 B1 协议的时间结构，但没有读取 stim 标签、
 *     没有使用 §6.5 的外部教学信号、没有预设 left/right 语义。
 *   - y 仍是 B1 结果反馈。g_y_mode=0 为 v1.8 的「平局视为失败(-1)」，
 *     已确认偏离 host；g_y_mode=1 为 host 原口径（平局 reward 0.0）。
 *   - 新增 calib：actor 增量除优势 A 外可叠加同类差分 d_same 与
 *     A/B 配对差分 d_pair（§6.8 下一步）。默认 calib=0 / y_mode=0
 *     与原实现逐位一致，用于回归比对 v1.8 的 30.00% / 30.00pp。
 *   - 前向模型自身常数（ETA_*、Z_GAIN、w_same=w_pair=0.5）固定未扫描；
 *     核心 lr=0.02 / psp=10 / trace_k=3 与判据阈值一律零改动。
 *   - 能力口径 g=0：训练后关闭全部注入，冻结投票，评估前后权重漂移 0；
 *     同时报告 A/B 预测分布差，防「恒定答案」冒充映射。
 *   - 即使有效也只证明「时序差分自监督目标能桥接 hidden 与输出选择」，
 *     **不构成 B1 通过**（新增了内部结构）。
 *
 * 编译：gcc -shared ... brain/benchmark1_ssp.c b1_core.o seed_bundle.o
 *       -o b1_ssp.dll -lm（见 build.ps1）。独立 DLL 可单独删除。
 * ============================================================ */

#include "benchmark1_core.h"
#include "seed_bundle.h"

#include <stdint.h>

#define OBSERVE_WINDOW 15
#define STIM_DURATION  10
#define COOLDOWN_TICKS 30

#define ETA_OJA    0.01
#define ETA_ACTOR  0.05
#define ETA_CRITIC 0.05
#define THETA_MAX  3.0
#define Z_GAIN     8.0

static NetworkState g_net;
static int g_ready = 0;

typedef struct {
    int pre;
    int post;
    double w;
} Edge;

static Edge* g_orig = NULL;
static int*  g_idx  = NULL;
static int   g_m    = 0;

static double* g_h = NULL;       /* 本试次 hidden 发放向量（归一化后为 ĥ） */
static double* g_h_prev = NULL;  /* 上一试次 ĥ（跨试次保留） */
static double* g_delta = NULL;   /* Δ = ĥ_t − ĥ_{t−1} */
static double* g_v = NULL;       /* 可学习读出方向 */
static int     g_hn = 0;
static int     g_has_prev = 0;

static double g_theta = 0.0, g_theta0 = 0.0, g_b = 0.0;

/* 模式与消融 */
static int      g_no_trace = 0, g_no_reward = 0, g_freeze_readout = 0;
static int      g_mode = 1;          /* 1=差分 Oja（新）；0=对 ĥ 做 Oja（§6.7 对照） */
static int      g_adv_mode = 1;      /* 0=host 原口径；1=可学习优势；2=冻结 critic */
static int      g_learn_theta = 1;
static int      g_learn_v = 1;       /* 0=冻结 v（随机投影对照） */
static int      g_u_mode = 0;        /* 0=学出的 u；1=恒定；2=shuffle；3=结构先验参照 */
static double   g_lambda = 1.0;
static uint64_t g_model_state = 0;

/* ---- §6.8 下一步（v1.9）：结果反馈口径 + actor 方向校准 ---- */
/* y_mode: 0 = 平局视为失败 -1（v1.8 口径，偏离 host）
 *         1 = host 原口径：平局 reward 0.0（host/benchmark1_host.cpp:368-371） */
static int      g_y_mode = 0;
/* calib: 0 = 仅优势 A（v1.8 原规则）
 *        1 = + 同类试次反馈差 d_same
 *        2 = + A/B 配对反馈差 d_pair
 *        3 = 两者都用（§6.8「同时利用 A/B 试次的结果反馈差」） */
static int      g_calib = 0;
static double   g_w_same = 0.5, g_w_pair = 0.5;

static double   g_last_y[2] = { 0.0, 0.0 };
static int      g_has_y[2] = { 0, 0 };
static double   g_prev_y = 0.0;
static int      g_has_prev_y = 0;   /* 上一个试次（交替结构下为异类）的 y */
static double   g_last_d_same = 0.0, g_last_d_pair = 0.0, g_last_delta = 0.0;

static uint64_t g_ov_val[NEURO_STREAM_COUNT];
static int      g_ov_on[NEURO_STREAM_COUNT];

/* 诊断 */
static double g_last_u = 0.0, g_last_raw = 0.0, g_last_delta_norm = 0.0;
static double g_last_d_raw = 0.0, g_last_vnorm = 1.0;
static double g_last_A = 0.0, g_last_gate_diff = 0.0, g_last_mirror_diff = 0.0;
static double g_last_baseline = 0.0;

double b1s_asymmetry(void);

static uint64_t ssp_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double ssp_uniform11(void) {
    uint64_t r = ssp_splitmix64(&g_model_state);
    return ((double)(r >> 11)) / 9007199254740992.0 * 2.0 - 1.0;
}

static void free_snapshot(void) {
    free(g_orig);   g_orig = NULL;
    free(g_idx);    g_idx = NULL;
    free(g_h);      g_h = NULL;
    free(g_h_prev); g_h_prev = NULL;
    free(g_delta);  g_delta = NULL;
    free(g_v);      g_v = NULL;
    g_m = 0; g_hn = 0;
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
    for (i = 0; i < g_net.synapse_count; i++) clear_traces(&g_net.synapses[i]);
    if (g_h) memset(g_h, 0, (size_t)g_hn * sizeof(double));
    /* g_h_prev 跨试次保留，这是时序差分的前提 */
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

void b1s_destroy(void);

int b1s_create(int sensory_a, int sensory_b, int hidden,
               int output_a, int output_b, double avg_conn,
               unsigned int seed) {
    int s, i, hbase, obase;
    NeuroSeedBundle bundle;
    uint64_t base[NEURO_STREAM_COUNT];
    uint64_t master = (uint64_t)seed;
    if (g_ready) { b1s_destroy(); }

    neuro_seed_bundle_init(&bundle, master);
    base[NEURO_STREAM_STRUCTURE]   = bundle.structure;
    base[NEURO_STREAM_WEIGHT]      = bundle.weight;
    base[NEURO_STREAM_NOISE]       = bundle.noise;
    base[NEURO_STREAM_TRIAL_ORDER] = bundle.trial_order;
    for (s = 0; s < NEURO_STREAM_COUNT; s++) {
        net_override_seed_stream(s, g_ov_on[s] ? g_ov_val[s] : base[s]);
    }
    g_model_state = ssp_splitmix64(&master) ^ 0x5A5A1234BEEF6789ULL;

    memset(&g_net, 0, sizeof(g_net));
    if (net_init(&g_net, sensory_a, sensory_b, hidden,
                 output_a, output_b, avg_conn, seed) != 0) return -1;
    g_ready = 1;

    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = hbase + g_net.hidden_count;
    g_hn = g_net.hidden_count;
    g_orig    = (Edge*)calloc((size_t)g_net.synapse_count, sizeof(Edge));
    g_idx     = (int*)calloc((size_t)g_net.synapse_count, sizeof(int));
    g_h       = (double*)calloc((size_t)g_hn, sizeof(double));
    g_h_prev  = (double*)calloc((size_t)g_hn, sizeof(double));
    g_delta   = (double*)calloc((size_t)g_hn, sizeof(double));
    g_v       = (double*)calloc((size_t)g_hn, sizeof(double));
    if (!g_orig || !g_idx || !g_h || !g_h_prev || !g_delta || !g_v) {
        b1s_destroy(); return -2;
    }
    {
        double norm = 0.0;
        for (i = 0; i < g_hn; i++) {
            g_v[i] = ssp_uniform11();
            norm += g_v[i] * g_v[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-9) for (i = 0; i < g_hn; i++) g_v[i] /= norm;
        g_theta0 = 0.5 * ssp_uniform11();
    }
    g_theta = g_theta0;
    g_b = 0.0;
    g_has_prev = 0;
    g_last_u = g_last_raw = g_last_delta_norm = 0.0;
    g_last_d_raw = g_last_A = 0.0;
    g_last_vnorm = 1.0;
    g_last_gate_diff = g_last_mirror_diff = g_last_baseline = 0.0;
    g_last_y[0] = g_last_y[1] = 0.0;
    g_has_y[0] = g_has_y[1] = 0;
    g_prev_y = 0.0;
    g_has_prev_y = 0;
    g_last_d_same = g_last_d_pair = g_last_delta = 0.0;

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

void b1s_override_stream(int which, uint64_t value) {
    if (which < 0 || which >= NEURO_STREAM_COUNT) return;
    if (value == 0) { g_ov_on[which] = 0; g_ov_val[which] = 0; }
    else { g_ov_on[which] = 1; g_ov_val[which] = value; }
}
void b1s_legacy_rng(int on) { net_use_legacy_rng(on); }

/* ---------- 配置 ---------- */

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

void b1s_set_model(int mode, int adv, int learn_theta, int learn_v,
                   int u_mode, double lambda, uint64_t model_seed) {
    g_mode = mode ? 1 : 0;
    g_adv_mode = (adv >= 0 && adv <= 2) ? adv : 1;
    g_learn_theta = learn_theta ? 1 : 0;
    g_learn_v = learn_v ? 1 : 0;
    g_u_mode = (u_mode >= 0 && u_mode <= 3) ? u_mode : 0;
    g_lambda = (lambda >= 0.0) ? lambda : 0.0;
    if (model_seed != 0 && g_ready && g_v) {
        int i;
        double norm = 0.0;
        g_model_state = model_seed;
        for (i = 0; i < g_hn; i++) {
            g_v[i] = ssp_uniform11();
            norm += g_v[i] * g_v[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-9) for (i = 0; i < g_hn; i++) g_v[i] /= norm;
        g_theta0 = 0.5 * ssp_uniform11();
        g_theta = g_theta0;
        g_b = 0.0;
        g_has_prev = 0;
        g_last_y[0] = g_last_y[1] = 0.0;
        g_has_y[0] = g_has_y[1] = 0;
        g_prev_y = 0.0;
        g_has_prev_y = 0;
        g_last_d_same = g_last_d_pair = g_last_delta = 0.0;
    }
}

void b1s_set_freeze_readout(int on) { g_freeze_readout = on ? 1 : 0; }

/* §6.8 下一步：calib = actor 更新是否叠加 A/B 结果反馈差；
 * y_mode = 结果反馈口径（0 = v1.8 平局 -1；1 = host 平局 0）。 */
void b1s_set_calib(int calib, int y_mode) {
    g_calib = (calib >= 0 && calib <= 3) ? calib : 0;
    g_y_mode = y_mode ? 1 : 0;
}
void b1s_set_calib_weights(double w_same, double w_pair) {
    g_w_same = (w_same >= 0.0) ? w_same : 0.0;
    g_w_pair = (w_pair >= 0.0) ? w_pair : 0.0;
}
double b1s_last_d_same(void) { return g_last_d_same; }
double b1s_last_d_pair(void) { return g_last_d_pair; }
double b1s_last_delta(void) { return g_last_delta; }
int    b1s_calib_mode(void) { return g_calib; }
int    b1s_y_mode(void) { return g_y_mode; }

double b1s_last_u(void) { return g_last_u; }
double b1s_last_raw(void) { return g_last_raw; }
double b1s_last_delta_norm(void) { return g_last_delta_norm; }
double b1s_last_theta(void) { return g_theta; }
double b1s_last_theta0(void) { return g_theta0; }
double b1s_last_b(void) { return g_b; }
double b1s_last_A(void) { return g_last_A; }
double b1s_last_vnorm(void) { return g_last_vnorm; }

/* ---------- 严格对称基线 ---------- */

double b1s_asymmetry(void) {
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

double b1s_symmetrize(double alpha) {
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
    if (g_m % 2 == 1) {
        Synapse* s = &g_net.synapses[g_idx[g_m - 1]];
        s->pre_id = g_orig[g_m - 1].pre;
        s->post_id = g_orig[g_m - 1].post;
        s->weight = (1.0 - a) * g_orig[g_m - 1].w;
        clear_traces(s);
    }
    return b1s_asymmetry();
}

void b1s_output_bias(double out[4]) {
    if (out) { out[0] = out[1] = out[2] = out[3] = 0.0; }
    if (g_ready) net_output_bias(&g_net, out);
}
double b1s_weight_checksum(void) { return g_ready ? checksum() : 0.0; }
double b1s_readout_checksum(void) {
    double s = 0.0;
    int k;
    if (!g_ready) return 0.0;
    for (k = 0; k < g_m; k++) s += g_net.synapses[g_idx[k]].weight;
    return s;
}

/* ---------- 时序差分自预测（核心机制） ---------- */

static void accum_hidden(void) {
    int i, hbase;
    if (!g_ready || !g_h) return;
    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    for (i = 0; i < g_hn; i++) {
        if (g_net.neurons[hbase + i].is_spiking) g_h[i] += 1.0;
    }
}

/* 归一化 g_h -> ĥ，算结构先验参照量，构造 Δ，对 Δ（或 ĥ）做 Oja，
 * 返回内部信号 u = tanh(Z_GAIN * (v·ĥ))。*/
static double ssp_step(void) {
    int i, mid;
    double hn = 0.0, sl = 0.0, sr = 0.0, dn = 0.0, proj = 0.0, vn = 0.0;
    double* feat;
    if (!g_ready || !g_h || !g_v) return 0.0;
    mid = g_hn / 2;
    for (i = 0; i < g_hn; i++) {
        if (i < mid) sl += g_h[i];
        else sr += g_h[i];
    }
    g_last_d_raw = (sl + sr > 1e-9) ? ((sl - sr) / (sl + sr)) : 0.0;
    for (i = 0; i < g_hn; i++) hn += g_h[i] * g_h[i];
    hn = sqrt(hn);
    if (hn < 1e-9) return 0.0;
    for (i = 0; i < g_hn; i++) g_h[i] /= hn;

    feat = g_h;   /* 默认：对 ĥ 做 Oja（§6.7 对照，mode=0） */
    if (g_mode == 1 && g_has_prev) {
        for (i = 0; i < g_hn; i++) g_delta[i] = g_h[i] - g_h_prev[i];
        for (i = 0; i < g_hn; i++) dn += g_delta[i] * g_delta[i];
        dn = sqrt(dn);
        if (dn > 1e-9) {
            for (i = 0; i < g_hn; i++) g_delta[i] /= dn;
            feat = g_delta;   /* 差分自监督：对 Δ̂ 做 Oja */
        }
    }
    g_last_delta_norm = dn;
    for (i = 0; i < g_hn; i++) proj += g_v[i] * feat[i];

    if (g_learn_v) {
        for (i = 0; i < g_hn; i++) {
            g_v[i] += ETA_OJA * proj * (feat[i] - proj * g_v[i]);
        }
        for (i = 0; i < g_hn; i++) vn += g_v[i] * g_v[i];
        vn = sqrt(vn);
        if (vn > 1e-9) for (i = 0; i < g_hn; i++) g_v[i] /= vn;
    }
    vn = 0.0;
    for (i = 0; i < g_hn; i++) vn += g_v[i] * g_v[i];
    g_last_vnorm = sqrt(vn);

    /* 内部信号用当前 ĥ 在 v 上的投影（不是 Δ），保证与当前试次对应 */
    {
        double raw = 0.0;
        for (i = 0; i < g_hn; i++) raw += g_v[i] * g_h[i];
        g_last_raw = raw;
        g_last_u = tanh(Z_GAIN * raw);
    }
    for (i = 0; i < g_hn; i++) g_h_prev[i] = g_h[i];
    g_has_prev = 1;
    return g_last_u;
}

static double apply_pathway_gate(double g) {
    double ma = 1.0 + g_lambda * g;
    double mb = 1.0 - g_lambda * g;
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
            if (syn->e[k] > 5.0) syn->e[k] = 5.0;
            if (syn->e[k] < -5.0) syn->e[k] = -5.0;
        }
        syn->eligibility_trace *= m;
    }
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

/* ---------- g=0 冻结投票 ---------- */

int b1s_frozen_vote(int stim, int* a_cnt, int* b_cnt) {
    int t, i, a = 0, b = 0, pred, saved_mod;
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

/* ---------- 完整试次 ---------- */

/* out[0]=a, [1]=b, [2]=即时 correct, [3]=注入 reward, [4]=权重 checksum 变化,
 * [5]=tick 数, [6]=门控量 g, [7]=|m_A-m_B|, [8]=镜像边迹平均绝对差, [9]=优势 A */
int b1s_run_trial(int stim, int learning_on, int delay, double out[10]) {
    int t, i, a = 0, b = 0, pred, expected, correct;
    int ticks = 0, arrival, delivery, gate_tick;
    double y, A, reward, u, gate, w0, delta;
    double* ro_saved = NULL;
    if (!g_ready) return -1;
    w0 = checksum();
    if (g_freeze_readout) {
        ro_saved = (double*)calloc((size_t)g_m, sizeof(double));
        if (ro_saved) for (i = 0; i < g_m; i++) ro_saved[i] = g_net.synapses[g_idx[i]].weight;
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
        accum_hidden();
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

    /* 结果反馈口径：g_y_mode=1 为 host 原口径（平局 reward 0.0，
     * 见 host/benchmark1_host.cpp:368-371）；0 为 v1.8 的「平局视为失败 -1」。 */
    if (g_y_mode == 1) y = (pred == 0) ? 0.0 : (correct ? 1.0 : -1.0);
    else               y = (pred == 0) ? -1.0 : (correct ? 1.0 : -1.0);
    if (learning_on) g_b += ETA_CRITIC * (y - g_b);
    A = (g_adv_mode == 1) ? (y - g_b) : y;
    g_last_A = A;

    /* actor 方向校准（§6.8 下一步）：除优势 A 外，再用两类结果反馈差。
     * d_same = y_t − y_{同类上一次}；d_pair = s·(y_t − y_{上一试次})，
     * s=+1(A)/−1(B)，即「A 相对 B 的反馈优势」。A/B 交替协议下上一试次
     * 必为异类，因此 d_pair 只利用时间结构，不读取 stim 标签语义。 */
    {
        int c = (stim == 0) ? 0 : 1;
        double sgn = (c == 0) ? 1.0 : -1.0;
        g_last_d_same = g_has_y[c] ? (y - g_last_y[c]) : 0.0;
        g_last_d_pair = g_has_prev_y ? (sgn * (y - g_prev_y)) : 0.0;
        g_last_y[c] = y;
        g_has_y[c] = 1;
        g_prev_y = y;
        g_has_prev_y = 1;
        delta = A;
        if (g_calib & 1) delta += g_w_same * g_last_d_same;
        if (g_calib & 2) delta += g_w_pair * g_last_d_pair;
        g_last_delta = delta;
    }

    u = ssp_step();
    if (g_u_mode == 1) u = 1.0;
    else if (g_u_mode == 2) {
        u = (ssp_splitmix64(&g_model_state) >> 63) ? fabs(u) : -fabs(u);
    } else if (g_u_mode == 3) {
        u = g_last_d_raw;
    }
    g_last_u = u;

    gate = 0.9 * tanh(g_theta * u);
    g_last_baseline = g_net.reward_baseline;
    g_last_gate_diff = 0.0;
    g_last_mirror_diff = 0.0;

    reward = (g_adv_mode == 0) ? y : (A + g_net.reward_baseline);
    if (learning_on) net_inject_reward(&g_net, reward, delay);

    delivery = g_net.current_tick + delay;
    gate_tick = (delay > 0) ? delivery - 1 : -1;
    if (learning_on && g_lambda > 0.0 && delay == 0) apply_pathway_gate(gate);
    arrival = delivery + 1;
    while (g_net.current_tick < arrival) {
        if (learning_on && g_lambda > 0.0 &&
            gate_tick >= 0 && g_net.current_tick == gate_tick) {
            apply_pathway_gate(gate);
        }
        net_tick(&g_net);
        ticks++;
    }
    for (t = 0; t < COOLDOWN_TICKS; t++) { net_tick(&g_net); ticks++; }

    if (learning_on && g_learn_theta) {
        g_theta += ETA_ACTOR * g_last_delta * u;
        if (g_theta > THETA_MAX) g_theta = THETA_MAX;
        if (g_theta < -THETA_MAX) g_theta = -THETA_MAX;
    }

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
        out[6] = gate;
        out[7] = g_last_gate_diff;
        out[8] = g_last_mirror_diff;
        out[9] = A;
    }
    return pred;
}
