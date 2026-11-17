/* ============================================================
 * benchmark1_fwd.c - B1「可学习内部前向模型」机制实验台
 *                    判据 v1.6 §6.6 / 大纲附录 H.4 的下一步
 *
 * 上一轮（b1_psel, §6.6 / 附录 H）已把「通路选择性」做进网络内部的资格迹，
 * 并证明它能破缺对称、产生可归因学习；但留下三个人工成分：
 *   (a) 内部读出 d 用了结构先验（感觉A->hidden左半->输出A 为「正确路径」）；
 *   (b) 学习方向靠人工翻转门控符号（sel_mode=6）才对；
 *   (c) 核心 reward_baseline 饱和到 -0.997，使 reward_error 幅度趋零，
 *       只有靠「基线补偿诊断组」才能出效果，而该组操纵了注入值。
 * 附录 H.4 要求：把「幅度 / 符号 / 基线」做成网络内部**可学习的**量，
 * 替代结构先验与人工方向，回到 host 原结果反馈口径，再重做配对消融。
 *
 * 本实验台用三个**在线可学习**的内部量替换上述三个人工成分：
 *   1. Oja 自监督读出 v（替代结构先验）：
 *      观察窗口累计 hidden 发放向量 h（400 维，**不预设左/右半**），
 *      v 由独立噪声流小随机初始化，用 Oja 规则在线学第一主成分：
 *        proj = v·(h/|h|);  v <- normalize(v + eta_o*proj*(h/|h| - proj*v))
 *      内部隐变量 z = proj。**无外部标签、无拓扑先验、无人工方向**：
 *      符号由随机初始化决定，相关性由网络自身活动学出。
 *   2. Actor 系数 theta（替代人工方向反转）：
 *      门控量 g = 0.9*tanh(theta*z)。theta 由独立噪声流小随机初始化
 *      （网络内部探索，不是人工指定方向），用优势在线更新：
 *        theta <- clamp(theta + eta_a*A*z)
 *      与资格迹 e 符号一致的方向由学习自动校准，不再有 sel_mode=6。
 *   3. Critic 基线 b 与优势 A（替代饱和的 reward_baseline）：
 *        y = B1 结果反馈（pred 正确 +1 / 错误 -1 / 平局 -1 视为失败）
 *        b <- b + eta_c*(y - b);  A = y - b
 *      注入 reward = A + g_net.reward_baseline，使核心结算出的
 *      reward_error 精确等于优势 A；幅度不再被核心滑动基线抵消。
 *      （只读写 NetworkState 的公开字段，核心源码一行未改。）
 *
 * 门控与上一轮相同（只改 Synapse.e[]，不写权重、不改 winner）：
 *   m_A = 1 + lambda*g（OUTPUT_A 输入边）, m_B = 1 - lambda*g（OUTPUT_B 输入边）。
 * 每试次清零资格迹并 ±5 截断（§6.6 的数值修复必须保留）。
 *
 * 诚实边界：
 *   - 结果反馈 y 仍来自 pred 与期望的比较（B1 任务定义），「平局视为失败」
 *     是对 host 口径的最小改动，需随报告记录；未使用 §6.5 的 mode=1
 *     刺激相关外部教学信号。
 *   - 前向模型自身的学习率（eta_o / eta_a / eta_c）是**固定机制常数**，
 *     未扫描、未挑最优；核心 lr=0.02 / psp=10 / trace_k=3 一律不动。
 *   - 能力口径一律 g=0：训练结束后关闭全部门控与奖励，冻结投票评估，
 *     评估前后权重漂移必须为 0；同时报告 A/B 预测分布差。
 *   - 判据阈值、学习规则、刺激生成器、benchmark1_core.c/.h、probe/sym/
 *     t1/psel、host 一律零改动；核心 SHA1 与 §6.2–§6.6 报告一致。
 *   - 即使本机制有效，也只证明「可学习内部前向模型足以让局部 R-STDP
 *     学出映射」，**不构成 B1 通过**（前向模型是新增的内部结构）。
 *
 * 编译：gcc -shared ... brain/benchmark1_fwd.c b1_core.o seed_bundle.o
 *       -o b1_fwd.dll -lm（见 build.ps1）。独立 DLL 可单独删除。
 * ============================================================ */

#include "benchmark1_core.h"
#include "seed_bundle.h"

#include <stdint.h>

#define OBSERVE_WINDOW 15
#define STIM_DURATION  10
#define COOLDOWN_TICKS 30

/* 前向模型自身的固定学习常数（非核心超参，全程未扫描） */
#define ETA_OJA    0.01
#define ETA_ACTOR  0.05
#define ETA_CRITIC 0.05
#define THETA_MAX  3.0

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

/* hidden 发放向量与 Oja 读出（动态按 hidden_count 分配） */
static double* g_h = NULL;      /* 本试次 hidden 发放计数 */
static double* g_v = NULL;      /* 可学习内部读出 */
static int     g_hn = 0;

/* 可学习状态 */
static double g_theta = 0.0;   /* actor：门控方向/幅度 */
static double g_b = 0.0;       /* critic：结果反馈基线 */
static double g_theta0 = 0.0;  /* 初始探索值（诊断用） */

/* 模式与消融 */
static int      g_no_trace = 0;
static int      g_no_reward = 0;
static int      g_freeze_readout = 0;
static int      g_fwd_mode = 1;      /* 1=Oja 学 v；0=冻结 v（随机投影） */
static int      g_adv_mode = 1;      /* 1=可学习优势；0=host 原口径（基线饱和） */
static int      g_learn_theta = 1;   /* 1=actor 更新；0=冻结初始探索值 */
static int      g_z_mode = 0;        /* 0=学出的 z；1=恒定 z=1；2=符号 shuffle */
static double   g_lambda = 1.0;
static uint64_t g_model_state = 0;   /* 前向模型独立随机流（从 master 派生） */

/* 四流覆盖 */
static uint64_t g_ov_val[NEURO_STREAM_COUNT];
static int      g_ov_on[NEURO_STREAM_COUNT];

/* 最近一次试次诊断量 */
static double g_last_z = 0.0, g_last_A = 0.0, g_last_g = 0.0;
static double g_last_y = 0.0, g_last_proj = 0.0, g_last_vnorm = 0.0;
static double g_last_d_raw = 0.0;   /* 结构先验参照量（仅 z_mode=3 诊断组用） */
static double g_last_gate_diff = 0.0, g_last_mirror_diff = 0.0;
static double g_last_baseline = 0.0;

/* ---------- 工具 ---------- */

double b1f_asymmetry(void);

static uint64_t fwd_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* [-1, 1) 均匀，来自前向模型独立流 */
static double fwd_uniform11(void) {
    uint64_t r = fwd_splitmix64(&g_model_state);
    return ((double)(r >> 11)) / 9007199254740992.0 * 2.0 - 1.0;
}

static void free_snapshot(void) {
    free(g_orig); g_orig = NULL;
    free(g_idx);  g_idx  = NULL;
    free(g_h);    g_h    = NULL;
    free(g_v);    g_v    = NULL;
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
    /* §6.6 数值修复：每试次从零迹开始，门控不跨试次指数累积 */
    for (i = 0; i < g_net.synapse_count; i++) {
        clear_traces(&g_net.synapses[i]);
    }
    if (g_h) memset(g_h, 0, (size_t)g_hn * sizeof(double));
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

void b1f_destroy(void);

int b1f_create(int sensory_a, int sensory_b, int hidden,
               int output_a, int output_b, double avg_conn,
               unsigned int seed) {
    int s, i, hbase, obase;
    NeuroSeedBundle bundle;
    uint64_t base[NEURO_STREAM_COUNT];
    uint64_t master = (uint64_t)seed;
    if (g_ready) { b1f_destroy(); }

    neuro_seed_bundle_init(&bundle, master);
    base[NEURO_STREAM_STRUCTURE]   = bundle.structure;
    base[NEURO_STREAM_WEIGHT]      = bundle.weight;
    base[NEURO_STREAM_NOISE]       = bundle.noise;
    base[NEURO_STREAM_TRIAL_ORDER] = bundle.trial_order;
    for (s = 0; s < NEURO_STREAM_COUNT; s++) {
        net_override_seed_stream(s, g_ov_on[s] ? g_ov_val[s] : base[s]);
    }
    /* 前向模型随机流独立于 M0 四流，但由同一 master 派生，逐位可复现 */
    g_model_state = fwd_splitmix64(&master) ^ 0xA5B4C3D2E1F00123ULL;

    memset(&g_net, 0, sizeof(g_net));
    if (net_init(&g_net, sensory_a, sensory_b, hidden,
                 output_a, output_b, avg_conn, seed) != 0) return -1;
    g_ready = 1;

    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    obase = hbase + g_net.hidden_count;
    g_hn = g_net.hidden_count;
    g_orig = (Edge*)calloc((size_t)g_net.synapse_count, sizeof(Edge));
    g_idx  = (int*)calloc((size_t)g_net.synapse_count, sizeof(int));
    g_h    = (double*)calloc((size_t)g_hn, sizeof(double));
    g_v    = (double*)calloc((size_t)g_hn, sizeof(double));
    if (!g_orig || !g_idx || !g_h || !g_v) { b1f_destroy(); return -2; }

    /* v 与 theta 都从网络内部噪声流小随机初始化 —— 不预设任何方向 */
    {
        double norm = 0.0;
        for (i = 0; i < g_hn; i++) {
            g_v[i] = fwd_uniform11();
            norm += g_v[i] * g_v[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-9) {
            for (i = 0; i < g_hn; i++) g_v[i] /= norm;
        }
        g_theta0 = 0.5 * fwd_uniform11();
    }
    g_theta = g_theta0;
    g_b = 0.0;
    g_last_z = g_last_A = g_last_g = g_last_y = 0.0;
    g_last_proj = g_last_vnorm = 1.0;
    g_last_gate_diff = g_last_mirror_diff = g_last_baseline = 0.0;

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

void b1f_destroy(void) {
    free_snapshot();
    if (g_ready) { net_destroy(&g_net); g_ready = 0; }
}

int b1f_ready(void) { return g_ready; }
int b1f_synapse_count(void) { return g_ready ? g_net.synapse_count : 0; }
int b1f_readout_edge_count(void) { return g_m; }

void b1f_override_stream(int which, uint64_t value) {
    if (which < 0 || which >= NEURO_STREAM_COUNT) return;
    if (value == 0) { g_ov_on[which] = 0; g_ov_val[which] = 0; }
    else { g_ov_on[which] = 1; g_ov_val[which] = value; }
}
void b1f_legacy_rng(int on) { net_use_legacy_rng(on); }

/* ---------- 配置 ---------- */

void b1f_set_hyper(double learning_rate, double psp_factor, int trace_k) {
    if (!g_ready) return;
    if (learning_rate > 0.0) g_net.hp.learning_rate = learning_rate;
    if (psp_factor > 0.0) g_net.hp.psp_factor = psp_factor;
    if (trace_k >= 1 && trace_k <= 3) g_net.hp.trace_k = trace_k;
}

void b1f_set_ablation(int no_trace, int no_reward) {
    g_no_trace = no_trace ? 1 : 0;
    g_no_reward = no_reward ? 1 : 0;
    if (g_ready) {
        g_net.use_eligibility_trace = g_no_trace ? 0 : 1;
        g_net.use_reward_modulation = g_no_reward ? 0 : 1;
    }
}

/* fwd: 1=Oja 学 v, 0=冻结 v；adv: 1=可学习优势, 0=host 原口径；
 * learn_theta: 1=actor 更新, 0=冻结；z_mode: 0=学出, 1=恒定, 2=shuffle。*/
void b1f_set_model(int fwd, int adv, int learn_theta, int z_mode,
                   double lambda, uint64_t model_seed) {
    g_fwd_mode = fwd ? 1 : 0;
    g_adv_mode = adv ? 1 : 0;
    g_learn_theta = learn_theta ? 1 : 0;
    g_z_mode = (z_mode >= 0 && z_mode <= 3) ? z_mode : 0;
    g_lambda = (lambda >= 0.0) ? lambda : 0.0;
    if (model_seed != 0) {
        g_model_state = model_seed;
        if (g_ready && g_v) {
            int i;
            double norm = 0.0;
            for (i = 0; i < g_hn; i++) {
                g_v[i] = fwd_uniform11();
                norm += g_v[i] * g_v[i];
            }
            norm = sqrt(norm);
            if (norm > 1e-9) {
                for (i = 0; i < g_hn; i++) g_v[i] /= norm;
            }
            g_theta0 = 0.5 * fwd_uniform11();
            g_theta = g_theta0;
            g_b = 0.0;
        }
    }
}

void b1f_set_freeze_readout(int on) { g_freeze_readout = on ? 1 : 0; }

double b1f_last_theta(void) { return g_theta; }
double b1f_last_theta0(void) { return g_theta0; }
double b1f_last_b(void) { return g_b; }
double b1f_last_z(void) { return g_last_z; }
double b1f_last_A(void) { return g_last_A; }
double b1f_last_vnorm(void) { return g_last_vnorm; }

/* ---------- 严格对称基线（与 sym/t1/psel 同构） ---------- */

double b1f_asymmetry(void) {
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

double b1f_symmetrize(double alpha) {
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
        return b1f_asymmetry();
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
    return b1f_asymmetry();
}

void b1f_output_bias(double out[4]) {
    if (out) { out[0] = out[1] = out[2] = out[3] = 0.0; }
    if (g_ready) net_output_bias(&g_net, out);
}
double b1f_weight_checksum(void) { return g_ready ? checksum() : 0.0; }
double b1f_readout_checksum(void) {
    double s = 0.0;
    int k;
    if (!g_ready) return 0.0;
    for (k = 0; k < g_m; k++) s += g_net.synapses[g_idx[k]].weight;
    return s;
}

/* ---------- 前向模型 ---------- */

static void accum_hidden(void) {
    int i, hbase;
    if (!g_ready || !g_h) return;
    hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    for (i = 0; i < g_hn; i++) {
        if (g_net.neurons[hbase + i].is_spiking) g_h[i] += 1.0;
    }
}

/* 归一化 h 并计算 proj；Oja 在线更新 v（无外部标签） */
static double oja_step(void) {
    int i, mid;
    double hn = 0.0, proj = 0.0, vn = 0.0, sl = 0.0, sr = 0.0;
    if (!g_ready || !g_h || !g_v) return 0.0;
    /* 归一化前顺带算出上一轮的结构先验量，仅作瓶颈定位参照（z_mode=3）。
     * 该量不是本实验台的机制，也不进入任何验收。*/
    mid = g_hn / 2;
    for (i = 0; i < g_hn; i++) {
        if (i < mid) sl += g_h[i];
        else sr += g_h[i];
    }
    g_last_d_raw = (sl + sr > 1e-9) ? ((sl - sr) / (sl + sr)) : 0.0;
    for (i = 0; i < g_hn; i++) hn += g_h[i] * g_h[i];
    hn = sqrt(hn);
    if (hn < 1e-9) return 0.0;
    for (i = 0; i < g_hn; i++) g_h[i] /= hn;   /* g_h 变为单位向量 */
    for (i = 0; i < g_hn; i++) proj += g_v[i] * g_h[i];
    if (g_fwd_mode) {
        for (i = 0; i < g_hn; i++) {
            g_v[i] += ETA_OJA * proj * (g_h[i] - proj * g_v[i]);
        }
        for (i = 0; i < g_hn; i++) vn += g_v[i] * g_v[i];
        vn = sqrt(vn);
        if (vn > 1e-9) {
            for (i = 0; i < g_hn; i++) g_v[i] /= vn;
        }
    }
    vn = 0.0;
    for (i = 0; i < g_hn; i++) vn += g_v[i] * g_v[i];
    g_last_vnorm = sqrt(vn);
    g_last_proj = proj;
    return proj;
}

/* 只改 Synapse.e[]，不写权重、不改 winner、不绕过核心 R-STDP */
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

int b1f_frozen_vote(int stim, int* a_cnt, int* b_cnt) {
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

/* out[0]=a, [1]=b, [2]=即时 correct（含注入，非能力口径）,
 * [3]=注入 reward 标量, [4]=权重 checksum 变化, [5]=tick 数,
 * [6]=门控量 g, [7]=|m_A-m_B|, [8]=镜像边迹平均绝对差,
 * [9]=优势 A（adv_mode=0 时为 y） */
int b1f_run_trial(int stim, int learning_on, int delay, double out[10]) {
    int t, i, a = 0, b = 0, pred, expected, correct;
    int ticks = 0, arrival, delivery, gate_tick;
    double y, A, reward, z, gate, w0, proj;
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

    /* B1 结果反馈：平局视为失败（-1）是对 host 口径的最小改动 */
    y = (pred == 0) ? -1.0 : (correct ? 1.0 : -1.0);
    g_last_y = y;

    /* adv=0: host 原口径（核心基线饱和）；adv=1: 可学习优势 A=y-b（主实验）；
     * adv=2: 冻结 critic 的参照，A=y 但注入补偿（检验优势稀疏性）。*/
    if (learning_on) g_b += ETA_CRITIC * (y - g_b);
    A = (g_adv_mode == 1) ? (y - g_b) : y;
    g_last_A = A;

    /* 可学习 Oja 读出 -> 内部隐变量 z */
    proj = oja_step();
    z = proj;
    if (g_z_mode == 1) z = 1.0;
    else if (g_z_mode == 2) {
        z = (fwd_splitmix64(&g_model_state) >> 63) ? fabs(proj) : -fabs(proj);
    } else if (g_z_mode == 3) {
        z = g_last_d_raw;   /* 结构先验参照，仅用于瓶颈定位，不计入验收 */
    }
    g_last_z = z;

    /* 可学习 actor 门控量 */
    gate = 0.9 * tanh(g_theta * z);
    g_last_g = gate;
    g_last_baseline = g_net.reward_baseline;
    g_last_gate_diff = 0.0;
    g_last_mirror_diff = 0.0;

    reward = (g_adv_mode == 0) ? y : (A + g_net.reward_baseline);
    if (learning_on) net_inject_reward(&g_net, reward, delay);

    delivery = g_net.current_tick + delay;
    gate_tick = (delay > 0) ? delivery - 1 : -1;
    if (learning_on && g_lambda > 0.0 && delay == 0) {
        apply_pathway_gate(gate);
    }
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

    /* actor 用本试次的优势在线校准方向（不写权重） */
    if (learning_on && g_learn_theta) {
        g_theta += ETA_ACTOR * A * z;
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
