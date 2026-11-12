/* ============================================================
 * benchmark1_probe.c - Benchmark-1 只读诊断适配层（T2 离线监督上界）
 *
 * 目的：让 Python 侧能以 flat buffer 读取 B1 网络的「刺激 → hidden 层活动」
 * 表示，从而在完全不注入奖励、不调用任何学习规则的前提下，用离线监督
 * 求解最优线性读出，测量当前架构的表达力上界。
 *
 * 设计约束（对应 spec/judgments.md v1.2 §6.2 约束 ④）：
 *   1. 本文件不修改 benchmark1_core.c / .h 的任何一行；仅调用其公开接口。
 *      核心源码 SHA1 保持不变，audit/b1_root_cause_2026-11-12.md 的证据链不断。
 *   2. 不注入奖励、不调用奖励队列、不改学习规则、不改刺激生成器、不改判据。
 *   3. b1p_step() 精确复刻 host/benchmark1_host.cpp 试次循环的单 tick 协议
 *      （刺激注入 / 清除 + net_tick），b1p_reset_trial() 复刻跨试次清零。
 *   4. 所有读取都是只读快照；b1p_weight_checksum() 用于自证权重零漂移。
 *
 * 编译：与 b1_core.o / seed_bundle.o 链接为 b1_probe.dll（见 build.ps1）。
 * 删除本文件与 b1_probe.dll 即可完全恢复原状，不影响 benchmark1.exe。
 * ============================================================ */

#include "benchmark1_core.h"

static NetworkState g_net;
static int g_ready = 0;

/* ---------- 生命周期 ---------- */

int b1p_create(int sensory_a, int sensory_b, int hidden,
               int output_a, int output_b, double avg_conn,
               unsigned int seed) {
    if (g_ready) {
        net_destroy(&g_net);
        g_ready = 0;
    }
    memset(&g_net, 0, sizeof(g_net));
    if (net_init(&g_net, sensory_a, sensory_b, hidden,
                 output_a, output_b, avg_conn, seed) != 0) {
        return -1;
    }
    g_ready = 1;
    return 0;
}

void b1p_destroy(void) {
    if (g_ready) {
        net_destroy(&g_net);
        g_ready = 0;
    }
}

int b1p_ready(void) { return g_ready; }

/* ---------- M0 四流契约（在 b1p_create 之前调用） ---------- */

void b1p_override_stream(int which, uint64_t value) {
    net_override_seed_stream(which, value);
}

void b1p_legacy_rng(int on) { net_use_legacy_rng(on); }

/* ---------- 形状查询 ---------- */

int b1p_neuron_count(void) { return g_ready ? g_net.neuron_count : 0; }
int b1p_hidden_count(void)  { return g_ready ? g_net.hidden_count : 0; }
int b1p_synapse_count(void) { return g_ready ? g_net.synapse_count : 0; }
int b1p_current_tick(void)  { return g_ready ? g_net.current_tick : 0; }

/* hidden 段在 neurons[] 中的起始下标 */
int b1p_hidden_offset(void) {
    return g_ready ? (g_net.sensory_a_count + g_net.sensory_b_count) : 0;
}

/* ---------- 试次协议（复刻 host，不属于学习规则） ---------- */

void b1p_reset_trial(void) {
    if (!g_ready) return;
    int i;
    for (i = 0; i < g_net.neuron_count; i++) {
        Neuron* n = &g_net.neurons[i];
        n->membrane_potential = MEMBRANE_REST;
        n->is_spiking = 0;
        n->refractory_remaining = 0;
        n->last_spike_tick = -1;
    }
}

/* stim: 0=刺激A, 1=刺激B, 其它=清除刺激（与 host 的 t>=stim_duration 分支一致） */
void b1p_step(int stim) {
    if (!g_ready) return;
    if (stim == 0)      net_inject_stimulus_a(&g_net);
    else if (stim == 1) net_inject_stimulus_b(&g_net);
    else                net_clear_stimulus(&g_net);
    net_tick(&g_net);
}

/* ---------- 只读快照 ---------- */

/* 本 tick 各 hidden 神经元是否发放，写入 buf（长度 >= hidden_count）。
 * 同时回传输出层本 tick 发放数，便于与 host 的投票口径对照。 */
void b1p_hidden_spikes(double* buf, int cap,
                       int* out_a_spikes, int* out_b_spikes,
                       int* hidden_spikes_total) {
    if (!g_ready) {
        if (out_a_spikes) *out_a_spikes = 0;
        if (out_b_spikes) *out_b_spikes = 0;
        if (hidden_spikes_total) *hidden_spikes_total = 0;
        return;
    }
    int i, a = 0, b = 0, h = 0;
    int off = g_net.sensory_a_count + g_net.sensory_b_count;
    for (i = 0; i < g_net.neuron_count; i++) {
        Neuron* n = &g_net.neurons[i];
        if (!n->is_spiking) continue;
        if (n->type == NT_HIDDEN) {
            if (buf && (i - off) < cap) buf[i - off] = 1.0;
            h++;
        } else if (n->type == NT_OUTPUT_A) a++;
        else if (n->type == NT_OUTPUT_B) b++;
    }
    if (buf) {
        int n = g_net.hidden_count < cap ? g_net.hidden_count : cap;
        for (i = 0; i < n; i++) {
            if (g_net.neurons[off + i].is_spiking == 0) buf[i] = 0.0;
        }
    }
    if (out_a_spikes) *out_a_spikes = a;
    if (out_b_spikes) *out_b_spikes = b;
    if (hidden_spikes_total) *hidden_spikes_total = h;
}

/* 输出层 initial/final 偏向（转调核心只读接口） */
void b1p_output_bias(double out[4]) {
    if (out) { out[0] = out[1] = out[2] = out[3] = 0.0; }
    if (g_ready) net_output_bias(&g_net, out);
}

/* hidden→output 权重按「hidden 左/右半 × 输出 A/B」四类求和后的扁平数组，
 * 与核心 net_output_bias 同口径，供 Python 侧独立复算。 */
void b1p_readout_weights(double out[4]) {
    if (out) { out[0] = out[1] = out[2] = out[3] = 0.0; }
    if (!g_ready) return;
    int hbase = g_net.sensory_a_count + g_net.sensory_b_count;
    int hmid  = g_net.hidden_count / 2;
    int obase = hbase + g_net.hidden_count;
    int oa_end = obase + g_net.output_a_count;
    int s;
    for (s = 0; s < g_net.synapse_count; s++) {
        int pre = g_net.synapses[s].pre_id;
        int post = g_net.synapses[s].post_id;
        if (pre < hbase || pre >= obase || post < obase) continue;
        int left = (pre < hbase + hmid);
        int to_a = (post < oa_end);
        int q = (left ? 0 : 2) + (to_a ? 0 : 1);
        out[q] += g_net.synapses[s].weight;
    }
}

/* 全网络权重校验和：用于证明诊断期间权重零漂移（没有学习发生） */
double b1p_weight_checksum(void) {
    if (!g_ready) return 0.0;
    double s = 0.0;
    int i;
    for (i = 0; i < g_net.synapse_count; i++) {
        s += g_net.synapses[i].weight;
    }
    return s;
}
