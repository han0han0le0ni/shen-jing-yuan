/* debug_b2_learn.c - 诊断 B2 学习失败原因
 * 迷你网络 36→200→10→10，50 episode
 * 打印：各层 spike 率 / 权重更新量 / trace 值分布 / omega_cmd 序列 / baseline 变化
 */
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "brain_b2/benchmark2_core.h"
#include <stdio.h>
#include <math.h>

void print_stats(NetworkState2* net, const char* label) {
    int sa=0, sh=0, m1s=0, m2s=0;
    for (int i = 0; i < net->neuron_count; i++) {
        if (!net->neurons[i].is_spiking) continue;
        switch(net->neurons[i].type) {
            case NT_INPUT_X: sa++; break;
            case NT_HIDDEN:  sh++; break;
            case NT_MOTOR_OMEGA1: m1s++; break;
            case NT_MOTOR_OMEGA2: m2s++; break;
        }
    }
    
    double w_sum=0, w_min=1e9, w_max=-1e9;
    double tr_sum=0, tr_abs_sum=0, tr_min=1e9, tr_max=-1e9;
    int w_saturated=0, tr_nonzero=0;
    for (int s = 0; s < net->synapse_count; s++) {
        double w = net->synapses[s].weight;
        double tr = net->synapses[s].eligibility_trace;
        w_sum += w;
        if (w < w_min) w_min = w;
        if (w > w_max) w_max = w;
        if (w >= WEIGHT_MAX * 0.99 || w <= WEIGHT_MIN * 1.01) w_saturated++;
        tr_sum += tr;
        tr_abs_sum += fabs(tr);
        if (tr < tr_min) tr_min = tr;
        if (tr > tr_max) tr_max = tr;
        if (fabs(tr) > 1e-6) tr_nonzero++;
    }
    
    printf("[%s] tick=%d | spike: in=%d h=%d m1=%d m2=%d | arm=(%.3f,%.3f) reward=%.3f | w: mean=%.3f [%.3f-%.3f] sat=%d/%d | trace: mean=%.4f abs_mean=%.4f [%.3f,%.3f] nonzero=%d/%d | baseline=%.4f\n",
           label, net->current_tick, sa, sh, m1s, m2s,
           net->arm.end_x, net->arm.end_y, net->arm.reward,
           w_sum/net->synapse_count, w_min, w_max, w_saturated, net->synapse_count,
           tr_sum/net->synapse_count, tr_abs_sum/net->synapse_count, tr_min, tr_max, tr_nonzero, net->synapse_count,
           net->reward_baseline);
}

int main() {
    NetworkState2 net;
    net2_init(&net, 36, 200, 10, 10, 8.0, 12345);
    net2_set_target(&net, 1.5, 0.5);
    printf("Net: %d neurons, %d synapses, target=(1.5, 0.5)\n\n", net.neuron_count, net.synapse_count);
    
    /* 保存初始权重用于对比 */
    double* init_w = (double*)malloc(net.synapse_count * sizeof(double));
    for (int s = 0; s < net.synapse_count; s++) init_w[s] = net.synapses[s].weight;
    
    /* 诊断 100 tick 内的 omega 序列 */
    printf("=== Omega cmd 序列 (tick 1-100) ===\n");
    double w1_mean=0, w2_mean=0;
    int m1_base = net.input_count + net.hidden_count;
    int m2_base = m1_base + net.motor1_count;
    
    for (int t = 0; t < 100; t++) {
        double w1=0, w2=0;
        /* 先跑 tick（里面 Step 4 会解码）*/
        net2_tick(&net);
        
        /* 手动解码检查 */
        for (int j = 0; j < 2; j++) {
            int m_base = (j == 0) ? m1_base : m2_base;
            int m_count = (j == 0) ? net.motor1_count : net.motor2_count;
            int pos_count=0, neg_count=0;
            double pos_sum=0, neg_sum=0;
            for (int i = 0; i < m_count; i++) {
                Neuron2* n = &net.neurons[m_base + i];
                if (n->is_spiking) {
                    if (n->pref_omega > 0.0) { pos_count++; pos_sum += n->pref_omega; }
                    else if (n->pref_omega < 0.0) { neg_count++; neg_sum += n->pref_omega; }
                }
            }
            double cmd;
            if (pos_count > neg_count && pos_count > 0) cmd = pos_sum/pos_count;
            else if (neg_count > pos_count && neg_count > 0) cmd = neg_sum/neg_count;
            else cmd = 0.0;
            cmd = fmax(-0.5, fmin(0.5, cmd));
            if (j == 0) w1 = cmd; else w2 = cmd;
        }
        
        w1_mean += w1; w2_mean += w2;
        /* 数一下 motor 层 spike */
        int m1_cnt=0, m2_cnt=0;
        for (int i=0;i<net.motor1_count;i++) if(net.neurons[m1_base+i].is_spiking) m1_cnt++;
        for (int i=0;i<net.motor2_count;i++) if(net.neurons[m2_base+i].is_spiking) m2_cnt++;
        if (t < 20 || (t+1) % 10 == 0) {
            printf("  Tick %3d | omega1=%+.3f omega2=%+.3f | arm=(%.3f,%.3f) | m1_spike=%d m2_spike=%d\n",
                   t, w1, w2, net.arm.end_x, net.arm.end_y, m1_cnt, m2_cnt);
        }
    }
    printf("  omega1 mean=%.4f, omega2 mean=%.4f (100 tick 平均)\n\n", w1_mean/100, w2_mean/100);
    
    print_stats(&net, "After 100 tick (pre-learning)");
    
    /* 跑 50 episode */
    printf("\n=== 50 Episode Learning ===\n");
    net.reward_baseline = 0.0;
    for (int ep = 0; ep < 50; ep++) {
        /* 重置关节 */
        net.arm.theta1 = 0.0; net.arm.theta2 = 0.0;
        net.arm.end_x = 2.0; net.arm.end_y = 0.0;
        net2_set_target(&net, 1.5, 0.5);
        
        double init_dist = sqrt((2.0-1.5)*(2.0-1.5) + (0.0-0.5)*(0.0-0.5));
        double prev_ratio = 999.0;
        
        for (int t = 0; t < 200; t++) {  /* 迷你版 200 tick/ep */
            net2_tick(&net);
            /* 密集 reward */
            double cur_dist = sqrt((net.arm.end_x-1.5)*(net.arm.end_x-1.5) + (net.arm.end_y-0.5)*(net.arm.end_y-0.5));
            double r = (init_dist - cur_dist) / init_dist;
            if (r < -1.0) r = -1.0; if (r > 1.0) r = 1.0;
            net2_inject_reward(&net, r, 0);
        }
        /* final reward + 5 tick */
        double fd = sqrt((net.arm.end_x-1.5)*(net.arm.end_x-1.5) + (net.arm.end_y-0.5)*(net.arm.end_y-0.5));
        double rw = (init_dist - fd) / init_dist;
        if (rw < -1.0) rw = -1.0; if (rw > 1.0) rw = 1.0;
        net2_inject_reward(&net, rw, 0);
        for (int t = 0; t < 5; t++) net2_tick(&net);
        
        double ratio = fd / init_dist;
        if (ep < 5 || (ep+1) % 10 == 0) {
            printf("Ep %2d | ratio=%.4f reward=%.4f baseline=%.4f | arm=(%.3f,%.3f)\n",
                   ep, ratio, rw, net.reward_baseline, net.arm.end_x, net.arm.end_y);
        }
        prev_ratio = ratio;
    }
    
    print_stats(&net, "After 50 episode");
    
    /* 对比前后权重变化 */
    double total_dw=0, max_dw=0, changed=0;
    for (int s = 0; s < net.synapse_count; s++) {
        double dw = fabs(net.synapses[s].weight - init_w[s]);
        total_dw += dw;
        if (dw > max_dw) max_dw = dw;
        if (dw > 0.001) changed++;
    }
    printf("\n=== 权重更新统计 ===\n");
    printf("总突触: %d, 变化突触: %d, 平均变化: %.6f, 最大变化: %.4f\n",
           net.synapse_count, changed, total_dw/net.synapse_count, max_dw);
    
    /* 检查特定突触：HIDDEN→MOTOR1 连接中，正 trace vs 负 trace */
    int h2m1_base = net.input_count + net.hidden_count;
    int h2m1_pos_trace=0, h2m1_neg_trace=0, h2m1_zero_trace=0;
    int h2m1_weight_up=0, h2m1_weight_down=0;
    double h2m1_dw = 0;
    for (int s = 0; s < net.synapse_count; s++) {
        Synapse2* syn = &net.synapses[s];
        if (syn->pre_id >= net.input_count && syn->pre_id < h2m1_base &&
            syn->post_id >= h2m1_base && syn->post_id < h2m1_base + net.motor1_count) {
            if (syn->eligibility_trace > 0.01) h2m1_pos_trace++;
            else if (syn->eligibility_trace < -0.01) h2m1_neg_trace++;
            else h2m1_zero_trace++;
            
            double dw = syn->weight - init_w[s];
            h2m1_dw += dw;
            if (dw > 0.001) h2m1_weight_up++;
            else if (dw < -0.001) h2m1_weight_down++;
        }
    }
    printf("\n=== HIDDEN→MOTOR1 突触分析 ===\n");
    printf("突触数: %d\n", h2m1_pos_trace+h2m1_neg_trace+h2m1_zero_trace);
    printf("trace 分布: pos=%d neg=%d zero=%d\n", h2m1_pos_trace, h2m1_neg_trace, h2m1_zero_trace);
    printf("权重变化: up=%d down=%d mean_dw=%.4f\n", h2m1_weight_up, h2m1_weight_down, h2m1_dw/(h2m1_pos_trace+h2m1_neg_trace+h2m1_zero_trace));
    
    free(init_w);
    net2_destroy(&net);
    return 0;
}
