/* debug_b2_learn.c - 固定目标验证网络能否学习 */
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "../brain_b2/benchmark2_core.h"
#include <stdio.h>
#include <math.h>

int main() {
    NetworkState2 net;
    net2_init(&net, 36, 200, 10, 10, 8.0, 12345);
    
    /* 固定目标在正下方 (0, -2) */
    double tx = 0.0, ty = -1.5;
    
    printf("=== 固定目标学习 ===\n");
    printf("初始 arm: (%.3f,%.3f)\n", net.arm.end_x, net.arm.end_y);
    printf("目标: (%.2f, %.2f)\n", tx, ty);
    printf("初始 dist: %.3f, reward=%.3f\n\n", 
           sqrt(net.arm.end_x*net.arm.end_x + net.arm.end_y*net.arm.end_y),
           net.arm.reward);
    
    int prev_best = 9999;
    for (int ep = 0; ep < 30; ep++) {
        /* 设置目标 + 重置 arm + 重置非输入层神经元 */
        net2_set_target(&net, tx, ty);
        net.arm.theta1 = 0.0;
        net.arm.theta2 = 0.0;
        net.arm.end_x = ARM_L1 + ARM_L2;
        net.arm.end_y = 0.0;
        for (int i = 0; i < net.neuron_count; i++) {
            if (net.neurons[i].type != NT_INPUT_X) {
                net.neurons[i].membrane_potential = MEMBRANE_REST;
                net.neurons[i].is_spiking = 0;
                net.neurons[i].refractory_remaining = 0;
                net.neurons[i].last_spike_tick = -1;
            }
        }
        
        double init_dist = sqrt(net.arm.end_x*net.arm.end_x + net.arm.end_y*net.arm.end_y);
        double best_dist_ep = 9999;
        
        for (int t = 0; t < 500; t++) {
            net2_tick(&net);
            double d = sqrt(pow(net.arm.end_x - tx, 2) + pow(net.arm.end_y - ty, 2));
            if (d < best_dist_ep) best_dist_ep = d;
        }
        
        double final_dist = sqrt(pow(net.arm.end_x - tx, 2) + pow(net.arm.end_y - ty, 2));
        printf("Ep %2d | best=%.3f final=%.3f ratio=%.3f improved=%d\n",
               ep, best_dist_ep, final_dist, final_dist/init_dist,
               (best_dist_ep < prev_best));
        prev_best = best_dist_ep;
    }
    
    net2_destroy(&net);
    return 0;
}
