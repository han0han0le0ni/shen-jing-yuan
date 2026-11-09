/* debug_b2_pot4.c - 逐步追踪 tick 循环 */
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "../brain_b2/benchmark2_core.h"
#include <stdio.h>

int main() {
    NetworkState2 net;
    net2_init(&net, 36, 200, 10, 10, 8.0, 12345);
    net2_set_target(&net, 0.5, -0.5);
    
    /* 检查 net2_set_target 后 INPUT_X 膜电位 */
    int best_i = 15;  /* 已知最好的 */
    printf("After net2_set_target:\n");
    printf("  Input[%d] pot=%.2f threshold=%.1f >= ? %d\n",
           best_i, net.neurons[best_i].membrane_potential,
           net.neurons[best_i].spike_threshold,
           net.neurons[best_i].membrane_potential >= net.neurons[best_i].spike_threshold);
    
    /* 手动模拟 Step 0（因为 net2_tick 会重置） */
    net.current_tick = 1;
    double tx = net.arm.target_x;
    double ty = net.arm.target_y;
    double sigma2 = ENCODE_SIGMA_POS * ENCODE_SIGMA_POS;
    printf("\nManual Step 0 (tick=%d):\n", net.current_tick);
    int above = 0;
    for (int i = 0; i < net.input_count; i++) {
        Neuron2* n = &net.neurons[i];
        double dx = n->pref_x - tx;
        double dy = n->pref_y - ty;
        double dist2 = dx*dx + dy*dy;
        double response = exp(-dist2 / (2.0 * sigma2));
        n->membrane_potential = MEMBRANE_REST + response * (SPIKE_THRESHOLD - MEMBRANE_REST + 5.0);
        if (n->membrane_potential >= n->spike_threshold) above++;
    }
    printf("  INPUT_X >= threshold: %d / %d\n", above, net.input_count);
    printf("  Best[%d] pot=%.2f\n", best_i, net.neurons[best_i].membrane_potential);
    
    /* Step 2 手动：清零 + 发放判定 */
    printf("\nManual Step 2:\n");
    for (int i = 0; i < net.neuron_count; i++) net.neurons[i].is_spiking = 0;
    
    for (int i = 0; i < net.neuron_count; i++) {
        Neuron2* n = &net.neurons[i];
        if (n->type != NT_INPUT_X) {
            if (n->refractory_remaining > 0) { n->refractory_remaining--; n->membrane_potential = MEMBRANE_REST; continue; }
            n->membrane_potential = n->membrane_potential * MEMBRANE_DECAY + MEMBRANE_REST * (1.0 - MEMBRANE_DECAY);
        }
        if (n->membrane_potential >= n->spike_threshold) {
            n->is_spiking = 1;
            n->last_spike_tick = net.current_tick;
            if (n->type != NT_INPUT_X) n->refractory_remaining = REFRACTORY_TICKS;
            n->membrane_potential = MEMBRANE_REST;
        }
    }
    
    int in_spike=0, h_spike=0;
    for (int i = 0; i < net.neuron_count; i++) {
        if (net.neurons[i].is_spiking) {
            if (net.neurons[i].type == NT_INPUT_X) in_spike++;
            if (net.neurons[i].type == NT_HIDDEN) h_spike++;
        }
    }
    printf("  After Step 2: in=%d h=%d\n", in_spike, h_spike);
    
    /* 现在跑 net2_tick 看看为什么不一样 */
    printf("\n--- Now run net2_tick() ---\n");
    net2_destroy(&net);
    net2_init(&net, 36, 200, 10, 10, 8.0, 12345);
    net2_set_target(&net, 0.5, -0.5);
    
    printf("Before tick: Input[%d] pot=%.2f\n", best_i, net.neurons[best_i].membrane_potential);
    net2_tick(&net);
    printf("After tick #%d: Input[%d] pot=%.2f spiking=%d\n",
           net.current_tick, best_i, net.neurons[best_i].membrane_potential, net.neurons[best_i].is_spiking);
    
    /* 再跑一个 tick 看看 */
    net2_tick(&net);
    printf("After tick #%d: Input[%d] pot=%.2f spiking=%d\n",
           net.current_tick, best_i, net.neurons[best_i].membrane_potential, net.neurons[best_i].is_spiking);
    
    net2_destroy(&net);
    return 0;
}
