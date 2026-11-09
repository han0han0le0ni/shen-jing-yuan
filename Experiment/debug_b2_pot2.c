/* debug_b2_pot2.c - 诊断 INPUT_X 膜电位 */
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
    net2_set_target(&net, 0.5, -0.5);
    
    printf("=== INPUT_X 膜电位（net2_set_target 之后）===\n");
    double sigma2 = ENCODE_SIGMA_POS * ENCODE_SIGMA_POS;
    int best_i = 0;
    double best_response = 0;
    for (int i = 0; i < net.input_count; i++) {
        Neuron2* n = &net.neurons[i];
        double dx = n->pref_x - 0.5;
        double dy = n->pref_y - (-0.5);
        double dist2 = dx*dx + dy*dy;
        double response = exp(-dist2 / (2.0 * sigma2));
        if (response > best_response) { best_response = response; best_i = i; }
        if (response > 0.5) {
            printf("  Neuron %2d pref=(%.2f,%.2f) dist=%.3f response=%.4f pot=%.2f\n",
                   i, n->pref_x, n->pref_y, sqrt(dist2), response, n->membrane_potential);
        }
    }
    
    printf("\nBest neuron %d: response=%.4f pot=%.2f threshold=%.1f\n",
           best_i, best_response, net.neurons[best_i].membrane_potential, 
           net.neurons[best_i].spike_threshold);
    
    printf("\n=== 手动算 Step 0 的增益 ===\n");
    printf("SPIKE_THRESHOLD=%.1f MEMBRANE_REST=%.1f\n", SPIKE_THRESHOLD, MEMBRANE_REST);
    printf("增益 = %.1f - (%.1f) + 5.0 = %.1f\n", SPIKE_THRESHOLD, MEMBRANE_REST, SPIKE_THRESHOLD - MEMBRANE_REST + 5.0);
    printf("Max response * 增益 = %.4f * %.1f = %.2f\n", best_response, SPIKE_THRESHOLD - MEMBRANE_REST + 5.0);
    printf("Membrane = %.1f + %.2f = %.2f\n", MEMBRANE_REST, best_response * (SPIKE_THRESHOLD - MEMBRANE_REST + 5.0),
           MEMBRANE_REST + best_response * (SPIKE_THRESHOLD - MEMBRANE_REST + 5.0));
    
    /* 手动执行 Step 0 看看 */
    printf("\n=== 手动执行 Step 0 后检查 ===\n");
    double tx = net.arm.target_x;
    double ty = net.arm.target_y;
    printf("target=(%.3f,%.3f)\n", tx, ty);
    
    for (int i = 0; i < net.input_count; i++) {
        Neuron2* n = &net.neurons[i];
        double dx = n->pref_x - tx;
        double dy = n->pref_y - ty;
        double dist2 = dx*dx + dy*dy;
        double response = exp(-dist2 / (2.0 * sigma2));
        n->membrane_potential = MEMBRANE_REST + response * (SPIKE_THRESHOLD - MEMBRANE_REST + 5.0);
    }
    
    int above_thresh = 0;
    for (int i = 0; i < net.input_count; i++) {
        if (net.neurons[i].membrane_potential >= SPIKE_THRESHOLD) above_thresh++;
    }
    printf("INPUT_X >= threshold (-50): %d / %d\n", above_thresh, net.input_count);
    
    /* 现在手动执行 Step 2 的发放判定 */
    for (int i = 0; i < net.neuron_count; i++) {
        Neuron2* n = &net.neurons[i];
        n->is_spiking = 0;
    }
    
    int spiking_now = 0;
    for (int i = 0; i < net.neuron_count; i++) {
        Neuron2* n = &net.neurons[i];
        if (n->membrane_potential >= n->spike_threshold) {
            n->is_spiking = 1;
            spiking_now++;
        }
    }
    printf("After manual Step 2: spiking=%d (input should be >0)\n", spiking_now);
    
    /* 直接跑一个 tick 看看 */
    printf("\n=== 跑 1 个 tick 看看 ===\n");
    net2_tick(&net);
    int in_spike=0, h_spike=0, m1_spike=0, m2_spike=0;
    for (int i = 0; i < net.neuron_count; i++) {
        if (!net.neurons[i].is_spiking) continue;
        switch(net.neurons[i].type) {
            case NT_INPUT_X: in_spike++; break;
            case NT_HIDDEN: h_spike++; break;
            case NT_MOTOR_OMEGA1: m1_spike++; break;
            case NT_MOTOR_OMEGA2: m2_spike++; break;
        }
    }
    printf("After 1 tick: in=%d h=%d m1=%d m2=%d\n", in_spike, h_spike, m1_spike, m2_spike);
    
    /* 再检查一次：tick 后 INPUT_X 膜电位 */
    printf("\nAfter tick INPUT_X best membrane: %.2f (threshold %.1f)\n",
           net.neurons[best_i].membrane_potential, SPIKE_THRESHOLD);
    
    net2_destroy(&net);
    return 0;
}
