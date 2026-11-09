/* debug_b2_pot.c - 只测 INPUT_X 膜电位 */
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "../brain_b2/benchmark2_core.h"
#include <stdio.h>

int main() {
    NetworkState2 net;
    net2_init(&net, 36, 10, 5, 5, 4.0, 12345);
    net2_set_target(&net, 0.5, -0.5);
    
    printf("target=(%.2f, %.2f)\n", net.arm.target_x, net.arm.target_y);
    printf("SPIKE_THRESHOLD=%f\n", SPIKE_THRESHOLD);
    printf("sigma_pos=%f\n", ENCODE_SIGMA_POS);
    
    /* 打印所有 INPUT_X 膜电位 */
    printf("\nINPUT_X membrane potentials & prefs:\n");
    for (int i = 0; i < net.input_count; i++) {
        Neuron2* n = &net.neurons[i];
        printf("  [%2d] pref=(%+.2f,%+.2f) mp=%+.1f  ", i, n->pref_x, n->pref_y, n->membrane_potential);
        double dx = n->pref_x - 0.5;
        double dy = n->pref_y - (-0.5);
        double r2 = dx*dx + dy*dy;
        printf("dist2=%.3f response=%.3f\n", r2, exp(-r2 / (2.0 * ENCODE_SIGMA_POS * ENCODE_SIGMA_POS)));
    }
    
    /* 手动跑一个 tick */
    printf("\n--- Running 1 tick ---\n");
    net2_tick(&net);
    printf("After tick 0:\n");
    for (int i = 0; i < net.input_count; i++) {
        Neuron2* n = &net.neurons[i];
        printf("  [%2d] spiking=%d mp=%+.1f\n", i, n->is_spiking, n->membrane_potential);
    }
    
    net2_destroy(&net);
    return 0;
}
