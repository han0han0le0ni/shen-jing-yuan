/* debug_b2_spike.c - 追踪迷你B2的传播链 */
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
    printf("Arm init: (%.3f,%.3f) → target (%.3f,%.3f)\n\n",
           net.arm.end_x, net.arm.end_y, net.arm.target_x, net.arm.target_y);
    
    for (int t = 0; t < 30; t++) {
        net2_tick(&net);  /* ← 关键！先跑 tick！ */
        
        int sa=0, sh=0, m1s=0, m2s=0;
        for (int i = 0; i < net.neuron_count; i++) {
            if (!net.neurons[i].is_spiking) continue;
            switch(net.neurons[i].type) {
                case NT_INPUT_X: sa++; break;
                case NT_HIDDEN:  sh++; break;
                case NT_MOTOR_OMEGA1: m1s++; break;
                case NT_MOTOR_OMEGA2: m2s++; break;
            }
        }
        printf("Tick %3d | in=%2d h=%2d m1=%2d m2=%2d | arm=(%.3f,%.3f) reward=%.3f\n",
               t, sa, sh, m1s, m2s, net.arm.end_x, net.arm.end_y, net.arm.reward);
    }
    
    net2_destroy(&net);
    return 0;
}
