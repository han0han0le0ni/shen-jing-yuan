/* debug_b2_omega2.c - 诊断固定目标下的 omega */
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "../brain_b2/benchmark2_core.h"
#include <stdio.h>

int main() {
    NetworkState2 net;
    net2_init(&net, 36, 200, 10, 10, 8.0, 12345);
    net2_set_target(&net, 0.0, -1.5);
    net.arm.theta1 = 0.0;
    net.arm.theta2 = 0.0;
    net.arm.end_x = 2.0;
    net.arm.end_y = 0.0;
    printf("Target=(0,-1.5) init arm=(2,0)\n\n");
    
    /* 手动检查解码 */
    int m1_base = net.input_count + net.hidden_count;
    int m2_base = m1_base + net.motor1_count;
    
    for (int t = 0; t < 10; t++) {
        net2_tick(&net);
        
        /* 手动解码看看 tick 后的 omega 是什么 */
        double omega1_cmd = 0, omega2_cmd = 0;
        double rs1 = 0, rs2 = 0;
        for (int i = 0; i < net.motor1_count; i++) {
            Neuron2* n = &net.neurons[m1_base+i];
            int since = t+1 - n->last_spike_tick;
            double r = (n->last_spike_tick >= 0 && since <= 5) ? 1.0 : 0.0;
            omega1_cmd += n->pref_omega * r; rs1 += r;
        }
        for (int i = 0; i < net.motor2_count; i++) {
            Neuron2* n = &net.neurons[m2_base+i];
            int since = t+1 - n->last_spike_tick;
            double r = (n->last_spike_tick >= 0 && since <= 5) ? 1.0 : 0.0;
            omega2_cmd += n->pref_omega * r; rs2 += r;
        }
        if (rs1 > 0) omega1_cmd /= rs1;
        if (rs2 > 0) omega2_cmd /= rs2;
        
        int ms1=0, ms2=0;
        for (int i = 0; i < net.motor1_count; i++) if (net.neurons[m1_base+i].is_spiking) ms1++;
        for (int i = 0; i < net.motor2_count; i++) if (net.neurons[m2_base+i].is_spiking) ms2++;
        
        printf("T%2d: omega=(%+.3f,%+.3f) ratesum=(%.0f,%.0f) spiking_m=(%d,%d) arm=(%.3f,%.3f)\n",
               t, omega1_cmd, omega2_cmd, rs1, rs2, ms1, ms2,
               net.arm.end_x, net.arm.end_y);
    }
    
    net2_destroy(&net);
    return 0;
}
