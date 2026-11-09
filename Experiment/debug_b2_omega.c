/* debug_b2_omega.c - 直接打印 omega 解码结果 */
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
    printf("Arm init: (%.3f,%.3f) → target (%.3f,%.3f)\n",
           net.arm.end_x, net.arm.end_y, net.arm.target_x, net.arm.target_y);
    printf("Motor pref_omega: ");
    int m1_base = net.input_count + net.hidden_count;
    int m2_base = m1_base + net.motor1_count;
    for (int i = 0; i < net.motor1_count; i++) printf("M1[%d]=%.2f ", i, net.neurons[m1_base+i].pref_omega);
    printf("\n");
    for (int i = 0; i < net.motor2_count; i++) printf("M2[%d]=%.2f ", i, net.neurons[m2_base+i].pref_omega);
    printf("\n\n");
    
    for (int t = 0; t < 20; t++) {
        /* 先手动做解码，看看 omega 值 */
        double omega1_cmd = 0.0, omega2_cmd = 0.0;
        double rate_sum1 = 0.0, rate_sum2 = 0.0;
        for (int i = 0; i < net.motor1_count; i++) {
            Neuron2* n = &net.neurons[m1_base + i];
            int since = t - n->last_spike_tick;
            double rate = (since >= 0 && since <= 5) ? 1.0 : 0.0;
            omega1_cmd += n->pref_omega * rate;
            rate_sum1 += rate;
        }
        for (int i = 0; i < net.motor2_count; i++) {
            Neuron2* n = &net.neurons[m2_base + i];
            int since = t - n->last_spike_tick;
            double rate = (since >= 0 && since <= 5) ? 1.0 : 0.0;
            omega2_cmd += n->pref_omega * rate;
            rate_sum2 += rate;
        }
        if (rate_sum1 > 0) omega1_cmd /= rate_sum1;
        if (rate_sum2 > 0) omega2_cmd /= rate_sum2;
        
        net2_tick(&net);
        
        printf("T%2d: omega_cmd=(%.3f,%.3f) rate_sum=(%.0f,%.0f) m1_spike=%d m2_spike=%d arm=(%.3f,%.3f)\n",
               t, omega1_cmd, omega2_cmd, rate_sum1, rate_sum2,
               net.neurons[m1_base].is_spiking, net.neurons[m2_base].is_spiking,
               net.arm.end_x, net.arm.end_y);
    }
    
    net2_destroy(&net);
    return 0;
}
