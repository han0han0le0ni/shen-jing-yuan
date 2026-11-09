/* debug_b2_pot3.c - 诊断 pref 点分布和 response 计算 */
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "../brain_b2/benchmark2_core.h"
#include <stdio.h>

int main() {
    NetworkState2 net;
    net2_init(&net, 36, 200, 10, 10, 8.0, 12345);
    
    double tx = 0.5, ty = -0.5;
    double sigma2 = ENCODE_SIGMA_POS * ENCODE_SIGMA_POS;
    printf("target=(%.3f,%.3f) sigma=%.2f sigma2=%.4f\n\n", tx, ty, ENCODE_SIGMA_POS, sigma2);
    
    double min_dist = 1e9;
    int best_i = 0;
    printf("All INPUT_X pref points and responses:\n");
    for (int i = 0; i < net.input_count; i++) {
        Neuron2* n = &net.neurons[i];
        double dx = n->pref_x - tx;
        double dy = n->pref_y - ty;
        double dist = sqrt(dx*dx + dy*dy);
        double response = exp(-dist*dist / (2.0 * sigma2));
        if (dist < min_dist) { min_dist = dist; best_i = i; }
        printf("  [%2d] pref=(%7.3f,%7.3f) dist=%.4f response=%.4f\n", 
               i, n->pref_x, n->pref_y, dist, response);
    }
    
    printf("\nBest: [%d] pref=(%.3f,%.3f) min_dist=%.4f\n", 
           best_i, net.neurons[best_i].pref_x, net.neurons[best_i].pref_y, min_dist);
    printf("Expected dist if grid-aligned: ~0.1414\n");
    
    /* 直接算膜电位 */
    double best_response = exp(-min_dist*min_dist / (2.0 * sigma2));
    double gain = SPIKE_THRESHOLD - MEMBRANE_REST + 5.0;
    double pot = MEMBRANE_REST + best_response * gain;
    printf("\nBest response=%.4f gain=%.1f pot=%.2f threshold=%.1f\n",
           best_response, gain, pot, SPIKE_THRESHOLD);
    
    /* 如果要让 max response 刚好到 threshold，需要什么增益？ */
    double needed_gain = (SPIKE_THRESHOLD - MEMBRANE_REST) / best_response;
    printf("Needed gain to reach threshold: %.1f (currently %.1f)\n", needed_gain, gain);
    
    /* 如果增大 sigma 呢？ */
    for (double s = 0.35; s <= 1.5; s += 0.25) {
        double r = exp(-min_dist*min_dist / (2.0 * s * s));
        double p = MEMBRANE_REST + r * gain;
        printf("  sigma=%.2f → response=%.4f pot=%.2f %s\n", 
               s, r, p, (p >= SPIKE_THRESHOLD) ? "✓" : "✗");
    }
    
    /* 如果降低阈值呢？ */
    for (double thresh = -50; thresh >= -60; thresh -= 2) {
        double p = MEMBRANE_REST + best_response * (thresh - MEMBRANE_REST + 5.0);
        printf("  threshold=%.1f → pot=%.2f %s\n", 
               thresh, p, (p >= thresh) ? "✓" : "✗");
    }
    
    net2_destroy(&net);
    return 0;
}
