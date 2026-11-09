/*
 * benchmark2_host.cpp - Benchmark-2 宿主测试驱动
 * 两关节虚拟臂到达任务
 *
 * 场景：随机生成 N 个目标点 → 每个目标连续运行 K ticks → 记录末端距离曲线
 * 通过判据：训练后末端距离中位数 ≤ 初始距离 20%
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
#include "../brain_b2/benchmark2_core.h"
}

struct B2Config {
    int input_n    = 100;    /* 位置编码输入神经元 */
    int hidden_n   = 2000;   /* 隐藏层 */
    int motor1_n   = 50;     /* 关节1运动神经元 */
    int motor2_n   = 50;     /* 关节2运动神经元 */
    double avg_conn = 8.0;
    
    int episodes     = 20;   /* 目标点数量 */
    int ticks_per_ep = 1000; /* 每个目标的运行 ticks */
    unsigned int seed = 12345;
    std::string out_csv = "E:/shen_jing_yuan/Experiment/data_b2/results.csv";
};

static unsigned int h_rand;
static inline double hu() {
    h_rand = h_rand * 1103515245 + 12345;
    return (double)((h_rand >> 16) & 0x7FFF) / 32767.0;
}

/* 生成目标点：离初始位置(2,0)在 [0.5, max_reach] 范围内 */
static void gen_target(double* tx, double* ty) {
    double max_reach = ARM_L1 + ARM_L2;
    double min_dist = 0.5;
    /* 距离初始位置在 [min_dist, max_reach] */
    double r = min_dist + hu() * (max_reach - min_dist);
    double theta = hu() * 2.0 * M_PI;
    double sx = max_reach;  /* 初始末端 x */
    double sy = 0.0;
    *tx = sx + r * cos(theta);
    *ty = sy + r * sin(theta);
    /* 确保整体可达 */
    double dist_from_origin = sqrt(*tx * *tx + *ty * *ty);
    if (dist_from_origin > max_reach) {
        *tx = *tx * max_reach / dist_from_origin;
        *ty = *ty * max_reach / dist_from_origin;
    }
    /* 同时确保不会离初始位置太近 */
    double dist_from_start = sqrt((*tx - sx) * (*tx - sx) + (*ty - sy) * (*ty - sy));
    if (dist_from_start < min_dist) {
        /* 重新生成角度 */
        theta = hu() * 2.0 * M_PI;
        *tx = sx + min_dist * cos(theta);
        *ty = sy + min_dist * sin(theta);
    }
}

int main(int argc, char* argv[]) {
    B2Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int& v) { if (i+1<argc) v = atoi(argv[++i]); };
        auto nextd = [&](double& v) { if (i+1<argc) v = atof(argv[++i]); };
        if (a == "--input")     next(cfg.input_n);
        else if (a == "--hidden")  next(cfg.hidden_n);
        else if (a == "--motor")   { next(cfg.motor1_n); cfg.motor2_n = cfg.motor1_n; }
        else if (a == "--episodes") next(cfg.episodes);
        else if (a == "--ticks")   next(cfg.ticks_per_ep);
        else if (a == "--seed")    { if (i+1<argc) cfg.seed = strtoul(argv[++i], nullptr, 10); }
        else if (a == "--out")     { if (i+1<argc) cfg.out_csv = argv[++i]; }
        else if (a == "--disable_trace")  { /* 编译时宏控制 */ }
        else if (a == "--disable_reward") { /* 编译时宏控制 */ }
        else if (a == "--help" || a == "-h") {
            printf("Usage: %s [--input N --hidden N --motor N --episodes N --ticks N --seed N --out FILE]\n", argv[0]);
            return 0;
        }
    }
    
    h_rand = cfg.seed + 777;
    
    printf("=== Benchmark-2: 2-Joint Reaching ===\n");
    printf("Config: in=%d h=%d m1=%d m2=%d episodes=%d ticks/ep=%d seed=%u\n",
           cfg.input_n, cfg.hidden_n, cfg.motor1_n, cfg.motor2_n,
           cfg.episodes, cfg.ticks_per_ep, cfg.seed);
    
    NetworkState2 net;
    if (net2_init(&net, cfg.input_n, cfg.hidden_n, cfg.motor1_n, cfg.motor2_n,
                  cfg.avg_conn, cfg.seed) != 0) {
        fprintf(stderr, "Init failed\n");
        return 1;
    }
    printf("Net: %d neurons, %d synapses\n", net.neuron_count, net.synapse_count);
    
    FILE* fp = fopen(cfg.out_csv.c_str(), "w");
    if (!fp) { net2_destroy(&net); fprintf(stderr, "Cannot open %s\n", cfg.out_csv.c_str()); return 1; }
    fprintf(fp, "episode,target_x,target_y,init_dist,final_dist,dist_ratio,final_reward\n");
    
    /* 存储所有最终距离用于中位数统计 */
    std::vector<double> final_dists;
    std::vector<double> init_dists;
    
    for (int ep = 0; ep < cfg.episodes; ep++) {
        double tx, ty;
        gen_target(&tx, &ty);
        net2_set_target(&net, tx, ty);
        
        /* 重置关节到初始位置 */
        net.arm.theta1 = 0.0;
        net.arm.theta2 = 0.0;
        net.arm.end_x = ARM_L1 + ARM_L2;
        net.arm.end_y = 0.0;
        
        double init_dist = sqrt((net.arm.end_x - tx)*(net.arm.end_x - tx) +
                                (net.arm.end_y - ty)*(net.arm.end_y - ty));
        init_dists.push_back(init_dist);
        
        /* 重置神经元膜电位（除了输入层被目标激活） */
        for (int i = 0; i < net.neuron_count; i++) {
            if (net.neurons[i].type != NT_INPUT_X) {
                net.neurons[i].membrane_potential = MEMBRANE_REST;
                net.neurons[i].is_spiking = 0;
                net.neurons[i].refractory_remaining = 0;
                net.neurons[i].last_spike_tick = -1;
            }
        }
        
        /* 运行 ticks_per_ep ticks */
        for (int t = 0; t < cfg.ticks_per_ep; t++) {
            net2_tick(&net);
        }
        
        double final_dist = sqrt((net.arm.end_x - tx)*(net.arm.end_x - tx) +
                                 (net.arm.end_y - ty)*(net.arm.end_y - ty));
        final_dists.push_back(final_dist);
        double ratio = final_dist / init_dist;
        
        fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                ep, tx, ty, init_dist, final_dist, ratio, net.arm.reward);
        
        printf("  Ep %2d | target=(%.2f,%.2f) | init=%.3f final=%.3f ratio=%.3f reward=%.3f\n",
               ep, tx, ty, init_dist, final_dist, ratio, net.arm.reward);
    }
    
    fclose(fp);
    
    /* 统计 */
    std::vector<double> ratios;
    for (size_t i = 0; i < final_dists.size(); i++) {
        ratios.push_back(final_dists[i] / init_dists[i]);
    }
    std::sort(ratios.begin(), ratios.end());
    double median_ratio = ratios[ratios.size() / 2];
    double mean_ratio = 0;
    for (auto r : ratios) mean_ratio += r;
    mean_ratio /= (double)ratios.size();
    double min_ratio = ratios.front();
    double max_ratio = ratios.back();
    
    printf("\n=== FINAL RESULTS ===\n");
    printf("Ratio (final/initial distance):\n");
    printf("  Median: %.4f\n", median_ratio);
    printf("  Mean:   %.4f\n", mean_ratio);
    printf("  Min:    %.4f\n", min_ratio);
    printf("  Max:    %.4f\n", max_ratio);
    printf("  Target: ≤ 0.20\n");
    printf("\n=== PASS CHECK ===\n");
    if (median_ratio <= 0.20) {
        printf("[PASS] Median ratio %.4f <= 0.20\n");
    } else {
        printf("[FAIL] Median ratio %.4f > 0.20\n", median_ratio);
    }
    
    net2_destroy(&net);
    return (median_ratio <= 0.20) ? 0 : 2;
}
