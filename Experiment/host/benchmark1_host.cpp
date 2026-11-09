/*
 * benchmark1_host.cpp - Benchmark-1 宿主测试驱动
 * 负责：生成刺激 → 运行tick → 注入延迟奖励 → 统计正确率 → CSV输出
 *
 * 场景：感觉层t=0受刺激A或B → 延迟D=200 ticks → 奖励/惩罚 → 下一试次
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <cmath>

/* C ABI 绑定 */
extern "C" {
#include "../brain/benchmark1_core.h"
}

/* ============================================================
 * 配置（冻结默认值，可通过命令行参数覆盖）
 * ============================================================ */
struct BenchmarkConfig {
    int sensory_a      = 50;     /* 感觉A神经元数 */
    int sensory_b      = 50;     /* 感觉B神经元数 */
    int hidden         = 400;    /* 中间神经元数 */
    int output_a       = 25;     /* 输出A神经元数 */
    int output_b       = 25;     /* 输出B神经元数 */
    double avg_conn    = 8.0;    /* 平均每神经元连接数 */
    
    int delay_ticks    = 200;    /* 奖励延迟D */
    int trials         = 1000;   /* 总试次数 */
    int stim_duration  = 10;     /* 刺激持续tick数 */
    int settle_ticks   = 50;     /* 刺激后稳定时间（给网络传播） */
    
    double correct_reward = 1.0; /* 正确奖励值 */
    double wrong_punish   = -1.0; /* 错误惩罚值 */
    
    unsigned int seed  = 12345;  /* 随机种子 */
};

/* ============================================================
 * 命令行解析
 * ============================================================ */
void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --sensory_a N      Sensory A neurons (default: 50)\n");
    printf("  --sensory_b N      Sensory B neurons (default: 50)\n");
    printf("  --hidden N         Hidden neurons (default: 400)\n");
    printf("  --output_a N       Output A neurons (default: 25)\n");
    printf("  --output_b N       Output B neurons (default: 25)\n");
    printf("  --delay N          Reward delay ticks D (default: 200)\n");
    printf("  --trials N         Total trials (default: 1000)\n");
    printf("  --seed N           Random seed (default: 12345)\n");
    printf("  --out FILE         Output CSV file path\n");
    printf("  --disable_trace     Disable eligibility trace (ablation)\n");
    printf("  --disable_reward    Disable reward modulation (ablation)\n");
}

int main(int argc, char* argv[]) {
    BenchmarkConfig cfg;
    std::string out_file = "E:/shen_jing_yuan/Experiment/data/benchmark1_results.csv";
    bool force_no_trace = false;
    bool force_no_reward = false;
    
    /* 解析命令行 */
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        auto next = [&](int& val) { if (i+1 < argc) val = atoi(argv[++i]); };
        auto nextu = [&](unsigned int& val) { if (i+1 < argc) val = (unsigned int)strtoul(argv[++i], nullptr, 10); };
        auto nextd = [&](double& val) { if (i+1 < argc) val = atof(argv[++i]); };
        
        if (arg == "--sensory_a") next(cfg.sensory_a);
        else if (arg == "--sensory_b") next(cfg.sensory_b);
        else if (arg == "--hidden") next(cfg.hidden);
        else if (arg == "--output_a") next(cfg.output_a);
        else if (arg == "--output_b") next(cfg.output_b);
        else if (arg == "--delay") next(cfg.delay_ticks);
        else if (arg == "--trials") next(cfg.trials);
        else if (arg == "--seed") nextu(cfg.seed);
        else if (arg == "--out") { if (i+1 < argc) out_file = argv[++i]; }
        else if (arg == "--disable_trace") force_no_trace = true;
        else if (arg == "--disable_reward") force_no_reward = true;
    }
    
    printf("=== Benchmark-1: Single-Synapse Delayed Credit Assignment ===\n");
    printf("Config: S_A=%d S_B=%d H=%d O_A=%d O_B=%d Delay=%d Trials=%d Seed=%u\n",
           cfg.sensory_a, cfg.sensory_b, cfg.hidden, cfg.output_a, cfg.output_b,
           cfg.delay_ticks, cfg.trials, cfg.seed);
    printf("El Trace: %s, Reward Mod: %s\n",
           force_no_trace ? "OFF (ABLATION)" : (ELIGIBILITY_TRACE_ENABLE ? "ON" : "OFF (COMPILE-TIME DISABLED)"),
           force_no_reward ? "OFF (ABLATION)" : (REWARD_MODULATION_ENABLE ? "ON" : "OFF (COMPILE-TIME DISABLED)"));
    
    /* 初始化网络 */
    NetworkState net;
    if (net_init(&net, cfg.sensory_a, cfg.sensory_b, cfg.hidden, cfg.output_a, cfg.output_b,
                 cfg.avg_conn, cfg.seed) != 0) {
        fprintf(stderr, "Failed to initialize network\n");
        return 1;
    }
    
    /* 运行时强制消融（可选） */
    if (force_no_trace) net.use_eligibility_trace = 0;
    if (force_no_reward) net.use_reward_modulation = 0;
    
    printf("Network initialized: %d neurons, %d synapses\n", net.neuron_count, net.synapse_count);
    
    /* 打开CSV输出 */
    FILE* fp = fopen(out_file.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s\n", out_file.c_str());
        net_destroy(&net);
        return 1;
    }
    
    /* 写CSV头 */
    fprintf(fp, "trial,stimulus_type,predicted_vote,correct,output_a_rate,output_b_rate,"
                "reward_given,reward_baseline,elapsed_ticks\n");
    
    /* 试次统计 */
    int correct_count = 0;
    int train_correct = 0, test_correct = 0;
    int train_total = cfg.trials / 2, test_total = cfg.trials - cfg.trials / 2;
    
    /* 试次循环 */
    unsigned int trial_seed = cfg.seed + 999;
    for (int trial = 0; trial < cfg.trials; trial++) {
        /* 生成刺激类型：0=A, 1=B */
        trial_seed = trial_seed * 1103515245 + 12345;
        unsigned int stim_rand = (trial_seed >> 16) & 0x7FFF;
        int stimulus_type = (stim_rand % 2 == 0) ? 0 : 1; /* 0=A, 1=B */
        
        int ticks_in_trial = 0;
        
        /* 重置所有神经元到静息状态（跨试次清零） */
        for (int i = 0; i < net.neuron_count; i++) {
            net.neurons[i].membrane_potential = -65.0;
            net.neurons[i].is_spiking = 0;
            net.neurons[i].refractory_remaining = 0;
            net.neurons[i].last_spike_tick = -1;
        }
        
        /* 输出观察窗口：累积观察期内OA和OB的发放次数 */
        const int OBSERVE_WINDOW = 15; /* 观察15 ticks */
        
        /* Phase 1: 刺激注入 + 同时观察输出 */
        int peak_a_count = 0, peak_b_count = 0;
        const int PEAK_TICK = 2;  /* 信号传播最强时刻 */
        
        for (int t = 0; t < OBSERVE_WINDOW; t++) {
            if (t < cfg.stim_duration) {
                /* 前N ticks注入刺激 */
                if (stimulus_type == 0) net_inject_stimulus_a(&net);
                else net_inject_stimulus_b(&net);
            } else {
                /* 之后清除刺激 */
                net_clear_stimulus(&net);
            }
            
            net_tick(&net);
            ticks_in_trial++;
            
            /* 在峰值tick记录发放计数 */
            if (t == PEAK_TICK) {
                for (int i = 0; i < net.neuron_count; i++) {
                    if (net.neurons[i].is_spiking) {
                        if (net.neurons[i].type == NT_OUTPUT_A) peak_a_count++;
                        if (net.neurons[i].type == NT_OUTPUT_B) peak_b_count++;
                    }
                }
            }
        }
        
        /* Phase 2: 用峰值时刻的快照投票 */
        double out_a_rate = (double)peak_a_count / net.output_a_count;
        double out_b_rate = (double)peak_b_count / net.output_b_count;
        
        /* Phase 3: 决策投票 */
        int predicted;
        if (peak_a_count > peak_b_count && peak_a_count > 0) predicted = 1;  /* 偏好A */
        else if (peak_b_count > peak_a_count && peak_b_count > 0) predicted = 2;  /* 偏好B */
        else predicted = 0;  /* 平局/无输出 */
        
        int expected = stimulus_type == 0 ? 1 : 2; /* A→1, B→2 */
        int is_correct = (predicted == expected) ? 1 : 0;
        
        /* 奖励策略：正确=+1.0, 错误=-1.0, 平局=0.0（打破死锁） */
        double reward_val;
        if (predicted == 0) reward_val = 0.0;      /* 平局不给惩罚 */
        else reward_val = is_correct ? 1.0 : -1.0;
        
        /* 统计 */
        if (trial < train_total) {
            if (is_correct) train_correct++;
        } else {
            if (is_correct) test_correct++;
        }
        if (is_correct) correct_count++;
        
        /* Phase 4: 注入延迟奖励 */
        net_inject_reward(&net, reward_val, cfg.delay_ticks);
        
        /* Phase 5: 等待奖励到达 + 冷却期（足够让奖励队列被处理完） */
        /* 先让奖励到达：推进到 delay_ticks */
        int reward_arrival_tick = ticks_in_trial + cfg.delay_ticks + 1;
        while (net.current_tick < reward_arrival_tick) {
            net_tick(&net);
            ticks_in_trial++;
        }
        
        /* 冷却期：让网络回到静息状态 */
        for (int t = 0; t < 30; t++) {
            net_tick(&net);
            ticks_in_trial++;
        }
        
        /* 写CSV */
        fprintf(fp, "%d,%d,%d,%d,%.4f,%.4f,%.2f,%.4f,%d\n",
                trial, stimulus_type, predicted, is_correct,
                out_a_rate, out_b_rate, reward_val,
                net.reward_baseline, ticks_in_trial);
        
        /* 进度输出 */
        if ((trial + 1) % 200 == 0) {
            double test_acc = (trial >= train_total) 
                ? (double)test_correct / (trial - train_total + 1) * 100.0
                : 0.0;
            printf("  Trial %d/%d | Train Acc: %.1f%% | Test Acc (so far): %.1f%%\n",
                   trial + 1, cfg.trials,
                   trial > 0 ? (double)correct_count / (trial + 1) * 100.0 : 0.0,
                   test_acc);
        }
    }
    
    fclose(fp);
    
    /* 最终结果 */
    double train_acc = (double)train_correct / train_total * 100.0;
    double test_acc = (double)test_correct / test_total * 100.0;
    double overall_acc = (double)correct_count / cfg.trials * 100.0;
    
    printf("\n=== FINAL RESULTS ===\n");
    printf("Overall Accuracy:  %.2f%%\n", overall_acc);
    printf("Train Accuracy:    %.2f%% (%d/%d)\n", train_acc, train_correct, train_total);
    printf("Test Accuracy:     %.2f%% (%d/%d)\n", test_acc, test_correct, test_total);
    printf("Total ticks:       %d\n", net.current_tick);
    printf("Output file:       %s\n", out_file.c_str());
    
    /* 诊断信息 */
    FILE* diag = fopen("E:/shen_jing_yuan/Experiment/logs/benchmark1_diagnostic.txt", "w");
    if (diag) {
        fprintf(diag, "=== Benchmark-1 Diagnostic Log ===\n");
        fprintf(diag, "Timestamp: %s\n", __DATE__ " " __TIME__);
        fprintf(diag, "Config: S_A=%d S_B=%d H=%d O_A=%d O_B=%d Delay=%d Trials=%d Seed=%u\n",
                cfg.sensory_a, cfg.sensory_b, cfg.hidden, cfg.output_a, cfg.output_b,
                cfg.delay_ticks, cfg.trials, cfg.seed);
        fprintf(diag, "El Trace: %d, Reward Mod: %d\n",
                net.use_eligibility_trace, net.use_reward_modulation);
        fprintf(diag, "\n");
        net_dump_weight_stats(&net, diag);
        net_dump_trace_stats(&net, diag);
        fprintf(diag, "\n");
        fprintf(diag, "Hyperparameters:\n");
        fprintf(diag, "  tau_stdp=%.1f, tau_trace=%.1f, lambda_trace=%.6f\n",
                net.hp.tau_stdp, net.hp.tau_trace, net.hp.lambda_trace);
        fprintf(diag, "  stdp_a_plus=%.2f, stdp_a_minus=%.2f, lr=%.4f\n",
                net.hp.stdp_a_plus, net.hp.stdp_a_minus, net.hp.learning_rate);
        fprintf(diag, "  baseline_window=%d\n", net.hp.reward_baseline_window);
        fclose(diag);
    }
    
    /* 通过判据检查 */
    printf("\n=== PASS CRITERIA CHECK ===\n");
    bool pass = true;
    if (test_acc < 80.0) {
        printf("[FAIL] Test accuracy %.2f%% < 80%%\n", test_acc);
        pass = false;
    } else {
        printf("[PASS] Test accuracy %.2f%% >= 80%%\n", test_acc);
    }
    if (net.use_eligibility_trace && net.use_reward_modulation) {
        printf("[NOTE] Ablation checks must be run separately with --disable_trace / --disable_reward\n");
    }
    
    net_destroy(&net);
    
    printf("\n%s\n", pass ? "RESULT: PASS (run ablations next)" : "RESULT: FAIL (tune hyperparameters)");
    return pass ? 0 : 2;
}
