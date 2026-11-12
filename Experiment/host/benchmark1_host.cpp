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
    printf("  --jsonl FILE       Output JSONL log (train.jsonl compatible)\n");
    printf("  --tag NAME         Tag written into every JSONL record\n");
    printf("  --trace-k N        Eligibility trace scales: 1..3 (default: 3)\n");
    printf("  --tau-traces a,b,c Trace time constants (default: 20,100,500)\n");
    printf("  --learning-rate X  Override reward-modulated learning rate\n");
    printf("  --disable_trace     Disable eligibility trace (ablation)\n");
    printf("  --disable_reward    Disable reward modulation (ablation)\n");
}

int main(int argc, char* argv[]) {
    BenchmarkConfig cfg;
    std::string out_file = "data/benchmark1_results.csv";  /* 相对 Experiment 目录 */
    std::string jsonl_path;
    std::string tag = "default";
    bool force_no_trace = false;
    bool force_no_reward = false;
    int trace_k = 3;
    double tau_traces[3] = { 20.0, 100.0, 500.0 };
    double learning_rate_override = -1.0;
    double psp_override = -1.0;
    bool credit_winner_only = false;   /* true: 只更新被选输出的输入突触 */
    
    /* 解析命令行 */
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        auto next = [&](int& val) { if (i+1 < argc) val = atoi(argv[++i]); };
        auto nextu = [&](unsigned int& val) { if (i+1 < argc) val = (unsigned int)strtoul(argv[++i], nullptr, 10); };
        
        if (arg == "--sensory_a") next(cfg.sensory_a);
        else if (arg == "--sensory_b") next(cfg.sensory_b);
        else if (arg == "--hidden") next(cfg.hidden);
        else if (arg == "--output_a") next(cfg.output_a);
        else if (arg == "--output_b") next(cfg.output_b);
        else if (arg == "--delay") next(cfg.delay_ticks);
        else if (arg == "--trials") next(cfg.trials);
        else if (arg == "--seed") nextu(cfg.seed);
        else if (arg == "--out") { if (i+1 < argc) out_file = argv[++i]; }
        else if (arg == "--jsonl") { if (i+1 < argc) jsonl_path = argv[++i]; }
        else if (arg == "--tag") { if (i+1 < argc) tag = argv[++i]; }
        else if (arg == "--trace-k") next(trace_k);
        else if (arg == "--learning-rate") { if (i+1 < argc) learning_rate_override = atof(argv[++i]); }
        else if (arg == "--psp-factor") { if (i+1 < argc) psp_override = atof(argv[++i]); }
        else if (arg == "--tau-traces") {
            if (i+1 < argc) {
                double a = tau_traces[0], b = tau_traces[1], c = tau_traces[2];
                int n = sscanf(argv[++i], "%lf,%lf,%lf", &a, &b, &c);
                if (n >= 1) tau_traces[0] = a;
                if (n >= 2) tau_traces[1] = b;
                if (n >= 3) tau_traces[2] = c;
            }
        }
        else if (arg == "--credit-winner") credit_winner_only = true;
        else if (arg == "--disable_trace") force_no_trace = true;
        else if (arg == "--disable_reward") force_no_reward = true;
    }
    
    if (cfg.trials < 2 || cfg.delay_ticks < 0 || cfg.sensory_a < 1 ||
        cfg.sensory_b < 1 || cfg.hidden < 1 || cfg.output_a < 1 || cfg.output_b < 1) {
        fprintf(stderr, "Invalid config: trials >= 2, delay >= 0, layer sizes >= 1\n");
        return 1;
    }
    
    printf("=== Benchmark-1: Single-Synapse Delayed Credit Assignment ===\n");
    printf("Config: S_A=%d S_B=%d H=%d O_A=%d O_B=%d Delay=%d Trials=%d Seed=%u\n",
           cfg.sensory_a, cfg.sensory_b, cfg.hidden, cfg.output_a, cfg.output_b,
           cfg.delay_ticks, cfg.trials, cfg.seed);
    printf("El Trace: %s, Reward Mod: %s\n",
           force_no_trace ? "OFF (ABLATION)" : (ELIGIBILITY_TRACE_ENABLE ? "ON" : "OFF (COMPILE-TIME DISABLED)"),
           force_no_reward ? "OFF (ABLATION)" : (REWARD_MODULATION_ENABLE ? "ON" : "OFF (COMPILE-TIME DISABLED)"));
    printf("Eligibility scales: K=%d, tau=[%.1f, %.1f, %.1f], alpha=1/K, lr=%.5f%s\n",
           trace_k, tau_traces[0], tau_traces[1], tau_traces[2],
           learning_rate_override > 0 ? learning_rate_override : 0.005,
           learning_rate_override > 0 ? " (override)" : "");
    
    /* 初始化网络 */
    NetworkState net;
    if (net_init(&net, cfg.sensory_a, cfg.sensory_b, cfg.hidden, cfg.output_a, cfg.output_b,
                 cfg.avg_conn, cfg.seed) != 0) {
        fprintf(stderr, "Failed to initialize network\n");
        return 1;
    }
    
    /* 应用多尺度迹 / 学习率配置（必须在消融开关之前） */
    {
        HyperParams hp = net.hp;
        hp.trace_k = trace_k;
        for (int k = 0; k < 3; k++) hp.tau_trace_k[k] = tau_traces[k];
        if (learning_rate_override > 0.0) hp.learning_rate = learning_rate_override;
        if (psp_override > 0.0) hp.psp_factor = psp_override;
        net_set_hyperparams(&net, &hp);
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

    /* JSONL 日志（M1）：每 trial 一行，末尾一条 summary */
    FILE* jf = nullptr;
    if (!jsonl_path.empty()) {
        jf = fopen(jsonl_path.c_str(), "w");
        if (!jf) fprintf(stderr, "WARN: cannot open jsonl: %s\n", jsonl_path.c_str());
    }
    
    /* 元数据 sidecar：与 B2 一致，记录超参与构建信息 */
    std::string meta_path = out_file + ".meta.json";
    FILE* mf = fopen(meta_path.c_str(), "w");
    if (mf) {
        std::string out_j = out_file;
        for (char& ch : out_j) if (ch == '\\') ch = '/';
        fprintf(mf, "{\n");
        fprintf(mf, "  \"benchmark\": 1,\n");
        fprintf(mf, "  \"build\": \"%s %s\",\n", __DATE__, __TIME__);
        fprintf(mf, "  \"out_csv\": \"%s\",\n", out_j.c_str());
        fprintf(mf, "  \"sensory_a\": %d, \"sensory_b\": %d, \"hidden\": %d,\n",
                cfg.sensory_a, cfg.sensory_b, cfg.hidden);
        fprintf(mf, "  \"output_a\": %d, \"output_b\": %d, \"avg_conn\": %.4f,\n",
                cfg.output_a, cfg.output_b, cfg.avg_conn);
        fprintf(mf, "  \"trials\": %d, \"delay_ticks\": %d, \"seed\": %u,\n",
                cfg.trials, cfg.delay_ticks, cfg.seed);
        fprintf(mf, "  \"el_trace\": %d, \"reward_mod\": %d,\n",
                net.use_eligibility_trace, net.use_reward_modulation);
        fprintf(mf, "  \"trace_k\": %d, \"tau_trace\": [%.1f, %.1f, %.1f],\n",
                net.hp.trace_k, net.hp.tau_trace_k[0], net.hp.tau_trace_k[1], net.hp.tau_trace_k[2]);
        fprintf(mf, "  \"trace_alpha\": [%.4f, %.4f, %.4f], \"learning_rate\": %.6f,\n",
                net.hp.trace_alpha[0], net.hp.trace_alpha[1], net.hp.trace_alpha[2],
                net.hp.learning_rate);
        {
            std::string jpath = jsonl_path;
            for (char& ch : jpath) if (ch == '\\') ch = '/';
            fprintf(mf, "  \"jsonl\": \"%s\",\n", jpath.c_str());
        }
        fprintf(mf, "  \"neurons\": %d, \"synapses\": %d\n", net.neuron_count, net.synapse_count);
        fprintf(mf, "}\n");
        fclose(mf);
    }
    
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
        
        /* Phase 1: 刺激注入 + 观察窗口内累计输出发放 */
        int peak_a_count = 0, peak_b_count = 0;
        
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
            
            /* 窗口内累计，避免单 tick 峰值噪声 */
            for (int i = 0; i < net.neuron_count; i++) {
                if (net.neurons[i].is_spiking) {
                    if (net.neurons[i].type == NT_OUTPUT_A) peak_a_count++;
                    if (net.neurons[i].type == NT_OUTPUT_B) peak_b_count++;
                }
            }
        }
        
        /* Phase 2: 用峰值时刻的快照投票 */
        double out_a_rate = (double)peak_a_count / (net.output_a_count * OBSERVE_WINDOW);
        double out_b_rate = (double)peak_b_count / (net.output_b_count * OBSERVE_WINDOW);
        
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
        /* 信用范围：默认全局 R-STDP；--credit-winner 只更新被选输出。
         * 两者都保留，用于 B1 消融对比。 */
        if (credit_winner_only) {
            net_inject_reward_vote(&net, reward_val, cfg.delay_ticks, predicted);
        } else {
            net_inject_reward(&net, reward_val, cfg.delay_ticks);
        }
        
        /* Phase 5: 等待奖励到达 + 冷却期（足够让奖励队列被处理完） */
        /* 先让奖励到达：推进到 delay_ticks */
        /* 必须用全局 tick：奖励队列的 delivery_tick 也是全局 tick。
         * 旧实现用 trial 局部 ticks_in_trial，导致延迟从未真正生效。 */
        int reward_arrival_tick = net.current_tick + cfg.delay_ticks + 1;
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

        /* 写 JSONL：一行一个 trial，字段与 spec/judgments.md 对齐 */
        if (jf) {
            fprintf(jf,
                "{\"event\":\"trial\",\"tag\":\"%s\",\"trial\":%d,\"stimulus\":%d,"
                "\"vote\":%d,\"correct\":%d,\"out_a\":%.4f,\"out_b\":%.4f,"
                "\"reward\":%.2f,\"baseline\":%.4f,\"ticks\":%d,"
                "\"delay\":%d,\"trace_k\":%d,\"seed\":%u,\"phase\":\"%s\"}\n",
                tag.c_str(), trial, stimulus_type, predicted, is_correct,
                out_a_rate, out_b_rate, reward_val, net.reward_baseline,
                ticks_in_trial, cfg.delay_ticks, net.hp.trace_k, cfg.seed,
                (trial < train_total) ? "train" : "test");
        }
        
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

    /* JSONL summary：一个条件一条，便于 tools/b1_report.py 汇总 */
    if (jf) {
        fprintf(jf,
            "{\"event\":\"summary\",\"tag\":\"%s\",\"trials\":%d,"
            "\"train_acc\":%.2f,\"test_acc\":%.2f,\"delay\":%d,"
            "\"trace_k\":%d,\"tau_trace\":[%.1f,%.1f,%.1f],"
            "\"alpha\":[%.4f,%.4f,%.4f],\"lr\":%.6f,"
            "\"seed\":%u,\"neurons\":%d,\"synapses\":%d,"
            "\"el_trace\":%d,\"reward_mod\":%d}\n",
            tag.c_str(), cfg.trials,
            train_total > 0 ? (double)train_correct / train_total * 100.0 : 0.0,
            test_total > 0 ? (double)test_correct / test_total * 100.0 : 0.0,
            cfg.delay_ticks, net.hp.trace_k,
            net.hp.tau_trace_k[0], net.hp.tau_trace_k[1], net.hp.tau_trace_k[2],
            net.hp.trace_alpha[0], net.hp.trace_alpha[1], net.hp.trace_alpha[2],
            net.hp.learning_rate, cfg.seed, net.neuron_count, net.synapse_count,
            net.use_eligibility_trace, net.use_reward_modulation);
        fclose(jf);
    }

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
    FILE* diag = fopen("logs/benchmark1_diagnostic.txt", "w");  /* 相对 Experiment 目录 */
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
        fprintf(diag, "  trace_k=%d, tau_k=[%.1f, %.1f, %.1f], alpha=[%.4f, %.4f, %.4f]\n",
                net.hp.trace_k, net.hp.tau_trace_k[0], net.hp.tau_trace_k[1], net.hp.tau_trace_k[2],
                net.hp.trace_alpha[0], net.hp.trace_alpha[1], net.hp.trace_alpha[2]);
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
