/*
 * benchmark2_host.cpp - Benchmark-2 宿主测试驱动
 * 两关节虚拟臂到达任务
 *
 * 场景：每轮生成目标（默认每轮随机换位置，也可固定/循环）→ 一路跑到碰到目标
 *       → 结算奖励并重置重来（无超时；时间与步数只用于评分）
 * 通过判据：训练后末端距离的标准中位数 ≤ 初始距离 20%
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
    int hidden_n   = 2000;   /* 隐藏层必须有！让 motor 层有足够突触电流 spike */
    int motor1_n   = 50;     /* 关节1运动神经元 */
    int motor2_n   = 50;     /* 关节2运动神经元 */
    double avg_conn = 8.0;
double rho_target = 0.0;   /* v20：H→H 目标谱半径，0=关闭（旧行为） */
    
    int episodes     = 20;   /* 尝试轮数：每轮一直跑到碰到目标 */
    int score_horizon = 1000; /* 速度评分参考步数：只参与奖励计算，不终止本轮 */
    int max_ticks_per_ep = 100000; /* 单轮安全阀，防止永远到不了目标；0 = 不设限（不是评分规则） */
    unsigned int seed = 12345;
    std::string out_csv = "data_b2/results.csv";  /* 相对 Experiment 目录，跨机器可复现 */
    
    /* 可回退的实验开关：默认值 = 旧版硬编码值 */
    double psp_factor = 25.0;
    /* v14（M1，负责人 2026-11-16 签字）：内在基电流，默认 = 旧字面量 */
    double hidden_base = 2.0;
    double motor_base  = 3.0;
    /* v15（M3，负责人 2026-11-16 签字「签字」）：学习增益与权重映射增益的独立缩放
     * lr_scale     = lr 阶梯整体乘子（默认 1.0 = 0.35/0.20/0.04 不变）
     * weight_scale = 初始化后全部权重整体乘子（默认 1.0 = 不调用 net2_scale_weights） */
    double lr_scale     = 1.0;
    double weight_scale = 1.0;

    /* v16（工具缺口，2026-11-17 负责人「可以」）：
     * save/load 权重 + 逐 tick 命令导出；三者默认空/关闭 → 旧行为逐位不变 */
    std::string save_weights;   /* 训练结束后导出权重的路径 */
    std::string save_init_weights; /* 初始化后、任何 tick 之前导出（作为 Δ 统计的基线） */
    std::string load_weights;   /* 初始化后导入权重（规模不符则报错退出） */
    std::string trace_cmd;      /* 逐 tick 命令 JSONL（ep,t,target,end,theta,om,om*,cf_err） */
    int trace_stride = 5;       /* 命令导出采样间隔（tick） */

    /* v17（M4，负责人 2026-11-17「签字」）：symmetric e-prop。
     * 默认 eprop=0 ⇒ 旧 R-STDP 路径逐位不变；开启后 R-STDP/误差通道整体关闭。 */
    int    eprop       = 0;      /* 0=旧规则 1=e-prop */
    int    eprop_scope = 0;      /* 0=全通路 1=仅 I→M（消融臂 D） */
    double eprop_alpha = 0.3;    /* 伪导数斜率 */
    double eprop_lr    = 1e-3;   /* ADAM 步长（P3 由单 seed 校准后写死） */

    /* v21（B2-ALIF A 路径，2026-11-19 签字）：alif_beta=0 默认关闭 ⇒ 旧路径逐位不变。
     * 只改前向动力学（阈值随发放适应），不触碰 e-prop。β 档只允许 {4,8}。 */
    double alif_beta   = 0.0;
    double alif_tau_a  = 100.0;
    int    alif_bpath  = 0;      /* 0=A 路径（仅前向）；1=B 路径（叠加适应资格向量） */

    /* v22（B2-EI，2026-11-20 签字）：IIN 侧抑制，默认关闭逐位不变 */
    int    ei_mode = 0;
    int    ei_n    = 0;
    double ei_w    = 0.0;
    double ei_tau  = 100.0;
    double ei_sigma = 0.25;

    /* v18（B2-struct，2026-11-18 签字，audit/b2_struct_prereg_2026-11-18.md）
     * 0=关闭（默认，逐位不变） 1=活动依赖生长 2=随机生长（对照） 3=按索引补入度 */
    int    struct_mode     = 0;
    int    struct_fmin     = 150;
    int    struct_interval = 100;
    int    struct_score    = 0;    /* v19：0=coact（旧，逐位不变） 1=rate[] */
    double trace_clamp = 1.0;
    double tau_trace = 2000.0;
    int decode_mode = 0;          /* 0=WTA 均值, 2=膜电位加权 */
    int wta_k = 3;
    int reward_mode = 0;          /* 0=combo1, 1=势能(1-dist/init) */
    int use_reward_baseline = 1;
    std::string preset = "legacy";
    int target_mode = 0;          /* 0=每轮随机换位置, 1=固定, 2=四目标循环 */
    double fixed_tx = 1.5, fixed_ty = 0.5;

    /* 一轮 = 从初始臂形一直到碰到目标；没有超时/步数终止，碰到即结算并重置重来。
     * 时间与步数只作为评分项：
     *   每 tick 扣 step_cost；首次到达奖励 = arrive_bonus +
     *   speed_bonus × max(0, (score_horizon-已用步数)/score_horizon)，
     *   到得越快，剩余比例越高、奖励越高；超过参考步数只是速度项为 0，不结束本轮。
     * speed_mode=0 回退到旧的「固定步数、不结算」对照行为。
     * 常数 step_cost 会被滑动基线大部分抵消，真正驱动“更快”的是首达奖励，请保持 baseline=1。 */
    int speed_mode       = 1;
    double step_cost     = 0.01;  /* 每 tick 步数成本 */
    double arrive_radius = 0.15;  /* 碰到目标的判定半径 */
    double arrive_bonus  = 1.0;   /* 首次碰到目标的基础奖励 */
    double speed_bonus   = 5.0;   /* 速度评分系数（按剩余步数比例给分） */

    /* v6：三因子调制 / 攀爬纤维误差 / 异构群体编码（默认全关 = 旧版） */
    double error_gain    = 0.0;   /* 小脑攀爬纤维：关节级有符号误差增益 */
    /* v13（T2r，签字项）：误差通道可塑范围。0=仅运动层输入突触（默认，旧路径逐位不变）；
     * 1=额外作用到隐藏层输入突触，方向用 H→M 权重的净贡献（反馈对齐式信用）。 */
    int    error_scope      = 0;
    double error_scope_gain = 0.1;
    double novelty_gain  = 0.0;   /* 乙酰胆碱：惊讶度放大学习率 */
    double cf_lambda     = 1.0;   /* 逆运动学期望速度缩放 */
    double hetero_tau    = 0.0;   /* 运动层膜时间常数对数正态 sigma */
    double hetero_thresh = 0.0;   /* 运动层阈值高斯 sigma (mV) */
    double hetero_tune   = 0.0;   /* 运动层调谐宽度对数正态 sigma */

    /* v7（M2 探针）：固定预算 + 评估段 + 纯状态输入 */
    int budget        = 0;   /* >0：每 episode 固定跑这么多 tick，到达也不提前结束（§0.3） */
    int prop_mode     = 0;   /* 0=full(含目标方位差) 1=raw(只 θ1/θ2) */
    int eval_episodes = 0;   /* >0：最后 N 轮冻结学习（use_reward_modulation=0）做评估 */

    /* v8（T2 解码可学性上界）：hidden 发放导出 / 外部线性读出 / 链路对照 */
    std::string dump_hidden;    /* 非空：按 stride 导出 hidden 发放到该二进制文件 */
    std::string load_readout;   /* 非空：读取 2*hidden 个 double 作为外部线性读出 */
    int dump_layer    = 0;      /* v9：导出源层 0=hidden 1=input */
    int readout_layer = 0;      /* v9：读出源层 0=hidden 1=input */
    int stride      = 10;       /* 导出采样间隔（tick） */
    int retarget    = 0;        /* v10：>0 时每 N tick 换一个新随机目标（仅采集用，保证状态覆盖） */

    /* v11（T2h 结构改动，负责人签字）：直通/残差通路 */
    int    direct_fan = 12;     /* I→M 每层扇出；12 = 旧拓扑 */
    double hm_scale   = 1.0;    /* H→M 初始权重缩放；0 = 不建该通路（纯直通） */
    double im_scale   = 1.0;    /* I→M 初始权重缩放（加扇出时保持总驱动） */

    /* v12（T2j 时序/速率编码） */
    double rate_tau      = 0.0;  /* >0 开启每神经元速率迹（EMA 发放率） */
    int    dump_values   = 0;    /* 0=瞬时 0/1；1=速率（uint8 量化 ×255） */
    int    readout_values = 0;   /* 0=瞬时 0/1；1=速率迹 */
    int drive_direct = 0;       /* 1=用解析 ω* 直接驱动臂（评估链路对照） */
};

static unsigned int h_rand;
static inline double hu() {
    h_rand = h_rand * 1103515245 + 12345;
    return (double)((h_rand >> 16) & 0x7FFF) / 32767.0;
}

/* 目标生成：target_mode 0=每轮随机换位置 1=固定 2=四目标循环
 * 随机模式限制在原点圆与 (2,0) 圆的交集内，保证两关节可达。 */
static void gen_target(const B2Config& cfg, int ep, double* tx, double* ty) {
    if (cfg.target_mode == 1) {
        *tx = cfg.fixed_tx;
        *ty = cfg.fixed_ty;
        return;
    }
    if (cfg.target_mode == 2) {
        static const double T4[4][2] = {{1.5,0.5},{1.5,-0.5},{1.0,0.9},{1.0,-0.9}};
        *tx = T4[ep % 4][0];
        *ty = T4[ep % 4][1];
        return;
    }
    double max_reach = ARM_L1 + ARM_L2;
    double sx = max_reach;
    for (int tries = 0; tries < 100; tries++) {
        double r = 0.3 + hu() * 1.5;
        double theta = hu() * 2.0 * M_PI;
        *tx = sx + r * cos(theta);
        *ty = r * sin(theta);

        double dist_origin = sqrt(*tx * *tx + *ty * *ty);
        if (dist_origin <= max_reach - 0.01) {
            double dist_from_start = sqrt((*tx - sx) * (*tx - sx) + (*ty) * (*ty));
            if (dist_from_start >= 0.4 && dist_from_start <= max_reach * 0.95) {
                return;
            }
        }
    }
    *tx = cfg.fixed_tx;
    *ty = cfg.fixed_ty;
}

int main(int argc, char* argv[]) {
    B2Config cfg;
    bool force_no_trace = false;
    bool force_no_reward = false;

    /* v7：--preset m2 的默认值在解析之前写入，令后续 CLI 参数可覆盖（便于冒烟与单变量 A/B）。
     * 旧预设 fix1/fix2/fix3 仍在解析之后按原顺序叠加，保持历史行为逐位不变。 */
    {
        std::string pre;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--preset" && i + 1 < argc) pre = argv[i + 1];
        }
        if (pre == "m2") {
            cfg.input_n = 1000; cfg.hidden_n = 3500;
            cfg.motor1_n = 250; cfg.motor2_n = 250;
            cfg.budget = 1200;
            cfg.decode_mode = 3;
            cfg.reward_mode = 1;
            cfg.use_reward_baseline = 1;
            cfg.cf_lambda = 0.0;
            cfg.error_gain = 0.0;
            cfg.novelty_gain = 0.0;
            cfg.hetero_tau = 0.25;
            cfg.hetero_thresh = 2.0;
            cfg.hetero_tune = 0.25;
            cfg.prop_mode = 1;
            cfg.episodes = 320;
            cfg.eval_episodes = 20;
            cfg.score_horizon = 1200;
            cfg.target_mode = 0;
        }
    }

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int& v) { if (i+1<argc) v = atoi(argv[++i]); };
        auto nextf = [&](double& v) { if (i+1<argc) v = atof(argv[++i]); };
        if (a == "--input")     next(cfg.input_n);
        else if (a == "--hidden")  next(cfg.hidden_n);
        else if (a == "--motor")   { next(cfg.motor1_n); cfg.motor2_n = cfg.motor1_n; }
        else if (a == "--episodes") next(cfg.episodes);
        else if (a == "--ticks")   next(cfg.score_horizon);   /* 兼容旧名：评分参考步数 */
        else if (a == "--score-horizon") next(cfg.score_horizon);
        else if (a == "--max-ticks")     next(cfg.max_ticks_per_ep);
        else if (a == "--target-mode") {
            std::string m = (i+1<argc) ? argv[++i] : "";
            cfg.target_mode = (m == "fixed") ? 1 : (m == "cycle" ? 2 : 0);
        }
        else if (a == "--target-x")     nextf(cfg.fixed_tx);
        else if (a == "--target-y")     nextf(cfg.fixed_ty);
        else if (a == "--random-target") cfg.target_mode = 0;
        else if (a == "--fixed-target")  cfg.target_mode = 1;
        else if (a == "--seed")    { if (i+1<argc) cfg.seed = strtoul(argv[++i], nullptr, 10); }
        else if (a == "--out")     { if (i+1<argc) cfg.out_csv = argv[++i]; }
        else if (a == "--disable_trace")  force_no_trace = true;
        else if (a == "--disable_reward") force_no_reward = true;
        else if (a == "--psp")          nextf(cfg.psp_factor);
        else if (a == "--hidden-base")  nextf(cfg.hidden_base);
        else if (a == "--motor-base")   nextf(cfg.motor_base);
        else if (a == "--lr-scale")     nextf(cfg.lr_scale);
        else if (a == "--weight-scale") nextf(cfg.weight_scale);
        else if (a == "--save-weights")  { if (i+1<argc) cfg.save_weights = argv[++i]; }
        else if (a == "--save-init-weights") { if (i+1<argc) cfg.save_init_weights = argv[++i]; }
        else if (a == "--load-weights")  { if (i+1<argc) cfg.load_weights = argv[++i]; }
        else if (a == "--trace-cmd")     { if (i+1<argc) cfg.trace_cmd = argv[++i]; }
        else if (a == "--trace-stride")  next(cfg.trace_stride);
        else if (a == "--eprop")         cfg.eprop = 1;
        else if (a == "--eprop-scope")   { std::string m = (i+1<argc) ? argv[++i] : "";
                                           cfg.eprop_scope = (m == "im" || m == "I-M" || m == "i2m") ? 1 : 0; }
        else if (a == "--eprop-alpha")   nextf(cfg.eprop_alpha);
        else if (a == "--eprop-lr")      nextf(cfg.eprop_lr);
        else if (a == "--alif-beta")     nextf(cfg.alif_beta);
        else if (a == "--alif-tau-a")    nextf(cfg.alif_tau_a);
        else if (a == "--alif-bpath")    { if (i+1<argc) cfg.alif_bpath = atoi(argv[++i]); }
        else if (a == "--ei-mode")       { if (i+1<argc) cfg.ei_mode = atoi(argv[++i]); }
        else if (a == "--ei-n")          { if (i+1<argc) cfg.ei_n = atoi(argv[++i]); }
        else if (a == "--ei-w")          nextf(cfg.ei_w);
        else if (a == "--ei-tau")        nextf(cfg.ei_tau);
        else if (a == "--ei-sigma")      nextf(cfg.ei_sigma);
        else if (a == "--struct")        { if (i+1<argc) cfg.struct_mode = atoi(argv[++i]); }
        else if (a == "--struct-fmin")   { if (i+1<argc) cfg.struct_fmin = atoi(argv[++i]); }
        else if (a == "--struct-score")  { if (i+1<argc) cfg.struct_score = atoi(argv[++i]); }
        else if (a == "--struct-interval") { if (i+1<argc) cfg.struct_interval = atoi(argv[++i]); }
        else if (a == "--trace-clamp")  nextf(cfg.trace_clamp);
        else if (a == "--tau-trace")    nextf(cfg.tau_trace);
        else if (a == "--wta-k")        next(cfg.wta_k);
        else if (a == "--decode-mode")  {
            std::string m = (i+1<argc) ? argv[++i] : "";
            cfg.decode_mode = (m == "weighted") ? 2 : (m == "pdc" ? 3 : 0);
        }
        else if (a == "--budget")        next(cfg.budget);
        else if (a == "--eval-episodes") next(cfg.eval_episodes);
        else if (a == "--avg-conn")      nextf(cfg.avg_conn);
        else if (a == "--rho-target")    nextf(cfg.rho_target);
        else if (a == "--dump-hidden")   { if (i+1<argc) cfg.dump_hidden = argv[++i]; }
        else if (a == "--load-readout")  { if (i+1<argc) cfg.load_readout = argv[++i]; }
        else if (a == "--dump-layer")    { std::string m = (i+1<argc) ? argv[++i] : ""; cfg.dump_layer = (m == "input") ? 1 : 0; }
        else if (a == "--readout-layer") { std::string m = (i+1<argc) ? argv[++i] : "";
                                           cfg.readout_layer = (m == "input") ? 1 : (m == "both") ? 2 : 0; }
        else if (a == "--stride")        next(cfg.stride);
        else if (a == "--retarget")      next(cfg.retarget);
        else if (a == "--direct-fan")    next(cfg.direct_fan);
        else if (a == "--hm-scale")      nextf(cfg.hm_scale);
        else if (a == "--im-scale")      nextf(cfg.im_scale);
        else if (a == "--rate-tau")      nextf(cfg.rate_tau);
        else if (a == "--dump-values")   { std::string m = (i+1<argc) ? argv[++i] : ""; cfg.dump_values = (m == "rate") ? 1 : 0; }
        else if (a == "--readout-values"){ std::string m = (i+1<argc) ? argv[++i] : ""; cfg.readout_values = (m == "rate") ? 1 : 0; }
        else if (a == "--drive-direct")  cfg.drive_direct = 1;
        else if (a == "--prop-mode")     { std::string m = (i+1<argc) ? argv[++i] : ""; cfg.prop_mode = (m == "raw") ? 1 : 0; }
        else if (a == "--reward-mode")  { std::string m = (i+1<argc) ? argv[++i] : ""; cfg.reward_mode = (m == "potential") ? 1 : 0; }
        else if (a == "--no-reward-baseline") cfg.use_reward_baseline = 0;
        else if (a == "--preset")       { if (i+1<argc) cfg.preset = argv[++i]; }
        else if (a == "--speed-reward")     cfg.speed_mode = 1;
        else if (a == "--no-speed-reward")  cfg.speed_mode = 0;
        else if (a == "--step-cost")        nextf(cfg.step_cost);
        else if (a == "--arrive-radius")    nextf(cfg.arrive_radius);
        else if (a == "--arrive-bonus")     nextf(cfg.arrive_bonus);
        else if (a == "--speed-bonus")      nextf(cfg.speed_bonus);
        else if (a == "--error-gain")       nextf(cfg.error_gain);
        else if (a == "--error-scope") {
            std::string m = (i+1<argc) ? argv[++i] : "";
            cfg.error_scope = (m == "hidden") ? 1 : 0;
        }
        else if (a == "--error-scope-gain") nextf(cfg.error_scope_gain);
        else if (a == "--novelty-gain")     nextf(cfg.novelty_gain);
        else if (a == "--cf-lambda")        nextf(cfg.cf_lambda);
        else if (a == "--hetero-tau")       nextf(cfg.hetero_tau);
        else if (a == "--hetero-thresh")    nextf(cfg.hetero_thresh);
        else if (a == "--hetero-tune")      nextf(cfg.hetero_tune);
        else if (a == "--help" || a == "-h") {
            printf("Usage: %s [--input N --hidden N --motor N --episodes N --seed N --out FILE]\n"
                   "       [--score-horizon N (速度评分参考步数，不终止本轮；--ticks 为旧别名)]\n"
                   "       [--max-ticks N (单轮安全阀，0=不设限，默认 100000)]\n"
                   "       [--target-mode random|fixed|cycle] [--target-x F --target-y F]\n"
                   "       [--random-target] [--fixed-target]\n"
                   "       [--disable_trace] [--disable_reward]\n"
                   "       [--preset legacy|fix1|fix2|fix3]\n"
                   "       [--psp F --wta-k K --decode-mode wta|weighted --reward-mode combo1|potential]\n"
                   "       [--hidden-base F --motor-base F]   (内在基电流 mV/tick，默认 2.0 / 3.0)\n"
                   "       [--lr-scale F --weight-scale F]  (学习增益 / 权重映射增益，默认 1.0)\n"
                   "       [--save-weights P --save-init-weights P --load-weights P --trace-cmd P [--trace-stride N]]\n"
                   "       [--eprop [--eprop-scope all|im] [--eprop-alpha F] [--eprop-lr F]]\n"
                   "       [--struct 0|1|2|3 [--struct-fmin N] [--struct-interval N]]  (结构可塑：只生长)\n"
                   "       [--no-reward-baseline --tau-trace F --trace-clamp F]\n"
                   "       [--speed-reward (默认：每步成本 + 首达奖励 + 碰到即结算重来)]\n"
                   "       [--no-speed-reward (旧对照：固定步数、不结算)]\n"
                   "       [--step-cost F --arrive-radius F --arrive-bonus F --speed-bonus F]\n"
                   "       [--error-gain F  攀爬纤维：关节级有符号运动误差增益(默认0=关)]\n"
                   "       [--novelty-gain F 乙酰胆碱：惊讶度放大学习率(默认0=关)]\n"
                   "       [--cf-lambda F    逆运动学期望速度缩放(默认1.0)]\n"
                   "       [--hetero-tau F --hetero-thresh F --hetero-tune F  运动层异质化]\n"
                   "       [--preset m2 --budget N --prop-mode raw|full --avg-conn F]\n"
                   "       [--dump-hidden FILE [--dump-layer hidden|input] --stride N --retarget N]\n"
                   "       [--load-readout FILE [--readout-layer hidden|input|both]] [--drive-direct]\n"
                   "       [--direct-fan N   I→M 每层扇出，默认 12 = 旧拓扑（T2h 结构改动）]\n"
                   "       [--hm-scale F     H→M 初始权重缩放，0 = 不建该通路（纯直通臂）]\n"
                   "       [--rate-tau F     速率迹时间常数（tick，0=关）；配 --dump-values rate / --readout-values rate]\n", argv[0]);
            return 0;
        }
    }
    
    /* 预设逐级叠加，便于 A/B；legacy 与旧版完全等价。
     * A/B 结论：fix1 有效（median 0.82→0.62）；fix2/fix3 的势能奖励与短 trace 实测更差，
     * 保留仅作对照，正式实验请使用 legacy 或 fix1。 */
    if (cfg.preset == "fix1" || cfg.preset == "fix2" || cfg.preset == "fix3") {
        cfg.wta_k = 1;             /* 1) 解码层：单 winner，避免正负 pref_omega 互相抵消 */
    }
    if (cfg.preset == "fix2" || cfg.preset == "fix3") {
        cfg.reward_mode = 1; cfg.use_reward_baseline = 0;          /* 2) 奖励层（对照用） */
    }
    if (cfg.preset == "fix3") {
        cfg.tau_trace = 200.0; cfg.trace_clamp = 2.0;              /* 3) 资格迹（对照用） */
    }
    /* v7：M2 预设的默认值已在解析前写入（见 main 开头），这里只做合法性校验，
     * 不再二次覆盖 CLI。 */
    if (cfg.preset != "legacy" && cfg.preset != "fix1" &&
        cfg.preset != "fix2" && cfg.preset != "fix3" && cfg.preset != "m2") {
        fprintf(stderr, "Unknown preset: %s\n", cfg.preset.c_str());
        return 1;
    }
    
    if (cfg.episodes < 1 || cfg.score_horizon < 1 || cfg.max_ticks_per_ep < 0 || cfg.input_n < 2 ||
        cfg.hidden_n < 1 || cfg.motor1_n < 2 || cfg.motor2_n < 2 ||
        cfg.psp_factor <= 0.0 || cfg.tau_trace < 1.0 ||
        cfg.wta_k < 1 || cfg.wta_k > 64) {
        fprintf(stderr, "Invalid config: episodes/ticks >= 1, input/motor >= 2, hidden >= 1, "
                        "psp > 0, tau_trace >= 1, 1 <= wta-k <= 64\n");
        return 1;
    }
    if (cfg.speed_mode &&
        (cfg.step_cost < 0.0 || cfg.arrive_radius <= 0.0 ||
         cfg.arrive_bonus < 0.0 || cfg.speed_bonus < 0.0)) {
        fprintf(stderr, "Invalid speed config: step_cost/arrive_bonus/speed_bonus >= 0, "
                        "arrive_radius > 0\n");
        return 1;
    }
    if (cfg.speed_mode && !cfg.use_reward_baseline) {
        fprintf(stderr, "[WARN] --no-reward-baseline 与 --speed-reward 并用时，每步成本会变成"
                        "恒定负误差，可能持续压低权重\n");
    }
    
    h_rand = cfg.seed + 777;
    
    printf("=== Benchmark-2: 2-Joint Reaching ===\n");
    printf("Config: in=%d h=%d m1=%d m2=%d episodes=%d score_horizon=%d max_ticks/ep=%d seed=%u\n",
           cfg.input_n, cfg.hidden_n, cfg.motor1_n, cfg.motor2_n,
           cfg.episodes, cfg.score_horizon, cfg.max_ticks_per_ep, cfg.seed);
    
    /* 调制队列含 8192 个事件（v6 每条 6 个 double），放静态区避免大栈帧 */
    static NetworkState2 net;
    if (net2_init_ex(&net, cfg.input_n, cfg.hidden_n, cfg.motor1_n, cfg.motor2_n,
                     cfg.avg_conn, cfg.seed, cfg.direct_fan, cfg.hm_scale,
                     cfg.im_scale) != 0) {
        fprintf(stderr, "Init failed\n");
        return 1;
    }
    /* v12（T2j）：速率迹必须先于外部读出开关设置（setter 依赖缓冲已分配） */
    if (cfg.rate_tau > 0.0) net2_set_rate_trace(&net, cfg.rate_tau);
    if (cfg.rate_tau > 0.0 && cfg.readout_values == 1) net2_set_ext_rate(&net, 1);

    /* 把 CLI/预设参数写入运行时超参（net2_init 里是旧版默认值） */
    net.hp.psp_factor = cfg.psp_factor;
    net.hp.hidden_base_current = cfg.hidden_base;
    net.hp.motor_base_current  = cfg.motor_base;
    /* v17（M4）：e-prop 开关（默认 0，旧路径逐位不变） */
    net.hp.eprop_mode  = cfg.eprop;
    net.hp.eprop_scope = cfg.eprop_scope;
    net.hp.eprop_alpha = cfg.eprop_alpha;
    net.hp.eprop_lr    = cfg.eprop_lr;
    /* v21（B2-ALIF A 路径）：前向适应开关（默认 0 ⇒ 旧路径逐位不变）。
     * β 档只允许 {4,8}（预注册 §3）；其他值直接拒绝，防事后调参。 */
    if (cfg.alif_beta != 0.0 && cfg.alif_beta != 4.0 && cfg.alif_beta != 8.0) {
        fprintf(stderr, "ALIF beta must be 0 (off), 4, or 8 (got %g)\n", cfg.alif_beta);
        net2_destroy(&net);
        return 2;
    }
    if (cfg.alif_bpath != 0 && cfg.alif_bpath != 1) {
        fprintf(stderr, "ALIF bpath must be 0 (A) or 1 (B)\n");
        net2_destroy(&net);
        return 2;
    }
    net.hp.alif_beta  = cfg.alif_beta;
    net.hp.alif_tau_a = cfg.alif_tau_a;
    net.hp.alif_bpath = cfg.alif_bpath;
    if (cfg.alif_beta > 0.0) {
        printf("v21 ALIF: beta=%.1f mV tau_a=%.1f rho_a=%.6f path=%s\n",
               cfg.alif_beta, cfg.alif_tau_a, exp(-1.0 / cfg.alif_tau_a),
               cfg.alif_bpath ? "B (forward + adaptive eligibility)" : "A (forward only)");
    }
    /* v22（B2-EI）：IIN 侧抑制。默认关闭；开启时校验档位与容量。 */
    if (cfg.ei_mode != 0 && cfg.ei_mode != 1 && cfg.ei_mode != 2) {
        fprintf(stderr, "EI mode must be 0 (off), 1 (v22 uniform), or 2 (v23 gaussian)\n");
        net2_destroy(&net);
        return 2;
    }
    if (cfg.ei_n < 0 || cfg.ei_n > 256) {
        fprintf(stderr, "EI n must be in [0,256] (EI_MAX_N)\n");
        net2_destroy(&net);
        return 2;
    }
    if (cfg.ei_mode == 1 && cfg.ei_n <= 0) {
        fprintf(stderr, "EI mode=1 requires ei_n>0\n");
        net2_destroy(&net);
        return 2;
    }
    net.hp.ei_mode = cfg.ei_mode;
    net.hp.ei_n    = cfg.ei_n;
    net.hp.ei_w    = cfg.ei_w;
    net.hp.ei_tau  = cfg.ei_tau;
    net.hp.ei_sigma = cfg.ei_sigma;
    if (cfg.ei_mode == 2) {
        printf("v23 EI2: mode=2 n=%d w=%.2f tau=%.1f sigma=%.2f (gaussian selective + activity-dependent)\n",
               cfg.ei_n, cfg.ei_w, cfg.ei_tau, cfg.ei_sigma);
    } else if (cfg.ei_mode == 1) {
        printf("v22 EI: mode=1 n=%d w=%.2f tau=%.1f (IIN uniform lateral inhibition)\n",
               cfg.ei_n, cfg.ei_w, cfg.ei_tau);
    }
    /* v18（B2-struct）：结构可塑开关（默认 0 ⇒ 旧路径逐位不变） */
    net.hp.struct_mode     = cfg.struct_mode;
    net.hp.struct_fmin     = cfg.struct_fmin;
    net.hp.struct_interval = cfg.struct_interval;
    /* v19（D5 修复）：候选打分改用 rate[] 时，必须保证速率迹已分配；
     * 用户没显式给 --rate-tau 时按 20 tick 开（EMA 到 ~63% 需 20 tick）。 */
    net.hp.struct_score    = cfg.struct_score;
    if (cfg.struct_score > 0 && cfg.rate_tau <= 0.0) {
        net2_set_rate_trace(&net, 20.0);
        if (cfg.readout_values == 1) net2_set_ext_rate(&net, 1);
    }
    net.hp.trace_clamp = cfg.trace_clamp;
    net.hp.tau_trace = cfg.tau_trace;
    net.hp.lambda_trace = exp(-1.0 / cfg.tau_trace);
    net.hp.decode_mode = cfg.decode_mode;
    net.hp.wta_k = cfg.wta_k;
    net.hp.use_reward_baseline = cfg.use_reward_baseline;
    /* v16：导入权重（必须在任何 tick 之前；weight_scale 仍会在其后生效） */
    if (!cfg.load_weights.empty()) {
        if (net2_import_weights(&net, cfg.load_weights.c_str()) != 0) {
            fprintf(stderr, "Cannot load weights from %s (bad format or size mismatch: need %d neurons / %d synapses)\n",
                    cfg.load_weights.c_str(), net.neuron_count, net.synapse_count);
            net2_destroy(&net);
            return 1;
        }
        printf("Weights: loaded from %s (%d synapses)\n", cfg.load_weights.c_str(), net.synapse_count);
    }
    /* v15（M3）：整体权重缩放，必须早于任何 tick；默认 1.0 时完全不调用 */
    if (cfg.weight_scale != 1.0) {
        net2_scale_weights(&net, cfg.weight_scale);
    }
    /* v16：初始权重导出（在 load / weight_scale 之后，任何 tick 之前） */
    if (!cfg.save_init_weights.empty()) {
        if (net2_export_weights(&net, cfg.save_init_weights.c_str()) == 0)
            printf("Weights: init saved to %s (%d synapses)\n",
                   cfg.save_init_weights.c_str(), net.synapse_count);
        else
            fprintf(stderr, "Cannot save init weights to %s\n", cfg.save_init_weights.c_str());
    }
    printf("Mode: preset=%s psp=%.2f wta_k=%d decode=%d reward=%d baseline=%d tau=%.0f clamp=%.2f"
           " hb=%.2f mb=%.2f lrs=%.3f ws=%.4f ep=%d ep_scope=%d ep_lr=%.5f\n",
           cfg.preset.c_str(), cfg.psp_factor, cfg.wta_k, cfg.decode_mode,
           cfg.reward_mode, cfg.use_reward_baseline, cfg.tau_trace, cfg.trace_clamp,
           cfg.hidden_base, cfg.motor_base, cfg.lr_scale, cfg.weight_scale,
           cfg.eprop, cfg.eprop_scope, cfg.eprop_lr);

    /* v6：三因子调制通道与运动层异质化。走 set_param 以便复用边界校验，
     * 并让 hetero_* 立即重建运动层的异质分布。默认值全 0 = 旧版行为。 */
    struct V6Param { const char* name; double v; };
    const V6Param v6[] = {
        {"error_gain", cfg.error_gain}, {"novelty_gain", cfg.novelty_gain},
        {"cf_lambda", cfg.cf_lambda},  {"hetero_tau", cfg.hetero_tau},
        {"hetero_thresh", cfg.hetero_thresh},         {"hetero_tune", cfg.hetero_tune},
        {"prop_mode", (double)cfg.prop_mode},
        {"error_scope", (double)cfg.error_scope},
        {"error_scope_gain", cfg.error_scope_gain},
    };
    for (const auto& p : v6) {
        if (net2_set_param(&net, p.name, p.v) != 0) {
            fprintf(stderr, "Invalid v6 param %s = %g\n", p.name, p.v);
            net2_destroy(&net);
            return 1;
        }
    }
    if (cfg.error_gain > 0.0 || cfg.novelty_gain > 0.0 ||
        cfg.hetero_tau > 0.0 || cfg.hetero_thresh > 0.0 || cfg.hetero_tune > 0.0) {
        printf("v6: error_gain=%.3f novelty_gain=%.3f cf_lambda=%.2f "
               "hetero(tau=%.3f thresh=%.3f tune=%.3f)\n",
               cfg.error_gain, cfg.novelty_gain, cfg.cf_lambda,
               cfg.hetero_tau, cfg.hetero_thresh, cfg.hetero_tune);
    }

    /* v20（B2-workpoint S2）：H→H 谱半径归一。默认 0 = 不调用，旧路径逐位不变。 */
    if (cfg.rho_target > 0.0) {
        double rho_meas = 0.0;
        if (net2_normalize_rho(&net, cfg.rho_target, &rho_meas) == 0) {
            printf("v20: rho target=%.3f measured=%.4f S=%d\n",
                   cfg.rho_target, rho_meas, net.synapse_count);
        } else {
            fprintf(stderr, "v20: rho normalize failed (target=%.3f)\n", cfg.rho_target);
        }
    }
    if (cfg.error_scope > 0) {
        printf("v13: error_scope=hidden error_scope_gain=%.3f (隐藏层输入突触接受误差塑形)\n",
               cfg.error_scope_gain);
    }
    if (cfg.speed_mode) {
        printf("Speed: ON cost=%.4f radius=%.4f bonus=%.2f speed_bonus=%.2f\n",
               cfg.step_cost, cfg.arrive_radius, cfg.arrive_bonus, cfg.speed_bonus);
    }
    
    /* 运行时消融（与 B1 一致；编译期宏仍决定默认值） */
    if (force_no_trace)  net.use_eligibility_trace = 0;
    if (force_no_reward) net.use_reward_modulation = 0;
    printf("El Trace: %d, Reward Mod: %d\n",
           net.use_eligibility_trace, net.use_reward_modulation);
    printf("Net: %d neurons, %d synapses\n", net.neuron_count, net.synapse_count);

    /* v8（T2 上界实验）：外部线性读出 / 链路对照 / hidden 发放导出 */
    if (!cfg.load_readout.empty()) {
        const int src_cnt = (cfg.readout_layer == 2) ? (net.input_count + net.hidden_count)
                          : (cfg.readout_layer == 1) ? net.input_count : net.hidden_count;
        size_t want = (size_t)src_cnt * 2;
        FILE* rf = fopen(cfg.load_readout.c_str(), "rb");
        if (!rf) { fprintf(stderr, "Cannot open %s\n", cfg.load_readout.c_str()); net2_destroy(&net); return 1; }
        double* beta = (double*)malloc(want * sizeof(double));
        size_t got = beta ? fread(beta, sizeof(double), want, rf) : 0;
        fclose(rf);
        if (got != want) {
            fprintf(stderr, "readout size mismatch: got %zu want %zu\n", got, want);
            free(beta); net2_destroy(&net); return 1;
        }
        net2_set_ext_readout_src(&net, beta, cfg.readout_layer);
        free(beta);
        net.use_reward_modulation = 0;
        printf("Ext readout: %zu coeffs loaded (layer=%s), learning OFF\n", want,
               (cfg.readout_layer == 2) ? "both" : (cfg.readout_layer == 1) ? "input" : "hidden");
    }
    if (cfg.drive_direct) {
        net2_set_drive_mode(&net, 1);
        net.use_reward_modulation = 0;
        printf("Drive: direct omega* (link check), learning OFF\n");
    }
    FILE* dh = NULL;
    const int dump_base  = (cfg.dump_layer == 1) ? 0 : net.input_count;
    const int dump_width = (cfg.dump_layer == 1) ? net.input_count : net.hidden_count;
    if (!cfg.dump_hidden.empty()) {
        dh = fopen(cfg.dump_hidden.c_str(), "wb");
        if (!dh) { fprintf(stderr, "Cannot open %s\n", cfg.dump_hidden.c_str()); net2_destroy(&net); return 1; }
        int hdr[3] = { 0x42325432, dump_width, cfg.stride };
        fwrite(hdr, sizeof(int), 3, dh);
    }
    /* v16：逐 tick 命令导出（JSONL），默认关闭 */
    FILE* tc = NULL;
    if (!cfg.trace_cmd.empty()) {
        tc = fopen(cfg.trace_cmd.c_str(), "w");
        if (!tc) { fprintf(stderr, "Cannot open %s\n", cfg.trace_cmd.c_str()); net2_destroy(&net); return 1; }
        fprintf(tc, "{\"header\":\"b2_cmd_trace\",\"stride\":%d}\n", cfg.trace_stride);
    }
    
    FILE* fp = fopen(cfg.out_csv.c_str(), "w");
    if (!fp) { net2_destroy(&net); fprintf(stderr, "Cannot open %s\n", cfg.out_csv.c_str()); return 1; }
    /* 原 7 列保持不变，末尾追加耗时/到达信息，旧脚本按前 7 列读仍兼容 */
    fprintf(fp, "episode,target_x,target_y,init_dist,final_dist,dist_ratio,final_reward,"
                "ticks_used,arrived,arrival_tick,is_eval\n");
    
    /* 元数据 sidecar：超参/构建信息随输出一起落盘，保证可复现 */
    std::string meta_path = cfg.out_csv + ".meta.json";
    FILE* mf = fopen(meta_path.c_str(), "w");
    if (mf) {
        std::string out_j = cfg.out_csv;
        for (char& ch : out_j) if (ch == '\\') ch = '/';
        /* D6（2026-11-18 记录，2026-11-19 修复）：路径字段须与 out_csv 同样转义，
         * 否则 Windows 反斜杠（E:\...）在 JSON 字符串中构成非法转义，json.load 失败。
         * 默认空串时输出与修复前逐位一致。 */
        std::string sw_j = cfg.save_weights, lw_j = cfg.load_weights, tc_j = cfg.trace_cmd;
        for (char& ch : sw_j) if (ch == '\\') ch = '/';
        for (char& ch : lw_j) if (ch == '\\') ch = '/';
        for (char& ch : tc_j) if (ch == '\\') ch = '/';
        fprintf(mf, "{\n");
        fprintf(mf, "  \"benchmark\": 2,\n");
        fprintf(mf, "  \"build\": \"%s %s\",\n", __DATE__, __TIME__);
        fprintf(mf, "  \"out_csv\": \"%s\",\n", out_j.c_str());
        fprintf(mf, "  \"input\": %d, \"hidden\": %d, \"motor1\": %d, \"motor2\": %d,\n",
                cfg.input_n, cfg.hidden_n, cfg.motor1_n, cfg.motor2_n);
        fprintf(mf, "  \"input_count\": %d, \"dump_layer\": \"%s\", \"readout_layer\": \"%s\",\n",
                net.input_count, (cfg.dump_layer == 1) ? "input" : "hidden",
                (cfg.readout_layer == 2) ? "both" : (cfg.readout_layer == 1) ? "input" : "hidden");
        fprintf(mf, "  \"retarget\": %d, \"stride\": %d,\n", cfg.retarget, cfg.stride);
        fprintf(mf, "  \"direct_fan\": %d, \"hm_scale\": %.4f, \"im_scale\": %.4f,\n",
                cfg.direct_fan, cfg.hm_scale, cfg.im_scale);
        fprintf(mf, "  \"rate_tau\": %.2f, \"dump_values\": \"%s\", \"readout_values\": \"%s\",\n",
                cfg.rate_tau, cfg.dump_values ? "rate" : "spike",
                (cfg.readout_values && cfg.rate_tau > 0.0) ? "rate" : "spike");
        fprintf(mf, "  \"avg_conn\": %.4f, \"seed\": %u,\n", cfg.avg_conn, cfg.seed);
        fprintf(mf, "  \"episodes\": %d, \"score_horizon\": %d, \"max_ticks_per_ep\": %d,\n",
                cfg.episodes, cfg.score_horizon, cfg.max_ticks_per_ep);
        fprintf(mf, "  \"target_mode\": %d, \"fixed_target\": [%.4f, %.4f],\n",
                cfg.target_mode, cfg.fixed_tx, cfg.fixed_ty);
        fprintf(mf, "  \"lr_schedule\": \"0.35(<500) / 0.20(<2000) / 0.04(else)\",\n");
        fprintf(mf, "  \"weight_max\": %.4f, \"psp_factor\": %.4f,\n", WEIGHT_MAX, cfg.psp_factor);
        fprintf(mf, "  \"hidden_base_current\": %.4f, \"motor_base_current\": %.4f,\n",
                cfg.hidden_base, cfg.motor_base);
        fprintf(mf, "  \"lr_scale\": %.4f, \"weight_scale\": %.4f,\n",
                cfg.lr_scale, cfg.weight_scale);
        fprintf(mf, "  \"save_weights\": \"%s\", \"load_weights\": \"%s\", \"trace_cmd\": \"%s\", \"trace_stride\": %d,\n",
                sw_j.c_str(), lw_j.c_str(), tc_j.c_str(), cfg.trace_stride);
        fprintf(mf, "  \"eprop\": %d, \"eprop_scope\": %d, \"eprop_alpha\": %.4f, \"eprop_lr\": %.6f,\n",
                cfg.eprop, cfg.eprop_scope, cfg.eprop_alpha, cfg.eprop_lr);
        fprintf(mf, "  \"struct_mode\": %d, \"struct_fmin\": %d, \"struct_interval\": %d, \"struct_added\": %d,\n",
                cfg.struct_mode, cfg.struct_fmin, cfg.struct_interval, net.struct_added);
        fprintf(mf, "  \"struct_score\": %d, \"rate_tau\": %.1f,\n", cfg.struct_score, net.rate_tau);
        fprintf(mf, "  \"rho_target\": %.3f,\n", cfg.rho_target);
        fprintf(mf, "  \"alif_beta\": %.2f, \"alif_tau_a\": %.2f, \"alif_rho_a\": %.6f, \"alif_bpath\": %d,\n",
                cfg.alif_beta, cfg.alif_tau_a,
                cfg.alif_tau_a > 0.0 ? exp(-1.0 / cfg.alif_tau_a) : 0.0, cfg.alif_bpath);
        fprintf(mf, "  \"ei_mode\": %d, \"ei_n\": %d, \"ei_w\": %.2f, \"ei_tau\": %.1f, \"ei_sigma\": %.2f,\n",
                cfg.ei_mode, cfg.ei_n, cfg.ei_w, cfg.ei_tau, cfg.ei_sigma);
        fprintf(mf, "  \"preset\": \"%s\", \"decode_mode\": %d, \"wta_k\": %d, \"reward_mode\": %d,\n",
                cfg.preset.c_str(), cfg.decode_mode, cfg.wta_k, cfg.reward_mode);
        fprintf(mf, "  \"use_reward_baseline\": %d, \"tau_trace\": %.1f, \"trace_clamp\": %.2f,\n",
                cfg.use_reward_baseline, cfg.tau_trace, cfg.trace_clamp);
        fprintf(mf, "  \"error_gain\": %.4f, \"novelty_gain\": %.4f, \"cf_lambda\": %.3f,\n"
                    "  \"hetero_tau\": %.4f, \"hetero_thresh\": %.4f, \"hetero_tune\": %.4f,\n",
                cfg.error_gain, cfg.novelty_gain, cfg.cf_lambda,
                cfg.hetero_tau, cfg.hetero_thresh, cfg.hetero_tune);
        fprintf(mf, "  \"error_scope\": %d, \"error_scope_gain\": %.4f,\n",
                cfg.error_scope, cfg.error_scope_gain);
        fprintf(mf, "  \"el_trace\": %d, \"reward_mod\": %d,\n",
                net.use_eligibility_trace, net.use_reward_modulation);
        fprintf(mf, "  \"speed_mode\": %d, \"step_cost\": %.4f, \"arrive_radius\": %.4f,\n",
                cfg.speed_mode, cfg.step_cost, cfg.arrive_radius);
        fprintf(mf, "  \"arrive_bonus\": %.2f, \"speed_bonus\": %.2f,\n",
                cfg.arrive_bonus, cfg.speed_bonus);
        fprintf(mf, "  \"neurons\": %d, \"synapses\": %d\n", net.neuron_count, net.synapse_count);
        fprintf(mf, "}\n");
        fclose(mf);
    }
    
    /* 存储所有最终距离用于中位数统计 */
    std::vector<double> final_dists;
    std::vector<double> init_dists;
    std::vector<int> used_ticks;
    std::vector<int> arrived_flags;
    std::vector<int> arrival_ticks;
    
    const int rom_base = net.use_reward_modulation;
    /* v21（B2-ALIF Gate 0）：hidden+motor 每 tick 平均发放数累计（只写 meta/stdout，
     * 不改 CSV 列）。beta=0 时同样统计，供同口径对比。 */
    long long gate0_spk_sum = 0, gate0_tick_sum = 0;
    for (int ep = 0; ep < cfg.episodes; ep++) {
        double tx, ty;
        gen_target(cfg, ep, &tx, &ty);
        net2_set_target(&net, tx, ty);

        /* v7：最后 eval_episodes 轮冻结学习（权重与迹不再更新），用于 g=0 能力评估。
         * 消融 --disable_reward 仍然全局生效。 */
        const bool is_eval = (cfg.eval_episodes > 0) &&
                             (ep >= cfg.episodes - cfg.eval_episodes);
        net.use_reward_modulation = (rom_base == 0 || is_eval) ? 0 : 1;
        
        /* 重置关节到初始位置 + EMA 平滑状态（含角速度/奖励/最大伸展） */
        net.arm.theta1 = 0.0;
        net.arm.theta2 = 0.0;
        net.arm.omega1 = 0.0;
        net.arm.omega2 = 0.0;
        net.arm.end_x = ARM_L1 + ARM_L2;
        net.arm.end_y = 0.0;
        net.arm.max_reach = ARM_L1 + ARM_L2;
        net.arm.reward = 1.0;
        net2_reset_ema();
        
        double init_dist = sqrt((net.arm.end_x - tx)*(net.arm.end_x - tx) +
                                (net.arm.end_y - ty)*(net.arm.end_y - ty));
        if (init_dist < 1e-6) {
            fprintf(stderr, "[WARN] episode %d: init_dist=%.6g, dist_ratio will be unstable\n",
                    ep, init_dist);
        }
        init_dists.push_back(init_dist);
        
        /* lr 阶梯：延长高 lr 到 Ep 2000，去掉中间档
         * v10 分析: Ep 600-999 lr=0.20 时 best=0.46, Ep 1000 lr 降到 0.08 反弹到 0.60
         * → lr=0.20 是稳定学习的最优区间，不要降到 0.08！ */
        /* v15（M3）：lr_scale 默认 1.0 ⇒ 三个阶梯值逐位不变 */
        if (ep < 500)       net.hp.learning_rate = 0.35 * cfg.lr_scale;
        else if (ep < 2000) net.hp.learning_rate = 0.20 * cfg.lr_scale;
        else                net.hp.learning_rate = 0.04 * cfg.lr_scale;
        
        /* 重置神经元膜电位（除了输入层被目标激活） */
        for (int i = 0; i < net.neuron_count; i++) {
            if (net.neurons[i].type != NT_INPUT_X) {
                net.neurons[i].membrane_potential = MEMBRANE_REST;
                net.neurons[i].is_spiking = 0;
                net.neurons[i].refractory_remaining = 0;
                net.neurons[i].last_spike_tick = -1;
            }
        }
        /* episode 重置：trace 保留（承重墙）；baseline 保留；queue 清零 */
        net.queue_head = 0;
        net.queue_tail = 0;
        
        /* combo1 reward: 0.5*cosine + 0.5*delta(gain=100) — 旧版最好配置 */
        double prev_ex = net.arm.end_x, prev_ey = net.arm.end_y;
        bool arrived = false;          /* 首次到达后不再重复发 bonus，避免刷分 */
        int arrival_tick = -1;         /* -1 = 本轮未到达 */
        int ticks_used = 0;
        /* 到达是唯一的正常结束条件；安全阀只用于避免永远到不了目标。
         * v7：--budget N 时改为固定预算（§0.3），到达也不提前结束。 */
        int ep_limit = (cfg.budget > 0) ? cfg.budget
                                       : (cfg.speed_mode ? cfg.max_ticks_per_ep : cfg.score_horizon);
        long long spk_t0 = (long long)net2_get_param(&net, "spike_count");
        for (int t = 0; ep_limit <= 0 || t < ep_limit; t++) {
            /* v10（T2e 采集）：每 retarget tick 换一个新随机目标，避免手臂到点后长期停住
             * 造成「绝大多数样本 ω*≈0」的退化分布。仅影响采集，不影响评估口径。 */
            if (cfg.retarget > 0 && t > 0 && (t % cfg.retarget == 0)) {
                double ntx = tx, nty = ty;
                gen_target(cfg, ep * 100000 + t, &ntx, &nty);
                tx = ntx; ty = nty;
                net2_set_target(&net, tx, ty);
            }
            net2_tick(&net);
            
            double cur_ex = net.arm.end_x, cur_ey = net.arm.end_y;
            double vx = cur_ex - prev_ex, vy = cur_ey - prev_ey;
            double tx = net.arm.target_x - cur_ex, ty = net.arm.target_y - cur_ey;
            
            double vmag = sqrt(vx*vx + vy*vy);
            double tmag = sqrt(tx*tx + ty*ty);
            
            /* cosine alignment */
            double cos_r = 0.0;
            if (vmag > 1e-6 && tmag > 1e-6) {
                cos_r = (vx*tx + vy*ty) / (vmag * tmag);
            }
            
            /* combo1: 0.5*cosine + 0.5*delta(gain=100) */
            double prev_d = sqrt((net.arm.target_x - prev_ex)*(net.arm.target_x - prev_ex) +
                                 (net.arm.target_y - prev_ey)*(net.arm.target_y - prev_ey));
            double cur_d = tmag;
            double delta_imp = (prev_d - cur_d) * 100.0;
            if (delta_imp > 1.0) delta_imp = 1.0;
            if (delta_imp < -1.0) delta_imp = -1.0;
            
            double r;
            if (cfg.reward_mode == 1) {
                /* 势能奖励：越靠近目标越强；配合 --no-reward-baseline 保留长期梯度 */
                double denom = init_dist > 1e-6 ? init_dist : 1e-6;
                r = 1.0 - tmag / denom;
            } else {
                r = 0.5 * cos_r + 0.5 * delta_imp;
            }
            /* 时间/步数维度：每步成本 + 首次到达奖励（越快到达，剩余步数越多，奖励越高） */
            bool just_arrived = false;
            if (cfg.speed_mode) {
                r -= cfg.step_cost;
                if (!arrived && tmag <= cfg.arrive_radius) {
                    arrived = true;
                    just_arrived = true;
                    arrival_tick = t;
                    double remain = (double)(cfg.score_horizon - 1 - t) /
                                    (double)cfg.score_horizon;
                    if (remain < 0.0) remain = 0.0;   /* 超过参考步数只是速度项为 0 */
                    r += cfg.arrive_bonus + cfg.speed_bonus * remain;
                }
            }
            net2_inject_reward(&net, r, 0);
            ticks_used = t + 1;

            /* v8：hidden 发放导出（T2 上界实验的数据采集）；v9：可切 input 层 */
            if (dh && (t % cfg.stride == 0)) {
                unsigned char bit;
                for (int hi = 0; hi < dump_width; hi++) {
                    if (cfg.dump_values == 1) {
                        double rv = (net.rate ? net.rate[dump_base + hi] : 0.0);
                        int q = (int)(rv * 255.0 + 0.5);
                        if (q < 0) q = 0;
                        if (q > 255) q = 255;
                        bit = (unsigned char)q;
                    } else {
                        bit = (unsigned char)(net.neurons[dump_base + hi].is_spiking ? 1 : 0);
                    }
                    fwrite(&bit, 1, 1, dh);
                }
                double meta[8] = { (double)t, net.cf_omega1_star, net.cf_omega2_star,
                                   net.arm.theta1, net.arm.theta2, net.arm.end_x,
                                   net.arm.end_y, (double)ep };
                fwrite(meta, sizeof(double), 8, dh);
            }
            /* v16：命令轨迹（解码指令 vs 解析期望指令 vs 攀爬纤维误差） */
            if (tc && (t % cfg.trace_stride == 0)) {
                fprintf(tc, "{\"ep\":%d,\"t\":%d,\"tx\":%.4f,\"ty\":%.4f,\"ex\":%.4f,\"ey\":%.4f,"
                            "\"th1\":%.4f,\"th2\":%.4f,\"om1\":%.4f,\"om2\":%.4f,"
                            "\"w1s\":%.4f,\"w2s\":%.4f,\"e1\":%.4f,\"e2\":%.4f}\n",
                        ep, t, net.arm.target_x, net.arm.target_y, net.arm.end_x, net.arm.end_y,
                        net.arm.theta1, net.arm.theta2, net.decoded_omega1, net.decoded_omega2,
                        net.cf_omega1_star, net.cf_omega2_star, net.cf_err1, net.cf_err2);
            }

            /* 碰到目标即结束本轮：先多跑 1 tick 消费刚注入的 bonus（reward 队列
             * 在下一 tick 开头才处理），否则下个 episode 开头 queue 清零会丢掉奖励。 */
            if (cfg.speed_mode && just_arrived && cfg.budget <= 0) {
                net2_tick(&net);   /* 先消费刚注入的奖励，再进入下一轮 */
                break;
            }
            
            prev_ex = cur_ex; prev_ey = cur_ey;
        }
        
        /* v21（Gate 0）：统计本 episode 的 hidden+motor 每 tick 平均发放数 */
        long long spk_t1 = (long long)net2_get_param(&net, "spike_count");
        gate0_spk_sum += (spk_t1 - spk_t0);
        gate0_tick_sum += ticks_used;
        double final_dist = sqrt((net.arm.end_x - tx)*(net.arm.end_x - tx) +
                                 (net.arm.end_y - ty)*(net.arm.end_y - ty));
        final_dists.push_back(final_dist);
        used_ticks.push_back(ticks_used);
        arrived_flags.push_back(arrived ? 1 : 0);
        arrival_ticks.push_back(arrival_tick);
        double safe_init = init_dist > 1e-6 ? init_dist : 1e-6;
        double ratio = final_dist / safe_init;
        
        if (fprintf(fp, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n",
                    ep, tx, ty, init_dist, final_dist, ratio, net.arm.reward,
                    ticks_used, arrived ? 1 : 0, arrival_tick, is_eval ? 1 : 0) < 0) {
            fprintf(stderr, "Failed to write %s\n", cfg.out_csv.c_str());
            fclose(fp);
            net2_destroy(&net);
            return 1;
        }
        fflush(fp);  /* 进程被中断时也尽量保住已完成的 episode */
        
        printf("  Ep %2d | target=(%.2f,%.2f) | init=%.3f final=%.3f ratio=%.3f reward=%.3f"
               " | ticks=%d arrived=%d | th1=%.3f th2=%.3f\n",
               ep, tx, ty, init_dist, final_dist, ratio, net.arm.reward,
               ticks_used, arrived ? 1 : 0, net.arm.theta1, net.arm.theta2);
        
        /* v18（B2-struct）：episode 边界执行结构轮。内部早退：未到 struct_interval 的倍数时
         * 只做统计窗口重置，开销≈0；struct_mode=0 时这段完全不进。 */
        if (net.hp.struct_mode > 0) {
            int s4[4] = {0, net.synapse_count, 0, 0};
            int added = net2_struct_round(&net, ep + 1, s4);
            if (added < 0) {
                fprintf(stderr, "[struct] round failed at ep %d\n", ep + 1);
            } else if (added > 0) {
                printf("[struct] after ep %d: round #%d +%d edges -> S=%d | fanin min=%d med=%d\n",
                       ep + 1, net.struct_rounds, added, s4[1], s4[2], s4[3]);
                fflush(stdout);
            }
        }

        /* 诊断：每 50ep 打印 INPUT→MOTOR 权重统计 */
        if (ep % 50 == 49) {
            int m1b = net.input_count + net.hidden_count;
            double wsum=0, wmin=1e9, wmax=-1e9; int n=0, sat_hi=0, sat_lo=0;
            for (int si = 0; si < net.synapse_count; si++) {
                int po = net.synapses[si].post_id;
                int pr = net.synapses[si].pre_id;
                if (pr < net.input_count && po >= m1b) {  /* INPUT→MOTOR 直连 */
                    double w = net.synapses[si].weight;
                    wsum += w; if (w<wmin) wmin=w; if (w>wmax) wmax=w; n++;
                    if (w > 4.9) sat_hi++;
                    if (w < 0.01) sat_lo++;
                }
            }
            printf("    [wstats] I->M n=%d mean=%.3f min=%.3f max=%.3f satHI=%d satLO=%d\n",
                   n, n?wsum/n:0.0, n?wmin:0.0, n?wmax:0.0, sat_hi, sat_lo);
        }
    }
    
    fclose(fp);
    if (dh) fclose(dh);
    if (tc) fclose(tc);
    /* v16：训练结束导出权重（供 wdump_stats.py 统计 I→H / H→H / H→M / I→M 分布） */
    if (!cfg.save_weights.empty()) {
        if (net2_export_weights(&net, cfg.save_weights.c_str()) == 0)
            printf("Weights: saved to %s (%d synapses)\n", cfg.save_weights.c_str(), net.synapse_count);
        else
            fprintf(stderr, "Cannot save weights to %s\n", cfg.save_weights.c_str());
    }
    
    /* 统计 */
    std::vector<double> ratios;
    for (size_t i = 0; i < final_dists.size(); i++) {
        double d = init_dists[i] > 1e-6 ? init_dists[i] : 1e-6;
        ratios.push_back(final_dists[i] / d);
    }
    std::sort(ratios.begin(), ratios.end());
    size_t mid = ratios.size() / 2;
    double median_ratio = (ratios.size() % 2) ? ratios[mid]
                                              : (ratios[mid - 1] + ratios[mid]) / 2.0;
    double mean_ratio = 0;
    for (auto r : ratios) mean_ratio += r;
    mean_ratio /= (double)ratios.size();
    double min_ratio = ratios.front();
    double max_ratio = ratios.back();
    
    /* v21（B2-ALIF Gate 0）：放电率操纵有效性，写 sidecar（不改 CSV 列）。
     * Gate 0 条件：beta=4 的每 tick 平均发放数比 beta=0 下降 ≥10%。 */
    double gate0_rate = (gate0_tick_sum > 0)
                        ? (double)gate0_spk_sum / (double)gate0_tick_sum : 0.0;
    printf("\n=== ALIF GATE 0 (firing-rate manipulation check) ===\n");
    printf("  beta=%.2f tau_a=%.1f rho_a=%.6f | hidden+motor spikes/tick=%.4f (n_spk=%lld n_tick=%lld)\n",
           cfg.alif_beta, cfg.alif_tau_a,
           cfg.alif_tau_a > 0.0 ? exp(-1.0 / cfg.alif_tau_a) : 0.0,
           gate0_rate, gate0_spk_sum, gate0_tick_sum);
    {
        std::string g0 = cfg.out_csv + ".gate0.json";
        FILE* gf = fopen(g0.c_str(), "w");
        if (gf) {
            std::string gj = g0;
            for (char& ch : gj) if (ch == '\\') ch = '/';
            fprintf(gf, "{\n  \"gate0_csv\": \"%s\",\n", gj.c_str());
            fprintf(gf, "  \"alif_beta\": %.2f, \"alif_tau_a\": %.2f, \"alif_rho_a\": %.6f, \"alif_bpath\": %d,\n",
                    cfg.alif_beta, cfg.alif_tau_a,
                    cfg.alif_tau_a > 0.0 ? exp(-1.0 / cfg.alif_tau_a) : 0.0, cfg.alif_bpath);
            fprintf(gf, "  \"spikes\": %lld, \"ticks\": %lld, \"spikes_per_tick\": %.6f,\n",
                    gate0_spk_sum, gate0_tick_sum, gate0_rate);
            fprintf(gf, "  \"med_ratio\": %.4f, \"mean_ratio\": %.4f,\n",
                    median_ratio, mean_ratio);
            fprintf(gf, "  \"episodes\": %d, \"seed\": %u\n}\n", cfg.episodes, cfg.seed);
            fclose(gf);
        }
    }

    printf("\n=== FINAL RESULTS ===\n");
    printf("Ratio (final/initial distance):\n");
    printf("  Median: %.4f\n", median_ratio);
    printf("  Mean:   %.4f\n", mean_ratio);
    printf("  Min:    %.4f\n", min_ratio);
    printf("  Max:    %.4f\n", max_ratio);
    printf("  Target: ≤ 0.20\n");
    printf("\n=== PASS CHECK ===\n");
    if (median_ratio <= 0.20) {
        printf("[PASS] Median ratio %.4f <= 0.20\n", median_ratio);
    } else {
        printf("[FAIL] Median ratio %.4f > 0.20\n", median_ratio);
    }

    if (cfg.speed_mode) {
        int n_arrived = 0, sum_used = 0, sum_arrival = 0;
        for (size_t i = 0; i < used_ticks.size(); i++) {
            sum_used += used_ticks[i];
            if (arrived_flags[i]) { n_arrived++; sum_arrival += arrival_ticks[i]; }
        }
        printf("\n=== SPEED / STEP STATS ===\n");
        printf("  Arrived:    %d/%d (%.1f%%)\n", n_arrived, cfg.episodes,
               100.0 * n_arrived / cfg.episodes);
        printf("  Mean ticks: %.1f\n", (double)sum_used / cfg.episodes);
        if (n_arrived) {
            printf("  Mean arrival tick: %.1f (smaller = faster)\n",
                   (double)sum_arrival / n_arrived);
        }
    }
    
    net2_destroy(&net);
    return (median_ratio <= 0.20) ? 0 : 2;
}
