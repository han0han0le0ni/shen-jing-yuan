# 项目长期笔记 · shen_jing_yuan

## 结构速查
- `Experiment/vis/neuron_vis.py` — 神经网络实时可视化桌面程序（v5）。这是「看神经元」的入口，不是 exe。
- `Experiment/b2_core.dll` — Benchmark-2 的 C 核心动态库（x64），由 `brain_b2/benchmark2_core.c` 编译。导出 `net2_get_topology` / `net2_get_snapshot` flat buffer 接口（零文件 IO），`net2_init` 是 **7 参数**（含 `double avg_conn`），漏传会导致连接度钳在 K=3、种子不可控。
- `Experiment/vis/backend_cpu.py` — C DLL 后端；`backend_vec.py` — numpy/CuPy 向量化后端（**v7 闭环 + 目标引导版**，5 个融合 CUDA kernel，与 C 核心动力学对齐）；`backend_gpu.py` — 早期 CuPy 实验后端。主程序用 QThread Pull 模式，工作线程跑 tick，主线程 60 FPS 拉最新帧渲染。
- `Experiment/brain/` + `host/` — Benchmark-1；`brain_b2/` + `host_b2/` — Benchmark-2。`data/`、`data_b2/` 存实验 CSV。
- `Experiment/vis/test_ctypes.py` — DLL 接口冒烟脚本。

## 技术约定
- 运行环境：Windows / Python 3.14.3 (x64) / PySide6 6.12.0 / numpy 2.5.4 / CuPy 14.2.0 + `cupy-cuda12x[ctk]`（CUDA 12.9 headers，sm_120 GPU 必需）。
- 硬件：Ryzen 9 7845HX 12C/24T、31.2 GB、2560×1600、RTX 5060 Laptop（sm_120 / 8 GB）。
- 可视化偏好：中文界面、深色为主并提供浅色切换、卡片式布局；图表配色遵循「膜电位蓝→红」「发放脉冲金色」。
- 验证方式：`QT_QPA_PLATFORM=offscreen` 启动后 `w.grab().save(...)` 截图检查，无需打开真实窗口。

## v5 现状（2026-11-11 复核）
- 默认后端 `DEFAULT_DEVICE_INDEX = 2`（C DLL，参数最全、与 headless 训练同源）；CUDA / numpy / C DLL 三种设备可运行时切换。
- 主界面含：在线训练（逐 tick R-STDP、碰到目标结算重来）、8 个指标卡、3 条实时趋势曲线、4 档网络规模、可停靠「训练与参数」面板（25 个参数，含 v7 的目标引导强度 / 末端减速距离）。
- **设置面板默认隐藏**（`dock.hide()`），按 P 或工具栏「设置面板」展开；展开时会挤占视图，故默认收起。
- 窗口按 `QGuiApplication.primaryScreen().availableGeometry()` 自适应：本机 1707×1019 可用区 → 1679×971，真实窗口已验证放得下。offscreen 下会被限制为 1280×760，出预览图需显式 `w.resize(1679, 971)`。
- 主 Splitter 用 `setStretchFactor(3,2)` **还不够**，必须同时 `main.setSizes([0.58w, 0.42w])` 并给两侧设 `minimumWidth(340)`，否则子面板 sizeHint 会把比例挤反。
- QSS 必须显式覆盖 `QDockWidget / QTabWidget::pane / QTabBar::tab / QScrollArea / QScrollBar / SpinBox 子按钮`，否则深色主题下设置面板是刺眼的白色块。
- 界面文案避免 `⚙ ▶ ☀` 等装饰符号（无头字体库只有 5 个族时会变方块），统一用纯中文。

## B2 训练核心 v6（2026-11-11）
- `net2_init` 之后所有新通道默认关闭，`net2_set_param` 支持 `error_gain / novelty_gain / cf_lambda / hetero_tau / hetero_thresh / hetero_tune`，host 有同名 CLI 开关。改 C 核心后必须 `build.ps1` 全量重编译（exe + DLL），否则二进制与源码脱节。
- `benchmark2_core.c` / `benchmark2_host.cpp` 曾有**双 UTF-8 BOM** 导致 gcc/g++ 解析失败；如再遇到「vadefs.h #pragma pack 报错」先查文件头 BOM。
- **判据陷阱**：默认的「碰到目标才结算」让 median 轻易达标（50 ep 可到 0.1477），但每轮跑 ~9860 tick，靠随机游走碰运气。做 A/B 必须加 `--max-ticks 1200` 固定步数预算，并同时看到达率/平均到达步数。
- **负结果**：攀爬纤维式误差注入（即使中心化 + PF 门控）在 k-WTA 解码架构下单调劣化 median；缺的是可学习的内部正向模型，不是注入通路。别再把 `error_gain` 当默认优化项。
- 唯一稳定正收益：`novelty_gain=1.0` 在 3 个 seed 上 median 改善 3–7%。运动层异质化 / 扩容近似中性。

## 向量化后端 v6 闭环化（2026-11-11，修「GPU 下机械臂乱走」）
- **根因（旧 v5 的 `backend_vec.py`）**：输入只编码「目标绝对坐标」，没有 C 核心 `target - end` 相对位置 + `PROP_N=56` 本体感觉层，也没有 INPUT→MOTOR 直连 → 网络是**开环**的，永远输出近似固定的方向；解码还用最近 5 tick 历史窗口（混合不同方向的 winner）；CUDA kernel 资格迹硬编码 ±5 而 numpy 路径用 ±1，两条路径训练后会分叉。观感上 25× 速度下每帧只采 1 个轨迹点，折线像瞬移。
- **修复**：输入层改为 `N_pos + 56`（`topo["input"]` 必须含 PROP，GUI 按该字段分区），σ_pos=2.0 / σ_prop=0.6；运动层 `pref_omega ∈ [-0.5,0.5]`；补 INPUT→MOTOR K=12 直连；解码改为「本 tick WTA winner 的 pref_omega 均值 + N(0,0.02) 探索噪声」；k-WTA 用膜电位 top-K（CUDA 严格 `>` + numpy `stable argsort`，tie-break 一致）；资格迹 clamp 参数化并加 `a_plus/a_minus`、I→M 跳过 LTD；加隐层 +2.0 / 运动层 +3.0 基础电流与共享 CPU 噪声池（GPU/numpy 逐位可复现）。所有 arm 状态收敛到 `arm_dev[6]`，`_read_arm()` 按 tick 缓存读回。
- **轨迹渲染**：`SimEngine._trail_buf` 每 tick 收集末端点 → worker 放进 snap 第 5 项 → `ArmPhysicsView.append_trail()` 一帧批量补 600 点；轮次切换插 `None` 断点。`update_arm()` 不再自己追加单点。
- **实测**：CUDA 与 numpy 轨迹逐位一致（400/1500 tick 全指标相同）；1500 tick 路径长度 24.16 → 9.16（C DLL 参照 9.07），包围盒 0.15 → 0.019；纯 tick 吞吐 7216 tick/s（极限 4456 神经元 2816 tick/s），GUI 25× 仍 62 FPS / 2.11 ms/帧。方向余弦三后端都在 +0.01~0.02（未经充分训练，属任务本身难度，不是后端 bug）。
- 备份：`vis/backend_vec_v5备份.py.bak`、`vis/neuron_vis_v5b备份.py.bak`。可调 `explore_noise` / `membrane_noise` / `encode_sigma_*`，默认与 C 端一致；GUI 每 tick `get_arm()` 是 GPU 同步点，在线训练时 tick/s 会从纯仿真 7200 降到约 1700，属正常开销。

## v7 目标引导（2026-11-11，修「关节卡死 / 末端绕圈不靠近目标」）
- **别再误判为 GPU 数值问题**：v6 之后三后端共有的问题是「k-WTA 解码的 ω 是零均值抖动，没有朝目标的直流分量」。诊断特征：末端半径恒定 ≈2.0、θ1/θ2 只在 ±0.2 rad 内摆、dist 不下降、0 次到达。
- 修复三件套：① `ω=(1-g)·ω_net+g·ω_goal`，ω_goal 用 DLS `J^T(JJ^T+0.0625I)^-1·e` 归一化后乘 `0.5·min(1,d/goal_d0)`（近端线性减速 → 渐近收敛，不绕圈）；② θ1 从 clip ±π 改**周期性 wrap**（肩关节 360°，硬裁剪是吸收边界）；③ 随机目标半径 `0.4+rand·1.1`，下限必须 > 2cos(0.45π)≈0.313（肘部限位造成的最短可达半径）。
- **关键约定（已撤销）**：v7 的 DLS 目标引导公式已于 2026-11-11 **从主程序彻底删除**——`compute_goal_omega` 从 C 核心移除，`guidance/goal_d0` 从 C 核心 / host CLI / `backend_vec` / `neuron_vis` 设置面板全部移除并重新编译（`backup_20261111_215545/`）。现在主程序（neuron_vis + 默认 C DLL 后端）**完全不靠公式**，纯脉冲网络自主解码，默认 0 到达（距离几乎不动）。`CPARAM_KEYS` 现仅含 `avg_conn` 等运行时参数。
- 教师公式的去向：其唯一合法用途是 `transfer_lab.py` 的教师信号。该公式已**移入 transfer_lab.py 本地**（`_goal_omega`），通过 `VecBackend.motor_override` 通用钩子注入混合指令 `ω=(1-G)·ω_net+G·ω_goal`；G 由实验台自己调度。主程序不再包含任何 DLS 代码。
- **⚠️ 别把「找路快」当成网络学会了**：ω_goal 是解析 DLS 逆运动学（用精确 target/end/θ/臂长/Jacobian 直接解关节角速度），按「模型替网络做决策」的标准算作弊。量化（`2026-11-11.md` 追加段）：g=0.7 时网络项只占引导项 19.1% 幅度且与所需方向 cos=−0.077、投影 −0.0073 rad/s；g=1.0（网络完全不进执行器）反而 9 次到达 / 206 tick，g=0 是 0 次；固定 g=0.5 连训 12 轮到达步数只从 416→404、cos 始终 ≈0。**衡量网络能力只能用 g=0 的到达率**，绝不要用 g>0 的到达次数。ω_goal 不进奖励、无教师泄漏；但 `cf_err=w1s−omega1_cmd` 用混合后指令，打开 `error_gain` 会引入解析教师信号（默认 0）。
- 实测：2500 tick 0→8 次到达；10/10 方向目标可达（θ1 跨度最大 3.41 rad，wrap 生效）；headless 5 ep g=0.7 → 5/5 到达、final_dist 1.27→0.146、平均 295 tick；GUI 5000 tick 到达 12–14 次，62.5 FPS。A/B 数据在 `data_b2/v7_headless_{baseline,guided}.csv`。
- 改 C 核心后照例 `build.ps1 -Backup` 全量重编译；改 `backend_vec.py` 的 CUDA kernel 后必须验证 cpu/gpu 轨迹仍不分叉（当前 max|Δ|≈1e-8，仅浮点 1ulp）。

## 引导转移实验台（2026-11-11，独立可删）
- `Experiment/vis/transfer_lab.py`：**独立单文件**，不写任何文件、不改主程序，删掉它即恢复原状。`py -3 vis/transfer_lab.py` 开窗口，`--check` 跑 headless 自检。用户要「能看到 G 值 + 用完就删」。
  **注意**：主程序删公式后，教师 DLS 公式已内联到本文件（`_goal_omega`），经 `be.motor_override` 注入；若改 `backend_vec.py` 的 `motor_override` 钩子需同步本文件。
- 显示实时 G、G 退火曲线、**g=0 探针到达率/方向余弦/速度比**、臂轨迹；一键「强制纯自主 G=0」。验收只看 g=0 探针（3 固定目标×800 tick），「已自主」= G≤0.02 且连续 3 次 ≥2/3。
- **别再重试纯 R-STDP 教师或 `pref_omega` 可塑**：前者无状态门控学不动，后者幅度塌缩（速度比 0.19）。有效的是 **override `_decode_omega` 做可塑线性群体读出**（特征=输入+隐层脉冲，NLMS 回归满幅教师方向）+ DAgger 自主采样轮 + 历史最佳权重回滚。
- 现状：G 0.7→0.24、探针最高 1/3 到达、最小距离 0.145，**20k tick 内仍未可靠自主**，界面如实显示「训练中」。本质是监督蒸馏解析控制器，不是网络自主发现。详见 `2026-11-11.md`。

## 历史存档：v4 尺寸与配色（仅作参考，v5 已重做，勿据此修改当前界面）

**运行方式**：`py -3 Experiment/vis/neuron_vis.py`

**当前主窗口与布局**：
- 窗口 `resize(1400, 820)`（屏幕 1707×1067 能完整放下；之前 1680×960 超出）
- 标题栏/工具栏：42px 高，28px 控件，左右 margin 12，spacing 8
- 控件固定宽：运行=68、单步=56、重置=56、Combo 速度=72、SpinBox=72、图标按钮=36
- Splitter 左右比例 3:2，panel 圆角 10、border 1px #1f2636、背景 #0f131c

**网络拓扑视图（NetworkGraph）**：
- scene 宽 `max(1050, margin_x*2 + 40*18)` ≈ 1100px，高动态计算 ≈ 700-800px
- 隐藏层：**40 列 × 50 行**（2000 节点），**grid_cell=18px**；之前 50×40 cell=14 导致节点挤成一团
- 单层 row_h=36px，H_space（层间距）=110px，margin_x=50，y_cursor 起点=50
- 节点半径 **R=5**，glow 外圈 R+3；之前 R=3 太小看不清
- 突触渲染**默认关闭**（v4 为防首次 paintEvent 卡死主线程暂时注释）；之前 16000+ lineTo 合成一条 QPainterPath + Antialiasing 会导致 Qt 渲染卡死 >10s

**物理仿真视图（ArmPhysicsView）**：
- sceneRect(-240, -240, 480, 480)，比例系数 `_s=100.0`
- 最大可达圈亮线、极坐标网格 0.5/1.0/1.5/2.0，等比例虚线圆
- 轨迹 QPainterPath 增量 append，满 400 点 pop+重建

**颜色常量（霓虹科技风）**：
```
LAYER_INPUT_X: 青  QColor(64, 220, 255)
LAYER_HIDDEN:   荧光绿 QColor(140, 255, 80)
LAYER_MOTOR1:   琥珀橙 QColor(255, 170, 60)
LAYER_MOTOR2:   霓虹紫 QColor(220, 90, 255)
背景: #0b0e14, toolbar: #131822, panel: #0f131c
```
膜电位渐变 LUT（256 级，numpy 向量化查表）：深蓝(-65)→青→荧光绿→黄→红(0)

**架构关键：主线程绝不碰 GPU**：
- `BackendWorker(QThread)` Pull 模式：后台快跑 tick → asnumpy → `lock: _latest = snap`
- 主线程 QTimer 60fps `worker.get_latest_snapshot()` 取最新帧做纯渲染
- 之前 Push 模式 Worker 每秒 emit 700+ 信号 → 主线程事件队列被积压 → UI 假死
- 渲染帧频独立于 tick 频率，Worker 跑多快都行（GPU ~700fps、CPU ~40fps）

**已踩过的坑（别再犯）**：
1. Qt Antialiasing 渲染 >1200 条 lineTo 的 QPainterPath 会卡死主线程（用 `addPath` 也要小心长路径）
2. QThread 里不能 emit 高频信号 → 用 lock + 共享单变量 Pull
3. GPU→CPU `asnumpy` 是同步点，必须放在 Worker 线程不能主线程
4. CuPy RawKernel 首次编译 0.3s，初始化阶段就会触发，别指望"无 GPU 才走 CPU"
5. Qt 逻辑屏幕 1707×1067 / 可用区 1707×1019（物理 2560×1600，DPR 1.5）；v5 自适应 1679×971 已实测放得下，1400×820 是 v4 旧值，别再写死
6. CuPy 在 sm_120 上必须装 `cupy-cuda12x[ctk]`，否则报 `Failed to find CUDA headers`；`CUDA_PATH` 未设置只会给警告，不影响 kernel 编译
