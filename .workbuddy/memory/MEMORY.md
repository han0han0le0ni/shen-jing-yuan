# 项目长期笔记 · shen_jing_yuan

## 结构速查
- `Experiment/vis/neuron_vis.py` — 神经网络实时可视化桌面程序（v5）。这是「看神经元」的入口，不是 exe。
- `Experiment/b2_core.dll` — Benchmark-2 的 C 核心动态库（x64），由 `brain_b2/benchmark2_core.c` 编译。导出 `net2_get_topology` / `net2_get_snapshot` flat buffer 接口（零文件 IO），`net2_init` 是 **7 参数**（含 `double avg_conn`），漏传会导致连接度钳在 K=3、种子不可控。
- `Experiment/vis/backend_cpu.py` — C DLL 后端；`backend_vec.py` — numpy/CuPy 向量化后端（含融合 CUDA kernel）；`backend_gpu.py` — 早期 CuPy 实验后端。主程序用 QThread Pull 模式，工作线程跑 tick，主线程 60 FPS 拉最新帧渲染。
- `Experiment/brain/` + `host/` — Benchmark-1；`brain_b2/` + `host_b2/` — Benchmark-2。`data/`、`data_b2/` 存实验 CSV。
- `Experiment/vis/test_ctypes.py` — DLL 接口冒烟脚本。

## 技术约定
- 运行环境：Windows / Python 3.14.3 (x64) / PySide6 6.12.0 / numpy 2.5.4 / CuPy 14.2.0 + `cupy-cuda12x[ctk]`（CUDA 12.9 headers，sm_120 GPU 必需）。
- 硬件：Ryzen 9 7845HX 12C/24T、31.2 GB、2560×1600、RTX 5060 Laptop（sm_120 / 8 GB）。
- 可视化偏好：中文界面、深色为主并提供浅色切换、卡片式布局；图表配色遵循「膜电位蓝→红」「发放脉冲金色」。
- 验证方式：`QT_QPA_PLATFORM=offscreen` 启动后 `w.grab().save(...)` 截图检查，无需打开真实窗口。

## 可视化 GUI 精确尺寸锁定（v4，2026-11-09 最后一版，勿乱改）

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
5. Windows 1707×1067 屏幕，窗口别超过 1400×820
