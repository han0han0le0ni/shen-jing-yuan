# -*- coding: utf-8 -*-
"""
neuron_vis.py — Benchmark-2 脉冲神经网络 · 实时可视化 (v5)

运行：cd Experiment && python vis/neuron_vis.py
快捷键：空格 运行/暂停 · S 单步 · R 重置 · T 随机目标 · F 适应窗口

v5 在 v4（QThread Pull 模式 + numpy/CuPy 向量化后端）基础上重做：
  1. 后端环境补齐：numpy / CuPy 14 / CUDA 12.9 toolkit headers
  2. 修正 net2_init 真实签名（7 参数，含 double avg_conn）
  3. 恢复被调试注释掉的突触连线：预渲染到 QPixmap 批量绘制，
     19882 条线不再是 19882 个图元，也不走抗锯齿长路径
  4. spike 光晕合并为 2 个 QPainterPath 图元（当前帧 + 残影）
  5. 膜电位颜色用 numpy 差分，只改变化的节点；命中检测 numpy 向量化
  6. 实时趋势曲线、指标卡、网络配置、性能仪表、神经元悬停/选中
  7. 中文字体自适应（无头/远程会话自动补载系统字体）
"""

import sys
import os
import math
import random
import time
import threading
from collections import deque
from string import Template

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QFrame, QSplitter, QToolTip, QMessageBox,
    QDockWidget, QTabWidget, QScrollArea, QFormLayout,
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
    QGraphicsPathItem, QGraphicsTextItem,
)
from PySide6.QtCore import Qt, QTimer, QRectF, QThread, Signal, QPointF, QLineF
from PySide6.QtGui import (
    QColor, QPen, QBrush, QPainter, QPainterPath, QFont, QLinearGradient,
    QAction, QKeySequence, QFontDatabase, QGuiApplication, QPixmap,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, APP_DIR)
from backend_cpu import CPUBackend
from backend_vec import VecBackend, L1, L2, MAX_OMEGA

# ============================================================
# 字体
# ============================================================
FONT_FAMILY = "Microsoft YaHei UI"
CJK_CANDIDATES = ("Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑",
                  "SimHei", "Noto Sans CJK SC", "PingFang SC", "Segoe UI")


def ensure_fonts():
    if QFontDatabase.families():
        return
    d = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            QFontDatabase.addApplicationFont(p)


def pick_font_family():
    fams = set(QFontDatabase.families())
    for n in CJK_CANDIDATES:
        if n in fams:
            return n
    return "sans-serif"


def gpu_label():
    try:
        import cupy as cp
        cap = cp.cuda.Device(0).compute_capability
        free, total = cp.cuda.runtime.memGetInfo()
        return f"GPU sm_{cap} · {total / 1024 ** 3:.1f} GB"
    except Exception:
        return "GPU 未启用"


# ============================================================
# 主题
# ============================================================
THEMES = {
    "dark": dict(
        bg="#080c13", panel="#111925", panel_alt="#0d1520", border="#22304a",
        text="#e6eefb", sub="#93a9c6", dim="#60728c",
        accent="#40dcff", accent_hover="#69e6ff", purple="#a78bfa",
        btn="#172132", btn_hover="#20304a",
        ok="#2ec492", warn="#ffb84d", danger="#ff6b6b",
        scene_net="#0a101a", scene_arm="#090e17", chart="#0a101a",
        band_edge="#1b2a40", grid="#16202f", grid_axis="#26374f",
        syn=[(90, 120, 165), (110, 150, 205), (130, 180, 235), (160, 210, 255)],
        active_syn=(255, 205, 90, 190), spike=(255, 225, 130, 225),
        spike_edge=(255, 245, 200, 245), ghost=(255, 200, 90, 70),
        layer=["#40dcff", "#8cff50", "#ffaa3c", "#dc5aff"],
        reach="#2c4f7a", target="#ff7070", target_fill=(255, 112, 112, 55),
        trail=(46, 196, 146, 165), link1="#4aa8ff", link2="#b06bff",
        joint="#c9d8ee", end="#2ec492", end_fill=(46, 196, 146, 55),
        scene_label="#7d90ab", select="#ffd166", node_edge="#2b3c55",
    ),
    "light": dict(
        bg="#f1f5fa", panel="#ffffff", panel_alt="#f5f8fd", border="#dbe4f0",
        text="#1b2738", sub="#556882", dim="#8797ac",
        accent="#0d8fc4", accent_hover="#1aa5e0", purple="#7c3aed",
        btn="#ffffff", btn_hover="#e8f3fb",
        ok="#0f9d6e", warn="#d97706", danger="#dc2626",
        scene_net="#fafbfe", scene_arm="#f8fafc", chart="#fafbfe",
        band_edge="#dfe8f4", grid="#e8eef6", grid_axis="#cbd8e9",
        syn=[(170, 184, 204), (135, 160, 200), (95, 135, 195), (55, 105, 185)],
        active_syn=(225, 155, 15, 195), spike=(240, 180, 40, 225),
        spike_edge=(210, 140, 10, 245), ghost=(235, 175, 45, 60),
        layer=["#0d8fc4", "#3aa85f", "#d97706", "#8b45d6"],
        reach="#a9bde0", target="#e05252", target_fill=(224, 82, 82, 42),
        trail=(16, 150, 110, 145), link1="#2f6fc0", link2="#8b5cf6",
        joint="#64748b", end="#0f9d6e", end_fill=(15, 157, 110, 45),
        scene_label="#64748b", select="#d97706", node_edge="#aebbd0",
    ),
}

QSS = Template("""
QWidget { font-family: "$font","Segoe UI",sans-serif;
          font-size: 12px; color: $text; }
QMainWindow, #Root { background: $bg; }
#Header, #Card, #ControlBar, #ConfigBar { background: $panel;
    border: 1px solid $border; border-radius: 12px; }
#Logo { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 $accent, stop:1 $purple);
        border-radius: 19px; color: #ffffff; font-size: 20px; font-weight: 700; }
#Title { font-size: 17px; font-weight: 600; color: $text; }
#Subtitle { font-size: 11px; color: $sub; }
#Badge { font-size: 12px; font-weight: 600; }
#Chip { background: $panel_alt; border: 1px solid $border; border-radius: 12px;
        color: $sub; font-size: 11px; padding: 5px 11px; }
#Perf { color: $dim; font-size: 11px; font-family: "Consolas",monospace; }
#CardHead { background: $panel_alt; border-bottom: 1px solid $border;
           border-top-left-radius: 11px; border-top-right-radius: 11px; }
#CardTitle { font-size: 13px; font-weight: 600; color: $text; }
#CardHint { font-size: 11px; color: $dim; }
#GroupLabel { color: $dim; font-size: 11px; font-weight: 600; }
#VSep { background: $border; }
#HLine { background: $border; }
#Hint { color: $dim; font-size: 11px; }
#MuteSmall { color: $dim; font-size: 11px; }
#TeacherCard { border-top: 1px solid $border; }
#MetricCard { background: $panel_alt; border: 1px solid $border; border-radius: 9px; }
#MetricLabel { color: $dim; font-size: 11px; }
#MetricValue { color: $text; font-size: 18px; font-weight: 600;
               font-family: "Consolas","$font",monospace; }
QPushButton { background: $btn; color: $text; border: 1px solid $border;
             border-radius: 8px; padding: 7px 13px; font-weight: 600; }
QPushButton:hover { background: $btn_hover; border-color: $accent; }
QPushButton:pressed { background: $accent; color: #ffffff; }
QPushButton:disabled { color: $dim; background: $panel_alt; }
QPushButton#Primary { background: $accent; color: #07121c;
                       border: 1px solid $accent; min-width: 92px; }
QPushButton#Primary:hover { background: $accent_hover; }
QPushButton#Ghost { background: transparent; }
QPushButton#Tiny { padding: 4px 10px; font-size: 11px; }
QDoubleSpinBox, QSpinBox, QComboBox { background: $panel_alt; color: $text;
    border: 1px solid $border; border-radius: 7px; padding: 5px 8px;
    min-width: 72px; selection-background-color: $accent; }
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus { border-color: $accent; }
QDoubleSpinBox::up-button, QSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 15px;
    background: $panel_alt; border-left: 1px solid $border; }
QDoubleSpinBox::down-button, QSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 15px;
    background: $panel_alt; border-left: 1px solid $border; }
QComboBox::drop-down { border: 0px; width: 18px; }
QComboBox QAbstractItemView { background: $panel; color: $text;
    border: 1px solid $border; selection-background-color: $accent; }
QCheckBox { color: $sub; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid $border;
    border-radius: 4px; background: $panel_alt; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; }
QDockWidget { background: $panel; color: $text;
              border: 1px solid $border; border-radius: 10px; }
QDockWidget::title { background: $panel_alt; color: $text;
    padding: 8px 12px; border-bottom: 1px solid $border;
    text-align: left; font-weight: 600; }
QDockWidget::content { background: $panel; }
#SettingsBody, #SettingsPage, #SettingsViewport { background: $panel; }
QTabWidget::pane { background: $panel; border: 1px solid $border;
    border-radius: 8px; top: -1px; }
QTabBar::tab { background: $panel_alt; color: $sub; padding: 6px 12px;
    border: 1px solid $border; border-bottom: none;
    border-top-left-radius: 7px; border-top-right-radius: 7px; }
QTabBar::tab:selected { background: $panel; color: $text; font-weight: 600; }
QTabBar::tab:hover { color: $text; }
QScrollArea { background: $panel; border: none; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: $border; border-radius: 4px;
    min-height: 26px; }
QScrollBar::handle:vertical:hover { background: $accent; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent; height: 0; }
QScrollBar:horizontal { background: transparent; height: 9px; margin: 2px; }
QScrollBar::handle:horizontal { background: $border; border-radius: 4px;
    min-width: 26px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent; width: 0; }
QGraphicsView { border: none; background: transparent; }
QSplitter { background: transparent; }
QSplitter::handle { background: $border; width: 4px; margin: 10px 3px;
                    border-radius: 2px; }
QSplitter::handle:vertical { height: 4px; margin: 3px 10px; }
QToolTip { background: $panel; color: $text; border: 1px solid $border;
           padding: 6px; font-size: 11px; }
QMessageBox { background: $panel; }
""")


def build_qss(T, font=None):
    return QSS.substitute(T, font=font or FONT_FAMILY)


# ============================================================
# 常量 / 预设
# ============================================================
LAYER_NAMES = ["输入层 · 位置 / 本体感觉", "隐藏层 · 脉冲整合",
               "运动层 1 · ω1", "运动层 2 · ω2"]
MEM_REST, MEM_THRESH = -65.0, -50.0
LUT_SIZE = 256

# 膜电位色带：静息深蓝 → 青 → 绿 → 黄 → 阈值红
LUT = np.zeros((LUT_SIZE, 4), dtype=np.uint8)


def _build_lut():
    stops = [(0.00, (40, 60, 200)), (0.25, (60, 200, 240)),
             (0.50, (80, 230, 120)), (0.75, (240, 220, 60)),
             (1.00, (240, 60, 60))]
    for i in range(LUT_SIZE):
        t = i / (LUT_SIZE - 1)
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0)
                LUT[i] = tuple(int(c0[q] + (c1[q] - c0[q]) * k) for q in range(3)) + (255,)
                break


_build_lut()
BRUSHES = [QBrush(QColor(int(LUT[i, 0]), int(LUT[i, 1]), int(LUT[i, 2])))
           for i in range(LUT_SIZE)]

# (名称, input, hidden, motor1, motor2)
PRESETS = [
    ("轻量 · 256", 36, 200, 10, 10),
    ("标准 · 520", 64, 400, 28, 28),
    ("大规模 · 2200", 100, 2000, 50, 50),
    ("极限 · 4400", 144, 4000, 128, 128),
]
DEVICES = [("CUDA · RTX 5060", "cuda"), ("向量化 · 12 核", "numpy"),
           ("C DLL · 原生 · 可训练", "cdll")]
DEFAULT_DEVICE_INDEX = 2   # C DLL：参数最全、与 headless 训练完全同源

# 可调设置：(key, 标签, 类型, 默认值, 范围/选项, 提示)
# 类型 bool / int / float / combo；范围 = (最小, 最大, 步长)
TRAIN_SPECS = [
    ("训练与奖励", [
        ("train_enabled", "启用在线训练", "bool", True, None,
         "关闭后窗口只仿真：不注入奖励、不结算轮次"),
        ("auto_target", "每轮重置随机换目标", "bool", True, None,
         "勾选：每次重置/新一轮随机换位置；取消：沿用当前目标重来"),
        ("arrive_radius", "碰到目标半径", "float", 0.15, (0.01, 2.0, 0.01),
         "末端进入该半径即算碰到目标，结算奖励并结束本轮"),
        ("arrive_bonus", "到达奖励", "float", 1.0, (0.0, 20.0, 0.1),
         "首次碰到目标的一次性奖励"),
        ("speed_bonus", "速度奖励系数", "float", 5.0, (0.0, 50.0, 0.5),
         "额外奖励 = 系数 × 剩余步数占比，越快到达越高"),
        ("step_cost", "每步成本", "float", 0.01, (0.0, 1.0, 0.001),
         "每个 tick 扣除，步数越多代价越大"),
        ("score_horizon", "评分参考步数", "int", 1000, (10, 20000, 10),
         "只用于速度评分归一化，超过也只把速度分清零，不结束本轮"),
        ("reward_mode", "即时奖励模式", "combo", "combo1",
         ["combo1", "potential"], "combo1 = 方向 + 距离缩短；potential = 越近越高"),
    ]),
    ("学习超参", [
        ("learning_rate", "学习率", "float", 0.15, (0.0, 2.0, 0.01),
         "R-STDP 权重更新幅度"),
        ("tau_trace", "资格迹时间常数", "float", 2000.0, (1.0, 5000.0, 50.0),
         "修改时同步重算 lambda_trace"),
        ("tau_stdp", "STDP 时间常数", "float", 50.0, (1.0, 500.0, 1.0), ""),
        ("psp_factor", "突触电流增益", "float", 25.0, (0.1, 200.0, 1.0), ""),
        ("trace_clamp", "资格迹截断", "float", 1.0, (0.0, 20.0, 0.1),
         "0 表示不截断"),
        ("wta_k", "运动层 WTA 数", "int", 3, (1, 64, 1), ""),
        ("decode_mode", "解码模式", "combo", 0,
         ["0 · WTA 均值", "2 · 膜电位加权"], "运动层群体解码方式"),
        ("use_reward_baseline", "减奖励基线", "bool", True, None,
         "关闭后常数步数成本会变成持续负误差"),
        ("reward_baseline_window", "基线窗口", "int", 200, (1, 5000, 10),
         "最近 N 个 tick 的平均奖励"),
        ("stdp_a_plus", "LTP 幅值", "float", 0.05, (0.0, 1.0, 0.01),
         "仅 C DLL 后端生效"),
        ("stdp_a_minus", "LTD 幅值", "float", 0.06, (0.0, 1.0, 0.01),
         "仅 C DLL 后端生效"),
        ("weight_decay_lambda", "权重衰减", "float", 0.00001,
         (0.0, 0.1, 0.00001), "仅 C DLL 后端生效"),
        ("el_trace", "启用资格迹", "bool", True, None, "消融开关"),
        ("reward_mod", "启用奖励调制", "bool", True, None, "消融开关"),
    ]),
    ("运行与网络", [
        ("avg_conn", "平均每神经元连接", "float", 8.0, (1.0, 32.0, 1.0),
         "修改后点工具条「应用配置」重建网络"),
    ]),
]
DEFAULT_TRAIN_CFG = {s[0]: s[3] for _, specs in TRAIN_SPECS for s in specs}
# 需要写入后端的运行时参数（avg_conn 只在重建网络时使用，不能下发）
CPARAM_KEYS = ({s[0] for s in TRAIN_SPECS[1][1]} |
               {s[0] for s in TRAIN_SPECS[2][1] if s[0] != "avg_conn"})


# ============================================================
# 后端封装
# ============================================================
# ============================================================
# 学习接口（蒸馏老师）：任何能根据当前状态给出「期望关节角速度」
# (omega1, omega2) 的算法都能被脉冲网络蒸馏进来——有现成的不用是傻子。
# 设计：先学（蒸馏老师）→ 再用（G=0 纯自主）→ 再创新（老师逐步放手，
# 网络有机会超越老师局限）。默认无老师，主程序跑纯脉冲网络；设置面板里
# 打开「蒸馏老师」开关后才走引导，且只在能承载引导的向量化后端生效。
# ============================================================
GOAL_D0 = 0.35   # 末端减速距离（仅老师信号使用）


def _goal_omega(t1, t2, ex, ey, tx, ty, max_omega, d0):
    """DLS 阻尼最小二乘逆运动学：已证明可达，现成的解析解。"""
    s1, c1 = math.sin(t1), math.cos(t1)
    s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
    J11 = -L1 * s1 - L2 * s12
    J21 = L1 * c1 + L2 * c12
    J12 = -L2 * s12
    J22 = L2 * c12
    err_x, err_y = tx - ex, ty - ey
    d = math.sqrt(err_x * err_x + err_y * err_y)
    if d < 1e-9 or d0 <= 1e-9:
        return 0.0, 0.0
    lam2 = 0.0625
    a11 = J11 * J11 + J12 * J12 + lam2
    a12 = J11 * J21 + J12 * J22
    a22 = J21 * J21 + J22 * J22 + lam2
    det = a11 * a22 - a12 * a12
    if det > 1e-9:
        vx = (a22 * err_x - a12 * err_y) / det
        vy = (-a12 * err_x + a11 * err_y) / det
    else:
        vx, vy = err_x, err_y
    g1 = J11 * vx + J21 * vy
    g2 = J12 * vx + J22 * vy
    gm = max(abs(g1), abs(g2))
    if gm < 1e-9:
        return 0.0, 0.0
    speed = max_omega * min(1.0, d / d0)
    return g1 / gm * speed, g2 / gm * speed


class Teacher:
    """蒸馏老师基类。子类只需实现 command(state)->(omega1, omega2)。

    state 统一字段：theta1, theta2, end_x, end_y, target_x, target_y,
    L1, L2, max_omega。想蒸馏新算法，只需新增一个继承 Teacher 的类。
    """

    name = "老师"
    desc = "蒸馏老师基类"
    normalize = True          # 训练目标是否归一到满速

    def reset(self, target_x, target_y):
        self.tx, self.ty = target_x, target_y

    def command(self, st):
        raise NotImplementedError

    def target(self, w1, w2):
        if not self.normalize:
            return w1, w2
        gm = max(abs(w1), abs(w2), 1e-12)
        sc = MAX_OMEGA / gm
        return w1 * sc, w2 * sc


class DLSTeacher(Teacher):
    """阻尼最小二乘逆运动学（现成解析解，已知可达）—— 默认的「现成老师」。"""

    name = "DLS 逆运动学（现成解析解）"

    def __init__(self, d0=GOAL_D0):
        self.d0 = d0

    def command(self, st):
        return _goal_omega(st["theta1"], st["theta2"], st["end_x"], st["end_y"],
                           st["target_x"], st["target_y"], st["max_omega"], self.d0)


class TransposeTeacher(Teacher):
    """转置雅可比梯度法（另一种 IK 算法）—— 演示「换一个算法当老师」。"""

    name = "转置雅可比（另一种 IK）"

    def __init__(self, gain=0.6, d0=GOAL_D0):
        self.gain, self.d0 = gain, d0

    def command(self, st):
        t1, t2 = st["theta1"], st["theta2"]
        L1_, L2_ = st["L1"], st["L2"]
        s1, c1 = math.sin(t1), math.cos(t1)
        s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
        J11, J21 = -L1_ * s1 - L2_ * s12, L1_ * c1 + L2_ * c12
        J12, J22 = -L2_ * s12, L2_ * c12
        errx, erry = st["target_x"] - st["end_x"], st["target_y"] - st["end_y"]
        d = math.hypot(errx, erry) + 1e-12
        sp = st["max_omega"] * min(1.0, d / self.d0)
        ux, uy = errx / d * sp, erry / d * sp
        return J11 * ux + J21 * uy, (J12 * ux + J22 * uy) * self.gain


class NullTeacher(Teacher):
    """不教（基线）—— 用以校准诚实预期（纯网络本就到不了）。"""

    name = "无老师（纯网络基线）"
    normalize = False

    def command(self, st):
        return 0.0, 0.0


class FuncTeacher(Teacher):
    """把你自己的 AI / 算法接进来：任意 callable(state)->(w1, w2)。

    用法：teacher = FuncTeacher(my_policy, name="我的RL策略")
        def my_policy(st):
            errx = st["target_x"] - st["end_x"]
            return errx * 0.5, 0.0
    """

    def __init__(self, fn, name="自定义老师", normalize=True):
        self.fn, self.name, self.normalize = fn, name, normalize

    def command(self, st):
        return self.fn(st)


TEACHERS = {
    "dls": DLSTeacher,
    "transpose": TransposeTeacher,
    "null": NullTeacher,
}
TEACHER_KEYS = ["none", "dls", "transpose", "null"]


class SimEngine:
    """统一三种后端 + 在线训练循环（每 tick 注入奖励、碰到目标结算并重置重来）"""

    def __init__(self, dll_path):
        self.dll = dll_path
        self.be = None
        self.initialized = False
        self.name = "—"
        self.on_gpu = False
        self.train_cfg = dict(DEFAULT_TRAIN_CFG)
        self.manual_target = (0.5, 0.5)
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_arrival_reward = 0.0
        self.episode = 0
        self.arrivals = 0
        self.trial_tick = 0
        self.last_episode_ticks = 0
        self.prev_ex, self.prev_ey = 2.0, 0.0
        self.arrived_this_episode = False
        self._trail_buf = []
        # 蒸馏老师（学习接口）：默认无老师，主程序跑纯脉冲网络
        self._teacher = None
        self._teacher_g = 0.7          # 引导比例：(1-G)·网络 + G·老师
        self._teacher_g_min = 0.02
        self._teacher_decay = 0.97     # 逐步放手退火系数
        self._teacher_anneal = False

    def drain_trail(self):
        """取走本轮循环内每个 tick 的末端点（供渲染层一次性画完）"""
        if not self._trail_buf:
            return None
        pts = self._trail_buf
        self._trail_buf = []
        return pts

    @property
    def training_supported(self):
        return bool(getattr(self.be, "can_train", True))

    def reset(self, device_key, cfg, seed, avg_conn=8.0):
        self.destroy()
        self._trail_buf = []
        i, h, m1, m2 = cfg
        if device_key == "cuda":
            be = VecBackend(device="gpu", seed=seed)
        elif device_key == "numpy":
            be = VecBackend(device="cpu", seed=seed)
        else:
            be = CPUBackend(self.dll)
        if device_key == "cdll":
            be.init(i, h, m1, m2, seed=seed, avg_conn=avg_conn)
        else:
            be.init(i, h, m1, m2, seed=seed)
        self.be = be
        self.initialized = True
        self.on_gpu = bool(getattr(be, "_on_gpu", False))
        self.name = ("CUDA · CuPy" if self.on_gpu else
                     ("向量化 · OpenBLAS" if device_key == "numpy" else
                      "C DLL · flat buffer"))
        self._reset_trial_state()
        return be.get_topology()

    def _reset_trial_state(self):
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_arrival_reward = 0.0
        self.episode = 0
        self.arrivals = 0
        self.trial_tick = 0
        self.last_episode_ticks = 0
        self.prev_ex, self.prev_ey = 2.0, 0.0
        self.arrived_this_episode = False

    def tick(self):
        self.be.tick()

    # ---------- 蒸馏老师（学习接口） ----------
    @property
    def _teacher_capable(self):
        """只有能承载 motor_override 的向量化后端才支持引导（C DLL 不认）。"""
        return (self._teacher is not None
                and hasattr(self.be, "motor_override")
                and hasattr(self.be, "_decode_omega"))

    def _net_output(self):
        """本 tick 网络原始解码（用上一 tick 的脉冲，不含引导、不含噪声）。"""
        be = self.be
        if not hasattr(be, "_decode_omega"):
            return 0.0, 0.0
        N = be.config["neuron_count"]
        return (be._decode_omega(be.m1_start, be.m2_start),
                be._decode_omega(be.m2_start, N))

    def _teacher_state(self):
        be = self.be
        ex, ey, tx, ty, _ = be.get_arm()
        return {"theta1": be.arm_theta1, "theta2": be.arm_theta2,
                "end_x": ex, "end_y": ey,
                "target_x": tx, "target_y": ty,
                "L1": L1, "L2": L2, "max_omega": MAX_OMEGA}

    def _apply_teacher(self):
        """若启用老师：计算 ω=(1-G)·网络 + G·老师，经后端 motor_override 注入。
        G=0 时退化为纯网络自主解码。C DLL 后端不认 motor_override，自动跳过。"""
        be = self.be
        if not self._teacher_capable:
            if hasattr(be, "motor_override"):
                be.motor_override = None
            return
        w1, w2 = self._net_output()
        st = self._teacher_state()
        g1, g2 = self._teacher.command(st)
        G = self._teacher_g
        o1 = max(-MAX_OMEGA, min(MAX_OMEGA, (1.0 - G) * w1 + G * g1))
        o2 = max(-MAX_OMEGA, min(MAX_OMEGA, (1.0 - G) * w2 + G * g2))
        be.motor_override = (o1, o2)

    def train_tick(self):
        """一个仿真步：推进物理 → 计算即时奖励 → 注入 → 判定结算/重来"""
        be = self.be
        c = self.train_cfg
        self._apply_teacher()          # 若启用老师：注入 motor_override（仅向量化后端生效）
        be.tick()
        ex, ey, tx, ty, _ = be.get_arm()
        self._trail_buf.append((ex, ey))
        if not c["train_enabled"] or not self.training_supported:
            return
        dist = math.hypot(ex - tx, ey - ty)
        vx, vy = ex - self.prev_ex, ey - self.prev_ey
        tvx, tvy = tx - ex, ty - ey
        vmag = math.hypot(vx, vy)
        cos_r = ((vx * tvx + vy * tvy) / (vmag * dist)
                 if vmag > 1e-6 and dist > 1e-6 else 0.0)
        prev_d = math.hypot(tx - self.prev_ex, ty - self.prev_ey)
        delta = max(-1.0, min(1.0, (prev_d - dist) * 100.0))
        if c["reward_mode"] == "potential":
            init_d = max(1e-6, math.hypot(tx - 2.0, ty))
            r = 1.0 - dist / init_d
        else:
            r = 0.5 * cos_r + 0.5 * delta
        r -= c["step_cost"]

        horizon = max(1, int(c["score_horizon"]))
        just_arrived = False
        if not self.arrived_this_episode and dist <= c["arrive_radius"]:
            self.arrived_this_episode = True
            just_arrived = True
            # 速度分：只按参考步数归一化，超过参考步数只是得 0 分，不结束本轮
            remain = max(0.0, (horizon - 1 - self.trial_tick) / horizon)
            r += c["arrive_bonus"] + c["speed_bonus"] * remain
            self.last_arrival_reward = r
        be.inject_reward(r, 0)
        self.last_reward = r
        self.total_reward += r
        self.trial_tick += 1
        self.prev_ex, self.prev_ey = ex, ey

        if just_arrived:   # 碰到目标是本轮唯一的结束条件
            be.tick()      # 消费刚注入的奖励（含到达奖励），否则重置时会被清队列丢掉
            self.episode += 1
            self.arrivals += 1
            self.last_episode_ticks = self.trial_tick
            # 老师逐步放手（先学 → 再用）：G 退火，让网络逐步接管控制权
            if self._teacher_anneal and self._teacher is not None:
                self._teacher_g = max(self._teacher_g_min,
                                     self._teacher_g * self._teacher_decay)
            ntx, nty = self._next_target()
            self.episode_reset(ntx, nty)

    def episode_reset(self, tx, ty):
        """重置一轮：后端归位，同时同步窗口侧的本轮计数（外部也可安全调用）"""
        self._trail_buf = []   # 清空轨迹：新一轮只画本次运动，不留上一轮残影/瞬移
        self.be.episode_reset(tx, ty)
        self.prev_ex, self.prev_ey = 2.0, 0.0
        self.trial_tick = 0
        self.arrived_this_episode = False

    def _next_target(self):
        if not self.train_cfg["auto_target"]:
            return self.manual_target
        # 肘关节限位 ±0.9π → 末端最短可达半径 2cos(0.45π) ≈ 0.313，
        # 下限取 0.4 保证物理可达；上限 1.5 < 臂长 2。
        r = 0.4 + random.random() * 1.1
        a = random.random() * math.pi * 2
        return (round(r * math.cos(a), 2), round(r * math.sin(a), 2))

    def set_params(self, values):
        """把运行时参数写入后端，返回当前后端不支持的参数名"""
        failed = []
        for key, value in values.items():
            try:
                if self.be.set_param(key, float(value)) != 0:
                    failed.append(key)
            except Exception:
                failed.append(key)
        return failed

    def get_param(self, name):
        try:
            return self.be.get_param(name)
        except Exception:
            return float("nan")

    def snapshot(self):
        # 统一成 (mem, spk, arm, meta)，丢掉用不到的 last_spike
        mem, spk, last, arm, meta = self.be.get_snapshot_arrays()
        return mem, spk, arm, meta

    def set_target(self, x, y):
        self.manual_target = (float(x), float(y))
        self.be.set_target(float(x), float(y))

    def destroy(self):
        if self.initialized:
            try:
                self.be.destroy()
            except Exception:
                pass
            self.initialized = False


class SimWorker(QThread):
    """后台快跑 tick，只保留最新一帧；异常上报而不是静默吞掉"""
    failed = Signal(str)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._running = False
        self._steps = 2
        self._lock = threading.Lock()
        self._latest = None
        self.tick_rate = 0.0
        self._count = 0
        self._t0 = time.perf_counter()

    def set_steps(self, n):
        self._steps = max(1, min(int(n), 50))

    def stop(self):
        self._running = False

    def latest(self):
        with self._lock:
            v = self._latest
            self._latest = None
        return v

    def run(self):
        self._running = True
        self._t0 = time.perf_counter()
        while self._running:
            try:
                for _ in range(self._steps):
                    self.engine.train_tick()
                mem, spk, arm, meta = self.engine.snapshot()
                snap = (np.ascontiguousarray(mem, dtype=np.float32),
                        np.ascontiguousarray(spk, dtype=np.int32),
                        np.ascontiguousarray(arm, dtype=np.float64),
                        np.ascontiguousarray(meta, dtype=np.int32),
                        self.engine.drain_trail())
                with self._lock:
                    self._latest = snap
                self._count += self._steps
            except Exception as e:
                self.failed.emit(f"{type(e).__name__}: {e}")
                break
            now = time.perf_counter()
            if now - self._t0 >= 0.5:
                self.tick_rate = self._count / (now - self._t0)
                self._count = 0
                self._t0 = now
            time.sleep(0.0002)


# ============================================================
# 小部件
# ============================================================
class GradientStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(10)
        self.setMinimumWidth(130)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 5, 5)
        grad = QLinearGradient(0, 0, self.width(), 0)
        for i in (0, 64, 128, 192, 255):
            c = LUT[i]
            grad.setColorAt(i / 255, QColor(int(c[0]), int(c[1]), int(c[2])))
        p.fillPath(path, QBrush(grad))
        p.end()


class MetricCard(QFrame):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setMinimumWidth(130)
        self.setFixedHeight(46)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(11, 6, 11, 6)
        lay.setSpacing(6)
        k = QLabel(label)
        k.setObjectName("MetricLabel")
        self.value = QLabel("—")
        self.value.setObjectName("MetricValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight |
                                 Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(k)
        lay.addStretch(1)
        lay.addWidget(self.value)

    def set_value(self, text, color=None):
        self.value.setText(text)
        if color:
            self.value.setStyleSheet(f"color:{color};")


class TrendChart(QWidget):
    """三条滚动曲线：即时奖励 / 末端距离 / 发放率"""
    SERIES = (("reward", "即时奖励", "ok"), ("dist", "末端距离", "accent"),
              ("rate", "发放率", "warn"))
    WINDOW = 260

    def __init__(self, T, parent=None):
        super().__init__(parent)
        self.T = T
        self.data = {k: deque(maxlen=self.WINDOW) for k, _, _ in self.SERIES}
        self.setMinimumHeight(140)

    def set_theme(self, T):
        self.T = T
        self.update()

    def clear(self):
        for d in self.data.values():
            d.clear()
        self.update()

    def push(self, reward, dist, rate):
        self.data["reward"].append(reward)
        self.data["dist"].append(dist)
        self.data["rate"].append(rate)
        self.update()

    def _range(self, key, vals):
        if key == "dist":
            return 0.0, 2.6
        if key == "rate":
            return 0.0, max(0.12, max(vals) * 1.2)
        lo, hi = min(vals), max(vals)
        pad = max(0.02, (hi - lo) * 0.2)
        return lo - pad, hi + pad

    def paintEvent(self, event):
        T = self.T
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        top, bottom, pad = 30, 16, 8
        ph = h - top - bottom
        p.fillRect(0, 0, w, h, QColor(T["chart"]))
        p.setPen(QPen(QColor(T["grid"]), 1))
        for i in range(5):
            y = top + ph * i / 4
            p.drawLine(0, y, w, y)
        for i in range(7):
            x = pad + (w - 2 * pad) * i / 6
            p.drawLine(x, top, x, top + ph)
        p.setFont(QFont(FONT_FAMILY, 9))
        lx = 10
        for key, label, ck in self.SERIES:
            col = QColor(T[ck])
            vals = list(self.data[key])
            p.fillRect(lx, 11, 8, 8, col)
            p.setPen(QColor(T["sub"]))
            p.drawText(lx + 13, 19, label)
            if vals:
                txt = (f"{vals[-1]:+.3f}" if key == "reward"
                       else f"{vals[-1]:.3f}")
                p.setPen(col)
                p.drawText(lx + 76, 19, txt)
            lx += 158
            if len(vals) < 2:
                continue
            lo, hi = self._range(key, vals)
            span = hi - lo or 1.0
            path = QPainterPath()
            first_x = last_x = pad
            for i, v in enumerate(vals):
                x = pad + i / (len(vals) - 1) * (w - 2 * pad)
                y = top + ph - max(0.0, min(1.0, (v - lo) / span)) * ph
                if i == 0:
                    path.moveTo(x, y)
                    first_x = x
                else:
                    path.lineTo(x, y)
                last_x = x
            fill = QPainterPath(path)
            fill.lineTo(last_x, top + ph)
            fill.lineTo(first_x, top + ph)
            fill.closeSubpath()
            fc = QColor(col)
            fc.setAlpha(30)
            p.fillPath(fill, QBrush(fc))
            p.setPen(QPen(col, 1.7))
            p.drawPath(path)
        p.setPen(QColor(T["dim"]))
        p.drawText(pad, h - 3, f"最近 {len(self.data['reward'])} 帧")
        p.end()


def make_card(title, hint, view, extra=None):
    card = QFrame()
    card.setObjectName("Card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    head = QWidget()
    head.setObjectName("CardHead")
    head.setFixedHeight(44)
    hl = QHBoxLayout(head)
    hl.setContentsMargins(16, 0, 12, 0)
    hl.setSpacing(8)
    t = QLabel(title)
    t.setObjectName("CardTitle")
    hl.addWidget(t)
    if hint:
        hh = QLabel(hint)
        hh.setObjectName("CardHint")
        hl.addWidget(hh)
    hl.addStretch(1)
    if extra is not None:
        hl.addWidget(extra)
    lay.addWidget(head)
    lay.addWidget(view, 1)
    return card


# ============================================================
# 网络拓扑视图
# ============================================================
class NetworkGraph(QGraphicsView):
    selectionChanged = Signal(int)

    def __init__(self, T, parent=None):
        super().__init__(parent)
        self.T = T
        self.scene_ = QGraphicsScene(self)
        self.scene_.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.W, self.H = 1000, 900
        self._syn_pix = None
        self.topo = None
        self.N = 0
        self.items = []
        self._px = np.zeros(0, dtype=np.float32)
        self._py = np.zeros(0, dtype=np.float32)
        self._radius = 4.0
        self._last_idx = np.zeros(0, dtype=np.int32)
        self._prev_spk = np.zeros(0, dtype=bool)
        self._last_m = np.zeros(0, dtype=np.float32)
        self._syn_pre = np.zeros(0, dtype=np.int32)
        self._syn_post = np.zeros(0, dtype=np.int32)
        self.layer_of = np.zeros(0, dtype=np.int32)
        self.in_deg = np.zeros(0, dtype=np.int32)
        self.out_deg = np.zeros(0, dtype=np.int32)
        self.hover_id = -1
        self.selected_id = -1
        self._user_zoom = False
        self._fitting = False
        self.spike_cur = self.spike_ghost = self.active_item = None
        self.hover_ring = self.select_ring = None
        self.apply_theme(T)

    def apply_theme(self, T):
        self.T = T
        self.scene_.setBackgroundBrush(QBrush(QColor(T["scene_net"])))
        if self.topo is not None:
            self.load_topology(self.topo)
        self._fit()

    def load_topology(self, topo):
        self.topo = topo
        self.scene_.clear()
        T = self.T
        N = topo["neuron_count"]
        self.N = N
        self.selected_id = self.hover_id = -1
        syns = topo["synapses"]
        self._syn_pre = np.fromiter((s["pre"] for s in syns), dtype=np.int32,
                                    count=len(syns))
        self._syn_post = np.fromiter((s["post"] for s in syns), dtype=np.int32,
                                     count=len(syns))
        self._syn_w = np.fromiter((abs(s["w"]) for s in syns), dtype=np.float32,
                                  count=len(syns))

        ranges = [
            (0, topo["input"], 0),
            (topo["input"], topo["input"] + topo["hidden"], 1),
            (topo["input"] + topo["hidden"],
             topo["input"] + topo["hidden"] + topo["motor1"], 2),
            (topo["input"] + topo["hidden"] + topo["motor1"], N, 3),
        ]
        hidden_n = topo["hidden"]
        cols = min(64, max(36, int(math.sqrt(hidden_n) * 1.5)))
        grid_cell = 14.0
        rows = math.ceil(hidden_n / cols)
        margin_x, row_h, gap = 44.0, 42.0, 95.0
        W = max(1000, int(margin_x * 2 + cols * grid_cell))
        heights = [row_h, rows * grid_cell, row_h, row_h]
        H = int(76 + sum(heights) + gap * 3 + 60)
        self.W, self.H = W, H
        self.scene_.setSceneRect(0, 0, W, H)

        self._px = np.zeros(N, dtype=np.float32)
        self._py = np.zeros(N, dtype=np.float32)
        self.layer_of = np.zeros(N, dtype=np.int32)
        y = 66.0
        for li, (start, end, lt) in enumerate(ranges):
            size = end - start
            self.layer_of[start:end] = lt
            bh = heights[li] + 16
            # 层带
            band = QPainterPath()
            band.addRoundedRect(QRectF(16, y - 8, W - 32, bh), 12, 12)
            col = QColor(T["layer"][lt])
            fill = QColor(col)
            fill.setAlpha(16)
            item = self.scene_.addPath(band, QPen(QColor(T["band_edge"]), 1),
                                       QBrush(fill))
            item.setZValue(-4)
            name = QGraphicsTextItem(LAYER_NAMES[lt])
            name.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
            name.setDefaultTextColor(col)
            name.setPos(26, y - 14)
            name.setZValue(3)
            self.scene_.addItem(name)
            cnt = QGraphicsTextItem(f"{size} 个神经元")
            cnt.setFont(QFont(FONT_FAMILY, 10))
            cnt.setDefaultTextColor(QColor(T["dim"]))
            cnt.setPos(26, y + 6)
            cnt.setZValue(3)
            self.scene_.addItem(cnt)

            if lt == 1:
                cw = cols * grid_cell
                x0 = (W - cw) / 2
                for k in range(size):
                    rr, cc = divmod(k, cols)
                    self._px[start + k] = x0 + (cc + 0.5) * grid_cell
                    self._py[start + k] = y + 10 + (rr + 0.5) * grid_cell
            else:
                step = (W - 2 * margin_x) / (size + 1)
                for k in range(size):
                    self._px[start + k] = margin_x + step * (k + 1)
                    self._py[start + k] = y + row_h / 2
            y += heights[li] + gap

        self._radius = min(4.2, grid_cell * 0.30)
        self.in_deg = np.bincount(self._syn_post, minlength=N).astype(np.int32)
        self.out_deg = np.bincount(self._syn_pre, minlength=N).astype(np.int32)

        # 突触连线：全部预渲染到一张 QPixmap（零图元、无 AA 长路径）
        self._build_syn_pixmap(syns)

        # 高亮层：当前发放突触 / 当前发放光晕 / 上一帧残影
        self.active_item = self.scene_.addPath(
            QPainterPath(), QPen(QColor(*T["active_syn"]), 1.5),
            QBrush(Qt.BrushStyle.NoBrush))
        self.active_item.setZValue(0)
        self.spike_ghost = self.scene_.addPath(
            QPainterPath(), QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(*T["ghost"])))
        self.spike_ghost.setZValue(3)
        self.spike_cur = self.scene_.addPath(
            QPainterPath(), QPen(QColor(*T["spike_edge"]), 1.4),
            QBrush(QColor(*T["spike"])))
        self.spike_cur.setZValue(4)

        # 神经元节点（单图元，不再为每个节点建外发光圈）
        R = self._radius
        edge = QPen(QColor(T["node_edge"]), 0.7)
        self.items = []
        for i in range(N):
            x, y = float(self._px[i]), float(self._py[i])
            e = QGraphicsEllipseItem(x - R, y - R, 2 * R, 2 * R)
            e.setBrush(BRUSHES[0])
            e.setPen(edge)
            e.setZValue(2)
            self.scene_.addItem(e)
            self.items.append(e)
        self._last_idx = np.full(N, -1, dtype=np.int32)
        self._prev_spk = np.zeros(N, dtype=bool)
        self._last_m = np.full(N, MEM_REST, dtype=np.float32)

        self.hover_ring = self.scene_.addEllipse(
            -1, -1, 2, 2, QPen(QColor(T["text"]), 1.2),
            QBrush(Qt.BrushStyle.NoBrush))
        self.hover_ring.setZValue(6)
        self.hover_ring.setVisible(False)
        self.select_ring = self.scene_.addEllipse(
            -1, -1, 2, 2, QPen(QColor(T["select"]), 2.2),
            QBrush(Qt.BrushStyle.NoBrush))
        self.select_ring.setZValue(7)
        self.select_ring.setVisible(False)
        self._fit()

    def _build_syn_pixmap(self, syns):
        T = self.T
        pix = QPixmap(int(self.W), int(self.H))
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        max_w = float(np.max(self._syn_w)) if len(self._syn_w) else 1.0
        max_w = max(max_w, 0.01)
        buckets = [[] for _ in range(4)]
        for s in syns:
            pre, post = s["pre"], s["post"]
            if not (0 <= pre < self.N and 0 <= post < self.N):
                continue
            b = min(3, int(abs(s["w"]) / max_w * 4))
            buckets[b].append(QLineF(float(self._px[pre]), float(self._py[pre]),
                                     float(self._px[post]), float(self._py[post])))
        widths = [0.5, 0.8, 1.1, 1.5]
        for b, lines in enumerate(buckets):
            if not lines:
                continue
            r, g, bl = T["syn"][b]
            p.setPen(QPen(QColor(r, g, bl, 34 if b == 0 else 58), widths[b]))
            try:
                p.drawLines(lines)
            except (TypeError, RuntimeError):
                for ln in lines:
                    p.drawLine(ln)
        p.end()
        self._syn_pix = pix

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor(self.T["scene_net"]))
        if self._syn_pix is not None:
            painter.drawPixmap(0, 0, self._syn_pix)

    # ---- 逐帧 ----
    def update_snapshot(self, mem, spk):
        N = self.N
        if len(mem) != N:
            return
        # numpy 差分：只更新颜色变化的节点
        t = (np.clip(mem, MEM_REST - 2.0, 0.0) - MEM_REST) / (MEM_THRESH - MEM_REST)
        idx = np.clip((t * (LUT_SIZE - 1)).astype(np.int32), 0, LUT_SIZE - 1)
        changed = np.nonzero(idx != self._last_idx)[0]
        for i in changed:
            self.items[int(i)].setBrush(BRUSHES[int(idx[i])])
        self._last_idx = idx
        self._last_m = mem

        cur = spk.astype(bool)
        cur_idx = np.nonzero(cur)[0]
        ghost_idx = np.nonzero(self._prev_spk & ~cur)[0]
        self.spike_cur.setPath(self._circles(cur_idx, self._radius + 3.0))
        self.spike_ghost.setPath(self._circles(ghost_idx, self._radius + 5.0))
        self._prev_spk = cur

        sel = self.selected_id
        path = QPainterPath()
        if sel >= 0:
            mask = (self._syn_pre == sel) | (self._syn_post == sel)
            chosen = np.nonzero(mask)[0][:1200]
        elif len(cur_idx):
            mask = np.isin(self._syn_pre, cur_idx)
            chosen = np.nonzero(mask)[0][:450]
        else:
            chosen = np.zeros(0, dtype=np.int32)
        if len(chosen):
            for j in chosen:
                a, b = int(self._syn_pre[j]), int(self._syn_post[j])
                path.moveTo(float(self._px[a]), float(self._py[a]))
                path.lineTo(float(self._px[b]), float(self._py[b]))
        self.active_item.setPath(path)

    def _circles(self, indices, r):
        path = QPainterPath()
        for i in indices:
            x, y = float(self._px[i]), float(self._py[i])
            path.addEllipse(x - r, y - r, 2 * r, 2 * r)
        return path

    # ---- 交互 ----
    def _nearest(self, sp):
        if self.N == 0:
            return -1
        d2 = (self._px - sp.x()) ** 2 + (self._py - sp.y()) ** 2
        i = int(np.argmin(d2))
        return i if d2[i] <= (self._radius + 6) ** 2 else -1

    def _place(self, item, nid, pad):
        if nid < 0:
            item.setVisible(False)
            return
        r = self._radius + pad
        x, y = float(self._px[nid]), float(self._py[nid])
        item.setRect(x - r, y - r, 2 * r, 2 * r)
        item.setVisible(True)

    def _rings(self):
        if self.hover_ring is not None:
            self._place(self.hover_ring, self.hover_id, 4.5)
        if self.select_ring is not None:
            self._place(self.select_ring, self.selected_id, 7.0)

    def _tip(self, i):
        return (f"神经元 #{i} · {LAYER_NAMES[int(self.layer_of[i])]}\n"
                f"膜电位   {float(self._last_m[i]):.2f} mV\n"
                f"状态     {'发放中' if self._prev_spk[i] else '静息'}\n"
                f"连接     入 {int(self.in_deg[i])} / 出 {int(self.out_deg[i])}")

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        i = self._nearest(self.mapToScene(event.pos()))
        if i != self.hover_id:
            self.hover_id = i
            self._rings()
            if i >= 0:
                QToolTip.showText(self.mapToGlobal(event.pos()), self._tip(i), self)
            else:
                QToolTip.hideText()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            i = self._nearest(self.mapToScene(event.pos()))
            self.selected_id = -1 if (i < 0 or i == self.selected_id) else i
            self._rings()
            self.selectionChanged.emit(self.selected_id)
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        f = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self._user_zoom = True
        self.scale(f, f)

    def reset_view(self):
        self._user_zoom = False
        self._fit()

    def _fit(self):
        if self._fitting:
            return
        self._fitting = True
        self.fitInView(self.scene_.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fitting = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoom:
            self._fit()


# ============================================================
# 机械臂视图
# ============================================================
class ArmPhysicsView(QGraphicsView):
    SCALE = 88.0
    EXTENT = 220.0

    def __init__(self, T, parent=None):
        super().__init__(parent)
        self.T = T
        self.scene_ = QGraphicsScene(self)
        self.scene_.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.scene_.setSceneRect(-self.EXTENT, -self.EXTENT,
                                 2 * self.EXTENT, 2 * self.EXTENT)
        self._target = (0.5, 0.5)
        self._fitting = False
        self.trail = deque(maxlen=600)
        self.trail_path = QPainterPath()
        self.apply_theme(T)
        self.setTarget(*self._target)
        self.update_arm(0, 0, 0, 0, 0, 0, 2.0)

    def apply_theme(self, T):
        self.T = T
        self._build()
        self._fit()

    def drawBackground(self, painter, rect):
        T = self.T
        painter.fillRect(rect, QColor(T["scene_net"]))
        step = 0.25 * self.SCALE
        grid = QPen(QColor(T["grid"]), 1)
        axis = QPen(QColor(T["grid_axis"]), 1.4)
        grid.setCosmetic(True)
        axis.setCosmetic(True)
        x = math.floor(rect.left() / step) * step
        while x <= rect.right():
            painter.setPen(axis if round(x / step) % 2 == 0 else grid)
            painter.drawLine(x, rect.top(), x, rect.bottom())
            x += step
        y = math.floor(rect.top() / step) * step
        while y <= rect.bottom():
            painter.setPen(axis if round(y / step) % 2 == 0 else grid)
            painter.drawLine(rect.left(), y, rect.right(), y)
            y += step

    def _build(self):
        T = self.T
        self.scene_.clear()
        s = self.SCALE
        self.reach = self.scene_.addEllipse(
            -2 * s, -2 * s, 4 * s, 4 * s,
            QPen(QColor(T["reach"]), 1.4, Qt.PenStyle.DashLine),
            QBrush(Qt.BrushStyle.NoBrush))
        self.reach.setZValue(-4)
        self.trail_path = QPainterPath()
        self.trail_item = self.scene_.addPath(
            self.trail_path,
            QPen(QColor(*T["trail"]), 2.2, Qt.PenStyle.SolidLine,
                 Qt.PenCapStyle.RoundCap), QBrush(Qt.BrushStyle.NoBrush))
        self.trail_item.setZValue(1)
        self.trail.clear()
        fill = QColor(*T["target_fill"])
        self.t_halo = self.scene_.addEllipse(
            -16, -16, 32, 32, QPen(Qt.PenStyle.NoPen), QBrush(fill))
        self.t_halo.setZValue(2)
        self.t_ring = self.scene_.addEllipse(
            -7, -7, 14, 14, QPen(QColor(T["target"]), 2.2),
            QBrush(Qt.BrushStyle.NoBrush))
        self.t_ring.setZValue(3)
        cp = QPen(QColor(T["target"]), 1.2)
        self.t_h = self.scene_.addLine(-15, 0, 15, 0, cp)
        self.t_v = self.scene_.addLine(0, -15, 0, 15, cp)
        self.t_h.setZValue(3)
        self.t_v.setZValue(3)
        self.t_lbl = self.scene_.addText("目标")
        self.t_lbl.setFont(QFont(FONT_FAMILY, 9))
        self.t_lbl.setDefaultTextColor(QColor(T["target"]))
        self.t_lbl.setZValue(4)
        self.shoulder = self.scene_.addEllipse(
            -6, -6, 12, 12, QPen(QColor(T["joint"]), 2),
            QBrush(QColor(T["panel_alt"])))
        self.shoulder.setZValue(4)
        self.link1 = self.scene_.addLine(
            0, 0, s, 0, QPen(QColor(T["link1"]), 8, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap))
        self.link1.setZValue(2)
        self.link2 = self.scene_.addLine(
            s, 0, 2 * s, 0, QPen(QColor(T["link2"]), 7, Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap))
        self.link2.setZValue(2)
        self.elbow = self.scene_.addEllipse(
            -5, -5, 10, 10, QPen(QColor(T["joint"]), 2),
            QBrush(QColor(T["panel"])))
        self.elbow.setZValue(4)
        self.e_halo = self.scene_.addEllipse(
            -13, -13, 26, 26, QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(*T["end_fill"])))
        self.e_halo.setZValue(3)
        self.e_dot = self.scene_.addEllipse(
            -5, -5, 10, 10, QPen(QColor(T["end"]), 2),
            QBrush(QColor(T["end"])))
        self.e_dot.setZValue(5)
        self.e_lbl = self.scene_.addText("末端")
        self.e_lbl.setFont(QFont(FONT_FAMILY, 9))
        self.e_lbl.setDefaultTextColor(QColor(T["end"]))
        self.e_lbl.setZValue(5)
        o = self.scene_.addText("肩关节")
        o.setFont(QFont(FONT_FAMILY, 8))
        o.setDefaultTextColor(QColor(T["scene_label"]))
        o.setPos(8, 6)
        o.setZValue(4)

    def setTarget(self, tx, ty):
        self._target = (tx, ty)
        x, y = tx * self.SCALE, -ty * self.SCALE
        self.t_halo.setRect(x - 16, y - 16, 32, 32)
        self.t_ring.setRect(x - 7, y - 7, 14, 14)
        self.t_h.setLine(x - 15, y, x + 15, y)
        self.t_v.setLine(x, y - 15, x, y + 15)
        self.t_lbl.setPos(x + 12, y - 22)

    def clearTrail(self):
        self.trail.clear()
        self.trail_path = QPainterPath()
        self.trail_item.setPath(self.trail_path)

    def append_trail(self, points):
        """批量追加一帧内所有 tick 的末端点。

        高速设备（GPU / 25x）一个渲染帧之间会跑几十个 tick，
        若每帧只画一个点，轨迹就变成稀疏折线、看起来像瞬移乱跳。
        这里把 worker 收集到的每一步都补上；None 表示一轮结束的断点。
        """
        if not points:
            return
        s = self.SCALE
        for p in points:
            if p is None:
                self.trail.append(None)
            else:
                self.trail.append((p[0] * s, -p[1] * s))
        self.trail_path = QPainterPath()
        down = False
        for p in self.trail:
            if p is None:
                down = False
                continue
            if down:
                self.trail_path.lineTo(*p)
            else:
                self.trail_path.moveTo(*p)
                down = True
        self.trail_item.setPath(self.trail_path)

    def update_arm(self, t1, t2, o1, o2, ex, ey, max_reach):
        s = self.SCALE
        e1x, e1y = math.cos(t1), math.sin(t1)
        e2x, e2y = e1x + math.cos(t1 + t2), e1y + math.sin(t1 + t2)
        ax, ay = e1x * s, -e1y * s
        bx, by = e2x * s, -e2y * s
        r = max_reach * s
        if abs(self.reach.rect().width() - 2 * r) > 0.5:
            self.reach.setRect(-r, -r, 2 * r, 2 * r)
        self.link1.setLine(0, 0, ax, ay)
        self.link2.setLine(ax, ay, bx, by)
        self.elbow.setRect(ax - 5, ay - 5, 10, 10)
        self.e_halo.setRect(bx - 13, by - 13, 26, 26)
        self.e_dot.setRect(bx - 5, by - 5, 10, 10)
        self.e_lbl.setPos(bx + 11, by - 20)

    def _fit(self):
        if self._fitting:
            return
        self._fitting = True
        self.fitInView(self.scene_.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fitting = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self, engine, theme_key="dark"):
        super().__init__()
        self.engine = engine
        self.theme_key = theme_key
        self.T = THEMES[theme_key]
        self._initializing = True      # 构建控件期间屏蔽信号触发的 _reset
        self.controls = {}
        self._last_episode = -1        # 轨迹按回合清空：检测 episode 变化
        self._teacher_forced_vec = False   # 是否因启用老师而强制切到向量化后端
        self.setWindowTitle("脉冲神经网络 · 实时训练可视化  |  Benchmark-2")
        # 按当前屏幕工作区自适应，避免 150% 缩放下窗口底边被任务栏遮住
        scr = QGuiApplication.primaryScreen()
        avail = scr.availableGeometry() if scr else None
        if avail and avail.isValid() and avail.width() > 0:
            win_w = max(1280, min(1680, avail.width() - 28))
            win_h = max(760, min(1000, avail.height() - 48))
        else:
            win_w, win_h = 1680, 1000
        self.resize(win_w, win_h)
        self.setMinimumSize(1280, 760)

        root = QWidget()
        root.setObjectName("Root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        self.setCentralWidget(root)

        self._build_settings_dock()
        lay.addWidget(self._build_header())
        lay.addWidget(self._build_controls())
        lay.addWidget(self._build_config())

        self.net_view = NetworkGraph(self.T)
        self.arm_view = ArmPhysicsView(self.T)
        self.chart = TrendChart(self.T)
        self.net_hint = QLabel("—")
        self.net_hint.setObjectName("CardHint")
        btn_fit = QPushButton("适应窗口")
        btn_fit.setObjectName("Tiny")
        btn_fit.setToolTip("复位网络视图缩放，快捷键 F")
        btn_fit.clicked.connect(self.net_view.reset_view)
        pair = QWidget()
        ph = QHBoxLayout(pair)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(10)
        ph.addWidget(self.net_hint)
        ph.addWidget(btn_fit)
        net_card = make_card("网络拓扑", None, self.net_view, pair)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(net_card)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(make_card("机械臂运动", "两关节 · 末端轨迹与目标追踪",
                                  self.arm_view))
        right.addWidget(make_card("实时趋势", "奖励 / 距离 / 发放率", self.chart))
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 1)
        main.addWidget(right)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        net_card.setMinimumWidth(340)
        right.setMinimumWidth(340)
        # 显式给出初始宽度，避免子面板的 sizeHint 把 3:2 比例挤反
        main.setSizes([int(win_w * 0.58), int(win_w * 0.42)])
        lay.addWidget(main, 1)
        lay.addWidget(self._build_footer())

        for key, fn in (("Space", self._toggle_play), ("S", self._step_once),
                        ("R", self._reset), ("T", self._random_target),
                        ("F", self.net_view.reset_view),
                        ("P", self._toggle_settings)):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(fn)
            self.addAction(act)
        self.net_view.selectionChanged.connect(self._on_selection)

        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(16)
        self.gui_timer.timeout.connect(self._render_frame)
        self.worker = None
        self.is_playing = False
        self._speed = 2.0
        self._render_ms = 0.0
        self._fps = 0.0
        self._total_reward = 0.0
        self._last_target = None
        self._error = ""

        self._initializing = False
        self._refresh_badge()
        self._reset()

    # ---------- UI ----------
    def _build_header(self):
        h = QWidget()
        h.setObjectName("Header")
        h.setFixedHeight(66)
        hl = QHBoxLayout(h)
        hl.setContentsMargins(16, 8, 16, 8)
        hl.setSpacing(10)
        logo = QLabel("N")
        logo.setObjectName("Logo")
        logo.setFixedSize(38, 38)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(logo)
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        t = QLabel("脉冲神经网络 · 在线训练可视化")
        t.setObjectName("Title")
        s = QLabel("Benchmark-2   |   碰到目标即结算并重来   |   逐 tick R-STDP · 参数实时可调")
        s.setObjectName("Subtitle")
        v.addWidget(t)
        v.addWidget(s)
        hl.addWidget(box)
        hl.addStretch(1)
        self.chip_gpu = QLabel(gpu_label())
        self.chip_gpu.setObjectName("Chip")
        hl.addWidget(self.chip_gpu)
        self.lbl_perf = QLabel("0 FPS · 0 tick/s · 0.0 ms")
        self.lbl_perf.setObjectName("Perf")
        hl.addWidget(self.lbl_perf)
        self.chip_dev = QLabel("后端 —")
        self.chip_dev.setObjectName("Chip")
        hl.addWidget(self.chip_dev)
        self.badge = QLabel("已暂停")
        self.badge.setObjectName("Badge")
        hl.addWidget(self.badge)
        self.btn_theme = QPushButton("浅色")
        self.btn_theme.setObjectName("Ghost")
        self.btn_theme.setToolTip("切换明暗主题")
        self.btn_theme.clicked.connect(self._toggle_theme)
        hl.addWidget(self.btn_theme)
        return h

    def _build_controls(self):
        bar = QFrame()
        bar.setObjectName("ControlBar")
        bar.setFixedHeight(60)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(14, 8, 14, 8)
        hl.setSpacing(8)

        def sep():
            f = QFrame()
            f.setObjectName("VSep")
            f.setFixedWidth(1)
            f.setFixedHeight(28)
            return f

        def group(t):
            l = QLabel(t)
            l.setObjectName("GroupLabel")
            return l

        hl.addWidget(group("仿真控制"))
        self.btn_play = QPushButton("运行")
        self.btn_play.setObjectName("Primary")
        self.btn_play.clicked.connect(self._toggle_play)
        hl.addWidget(self.btn_play)
        b = QPushButton("单步")
        b.setToolTip("快捷键 S")
        b.clicked.connect(self._step_once)
        hl.addWidget(b)
        b = QPushButton("重置")
        b.setToolTip("快捷键 R")
        b.clicked.connect(self._reset)
        hl.addWidget(b)
        hl.addSpacing(4)
        hl.addWidget(sep())
        hl.addSpacing(4)
        hl.addWidget(group("计算设备"))
        self.combo_dev = QComboBox()
        self.combo_dev.addItems([d[0] for d in DEVICES])
        self.combo_dev.setMinimumWidth(170)
        self.combo_dev.currentIndexChanged.connect(lambda _: self._reset())
        self.combo_dev.setCurrentIndex(DEFAULT_DEVICE_INDEX)
        hl.addWidget(self.combo_dev)
        hl.addWidget(group("速度"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5×", "1×", "2×", "5×", "10×", "25×"])
        self.combo_speed.setCurrentIndex(2)
        self.combo_speed.currentIndexChanged.connect(self._on_speed)
        hl.addWidget(self.combo_speed)
        hl.addSpacing(4)
        hl.addWidget(sep())
        hl.addSpacing(4)
        hl.addWidget(group("目标"))
        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-1.5, 1.5)
        self.spin_tx.setSingleStep(0.05)
        self.spin_tx.setValue(0.5)
        self.spin_tx.setPrefix("X  ")
        self.spin_tx.setDecimals(2)
        hl.addWidget(self.spin_tx)
        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-1.5, 1.5)
        self.spin_ty.setSingleStep(0.05)
        self.spin_ty.setValue(0.5)
        self.spin_ty.setPrefix("Y  ")
        self.spin_ty.setDecimals(2)
        hl.addWidget(self.spin_ty)
        b = QPushButton("设定")
        b.clicked.connect(self._set_target)
        hl.addWidget(b)
        b = QPushButton("随机")
        b.setToolTip("快捷键 T")
        b.clicked.connect(self._random_target)
        hl.addWidget(b)
        hl.addSpacing(4)
        hl.addWidget(sep())
        b = QPushButton("设置面板")
        b.setObjectName("Ghost")
        b.setToolTip("显示/隐藏训练与参数面板（快捷键 P）")
        b.clicked.connect(self._toggle_settings)
        hl.addWidget(b)
        self.lbl_trial = QLabel("第 0 轮 · 到达 0")
        self.lbl_trial.setObjectName("Perf")
        self.lbl_trial.setToolTip("累计完成轮次 / 碰到目标次数 / 本轮步数")
        hl.addWidget(self.lbl_trial)
        self.lbl_teacher_status = QLabel("老师：无")
        self.lbl_teacher_status.setObjectName("Perf")
        self.lbl_teacher_status.setToolTip(
            "蒸馏老师状态（学习接口）：网络先模仿老师，再逐步自主")
        hl.addWidget(self.lbl_teacher_status)
        hl.addStretch(1)
        return bar

    def _build_config(self):
        bar = QFrame()
        bar.setObjectName("ConfigBar")
        bar.setFixedHeight(58)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(14, 8, 14, 8)
        hl.setSpacing(8)
        l = QLabel("网络规模")
        l.setObjectName("GroupLabel")
        hl.addWidget(l)
        self.combo_preset = QComboBox()
        self.combo_preset.addItems([p[0] for p in PRESETS])
        self.combo_preset.setCurrentIndex(2)
        self.combo_preset.setMinimumWidth(150)
        self.combo_preset.currentIndexChanged.connect(lambda _: self._reset())
        hl.addWidget(self.combo_preset)
        hl.addWidget(QLabel("种子"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        self.spin_seed.setMinimumWidth(94)
        hl.addWidget(self.spin_seed)
        self.chk_seed = QCheckBox("固定种子")
        self.chk_seed.setChecked(True)
        self.chk_seed.setToolTip("固定后每次重置得到完全相同的网络，便于复现实验")
        hl.addWidget(self.chk_seed)
        b = QPushButton("应用配置")
        b.setObjectName("Primary")
        b.clicked.connect(self._reset)
        hl.addWidget(b)
        self.lbl_cfg = QLabel("—")
        self.lbl_cfg.setObjectName("CardHint")
        hl.addSpacing(8)
        hl.addWidget(self.lbl_cfg)
        hl.addStretch(1)
        return bar

    # ---------- 训练与参数面板（可停靠 / 可隐藏，不挤压可视化区域） ----------
    def _make_param_widget(self, spec):
        key, _label, kind, default, rng, tip = spec
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
        elif kind == "int":
            lo, hi, step = rng
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setSingleStep(int(step))
            w.setValue(int(default))
        elif kind == "float":
            lo, hi, step = rng
            w = QDoubleSpinBox()
            w.setRange(float(lo), float(hi))
            w.setSingleStep(float(step))
            w.setDecimals(5 if step < 0.001 else (3 if step < 0.1 else 2))
            w.setValue(float(default))
        else:
            w = QComboBox()
            w.addItems([str(o) for o in rng])
            if key == "decode_mode":
                w.setCurrentIndex(1 if int(default) == 2 else 0)
            elif default in rng:
                w.setCurrentIndex(rng.index(default))
        w.setToolTip(tip)
        return w

    def _build_settings_dock(self):
        dock = QDockWidget("训练与参数", self)
        dock.setObjectName("SettingsDock")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.RightDockWidgetArea)
        tabs = QTabWidget()
        for group, specs in TRAIN_SPECS:
            page = QWidget()
            page.setObjectName("SettingsPage")
            form = QFormLayout(page)
            form.setContentsMargins(10, 10, 10, 10)
            form.setSpacing(7)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for spec in specs:
                w = self._make_param_widget(spec)
                self.controls[spec[0]] = w
                lab = QLabel(spec[1])
                lab.setToolTip(spec[5])
                form.addRow(lab, w)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.viewport().setObjectName("SettingsViewport")
            scroll.setWidget(page)
            tabs.addTab(scroll, group)
        btn_apply = QPushButton("应用设置")
        btn_apply.setObjectName("Primary")
        btn_apply.setToolTip("立即写入后端；连接数等规模参数需再点「应用配置」重建网络")
        btn_apply.clicked.connect(self._apply_settings)
        btn_default = QPushButton("恢复默认")
        btn_default.clicked.connect(self._restore_defaults)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 6, 8, 8)
        h.setSpacing(8)
        h.addWidget(btn_default)
        h.addWidget(btn_apply, 1)
        box = QWidget()
        box.setObjectName("SettingsBody")
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(tabs, 1)
        v.insertWidget(1, self._build_teacher_card())
        v.addWidget(row)
        dock.setWidget(box)
        dock.setMinimumWidth(340)
        self.settings_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.hide()          # 默认收起，避免挤压网络/机械臂视图；P 或「设置面板」展开
        self._apply_settings(silent=True)

    def _build_teacher_card(self):
        """蒸馏老师（学习接口）设置卡片：选老师 + 引导比例 G + 逐步放手。"""
        card = QWidget()
        card.setObjectName("TeacherCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)
        ttl = QLabel("蒸馏老师（学习接口）")
        ttl.setObjectName("CardTitle")
        v.addWidget(ttl)

        self.combo_teacher = QComboBox()
        self.combo_teacher.addItems([
            "无老师（纯脉冲网络）", "DLS 逆运动学（现成解析解）",
            "转置雅可比（另一种 IK）", "无老师基线"])
        self.combo_teacher.setMinimumWidth(180)
        self.combo_teacher.currentIndexChanged.connect(self._on_teacher)
        v.addWidget(self.combo_teacher)

        gh = QWidget()
        ghb = QHBoxLayout(gh)
        ghb.setContentsMargins(0, 0, 0, 0)
        ghb.setSpacing(8)
        ghb.addWidget(QLabel("引导比例 G"))
        self.spin_teacher_g = QDoubleSpinBox()
        self.spin_teacher_g.setRange(0.0, 1.0)
        self.spin_teacher_g.setSingleStep(0.05)
        self.spin_teacher_g.setValue(0.7)
        self.spin_teacher_g.setDecimals(2)
        self.spin_teacher_g.valueChanged.connect(self._on_teacher_g)
        ghb.addWidget(self.spin_teacher_g, 1)
        v.addWidget(gh)

        self.chk_anneal = QCheckBox("逐步放手（G 退火 · 先学再用）")
        self.chk_anneal.setToolTip("勾选后每碰到一次目标 G 乘以 0.97，网络逐步接管控制权")
        self.chk_anneal.toggled.connect(self._on_teacher_anneal)
        v.addWidget(self.chk_anneal)

        tip = QLabel("先学（蒸馏老师）→ 再用（G=0 纯自主）→ 再创新。引导需在向量化"
                     "后端承载：选老师会自动从 C DLL 切过去。")
        tip.setObjectName("MuteSmall")
        tip.setWordWrap(True)
        v.addWidget(tip)
        return card

    def _param_values(self):
        vals = {}
        for key, w in self.controls.items():
            if isinstance(w, QCheckBox):
                vals[key] = bool(w.isChecked())
            elif isinstance(w, QComboBox):
                if key == "decode_mode":
                    vals[key] = 2 if w.currentIndex() == 1 else 0
                else:
                    vals[key] = w.currentText()
            else:
                vals[key] = float(w.value())
        return vals

    def _apply_settings(self, silent=False):
        vals = self._param_values()
        self.engine.train_cfg.update(vals)
        failed = []
        if self.engine.initialized:
            failed = self.engine.set_params({k: v for k, v in vals.items()
                                             if k in CPARAM_KEYS})
        self._update_trial_label()
        if not silent:
            if failed:
                self.lbl_cfg.setText("当前后端不支持：" + "、".join(failed))
            else:
                self.lbl_cfg.setText("设置已应用 · 在线生效（连接数需重建网络）")
        return failed

    def _restore_defaults(self):
        for _, specs in TRAIN_SPECS:
            for key, _label, kind, default, rng, _tip in specs:
                w = self.controls.get(key)
                if w is None:
                    continue
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(default))
                elif isinstance(w, QComboBox):
                    if key == "decode_mode":
                        w.setCurrentIndex(1 if int(default) == 2 else 0)
                    elif default in rng:
                        w.setCurrentIndex(rng.index(default))
                else:
                    w.setValue(default)
        self._apply_settings(silent=True)
        self.lbl_cfg.setText("已恢复默认设置")

    def _toggle_settings(self):
        self.settings_dock.setVisible(not self.settings_dock.isVisible())

    # ---------- 蒸馏老师（学习接口） ----------
    def _on_teacher(self, idx):
        if not hasattr(self, "combo_dev"):
            return   # 构建阶段 combo_dev 尚未创建，跳过
        key = TEACHER_KEYS[max(0, idx)]
        if key == "none":
            self.engine._teacher = None
            self._teacher_status("老师：无")
            if self._teacher_forced_vec:
                self._teacher_forced_vec = False
                self.combo_dev.setCurrentIndex(DEFAULT_DEVICE_INDEX)  # 触发 _reset
            return
        self.engine._teacher = TEACHERS[key]()
        self.engine._teacher_g = float(self.spin_teacher_g.value())
        self.engine._teacher_anneal = self.chk_anneal.isChecked()
        self._teacher_forced_vec = False
        # 引导钩子只在向量化后端生效：选老师且当前是 C DLL 时自动切过去
        if DEVICES[self.combo_dev.currentIndex()][1] == "cdll":
            self._teacher_forced_vec = True
            idx_vec = 0 if DEVICES[0][1] == "cuda" else 1
            self.combo_dev.setCurrentIndex(idx_vec)   # 触发 _reset（engine 复用，保留 _teacher）
            self._teacher_status("老师：%s · G=%.2f（已切向量化后端）" %
                                 (self.engine._teacher.name, self.engine._teacher_g))
        else:
            self._teacher_status("老师：%s · G=%.2f" %
                                 (self.engine._teacher.name, self.engine._teacher_g))

    def _on_teacher_g(self, v):
        self.engine._teacher_g = float(v)

    def _on_teacher_anneal(self, v):
        self.engine._teacher_anneal = bool(v)

    def _teacher_status(self, text):
        if hasattr(self, "lbl_teacher_status"):
            self.lbl_teacher_status.setText(text)

    def _update_trial_label(self):
        if not hasattr(self, "lbl_trial"):
            return
        e = self.engine
        horizon = max(1, int(e.train_cfg["score_horizon"]))
        state = ("训练中" if e.train_cfg["train_enabled"] and e.training_supported
                 else "仅仿真")
        self.lbl_trial.setText(
            f"{state} · 第 {e.episode} 轮 · 到达 {e.arrivals} · "
            f"本轮 {e.trial_tick} tick（评分参考 {horizon}） · "
            f"上轮 {e.last_episode_ticks} tick · 到达奖励 {e.last_arrival_reward:+.2f}")

    def _build_footer(self):
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(8)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        self.metrics = {}
        for key, label in (("step", "仿真步数"), ("spike", "发放神经元"),
                          ("dist", "末端距离"), ("ratio", "距离 / 臂长"),
                          ("reward", "即时奖励"), ("total", "累计奖励"),
                          ("episode", "完成轮次"), ("trial", "本轮步数")):
            m = MetricCard(label)
            h.addWidget(m, 1)
            self.metrics[key] = m
        lay.addWidget(row)
        line = QFrame()
        line.setObjectName("HLine")
        line.setFixedHeight(1)
        lay.addWidget(line)
        row2 = QWidget()
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.setSpacing(8)
        self.strip = GradientStrip()
        for text in ("膜电位", "−65 静息"):
            q = QLabel(text)
            q.setObjectName("CardHint")
            h2.addWidget(q)
        h2.addWidget(self.strip)
        for text in ("−50 mV 阈值", "金色光圈 = 发放脉冲",
                     "连线粗细 = 突触权重"):
            q = QLabel(text)
            q.setObjectName("CardHint")
            h2.addSpacing(5)
            h2.addWidget(q)
        h2.addStretch(1)
        q = QLabel("空格 运行/暂停 · S 单步 · R 重置 · T 随机 · F 适应窗口 · P 设置面板")
        q.setObjectName("Hint")
        h2.addWidget(q)
        lay.addWidget(row2)
        return card

    # ---------- 主题 ----------
    def _toggle_theme(self):
        self.theme_key = "light" if self.theme_key == "dark" else "dark"
        self.T = THEMES[self.theme_key]
        QApplication.instance().setStyleSheet(build_qss(self.T))
        self.net_view.apply_theme(self.T)
        self.arm_view.apply_theme(self.T)
        self.chart.set_theme(self.T)
        self.btn_theme.setText("浅色" if self.theme_key == "dark" else "深色")
        self._refresh_badge()
        self._refresh_frame(self._last_snap)

    def _refresh_badge(self):
        T = self.T
        text, color = ("运行中", T["ok"]) if self.is_playing else ("已暂停", T["dim"])
        if self._error:
            text, color = "后端异常", T["danger"]
        self.badge.setText(text)
        self.badge.setStyleSheet(
            f"color:{color}; background:{T['panel_alt']};"
            f" border:1px solid {T['border']}; border-radius:14px;"
            f" padding:6px 14px; font-weight:600;")

    # ---------- 控制 ----------
    def _current_cfg(self):
        idx = max(0, self.combo_preset.currentIndex())
        return PRESETS[idx][1:]

    def _reset(self):
        if self._initializing:
            return
        self._pause()
        self._error = ""
        cfg = self._current_cfg()
        dev = DEVICES[max(0, self.combo_dev.currentIndex())][1]
        seed = (self.spin_seed.value() if self.chk_seed.isChecked()
                else random.randint(0, 999999))
        avg_conn = (float(self.controls["avg_conn"].value())
                    if "avg_conn" in self.controls else 8.0)
        try:
            topo = self.engine.reset(dev, cfg, seed, avg_conn=avg_conn)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self._refresh_badge()
            self.lbl_cfg.setText(f"后端初始化失败：{self._error}")
            return
        self.net_view.load_topology(topo)
        self.chart.clear()
        self.arm_view.clearTrail()
        self._last_target = None
        self._apply_settings(silent=True)
        if self.engine.train_cfg.get("auto_target"):
            self._random_target()
        else:
            self._set_target()
        self._total_reward = 0.0
        self._last_snap = self.engine.snapshot()
        self.lbl_cfg.setText(
            f"{topo['neuron_count']} 神经元 · {topo['synapse_count']} 突触 · 种子 {seed}")
        self.net_hint.setText(
            f"{topo['neuron_count']} 神经元 · {topo['synapse_count']} 突触 · 种子 {seed}")
        self.chip_dev.setText(f"后端 {self.engine.name}")
        self._refresh_badge()
        self._refresh_frame(self._last_snap)

    def _toggle_play(self):
        self._pause() if self.is_playing else self._play()

    def _play(self):
        if not self.engine.initialized or self._error:
            return
        self.is_playing = True
        self.btn_play.setText("暂停")
        self.worker = SimWorker(self.engine)
        self.worker.set_steps(max(1, int(self._speed)))
        self.worker.failed.connect(self._on_worker_error)
        self.worker.start()
        self.gui_timer.start()
        self._refresh_badge()

    def _pause(self):
        self.is_playing = False
        self.btn_play.setText("运行")
        self.gui_timer.stop()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        if hasattr(self, "badge"):
            self._refresh_badge()

    def _on_worker_error(self, msg):
        self._error = msg
        self.lbl_cfg.setText(f"后端异常：{msg}")
        self._pause()
        self._refresh_badge()

    def _on_speed(self, idx):
        self._speed = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0][idx]
        if self.worker is not None:
            self.worker.set_steps(max(1, int(self._speed)))

    def _step_once(self):
        if self.is_playing or not self.engine.initialized:
            return
        self.engine.train_tick()
        self._last_snap = self.engine.snapshot()
        self._refresh_frame(self._last_snap)

    def _set_target(self):
        tx, ty = self.spin_tx.value(), self.spin_ty.value()
        if self.engine.initialized:
            self.engine.set_target(tx, ty)
        self.arm_view.setTarget(tx, ty)
        self._last_target = (tx, ty)

    def _random_target(self):
        r = 0.4 + random.random() * 1.1
        a = random.random() * math.pi * 2
        self.spin_tx.setValue(round(r * math.cos(a), 2))
        self.spin_ty.setValue(round(r * math.sin(a), 2))
        self._set_target()

    def _on_selection(self, nid):
        if nid < 0:
            self.net_hint.setText(
                f"{self.net_view.N} 神经元 · {len(self.net_view._syn_pre)} 突触")
        else:
            v = self.net_view
            self.net_hint.setText(
                f"已选中 #{nid} · {LAYER_NAMES[int(v.layer_of[nid])]} · "
                f"入 {int(v.in_deg[nid])} / 出 {int(v.out_deg[nid])} · 再次点击取消")

    # ---------- 渲染循环 ----------
    def _render_frame(self):
        if self.worker is None:
            return
        snap = self.worker.latest()
        if snap is None:
            return
        t0 = time.perf_counter()
        self._last_snap = snap
        self._refresh_frame(snap)
        dt = (time.perf_counter() - t0) * 1000
        self._render_ms = dt if self._render_ms == 0 else self._render_ms * 0.8 + dt * 0.2
        fps = 1000.0 / max(self.gui_timer.interval(), dt)
        self._fps = fps if self._fps == 0 else self._fps * 0.8 + fps * 0.2
        rate = self.worker.tick_rate if self.worker else 0.0
        self.lbl_perf.setText(
            f"{self._fps:.0f} FPS · {rate:,.0f} tick/s · {self._render_ms:.2f} ms/帧")

    def _refresh_frame(self, snap):
        if snap is None:
            return
        mem, spk, arm, meta = snap[:4]
        trail = snap[4] if len(snap) > 4 else None
        self.net_view.update_snapshot(mem, spk)
        t1, t2 = float(arm[0]), float(arm[1])
        tx, ty = float(arm[6]), float(arm[7])
        ex, ey = float(arm[4]), float(arm[5])
        mr = float(arm[9])
        self.arm_view.update_arm(t1, t2, float(arm[2]), float(arm[3]), ex, ey, mr)
        # 新一轮开始：清空轨迹，只显示当前轮运动（去掉上一轮残影/瞬移）
        if self.engine.episode != self._last_episode:
            self.arm_view.clearTrail()
            self._last_episode = self.engine.episode
        self.arm_view.append_trail(trail if trail else [(ex, ey)])
        if (tx, ty) != self._last_target:
            self.arm_view.setTarget(tx, ty)
            self._last_target = (tx, ty)
        sp = int(np.count_nonzero(spk))
        dist = math.hypot(ex - tx, ey - ty)
        ratio = dist / mr if mr > 0 else 0.0
        r = (float(self.engine.last_reward)
             if self.engine.train_cfg["train_enabled"] else float(arm[8]))
        self._total_reward = self.engine.total_reward
        T = self.T
        self.metrics["step"].set_value(f"{int(meta[0]):,}")
        self.metrics["spike"].set_value(f"{sp} / {len(mem)}")
        self.metrics["dist"].set_value(f"{dist:.3f}")
        self.metrics["ratio"].set_value(f"{ratio:.3f}")
        self.metrics["reward"].set_value(f"{r:+.4f}",
                                        T["ok"] if r >= 0 else T["warn"])
        self.metrics["total"].set_value(f"{self._total_reward:+.1f}")
        self.metrics["episode"].set_value(f"{self.engine.episode}")
        self.metrics["trial"].set_value(f"{self.engine.trial_tick}")
        if self.engine._teacher is not None:
            self.lbl_teacher_status.setText(
                "老师：%s · G=%.2f%s" % (
                    self.engine._teacher.name, self.engine._teacher_g,
                    " · 退火中" if self.engine._teacher_anneal else ""))
        self.chart.push(r, dist, sp / len(mem) if len(mem) else 0.0)
        self._update_trial_label()

    def closeEvent(self, event):
        self.gui_timer.stop()
        self._pause()
        self.engine.destroy()
        super().closeEvent(event)


# ============================================================
# 入口
# ============================================================
def find_dll():
    for p in (os.path.join(ROOT_DIR, "b2_core.dll"),
              os.path.join(ROOT_DIR, "Debug", "b2_core.dll")):
        if os.path.exists(p):
            return p
    return None


def main():
    global FONT_FAMILY
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ensure_fonts()
    FONT_FAMILY = pick_font_family()
    app.setStyleSheet(build_qss(THEMES["dark"], FONT_FAMILY))
    dll = find_dll()
    if not dll:
        QMessageBox.critical(None, "缺少核心库",
                             "找不到 b2_core.dll。\n请先编译 C 核心后再启动可视化。")
        return 1
    win = MainWindow(SimEngine(dll))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
