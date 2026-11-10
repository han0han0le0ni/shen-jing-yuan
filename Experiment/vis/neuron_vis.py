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
from backend_vec import VecBackend

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
QComboBox::drop-down { border: 0px; width: 18px; }
QComboBox QAbstractItemView { background: $panel; color: $text;
    border: 1px solid $border; selection-background-color: $accent; }
QCheckBox { color: $sub; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid $border;
    border-radius: 4px; background: $panel_alt; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; }
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
LAYER_NAMES = ["输入层 · 位置编码", "隐藏层 · 脉冲整合",
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
           ("C DLL · 原生", "cdll")]


# ============================================================
# 后端封装
# ============================================================
class SimEngine:
    """统一三种后端；安全销毁、统一拓扑/快照接口"""

    def __init__(self, dll_path):
        self.dll = dll_path
        self.be = None
        self.initialized = False
        self.name = "—"
        self.on_gpu = False

    def reset(self, device_key, cfg, seed, avg_conn=8.0):
        self.destroy()
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
        return be.get_topology()

    def tick(self):
        self.be.tick()

    def snapshot(self):
        # 统一成 (mem, spk, arm, meta)，丢掉用不到的 last_spike
        mem, spk, last, arm, meta = self.be.get_snapshot_arrays()
        return mem, spk, arm, meta

    def set_target(self, x, y):
        self.be.set_target(x, y)

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
                    self.engine.tick()
                mem, spk, arm, meta = self.engine.snapshot()
                snap = (np.ascontiguousarray(mem, dtype=np.float32),
                        np.ascontiguousarray(spk, dtype=np.int32),
                        np.ascontiguousarray(arm, dtype=np.float64),
                        np.ascontiguousarray(meta, dtype=np.int32))
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
        self.trail = deque(maxlen=240)
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
        if len(self.trail) == self.trail.maxlen:
            self.trail.popleft()
            self.trail.append((bx, by))
            self.trail_path = QPainterPath()
            pts = list(self.trail)
            self.trail_path.moveTo(*pts[0])
            for px, py in pts[1:]:
                self.trail_path.lineTo(px, py)
        else:
            if self.trail:
                self.trail_path.lineTo(bx, by)
            else:
                self.trail_path.moveTo(bx, by)
            self.trail.append((bx, by))
        self.trail_item.setPath(self.trail_path)

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
        self.setWindowTitle("脉冲神经网络 · 实时可视化  |  Benchmark-2 · CUDA")
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 760)

        root = QWidget()
        root.setObjectName("Root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        self.setCentralWidget(root)

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
        lay.addWidget(main, 1)
        lay.addWidget(self._build_footer())

        for key, fn in (("Space", self._toggle_play), ("S", self._step_once),
                        ("R", self._reset), ("T", self._random_target),
                        ("F", self.net_view.reset_view)):
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
        t = QLabel("脉冲神经网络 · CUDA 实时可视化")
        t.setObjectName("Title")
        s = QLabel("Benchmark-2   |   QThread + CuPy 融合核   |   大规模脉冲网络交互分析")
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
        self.combo_dev.setMinimumWidth(150)
        self.combo_dev.currentIndexChanged.connect(lambda _: self._reset())
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
                          ("reward", "即时奖励"), ("total", "累计奖励")):
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
        q = QLabel("空格 运行/暂停 · S 单步 · R 重置 · T 随机 · F 适应窗口")
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
        self._pause()
        self._error = ""
        cfg = self._current_cfg()
        dev = DEVICES[max(0, self.combo_dev.currentIndex())][1]
        seed = (self.spin_seed.value() if self.chk_seed.isChecked()
                else random.randint(0, 999999))
        try:
            topo = self.engine.reset(dev, cfg, seed)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self._refresh_badge()
            self.lbl_cfg.setText(f"后端初始化失败：{self._error}")
            return
        self.net_view.load_topology(topo)
        self.chart.clear()
        self.arm_view.clearTrail()
        self._last_target = None
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
        self.engine.tick()
        self._last_snap = self.engine.snapshot()
        self._refresh_frame(self._last_snap)

    def _set_target(self):
        tx, ty = self.spin_tx.value(), self.spin_ty.value()
        if self.engine.initialized:
            self.engine.set_target(tx, ty)
        self.arm_view.setTarget(tx, ty)
        self._last_target = (tx, ty)

    def _random_target(self):
        r = 0.3 + random.random() * 1.2
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
        mem, spk, arm, meta = snap
        self.net_view.update_snapshot(mem, spk)
        t1, t2 = float(arm[0]), float(arm[1])
        tx, ty = float(arm[6]), float(arm[7])
        ex, ey = float(arm[4]), float(arm[5])
        mr = float(arm[9])
        self.arm_view.update_arm(t1, t2, float(arm[2]), float(arm[3]), ex, ey, mr)
        if (tx, ty) != self._last_target:
            self.arm_view.setTarget(tx, ty)
            self._last_target = (tx, ty)
        sp = int(np.count_nonzero(spk))
        dist = math.hypot(ex - tx, ey - ty)
        ratio = dist / mr if mr > 0 else 0.0
        r = float(arm[8])
        self._total_reward += r
        T = self.T
        self.metrics["step"].set_value(f"{int(meta[0]):,}")
        self.metrics["spike"].set_value(f"{sp} / {len(mem)}")
        self.metrics["dist"].set_value(f"{dist:.3f}")
        self.metrics["ratio"].set_value(f"{ratio:.3f}")
        self.metrics["reward"].set_value(f"{r:+.4f}",
                                        T["ok"] if r >= 0 else T["warn"])
        self.metrics["total"].set_value(f"{self._total_reward:+.1f}")
        self.chart.push(r, dist, sp / len(mem) if len(mem) else 0.0)

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
