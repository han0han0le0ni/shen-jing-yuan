"""引导监控面板：实时显示老师指令 / 网络解码 / 最终注入 / 末端距离。

参考 transfer_lab 的 Chart 与「解析引导 X% / 网络自主 Y%」分割条，
但直接读主窗口 SimEngine 的 guidance / guidance_hist，不额外跑仿真。

独立自检： py -3 vis/guidance_monitor.py --check
主窗口：    工具栏「引导监控」
"""
import math
import os
import sys
import tempfile
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

DEFAULT_THEME = {
    "bg": "#0b0e14", "panel": "#0f131c", "panel_alt": "#141b28",
    "chart": "#0c111b", "border": "#1f2636", "grid": "#20293a",
    "text": "#dfe7f5", "sub": "#8a9ab3", "dim": "#5c6a83",
    "accent": "#22d3ee", "purple": "#a78bfa", "warn": "#ffbf47",
    "ok": "#34d399", "danger": "#fb7185", "btn": "#182033",
    "btn_hover": "#22304a",
}


def build_qss(T):
    return """
    QWidget { font-family: "Microsoft YaHei","Segoe UI",sans-serif;
              font-size: 12px; color: %s; }
    QMainWindow, #Root { background: %s; }
    QFrame#Card { background: %s; border: 1px solid %s; border-radius: 10px; }
    QLabel#Mute { color: %s; }
    QDoubleSpinBox { background: %s; color: %s; border: 1px solid %s;
                     border-radius: 6px; padding: 4px 8px; }
    """ % (T["text"], T["bg"], T["panel"], T["border"], T["sub"],
           T["panel_alt"], T["text"], T["border"])


class MiniChart(QWidget):
    """多序列折线图（纯 QPainter）。series: [(name, color), ...]"""

    def __init__(self, title, series, ymin, ymax, T, unit=""):
        super().__init__()
        self.title, self.series = title, series
        self.ymin, self.ymax, self.unit = ymin, ymax, unit
        self.T = T
        self.data = {}
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def set_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        T = self.T
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(T["chart"]))
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(T["text"]))
        p.drawText(10, 16, self.title)

        # 标题在第 1 行、图例在第 2 行，绘图区从第 3 行开始（避免文字重叠）
        m = QRectF(48, 46, max(10, w - 62), max(10, h - 70))
        for k in range(5):
            y = m.top() + (m.height()) * k / 4.0
            v = self.ymax - (self.ymax - self.ymin) * k / 4.0
            p.setPen(QPen(QColor(T["grid"]), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(m.left()), int(y), int(m.right()), int(y))
            p.setPen(QColor(T["dim"]))
            p.drawText(QRectF(2, y - 9, 42, 18),
                       Qt.AlignmentFlag.AlignRight, "%.2f" % v)

        n = max(1, max((len(v) for v in self.data.values()), default=1) - 1)
        for name, color in self.series:
            pts = self.data.get(name, [])
            if len(pts) < 2:
                continue
            col = QColor(color)
            path = QPainterPath()
            for i, (_x, y) in enumerate(pts):
                px = m.left() + (i / max(1, len(pts) - 1)) * m.width()
                yy = max(self.ymin, min(self.ymax, y))
                py = m.bottom() - (yy - self.ymin) / (self.ymax - self.ymin) * m.height()
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            p.setBrush(QBrush())       # 开放折线不能带画刷
            p.setPen(QPen(col, 2.0))
            p.drawPath(path)

        lx = m.left() + 8
        for name, color in self.series:
            p.setBrush(QBrush(QColor(color)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(lx, 32, 12, 4))
            p.setPen(QColor(T["sub"]))
            p.drawText(QRectF(lx + 17, 24, 130, 18),
                       Qt.AlignmentFlag.AlignLeft, name)
            lx += 120
        p.setPen(QColor(T["dim"]))
        p.drawText(QRectF(m.right() - 170, h - 16, 168, 14),
                   Qt.AlignmentFlag.AlignRight, "最近 %d tick · %s" % (n, self.unit))


def value_card(T, title, value, color):
    f = QFrame()
    f.setObjectName("Card")
    v = QVBoxLayout(f)
    v.setContentsMargins(10, 8, 10, 8)
    v.setSpacing(2)
    a = QLabel(title)
    a.setStyleSheet("color:%s;" % T["sub"])
    b = QLabel(value)
    b.setStyleSheet("color:%s; font-size:18px; font-weight:600;" % color)
    v.addWidget(a)
    v.addWidget(b)
    return f


class GuidanceWindow(QMainWindow):
    def __init__(self, engine, theme=None, set_g_callback=None):
        super().__init__()
        self.engine = engine
        self.T = theme or DEFAULT_THEME
        self.set_g_callback = set_g_callback
        self.setWindowTitle("引导监控 · 老师 / 网络 / 最终指令")
        self.resize(1060, 820)
        self.setMinimumSize(900, 700)
        self.setStyleSheet(build_qss(self.T))

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        head = QHBoxLayout()
        box = QVBoxLayout()
        t = QLabel("引导监控 · 实时指令分解")
        t.setStyleSheet("font-size:19px; font-weight:600;")
        s = QLabel("老师指令 ω → 与网络解码按 G 混合 → 注入 motor_override")
        s.setObjectName("Mute")
        box.addWidget(t)
        box.addWidget(s)
        head.addLayout(box)
        head.addStretch(1)
        self.badge = QLabel("等待数据")
        head.addWidget(self.badge)
        lay.addLayout(head)

        split = QFrame()
        split.setObjectName("Card")
        sh = QHBoxLayout(split)
        sh.setContentsMargins(14, 10, 14, 10)
        self.lbl_split = QLabel("解析引导 0% / 网络自主 100%")
        self.lbl_split.setStyleSheet("font-size:15px; font-weight:600; color:%s;"
                                     % self.T["accent"])
        sh.addWidget(self.lbl_split)
        sh.addStretch(1)
        sh.addWidget(QLabel("当前 G（1=全引导，0=全自主）"))
        self.spin_g = QDoubleSpinBox()
        self.spin_g.setRange(0.0, 1.0)
        self.spin_g.setSingleStep(0.05)
        self.spin_g.setDecimals(2)
        self.spin_g.setFixedWidth(90)
        self.spin_g.setValue(float(getattr(engine, "_teacher_g", 0.0)))
        self.spin_g.valueChanged.connect(self._on_g)
        sh.addWidget(self.spin_g)
        lay.addWidget(split)

        self.cards = QGridLayout()
        self.cards.setSpacing(8)
        lay.addLayout(self.cards)

        self.chart_omega = MiniChart(
            "关节 1 角速度 ω₁（网络 / 老师 / 最终注入）",
            [("网络", self.T["purple"]), ("最终", self.T["accent"]),
             ("老师", self.T["warn"])],
            -0.6, 0.6, self.T, unit="rad/s")
        self.chart_dist = MiniChart(
            "末端到目标距离",
            [("距离", self.T["ok"])], 0.0, 3.0, self.T, unit="arm length")
        self.chart_g = MiniChart(
            "引导依赖 G（老师占比 · 退火时应逐轮下降）",
            [("G", self.T["warn"])], 0.0, 1.0, self.T, unit="ratio")
        lay.addWidget(self.chart_omega, 1)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.chart_dist, 1)
        row.addWidget(self.chart_g, 1)
        lay.addLayout(row, 1)

        self.foot = QLabel("")
        self.foot.setObjectName("Mute")
        self.foot.setWordWrap(True)
        lay.addWidget(self.foot)

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def _on_g(self, v):
        self.engine._teacher_g = float(v)
        if self.set_g_callback:
            self.set_g_callback(float(v))

    def refresh(self):
        T = self.T
        rows = list(getattr(self.engine, "guidance_hist", ()) or ())
        cur = getattr(self.engine, "guidance", {}) or {}

        if not self.spin_g.hasFocus():
            self.spin_g.blockSignals(True)
            self.spin_g.setValue(float(getattr(self.engine, "_teacher_g",
                                               cur.get("g", 0.0))))
            self.spin_g.blockSignals(False)

        g = float(cur.get("g", getattr(self.engine, "_teacher_g", 0.0)))
        self.lbl_split.setText("解析引导 %d%%  /  网络自主 %d%%"
                               % (round(g * 100), round((1 - g) * 100)))

        eff = bool(cur.get("effective", False))
        if not rows:
            badge, color = "等待运行…", T["dim"]
        elif eff:
            badge = "引导生效中 · %s · %s" % (
                cur.get("backend", "?"), cur.get("teacher", ""))
            color = T["ok"]
        else:
            badge = "引导未生效 · " + (cur.get("reason") or "未知原因")
            color = T["danger"]
        self.badge.setText(badge)
        self.badge.setStyleSheet(
            "background:%s; color:%s; border-radius:8px; padding:7px 12px; "
            "font-weight:600;" % (T["panel_alt"], color))

        net = cur.get("net", (0.0, 0.0))
        cmd = cur.get("cmd", (0.0, 0.0))
        final = cur.get("final", (0.0, 0.0))
        dist = float(cur.get("dist", 0.0))
        specs = [
            ("老师 ω₁", "%+.3f" % cmd[0], T["warn"]),
            ("老师 ω₂", "%+.3f" % cmd[1], T["warn"]),
            ("网络 ω₁", "%+.3f" % net[0], T["purple"]),
            ("网络 ω₂", "%+.3f" % net[1], T["purple"]),
            ("注入 ω₁", "%+.3f" % final[0], T["accent"]),
            ("注入 ω₂", "%+.3f" % final[1], T["accent"]),
            ("末端距离", "%.3f" % dist, T["ok"]),
            ("tick", str(cur.get("tick", 0)), T["text"]),
        ]
        while self.cards.count():
            it = self.cards.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for i, (title, val, color) in enumerate(specs):
            self.cards.addWidget(value_card(T, title, val, color), i // 4, i % 4)

        self.chart_omega.set_data({
            "网络": [(i, r["net"][0]) for i, r in enumerate(rows)],
            "老师": [(i, r["cmd"][0]) for i, r in enumerate(rows)],
            "最终": [(i, r["final"][0]) for i, r in enumerate(rows)],
        })
        self.chart_dist.set_data({
            "距离": [(i, r["dist"]) for i, r in enumerate(rows)],
        })
        self.chart_g.set_data({
            "G": [(i, r.get("g", 0.0)) for i, r in enumerate(rows)],
        })
        self.foot.setText(
            "生效条件：选择学习接口 + 「向量化 · 12 核」后端（CUDA 的 _k_arm kernel "
            "不消费 motor_override，C DLL 无注入钩子）。G=1 时注入指令应等于老师指令；"
            "G=0 时等于网络解码。G 曲线每次碰到目标才降一档（×0.97，下限 0.02），"
            "碰不到目标就不变。数据每 100 ms 刷新，保留最近 %d tick。" % len(rows))


class _FakeEngine:
    """--check 用的模拟数据源，不依赖仿真后端。"""

    def __init__(self):
        self._teacher_g = 0.7
        self.guidance_hist = deque(maxlen=360)
        for i in range(240):
            cmd = (0.48 * math.sin(i / 30.0), 0.40 * math.cos(i / 25.0))
            net = (0.12 * math.sin(i / 17.0), 0.10 * math.cos(i / 13.0))
            final = (0.3 * net[0] + 0.7 * cmd[0], 0.3 * net[1] + 0.7 * cmd[1])
            self.guidance_hist.append({
                "tick": i, "g": 0.7, "teacher": "DLS 逆运动学（自检数据）",
                "net": net, "cmd": cmd, "final": final,
                "dist": max(0.05, 1.9 - i * 0.006),
                "effective": True, "reason": "", "backend": "向量化·12核",
            })
        self.guidance = self.guidance_hist[-1]


def main():
    check = "--check" in sys.argv
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))
    w = GuidanceWindow(_FakeEngine() if check else None)
    if w.engine is None:
        print("usage: --check (standalone self test), or open from neuron_vis")
        return 1
    w.show()
    if check:
        out = os.path.join(tempfile.gettempdir(), "guidance_monitor_check.png")
        QTimer.singleShot(900, lambda: (w.grab().save(out), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
