"""Benchmark-1 成果窗口：读取 data/b1_m1/*.jsonl，展示延迟扫描、消融与判据。

可独立运行：  py -3 vis/results_panel.py
离屏自检：    py -3 vis/results_panel.py --check
也可从 neuron_vis.py 的「基准成果」按钮打开（共用主题配色）。
"""
import glob
import json
import os
import sys
import tempfile

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QAbstractItemView,
)
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "b1_m1")


def load_summaries():
    """tag -> summary record（只认 summary 事件，忽略 trial 行）。"""
    out = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl"))):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") == "summary" and rec.get("tag"):
                        out[rec["tag"]] = rec
        except OSError:
            continue
    return out


def sweep_curve(rows, cond):
    pts = []
    prefix = "sweep_%s_D" % cond
    for tag, rec in rows.items():
        if not tag.startswith(prefix):
            continue
        try:
            d = int(tag.rsplit("_D", 1)[1])
        except (ValueError, IndexError):
            continue
        pts.append((d, float(rec.get("test_acc", 0.0))))
    return sorted(pts)


def d_half(points):
    """正确率首次跌破 50% 的延迟（线性插值）。"""
    prev = None
    for d, acc in sorted(points):
        if acc < 50.0:
            if prev is None:
                return 0.0
            d0, a0 = prev
            if a0 <= acc:
                return float(d0)
            return d0 + (a0 - 50.0) / (a0 - acc) * (d - d0)
        prev = (d, acc)
    return None


def build_qss(T):
    return """
    QWidget { font-family: "Microsoft YaHei","Segoe UI",sans-serif;
              font-size: 12px; color: %s; }
    QMainWindow, #Root { background: %s; }
    QFrame#Card { background: %s; border: 1px solid %s; border-radius: 10px; }
    QPushButton { background: %s; color: %s; border: 1px solid %s;
                  border-radius: 6px; padding: 6px 14px; }
    QPushButton:hover { background: %s; }
    QTableWidget { background: %s; color: %s; border: 1px solid %s;
                   gridline-color: %s; }
    QHeaderView::section { background: %s; color: %s; border: none; padding: 5px; }
    QTableWidget::item { background: transparent; color: %s; padding: 3px; }
    QTableCornerButton::section { background: %s; border: none; }
    """ % (T["text"], T["bg"], T["panel"], T["border"], T["btn"], T["text"],
           T["border"], T["btn_hover"], T["panel_alt"], T["text"],
           T["border"], T["grid"], T["panel"], T["sub"],
           T["text"], T["panel"])


class LineChart(QWidget):
    """延迟-正确率曲线，纯 QPainter 绘制，无第三方图表依赖。"""

    def __init__(self, T):
        super().__init__()
        self.T = T
        self.series = []
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_series(self, series):
        self.series = series
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        T = self.T
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(T["chart"]))
        m = QRectF(58, 44, max(10, w - 78), max(10, h - 86))
        if not self.series:
            p.setPen(QColor(T["dim"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "暂无扫描数据 · 运行 tools/b1_sweep.ps1")
            return

        # 网格与 y 轴刻度
        p.setFont(QFont("Segoe UI", 9))
        for v in (0, 20, 40, 60, 80, 100):
            y = m.bottom() - v / 100.0 * m.height()
            p.setPen(QPen(QColor(T["grid"]), 1))
            p.drawLine(int(m.left()), int(y), int(m.right()), int(y))
            p.setPen(QColor(T["dim"]))
            p.drawText(QRectF(4, y - 9, 48, 18), Qt.AlignmentFlag.AlignRight, "%d%%" % v)

        all_x = sorted({d for _n, _c, pts in self.series for d, _a in pts})
        if not all_x:
            return
        xpos = {d: m.left() + (i / max(1, len(all_x) - 1)) * m.width()
                for i, d in enumerate(all_x)}

        for d, x in xpos.items():
            p.setPen(QPen(QColor(T["grid"]), 1, Qt.PenStyle.DotLine))
            p.drawLine(int(x), int(m.top()), int(x), int(m.bottom()))
            p.setPen(QColor(T["dim"]))
            p.drawText(QRectF(x - 24, m.bottom() + 6, 48, 18),
                       Qt.AlignmentFlag.AlignCenter, str(d))

        # 80% 判据线
        y80 = m.bottom() - 0.8 * m.height()
        p.setPen(QPen(QColor(T["warn"]), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(m.left()), int(y80), int(m.right()), int(y80))
        p.setPen(QColor(T["warn"]))
        p.drawText(QRectF(m.right() - 66, y80 - 18, 64, 16),
                   Qt.AlignmentFlag.AlignRight, "判据 80%")

        # 曲线
        for name, color, pts in self.series:
            pts = sorted(pts)
            if len(pts) < 1:
                continue
            col = QColor(color)
            path = QPainterPath()
            for i, (d, acc) in enumerate(pts):
                x = xpos.get(d)
                if x is None:
                    continue
                y = m.bottom() - max(0.0, min(100.0, acc)) / 100.0 * m.height()
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.setBrush(QBrush())       # 开放折线不能带画刷，否则被闭合填充
            p.setPen(QPen(col, 2.2))
            p.drawPath(path)
            p.setBrush(QBrush(col))
            p.setPen(Qt.PenStyle.NoPen)
            for d, acc in pts:
                x = xpos.get(d)
                if x is None:
                    continue
                y = m.bottom() - max(0.0, min(100.0, acc)) / 100.0 * m.height()
                p.drawEllipse(QRectF(x - 3.5, y - 3.5, 7, 7))

        # 图例
        lx = m.left() + 6
        for name, color, _pts in self.series:
            p.setBrush(QBrush(QColor(color)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(lx, 12, 12, 4))
            p.setPen(QColor(T["sub"]))
            p.drawText(QRectF(lx + 18, 4, 150, 20), Qt.AlignmentFlag.AlignLeft, name)
            lx += 138


class BarChart(QWidget):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.bars = []
        self.setMinimumHeight(230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_bars(self, bars):
        self.bars = bars
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        T = self.T
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(T["chart"]))
        m = QRectF(46, 30, max(10, w - 64), max(10, h - 70))
        if not self.bars:
            p.setPen(QColor(T["dim"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "暂无消融数据")
            return

        for v in (0, 50, 100):
            y = m.bottom() - v / 100.0 * m.height()
            p.setPen(QPen(QColor(T["grid"]), 1))
            p.drawLine(int(m.left()), int(y), int(m.right()), int(y))
            p.setPen(QColor(T["dim"]))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(2, y - 9, 40, 18), Qt.AlignmentFlag.AlignRight, "%d" % v)

        y80 = m.bottom() - 0.8 * m.height()
        p.setPen(QPen(QColor(T["warn"]), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(m.left()), int(y80), int(m.right()), int(y80))

        n = len(self.bars)
        slot = m.width() / n
        bw = min(64, slot * 0.52)
        for i, (label, value, color) in enumerate(self.bars):
            cx = m.left() + slot * (i + 0.5)
            hh = max(0.0, min(100.0, value)) / 100.0 * m.height()
            p.setBrush(QBrush(QColor(color)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(cx - bw / 2, m.bottom() - hh, bw, hh))
            p.setPen(QColor(T["text"]))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(cx - bw, m.bottom() - hh - 20, bw * 2, 18),
                       Qt.AlignmentFlag.AlignCenter, "%.1f%%" % value)
            p.setPen(QColor(T["sub"]))
            p.drawText(QRectF(cx - slot / 2, m.bottom() + 8, slot, 30),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, label)


def card(T, title, value, note, color):
    f = QFrame()
    f.setObjectName("Card")
    v = QVBoxLayout(f)
    v.setContentsMargins(14, 12, 14, 12)
    v.setSpacing(4)
    a = QLabel(title)
    a.setStyleSheet("color:%s;" % T["sub"])
    b = QLabel(value)
    b.setStyleSheet("color:%s; font-size:26px; font-weight:600;" % color)
    c = QLabel(note)
    c.setStyleSheet("color:%s;" % T["dim"])
    v.addWidget(a)
    v.addWidget(b)
    v.addWidget(c)
    return f


class ResultsWindow(QMainWindow):
    def __init__(self, theme=None):
        super().__init__()
        self.T = theme
        if self.T is None:
            from neuron_vis import THEMES
            self.T = THEMES["dark"]
        self.setWindowTitle("开发成果 · Benchmark-1 多尺度资格迹")
        self.resize(1400, 900)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(build_qss(self.T))

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)

        # 标题行
        head = QHBoxLayout()
        box = QVBoxLayout()
        t = QLabel("Benchmark-1 · 多尺度资格迹验证")
        t.setStyleSheet("font-size:20px; font-weight:600; color:%s;" % self.T["text"])
        sub = QLabel("延迟信用分配 · 数据来源 data/b1_m1/*.jsonl · 每次运行为独立脑实例")
        sub.setStyleSheet("color:%s;" % self.T["sub"])
        box.addWidget(t)
        box.addWidget(sub)
        head.addLayout(box)
        head.addStretch(1)
        self.badge = QLabel("载入中")
        self.badge.setStyleSheet(
            "background:%s; color:%s; border-radius:8px; padding:8px 14px; font-weight:600;"
            % (self.T["panel_alt"], self.T["warn"]))
        head.addWidget(self.badge)
        btn = QPushButton("刷新数据")
        btn.clicked.connect(self.refresh)
        head.addWidget(btn)
        lay.addLayout(head)

        # 指标卡
        self.cards = QGridLayout()
        self.cards.setSpacing(10)
        lay.addLayout(self.cards)

        # 图区
        charts = QHBoxLayout()
        charts.setSpacing(12)
        left = QFrame()
        left.setObjectName("Card")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 10, 12, 10)
        lv.addWidget(QLabel("延迟 — 正确率曲线（判据 D=200 ≥ 80%）"))
        self.line = LineChart(self.T)
        lv.addWidget(self.line)
        charts.addWidget(left, 3)

        right = QFrame()
        right.setObjectName("Card")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 10, 12, 10)
        rv.addWidget(QLabel("D=200 消融对照"))
        self.bars = BarChart(self.T)
        rv.addWidget(self.bars, 1)
        self.criteria_box = QWidget()
        self.criteria_lay = QVBoxLayout(self.criteria_box)
        self.criteria_lay.setContentsMargins(0, 6, 0, 0)
        self.criteria_lay.setSpacing(4)
        rv.addWidget(self.criteria_box)
        charts.addWidget(right, 2)
        lay.addLayout(charts, 1)

        # 数据表
        self.table = QTableWidget()
        self.table.setMinimumHeight(205)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setCornerButtonEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table)

        foot = QLabel("")
        foot.setStyleSheet("color:%s;" % self.T["dim"])
        self.foot = foot
        lay.addWidget(foot)

        self.refresh()

    def refresh(self):
        T = self.T
        rows = load_summaries()
        curves = {
            "多尺度 K=3": sweep_curve(rows, "k3"),
            "单尺度 K=1": sweep_curve(rows, "k1_200"),
            "无资格迹": sweep_curve(rows, "notrace"),
        }
        colors = [T["accent"], T["warn"], T["dim"]]
        self.line.set_series([
            (name, colors[i], pts)
            for i, (name, pts) in enumerate(curves.items())
        ])

        def acc(tag, default=0.0):
            r = rows.get(tag)
            return float(r.get("test_acc", default)) if r else default

        k3_200 = acc("sweep_k3_D200")
        k1_200 = acc("sweep_k1_200_D200")
        notrace_200 = acc("sweep_notrace_D200")
        noreward = acc("f_noreward", acc("abl_noreward"))
        k3_500 = acc("sweep_k3_D500")
        k1_500 = acc("sweep_k1_200_D500")
        drop = k3_200 - notrace_200
        dh = d_half(curves["多尺度 K=3"])
        gain = k3_500 - k1_500

        self.bars.set_bars([
            ("多尺度\nK=3", k3_200, T["accent"]),
            ("单尺度\nK=1", k1_200, T["purple"]),
            ("无资格迹", notrace_200, T["danger"]),
            ("无奖励", noreward, T["dim"]),
        ])

        ok, bad = T["ok"], T["danger"]
        items = [
            ("D=200 正确率 ≥80%", "%.1f%%" % k3_200, k3_200 >= 80),
            ("资格迹消融下降 ≥30pp", "%.1fpp" % drop, drop >= 30),
            ("半衰延迟 D½ ≥200", "%.0f" % (dh or 0), (dh or 0) >= 200),
            ("无奖励调制应退化 <80%", "%.1f%%" % noreward, 0 < noreward < 80),
        ]
        while self.criteria_lay.count():
            it = self.criteria_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for label, value, passed in items:
            mark = "PASS" if passed else "FAIL"
            q = QLabel("[%s]  %s  %s" % (mark, label, value))
            q.setStyleSheet("color:%s;" % (ok if passed else bad))
            self.criteria_lay.addWidget(q)

        # 指标卡重建
        while self.cards.count():
            item = self.cards.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        specs = [
            ("D=200 测试正确率", "%.1f%%" % k3_200, "判据 ≥ 80%", ok if k3_200 >= 80 else bad),
            ("资格迹消融下降", "%.1fpp" % drop, "判据 ≥ 30pp", ok if drop >= 30 else bad),
            ("半衰延迟 D½", "%.0f" % (dh or 0), "判据 ≥ 200 ticks", ok if (dh or 0) >= 200 else bad),
            ("多尺度 D=500 增益", "%+.1fpp" % gain, "K=3 相对 K=1", T["accent"]),
        ]
        for i, (title, value, note, color) in enumerate(specs):
            self.cards.addWidget(card(T, title, value, note, color), 0, i)

        delays = sorted({d for pts in curves.values() for d, _ in pts})
        names = list(curves.keys())
        self.table.setRowCount(len(delays))
        self.table.setColumnCount(1 + len(names))
        self.table.setHorizontalHeaderLabels(["延迟 D"] + names)
        lookup = {name: dict(pts) for name, pts in curves.items()}
        for r, d in enumerate(delays):
            self.table.setItem(r, 0, QTableWidgetItem(str(d)))
            for c, name in enumerate(names):
                v = lookup[name].get(d)
                self.table.setItem(r, c + 1,
                                   QTableWidgetItem("-" if v is None else "%.1f%%" % v))

        pending = "τe–N 曲线 / neuromod 交叉 / 稀疏化 / 10 种子分布"
        self.badge.setText("核心判据通过 · 强制报告项待补" if k3_200 >= 80 else "数据未达标")
        self.badge.setStyleSheet(
            "background:%s; color:%s; border-radius:8px; padding:8px 14px; font-weight:600;"
            % (T["panel_alt"], T["warn"] if k3_200 >= 80 else T["danger"]))
        self.foot.setText(
            "强制报告项（待补）：%s　|　扫描复现：tools/b1_sweep.ps1　|　"
            "汇总：tools/b1_report.py　|　共 %d 条 summary 记录" % (pending, len(rows)))


def main():
    check = "--check" in sys.argv
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    w = ResultsWindow()
    w.show()
    if check:
        out = os.path.join(tempfile.gettempdir(), "results_panel_check.png")
        QTimer.singleShot(1200, lambda: (w.grab().save(out), app.quit()))
    rc = app.exec()
    if check:
        print("check screenshot:", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
