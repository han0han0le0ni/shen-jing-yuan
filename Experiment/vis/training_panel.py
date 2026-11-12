"""训练存档窗口：把「学到的权重」显式保存 / 恢复，并展示可核验的指纹。

主窗口工具栏「训练存档」打开；独立自检：
    py -3 vis/training_panel.py --check

为什么需要它：engine.reset() 会按 seed 重新随机初始化权重，进程退出也不保存，
所以以前刷新一次就和没训练过一样。本窗口调用 training_store 落盘/回读。
"""
import os
import sys
import tempfile

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QListWidget,
    QListWidgetItem, QTextBrowser, QFrame,
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont

import training_store as store

DEFAULT_THEME = {
    "bg": "#0b0e14", "panel": "#0f131c", "panel_alt": "#141b28",
    "border": "#1f2636", "text": "#dfe7f5", "sub": "#8a9ab3",
    "dim": "#5c6a83", "accent": "#22d3ee", "warn": "#ffbf47",
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
    QListWidget, QTextBrowser { background: %s; color: %s;
                                border: 1px solid %s; border-radius: 6px; padding: 6px; }
    QListWidget::item:selected { background: %s; color: %s; }
    QPushButton { background: %s; color: %s; border: 1px solid %s;
                  border-radius: 6px; padding: 6px 12px; }
    QPushButton:hover { background: %s; }
    QCheckBox { color: %s; }
    """ % (T["text"], T["bg"], T["panel"], T["border"], T["sub"],
           T["panel_alt"], T["text"], T["border"], T["btn_hover"],
           T["text"], T["btn"], T["text"], T["border"], T["btn_hover"], T["sub"])


class SnapshotWindow(QMainWindow):
    def __init__(self, engine=None, theme=None, flags=None, on_change=None,
                 extra=None):
        super().__init__()
        self.engine = engine
        self.T = theme or DEFAULT_THEME
        self.flags = flags if flags is not None else {"auto_save": True,
                                                     "auto_load": True}
        self.on_change = on_change
        self.extra = extra or {}
        self.setWindowTitle("训练存档 · 权重与进度持久化")
        self.resize(900, 660)
        self.setMinimumSize(760, 540)
        self.setStyleSheet(build_qss(self.T))

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        head = QHBoxLayout()
        box = QVBoxLayout()
        t = QLabel("训练存档 · 学到的权重不再丢")
        t.setStyleSheet("font-size:19px; font-weight:600;")
        s = QLabel("保存突触权重 W、资格迹、训练轮次、引导依赖 G 与最近 360 tick 引导记录")
        s.setObjectName("Mute")
        box.addWidget(t)
        box.addWidget(s)
        head.addLayout(box)
        head.addStretch(1)
        self.badge = QLabel("—")
        head.addWidget(self.badge)
        lay.addLayout(head)

        card = QFrame()
        card.setObjectName("Card")
        grid = QGridLayout(card)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setSpacing(8)
        self.vals = {}
        fields = [
            ("后端", "device"), ("神经元 / 突触", "size"),
            ("权重指纹", "w_hash"), ("权重总和", "w_sum"),
            ("训练轮次", "episode"), ("到达次数", "arrivals"),
            ("引导依赖 G", "g"), ("拓扑指纹", "topo_sig"),
        ]
        for i, (label, key) in enumerate(fields):
            a = QLabel(label)
            a.setStyleSheet("color:%s;" % self.T["sub"])
            b = QLabel("—")
            b.setStyleSheet("font-weight:600;")
            grid.addWidget(a, i // 4 * 2, i % 4)
            grid.addWidget(b, i // 4 * 2 + 1, i % 4)
            self.vals[key] = b
        lay.addWidget(card)

        lay.addWidget(QLabel("存档列表（选中后可载入 / 删除）"))
        self.lst = QListWidget()
        self.lst.itemDoubleClicked.connect(lambda _it: self._load())
        lay.addWidget(self.lst, 1)

        row = QHBoxLayout()
        for text, fn, tip in (
            ("保存到自动存档", self._save, "覆盖 autosave.npz；退出时也会自动保存"),
            ("另存副本", self._save_as, "按时间戳另存一份，不覆盖自动存档"),
            ("载入选中", self._load, "把权重与进度写回当前网络（拓扑必须一致）"),
            ("删除选中", self._delete, "删除选中的存档文件"),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)

        opts = QHBoxLayout()
        self.chk_save = QCheckBox("退出时自动保存")
        self.chk_save.setChecked(bool(self.flags.get("auto_save", True)))
        self.chk_save.toggled.connect(
            lambda v: self.flags.__setitem__("auto_save", bool(v)))
        self.chk_load = QCheckBox("启动时自动载入")
        self.chk_load.setChecked(bool(self.flags.get("auto_load", True)))
        self.chk_load.toggled.connect(
            lambda v: self.flags.__setitem__("auto_load", bool(v)))
        opts.addWidget(self.chk_save)
        opts.addWidget(self.chk_load)
        opts.addStretch(1)
        lay.addLayout(opts)

        self.log = QTextBrowser()
        self.log.setFixedHeight(110)
        lay.addWidget(self.log)

        self.foot = QLabel(
            "存档目录 Experiment/data/training/。载入前会核对拓扑指纹：换网络规模、种子或后端后"
            "旧权重不适用，会拒绝载入。C DLL 后端只有读权重的接口、没有写接口，"
            "需要保存训练成果请切到「向量化 · 12 核」（选老师时会自动切过去）。")
        self.foot.setObjectName("Mute")
        self.foot.setWordWrap(True)
        lay.addWidget(self.foot)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh_info)
        self.timer.start()
        self.refresh()
        self.log.append("就绪。训练一段时间后点「保存到自动存档」，下次启动会自动恢复。")

    # ---------- 展示 ----------
    def _set_badge(self, text, color):
        self.badge.setText(text)
        self.badge.setStyleSheet(
            "background:%s; color:%s; border-radius:8px; padding:7px 12px; "
            "font-weight:600;" % (self.T["panel_alt"], color))

    def refresh_info(self):
        if self.engine is None:
            self._set_badge("自检模式", self.T["warn"])
            return
        info = store.current_info(self.engine)
        if not info.get("supported"):
            self._set_badge("当前后端不支持存档", self.T["danger"])
            for b in self.vals.values():
                b.setText("—")
            self.vals["device"].setText("C DLL（只读权重）")
            return
        self._set_badge("后端可存档 · %s" % info["device"], self.T["ok"])
        self.vals["device"].setText(info["device"])
        self.vals["size"].setText("%d / %d" % (info["neuron_count"],
                                               info["synapse_count"]))
        self.vals["w_hash"].setText(info["w_hash"])
        self.vals["w_sum"].setText("%.2f" % info["w_sum"])
        self.vals["episode"].setText(str(info["episode"]))
        self.vals["arrivals"].setText(str(info["arrivals"]))
        self.vals["g"].setText("%.3f" % info["teacher_g"])
        self.vals["topo_sig"].setText(info["topo_sig"])

    def refresh(self):
        self.refresh_info()
        self.lst.clear()
        for p in store.list_snapshots():
            m = store.describe(p)
            name = os.path.basename(p)
            if m is None:
                label = "%s · 存档损坏" % name
            else:
                label = ("%s · %s · 第 %s 轮 / 到达 %s · G=%.2f · w=%s · %.0f KB"
                         % (name, m.get("saved_at", "?"), m.get("episode", 0),
                            m.get("arrivals", 0), float(m.get("teacher_g", 0)),
                            m.get("w_hash", "?"), m["size"] / 1024.0))
            it = QListWidgetItem(label)
            it.setData(0x0100, p)
            self.lst.addItem(it)
        if self.lst.count() and not self.lst.selectedItems():
            self.lst.setCurrentRow(0)

    def _selected_path(self):
        items = self.lst.selectedItems()
        return items[0].data(0x0100) if items else None

    # ---------- 操作 ----------
    def _save(self, path=None):
        if self.engine is None:
            self.log.append("自检模式不写文件。")
            return
        try:
            m = store.save(self.engine, path=path, extra=dict(self.extra))
            self.log.append("已保存 %s：第 %d 轮，权重指纹 %s，G=%.3f"
                            % (os.path.basename(m.get("path", path or store.DEFAULT_NAME)),
                               m["episode"], m["w_hash"], m["teacher_g"]))
            self.refresh()
            if self.on_change:
                self.on_change("已保存 · 第 %d 轮" % m["episode"])
        except store.SnapshotError as exc:
            self.log.append("保存失败：" + str(exc))

    def _save_as(self):
        import time as _t
        self._save(os.path.join(store.snap_dir(),
                                "snap_%s.npz" % _t.strftime("%Y%m%d_%H%M%S")))

    def _load(self):
        if self.engine is None:
            self.log.append("自检模式不载入。")
            return
        p = self._selected_path()
        if not p:
            self.log.append("请先选中一个存档。")
            return
        try:
            m = store.load(self.engine, path=p)
            self.log.append("已载入 %s：第 %d 轮，权重指纹 %s，G=%.3f（已写回当前网络）"
                            % (os.path.basename(p), m["episode"],
                               m["w_hash"], m["teacher_g"]))
            self.refresh()
            if self.on_change:
                self.on_change("已恢复存档 · 第 %d 轮 · G=%.2f"
                               % (m["episode"], m["teacher_g"]))
        except store.SnapshotError as exc:
            self.log.append("载入失败：" + str(exc))

    def _delete(self):
        p = self._selected_path()
        if not p:
            self.log.append("请先选中一个存档。")
            return
        ok = store.delete(p)
        self.log.append(("已删除 " if ok else "删除失败 ") + os.path.basename(p))
        self.refresh()


def main():
    check = "--check" in sys.argv
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))
    eng = None
    if check:
        import neuron_vis as nv
        eng = nv.SimEngine(None)
        eng.reset("numpy", nv.PRESETS[1][1:], 20261112)
        for _ in range(150):
            eng.train_tick()
    w = SnapshotWindow(eng)
    w.show()
    if check:
        w.log.append("自检：用真实 numpy 后端跑了 150 tick，权重已变化；此模式不写文件。")
        out = os.path.join(tempfile.gettempdir(), "training_panel_check.png")
        QTimer.singleShot(900, lambda: (w.grab().save(out), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
