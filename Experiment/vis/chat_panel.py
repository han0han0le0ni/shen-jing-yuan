"""AI 对话窗口：连接本机 LM Studio，聊天 + 查看当前引导/蒸馏状态。

主窗口工具栏「AI 对话」打开；也可独立自检：
    py -3 vis/chat_panel.py --check

链路（与蒸馏老师插件共用同一个接口配置）：
    vis/lmstudio.json  ->  lmstudio_client.py  ->  LM Studio /v1/chat/completions
    LM Studio 老师插件  ->  command(state)->(w1,w2)  ->  G 混合 -> motor_override
"""
import html
import json
import os
import sys
import tempfile
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextBrowser, QPushButton, QComboBox, QCheckBox,
    QFrame,
)
from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtGui import QFont

import lmstudio_client as lm

DEFAULT_THEME = {
    "bg": "#0b0e14", "panel": "#0f131c", "panel_alt": "#141b28",
    "border": "#1f2636", "text": "#dfe7f5", "sub": "#8a9ab3",
    "dim": "#5c6a83", "accent": "#22d3ee", "purple": "#a78bfa",
    "warn": "#ffbf47", "ok": "#34d399", "danger": "#fb7185",
    "btn": "#182033", "btn_hover": "#22304a",
}


def build_qss(T):
    return """
    QWidget { font-family: "Microsoft YaHei","Segoe UI",sans-serif;
              font-size: 12px; color: %s; }
    QMainWindow, #Root { background: %s; }
    QFrame#Card { background: %s; border: 1px solid %s; border-radius: 10px; }
    QLabel#Mute { color: %s; }
    QLineEdit, QComboBox, QTextBrowser {
        background: %s; color: %s; border: 1px solid %s;
        border-radius: 6px; padding: 6px 8px; }
    QTextBrowser { padding: 10px; }
    QPushButton { background: %s; color: %s; border: 1px solid %s;
                  border-radius: 6px; padding: 6px 12px; }
    QPushButton:hover { background: %s; }
    QPushButton:disabled { color: %s; }
    QCheckBox { color: %s; }
    """ % (T["text"], T["bg"], T["panel"], T["border"], T["sub"],
           T["panel_alt"], T["text"], T["border"], T["btn"], T["text"],
           T["border"], T["btn_hover"], T["dim"], T["sub"])


class _Bridge(QObject):
    """后台线程 -> GUI 线程：Qt 信号自动排队，避免跨线程操作控件。"""
    done = Signal(str, str)


class _FakeEngine:
    """--check 用的假仿真状态。"""
    guidance = {
        "tick": 300, "g": 1.0, "teacher": "DLS 逆运动学（自检数据）",
        "net": (0.068, 0.228), "cmd": (-0.190, 0.500),
        "final": (-0.190, 0.500), "dist": 2.831,
        "effective": True, "backend": "向量化·12核", "reason": "",
    }


class ChatWindow(QMainWindow):
    def __init__(self, engine=None, theme=None):
        super().__init__()
        self.engine = engine
        self.T = theme or DEFAULT_THEME
        self.history = []
        self._busy = False
        self._kind = ""
        self._pending_user = ""
        self.cfg = lm.load_config()
        self.setWindowTitle("AI 对话 · LM Studio（本地模型）")
        self.resize(880, 680)
        self.setMinimumSize(720, 520)
        self.setStyleSheet(build_qss(self.T))

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        head = QHBoxLayout()
        box = QVBoxLayout()
        t = QLabel("AI 对话 · 本地模型接入")
        t.setStyleSheet("font-size:19px; font-weight:600;")
        s = QLabel("OpenAI 兼容 HTTP 接口 · 与「LM Studio 老师」插件共用配置")
        s.setObjectName("Mute")
        box.addWidget(t)
        box.addWidget(s)
        head.addLayout(box)
        head.addStretch(1)
        self.badge = QLabel("未连接")
        head.addWidget(self.badge)
        lay.addLayout(head)

        conn = QFrame()
        conn.setObjectName("Card")
        ch = QHBoxLayout(conn)
        ch.setContentsMargins(12, 10, 12, 10)
        ch.setSpacing(8)
        ch.addWidget(QLabel("接口"))
        self.base_edit = QLineEdit(self.cfg["base_url"])
        self.base_edit.setToolTip("LM Studio 的 OpenAI 兼容地址，默认 http://127.0.0.1:1234/v1")
        ch.addWidget(self.base_edit, 2)
        ch.addWidget(QLabel("模型"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItem(self.cfg["model"] or "自动（第一个已加载模型）")
        self.model_combo.setMinimumWidth(190)
        ch.addWidget(self.model_combo, 1)
        self.btn_probe = QPushButton("测试连接 / 刷新模型")
        self.btn_probe.clicked.connect(self._probe)
        ch.addWidget(self.btn_probe)
        self.btn_save = QPushButton("保存设置")
        self.btn_save.setToolTip("写入 vis/lmstudio.json；老师插件重新选择后即生效")
        self.btn_save.clicked.connect(self._save)
        ch.addWidget(self.btn_save)
        lay.addWidget(conn)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        lay.addWidget(self.chat_view, 1)

        self.lbl_teacher = QLabel("老师插件：未启用")
        self.lbl_teacher.setObjectName("Mute")
        lay.addWidget(self.lbl_teacher)

        bar = QHBoxLayout()
        self.chk_state = QCheckBox("附上当前仿真状态")
        self.chk_state.setChecked(True)
        self.chk_state.setToolTip("把 tick / G / 网络 / 老师 / 注入指令一起发给模型")
        bar.addWidget(self.chk_state)
        self.btn_explain = QPushButton("解释当前引导")
        self.btn_explain.clicked.connect(self._explain)
        bar.addWidget(self.btn_explain)
        bar.addStretch(1)
        lay.addLayout(bar)

        inp = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("问点什么，例如：现在的引导生效吗？下一步该往哪走？")
        self.input_edit.returnPressed.connect(self._send)
        inp.addWidget(self.input_edit, 1)
        self.btn_send = QPushButton("发送")
        self.btn_send.clicked.connect(self._send)
        inp.addWidget(self.btn_send)
        lay.addLayout(inp)

        self.foot = QLabel(
            "蒸馏方式：模型输出方向意图/角速度 JSON → 解析层转成 (ω₁,ω₂) → 按 G 与网络解码混合 "
            "→ motor_override 注入 → R-STDP 模仿。接口只需 LM Studio 开启 Start Server；"
            "服务不可用时老师自动回退 DLS 解析解。")
        self.foot.setObjectName("Mute")
        self.foot.setWordWrap(True)
        lay.addWidget(self.foot)

        self._bridge = _Bridge(self)
        self._bridge.done.connect(self._on_done)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_teacher)
        self._timer.start()

        self._append("系统", "已就绪。先点「测试连接 / 刷新模型」确认 LM Studio 服务，再开始对话。"
                            "接口地址与模型名保存后，蒸馏老师插件会共用。")
        self._refresh_teacher()

    # ---------- 展示 ----------
    def _append(self, who, text):
        color = {"你": self.T["accent"], "LM Studio": self.T["ok"],
                 "系统": self.T["sub"]}.get(who, self.T["text"])
        safe = html.escape(str(text)).replace("\n", "<br>")
        self.chat_view.append(
            '<div style="margin:4px 0;"><span style="color:%s;font-weight:600;">%s</span>'
            '&nbsp;&nbsp;<span style="color:%s;">%s</span></div>'
            % (color, html.escape(who), self.T["text"], safe))
        self.chat_view.verticalScrollBar().setValue(
            self.chat_view.verticalScrollBar().maximum())

    def _set_badge(self, text, color):
        self.badge.setText(text)
        self.badge.setStyleSheet(
            "background:%s; color:%s; border-radius:8px; padding:7px 12px; "
            "font-weight:600;" % (self.T["panel_alt"], color))

    def _state_text(self):
        g = getattr(self.engine, "guidance", None) or {}
        if not g:
            return "（仿真还没产生引导数据）"
        net, cmd, fin = g.get("net", (0, 0)), g.get("cmd", (0, 0)), g.get("final", (0, 0))
        return ("tick=%s · G=%.2f · 老师=%s · 后端=%s · 生效=%s%s · "
                "网络ω=(%.3f,%.3f) · 老师ω=(%.3f,%.3f) · 注入ω=(%.3f,%.3f) · 末端距离=%.3f"
                % (g.get("tick", 0), float(g.get("g", 0)), g.get("teacher", "无"),
                   g.get("backend", "?"), g.get("effective", False),
                   ("（" + g["reason"] + "）") if g.get("reason") else "",
                   net[0], net[1], cmd[0], cmd[1], fin[0], fin[1],
                   float(g.get("dist", 0))))

    def _refresh_teacher(self):
        teacher = getattr(self.engine, "_teacher", None)
        if teacher is None:
            self.lbl_teacher.setText(
                "老师插件：未启用 · 选择「LM Studio 老师」后此处显示请求/命中与回退情况")
        elif hasattr(teacher, "info"):
            self.lbl_teacher.setText("老师插件：%s · %s" % (teacher.name, teacher.info()))
        else:
            self.lbl_teacher.setText("老师插件：%s（无 LLM 请求统计）" % teacher.name)

    # ---------- 异步调用 ----------
    def _run(self, kind, fn):
        if self._busy:
            return
        self._busy, self._kind = True, kind
        self.btn_send.setEnabled(False)
        self.btn_probe.setEnabled(False)
        self.btn_explain.setEnabled(False)
        self._set_badge("请求中…", self.T["warn"])

        def work():
            try:
                self._bridge.done.emit(str(fn()), "")
            except Exception as exc:
                self._bridge.done.emit("", "%s: %s" % (type(exc).__name__, exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, text, err):
        self._busy = False
        self.btn_send.setEnabled(True)
        self.btn_probe.setEnabled(True)
        self.btn_explain.setEnabled(True)
        if err:
            self._append("系统", "请求失败：" + err)
            self._set_badge("连接失败", self.T["danger"])
            return
        if self._kind == "probe":
            try:
                info = json.loads(text)
            except ValueError:
                self._append("系统", "自检返回无法解析：" + text[:200])
                return
            self.model_combo.clear()
            if info.get("models"):
                self.model_combo.addItems(info["models"])
                if info.get("model"):
                    self.model_combo.setCurrentText(info["model"])
            else:
                self.model_combo.addItem("自动（第一个已加载模型）")
            if info.get("ok"):
                self._set_badge("已连接 · %d 个模型" % len(info["models"]), self.T["ok"])
                self._append("系统", "已连接，可用模型：" + ", ".join(info["models"]))
            else:
                self._set_badge("未连接", self.T["danger"])
                self._append("系统", info.get("error") or "未连接")
            return
        self.history.append({"role": "user", "content": self._pending_user})
        self.history.append({"role": "assistant", "content": text})
        self._append("LM Studio", text)
        self._set_badge("已连接", self.T["ok"])

    def _base(self):
        return lm.normalize_base(self.base_edit.text())

    def _model(self):
        cur = self.model_combo.currentText().strip()
        return "" if cur.startswith("自动") else cur

    def _probe(self):
        base = self._base()
        self._run("probe", lambda: json.dumps(
            lm.probe(base, timeout=2.0), ensure_ascii=False))

    def _save(self):
        try:
            cfg = lm.save_config({
                "base_url": self._base(),
                "model": self._model(),
                "temperature": self.cfg.get("temperature", 0.2),
                "timeout": self.cfg.get("timeout", 8.0),
            })
            self.cfg = cfg
            self.base_edit.setText(cfg["base_url"])
            self._append("系统", "设置已写入 %s。老师插件重新选择一次即生效（插件目录热加载也可）。"
                         % lm.CONFIG_PATH)
        except OSError as exc:
            self._append("系统", "保存失败：%s" % exc)

    def _messages(self, question):
        cfg = self.cfg
        msgs = [{"role": "system", "content": lm.SYSTEM_PROMPT}]
        state = self._state_text() if self.chk_state.isChecked() else ""
        if state:
            msgs.append({"role": "system", "content": "当前仿真状态：" + state})
        msgs.extend(self.history[-10:])
        msgs.append({"role": "user", "content": question})
        return msgs

    def _ask(self, question):
        self._append("你", question)
        self._pending_user = question
        self.input_edit.clear()
        msgs = self._messages(question)
        base, model = self._base(), self._model()
        self._run("chat", lambda: lm.chat(msgs, base_url=base, model=model,
                                          temperature=float(self.cfg.get("temperature", 0.2)),
                                          timeout=float(self.cfg.get("timeout", 8.0))))

    def _send(self):
        q = self.input_edit.text().strip()
        if q:
            self._ask(q)

    def _explain(self):
        self._ask("请根据当前仿真状态，简短说明引导是否生效、G 的含义，以及下一步该往哪个方向走。")


def main():
    check = "--check" in sys.argv
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))
    w = ChatWindow(_FakeEngine() if check else None)
    w.show()
    if check:
        w._append("你", "现在的引导生效吗？")
        w._append("LM Studio", "生效中：G=1.00，后端向量化·12核，注入指令与老师指令一致 "
                               "(-0.190, +0.500)，网络自主输出 (+0.068, +0.228)。"
                               "末端距离 2.831，建议继续朝目标方向推进。（自检示例）")
        out = os.path.join(tempfile.gettempdir(), "chat_panel_check.png")
        QTimer.singleShot(900, lambda: (w.grab().save(out), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
