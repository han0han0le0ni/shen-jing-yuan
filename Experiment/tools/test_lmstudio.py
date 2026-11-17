"""LM Studio 接入回归测试：不依赖真实 LM Studio，本地起一个假 OpenAI 兼容服务。

运行： py -3 tools/test_lmstudio.py
覆盖：接口层 / 契约发现 / 蒸馏解析 / 异步不阻塞 / 失败回退 / 对话窗口联网自检。
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS = os.path.join(ROOT, "vis")
sys.path.insert(0, VIS)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import lmstudio_client as lm                       # noqa: E402
import learning_registry as reg                     # noqa: E402
from learning_api import DLSTeacher                 # noqa: E402

FAILED = []


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


class FakeLMStudio(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            time.sleep(getattr(self.server, "delay", 0.0))
            self._json({"data": [{"id": "fake-local-7b"}, {"id": "fake-local-3b"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b"{}"
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._json({"error": "not found"}, 404)
            return
        time.sleep(getattr(self.server, "delay", 0.0))
        try:
            payload = json.loads(body.decode("utf-8"))
            self.server.last_messages = payload.get("messages", [])
        except ValueError:
            pass
        self._json({"choices": [{"message": {
            "role": "assistant", "content": getattr(self.server, "reply", "{\"w1\":0.1,\"w2\":-0.1}")
        }}]})


def start_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLMStudio)
    srv.reply = '{"w1":0.12,"w2":-0.08}'
    srv.delay = 0.0
    srv.last_messages = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


STATE = {
    "theta1": 0.4, "theta2": -0.6, "end_x": 1.2, "end_y": 0.5,
    "target_x": 0.6, "target_y": 0.9, "L1": 1.0, "L2": 1.0, "max_omega": 0.5,
}


def test_client(base):
    print("[1] 接口层（OpenAI 兼容 HTTP）")
    check(lm.normalize_base("127.0.0.1:1234") == "http://127.0.0.1:1234/v1",
          "地址自动补协议与 /v1")
    check(lm.normalize_base("http://127.0.0.1:1234/v1/") == "http://127.0.0.1:1234/v1",
          "已带 /v1 的地址不重复追加")
    models = lm.list_models(base, timeout=2.0)
    check(models == ["fake-local-7b", "fake-local-3b"], "GET /v1/models 返回模型列表")
    txt = lm.chat([{"role": "user", "content": "ping"}], base_url=base,
                  model="fake-local-7b", timeout=2.0)
    check(txt == '{"w1":0.12,"w2":-0.08}', "POST /v1/chat/completions 取回 content")
    info = lm.probe(base, timeout=2.0)
    check(info["ok"] and info["model"] == "fake-local-7b", "probe 报告已连接与自动模型")
    bad = lm.probe("http://127.0.0.1:1/v1", timeout=0.4)
    check(not bad["ok"] and "Start Server" in bad["error"], "服务未启动 → 可读错误提示")


def test_config():
    print("[2] 配置读写（写临时文件，不污染工作区）")
    orig = lm.CONFIG_PATH
    with tempfile.TemporaryDirectory() as d:
        lm.CONFIG_PATH = os.path.join(d, "lmstudio.json")
        cfg = lm.load_config()
        check(cfg["base_url"] == lm.DEFAULT_CONFIG["base_url"], "无配置时用默认值")
        out = lm.save_config({"base_url": "127.0.0.1:9999", "model": "fake-local-7b"})
        check(out["base_url"] == "http://127.0.0.1:9999/v1", "保存并规范化地址")
        check(lm.load_config()["model"] == "fake-local-7b", "重新加载读回模型名")
    lm.CONFIG_PATH = orig


def test_plugin_discovery():
    print("[3] 插件自动发现与契约")
    ps, errs = reg.discover()
    p = next((x for x in ps if x["id"] == "lmstudio"), None)
    check(p is not None, "LM Studio 老师出现在选项栏")
    check(not errs, "插件目录无加载错误：%s" % errs)
    if p is None:
        return None
    check(len(p["params"]) == 5, "5 个动态参数控件（every/temperature/timeout/fallback/port）")
    obj = p["create"]({s["key"]: s.get("default") for s in p["params"]})
    check(hasattr(obj, "command"), "create() 返回符合 Teacher 契约的对象")
    return obj


def test_distill(srv, base):
    print("[4] 蒸馏方式：JSON 角速度 / 方向意图 / 回退")
    import importlib.util
    path = os.path.join(VIS, "learning_plugins", "lmstudio.py")
    spec = importlib.util.spec_from_file_location("_lm_plugin_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    srv.reply = '{"w1":0.12,"w2":-0.08}'
    w = mod.parse_teacher_reply(srv.reply, STATE)
    check(abs(w[0] - 0.12) < 1e-9 and abs(w[1] + 0.08) < 1e-9, "解析 {\"w1\",\"w2\"}")
    srv.reply = '```json\n{"action":"up_right"}\n```'
    w = mod.parse_teacher_reply(srv.reply, STATE)
    check(len(w) == 2 and all(abs(v) <= STATE["max_omega"] + 1e-9 for v in w),
          "方向意图经解析层转成钳位后的关节角速度")
    w2 = mod.parse_teacher_reply("应该往右上走", STATE)
    check(len(w2) == 2, "纯中文方向词也能解析")
    try:
        mod.parse_teacher_reply("我不知道", STATE)
        check(False, "无法解析时抛 ValueError")
    except ValueError:
        check(True, "无法解析时抛 ValueError")
    w3 = mod.parse_teacher_reply('{"w1":99,"w2":-99}', STATE)
    check(abs(w3[0]) <= STATE["max_omega"] and abs(w3[1]) <= STATE["max_omega"],
          "越界数值被钳到 ±max_omega")

    print("[5] 老师插件：异步不阻塞 / 命中 / 失败回退 DLS")
    srv.reply = '{"w1":0.12,"w2":-0.08}'
    srv.delay = 0.35
    t = mod.LMStudioTeacher(every=1, temperature=0.1, timeout=3.0,
                            fallback=True, base_url=base, model="fake-local-7b")
    t0 = time.perf_counter()
    for _ in range(60):
        t.command(STATE)
    elapsed = time.perf_counter() - t0
    check(elapsed < 0.20, "60 次 command() 不被网络请求阻塞（%.3fs）" % elapsed)
    deadline = time.time() + 3.0
    while t._busy and time.time() < deadline:
        time.sleep(0.02)
    check(t.hits >= 1 and t.last_source == "lm", "后台请求命中并缓存模型指令")
    check(t._cache is not None and len(t._cache) == 2, "缓存是 (w1,w2)")
    check("命中" in t.info(), "info() 提供请求/命中/来源统计")

    srv.reply = "模型没听懂"
    t2 = mod.LMStudioTeacher(every=1, temperature=0.1, timeout=3.0,
                             fallback=True, base_url=base, model="fake-local-7b")
    t2.command(STATE)
    deadline = time.time() + 3.0
    while t2._busy and time.time() < deadline:
        time.sleep(0.02)
    check(t2.last_source == "dls" and t2.last_error, "解析失败 → 记录原因并切回 DLS")
    dls_w = DLSTeacher().command(STATE)
    got = t2.command(STATE)
    check(abs(got[0] - dls_w[0]) < 1e-9 and abs(got[1] - dls_w[1]) < 1e-9,
          "回退输出与 DLS 解析解一致")

    t3 = mod.LMStudioTeacher(every=1, timeout=1.0, fallback=True,
                             base_url="http://127.0.0.1:1/v1", model="fake-local-7b")
    t3.command(STATE)
    deadline = time.time() + 3.0
    while t3._busy and time.time() < deadline:
        time.sleep(0.02)
    check(t3.last_source == "dls" and "Start Server" in t3.last_error,
          "服务未启动 → 给出可读原因并回退")
    srv.delay = 0.0


def test_window(srv, base):
    print("[6] 对话窗口：连接自检 + 一轮真实往返")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    import chat_panel

    app = QApplication([])
    w = chat_panel.ChatWindow(chat_panel._FakeEngine())
    w.resize(880, 680)
    w.show()
    w.base_edit.setText(base)
    srv.reply = "引导已生效：G=1，注入指令等于老师指令。"
    w._probe()

    def wait_probe():
        if w._busy:
            QTimer.singleShot(30, wait_probe)
            return
        check("已连接" in w.badge.text(), "窗口徽标显示已连接")
        check(w.model_combo.count() >= 2, "模型下拉自动填充")
        w._ask("现在的引导生效吗？")
        QTimer.singleShot(30, wait_chat)

    def wait_chat():
        if w._busy:
            QTimer.singleShot(30, wait_chat)
            return
        check(any(m["role"] == "assistant" for m in w.history), "对话收到模型回复")
        check(len([m for m in w.history if m["role"] == "user"]) == 1,
              "用户消息只入历史一次（上下文不重复）")
        check("当前仿真状态" in " ".join(
            m["content"] for m in srv.last_messages if m["role"] == "system"),
            "当前引导状态随请求附上")
        w.close()
        app.quit()

    QTimer.singleShot(30, wait_probe)
    QTimer.singleShot(15000, app.quit)
    app.exec()


def main():
    srv = start_server()
    base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
    try:
        test_client(base)
        test_config()
        test_plugin_discovery()
        test_distill(srv, base)
        test_window(srv, base)
    finally:
        srv.shutdown()
        srv.server_close()
    print()
    if FAILED:
        print("FAILED %d: %s" % (len(FAILED), "; ".join(FAILED)))
        return 1
    print("LMSTUDIO OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
