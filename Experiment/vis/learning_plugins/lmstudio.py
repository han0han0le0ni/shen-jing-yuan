"""LM Studio 蒸馏老师插件：把本地大模型的「决策意图」蒸馏成网络老师指令。

契约与 DLS 老师完全一致：command(state) -> (w1, w2)，因此会被自动发现、
自动出现在学习接口选项栏、可在引导监控面板里看到实时曲线。

蒸馏方式（刻意不让 LLM 直接算逆运动学，本地小模型算不准）：
    1. 每 every 个 tick，把当前臂状态序列化成 prompt 异步发给 LM Studio；
    2. 模型只需输出一个方向意图或一对角速度，严格 JSON：
           {"action": "up_right"}   或   {"w1": 0.12, "w2": -0.08}
    3. 方向意图经雅可比/ DLS 解析层转成 (w1, w2)，再按 G 与网络解码混合，
       经 motor_override 注入；网络通过 R-STDP 模仿这个老师（决策蒸馏）。
    4. 请求在后台线程进行，command() 绝不阻塞 GUI；服务没开、超时、
       输出无法解析时自动回退到 DLS 解析解，仿真不会卡死也不会乱跑。

接口地址/模型名不在选项栏重复配置，统一读 vis/lmstudio.json
（在「AI 对话」窗口里点「保存设置」即写入，插件热加载后立即生效）。
"""
import json
import threading
import time

from learning_api import Teacher, DLSTeacher, GOAL_D0, _goal_omega
import lmstudio_client as lm

DIRS = {
    "up": (0.0, 1.0), "down": (0.0, -1.0),
    "left": (-1.0, 0.0), "right": (1.0, 0.0),
    "up_left": (-0.707, 0.707), "up_right": (0.707, 0.707),
    "down_left": (-0.707, -0.707), "down_right": (0.707, -0.707),
    "up-left": (-0.707, 0.707), "up-right": (0.707, 0.707),
    "down-left": (-0.707, -0.707), "down-right": (0.707, -0.707),
    "stop": (0.0, 0.0),
    "上": (0.0, 1.0), "下": (0.0, -1.0), "左": (-1.0, 0.0), "右": (1.0, 0.0),
    "左上": (-0.707, 0.707), "右上": (0.707, 0.707),
    "左下": (-0.707, -0.707), "右下": (0.707, -0.707), "停": (0.0, 0.0),
}

SYSTEM_PROMPT = (
    "你是二连杆机械臂的运动老师。末端要靠近目标点。"
    "只输出一行 JSON，不要解释、不要 markdown。"
    '两种格式任选其一：{"action":"方向"} 或 {"w1":数字,"w2":数字}。'
    "方向可选 up/down/left/right/up_left/up_right/down_left/down_right/stop；"
    "w1,w2 是两个关节角速度，取值范围 [-0.5, 0.5]。"
)


def parse_teacher_reply(text, st):
    """把模型回复解析成 (w1,w2)；解析不了就抛 ValueError（由调用方回退）。"""
    raw = str(text or "").strip()
    s, e = raw.find("{"), raw.rfind("}")
    obj = None
    if s >= 0 and e > s:
        try:
            obj = json.loads(raw[s:e + 1])
        except ValueError:
            obj = None
    if isinstance(obj, dict):
        if "w1" in obj and "w2" in obj:
            w1, w2 = float(obj["w1"]), float(obj["w2"])
            cap = float(st["max_omega"])
            return (max(-cap, min(cap, w1)), max(-cap, min(cap, w2)))
        act = obj.get("action", obj.get("direction", obj.get("dir")))
        if act is not None and str(act).strip().lower() in DIRS:
            return _dir_to_omega(st, str(act).strip().lower())
    low = raw.lower()
    for word in ("up_left", "up_right", "down_left", "down_right",
                 "up-left", "up-right", "down-left", "down-right",
                 "up", "down", "left", "right", "stop",
                 "左上", "右上", "左下", "右下", "上", "下", "左", "右", "停"):
        if word in low:
            return _dir_to_omega(st, word)
    raise ValueError("模型回复里找不到 action 或 w1/w2：%s" % raw[:120])


def _dir_to_omega(st, word, step=0.30):
    """方向意图 -> 一个临时小目标 -> DLS 解析解 -> 关节角速度。"""
    vx, vy = DIRS[word]
    if vx == 0.0 and vy == 0.0:
        return 0.0, 0.0
    tx = st["end_x"] + vx * step
    ty = st["end_y"] + vy * step
    return _goal_omega(st["theta1"], st["theta2"], st["end_x"], st["end_y"],
                       tx, ty, st["max_omega"], GOAL_D0)


class LMStudioTeacher(Teacher):
    """本地 LLM 老师。异步节流请求 + 结果缓存 + DLS 兜底。"""

    name = "LM Studio 老师（本地模型 · 决策蒸馏）"
    desc = "本地模型给方向意图，经解析层转成老师指令；服务不可用时回退 DLS。"

    def __init__(self, every=30, temperature=0.1, timeout=8.0,
                 fallback=True, base_url=None, model=None, port=1234):
        cfg = lm.load_config()
        self.base_url = lm.normalize_base(base_url or cfg["base_url"])
        if base_url is None and int(port) != 1234:
            self.base_url = "http://127.0.0.1:%d/v1" % int(port)
        self.model = model if model is not None else cfg["model"]
        self.every = max(1, int(every))
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.fallback = bool(fallback)
        self._dls = DLSTeacher()
        self._lock = threading.Lock()
        self._busy = False
        self._tick = 0
        self._cache = None
        self._cache_tick = 0
        self.calls = 0
        self.hits = 0
        self.last_source = "dls"
        self.last_error = ""
        self.last_latency = 0.0

    def info(self):
        with self._lock:
            return ("请求 %d / 命中 %d · 当前来源 %s · 上次 %.0f ms%s"
                    % (self.calls, self.hits, self.last_source,
                       self.last_latency * 1000,
                       (" · " + self.last_error[:60]) if self.last_error else ""))

    def command(self, st):
        self._tick += 1
        if self._tick % self.every == 0 and not self._busy:
            self._ask_async(dict(st))
        with self._lock:
            cached, age = self._cache, self._tick - self._cache_tick
        # 缓存最多用两个提问周期，过期就交给解析解，避免拿着旧指令乱转
        if cached is not None and age <= self.every * 2:
            return cached
        if self.fallback:
            return self._dls.command(st)
        return 0.0, 0.0

    def _ask_async(self, st):
        self._busy = True
        threading.Thread(target=self._ask, args=(st,), daemon=True).start()

    def _ask(self, st):
        t0 = time.perf_counter()
        user = json.dumps({
            "theta1": round(st["theta1"], 3), "theta2": round(st["theta2"], 3),
            "end": [round(st["end_x"], 3), round(st["end_y"], 3)],
            "target": [round(st["target_x"], 3), round(st["target_y"], 3)],
            "max_omega": st["max_omega"],
        }, ensure_ascii=False)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "状态：" + user + "\n下一拍选哪个方向或角速度？"},
        ]
        try:
            model = self.model
            if not model:
                models = lm.list_models(self.base_url, timeout=min(2.0, self.timeout))
                if not models:
                    raise lm.LMStudioError("服务里没有已加载模型")
                model = models[0]
                self.model = model
            reply = lm.chat(messages, base_url=self.base_url, model=model,
                            temperature=self.temperature, timeout=self.timeout)
            w1, w2 = parse_teacher_reply(reply, st)
            with self._lock:
                self._cache = (w1, w2)
                self._cache_tick = self._tick
                self.calls += 1
                self.hits += 1
                self.last_source = "lm"
                self.last_error = ""
                self.last_latency = time.perf_counter() - t0
        except Exception as exc:
            with self._lock:
                self.calls += 1
                self.last_source = "dls" if self.fallback else "zero"
                self.last_error = "%s: %s" % (type(exc).__name__, str(exc)[:120])
                self.last_latency = time.perf_counter() - t0
        finally:
            self._busy = False


PLUGIN = {
    "id": "lmstudio",
    "name": "LM Studio 老师（本地模型 · 决策蒸馏）",
    "description": "本地模型输出方向意图，经解析层转成老师指令；异步请求、失败自动回退 DLS。"
                   "接口地址在「AI 对话」窗口设置（vis/lmstudio.json）。",
    "params": [
        {"key": "every", "label": "每 N tick 问一次", "type": "float",
         "min": 1.0, "max": 300.0, "step": 1.0, "default": 30.0},
        {"key": "temperature", "label": "采样温度", "type": "float",
         "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.1},
        {"key": "timeout", "label": "单次超时（秒）", "type": "float",
         "min": 1.0, "max": 60.0, "step": 1.0, "default": 8.0},
        {"key": "fallback", "label": "失败回退 DLS 解析解", "type": "bool",
         "default": True},
        {"key": "port", "label": "服务端口（改了才覆盖配置）", "type": "float",
         "min": 1.0, "max": 65535.0, "step": 1.0, "default": 1234.0},
    ],
    "create": lambda p: LMStudioTeacher(
        every=p.get("every", 30), temperature=p.get("temperature", 0.1),
        timeout=p.get("timeout", 8.0), fallback=p.get("fallback", True),
        port=p.get("port", 1234)),
}
