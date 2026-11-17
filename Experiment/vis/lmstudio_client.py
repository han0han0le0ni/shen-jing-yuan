"""LM Studio 本地模型接入层：OpenAI 兼容 HTTP，仅用 Python 标准库。

LM Studio 开启「Start Server」后默认监听 http://127.0.0.1:1234 ，提供：
    GET  /v1/models               -> {"data": [{"id": "模型名"}, ...]}
    POST /v1/chat/completions     -> {"choices": [{"message": {"content": "..."}}]}

本模块只负责「一个接口」：连接配置、模型列表、单轮/多轮对话。
所有网络调用都是同步的，调用方负责放到后台线程（对话窗口与老师插件已各自处理）。

配置写在同目录 lmstudio.json，供对话窗口与蒸馏老师插件共用：
    {"base_url": "http://127.0.0.1:1234/v1", "model": "", "temperature": 0.2,
     "timeout": 8.0}
"""
import json
import os
import urllib.error
import urllib.request

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(VIS_DIR, "lmstudio.json")

DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:1234/v1",
    "model": "",              # 空 = 自动选服务端第一个已加载模型
    "temperature": 0.2,
    "timeout": 8.0,
}


class LMStudioError(Exception):
    """可读的连接/协议错误（不把 traceback 抛给 GUI）。"""


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            cfg.update({k: raw[k] for k in DEFAULT_CONFIG if k in raw})
    except (OSError, ValueError):
        pass
    cfg["base_url"] = normalize_base(cfg.get("base_url", ""))
    return cfg


def save_config(cfg):
    """只保存已知键，避免把临时字段写进配置。"""
    out = dict(DEFAULT_CONFIG)
    out.update({k: cfg[k] for k in DEFAULT_CONFIG if k in cfg})
    out["base_url"] = normalize_base(out["base_url"])
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def normalize_base(url):
    """允许用户填 host 或 /v1 两种形式。"""
    u = str(url or DEFAULT_CONFIG["base_url"]).strip().rstrip("/")
    if not u:
        return DEFAULT_CONFIG["base_url"]
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "http://" + u
    if not u.endswith("/v1"):
        u += "/v1"
    return u


def _request(base_url, path, payload=None, timeout=8.0):
    url = normalize_base(base_url) + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LMStudioError("HTTP %d：%s" % (exc.code, detail)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LMStudioError(
            "连不上 %s —— 请在 LM Studio 里点 Start Server（默认端口 1234）。%s"
            % (normalize_base(base_url), exc)) from exc
    try:
        return json.loads(body)
    except ValueError as exc:
        raise LMStudioError("返回不是 JSON：%s" % body[:200]) from exc


def list_models(base_url=None, timeout=2.0):
    """返回已加载模型 id 列表；服务未启动时抛 LMStudioError。"""
    cfg = load_config()
    obj = _request(base_url or cfg["base_url"], "/models", timeout=timeout)
    data = obj.get("data", []) if isinstance(obj, dict) else []
    return [str(m.get("id", "")) for m in data if isinstance(m, dict) and m.get("id")]


def _pick_model(model, base_url, timeout):
    if model:
        return model
    models = list_models(base_url, timeout=timeout)
    if not models:
        raise LMStudioError("服务已连接，但没有已加载的模型（先在 LM Studio 里 Load 一个模型）")
    return models[0]


def chat(messages, base_url=None, model=None, temperature=None, timeout=None):
    """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...] -> 回复文本。"""
    cfg = load_config()
    base = base_url or cfg["base_url"]
    t = float(cfg["timeout"] if timeout is None else timeout)
    temp = float(cfg["temperature"] if temperature is None else temperature)
    chosen = _pick_model(model if model is not None else cfg["model"], base, t)
    payload = {
        "model": chosen,
        "messages": [dict(m) for m in messages],
        "temperature": temp,
        "stream": False,
    }
    obj = _request(base, "/chat/completions", payload=payload, timeout=t)
    try:
        return str(obj["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LMStudioError("响应缺少 choices[0].message.content：%s" % str(obj)[:200]) from exc


def probe(base_url=None, timeout=2.0):
    """连接自检：返回 {"ok", "models", "model", "error"}，不抛异常。"""
    base = normalize_base(base_url or load_config()["base_url"])
    try:
        models = list_models(base, timeout=timeout)
        cfg = load_config()
        model = cfg["model"] if cfg["model"] in models else (models[0] if models else "")
        return {"ok": True, "models": models, "model": model, "error": ""}
    except LMStudioError as exc:
        return {"ok": False, "models": [], "model": "", "error": str(exc)}


SYSTEM_PROMPT = (
    "你运行在「脉冲神经网络 · 二连杆机械臂」实验台旁，是本地助手。"
    "回答要简短、具体，优先用中文。涉及仿真时，数字以用户给出的状态为准，不要编造。"
)
