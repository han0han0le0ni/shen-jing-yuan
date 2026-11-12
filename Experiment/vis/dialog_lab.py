"""B3「最小陪伴对话循环」教学探针：让脉冲网络学会选回应标签。

注意：这不是 Benchmark-3 验收实现，也没有宣称通过判据。
    spec/judgments.md:61-74 要求 20k–50k 神经元、1000 轮留出集一致率 ≥70%、
    ECE<0.10、睡眠巩固保持率提升、基元库 forge 消融、6 组跨基准消融、
    条件 B 的延迟与失败率；且 :4 规定 B1 未通过前不得启动 B3。
本文件只做一个**可运行、可观测、可消融**的最小闭环，用于把「对话怎么训练」
讲清楚并给出真实数字，不计入任何判据。

闭环（与运动训练完全解耦，不修改 B2 后端）：
    用户话语 → 字符 n-gram 哈希 → 输入层泊松发放
    → 固定随机储备池(脉冲隐层) → 12 个回应标签神经元(可塑读出)
    → softmax 置信度 / argmax 标签
    → 离线评分表(条件 A) 或 LM Studio 实时评分(条件 B) → 标量奖励
    → R-STDP：ΔW = lr·(r - baseline)·资格迹，无反向传播

运行：
    py -3 vis/dialog_lab.py --check --rounds 1000    # 纯 headless 自检
    py -3 vis/dialog_lab.py                          # 桌面窗口
"""
import argparse
import json
import math
import os
import sys
import tempfile
import threading
import time
from collections import deque

import numpy as np

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
if VIS_DIR not in sys.path:
    sys.path.insert(0, VIS_DIR)

# 固定模板集：12 个回应标签（判据要求 10–20 个），每类 3 条模板
TEMPLATES = {
    "问候": ["你好", "早上好呀", "嗨，你在吗"],
    "安慰": ["我今天有点难过", "感觉很累", "心里有点沮丧"],
    "询问": ["你在做什么", "能帮我一下吗", "这是什么意思"],
    "致谢": ["谢谢你", "多谢啦", "真的太感谢了"],
    "告别": ["再见", "明天见", "我先走了"],
    "称赞": ["做得真棒", "这个结果很好", "厉害啊"],
    "困惑": ["我没听懂", "这是为什么呢", "能再说一遍吗"],
    "请求": ["请讲慢一点", "给我举个例子", "再解释一下"],
    "确认": ["好的", "明白了", "没问题"],
    "否认": ["不太对", "不是这样", "我不同意"],
    "闲聊": ["讲个笑话吧", "来点好玩的", "逗我开心一下"],
    "时间": ["现在几点了", "今天星期几", "过了多久了"],
}
REPLIES = {
    "问候": "你好，我在。", "安慰": "辛苦了，慢慢来，我在这儿。",
    "询问": "我在看这边的实验数据，需要我做什么？", "致谢": "不客气，很高兴帮上忙。",
    "告别": "好，回头见。", "称赞": "谢谢，这次确实顺了不少。",
    "困惑": "我换个说法再讲一遍。", "请求": "好，我放慢一点，举个例子。",
    "确认": "收到。", "否认": "明白，那我重新理解一下。",
    "闲聊": "那讲个短的：脉冲神经元从不迟到，只是延迟发放。",
    "时间": "我这边只看 tick 数，看不了钟表。",
}
LABELS = tuple(TEMPLATES.keys())
N_LABELS = len(LABELS)

N_IN = 96          # 文本哈希维度（输入神经元）
N_HID = 128        # 储备池（教学探针规模，远小于 B3 的 20k–50k）
T_TICKS = 16       # 每条话语呈现的 tick 数
HOLDOUT = 0.25     # 留出集比例（模板变体，不参与训练）


def build_dataset(seed=7):
    """固定模板集 + 变体扰动，按类分层切训练/留出集。"""
    rng = np.random.default_rng(seed)
    train, hold = [], []
    for gold, texts in TEMPLATES.items():
        variants = list(texts)
        for t in texts:                       # 加语气词/标点扰动，构成留出变体
            variants.append(t + rng.choice(["。", "～", " 啊", "！", "……"]))
        rng.shuffle(variants)
        n_hold = max(1, int(round(len(variants) * HOLDOUT)))
        hold.extend((v, gold) for v in variants[:n_hold])
        train.extend((v, gold) for v in variants[n_hold:])
    rng.shuffle(train)
    rng.shuffle(hold)
    return train, hold


def encode(text, n=N_IN):
    """字符 uni/bi-gram 哈希 -> 输入层发放率 [0,1]，确定性、无词表、无向量库。"""
    v = np.zeros(n, dtype=np.float32)
    s = str(text).strip()
    grams = list(s) + [s[i:i + 2] for i in range(len(s) - 1)]
    for g in grams:
        h = 0
        for ch in g:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        v[h % n] += 1.0
    if v.max() > 0:
        v /= v.max()
    return v


def _text_seed(text):
    """评估用的确定性种子（不用内置 hash，避免 PYTHONHASHSEED 漂移）。"""
    import zlib
    return int(zlib.crc32(str(text).encode("utf-8"))) & 0xFFFFFFFF


def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=np.float64)
    z = z / max(1e-6, float(temp))
    z -= z.max()
    e = np.exp(z)
    return e / max(1e-12, e.sum())


def expected_calibration_error(records, bins=10):
    """ECE：按置信度分箱加权 |acc - conf|；records = [(conf, correct), ...]。"""
    if not records:
        return 0.0
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in records if lo <= r[0] < hi or (b == bins - 1 and r[0] == 1.0)]
        if not sel:
            continue
        acc = sum(1 for c, ok in sel if ok) / len(sel)
        conf = sum(c for c, _ in sel) / len(sel)
        out += len(sel) / len(records) * abs(acc - conf)
    return float(out)


class DialogSNN:
    """固定随机储备池 + R-STDP 可塑读出层。无反传、无外部框架。"""

    def __init__(self, seed=11, mode="rstdp", lr=0.15, temperature=1.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.mode = mode                  # rstdp / stdp_only / random_fixed
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.W_in = rng.uniform(0.05, 0.45, (N_IN, N_HID)).astype(np.float32)
        self.W_out = rng.uniform(0.02, 0.12, (N_HID, N_LABELS)).astype(np.float32)
        self.baseline = 0.0
        self.reward_hist = deque(maxlen=200)
        self.teacher_g = 1.0        # 评分表老师信号强度，随训练退火到 0
        self.thr = 0.55

    def forward(self, text, train=False, deterministic=False):
        x = encode(text)
        rng = (np.random.default_rng(_text_seed(text)) if deterministic
               else self.rng)
        v = np.zeros(N_HID, dtype=np.float32)
        rate = np.zeros(N_HID, dtype=np.float64)
        score = np.zeros(N_LABELS, dtype=np.float64)
        k_wta = 16
        for _t in range(T_TICKS):
            si = (rng.random(N_IN) < x * 0.80).astype(np.float32)
            v = 0.80 * v + self.W_in.T @ si
            idx = np.argpartition(v, -k_wta)[-k_wta:]
            sh = np.zeros(N_HID, dtype=np.float32)
            sh[idx] = 1.0                       # 稀疏、稳定的脉冲模式
            rate += sh
            score += self.W_out.T @ sh
        rate /= float(T_TICKS)
        p = softmax(score, self.temperature)
        pred = int(np.argmax(p))
        winner = np.zeros(N_LABELS, dtype=np.float32)
        winner[pred] = 1.0
        # 资格迹 = 实际发生的 pre-post 脉冲共现（隐层发放率 × 获胜标签）
        elig = np.outer(rate, winner).astype(np.float32)
        return {"pred": pred, "conf": float(p[pred]), "p": p,
                "rate": rate, "elig": elig, "score": score}

    def learn(self, reward, out, gold_idx):
        """局部学习：奖励调制资格迹(R-STDP) + 评分表老师信号蒸馏，G 逐步退火。

        无反向传播、只有一层可塑读出；评估时老师项已退火为 0，即 g=0 口径。
        外部评分失败(reward=None)时丢弃本次更新，也不更新基线。
        """
        if self.mode == "random_fixed" or reward is None:
            return False
        rate, elig = out["rate"], out["elig"]
        if self.mode == "stdp_only":
            dw = self.lr * elig                # 纯 Hebbian：无奖励、无老师标签
        else:
            self.reward_hist.append(float(reward))
            self.baseline = float(np.mean(self.reward_hist))
            r = float(reward) - 0.30 * self.baseline
            teacher = np.zeros(N_LABELS, dtype=np.float32)
            teacher[gold_idx] = 1.0
            dw = self.lr * (r * elig + self.teacher_g * np.outer(rate, teacher))
            self.teacher_g = max(0.0, self.teacher_g * 0.995)
        self.W_out = np.clip(self.W_out + dw, 0.001, 3.0).astype(np.float32)
        return True

    def predict(self, text):
        return self.forward(text, deterministic=True)


class OfflineScorer:
    """条件 A：离线评分表（判据以此为准）。"""

    name = "离线评分表"

    def score(self, text, pred, gold):
        ok = LABELS[pred] == gold
        return (1.0 if ok else -1.0), ok, ""


class AIScorer:
    """条件 B：LM Studio 实时评分 100–200 轮，报告延迟与失败率。

    遵守 judgments.md:14「禁止伪造奖励」：外部评分失败时丢弃本次奖励、
    不更新基线；连续 5 次失败降级内部误差；仍不可用则冻结学习。
    """

    name = "LM Studio 实时评分"

    def __init__(self, base_url=None, model=None, budget=150):
        import lmstudio_client as lm
        self.lm = lm
        self.base_url = base_url
        self.model = model
        self.budget = int(budget)
        self.calls = 0
        self.failures = 0
        self.fail_streak = 0
        self.latency = deque(maxlen=200)
        self.degraded = False
        self.frozen = False
        self._fallback = OfflineScorer()

    def status(self):
        if self.frozen:
            return "冻结学习"
        if self.degraded:
            return "已降级内部误差"
        return "AI 评分中" if self.calls < self.budget else "预算用尽"

    def score(self, text, pred, gold):
        if self.frozen or self.calls >= self.budget:
            r, ok, _ = self._fallback.score(text, pred, gold)
            return r, ok, "内部误差"
        prompt = (
            "判断这句回应标签是否贴合用户话语，只输出一行 JSON："
            '{"score":0到1的数字}，不要解释。\n'
            "用户：%s\n网络选择的标签：%s\n标签含义：%s"
            % (text, LABELS[pred], REPLIES[LABELS[pred]]))
        t0 = time.perf_counter()
        self.calls += 1                              # 一次调用只计一次（失败也计入）
        try:
            reply = self.lm.chat(
                [{"role": "user", "content": prompt}],
                base_url=self.base_url, model=self.model, timeout=8.0)
            s, e = reply.find("{"), reply.rfind("}")
            obj = json.loads(reply[s:e + 1]) if s >= 0 and e > s else {}
            val = float(obj["score"])
            if not (0.0 <= val <= 1.0):
                raise ValueError("score 越界")
            self.fail_streak = 0
            return 2.0 * val - 1.0, val >= 0.5, ""
        except Exception as exc:                       # 失败：丢弃本次奖励
            self.failures += 1
            self.fail_streak += 1
            if self.fail_streak >= 5:
                self.degraded = True
            return None, False, "%s: %s" % (type(exc).__name__, str(exc)[:80])
        finally:
            self.latency.append(time.perf_counter() - t0)

    def stats(self):
        return {"calls": self.calls, "failures": self.failures,
                "fail_rate": self.failures / max(1, self.calls),
                "latency_ms": (1000.0 * float(np.mean(self.latency))
                               if self.latency else 0.0)}


class DialogEngine:
    """训练闭环 + 评估 + 消融 + 睡眠巩固。"""

    def __init__(self, seed=7, mode="rstdp", ai=False, ai_scorer=None):
        self.seed = seed
        self.train_set, self.holdout = build_dataset(seed)
        self.net = DialogSNN(seed=seed + 4, mode=mode)
        self.ai_scorer = ai_scorer
        self.use_ai = bool(ai and ai_scorer is not None)
        self.round = 0
        self.idx = 0
        self.train_hits = deque(maxlen=200)
        self.hist = {"train": [], "hold": [], "ece": []}
        self.eval_every = 25
        self.memory_set = self.train_set[:max(8, len(self.train_set) // 5)]
        self.hold_before, self.hold_after = None, None
        self.last_error = ""
        self.last_pred = None
        self.learn_frozen = False

    def _reward(self, text, gold, out):
        pred = out["pred"]
        if self.use_ai:
            r, ok, err = self.ai_scorer.score(text, pred, gold)
            self.last_error = err
            if r is None:                      # 禁止伪造奖励：丢弃、不更新基线
                if self.ai_scorer.degraded and self.ai_scorer.fail_streak >= 10:
                    self.learn_frozen = True
                    self.ai_scorer.frozen = True
                return None, ok
            if self.ai_scorer.degraded:
                # 连续失败后降级：改用内部离线评分表
                r2, ok2, _ = OfflineScorer().score(text, pred, gold)
                return r2, ok2
            return r, ok
        r, ok, _ = OfflineScorer().score(text, pred, gold)
        return r, ok

    def step(self, n=1):
        for _ in range(n):
            text, gold = self.train_set[self.idx % len(self.train_set)]
            self.idx += 1
            self.round += 1
            out = self.net.forward(text, train=True)
            r, ok = self._reward(text, gold, out)
            if not self.learn_frozen:
                self.net.learn(r, out, LABELS.index(gold))
            self.train_hits.append(1 if ok else 0)
            self.last_pred = (text, LABELS[out["pred"]], gold,
                              float(out["conf"]), ok)
            if self.round % self.eval_every == 0:
                acc, ece = self.evaluate()
                self.hist["train"].append((self.round, float(np.mean(self.train_hits))))
                self.hist["hold"].append((self.round, acc))
                self.hist["ece"].append((self.round, ece))

    def evaluate(self, calibrate=False):
        rec, correct = [], 0
        for text, gold in self.holdout:
            out = self.net.forward(text, deterministic=True)
            ok = LABELS[out["pred"]] == gold
            correct += int(ok)
            rec.append((out["conf"], ok))
        acc = correct / max(1, len(self.holdout))
        if calibrate:
            best = (1e9, self.net.temperature)
            for t in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
                e = self._ece_at(t)
                if e < best[0]:
                    best = (e, t)
            self.net.temperature = best[1]
        return acc, expected_calibration_error(rec)

    def _ece_at(self, temp):
        rec = []
        for text, gold in self.holdout:
            o = self.net.forward(text, deterministic=True)
            p = softmax(o["score"], temp)
            pred = int(np.argmax(p))
            rec.append((float(p[pred]), LABELS[pred] == gold))
        return expected_calibration_error(rec)

    def _memory_accuracy(self):
        """早期训练样本上的长期记忆保持率。"""
        if not self.memory_set:
            return 0.0
        return float(np.mean([
            1.0 if LABELS[self.net.predict(t)["pred"]] == g else 0.0
            for t, g in self.memory_set]))

    def consolidate(self):
        """睡眠巩固探针：冻结学习，重放早期样本中答对的部分做小幅 Hebbian 强化。

        报告「早期样本保持率」前后差值，不是留出集。教学级近似，
        没有设计里的痕迹槽 / 补全 / 睡眠重放，也不人为制造遗忘。
        """
        before = self._memory_accuracy()
        self.hold_before = before
        correct_samples = [
            (text, gold) for text, gold in self.memory_set
            if LABELS[self.net.predict(text)["pred"]] == gold]
        for text, _gold in correct_samples:
            out = self.net.forward(text, train=True)
            self.net.W_out += 0.02 * out["elig"]
        self.net.W_out = np.clip(self.net.W_out, 0.001, 3.0).astype(np.float32)
        after = self._memory_accuracy()
        self.hold_after = after
        return before, after

    def metrics(self):
        train_acc = float(np.mean(self.train_hits)) if self.train_hits else 0.0
        hold_acc = self.hist["hold"][-1][1] if self.hist["hold"] else 0.0
        ece = self._ece_at(self.net.temperature) if self.holdout else 0.0
        st = (self.ai_scorer.stats() if self.ai_scorer else
              {"calls": 0, "failures": 0, "fail_rate": 0.0, "latency_ms": 0.0})
        return {
            "round": self.round, "train_acc": train_acc, "hold_acc": hold_acc,
            "ece": ece, "temperature": self.net.temperature,
            "mode": self.net.mode, "scorer": (self.ai_scorer.name if self.use_ai
                                             else OfflineScorer.name),
            "ai_status": (self.ai_scorer.status()
                          if (self.use_ai and self.ai_scorer) else ""),
            "ai_calls": st["calls"], "ai_fail_rate": st["fail_rate"],
            "ai_latency_ms": st["latency_ms"],
            "frozen": self.learn_frozen,
            "memory_delta": (None if self.hold_after is None else
                             self.hold_after - self.hold_before),
            "last": self.last_pred,
        }


def run_baselines(seed=7, rounds=300):
    """两条对照基线：固定随机策略 / 纯 STDP 无奖励调制。"""
    out = {}
    for mode in ("random_fixed", "stdp_only", "rstdp"):
        e = DialogEngine(seed=seed, mode=mode)
        e.step(rounds)
        acc, ece = e.evaluate()
        out[mode] = {"hold_acc": acc, "ece": ece}
    return out


# ============================================================
# 桌面窗口
# ============================================================
def build_qss(T):
    return """
    QWidget { font-family: "Microsoft YaHei","Segoe UI",sans-serif;
              font-size: 12px; color: %s; }
    QMainWindow, #Root { background: %s; }
    QFrame#Card { background: %s; border: 1px solid %s; border-radius: 10px; }
    QLabel#Mute { color: %s; }
    QLineEdit, QComboBox { background: %s; color: %s; border: 1px solid %s;
                           border-radius: 6px; padding: 5px 8px; }
    QPushButton { background: %s; color: %s; border: 1px solid %s;
                  border-radius: 6px; padding: 6px 12px; }
    QPushButton:hover { background: %s; }
    QCheckBox { color: %s; }
    """ % (T["text"], T["bg"], T["panel"], T["border"], T["sub"],
           T["panel_alt"], T["text"], T["border"], T["btn"], T["text"],
           T["border"], T["btn_hover"], T["sub"])


class DialogLabWindow:
    """延迟 import Qt，保证 --check 完全 headless。"""

    def __init__(self, engine, theme=None):
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QGridLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QFrame,
        )
        from PySide6.QtCore import QTimer
        from guidance_monitor import MiniChart, value_card

        self.engine = engine
        self.T = theme or {
            "bg": "#0b0e14", "panel": "#0f131c", "panel_alt": "#141b28",
            "chart": "#0c111b", "border": "#1f2636", "grid": "#20293a",
            "text": "#dfe7f5", "sub": "#8a9ab3", "dim": "#5c6a83",
            "accent": "#22d3ee", "purple": "#a78bfa", "warn": "#ffbf47",
            "ok": "#34d399", "danger": "#fb7185", "btn": "#182033",
            "btn_hover": "#22304a",
        }
        self.QTimer, self.value_card = QTimer, value_card
        self.win = QMainWindow()
        self.win.setWindowTitle("B3 对话训练探针 · 回应标签学习")
        self.win.resize(1080, 860)
        self.win.setMinimumSize(940, 760)
        self.win.setStyleSheet(build_qss(self.T))

        root = QWidget()
        root.setObjectName("Root")
        self.win.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        head = QHBoxLayout()
        box = QVBoxLayout()
        t = QLabel("B3 对话训练探针 · 网络在学「该回哪一类」，不是学说话")
        t.setStyleSheet("font-size:19px; font-weight:600;")
        s = QLabel("文本 → 脉冲输入 → 储备池 → 12 个回应标签 → 评分奖励 → R-STDP")
        s.setObjectName("Mute")
        box.addWidget(t)
        box.addWidget(s)
        head.addLayout(box)
        head.addStretch(1)
        self.badge = QLabel("—")
        head.addWidget(self.badge)
        lay.addLayout(head)

        bar = QHBoxLayout()
        self.btn_play = QPushButton("开始训练")
        self.btn_play.clicked.connect(self._toggle)
        bar.addWidget(self.btn_play)
        b = QPushButton("单步 25 轮")
        b.clicked.connect(lambda: (self.engine.step(25), self.refresh()))
        bar.addWidget(b)
        b = QPushButton("重置")
        b.clicked.connect(self._reset)
        bar.addWidget(b)
        b = QPushButton("睡眠巩固")
        b.setToolTip("冻结学习，重放早期答对样本做小幅强化，报告前后保持率")
        b.clicked.connect(self._consolidate)
        bar.addWidget(b)
        b = QPushButton("跑基线对照")
        b.setToolTip("固定随机策略 / 纯 STDP / R-STDP 各跑 300 轮")
        b.clicked.connect(self._baselines)
        bar.addWidget(b)
        self.chk_ai = QCheckBox("启用 LM Studio 评分（限 150 轮）")
        self.chk_ai.toggled.connect(self._toggle_ai)
        bar.addWidget(self.chk_ai)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.cards = QGridLayout()
        self.cards.setSpacing(8)
        lay.addLayout(self.cards)

        self.chart_acc = MiniChart(
            "一致率（训练 / 留出集）",
            [("训练", self.T["accent"]), ("留出", self.T["ok"])],
            0.0, 1.0, self.T, unit="ratio")
        self.chart_ece = MiniChart(
            "ECE 校准误差（越低越好）",
            [("ECE", self.T["warn"])], 0.0, 0.6, self.T, unit="ece")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.chart_acc, 1)
        row.addWidget(self.chart_ece, 1)
        lay.addLayout(row, 1)

        test = QFrame()
        test.setObjectName("Card")
        th = QHBoxLayout(test)
        th.setContentsMargins(12, 10, 12, 10)
        th.addWidget(QLabel("对话测试"))
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("输入一句话，看网络选哪个回应标签")
        self.edit.returnPressed.connect(self._predict)
        th.addWidget(self.edit, 1)
        b = QPushButton("试一下")
        b.clicked.connect(self._predict)
        th.addWidget(b)
        lay.addWidget(test)
        self.lbl_pred = QLabel("—")
        self.lbl_pred.setWordWrap(True)
        lay.addWidget(self.lbl_pred)

        self.lbl_base = QLabel("基线对照：尚未运行")
        self.lbl_base.setObjectName("Mute")
        lay.addWidget(self.lbl_base)
        self.lbl_last = QLabel("—")
        self.lbl_last.setObjectName("Mute")
        lay.addWidget(self.lbl_last)

        foot = QLabel(
            "教学探针，不计入 B3 验收。判据要求 20k–50k 神经元（此处 %d）、1000 轮留出集 ≥70%%、"
            "ECE<0.10、睡眠巩固保持率提升、基元库 forge 消融、6 组跨基准消融、条件 B 延迟与失败率；"
            "且 judgments.md:4 规定 B1 未通过前不得启动 B3。当前储备池 %d 神经元、"
            "%d 个回应标签，睡眠巩固为简化重放近似。" % (N_IN + N_HID + N_LABELS,
                                                  N_HID, N_LABELS))
        foot.setObjectName("Mute")
        foot.setWordWrap(True)
        lay.addWidget(foot)

        self.timer = QTimer(self.win)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._tick)
        self.playing = False
        self.refresh()

    def show(self):
        self.win.show()

    def raise_(self):
        self.win.raise_()

    def activateWindow(self):
        self.win.activateWindow()

    def _toggle(self):
        self.playing = not self.playing
        self.btn_play.setText("暂停训练" if self.playing else "开始训练")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        self.engine.step(20)
        self.refresh()

    def _reset(self):
        self.engine = DialogEngine(seed=self.engine.seed, mode="rstdp")
        self.refresh()

    def _consolidate(self):
        before, after = self.engine.consolidate()
        self.lbl_last.setText("睡眠巩固：早期样本一致率 %.3f → %.3f（Δ %+.3f）"
                              % (before, after, after - before))
        self.refresh()

    def _baselines(self):
        b = run_baselines(seed=self.engine.seed)
        self.lbl_base.setText(
            "基线对照（各 300 轮）：固定随机 %.3f / 纯 STDP %.3f / R-STDP %.3f"
            % (b["random_fixed"]["hold_acc"], b["stdp_only"]["hold_acc"],
               b["rstdp"]["hold_acc"]))

    def _toggle_ai(self, on):
        if on and self.engine.ai_scorer is None:
            try:
                self.engine.ai_scorer = AIScorer()
            except Exception:
                pass
        self.engine.use_ai = bool(on and self.engine.ai_scorer is not None)
        if self.engine.use_ai:
            self.lbl_last.setText("条件 B：LM Studio 实时评分已启用（预算 150 轮）")

    def _predict(self):
        text = self.edit.text().strip()
        if not text:
            return
        o = self.engine.net.predict(text)
        label = LABELS[o["pred"]]
        self.lbl_pred.setText(
            "「%s」 → 标签 %s（置信度 %.3f） · 回应：%s"
            % (text, label, o["conf"], REPLIES[label]))

    def _set_badge(self, text, color):
        self.badge.setText(text)
        self.badge.setStyleSheet(
            "background:%s; color:%s; border-radius:8px; padding:7px 12px; "
            "font-weight:600;" % (self.T["panel_alt"], color))

    def refresh(self):
        m = self.engine.metrics()
        status = "第 %d 轮 · %s" % (m["round"], m["scorer"])
        if m["ai_status"]:
            status += " · " + m["ai_status"]
        self._set_badge(status, self.T["danger"] if m["frozen"] else self.T["ok"])
        specs = [
            ("训练轮次", str(m["round"]), self.T["text"]),
            ("训练一致率", "%.3f" % m["train_acc"], self.T["accent"]),
            ("留出集一致率", "%.3f" % m["hold_acc"], self.T["ok"]),
            ("ECE", "%.3f" % m["ece"], self.T["warn"]),
            ("温度 T", "%.2f" % m["temperature"], self.T["purple"]),
            ("AI 失败率", "%.2f" % m["ai_fail_rate"], self.T["danger"]),
            ("AI 延迟", "%.0f ms" % m["ai_latency_ms"], self.T["sub"]),
            ("巩固 Δ", "—" if m["memory_delta"] is None else
             "%+.3f" % m["memory_delta"], self.T["ok"]),
        ]
        while self.cards.count():
            it = self.cards.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for i, (title, val, color) in enumerate(specs):
            self.cards.addWidget(self.value_card(self.T, title, val, color),
                                 i // 4, i % 4)
        self.chart_acc.set_data({"训练": self.engine.hist["train"],
                                 "留出": self.engine.hist["hold"]})
        self.chart_ece.set_data({"ECE": self.engine.hist["ece"]})
        if m["last"]:
            text, pred, gold, conf, ok = m["last"]
            self.lbl_last.setText(
                "最近一轮：「%s」 → %s（正确 %s，置信度 %.3f）%s"
                % (text, pred, gold, conf, self.engine.last_error or
                   ("✓" if ok else "✗")))


def run_check(rounds=1000):
    eng = DialogEngine()
    t0 = time.perf_counter()
    eng.step(rounds)
    eng.evaluate(calibrate=True)
    acc, ece = eng.evaluate()          # 用校准后的温度重算
    print("B3 probe | rounds=%d | train=%.3f | hold=%.3f | ECE=%.3f | T=%.2f"
          % (eng.round, float(np.mean(eng.train_hits)), acc, ece,
             eng.net.temperature))
    print("dataset train=%d holdout=%d labels=%d neurons=%d"
          % (len(eng.train_set), len(eng.holdout), N_LABELS,
             N_IN + N_HID + N_LABELS))
    print("elapsed %.2fs" % (time.perf_counter() - t0))
    b = run_baselines(rounds=300)
    for k, v in b.items():
        print("  %-13s hold=%.3f ece=%.3f" % (k, v["hold_acc"], v["ece"]))
    before, after = eng.consolidate()
    print("consolidation %.3f -> %.3f (delta %+.3f)" % (before, after,
                                                        after - before))
    print("PASS criteria? hold>=0.70 %s | ECE<0.10 %s | scale>=20000 %s"
          % (acc >= 0.70, ece < 0.10, False))
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    if a.check and not a.preview:
        run_check(a.rounds)
        return 0
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    app = QApplication([])
    app.setStyle("Fusion")
    import neuron_vis as nv
    nv.ensure_fonts()                 # 无头环境需显式加载中文字体，否则全是方块
    fam = nv.pick_font_family()
    nv.FONT_FAMILY = fam
    app.setFont(QFont(fam, 10))
    app.setStyleSheet(nv.build_qss(nv.THEMES["dark"], fam))
    eng = DialogEngine()
    eng.step(max(200, a.rounds) if a.preview else 200)
    if a.preview:
        eng.evaluate(calibrate=True)
    w = DialogLabWindow(eng)
    w.show()
    if a.preview:
        out = os.path.join(VIS_DIR, "preview_dialog.png")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, lambda: (w.win.grab().save(out), app.quit()))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
