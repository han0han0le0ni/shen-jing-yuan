# -*- coding: utf-8 -*-
"""独立实验程序：让脉冲网络向外部老师（AI / 算法）学习后逐步自主。

运行：  py -3 Experiment/vis/transfer_lab.py
自检：  py -3 Experiment/vis/transfer_lab.py --check
删除：  直接删掉本文件即可。本程序不修改任何其他源码，不写日志、
        不导出 CSV、不生成截图，只在内存里跑。

这是「学习接口」：任何能根据当前状态给出期望关节角速度 (omega1, omega2)
的算法都能当老师，被脉冲网络通过可塑群体读出蒸馏进来。
  · DLSTeacher      —— 阻尼最小二乘逆运动学（现成解析解，已知可达）
  · TransposeTeacher—— 转置雅可比梯度法（另一种 IK 算法，演示换老师）
  · NullTeacher     —— 不教（基线，证明纯网络单独做不到）
  · FuncTeacher     —— 把你自己的 AI/算法（RL 策略、外部模型、LLM 规划器…）
                       用一行 callable(state)->(w1, w2) 接进来
老师列表见文件底部的 TEACHERS 注册表；想蒸馏新算法，只需新增一个
继承 Teacher、实现 command() 的类即可，无需改动训练/评估循环。

哲学：神经元是用来「学」的，不是神——先学（蒸馏老师），再用
（g=0 纯自主），再创新（DAgger 让网络在自己造成的状态下也被教，
有机会超越老师本身的局限）。验收判据只看 g=0 探针的到达率，
带引导时的到达次数不算数。

蒸馏机制：
  1. 可塑线性群体读出：以「输入层+隐层」脉冲活动为特征，两关节各学
     一组突触权重（NLMS 在线回归）逼近老师命令；推理期 g=0 完全由
     网络活动产生。
  2. DAgger：定期用 g=0 跑训练轮，让网络在自己造成的状态下也被教，
     避免「训练看引导轨迹、评估看自主轨迹」的分布不匹配死锁。
  3. 表现驱动退火 + 缓慢时间表：g=0 探针达标才允许突破 0.30 下限，
     降到 0.02；连续达标才判定「已自主」。
  4. 诚实声明：G 降到 0 只证明网络复现了老师策略；界面同时显示 G 与
     探针到达率，两者背离说明还没真正学会（可能只是背下了引导轨迹）。

G 的含义：omega = (1-G)*omega_net + G*omega_teacher
  G=1 全程老师接管；G=0 网络完全自己找路。
"""
import sys
sys.dont_write_bytecode = True          # 连 __pycache__ 都不留下

import os
import math
import time
import argparse
import random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backend_vec as bv                # 只读取，绝不修改

SIZE = (100, 2000, 50, 50)             # 与可视化程序默认档一致
ARRIVE_R = 0.15
PROBE_TARGETS = ((0.8, 0.6), (-1.0, 0.4), (0.2, -1.0))
PROBE_TICKS = 800
MAX_TRIAL_TICKS = 1500

C_BG = "#0b0e14"
C_CARD = "#121826"
C_LINE = "#222c40"
C_TEXT = "#e6edf7"
C_MUTE = "#8b98ae"
C_GUIDE = "#ffa83c"                    # 引导（解析控制器）
C_AUTO = "#6ee7a8"                     # 自主（网络）
C_TEACH = "#58a6ff"                    # 教师信号 / 余弦


# ============================================================
# 后端：分通道教师信号
# ============================================================
class TransferBackend(bv.VecBackend):
    """在独立文件里子类化，主程序 backend_vec.py 一行都不动。"""

    def __init__(self, seed=12345):
        super().__init__(device="cpu", seed=seed)   # 固定 numpy 路径，状态可快照
        self._teacher_q = []
        self._m1_mask = None
        self._m2_mask = None
        self.teacher_lr = 0.03
        self.teacher_on = True
        self.last_teacher = (0.0, 0.0)
        # 可塑线性群体读出：以「输入层+隐层」的脉冲活动为特征，
        # 两个关节各有一组突触权重。推理期输出完全由网络活动决定。
        self.readout_w = None
        self.readout_on = True
        self.readout_lr = 0.3

    def init(self, *a, **kw):
        super().init(*a, **kw)
        post = self.syn_post
        self._m1_mask = (post >= self.m1_start) & (post < self.m2_start)
        self._m2_mask = (post >= self.m2_start)
        self._hidden_mask = ~(self._m1_mask | self._m2_mask)
        self.readout_w = np.zeros((2, self.m1_start), dtype=np.float64)
        return self

    def _decode_omega(self, start, end):
        """替代 k-WTA 的 pref 均值解码：线性群体读出。"""
        if not self.readout_on or self.readout_w is None:
            return super()._decode_omega(start, end)
        x = self.is_spiking[:self.m1_start]
        j = 0 if start == self.m1_start else 1
        return float(np.dot(self.readout_w[j], x))

    def update_readout(self, t1, t2):
        """NLMS 在线回归：只更新本 tick 真正发放的特征维度（状态门控）。"""
        if not self.readout_on:
            return
        x = self.is_spiking[:self.m1_start].astype(np.float64)
        n = float(x.sum())
        if n < 1.0:
            return
        w = self.readout_w
        p1, p2 = float(w[0] @ x), float(w[1] @ x)
        a = self.readout_lr / (n + 1.0)
        w[0] += a * (t1 - p1) * x
        w[1] += a * (t2 - p2) * x

    def episode_reset(self, tx, ty):
        super().episode_reset(tx, ty)
        self._teacher_q.clear()

    def push_teacher(self, r1, r2):
        self._teacher_q.append((float(r1), float(r2)))

    def _apply_reward(self):
        """替代父类的单标量奖励：两个运动层各用自己的教师信号，
        隐层突触用两者均值。正=强化刚共同发放的通路，负=削弱。"""
        if not self._teacher_q:
            return
        r1, r2 = self._teacher_q.pop(0)
        self.last_teacher = (r1, r2)
        hp = self.hp
        if not self.teacher_on or not hp["reward_mod"] or not hp["el_trace"]:
            return
        if abs(r1) < 1e-4 and abs(r2) < 1e-4:
            return
        rmap = np.zeros(len(self.syn_post), dtype=np.float32)
        rmap[self._m1_mask] = r1
        rmap[self._m2_mask] = r2
        rmap[self._hidden_mask] = 0.5 * (r1 + r2)
        dw = (hp["learning_rate"] * self.teacher_lr
              * rmap * self.eligibility_trace).astype(np.float32)
        pre, post = self.syn_pre, self.syn_post
        wv = np.clip(self.Wt[post, pre] + dw, 0.001, 10.0)
        self.Wt[post, pre] = wv
        self.W[pre, post] = wv


GOAL_D0 = 0.35   # 末端减速距离（仅本实验台教师信号使用，与旧 GUI 默认值一致）


def _goal_omega(t1, t2, ex, ey, tx, ty, max_omega, d0):
    """DLS 逆运动学教师信号（与已删除的共享核心公式同构）。
    只存在于本实验台，作为模仿式蒸馏的教师；推理期 G=0 时网络完全自主。"""
    s1, c1 = math.sin(t1), math.cos(t1)
    s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
    J11 = -bv.L1 * s1 - bv.L2 * s12
    J21 = bv.L1 * c1 + bv.L2 * c12
    J12 = -bv.L2 * s12
    J22 = bv.L2 * c12
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


def net_output(be):
    """本 tick 网络原始解码（不含引导、不含探索噪声）。"""
    n = be.config["neuron_count"]
    return (be._decode_omega(be.m1_start, be.m2_start),
            be._decode_omega(be.m2_start, n))


# ============================================================
# 学习接口：可蒸馏的老师（任意给出期望关节角速度的算法都能接进来）
# ============================================================
class Teacher:
    """蒸馏老师基类。

    任何外部 AI / 算法只要能根据当前状态给出「期望关节角速度」
    (omega1, omega2)，就能被脉冲网络通过可塑群体读出学走。
    子类只需实现 command()，返回 [-max_omega, max_omega] 范围内的指令。
    """

    name = "老师"
    desc = "蒸馏老师基类"
    normalize = True          # 训练目标是否归一到满速（避免近目标处读出塌缩为 0）

    def reset(self, target_x, target_y):
        self.tx, self.ty = target_x, target_y

    def command(self, st):
        """st: dict(theta1, theta2, end_x, end_y, target_x, target_y, L1, L2, max_omega)
        返回期望 (omega1, omega2)。"""
        raise NotImplementedError

    def target(self, w1, w2):
        """蒸馏训练目标：可把老师指令归一到满速，让读出只学方向+相对速度比。"""
        if not self.normalize:
            return w1, w2
        gm = max(abs(w1), abs(w2), 1e-12)
        sc = bv.MAX_OMEGA / gm
        return w1 * sc, w2 * sc


class DLSTeacher(Teacher):
    """阻尼最小二乘逆运动学（现成解析解，已知可达）—— 默认的「现成老师」。"""

    name = "DLS 逆运动学（现成解析解）"
    desc = "已证明可达目标；直接蒸馏，不必重新发明轮子。"

    def __init__(self, d0=GOAL_D0):
        self.d0 = d0

    def command(self, st):
        return _goal_omega(st["theta1"], st["theta2"], st["end_x"], st["end_y"],
                           st["target_x"], st["target_y"], st["max_omega"], self.d0)


class TransposeTeacher(Teacher):
    """转置雅可比梯度法（另一种 IK 算法）—— 演示「换一个算法当老师」。"""

    name = "转置雅可比梯度法（另一种 IK 算法）"
    desc = "用 Jᵀ 做梯度下降式逼近，区别于 DLS；验证接口可蒸馏不同算法。"

    def __init__(self, gain=0.6, d0=GOAL_D0):
        self.gain, self.d0 = gain, d0

    def command(self, st):
        t1, t2 = st["theta1"], st["theta2"]
        L1, L2 = st["L1"], st["L2"]
        s1, c1 = math.sin(t1), math.cos(t1)
        s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
        J11, J21 = -L1 * s1 - L2 * s12, L1 * c1 + L2 * c12
        J12, J22 = -L2 * s12, L2 * c12
        errx, erry = st["target_x"] - st["end_x"], st["target_y"] - st["end_y"]
        d = math.hypot(errx, erry) + 1e-12
        sp = st["max_omega"] * min(1.0, d / self.d0)
        ux, uy = errx / d * sp, erry / d * sp
        return J11 * ux + J21 * uy, (J12 * ux + J22 * uy) * self.gain


class NullTeacher(Teacher):
    """不教（基线）—— 证明纯脉冲网络在没有任何老师时确实做不到。"""

    name = "无老师（纯网络基线）"
    desc = "什么都不教；g=0 探针应几乎到不了，用以校准诚实预期。"
    normalize = False

    def command(self, st):
        return 0.0, 0.0


class FuncTeacher(Teacher):
    """把你自己的 AI / 算法接进来：任意 callable(state)->(w1, w2)。

    用法示例（在代码里构造后传给 TransferEngine(teacher=...)）：
        def my_policy(st):
            # st 含 theta1,theta2,end_x,end_y,target_x,target_y,L1,L2,max_omega
            errx = st["target_x"] - st["end_x"]
            return errx * 0.5, 0.0
        teacher = FuncTeacher(my_policy, name="我的RL策略")
    """

    def __init__(self, fn, name="自定义老师", normalize=True):
        self.fn, self.name, self.normalize = fn, name, normalize

    def command(self, st):
        return self.fn(st)


# 老师注册表：CLI / GUI 下拉框用的名字 -> 构造器
TEACHERS = {
    "dls": DLSTeacher,
    "transpose": TransposeTeacher,
    "null": NullTeacher,
}


# ============================================================
# 引擎：训练 + 退火 + g=0 探针
# ============================================================
class TransferEngine:
    def __init__(self, seed=12345, g0=0.7, teacher_lr=0.03,
                 readout_lr=0.01, anneal_step=0.20, probe_every=10,
                 teacher_on=True, random_target=True,
                 dagger_every=3, time_decay=0.98, wta_k=5,
                 teacher=None):
        self.seed = seed
        self.wta_k = int(wta_k)
        self.teacher = teacher or DLSTeacher()      # 学习接口：可换成任意 Teacher
        self.teacher.reset(*PROBE_TARGETS[0])
        self.g0 = float(g0)
        self.g = float(g0)
        self.readout_lr = float(readout_lr)
        self.dagger_every = int(dagger_every)
        self.time_decay = float(time_decay)
        self.round_g = float(g0)
        self.dagger_round = False
        self.g_min = 0.02
        self.anneal_step = float(anneal_step)
        self.probe_every = int(probe_every)
        self.random_target = bool(random_target)
        self.force_autonomous = False
        self.autonomous = False
        self._streak = 0

        self.be = TransferBackend(seed)
        self.be.init(*SIZE, seed=seed)
        self.be.set_param("tau_trace", 200.0)     # 局部信用分配，比 2000 更适合逐 tick 教师
        self.be.set_param("wta_k", self.wta_k)
        self.be.teacher_lr = float(teacher_lr)
        self.be.teacher_on = bool(teacher_on)
        self.be.readout_lr = float(readout_lr)

        # 隔离的探针后端：只建一次，每次仅覆盖权重
        self.probe = TransferBackend(seed)
        self.probe.init(*SIZE, seed=seed)
        self.probe.set_param("tau_trace", 200.0)
        self.probe.set_param("wta_k", self.wta_k)
        self.probe.set_param("reward_mod", 0)     # 评估期间绝不学习
        self.probe.teacher_on = False

        self.rng = random.Random(seed + 7)
        self.target = PROBE_TARGETS[0]
        self.be.episode_reset(*self.target)

        self.ticks = 0
        self.episode = 0
        self.arrivals = 0
        self.trial_tick = 0
        self.prev = (2.0, 0.0)
        self.hit = False
        self.cos_ema = 0.0
        self.last_probe = None
        self.probe_count = 0
        self.g_hist = [(0, self.g)]
        self.best_score = -1e9
        self.best_w = None
        self.bad_streak = 0
        self.rate_hist = []
        self.cos_hist = []
        self.dist_hist = []
        self.trail = []
        self.pending = []
        self.probing = False

    # ---------- 目标 ----------
    def _next_target(self):
        if not self.random_target:
            return PROBE_TARGETS[0]
        r = 0.4 + self.rng.random() * 1.1
        a = self.rng.random() * math.pi * 2
        return (round(r * math.cos(a), 2), round(r * math.sin(a), 2))

    def effective_g(self):
        return self.round_g

    def _sync_g(self):
        # round_g 直接驱动 step_once 里的 motor_override，无需再下发后端
        self.round_g = 0.0 if self.force_autonomous else self.g

    def _state(self, be, pre):
        """构造传给老师的当前状态字典（学习接口的统一输入格式）。"""
        return {"theta1": pre[0], "theta2": pre[1],
                "end_x": pre[4], "end_y": pre[5],
                "target_x": be.target_x, "target_y": be.target_y,
                "L1": bv.L1, "L2": bv.L2, "max_omega": bv.MAX_OMEGA}

    def _plastic_readout(self, g1, g2):
        """教师目标 -> 可塑群体读出（NLMS）。
        教师只在训练期出现；推理期 g=0 时输出完全由网络脉冲活动决定。
        """
        if self.readout_lr <= 0.0:
            return
        t1, t2 = self.teacher.target(g1, g2)     # 默认归一到满速，避免近目标塌缩
        if max(abs(t1), abs(t2)) < 1e-6:
            return
        self.be.update_readout(t1, t2)

    # ---------- 单步 ----------
    def step_once(self):
        be = self.be
        pre = [float(x) for x in be.arm_dev[:6]]
        # 本 tick 命令 = (1-G)·网络解码 + G·教师，经后端 motor_override 注入；
        # 老师来自学习接口（任意 Teacher），G=0 时退化为纯网络自主解码。
        w1, w2 = net_output(be)
        g1, g2 = self.teacher.command(self._state(be, pre))
        G = self.effective_g()
        o1 = (1.0 - G) * w1 + G * g1
        o2 = (1.0 - G) * w2 + G * g2
        o1 = max(-bv.MAX_OMEGA, min(bv.MAX_OMEGA, o1))
        o2 = max(-bv.MAX_OMEGA, min(bv.MAX_OMEGA, o2))
        be.motor_override = (o1, o2)
        be.tick()
        self.ticks += 1
        self.trial_tick += 1

        ex, ey, tx, ty, _ = be.get_arm()
        self.pending.append((ex, ey))

        gm = max(abs(g1), abs(g2), 1e-6)
        u1, u2 = g1 / gm, g2 / gm
        # 有符号教师信号：网络输出朝教师方向为正、反向为负
        r1 = max(-1.0, min(1.0, w1 * u1 / bv.MAX_OMEGA))
        r2 = max(-1.0, min(1.0, w2 * u2 / bv.MAX_OMEGA))
        be.push_teacher(r1, r2)
        self._plastic_readout(g1, g2)

        # 方向一致性（0=无定向信息，1=完全学会教师）
        nn = math.hypot(w1, w2)
        ng = math.hypot(g1, g2)
        c = ((w1 * g1 + w2 * g2) / (nn * ng)) if nn > 1e-6 and ng > 1e-6 else 0.0
        self.cos_ema += 0.01 * (c - self.cos_ema)

        d = math.hypot(ex - tx, ey - ty)
        just = (not self.hit) and d <= ARRIVE_R
        if just:
            self.hit = True
            self.arrivals += 1
            be.tick()                        # 消费最后一条教师信号
        self.prev = (ex, ey)

        if just or self.trial_tick >= MAX_TRIAL_TICKS:
            self._end_episode()
        return d

    def _end_episode(self):
        self.episode += 1
        self.pending.append(None)
        # 缓慢时间表，避免「探针不达标 -> G 不降 -> 训练永远是引导轨迹」
        # 的分布不匹配死锁；但只降到 0.30，突破必须靠 g=0 探针证明。
        if not self.force_autonomous:
            self.g = max(0.30, self.g * self.time_decay)
            self.g_hist.append((self.ticks, self.g))
        # DAgger：定期让网络在自己造成的状态下也接受教师指导
        self.dagger_round = (self.dagger_every > 0
                             and self.episode % self.dagger_every == 0)
        self.round_g = (0.0 if (self.force_autonomous or self.dagger_round)
                        else self.g)
        self.target = self._next_target()
        self.be.episode_reset(*self.target)
        self.prev = (2.0, 0.0)
        self.trial_tick = 0
        self.hit = False
        if self.probe_every > 0 and self.episode % self.probe_every == 0:
            self.run_probe()

    # ---------- g=0 探针（唯一验收口径）----------
    def run_probe(self):
        self.probing = True
        p = self.probe
        p.Wt[:] = self.be.Wt                 # 突触权重快照
        p.W[:] = self.be.W
        p.readout_w[:] = self.be.readout_w    # 可塑读出权重必须一起快照
        hit_list, min_list, cos_list, amp_net, amp_goal = [], [], [], [], []
        for tx, ty in PROBE_TARGETS:
            p.tick_count = 0
            p.episode_reset(tx, ty)
            best, arrived = 1e9, False
            for _ in range(PROBE_TICKS):
                pre = [float(x) for x in p.arm_dev[:6]]
                st = {"theta1": pre[0], "theta2": pre[1],
                      "end_x": pre[4], "end_y": pre[5],
                      "target_x": tx, "target_y": ty,
                      "L1": bv.L1, "L2": bv.L2, "max_omega": bv.MAX_OMEGA}
                g1, g2 = self.teacher.command(st)      # 与训练同口径的方向基准
                p.tick()
                ex, ey, _, _, _ = p.get_arm()
                d = math.hypot(ex - tx, ey - ty)
                best = min(best, d)
                w1, w2 = net_output(p)
                nn, ng = math.hypot(w1, w2), math.hypot(g1, g2)
                amp_net.append(nn)
                amp_goal.append(ng)
                if nn > 1e-6 and ng > 1e-6:
                    cos_list.append((w1 * g1 + w2 * g2) / (nn * ng))
                if d <= ARRIVE_R:
                    arrived = True
                    break
            hit_list.append(arrived)
            min_list.append(best)
        rate = sum(hit_list) / len(hit_list)
        mind = min(min_list)
        cosv = sum(cos_list) / len(cos_list) if cos_list else 0.0
        ampr = ((sum(amp_net) / len(amp_net)) /
                max(1e-6, sum(amp_goal) / len(amp_goal)))

        self.probe_count += 1
        self.last_probe = (rate, mind, cosv, ampr)
        self.rate_hist.append((self.ticks, rate))
        self.dist_hist.append((self.ticks, mind))
        self.cos_hist.append((self.ticks, cosv))

        # 历史最佳检查点：连续两次评估变差就回滚读出权重，避免灾难性漂移
        score = 2.0 * rate - mind
        if score > self.best_score + 1e-6:
            self.best_score = score
            self.best_w = self.be.readout_w.copy()
            self.bad_streak = 0
        else:
            self.bad_streak += 1
            if self.bad_streak >= 2 and self.best_w is not None:
                self.be.readout_w[:] = self.best_w
                self.bad_streak = 0

        # 表现驱动退火：有到达、或已能逼近目标，才允许突破时间表下限
        if not self.force_autonomous:
            good = (rate >= 1.0 / 3.0) or (mind < 0.35)
            if good:
                self.g = max(self.g_min, self.g * (1.0 - self.anneal_step))
                self._streak += 1
            elif rate <= 0.0 and mind > 0.8:
                self.g = min(self.g0, self.g / (1.0 - 0.5 * self.anneal_step))
                self._streak = 0
            else:
                self._streak = 0
            self._sync_g()
        if (self.g <= self.g_min + 1e-6 and self._streak >= 3
                and rate >= 2.0 / 3.0):
            self.autonomous = True
        self.g_hist.append((self.ticks, self.g))
        self.probing = False
        return self.last_probe

    # ---------- 对外状态 ----------
    def snapshot(self):
        be = self.be
        a = [float(x) for x in be.arm_dev[:6]]
        g = 0.0 if self.force_autonomous else self.g
        return {
            "g": g, "g_sched": self.g, "round_g": self.round_g,
            "dagger": self.dagger_round, "ticks": self.ticks,
            "episode": self.episode, "arrivals": self.arrivals,
            "trial_tick": self.trial_tick, "cos": self.cos_ema,
            "probe": self.last_probe, "probe_n": self.probe_count,
            "autonomous": self.autonomous, "probing": self.probing,
            "teacher_name": self.teacher.name, "teacher_desc": self.teacher.desc,
            "teacher": be.last_teacher, "target": (be.target_x, be.target_y),
            "arm": a, "g_hist": list(self.g_hist),
            "rate_hist": list(self.rate_hist),
            "cos_hist": list(self.cos_hist),
            "dist_hist": list(self.dist_hist),
            "force_auto": self.force_autonomous,
        }

    def take_trail(self):
        t = self.pending
        self.pending = []
        return t


# ============================================================
# headless 自检
# ============================================================
def run_check(args):
    e = TransferEngine(seed=args.seed, g0=0.7, teacher_lr=args.teacher_lr,
                       readout_lr=args.readout_lr,
                       dagger_every=args.dagger_every, wta_k=args.wta_k,
                       teacher_on=args.rstdp,
                       probe_every=args.probe_every, random_target=True,
                       teacher=TEACHERS[args.distill]())
    print("蒸馏老师：%s" % e.teacher.name)
    print("R-STDP 教师：%s  lr=%.3f   读出可塑(监督)：lr=%.3f   每 %d 轮一次探针"
          % ("开" if e.be.teacher_on else "关", e.be.teacher_lr,
             e.readout_lr, e.probe_every))
    print("探针口径：g=0，%d 个固定目标，每个最多 %d tick"
          % (len(PROBE_TARGETS), PROBE_TICKS))
    print("-" * 72)
    t0 = time.time()
    last_n = 0
    while e.ticks < args.ticks:
        e.step_once()
        if e.probe_count > last_n:
            last_n = e.probe_count
            rate, mind, cosv, ampr = e.last_probe
            print("探针 %2d  tick %6d  G=%.3f  自主到达 %d/%d  最小距离 %.3f  "
                  "网络/教师 cos %+.4f  速度比 %.2f"
                  % (last_n, e.ticks, e.g, round(rate * len(PROBE_TARGETS)),
                     len(PROBE_TARGETS), mind, cosv, ampr))
    print("-" * 72)
    print("总 tick %d  %.1f tick/s  轮次 %d  带引导到达 %d  最终 G=%.3f  已自主=%s"
          % (e.ticks, e.ticks / max(1e-6, time.time() - t0), e.episode,
             e.arrivals, e.g, e.autonomous))
    rw = e.be.readout_w
    print("读出权重: |w1|=%.3f  |w2|=%.3f  非零维度 %d/%d"
          % (float(np.linalg.norm(rw[0])), float(np.linalg.norm(rw[1])),
             int(np.count_nonzero(rw)), rw.size))
    return 0


# ============================================================
# GUI（仅在非 --check 时构建）
# ============================================================
def run_gui(args):
    from PySide6.QtCore import Qt, QTimer, QThread, QMutex, QPointF, QRectF
    from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                                  QPushButton, QCheckBox, QDoubleSpinBox,
                                  QSpinBox, QComboBox, QVBoxLayout, QHBoxLayout,
                                  QGridLayout, QFrame, QSizePolicy)

    QSS = """
    QWidget { background:%s; color:%s; font-family:'Microsoft YaHei','SimHei',sans-serif; }
    QMainWindow { background:%s; }
    QFrame#card { background:%s; border:1px solid %s; border-radius:10px; }
    QLabel#big { font-size:52px; font-weight:700; color:%s; }
    QLabel#stage { font-size:15px; font-weight:600; color:%s; }
    QLabel#mute { color:%s; font-size:12px; }
    QLabel#val { font-size:19px; font-weight:600; }
    QPushButton { background:#1b2436; color:%s; border:1px solid %s;
                  border-radius:6px; padding:6px 12px; font-size:13px; }
    QPushButton:hover { background:#26344c; }
    QPushButton:checked { background:#2d4a3e; border-color:%s; }
    QCheckBox, QDoubleSpinBox, QSpinBox { color:%s; font-size:12px; }
    QDoubleSpinBox, QSpinBox { background:#151d2c; border:1px solid %s;
                               border-radius:4px; padding:3px; }
    """ % (C_BG, C_TEXT, C_BG, C_CARD, C_LINE, C_GUIDE, C_AUTO,
           C_MUTE, C_TEXT, C_LINE, C_AUTO, C_TEXT, C_LINE)

    class Chart(QWidget):
        """轻量折线图，避免任何外部图表依赖。"""
        def __init__(self, title, series, ymin, ymax, unit="", parent=None):
            super().__init__(parent)
            self.title, self.series = title, series
            self.ymin, self.ymax, self.unit = ymin, ymax, unit
            self.data = {}
            self.setMinimumHeight(112)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def set_series(self, d):
            self.data = d
            self.update()

        def paintEvent(self, ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()
            p.fillRect(0, 0, w, h, QColor(C_CARD))
            p.setPen(QPen(QColor(C_LINE), 1))
            p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 9, 9)
            p.setPen(QColor(C_MUTE))
            f = QFont("Microsoft YaHei", 9)
            p.setFont(f)
            p.drawText(10, 16, self.title)
            left, top, right, bot = 34, 26, w - 8, h - 20
            for k in range(5):
                y = top + (bot - top) * k / 4.0
                p.setPen(QPen(QColor(C_LINE), 1, Qt.DashLine))
                p.drawLine(int(left), int(y), int(right), int(y))
                p.setPen(QColor(C_MUTE))
                v = self.ymax - (self.ymax - self.ymin) * k / 4.0
                p.drawText(2, int(y) + 3, ("%.2f" % v) if self.ymax <= 2 else ("%.0f" % v))
            for key, (label, color) in self.series.items():
                pts = self.data.get(key, [])
                if len(pts) < 2:
                    continue
                xs = [q[0] for q in pts]
                x0, x1 = min(xs), max(xs)
                span = max(1.0, x1 - x0)
                path = QPainterPath()
                for i, (x, y) in enumerate(pts):
                    px = left + (right - left) * (x - x0) / span
                    py = bot - (bot - top) * max(0.0, min(1.0, (y - self.ymin) /
                                                          max(1e-9, self.ymax - self.ymin)))
                    if i == 0:
                        path.moveTo(px, py)
                    else:
                        path.lineTo(px, py)
                p.setPen(QPen(QColor(color), 2))
                p.drawPath(path)
            p.setPen(QColor(C_MUTE))
            p.drawText(int(right) - 190, h - 5, self.unit)

    class ArmView(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.snap = None
            self.trail = []
            self.setMinimumSize(330, 330)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def add_points(self, pts):
            for q in pts:
                if q is None:
                    # 进入新一轮（碰到目标重新测试）：清空旧轨迹，
                    # 只显示本次运动，避免「瞬移」式的跨轮残影
                    self.trail = []
                else:
                    self.trail.append(q)
            if len(self.trail) > 1500:
                self.trail = self.trail[-1500:]
            self.update()

        def paintEvent(self, ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()
            p.fillRect(0, 0, w, h, QColor(C_CARD))
            p.setPen(QPen(QColor(C_LINE), 1))
            p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 9, 9)
            s = min(w, h) / 4.3
            cx, cy = w / 2.0, h / 2.0
            p.setPen(QColor(C_MUTE))
            p.setFont(QFont("Microsoft YaHei", 9))
            p.drawText(12, 20, "机械臂轨迹（橙=引导主导，绿=纯自主）")

            def pt(x, y):
                return QPointF(cx + x * s, cy - y * s)

            # 网格与最大可达圈
            for r in (0.5, 1.0, 1.5, 2.0):
                p.setPen(QPen(QColor(C_LINE), 1, Qt.DashLine))
                p.drawEllipse(pt(0, 0), r * s, r * s)
            p.setPen(QPen(QColor("#3a4a66"), 1))
            p.drawLine(pt(-2.2, 0), pt(2.2, 0))
            p.drawLine(pt(0, -2.2), pt(0, 2.2))
            p.setPen(QPen(QColor("#41577a"), 2))
            p.drawEllipse(pt(0, 0), 2.0 * s, 2.0 * s)

            snap = self.snap
            if not snap:
                return
            a = snap["arm"]
            tx, ty = snap["target"]
            g = snap["g"]
            col = QColor(C_AUTO if g < 0.05 else C_GUIDE)

            # 轨迹
            tc = QColor(C_TEACH)
            tc.setAlpha(150)
            pen = QPen(tc, 1.4)
            p.setPen(pen)
            path = QPainterPath()
            started = False
            for q in self.trail:
                if q is None:
                    started = False
                    continue
                qp = pt(q[0], q[1])
                if not started:
                    path.moveTo(qp)
                    started = True
                else:
                    path.lineTo(qp)
            p.drawPath(path)

            # 连杆
            L1, L2 = bv.L1, bv.L2
            e1 = (L1 * math.cos(a[0]), L1 * math.sin(a[0]))
            e2 = (a[4], a[5])
            p.setPen(QPen(col, 5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(pt(0, 0), pt(*e1))
            p.setPen(QPen(col, 4, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(pt(*e1), pt(*e2))
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawEllipse(pt(0, 0), 5, 5)
            p.drawEllipse(pt(*e1), 4, 4)
            p.setBrush(QBrush(QColor(C_TEXT)))
            p.drawEllipse(pt(*e2), 6, 6)
            # 目标
            p.setBrush(Qt.NoBrush)
            pen = QPen(QColor(C_AUTO), 2)
            p.setPen(pen)
            p.drawEllipse(pt(tx, ty), 9, 9)
            p.drawLine(pt(tx - 0.13, ty), pt(tx + 0.13, ty))
            p.drawLine(pt(tx, ty - 0.13), pt(tx, ty + 0.13))
            # 距离线
            p.setPen(QPen(QColor(C_MUTE), 1, Qt.DashLine))
            p.drawLine(pt(*e2), pt(tx, ty))
            d = math.hypot(e2[0] - tx, e2[1] - ty)
            p.setPen(QColor(C_TEXT))
            p.setFont(QFont("Microsoft YaHei", 10))
            p.drawText(12, h - 12, "末端距离 %.3f   %s" %
                       (d, "网络自主" if g < 0.05 else "引导占比 %d%%" % round(g * 100)))

    class Worker(QThread):
        def __init__(self, engine):
            super().__init__()
            self.e = engine
            self.mutex = QMutex()
            self.latest = engine.snapshot()
            self.running = True
            self.paused = False
            self.request_probe = False

        def run(self):
            while self.running:
                if self.paused:
                    self.msleep(20)
                    continue
                self.e.step_once()
                self.mutex.lock()
                self.latest = self.e.snapshot()
                self.mutex.unlock()

        def state(self):
            self.mutex.lock()
            s = self.latest
            self.mutex.unlock()
            return s

        def set_paused(self, v):
            self.paused = bool(v)

    class LabWindow(QMainWindow):
        def __init__(self, engine):
            super().__init__()
            self.setWindowTitle("引导转移实验台 · 网络何时能自己找路")
            self.e = engine
            self.worker = Worker(engine)
            self.auto_demo = QCheckBox("强制纯自主（G=0，随时看真实水平）")
            self.teacher_box = QCheckBox("附加 R-STDP 教师（默认关，用于对照）")
            self.teacher_box.setChecked(engine.be.teacher_on)
            self.btn_play = QPushButton("暂停")
            self.btn_play.setCheckable(True)
            self.btn_probe = QPushButton("立即评估")
            self.btn_reset = QPushButton("重置")
            self.lr = QDoubleSpinBox()
            self.lr.setRange(0.0, 2.0)
            self.lr.setSingleStep(0.05)
            self.lr.setDecimals(3)
            self.lr.setValue(engine.readout_lr)
            self.lr.setFixedWidth(78)
            self.every = QSpinBox()
            self.every.setRange(2, 100)
            self.every.setValue(engine.probe_every)
            self.every.setFixedWidth(62)
            self.teacher_sel = QComboBox()
            self.teacher_sel.addItems(list(TEACHERS.keys()))
            self.teacher_sel.setCurrentText("dls")
            self.teacher_sel.setFixedWidth(120)
            self.teacher_sel.setToolTip("选择要蒸馏的老师（学习接口）")

            bar = QWidget()
            hb = QHBoxLayout(bar)
            hb.setContentsMargins(10, 8, 10, 8)
            hb.setSpacing(8)
            for wgt in (self.btn_play, self.btn_probe, self.btn_reset,
                        self.auto_demo, self.teacher_box):
                hb.addWidget(wgt)
            hb.addStretch(1)
            self.setMenuWidget(bar)

            root = QWidget()
            top = QHBoxLayout(root)
            top.setContentsMargins(12, 12, 12, 12)
            top.setSpacing(12)

            left = QWidget()
            lv = QVBoxLayout(left)
            lv.setSpacing(10)
            gcard = QFrame()
            gcard.setObjectName("card")
            gv = QVBoxLayout(gcard)
            gv.setContentsMargins(16, 10, 16, 12)
            self.lbl_g = QLabel("0.70")
            self.lbl_g.setObjectName("big")
            self.lbl_g.setAlignment(Qt.AlignCenter)
            self.lbl_stage = QLabel("模仿学习中")
            self.lbl_stage.setObjectName("stage")
            self.lbl_stage.setAlignment(Qt.AlignCenter)
            self.lbl_split = QLabel("解析引导 70%  /  网络自主 30%")
            self.lbl_split.setObjectName("mute")
            self.lbl_split.setAlignment(Qt.AlignCenter)
            gv.addWidget(QLabel("当前 G 值（1=全引导，0=全自主）", objectName="mute"))
            for x in (self.lbl_g, self.lbl_stage, self.lbl_split):
                gv.addWidget(x)
            lv.addWidget(gcard)

            self.lbl_teacher = QLabel("蒸馏老师：—")
            self.lbl_teacher.setObjectName("mute")
            self.lbl_teacher.setWordWrap(True)
            lv.addWidget(self.lbl_teacher)

            # 设置卡片：决定加载哪个老师及蒸馏参数（学习接口开关）
            scard = QFrame()
            scard.setObjectName("card")
            sv = QVBoxLayout(scard)
            sv.setContentsMargins(12, 8, 12, 10)
            sv.addWidget(QLabel("设置 · 蒸馏老师与参数", objectName="mute"))
            srow1 = QHBoxLayout()
            srow1.setSpacing(8)
            srow1.addWidget(QLabel("蒸馏老师"))
            self.teacher_sel.setFixedWidth(180)
            srow1.addWidget(self.teacher_sel, 1)
            sv.addLayout(srow1)
            srow2 = QHBoxLayout()
            srow2.setSpacing(8)
            srow2.addWidget(QLabel("读出可塑强度"))
            srow2.addWidget(self.lr)
            srow2.addWidget(QLabel("评估间隔(轮)"))
            srow2.addWidget(self.every)
            srow2.addStretch(1)
            sv.addLayout(srow2)
            lv.addWidget(scard)

            grid = QGridLayout()
            grid.setSpacing(8)
            self.cards = {}
            items = (("ticks", "累计 tick"), ("episode", "训练轮次"),
                     ("arrivals", "带引导到达"), ("probe_n", "评估次数"),
                     ("rate", "自主到达率"), ("mind", "自主最小距离"),
                     ("cos", "探针方向余弦"), ("amp", "自主/教师速度比"))
            for i, (k, name) in enumerate(items):
                f = QFrame()
                f.setObjectName("card")
                v = QVBoxLayout(f)
                v.setContentsMargins(10, 6, 10, 8)
                lab = QLabel(name, objectName="mute")
                val = QLabel("—", objectName="val")
                v.addWidget(lab)
                v.addWidget(val)
                grid.addWidget(f, i // 4, i % 4)
                self.cards[k] = val
            lv.addLayout(grid)

            self.ch_g = Chart("G 值退火轨迹", {"g": ("G", C_GUIDE)}, 0, 1, "越接近 0 越自主")
            self.ch_rate = Chart("g=0 探针：自主到达率 / 方向一致性",
                                 {"rate": ("到达率", C_AUTO),
                                  "cos": ("方向余弦", C_TEACH)}, -0.25, 1.0,
                                 "到达率=1 且 G=0 才算学会")
            lv.addWidget(self.ch_g)
            lv.addWidget(self.ch_rate)
            top.addWidget(left, 3)

            self.arm = ArmView()
            top.addWidget(self.arm, 2)

            note = QLabel("独立单文件 · 默认不写任何文件 · 退出后删除本文件即可 · "
                          "验收只看 g=0 评估（G 到 0 且连续 3 次到达 2/3）· "
                          "这是监督蒸馏，不是网络自主发现", objectName="mute")
            self.setCentralWidget(root)
            self.statusBar().addWidget(note)

            self.btn_play.toggled.connect(self.worker.set_paused)
            self.btn_probe.clicked.connect(self._probe_now)
            self.btn_reset.clicked.connect(self._reset)
            self.auto_demo.toggled.connect(self._auto)
            self.teacher_box.toggled.connect(self._teacher)
            self.lr.valueChanged.connect(self._lr)
            self.every.valueChanged.connect(self._every)
            self.teacher_sel.currentTextChanged.connect(self._teacher_changed)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(33)
            self.worker.start()
            self.resize(1120, 760)
            self.setStyleSheet(QSS)
            self.refresh()

        # --- 控件 ---
        def _make_engine(self, seed):
            return TransferEngine(seed=seed, g0=self.e.g0,
                                  teacher_lr=self.e.be.teacher_lr,
                                  readout_lr=self.e.readout_lr,
                                  dagger_every=self.e.dagger_every,
                                  wta_k=self.e.wta_k,
                                  teacher_on=self.e.be.teacher_on,
                                  probe_every=int(self.every.value()),
                                  teacher=TEACHERS[self.teacher_sel.currentText()]())
        def _probe_now(self):
            self.e.run_probe()
        def _reset(self):
            self.worker.set_paused(True)
            ne = self._make_engine(self.e.seed + 1)
            self.e = ne
            self.worker.e = ne
            self.arm.trail = []
            self.worker.set_paused(False)
        def _auto(self, v):
            self.e.force_autonomous = bool(v)
            self.e._sync_g()
        def _teacher(self, v):
            self.e.be.teacher_on = bool(v)
        def _lr(self, v):
            self.e.readout_lr = float(v)
        def _every(self, v):
            self.e.probe_every = int(v)
        def _teacher_changed(self, _):
            self._reset()

        def refresh(self):
            s = self.worker.state()
            g = s["g"]
            self.lbl_g.setText("%.3f" % g)
            self.lbl_g.setStyleSheet("color:%s" % (C_AUTO if g < 0.05 else C_GUIDE))
            self.lbl_split.setText("解析引导 %d%%  /  网络自主 %d%%"
                                   % (round(g * 100), round((1 - g) * 100)))
            if s["autonomous"]:
                stage, col = "已自主：网络可以自己找路", C_AUTO
            elif s["force_auto"]:
                stage, col = "纯自主演示中（G 强制为 0）", C_AUTO
            elif s["probing"]:
                stage, col = "正在做 g=0 评估", C_TEACH
            elif s["g_sched"] <= 0.05:
                stage, col = "G 已接近 0，等待连续评估通过", C_TEACH
            else:
                stage, col = "可塑群体读出训练中（监督蒸馏）", C_GUIDE
            if s.get("dagger") and not s["force_auto"]:
                stage += "（本轮自主采样）"
            self.lbl_stage.setText(stage)
            self.lbl_stage.setStyleSheet("color:%s;font-size:15px;font-weight:600" % col)

            self.lbl_teacher.setText("蒸馏老师：%s\n%s" % (s["teacher_name"], s["teacher_desc"]))

            pr = s["probe"]
            self.cards["ticks"].setText("%d" % s["ticks"])
            self.cards["episode"].setText("%d" % s["episode"])
            self.cards["arrivals"].setText("%d" % s["arrivals"])
            self.cards["probe_n"].setText("%d" % s["probe_n"])
            self.cards["rate"].setText(("%d/%d" % (round(pr[0] * 3), 3)) if pr else "未评估")
            self.cards["mind"].setText("%.3f" % pr[1] if pr else "—")
            self.cards["cos"].setText("%+.3f" % pr[2] if pr else "—")
            self.cards["amp"].setText("%.2f" % pr[3] if pr else "—")

            self.ch_g.set_series({"g": s["g_hist"]})
            self.ch_rate.set_series({"rate": s["rate_hist"], "cos": s["cos_hist"]})
            self.arm.snap = s
            self.arm.add_points(self.e.take_trail())

        def closeEvent(self, ev):
            self.timer.stop()
            self.worker.running = False
            self.worker.wait(2000)
            super().closeEvent(ev)

    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    eng = TransferEngine(seed=args.seed, g0=0.7, teacher_lr=args.teacher_lr,
                         readout_lr=args.readout_lr,
                         dagger_every=args.dagger_every, wta_k=args.wta_k,
                         teacher_on=args.rstdp,
                         probe_every=args.probe_every,
                         teacher=TEACHERS[args.distill]())
    w = LabWindow(eng)
    w.show()
    return app.exec()


def main():
    ap = argparse.ArgumentParser(description="引导转移实验台（独立程序）")
    ap.add_argument("--check", action="store_true", help="headless 自检，不开窗口")
    ap.add_argument("--ticks", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--teacher-lr", type=float, default=0.03)
    ap.add_argument("--readout-lr", type=float, default=0.3)
    ap.add_argument("--rstdp", action="store_true",
                    help="附加纯 R-STDP 教师通道（默认只用监督式读出可塑）")
    ap.add_argument("--distill", choices=list(TEACHERS.keys()), default="dls",
                    help="选择要蒸馏的老师：dls=现成解析解 / transpose=另一种IK / null=无老师基线")
    ap.add_argument("--dagger-every", type=int, default=3)
    ap.add_argument("--wta-k", type=int, default=5)
    ap.add_argument("--probe-every", type=int, default=8)
    a = ap.parse_args()
    return run_check(a) if a.check else run_gui(a)


if __name__ == "__main__":
    sys.exit(main())
