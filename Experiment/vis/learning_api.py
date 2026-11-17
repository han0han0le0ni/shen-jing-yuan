"""学习接口公共 API：蒸馏老师基类、内置实现与工具函数。

具体老师以插件形式放在 ``learning_plugins/*.py``，由 learning_registry 自动发现。
插件 import 本模块即可，不要反向依赖 neuron_vis，避免循环导入。
"""
import math

from backend_vec import L1, L2, MAX_OMEGA

GOAL_D0 = 0.35   # 末端减速距离（仅老师信号使用）


def _goal_omega(t1, t2, ex, ey, tx, ty, max_omega, d0):
    """DLS 阻尼最小二乘逆运动学：已证明可达，现成的解析解。"""
    s1, c1 = math.sin(t1), math.cos(t1)
    s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
    J11 = -L1 * s1 - L2 * s12
    J21 = L1 * c1 + L2 * c12
    J12 = -L2 * s12
    J22 = L2 * c12
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


class Teacher:
    """蒸馏老师基类。子类只需实现 command(state)->(omega1, omega2)。

    state 统一字段：theta1, theta2, end_x, end_y, target_x, target_y,
    L1, L2, max_omega。
    """

    name = "老师"
    desc = "蒸馏老师基类"
    normalize = True          # 训练目标是否归一到满速

    def reset(self, target_x, target_y):
        self.tx, self.ty = target_x, target_y

    def command(self, st):
        raise NotImplementedError

    def target(self, w1, w2):
        if not self.normalize:
            return w1, w2
        gm = max(abs(w1), abs(w2), 1e-12)
        sc = MAX_OMEGA / gm
        return w1 * sc, w2 * sc


class DLSTeacher(Teacher):
    """阻尼最小二乘逆运动学（现成解析解，已知可达）。"""

    name = "DLS 逆运动学（现成解析解）"
    desc = "解析法给出期望关节角速度，用于蒸馏；不是网络自主能力。"

    def __init__(self, d0=GOAL_D0):
        self.d0 = d0

    def command(self, st):
        return _goal_omega(st["theta1"], st["theta2"], st["end_x"], st["end_y"],
                           st["target_x"], st["target_y"], st["max_omega"], self.d0)


class TransposeTeacher(Teacher):
    """转置雅可比梯度法（另一种 IK 算法）。"""

    name = "转置雅可比（另一种 IK）"
    desc = "用 J^T 乘末端误差方向，作为另一种引导信号。"

    def __init__(self, gain=0.6, d0=GOAL_D0):
        self.gain, self.d0 = gain, d0

    def command(self, st):
        t1, t2 = st["theta1"], st["theta2"]
        L1_, L2_ = st["L1"], st["L2"]
        s1, c1 = math.sin(t1), math.cos(t1)
        s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
        J11, J21 = -L1_ * s1 - L2_ * s12, L1_ * c1 + L2_ * c12
        J12, J22 = -L2_ * s12, L2_ * c12
        errx, erry = st["target_x"] - st["end_x"], st["target_y"] - st["end_y"]
        d = math.hypot(errx, erry) + 1e-12
        sp = st["max_omega"] * min(1.0, d / self.d0)
        ux, uy = errx / d * sp, erry / d * sp
        return J11 * ux + J21 * uy, (J12 * ux + J22 * uy) * self.gain


class NullTeacher(Teacher):
    """不教（基线）—— 用以校准诚实预期（纯网络本就到不了）。"""

    name = "无老师（纯网络基线）"
    desc = "总是返回 (0,0)，用于对照。"
    normalize = False

    def command(self, st):
        return 0.0, 0.0


class FuncTeacher(Teacher):
    """把任意 callable(state)->(w1, w2) 包装成老师。"""

    def __init__(self, fn, name="自定义老师", normalize=True):
        self.fn, self.name, self.normalize = fn, name, normalize

    def command(self, st):
        return self.fn(st)
