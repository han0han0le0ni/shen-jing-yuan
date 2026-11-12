"""DLS 引导老师插件。

PLUGIN 契约：id / name / description / params / create(params) -> teacher。
本文件被删除或改名时，选项栏会自动移除该条目（热加载）。
"""
from learning_api import DLSTeacher, GOAL_D0

PLUGIN = {
    "id": "dls",
    "name": "DLS 逆运动学（现成解析解）",
    "description": "阻尼最小二乘解析解，提供期望关节角速度；用于蒸馏，不计入网络能力。",
    "params": [
        {
            "key": "d0",
            "label": "减速距离 d0",
            "type": "float",
            "min": 0.05,
            "max": 1.00,
            "step": 0.05,
            "default": GOAL_D0,
        },
    ],
    "create": lambda p: DLSTeacher(d0=float(p.get("d0", GOAL_D0))),
}
