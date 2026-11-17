"""转置雅可比引导老师插件（两个可调参数，演示选项栏动态生成）。"""
from learning_api import TransposeTeacher, GOAL_D0

PLUGIN = {
    "id": "transpose",
    "name": "转置雅可比（另一种 IK）",
    "description": "J^T 乘末端误差方向；与 DLS 形成对照。",
    "params": [
        {"key": "gain", "label": "引导增益", "type": "float",
         "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.6},
        {"key": "d0", "label": "减速距离 d0", "type": "float",
         "min": 0.05, "max": 1.00, "step": 0.05, "default": GOAL_D0},
    ],
    "create": lambda p: TransposeTeacher(
        gain=float(p.get("gain", 0.6)),
        d0=float(p.get("d0", GOAL_D0))),
}
