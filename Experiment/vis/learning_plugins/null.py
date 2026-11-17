"""空老师插件：故意不声明 params，验证「没有的选项不出现」。"""
from learning_api import NullTeacher

PLUGIN = {
    "id": "null",
    "name": "无老师基线（纯网络对照）",
    "description": "始终输出 (0,0)，用于校准纯网络基线。",
    "params": [],
    "create": lambda p: NullTeacher(),
}
