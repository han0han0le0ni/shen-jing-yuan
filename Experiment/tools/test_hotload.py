"""学习接口插件回归测试：自动发现 / 契约校验 / 空目录隐藏 / GUI 热加载。

运行： py -3 tools/test_hotload.py
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS = os.path.join(ROOT, "vis")
sys.path.insert(0, VIS)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6.QtWidgets import QApplication          # noqa: E402
from PySide6.QtCore import QTimer                   # noqa: E402

import learning_registry as reg                     # noqa: E402
import neuron_vis as nv                             # noqa: E402

FAILED = []


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


GOOD = """\
PLUGIN = {
    "id": "tmp",
    "name": "临时插件",
    "params": [{"key": "g", "label": "增益", "type": "float",
                "min": 0.0, "max": 1.0, "step": 0.1, "default": 0.3}],
    "create": lambda p: object(),
}
"""
RENAMED = GOOD.replace('"临时插件"', '"改名后的插件"')


def test_discover():
    print("[1] 自动发现（临时目录，不触碰真实插件）")
    with tempfile.TemporaryDirectory() as d:
        ps, errs = reg.discover(d)
        check(not ps and not errs, "空目录 → 0 插件")

        p = os.path.join(d, "tmp.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(GOOD)
        ps, errs = reg.discover(d)
        check(len(ps) == 1 and ps[0]["id"] == "tmp", "新增文件 → 自动出现")
        check(ps[0]["params"][0]["default"] == 0.3, "参数契约解析正确")

        time.sleep(0.02)
        with open(p, "w", encoding="utf-8") as f:
            f.write(RENAMED)
        ps, errs = reg.discover(d)
        check(len(ps) == 1 and ps[0]["name"] == "改名后的插件", "改写文件 → 立即重载")

        with open(os.path.join(d, "broken.py"), "w", encoding="utf-8") as f:
            f.write("raise RuntimeError('boom')")
        ps, errs = reg.discover(d)
        check(len(ps) == 1 and len(errs) == 1, "坏插件被跳过且错误可报告")

        os.remove(p)
        ps, errs = reg.discover(d)
        check(not ps, "删除文件 → 自动消失")


def test_window():
    print("[2] 主窗口：动态选项 / 无参数不出现 / 热加载")
    app = QApplication([])
    nv.ensure_fonts()
    nv.FONT_FAMILY = nv.pick_font_family()
    app.setStyleSheet(nv.build_qss(nv.THEMES["dark"], nv.FONT_FAMILY))
    w = nv.MainWindow(nv.SimEngine(nv.find_dll()))
    w.resize(1400, 900)
    w.show()

    expected = ["none"] + [p["id"] for p in w._plugins]
    check(w.combo_teacher.count() == len(expected), "选项栏 = 无老师 + 检测到的插件")
    # isVisible() 受父级设置面板默认收起影响；用 isHidden() 判断显式显隐
    check(not w.teacher_card.isHidden(), "有插件 → 卡片出现")

    by_id = {pid: i for i, pid in enumerate(w._plugin_ids)}
    for pid, n_params in (("dls", 1), ("transpose", 2), ("null", 0), ("none", 0)):
        if pid not in by_id:
            check(False, "%s 存在于选项栏" % pid)
            continue
        w.combo_teacher.blockSignals(True)
        w.combo_teacher.setCurrentIndex(by_id[pid])
        w.combo_teacher.blockSignals(False)
        w._rebuild_plugin_params()
        check(w.plugin_param_lay.count() == n_params,
              "%s → %d 个参数控件" % (pid, n_params))
        check(w.plugin_param_box.isHidden() == (n_params == 0),
              "%s → 参数区%s" % (pid, "出现" if n_params else "不出现"))

    # 目录为空时整卡隐藏（用替换 discover 模拟，不动真实文件）
    orig = nv.discover
    nv.discover = lambda: ([], [])
    w._refresh_plugin_ui(initial=False)
    check(w.teacher_card.isHidden(), "0 插件 → 整张卡片不出现")
    nv.discover = orig
    w._refresh_plugin_ui(initial=False)
    check(not w.teacher_card.isHidden(), "恢复插件 → 卡片重新出现")

    # 热加载：运行时新增/删除文件
    hot = os.path.join(reg.plugin_dir(), "zz_hot_test.py")
    with open(hot, "w", encoding="utf-8") as f:
        f.write("""\
from learning_api import NullTeacher
PLUGIN = {"id": "zz_hot_test", "name": "热加载测试插件",
          "params": [], "create": lambda p: NullTeacher()}
""")
    q = {"base": w.combo_teacher.count(), "after_add": None, "after_del": None}

    def on_add():
        q["after_add"] = w.combo_teacher.count()

    def on_del():
        q["after_del"] = w.combo_teacher.count()
        check(q["after_add"] == q["base"] + 1, "运行中新增插件 → 选项栏自动增加")
        check(q["after_del"] == q["base"], "运行中删除插件 → 选项栏自动移除")
        w.close()
        app.quit()

    def do_del():
        try:
            os.remove(hot)
        except OSError:
            pass

    QTimer.singleShot(2200, on_add)
    QTimer.singleShot(2400, do_del)
    QTimer.singleShot(3900, on_del)
    QTimer.singleShot(4400, app.quit)
    app.exec()


def main():
    test_discover()
    test_window()
    print()
    if FAILED:
        print("FAILED %d: %s" % (len(FAILED), "; ".join(FAILED)))
        return 1
    print("HOTLOAD OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
