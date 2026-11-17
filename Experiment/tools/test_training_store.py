"""训练快照 / G 退火回归测试（不污染工作区数据目录）。

运行： py -3 tools/test_training_store.py
覆盖：退火逐轮下降、权重训练后变化、reset 归零、save/load 精确恢复、
      拓扑不匹配拒绝、C DLL 拒绝、主窗口 G 回写、存档窗口自检。
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS = os.path.join(ROOT, "vis")
sys.path.insert(0, VIS)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np                                  # noqa: E402
import neuron_vis as nv                              # noqa: E402
import training_store as store                      # noqa: E402
from learning_api import DLSTeacher                  # noqa: E402

FAILED = []


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def make_engine(seed=4242):
    eng = nv.SimEngine(None)
    eng.reset("numpy", nv.PRESETS[1][1:], seed)
    return eng


def test_anneal_and_weights():
    print("[1] G 退火与权重变化（真实仿真）")
    eng = make_engine()
    W0 = np.array(eng.be.W, copy=True)
    eng._teacher = DLSTeacher()
    eng._teacher_g = 1.0
    eng._teacher_anneal = True

    g_at_arrival = []
    last = eng.episode
    for _ in range(2400):
        eng.train_tick()
        if eng.episode != last:
            last = eng.episode
            g_at_arrival.append(eng._teacher_g)

    W1 = np.array(eng.be.W, copy=True)
    check(len(g_at_arrival) >= 3, "2400 tick 内至少到达 3 次（实际 %d）" % len(g_at_arrival))
    check(all(b < a for a, b in zip(g_at_arrival, g_at_arrival[1:])),
          "每次到达后 G 严格下降：%s" % [round(x, 4) for x in g_at_arrival[:5]])
    check(abs(g_at_arrival[0] - 0.97) < 1e-9, "第一次 1.00 × 0.97 = 0.97")
    check(not np.array_equal(W0, W1), "训练确实改变了权重")
    print("       W.sum %.2f -> %.2f，G 1.00 -> %.4f，到达 %d 次"
          % (float(W0.sum()), float(W1.sum()), eng._teacher_g, eng.arrivals))
    return eng, W0, W1


def test_roundtrip(eng, W0, W1, tmpdir):
    print("[2] 存档：save -> reset 归零 -> load 精确恢复")
    m = store.save(eng, extra={"preset": "标准 · 520", "seed": 4242})
    p = store.default_path()
    check(os.path.exists(p) and not os.path.exists(p + ".tmp"),
          "原子写入，无 .tmp 残留")
    check(m["w_hash"] == store._weight_fingerprint(W1), "存档指纹 = 训练后权重指纹")
    d = store.describe()
    check(d is not None and d["episode"] == eng.episode, "describe 读回轮次")
    check(store.list_snapshots()[0] == p, "存档出现在列表首位")

    episode, arrivals, g = eng.episode, eng.arrivals, eng._teacher_g
    trace = np.array(eng.be.eligibility_trace, copy=True)
    ghist = list(eng.guidance_hist)
    eng.reset("numpy", nv.PRESETS[1][1:], 4242)
    W_reset = np.array(eng.be.W, copy=True)
    check(np.array_equal(W0, W_reset), "reset 后权重精确回到初始（旧行为：学习成果归零）")
    check(not np.array_equal(W1, W_reset), "reset 后的权重不等于训练后的权重")

    m2 = store.load(eng)
    W_back = np.array(eng.be.W, copy=True)
    check(np.array_equal(W1, W_back), "load 后权重与训练后逐元素一致")
    check(np.array_equal(trace, np.array(eng.be.eligibility_trace)),
          "资格迹一并恢复")
    check(eng.episode == episode and eng.arrivals == arrivals,
          "训练轮次 / 到达次数恢复")
    check(abs(eng._teacher_g - g) < 1e-9, "引导依赖 G 恢复（%.4f）" % g)
    check(len(eng.guidance_hist) == len(ghist), "引导历史恢复（最近 %d tick）"
          % len(ghist))
    check(np.array_equal(np.array(eng.be.Wt), np.ascontiguousarray(W1.T)),
          "Wt 与 W 同步重建")
    check(m2["w_hash"] == m["w_hash"], "载入返回的指纹与存档一致")
    return p


def test_guards(p, tmpdir):
    print("[3] 安全护栏：拓扑不匹配 / 后端不支持 / 损坏文件")
    other = make_engine(seed=999)
    try:
        store.load(other)
        check(False, "换种子后拒绝载入")
    except store.SnapshotError as exc:
        check("拓扑不匹配" in str(exc), "换种子后拒绝载入：%s" % str(exc)[:60])

    class _Fake:
        def __init__(self):
            self.be = object()
    try:
        store.save(_Fake())
        check(False, "C DLL 后端拒绝保存")
    except store.SnapshotError as exc:
        check("C DLL" in str(exc), "C DLL 后端拒绝保存并给出提示")

    bad = os.path.join(tmpdir, "broken.npz")
    with open(bad, "wb") as f:
        f.write(b"not an npz")
    check(store.describe(bad) is None, "损坏存档 describe 返回 None")
    try:
        store.load(other, path=bad)
        check(False, "损坏存档拒绝载入")
    except store.SnapshotError:
        check(True, "损坏存档拒绝载入")
    check(store.delete(p) and not os.path.exists(p), "删除存档")


def test_gui(tmpdir):
    print("[4] 主窗口：G 回写 / 自动存档不抛异常 / 存档窗口")
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    nv.ensure_fonts()
    nv.FONT_FAMILY = nv.pick_font_family()
    app.setStyleSheet(nv.build_qss(nv.THEMES["dark"], nv.FONT_FAMILY))

    w = nv.MainWindow(nv.SimEngine(nv.find_dll()))
    w.engine._teacher_g = 0.42
    w._sync_teacher_g()
    check(abs(w.spin_teacher_g.value() - 0.42) < 1e-9,
          "退火后的 G 回写主窗口数字框")
    check(abs(w.engine._teacher_g - 0.42) < 1e-9,
          "回写不再反向污染引擎值")

    w.combo_dev.blockSignals(True)
    w.combo_dev.setCurrentIndex(1)       # 向量化 · 12 核
    w.combo_dev.blockSignals(False)
    w._reset()
    w._autosave_training()
    check(os.path.exists(store.default_path()), "退出自动保存已写入（调用 _autosave_training）")
    check("已保存" in w.lbl_snapshot.text(), "工具栏显示存档状态")

    import training_panel
    sw = training_panel.SnapshotWindow(
        w.engine, theme=w.T, flags=w._training_flags,
        on_change=w._snapshot_status, extra=w._snapshot_extra())
    sw.refresh()
    check(sw.lst.count() >= 1, "存档窗口列出快照")
    check(sw.vals["w_hash"].text() != "—", "存档窗口显示当前权重指纹")
    sw._save()
    check("已保存" in sw.log.toPlainText(), "窗口内保存可用")
    w.close()
    app.quit()


def test_restart(eng, W1, tmpdir):
    print("[5] 模拟刷新：退出自动保存 -> 重开窗口 -> 启动自动恢复")
    from PySide6.QtWidgets import QApplication
    store.save(eng)                       # 模拟上一次退出时自动保存的训练成果
    app = QApplication.instance() or QApplication([])
    w = nv.MainWindow(nv.SimEngine(nv.find_dll()))
    check(not w._autoload_done,
          "默认 C DLL 启动时不占用唯一一次自动载入机会（状态：%s）"
          % w.lbl_snapshot.text())

    # 恢复成训练时的配置：标准 · 520 / seed 4242 / 向量化 · 12 核
    w.combo_preset.blockSignals(True)
    w.combo_preset.setCurrentIndex(1)
    w.combo_preset.blockSignals(False)
    w.spin_seed.setValue(4242)
    w.chk_seed.setChecked(True)
    w.combo_dev.blockSignals(True)
    w.combo_dev.setCurrentIndex(1)
    w.combo_dev.blockSignals(False)
    w._reset()
    check(w._autoload_done and "已恢复" in w.lbl_snapshot.text(),
          "切到向量化后端后自动恢复：%s" % w.lbl_snapshot.text())
    check(np.array_equal(np.array(w.engine.be.W), W1),
          "刷新后权重与训练后逐元素一致（不再归零）")
    w.close()
    app.quit()


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        store.SNAP_DIR = os.path.join(tmpdir, "training")
        eng, W0, W1 = test_anneal_and_weights()
        p = test_roundtrip(eng, W0, W1, tmpdir)
        test_guards(p, tmpdir)
        test_gui(tmpdir)
        test_restart(eng, W1, tmpdir)
    print()
    if FAILED:
        print("FAILED %d: %s" % (len(FAILED), "; ".join(FAILED)))
        return 1
    print("TRAININGSTORE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
