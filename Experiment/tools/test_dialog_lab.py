"""B3 对话训练探针回归测试（含假 LM Studio 评分服务）。

运行： py -3 tools/test_dialog_lab.py
覆盖：数据集/编码、学习效果、三条基线、ECE 校准、巩固、
      条件 B 评分解析与「禁止伪造奖励」、GUI 窗口与主窗口接线。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS = os.path.join(ROOT, "vis")
sys.path.insert(0, VIS)
sys.path.insert(0, os.path.join(ROOT, "tools"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np                                  # noqa: E402
import dialog_lab as dl                              # noqa: E402
import test_lmstudio as t                            # noqa: E402

FAILED = []


def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def test_dataset():
    print("[1] 数据集与脉冲编码")
    train, hold = dl.build_dataset(7)
    check(len(dl.LABELS) == 12, "10–20 个回应标签（实际 %d）" % len(dl.LABELS))
    check(len(train) > 30 and len(hold) >= 12,
          "训练集 %d / 留出集 %d" % (len(train), len(hold)))
    check(not set(x[0] for x in train) & set(x[0] for x in hold),
          "训练/留出集样本不重叠")
    a, b = dl.encode("你好"), dl.encode("你好")
    check(np.array_equal(a, b) and a.shape == (dl.N_IN,),
          "文本编码确定、维度固定、无词表")
    check(not np.array_equal(dl.encode("你好"), dl.encode("再见")),
          "不同话语产生不同输入发放率")
    check(a.max() <= 1.0 and a.min() >= 0.0, "发放率落在 [0,1]")
    check(dl.expected_calibration_error([]) == 0.0, "ECE 空输入返回 0")


def test_learning():
    print("[2] 对话训练是否真的学会（1000 轮）")
    eng = dl.DialogEngine(seed=7)
    eng.step(1000)
    m = eng.metrics()
    acc, ece_raw = eng.evaluate()
    eng.evaluate(calibrate=True)
    _, ece_cal = eng.evaluate()
    print("       train=%.3f hold=%.3f ECE %.3f→%.3f T=%.1f G=%.3f"
          % (m["train_acc"], acc, ece_raw, ece_cal,
             eng.net.temperature, eng.net.teacher_g))
    check(m["train_acc"] > 0.55, "训练一致率显著高于随机 1/12")
    check(acc > 0.45, "留出集一致率 > 0.45（随机约 0.083）")
    check(ece_cal <= ece_raw + 1e-9, "温度缩放校准不恶化 ECE")
    check(0.0 <= ece_cal <= 1.0, "ECE 落在 [0,1]")
    check(eng.net.teacher_g < 0.02, "老师信号已退火，评估为 g=0 口径")

    b = dl.run_baselines(seed=7, rounds=300)
    print("       baselines " + " ".join("%s=%.3f" % (k, v["hold_acc"])
                                        for k, v in b.items()))
    check(b["rstdp"]["hold_acc"] > b["stdp_only"]["hold_acc"] + 0.2,
          "R-STDP 显著高于「纯 STDP 无奖励调制」")
    check(b["rstdp"]["hold_acc"] > b["random_fixed"]["hold_acc"] + 0.2,
          "R-STDP 显著高于「不学习固定随机策略」")

    before, after = eng.consolidate()
    check(after >= before - 1e-9, "睡眠巩固前后保持率不下降（Δ=%+.3f）"
          % (after - before))
    check(eng.metrics()["memory_delta"] is not None, "巩固差值写入指标")
    return eng


def test_ai_scoring():
    print("[3] 条件 B：LM Studio 实时评分 / 禁止伪造奖励")
    srv = t.start_server()
    base = "http://127.0.0.1:%d/v1" % srv.server_address[1]
    try:
        srv.reply = '{"score":0.9}'
        sc = dl.AIScorer(base_url=base, model="fake-local-7b", budget=3)
        r, ok, err = sc.score("你好", dl.LABELS.index("问候"), "问候")
        check(abs(r - 0.8) < 1e-9 and ok and not err, "AI 评分解析 0.9 → 奖励 +0.8")
        check(sc.stats()["latency_ms"] >= 0.0, "记录延迟")
        for _ in range(5):
            sc.score("你好", 0, "问候")
        check(sc.calls == 3, "预算 150→3 轮上限生效（calls=%d）" % sc.calls)

        srv.reply = "模型没给 JSON"
        sc2 = dl.AIScorer(base_url=base, model="fake-local-7b", budget=100)
        eng = dl.DialogEngine(seed=7, ai=True, ai_scorer=sc2)
        w0 = np.array(eng.net.W_out, copy=True)
        base_before, hist_before = eng.net.baseline, len(eng.net.reward_hist)
        for _ in range(10):
            text, gold = eng.train_set[0]
            out = eng.net.forward(text)
            rr, ok2 = eng._reward(text, gold, out)
            check_last = rr
            eng.net.learn(rr, out, dl.LABELS.index(gold))
        check(check_last is None, "评分失败 → 丢弃本次奖励")
        check(np.array_equal(w0, eng.net.W_out), "丢弃奖励时权重不更新")
        check(len(eng.net.reward_hist) == hist_before and
              eng.net.baseline == base_before, "丢弃奖励时不更新基线")
        check(sc2.degraded and sc2.fail_streak >= 5,
              "连续 5 次失败 → 降级内部误差")
        check(eng.learn_frozen, "连续 10 次失败 → 冻结学习")
        st = sc2.stats()
        check(st["calls"] == 10 and st["failures"] == 10,
              "调用 / 失败次数不重复计数（%d / %d）"
              % (st["calls"], st["failures"]))
        check(st["fail_rate"] > 0.9, "失败率可报告")
    finally:
        srv.shutdown()
        srv.server_close()


def test_gui():
    print("[4] 桌面窗口与主窗口接线")
    from PySide6.QtWidgets import QApplication, QPushButton
    import neuron_vis as nv
    app = QApplication([])
    nv.ensure_fonts()
    nv.FONT_FAMILY = nv.pick_font_family()
    app.setStyleSheet(nv.build_qss(nv.THEMES["dark"], nv.FONT_FAMILY))

    eng = dl.DialogEngine(seed=7)
    eng.step(300)
    eng.evaluate(calibrate=True)
    w = dl.DialogLabWindow(eng, theme=nv.THEMES["dark"])
    w.show()
    w.refresh()
    check(w.cards.count() == 8, "8 张指标卡")
    check(len(w.chart_acc.data.get("留出", [])) > 0, "留出集一致率曲线有数据")
    check(len(w.chart_ece.data.get("ECE", [])) > 0, "ECE 曲线有数据")
    w.edit.setText("谢谢你")
    w._predict()
    check("标签" in w.lbl_pred.text() and "回应" in w.lbl_pred.text(),
          "对话测试区返回标签与回应")
    w._baselines()
    check("基线对照" in w.lbl_base.text(), "基线对照可一键运行")

    mw = nv.MainWindow(nv.SimEngine(nv.find_dll()))
    btn = next((b for b in mw.findChildren(QPushButton)
                if b.text() == "对话训练"), None)
    check(btn is not None, "主工具栏出现「对话训练」按钮")
    if btn is not None:
        btn.click()
        check(mw._dialog_win is not None, "按钮打开对话训练探针窗口")
        check(mw._dialog_win.win.windowTitle().startswith("B3"),
              "窗口标题标明 B3 探针")
    mw.close()
    app.quit()


def main():
    test_dataset()
    test_learning()
    test_ai_scoring()
    test_gui()
    print()
    if FAILED:
        print("FAILED %d: %s" % (len(FAILED), "; ".join(FAILED)))
        return 1
    print("DIALOGLAB OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
