"""单突触资格迹对照：B1 规则（C 侧手写，核心零改动）vs neuromod 0.6.0（Rust 侧）。

驱动的 Δt 序列两侧完全相同（同一 seq 文件），τe 对齐到给定值。
只比「迹的衰减律 + 核形状 + 归一化迹曲线」，不比整网正确率（口径见
audit/b1_prereg_neuromod_rule_2026-11-14.md 与 v1.17 结论）。

用法：py -3 tools/nm_synapse_compare.py [--tau 100]
产物：data/b1_nm_syn/pairs.jsonl + audit/b1_neuromod_synapse_compare_2026-11-14.md
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "b1_nm_syn"
REPORT = ROOT / "audit" / "b1_neuromod_synapse_compare_2026-11-14.md"
C_EXE = ROOT / "nm_compare.exe"
R_EXE = ROOT / "ref_neuromod" / "target" / "release" / "nm_ref_trace.exe"
SEQ = DATA / "seq.txt"
NPairs = 3000
KERNEL_PTS = [-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0]


def make_seq(path):
    """确定性 LCG 生成 "gap dt" 序列（两侧共用，可复算）。"""
    s = 20261114
    gaps = [1, 2, 5, 10, 30, 100]
    lines = []
    for _ in range(NPairs):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        gap = gaps[(s >> 8) % len(gaps)]
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        dt = float((s >> 8) % 41 - 20)          # -20..20 整数步
        lines.append(f"{gap} {dt:.0f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def parse(path):
    out = {"vals": []}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("KERNEL"):
            out["kernel"] = [float(x) for x in line.split()[1:]]
        elif line.startswith("DECAY"):
            out["decay"] = float(line.split()[1])
        elif line.startswith("RATIO"):
            out["ratio"] = float(line.split()[1])
        elif line.startswith("N "):
            out["n"] = int(line.split()[1])
        else:
            out["vals"].append(float(line))
    return out


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 1e-12 else float("nan")


def pearson(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 1e-12 else float("nan")


def norm(xs):
    m = max(abs(x) for x in xs) or 1.0
    return [x / m for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=100.0)
    args = ap.parse_args()
    tau = args.tau

    if not C_EXE.exists():
        raise SystemExit(f"缺少 {C_EXE}（gcc -O2 -o nm_compare.exe tools/nm_compare.c -lm）")
    if not R_EXE.exists():
        raise SystemExit(f"缺少 {R_EXE}（cd ref_neuromod && cargo build --release）")

    n = make_seq(SEQ)
    cout, rout = DATA / "c_out.txt", DATA / "r_out.txt"
    subprocess.run([str(C_EXE), str(SEQ), f"{tau}"] , check=True,
                   stdout=open(cout, "w", encoding="utf-8"))
    subprocess.run([str(R_EXE), str(SEQ), f"{tau}", str(rout)], check=True)
    c, r = parse(cout), parse(rout)

    cv, rv = c["vals"], r["vals"]
    m = min(len(cv), len(rv))
    cv, rv = cv[:m], rv[:m]
    cn, rn = norm(cv), norm(rv)

    kc, kr = c["kernel"], r["kernel"]
    kn_c, kn_r = norm(kc), norm(kr)
    kdev = max(abs(a - b) for a, b in zip(kn_c, kn_r))

    decay_err = abs(r["decay"] - c["decay"]) / max(abs(c["decay"]), 1e-30)
    amp_ratio = statistics.fmean([abs(a) for a in kr]) / max(
        statistics.fmean([abs(a) for a in kc]), 1e-30)
    rho_trace = spearman(cn, rn)
    r_trace = pearson(cn, rn)

    ok_decay = decay_err <= 1e-5
    ok_kernel = kdev <= 0.05 and spearman(kn_c, kn_r) >= 0.99
    ok_trace = rho_trace >= 0.8

    DATA.mkdir(parents=True, exist_ok=True)
    rec = {
        "tau": tau, "n_pairs": n, "n_curves": m,
        "c_decay": c["decay"], "r_decay": r["decay"], "decay_rel_err": decay_err,
        "kernel_c": kc, "kernel_r": kr, "kernel_norm_max_dev": kdev,
        "kernel_amp_ratio_r_over_c": amp_ratio,
        "trace_spearman": rho_trace, "trace_pearson": r_trace,
        "c_first_last": [cv[0], cv[-1]], "r_first_last": [rv[0], rv[-1]],
        "ok_decay": ok_decay, "ok_kernel": ok_kernel, "ok_trace": ok_trace,
    }
    with (DATA / "pairs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    a = []
    ap_ = a.append
    ap_("# 单突触资格迹对照：B1 规则 vs neuromod 0.6.0（§1 第 4 项 / v1.19）")
    ap_("")
    ap_(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')} · τe 对齐 **{tau:g}** · "
        f"驱动序列 {n} 个脉冲对（gap∈{{1,2,5,10,30,100}}，dt∈[-20,20] 整数步，"
        f"确定性 LCG 生成，两侧同一文件）")
    ap_("> C 侧：`tools/nm_compare.c` 按**核心同常数**手写（τ_stdp=50、a₊=0.1、a₋=0.12），"
        "核心 `benchmark1_core.c` 零改动；Rust 侧：`ref_neuromod` 调 crate 的 `EligibilityTrace`。")
    ap_(f"> Rust 工具链：rustc 1.100.0 / cargo 1.100.0；crate `neuromod 0.6.0`。")
    ap_("")
    ap_("## 0. 结论")
    ap_("")
    ap_(f"1. **衰减律**：C 侧 `exp(-1/τ)` = {c['decay']:.12f}，Rust 侧 = {r['decay']:.12f}"
        f"（相对误差 {decay_err:.2e}）→ {'**等价**' if ok_decay else '**不等价**'}。")
    ap_(f"2. **STDP 核**：归一化后在 7 个采样点上的最大偏差 **{kdev:.4f}**，"
        f"τ₊ 不同（B1 50 vs 20）；幅度比 Rust/C = {amp_ratio:.4f}（A₊ 0.01 vs 0.1，约 10×）→ "
        f"{'**等价**' if ok_kernel else '**不等价**'}。")
    ap_(f"3. **迹曲线**（{m} 点，按各自峰值归一化）：Spearman ρ = **{rho_trace:.3f}**，"
        f"Pearson r = {r_trace:.3f} → {'同形' if ok_trace else '**不同形**'}。")
    ap_("")
    if ok_decay and not ok_kernel:
        ap_("**判定**：两条规则的**迹衰减律是同一个公式**，但**核不等价**——"
            "`neuromod` 的 A₊/A₋/τ± 是**编译期常量**（0.01/0.012/20），"
            "`RmStdpConfig` 只暴露 τe / reward_lr / w_min / w_max，**无法对齐到 B1 的 0.1/0.12/50**；"
            "因此「统计等价」在核层面**结构性不成立**，不是调参能解决的。")
        ap_("")
        ap_("这意味着 §1 第 4 项要么（a）由负责人重新定义等价口径（只比衰减律/归一化迹曲线，不比核），"
            "要么（b）如实记为「两侧规则不等价」并保留为 B1 的未通过理由。**本轮不代为选择。**")
    elif ok_decay and ok_kernel:
        ap_("**判定**：衰减律与核在给定容差内都等价（需要与判据口径再核对一次才算销项）。")
    else:
        ap_("**判定**：连衰减律都不等价，需先查两侧衰减实现。")
    ap_("")
    ap_("## 1. 核采样（同一 Δt 点）")
    ap_("")
    ap_("| Δt | C（B1 规则） | Rust（neuromod） | C 归一化 | Rust 归一化 | 归一化偏差 |")
    ap_("|---|---|---|---|---|---|")
    for i, dt in enumerate(KERNEL_PTS):
        ap_(f"| {dt:+.0f} | {kc[i]:.6f} | {kr[i]:.6f} | {kn_c[i]:.4f} | {kn_r[i]:.4f} | "
            f"{abs(kn_c[i] - kn_r[i]):.4f} |")
    ap_("")
    ap_("## 2. 迹曲线统计")
    ap_("")
    ap_("| 量 | 值 |")
    ap_("|---|---|")
    ap_(f"| 脉冲对数 | {n} |")
    ap_(f"| 单步衰减因子（C / Rust） | {c['decay']:.12f} / {r['decay']:.12f} |")
    ap_(f"| 归一化迹曲线 Spearman ρ | {rho_trace:.4f} |")
    ap_(f"| 归一化迹曲线 Pearson r | {r_trace:.4f} |")
    ap_(f"| 首/末值（C） | {cv[0]:.6f} / {cv[-1]:.6f} |")
    ap_(f"| 首/末值（Rust） | {rv[0]:.6f} / {rv[-1]:.6f} |")
    ap_("")
    ap_("## 3. 诚实边界")
    ap_("")
    ap_("- 这**不是**整网等价：拓扑、RNG、权重归一化（L1）都不同，本节只比单突触层面的迹与核。")
    ap_("- C 侧是**按核心同常数手写**的对照实现（核心零改动），不是直接调用核心；"
       "因此结论覆盖「规则」，不覆盖「核心代码路径」。")
    ap_("- Rust 侧用 f32、C 侧用 double，衰减因子存在 ~1e-7 量级差异，已在容差 1e-5 内。")
    ap_("- 驱动序列是合成的（非 B1 真实脉冲对）；B1 核心不暴露单突触 Δt 序列，"
       "要拿真实序列必须改核心，违反冻结。")
    ap_("")
    ap_("## 4. 复现")
    ap_("")
    ap_("```")
    ap_("cd e:/shen_jing_yuan/Experiment")
    ap_("gcc -O2 -o nm_compare.exe tools/nm_compare.c -lm")
    ap_("cd ref_neuromod && cargo build --release && cd ..")
    ap_("py -3 tools/nm_synapse_compare.py --tau 100")
    ap_("```")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(a) + "\n", encoding="utf-8")
    print(f"tau={tau:g} decay_err={decay_err:.2e} kernel_dev={kdev:.4f} "
          f"rho_trace={rho_trace:.3f} -> {REPORT}")
    return rec


if __name__ == "__main__":
    main()
