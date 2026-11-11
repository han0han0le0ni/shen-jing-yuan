"""SNN 实验项目全面体检 —— 数据 / 日志只读扫描器。

用法:  py audit/scan_data.py
输出:  audit/scan_results.json
不修改任何实验数据，仅新建/覆盖 audit/ 下的检测结果文件。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit"
DATA = ROOT / "data"
DATA_B2 = ROOT / "data_b2"

B2_HEADER = "episode,target_x,target_y,init_dist,final_dist,dist_ratio,final_reward"
B1_HEADER = ("trial,stimulus_type,predicted_vote,correct,output_a_rate,"
             "output_b_rate,reward_given,reward_baseline,elapsed_ticks")


def iso(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def read_rows(p: Path) -> list[list[str]]:
    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.reader(f))


def q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def finite(x: float) -> bool:
    return math.isfinite(x)


def scan_b2_csv(p: Path) -> dict:
    out = {"file": p.name, "path": str(p), "mtime": iso(p), "bytes": p.stat().st_size,
           "flags": [], "n": 0}
    if p.stat().st_size == 0:
        out["flags"].append("empty_file")
        return out
    rows = read_rows(p)
    if not rows:
        out["flags"].append("no_header")
        return out
    header = ",".join(rows[0]).strip()
    if header != B2_HEADER:
        out["flags"].append("header_mismatch")
    body = rows[1:]
    ratios, inits, finals, rewards, targets = [], [], [], [], []
    eps = []
    ragged = nonnumeric = nan_inf = ratio_bad = reward_bad = neg = 0
    for i, row in enumerate(body):
        if len(row) != 7:
            ragged += 1
            if i == len(body) - 1:
                out["flags"].append("last_row_truncated")
            continue
        try:
            v = [float(x) for x in row]
        except ValueError:
            nonnumeric += 1
            continue
        if not all(finite(x) for x in v):
            nan_inf += 1
            continue
        ep, tx, ty, init, final, ratio, rew = v
        eps.append(ep)
        ratios.append(ratio)
        inits.append(init)
        finals.append(final)
        rewards.append(rew)
        targets.append((tx, ty))
        if init <= 0 or abs(ratio - final / init) > 0.0011:
            ratio_bad += 1
        if abs(rew - (1.0 - final / 2.0)) > 0.002:
            reward_bad += 1
        if final < 0 or init < 0 or ratio < 0:
            neg += 1
    for key, cnt in (("ragged_rows", ragged), ("nonnumeric_rows", nonnumeric),
                     ("nan_inf_rows", nan_inf), ("ratio_or_init_mismatch", ratio_bad),
                     ("reward_formula_mismatch", reward_bad), ("negative_values", neg)):
        out[key] = cnt
        if cnt:
            out["flags"].append(key)
    if eps and eps != list(range(len(eps))):
        out["flags"].append("episode_gap_or_duplicate")
    if len(set(targets)) > 1:
        out["flags"].append("multiple_targets")
    if not ratios:
        out["flags"].append("no_valid_rows")
        return out
    out.update({
        "n": len(ratios),
        "median_ratio": round(statistics.median(ratios), 4),
        "mean_ratio": round(statistics.fmean(ratios), 4),
        "min_ratio": round(min(ratios), 4),
        "max_ratio": round(max(ratios), 4),
        "p25": round(q(ratios, 0.25), 4),
        "p75": round(q(ratios, 0.75), 4),
        "frac_le_target": round(sum(r <= 0.20 for r in ratios) / len(ratios), 4),
        "frac_worse_than_start": round(sum(r > 1.0 for r in ratios) / len(ratios), 4),
        "init_dist_unique": len(set(round(x, 4) for x in inits)),
        "target_unique": len(set(targets)),
        "target_sample": list(targets[0]) if targets else None,
        "final_reward_min": round(min(rewards), 4),
        "final_reward_max": round(max(rewards), 4),
    })
    k = max(1, len(ratios) // 10)
    out["first_decile_median"] = round(statistics.median(ratios[:k]), 4)
    out["last_decile_median"] = round(statistics.median(ratios[-k:]), 4)
    out["learning_delta"] = round(out["first_decile_median"] - out["last_decile_median"], 4)
    step = max(1, len(ratios) // 400)
    out["curve"] = [round(x, 4) for x in ratios[::step]]
    out["csv_status"] = "PASS" if out["median_ratio"] <= 0.20 else "FAIL"
    return out


def scan_b1_csv(p: Path) -> dict:
    out = {"file": p.name, "path": str(p), "mtime": iso(p), "bytes": p.stat().st_size,
           "flags": [], "n": 0}
    if p.stat().st_size == 0:
        out["flags"].append("empty_file")
        return out
    rows = read_rows(p)
    header = ",".join(rows[0]).strip()
    if header != B1_HEADER:
        out["flags"].append("header_mismatch")
    trials, correct, bad_correct, ragged, nonnumeric, nan_inf, bad_rates = [], [], 0, 0, 0, 0, 0
    for i, row in enumerate(rows[1:]):
        if len(row) != 9:
            ragged += 1
            if i == len(rows) - 2:
                out["flags"].append("last_row_truncated")
            continue
        try:
            v = [float(x) for x in row]
        except ValueError:
            nonnumeric += 1
            continue
        if not all(finite(x) for x in v):
            nan_inf += 1
            continue
        trial, stim, pred, cor, ra, rb, rew, base, ticks = v
        trials.append(trial)
        correct.append(cor)
        expected = 1 if stim == 0 else 2
        if int(cor) != int(pred == expected):
            bad_correct += 1
        if not (0.0 <= ra <= 1.0 and 0.0 <= rb <= 1.0):
            bad_rates += 1
    for key, cnt in (("ragged_rows", ragged), ("nonnumeric_rows", nonnumeric),
                     ("nan_inf_rows", nan_inf), ("correct_flag_mismatch", bad_correct),
                     ("rate_out_of_range", bad_rates)):
        out[key] = cnt
        if cnt:
            out["flags"].append(key)
    if trials and trials != list(range(len(trials))):
        out["flags"].append("trial_gap_or_duplicate")
    if not correct:
        out["flags"].append("no_valid_rows")
        return out
    n = len(correct)
    half = n // 2
    out.update({
        "n": n,
        "overall_acc": round(100.0 * sum(correct) / n, 2),
        "train_acc": round(100.0 * sum(correct[:half]) / half, 2) if half else None,
        "test_acc": round(100.0 * sum(correct[half:]) / (n - half), 2) if n > half else None,
        "csv_status": "PASS" if (n > half and 100.0 * sum(correct[half:]) / (n - half) >= 80) else "FAIL",
    })
    return out


RE_CFG = re.compile(r"Config:\s*in=(\d+)\s+h=(\d+)\s+m1=(\d+)\s+m2=(\d+)\s+episodes=(\d+)\s+ticks/ep=(\d+)\s+seed=(\d+)")
RE_NET = re.compile(r"Net:\s*(\d+)\s+neurons,\s*(\d+)\s+synapses")
RE_EP = re.compile(r"^\s*Ep\s+(\d+)\s*\|")
RE_NUM = re.compile(r"^\s*(Median|Mean|Min|Max):\s*([-\d.]+)")


def parse_b2_log(p: Path) -> dict:
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out = {"file": p.name, "path": str(p), "mtime": iso(p), "bytes": p.stat().st_size,
           "lines": len(lines), "flags": []}
    m = RE_CFG.search(text)
    if m:
        out["config"] = dict(zip(("input", "hidden", "motor1", "motor2", "episodes",
                                  "ticks", "seed"), (int(x) for x in m.groups())))
    m = RE_NET.search(text)
    if m:
        out["net"] = {"neurons": int(m.group(1)), "synapses": int(m.group(2))}
    ep_ids = [int(mm.group(1)) for line in lines if (mm := RE_EP.match(line))]
    out["episode_lines"] = len(ep_ids)
    out["last_episode"] = ep_ids[-1] if ep_ids else None
    for line in lines:
        mm = RE_NUM.match(line)
        if mm:
            out[mm.group(1).lower()] = float(mm.group(2))
    mm = re.search(r"\[(PASS|FAIL)\][^\n]*", text)
    out["status"] = mm.group(1) if mm else None
    out["pass_line"] = mm.group(0).strip() if mm else None
    if "Median:" not in text:
        out["flags"].append("no_final_results")
    if "NativeCommandError" in text or "RemoteException" in text:
        out["flags"].append("powershell_stderr_noise")
    if "REWARD_DEBUG" in text or "MAX_TRACE" in text:
        out["flags"].append("legacy_debug_print")
    if out["episode_lines"] == 0:
        out["flags"].append("no_episode_lines")
    return out


def parse_b1_results(p: Path) -> list[dict]:
    text = p.read_text(encoding="utf-8", errors="replace")
    blocks = [b for b in re.split(r"(?==== Benchmark-1)", text) if "Seed=" in b]
    out = []
    for b in blocks:
        seed = re.search(r"Seed=(\d+)", b)
        test = re.search(r"Test Accuracy:\s*([\d.]+)%", b)
        overall = re.search(r"Overall Accuracy:\s*([\d.]+)%", b)
        train = re.search(r"Train Accuracy:\s*([\d.]+)%", b)
        trace = re.search(r"El Trace:\s*([^\n,]+)", b)
        status = re.search(r"\[(PASS|FAIL)\]", b)
        out.append({
            "source": p.name, "seed": int(seed.group(1)) if seed else None,
            "test_acc": float(test.group(1)) if test else None,
            "train_acc": float(train.group(1)) if train else None,
            "overall_acc": float(overall.group(1)) if overall else None,
            "el_trace": trace.group(1).strip() if trace else None,
            "status": status.group(1) if status else None,
        })
    return out


def find_log_for(stem: str) -> Path | None:
    candidates = [DATA_B2 / (stem + ".log")]
    if stem.startswith("results_"):
        tail = stem[len("results_"):]
        candidates += [DATA_B2 / ("run_" + tail + ".log"), DATA_B2 / (tail + ".log")]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> None:
    result = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "root": str(ROOT), "method": "只读扫描：CSV 逐行解析 + 日志正则解析 + 二进制标记检查"}

    b2_csvs = sorted(DATA_B2.glob("*.csv"))
    b2 = [scan_b2_csv(p) for p in b2_csvs]
    paired_logs = set()
    for rec in b2:
        stem = Path(rec["file"]).stem
        lp = find_log_for(stem)
        if lp:
            paired_logs.add(lp.name)
            rec["paired_log"] = lp.name
            rec["log"] = parse_b2_log(lp)
            rec["status"] = rec["log"].get("status") or rec.get("csv_status")
            if rec["log"].get("median") is not None and rec.get("median_ratio") is not None:
                if abs(rec["log"]["median"] - rec["median_ratio"]) > 0.005:
                    rec["flags"].append("csv_log_median_mismatch")
            cfg = rec["log"].get("config") or {}
            if cfg.get("episodes") and rec["n"] and rec["n"] != cfg["episodes"]:
                rec["flags"].append("csv_log_episode_count_mismatch")
                rec["declared_episodes"] = cfg["episodes"]
        else:
            rec["paired_log"] = None
            rec["status"] = rec.get("csv_status")
    result["b2_runs"] = b2

    b1_csvs = sorted(DATA.glob("*.csv")) + sorted((DATA / "multi_seed").glob("*.csv"))
    b1 = [scan_b1_csv(p) for p in b1_csvs]
    result["b1_runs"] = b1

    seed_results = []
    for nm in ("results.txt", "abl_trace.txt", "abl_reward.txt"):
        p = DATA / "multi_seed" / nm
        if p.is_file():
            seed_results += parse_b1_results(p)
    result["b1_seeds"] = seed_results

    early_logs = []
    for folder in (ROOT, ROOT / "logs", ROOT / "logs_b2"):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.log")):
            text_head = p.read_text(encoding="utf-8", errors="replace")[:4000]
            if "Benchmark-2" not in text_head and "Median:" not in p.read_text(encoding="utf-8", errors="replace")[-4000:]:
                continue
            rec = parse_b2_log(p)
            rec["folder"] = folder.name
            early_logs.append(rec)
    result["root_logs"] = early_logs

    all_b2_logs = sorted(DATA_B2.glob("*.log"))
    result["orphan_logs"] = [p.name for p in all_b2_logs if p.name not in paired_logs]
    result["orphan_csvs"] = [r["file"] for r in b2 if not r["paired_log"]]

    key = ["brain_b2/benchmark2_core.c", "brain_b2/benchmark2_core.h",
           "host_b2/benchmark2_host.cpp", "benchmark2.exe", "b2_core.dll", "b2_core.o",
           "brain/benchmark1_core.c", "brain/benchmark1_core.h",
           "host/benchmark1_host.cpp", "benchmark1.exe"]
    result["artifacts"] = {}
    for rel in key:
        p = ROOT / rel
        if p.is_file():
            result["artifacts"][rel] = {"mtime": iso(p), "bytes": p.stat().st_size, "sha16": sha(p)}

    markers = [b"REWARD_DEBUG", b"MAX_TRACE", b"combo1", b"wstats", b"PASS CHECK", b"Benchmark-2"]
    result["binary_markers"] = {}
    for nm in ("benchmark2.exe", "b2_core.dll", "benchmark1.exe"):
        p = ROOT / nm
        if p.is_file():
            raw = p.read_bytes()
            result["binary_markers"][nm] = {m.decode(): (m in raw) for m in markers}

    qc = {
        "b2_csv_count": len(b2), "b2_log_count": len(all_b2_logs),
        "b1_csv_count": len(b1), "root_log_count": len(early_logs),
        "b2_total_episodes": sum(r.get("n", 0) for r in b2),
        "b1_total_trials": sum(r.get("n", 0) for r in b1),
        "b2_empty_files": [r["file"] for r in b2 if "empty_file" in r["flags"]],
        "b1_empty_files": [r["file"] for r in b1 if "empty_file" in r["flags"]],
        "b2_bad_files": [{"file": r["file"], "flags": r["flags"]} for r in b2 if r["flags"]],
        "b1_bad_files": [{"file": r["file"], "flags": r["flags"]} for r in b1 if r["flags"]],
        "b2_ragged_rows": sum(r.get("ragged_rows", 0) for r in b2),
        "b2_ratio_mismatch_rows": sum(r.get("ratio_or_init_mismatch", 0) for r in b2),
        "b2_reward_mismatch_rows": sum(r.get("reward_formula_mismatch", 0) for r in b2),
        "b1_correct_mismatch_rows": sum(r.get("correct_flag_mismatch", 0) for r in b1),
        "logs_without_final_results": [r["file"] for r in early_logs + [x["log"] for x in b2 if x.get("log")]
                                       if "no_final_results" in r["flags"]],
        "logs_with_legacy_debug": [r["file"] for r in early_logs + [x["log"] for x in b2 if x.get("log")]
                                   if "legacy_debug_print" in r["flags"]],
        "logs_with_powershell_noise": [r["file"] for r in early_logs + [x["log"] for x in b2 if x.get("log")]
                                       if "powershell_stderr_noise" in r["flags"]],
    }
    result["quality"] = qc
    result["summary"] = {
        "b2_pass": sum(r.get("status") == "PASS" for r in b2),
        "b2_fail": sum(r.get("status") == "FAIL" for r in b2),
        "b2_unknown": sum(r.get("status") is None for r in b2),
        "b1_pass": sum(r.get("csv_status") == "PASS" for r in b1),
        "b1_fail": sum(r.get("csv_status") == "FAIL" for r in b1),
        "root_log_pass": sum(r.get("status") == "PASS" for r in early_logs),
        "root_log_fail": sum(r.get("status") == "FAIL" for r in early_logs),
        "b1_seed_pass": sum(r.get("status") == "PASS" for r in seed_results),
        "b1_seed_fail": sum(r.get("status") == "FAIL" for r in seed_results),
    }

    AUDIT.mkdir(exist_ok=True)
    out_path = AUDIT / "scan_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", out_path)
    print("SUMMARY", json.dumps(result["summary"], ensure_ascii=False))
    print("QUALITY", json.dumps({k: v for k, v in qc.items() if not isinstance(v, list)}, ensure_ascii=False))
    print("B2 BAD FILES", len(qc["b2_bad_files"]), "B1 BAD FILES", len(qc["b1_bad_files"]))
    print("ORPHAN CSV", len(result["orphan_csvs"]), "ORPHAN LOG", len(result["orphan_logs"]))
    for r in qc["b2_bad_files"][:20]:
        print("  B2", r["file"], r["flags"])
    for r in qc["b1_bad_files"][:20]:
        print("  B1", r["file"], r["flags"])


if __name__ == "__main__":
    main()
