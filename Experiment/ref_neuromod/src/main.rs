// 与 neuromod 0.6.0 的 EligibilityTrace 做单突触迹对照（Rust 侧）。
// 输入：seq 文件，每行 "gap dt"（先空转 gap 步衰减，再按 dt 累积一次）
// 输出：KERNEL 行 + DECAY 行 + 每次累积后的迹值（一行一个）
use neuromod::EligibilityTrace;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: nm_ref_trace <seq> <tau> [out]");
        std::process::exit(2);
    }
    let seq_path = &args[1];
    let tau: f32 = args[2].parse().expect("tau");
    let out_path = args.get(3).cloned();

    let text = fs::read_to_string(seq_path).expect("read seq");
    let mut tr = EligibilityTrace::new(tau);

    let mut kernel = String::new();
    for dt in [-20.0f32, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0] {
        kernel.push_str(&format!("{:.10} ", EligibilityTrace::kernel(dt)));
    }

    let mut vals: Vec<f64> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let gap: u32 = it.next().expect("gap").parse().expect("gap parse");
        let dt: f32 = it.next().expect("dt").parse().expect("dt parse");
        for _ in 0..gap {
            tr.decay();
        }
        tr.accumulate(dt);
        vals.push(tr.value as f64);
    }

    // 单步衰减因子：取末值前后的比值
    let before = tr.value;
    tr.decay();
    let ratio = if before.abs() > 1e-12 {
        (tr.value / before) as f64
    } else {
        f64::NAN
    };

    let mut out = String::new();
    out.push_str(&format!("KERNEL {}\n", kernel.trim_end()));
    out.push_str(&format!("DECAY {:.12}\n", ratio));
    out.push_str(&format!("N {}\n", vals.len()));
    for v in &vals {
        out.push_str(&format!("{:.10}\n", v));
    }

    match out_path {
        Some(p) => fs::write(p, out).expect("write out"),
        None => print!("{}", out),
    }
}
