/* nm_compare.c - 单突触资格迹对照（C 侧，按 B1 核心同常数手写更新）
 *
 * 用途：与 neuromod 0.6.0 的 EligibilityTrace 在**同一 Δt 驱动序列**上对照。
 * 这不是核心代码：核心 benchmark1_core.c 零改动，本文件独立编译成 nm_compare.exe。
 * 常数取核心默认：tau_stdp=50、a_plus=0.1、a_minus=0.12、alpha=1（单尺度消融配置）。
 *
 * 输入：seq 文件，每行 "gap dt"：先空转 gap 步衰减，再按 dt 累积一次核值。
 * 输出：KERNEL 行（固定采样点）+ DECAY 行（单步衰减因子）+ N 行 + 每次累积后的迹值。
 *
 * 编译：gcc -O2 -o nm_compare.exe tools/nm_compare.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TAU_STDP   50.0
#define A_PLUS     0.1
#define A_MINUS    0.12

static double b1_kernel(double dt)
{
    if (dt >= 0.0) return A_PLUS * exp(-dt / TAU_STDP);
    return -A_MINUS * exp(dt / TAU_STDP);
}

int main(int argc, char** argv)
{
    FILE* fp;
    FILE* out;
    double tau_e, decay, e = 0.0, ratio;
    char line[256];
    int n = 0, first = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: nm_compare <seq> <tau_e> [out]\n");
        return 2;
    }
    tau_e = atof(argv[2]);
    if (!(tau_e > 0.0)) { fprintf(stderr, "bad tau\n"); return 2; }
    decay = exp(-1.0 / tau_e);

    fp = fopen(argv[1], "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    out = (argc > 3) ? fopen(argv[3], "w") : stdout;
    if (!out) { fclose(fp); return 2; }

    fprintf(out, "KERNEL");
    {
        const double pts[] = { -20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0 };
        int i;
        for (i = 0; i < 7; i++) fprintf(out, " %.10f", b1_kernel(pts[i]));
        fprintf(out, "\n");
    }
    fprintf(out, "DECAY %.12f\n", decay);

    /* 先数行数，便于输出 N；同时把值缓存起来 */
    {
        double* vals = NULL;
        int cap = 0;
        double before;
        while (fgets(line, sizeof(line), fp)) {
            int gap;
            double dt;
            if (sscanf(line, "%d %lf", &gap, &dt) != 2) continue;
            while (gap-- > 0) e *= decay;
            e += b1_kernel(dt);
            if (n >= cap) {
                cap = cap ? cap * 2 : 1024;
                vals = (double*)realloc(vals, (size_t)cap * sizeof(double));
                if (!vals) { fclose(fp); return 3; }
            }
            vals[n++] = e;
        }
        fprintf(out, "N %d\n", n);
        {
            int i;
            for (i = 0; i < n; i++) fprintf(out, "%.10f\n", vals[i]);
        }
        before = e;
        ratio = (fabs(before) > 1e-12) ? (before * decay / before) : NAN;
        fprintf(out, "RATIO %.12f\n", ratio);
        free(vals);
        (void)first;
    }

    if (out != stdout) fclose(out);
    fclose(fp);
    return 0;
}
