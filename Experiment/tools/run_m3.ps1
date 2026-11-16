# M3 · 学习增益 × 权重映射增益（负责人 2026-11-16「签字」）
# 承接：audit/m1_m2_workpoint_report_2026-11-16.md §5/§7
#   C1 工作点（hb=0, psp=1.5）机制修好但闭环 +29% → 检验"是不是学习增益与 PSP 失配"
# 四新臂 × 3 seed，判定线不变（median ≤ 0.40，阈值零变更）：
#   L3   : C1 + lr ×3        （学习增益上调一档）
#   L16  : C1 + lr ×16.7     （把"电压意义上的学习增益"拉回基线水平）
#   W6   : psp=25 + 权重×0.06（前向驱动与 C1 对齐，而学习增益留在基线）
#   W6L  : psp=25 + 权重×0.06 + lr×0.06（相对权重步长也与基线对齐）
# 复用：C1 三 seed = data_b2/m2_c1（0.6956）; 基线 = data_b2/t2o/k1_g1（8 seed 0.5398）
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_m3.ps1 -Launch
param([switch]$Launch, [switch]$SummaryOnly, [string]$SeedList = '101,202,303',
      [int]$Episodes = 320, [int]$EvalEpisodes = 20)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$seeds = @($SeedList.Split(',') | ForEach-Object { [int]$_.Trim() })

$arms = @(
    @{ n = 'L3';  psp = '1.5'; hb = '0'; ws = '1';    lrs = '3' },
    @{ n = 'L16'; psp = '1.5'; hb = '0'; ws = '1';    lrs = '16.667' },
    @{ n = 'W6';  psp = '25';  hb = '0'; ws = '0.06'; lrs = '1' },
    @{ n = 'W6L'; psp = '25';  hb = '0'; ws = '0.06'; lrs = '0.06' }
)

function Show-Summary {
    & py -3 (Join-Path $root 'tools\t2_arm_summary.py') `
        --arm "base|K1g1=data_b2/t2o/k1_g1/t2o_k1_g1_s*.csv" `
        --arm "C1|psp1.5=data_b2/m2_c1/m2c1_s*.csv" `
        --arm "L3|psp1.5lr3=data_b2/m3/L3_s*.csv" `
        --arm "L16|psp1.5lr16=data_b2/m3/L16_s*.csv" `
        --arm "W6|w0.06psp25=data_b2/m3/W6_s*.csv" `
        --arm "W6L|w0.06lr0.06=data_b2/m3/W6L_s*.csv"
}

if ($SummaryOnly) { Show-Summary; exit 0 }

foreach ($a in $arms) {
    $dir = Join-Path $root ("data_b2\m3")
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($s in $seeds) {
        $out = Join-Path $dir ("{0}_s{1}.csv" -f $a.n, $s)
        if (Test-Path $out) { Write-Host ("skip {0} s{1}" -f $a.n, $s); continue }
        $cmdArgs = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--error-gain', '1',
                     '--hidden-base', $a.hb, '--psp', $a.psp,
                     '--weight-scale', $a.ws, '--lr-scale', $a.lrs,
                     '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
                     '--seed', "$s", '--out', $out)
        Start-Process -FilePath $exe -ArgumentList $cmdArgs -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $dir ("log_{0}_s{1}.txt" -f $a.n, $s)) `
            -RedirectStandardError  (Join-Path $dir ("err_{0}_s{1}.txt" -f $a.n, $s))
    }
}

$total = $arms.Count * $seeds.Count
while ($true) {
    $done = (Get-ChildItem (Join-Path $root 'data_b2\m3') -Filter *.csv -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host ("{0}  {1}/{2} csv" -f (Get-Date -Format 'HH:mm:ss'), $done, $total)
    if ($done -ge $total) { break }
    Start-Sleep -Seconds 30
}
Show-Summary
