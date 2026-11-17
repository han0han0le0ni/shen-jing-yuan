# 早停验证（零代码）：`--episodes N --eval-episodes 20` ⇒ 评测的是"训练到 N-20 轮"的状态
# 目的：验证 §6.41 的可检验推论——在 corr(ω,ω*) 峰值（约 ep 40–80）早停，medR/medD 是否优于跑满 320 ep
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_earlystop.ps1
param([int[]]$Episodes = @(40, 60, 80, 120, 160, 240),
      [int[]]$Seeds = @(101, 202, 303, 404, 505, 606, 707, 808),
      [int]$EvalEpisodes = 20)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root 'data_b2\early'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

foreach ($ep in $Episodes) {
    foreach ($s in $Seeds) {
        $csv = Join-Path $dir ("e{0}_s{1}.csv" -f $ep, $s)
        if (Test-Path $csv) { continue }
        $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--error-gain', '1',
               '--episodes', "$ep", '--eval-episodes', "$EvalEpisodes",
               '--seed', "$s", '--out', $csv)
        Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $dir ("log_e{0}_s{1}.txt" -f $ep, $s)) `
            -RedirectStandardError  (Join-Path $dir ("err_e{0}_s{1}.txt" -f $ep, $s))
    }
    # 每档等跑完再进下一档，避免 40 个进程同时抢核
    $n = 0
    while ($n -lt 40) {
        if ((Get-Process benchmark2 -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) { break }
        Start-Sleep -Seconds 10
        $n++
    }
    Write-Host ("ep={0} done (~{1}s)" -f $ep, ($n * 10))
}

$arms = @()
foreach ($ep in $Episodes) { $arms += @('--arm', ("e{0}|K1g1=data_b2/early/e{0}_s*.csv" -f $ep)) }
$arms += @('--arm', 'full320|K1g1=data_b2/t2o/k1_g1/t2o_k1_g1_s*.csv')
& py -3 (Join-Path $root 'tools\t2_arm_summary.py') @arms
