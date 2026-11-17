# v16 批量：8 seed × 320 ep，导出 init/final 权重 + 命令轨迹
# 目的：验证「学习把 H→M 压到下限」是否跨 seed 稳定（§6.40 只有 seed 202 单点）
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_v16_8seed.ps1
param([int[]]$Seeds = @(101, 202, 303, 404, 505, 606, 707, 808),
      [int]$Episodes = 320, [int]$EvalEpisodes = 20, [int]$TraceStride = 20,
      [switch]$SummaryOnly)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root 'data_b2\v16_8s'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

if (-not $SummaryOnly) {
    foreach ($s in $Seeds) {
        $csv = Join-Path $dir ("s{0}.csv" -f $s)
        if (Test-Path $csv) { Write-Host ("skip s{0}" -f $s); continue }
        $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--error-gain', '1',
               '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
               '--seed', "$s",
               '--save-init-weights', (Join-Path $dir ("w_init_s{0}.txt" -f $s)),
               '--save-weights',      (Join-Path $dir ("w_final_s{0}.txt" -f $s)),
               '--trace-cmd',         (Join-Path $dir ("cmd_s{0}.jsonl" -f $s)),
               '--trace-stride', "$TraceStride",
               '--out', $csv)
        Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $dir ("log_s{0}.txt" -f $s)) `
            -RedirectStandardError  (Join-Path $dir ("err_s{0}.txt" -f $s))
        Write-Host ("launched s{0}" -f $s)
    }

    $n = 0
    while ($n -lt 60) {
        $running = (Get-Process benchmark2 -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($running -eq 0) { break }
        Start-Sleep -Seconds 15
        $n++
    }
    Write-Host ("waited ~{0}s" -f ($n * 15))
}

& py (Join-Path $PSScriptRoot 'v16_agg.py') --dir data_b2/v16_8s
