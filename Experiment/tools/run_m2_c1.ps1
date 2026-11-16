# M2 · 工作点 C1（hb=0, psp=1.5）× K=1 + error_gain=1 闭环
# 承接 M1：audit/m1_workpoint_prereg_2026-11-16.md §3 通过分支
# 基线：data_b2/t2o/k1_g1/（§6.33/§6.34，合并 0.5398 / 三 seed 0.5843）
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_m2_c1.ps1 -Launch
param([switch]$Launch, [switch]$SummaryOnly, [string]$SeedList = '101,202,303',
      [int]$Episodes = 320, [int]$EvalEpisodes = 20)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root 'data_b2\m2_c1'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$seeds = @($SeedList.Split(',') | ForEach-Object { [int]$_.Trim() })

function Show-Summary {
    & py -3 (Join-Path $root 'tools\t2_arm_summary.py') `
        --arm "base|K1g1=data_b2/t2o/k1_g1/t2o_k1_g1_s*.csv" `
        --arm "C1|K1g1=data_b2/m2_c1/m2c1_s*.csv"
}

if ($SummaryOnly) { Show-Summary; exit 0 }

foreach ($s in $seeds) {
    $out = Join-Path $dir "m2c1_s$s.csv"
    if (Test-Path $out) { Write-Host "skip seed $s"; continue }
    $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--error-gain', '1',
           '--hidden-base', '0', '--psp', '1.5',
           '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
           '--seed', "$s", '--out', $out)
    Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $dir "log_s$s.txt") `
        -RedirectStandardError  (Join-Path $dir "err_s$s.txt")
    Write-Host ("launched seed {0}" -f $s)
}

if (-not $Launch) { Write-Host 'launched (waiting)'; }

$total = $seeds.Count
while ($true) {
    $done = (Get-ChildItem $dir -Filter *.csv -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host ("{0}  {1}/{2} csv" -f (Get-Date -Format 'HH:mm:ss'), $done, $total)
    if ($done -ge $total) { break }
    Start-Sleep -Seconds 60
}
Show-Summary
