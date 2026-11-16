# T2k · 攀爬纤维误差通道（error_gain）单变量扫尾
# 预注册：audit/t2k_prereg_2026-11-16.md
# 臂：g0(对照,复现 §6.22) / g1 / g5；其余全为 --preset m2 默认（cf_lambda=0, novelty_gain=0）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\run_t2k.ps1 -Launch -Parallel          # 9 臂并行，后台
#   powershell -ExecutionPolicy Bypass -File .\tools\run_t2k.ps1 -GainList "0,1" -SeedList "101" -Episodes 100   # 筛查
#   powershell -ExecutionPolicy Bypass -File .\tools\run_t2k.ps1 -SummaryOnly               # 只汇总
param([switch]$Launch,
      [switch]$Parallel,
      [switch]$SummaryOnly,
      [string]$SeedList = '101,202,303',
      [string]$GainList = '0,1,5',
      [int]$Episodes = 320,
      [int]$EvalEpisodes = 20)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$exe = Join-Path $root 'benchmark2.exe'
$seeds = @($SeedList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { [int]$_.Trim() })
$gains = @($GainList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $_.Trim() })

function Get-ArmDir([string]$g) { Join-Path $root "data_b2\t2k\g$g" }

function Show-Summary {
    $rows = @()
    foreach ($g in $gains) {
        $dir = Get-ArmDir $g
        foreach ($s in $seeds) {
            $log = Join-Path $dir "log_g${g}_s$s.txt"
            if (-not (Test-Path $log)) { continue }
            $t = Get-Content $log -Raw
            $med = if ($t -match 'Median:\s*([0-9.]+)') { [double]$Matches[1] } else { [double]'nan' }
            $arr = if ($t -match 'Arrived:\s*(\d+)/(\d+)') { "$($Matches[1])/$($Matches[2])" } else { '-' }
            $rows += [pscustomobject]@{ gain = $g; seed = $s; median = $med; arrived = $arr }
        }
    }
    if ($rows.Count -eq 0) { Write-Host 'no logs yet'; return }
    $rows | Sort-Object gain, seed | Format-Table -AutoSize | Out-String | Write-Host
    $rows | Group-Object gain | Sort-Object Name | ForEach-Object {
        $m = ($_.Group | Where-Object { -not [double]::IsNaN($_.median) } | Measure-Object median -Average -Minimum)
        "g={0}  n={1}  median avg={2:N4} min={3:N4}" -f $_.Name, $m.Count, $m.Average, $m.Minimum | Write-Host
    }
}

if ($SummaryOnly) { Show-Summary; exit 0 }

foreach ($g in $gains) {
    $dir = Get-ArmDir $g
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($s in $seeds) {
        $out = Join-Path $dir "t2k_g${g}_s$s.csv"
        if (Test-Path $out) { continue }
        $a = @('--preset', 'm2', '--error-gain', "$g",
               '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
               '--seed', "$s", '--out', $out)
        $sp = @{ FilePath = $exe; ArgumentList = $a
                 RedirectStandardOutput = (Join-Path $dir "log_g${g}_s$s.txt")
                 RedirectStandardError  = (Join-Path $dir "err_g${g}_s$s.txt") }
        if ($Parallel) { $sp.WindowStyle = 'Hidden' } else { $sp.Wait = $true; $sp.NoNewWindow = $true }
        Start-Process @sp
        Write-Host ("launched g={0} s={1} ({2} ep)" -f $g, $s, $Episodes)
    }
}

if ($Launch) { Write-Host 'launched (detached)'; exit 0 }
if ($Parallel) {
    $total = $gains.Count * $seeds.Count
    while ($true) {
        $done = 0
        foreach ($g in $gains) { $done += (Get-ChildItem (Get-ArmDir $g) -Filter *.csv -ErrorAction SilentlyContinue | Measure-Object).Count }
        Write-Host ("{0}  {1}/{2} csv" -f (Get-Date -Format 'HH:mm:ss'), $done, $total)
        if ($done -ge $total) { break }
        Start-Sleep -Seconds 60
    }
}
Write-Host 'all done'
Show-Summary
