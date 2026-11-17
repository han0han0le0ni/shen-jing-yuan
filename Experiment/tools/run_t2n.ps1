# T2n · 误差项 × 解码方式（PDc vs k-WTA）2×2
# 预注册：audit/t2n_prereg_2026-11-16.md
# 新跑两臂：C = WTA+g0 / D = WTA+g1；A/B（PDc g0/g1）复用 data_b2/t2k/g{0,1}/
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\run_t2n.ps1 -Launch -Parallel
#   powershell -ExecutionPolicy Bypass -File .\tools\run_t2n.ps1 -SummaryOnly
param([switch]$Launch,
      [switch]$Parallel,
      [switch]$SummaryOnly,
      [string]$SeedList = '101,202,303',
      [string]$GainList = '0,1',
      [int]$Episodes = 320,
      [int]$EvalEpisodes = 20,
      [int]$WtaK = 3)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$exe = Join-Path $root 'benchmark2.exe'
$seeds = @($SeedList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { [int]$_.Trim() })
$gains = @($GainList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $_.Trim() })

function Get-ArmDir([string]$g) { Join-Path $root "data_b2\t2n\wta_g$g" }

function Read-Arm([string]$csv, [string]$tag) {
    if (-not (Test-Path $csv)) { return $null }
    $rows = Import-Csv $csv | Where-Object { $_.is_eval -eq '1' }
    if (-not $rows) { return $null }
    $d = @($rows | ForEach-Object { [double]$_.final_dist } | Sort-Object)
    $r = @($rows | ForEach-Object { [double]$_.dist_ratio } | Sort-Object)
    $arr = ($rows | Where-Object { $_.arrived -eq '1' } | Measure-Object).Count
    $mid = [int][math]::Floor($d.Count / 2)
    $medD = if ($d.Count % 2) { $d[$mid] } else { ($d[$mid-1] + $d[$mid]) / 2 }
    $medR = if ($r.Count % 2) { $r[$mid] } else { ($r[$mid-1] + $r[$mid]) / 2 }
    [pscustomobject]@{ arm = $tag; n = $d.Count; medD = $medD; medR = $medR; arr = $arr }
}

function Show-Summary {
    # 统一口径由 tools/t2_arm_summary.py 负责（PS 里对「每 seed 中位数再取中位数」会与合并口径不一致）
    $armArgs = @(
        "--arm", "A|pdc_g0=data_b2/t2k/g0/t2k_g0_s*.csv",
        "--arm", "B|pdc_g1=data_b2/t2k/g1/t2k_g1_s*.csv",
        "--arm", "C|wta_g0=data_b2/t2n/wta_g0/t2n_wta_g0_s*.csv",
        "--arm", "D|wta_g1=data_b2/t2n/wta_g1/t2n_wta_g1_s*.csv"
    )
    & py -3 (Join-Path $root 'tools\t2_arm_summary.py') @armArgs
}

if ($SummaryOnly) { Show-Summary; exit 0 }

foreach ($g in $gains) {
    $dir = Get-ArmDir $g
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($s in $seeds) {
        $out = Join-Path $dir "t2n_wta_g${g}_s$s.csv"
        if (Test-Path $out) { continue }
        $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', "$WtaK", '--error-gain', "$g",
               '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
               '--seed', "$s", '--out', $out)
        $sp = @{ FilePath = $exe; ArgumentList = $a
                 RedirectStandardOutput = (Join-Path $dir "log_wta_g${g}_s$s.txt")
                 RedirectStandardError  = (Join-Path $dir "err_wta_g${g}_s$s.txt") }
        if ($Parallel) { $sp.WindowStyle = 'Hidden' } else { $sp.Wait = $true; $sp.NoNewWindow = $true }
        Start-Process @sp
        Write-Host ("launched wta g={0} s={1} ({2} ep)" -f $g, $s, $Episodes)
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
