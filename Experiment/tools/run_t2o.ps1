# T2o · 选优强度 K 的剂量响应：K∈{1,8} x g∈{0,1}
# 预注册：audit/t2o_prereg_2026-11-16.md
# K=3 的两格复用 data_b2/t2n/（T2n 结果），本脚本只跑 K=1 / K=8
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\run_t2o.ps1 -Launch -Parallel
#   powershell -ExecutionPolicy Bypass -File .\tools\run_t2o.ps1 -SummaryOnly
param([switch]$Launch,
      [switch]$Parallel,
      [switch]$SummaryOnly,
      [string]$SeedList = '101,202,303',
      [string]$KList = '1,8',
      [string]$GainList = '0,1',
      [int]$Episodes = 320,
      [int]$EvalEpisodes = 20)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'

$seeds = @($SeedList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { [int]$_.Trim() })
$ks    = @($KList.Split(',')    | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $_.Trim() })
$gains = @($GainList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $_.Trim() })

function Get-ArmDir([string]$k, [string]$g) { Join-Path $root "data_b2\t2o\k${k}_g$g" }

function Get-Med([double[]]$v) {
    $s = @($v | Sort-Object)
    if ($s.Count -eq 0) { return [double]::NaN }
    $mid = [int][math]::Floor($s.Count / 2)
    if ($s.Count % 2) { return $s[$mid] } else { return ($s[$mid-1] + $s[$mid]) / 2 }
}

function Read-Cell([string[]]$paths) {
    $D = New-Object 'System.Collections.Generic.List[double]'
    $arr = 0; $n = 0; $perSeed = @()
    foreach ($p in $paths) {
        if (-not (Test-Path $p)) { continue }
        $rows = @(Import-Csv $p | Where-Object { $_.is_eval -eq '1' })
        if ($rows.Count -eq 0) { continue }
        $d = @($rows | ForEach-Object { [double]$_.final_dist })
        $perSeed += (Get-Med $d)
        foreach ($x in $d) { $D.Add($x) }
        $arr += ($rows | Where-Object { $_.arrived -eq '1' } | Measure-Object).Count
        $n += $rows.Count
    }
    if ($D.Count -eq 0) { return $null }
    [pscustomobject]@{ medD = [math]::Round((Get-Med $D.ToArray()), 4); perSeed = ($perSeed | ForEach-Object { [math]::Round($_, 3) }) -join '/'; arr = "$arr/$n" }
}

function Show-Summary {
    # 统一口径由 tools/t2_arm_summary.py 负责（PS 的数组累加在此处不可靠）
    $armArgs = @(
        "--arm", "K1|g0=data_b2/t2o/k1_g0/t2o_k1_g0_s*.csv",
        "--arm", "K1|g1=data_b2/t2o/k1_g1/t2o_k1_g1_s*.csv",
        "--arm", "K3|g0=data_b2/t2n/wta_g0/t2n_wta_g0_s*.csv",
        "--arm", "K3|g1=data_b2/t2n/wta_g1/t2n_wta_g1_s*.csv",
        "--arm", "K8|g0=data_b2/t2o/k8_g0/t2o_k8_g0_s*.csv",
        "--arm", "K8|g1=data_b2/t2o/k8_g1/t2o_k8_g1_s*.csv"
    )
    & py -3 (Join-Path $root 'tools\t2_arm_summary.py') @armArgs
}

if ($SummaryOnly) { Show-Summary; exit 0 }

foreach ($k in $ks) {
    foreach ($g in $gains) {
        $dir = Get-ArmDir $k $g
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        foreach ($s in $seeds) {
            $out = Join-Path $dir "t2o_k${k}_g${g}_s$s.csv"
            if (Test-Path $out) { continue }
            $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', "$k", '--error-gain', "$g",
                   '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
                   '--seed', "$s", '--out', $out)
            $sp = @{ FilePath = $exe; ArgumentList = $a
                     RedirectStandardOutput = (Join-Path $dir "log_k${k}_g${g}_s$s.txt")
                     RedirectStandardError  = (Join-Path $dir "err_k${k}_g${g}_s$s.txt") }
            if ($Parallel) { $sp.WindowStyle = 'Hidden' } else { $sp.Wait = $true; $sp.NoNewWindow = $true }
            Start-Process @sp
            Write-Host ("launched K={0} g={1} s={2} ({3} ep)" -f $k, $g, $s, $Episodes)
        }
    }
}

if ($Launch) { Write-Host 'launched (detached)'; exit 0 }
if ($Parallel) {
    $total = $ks.Count * $gains.Count * $seeds.Count
    while ($true) {
        $done = 0
        foreach ($k in $ks) { foreach ($g in $gains) { $done += (Get-ChildItem (Get-ArmDir $k $g) -Filter *.csv -ErrorAction SilentlyContinue | Measure-Object).Count } }
        Write-Host ("{0}  {1}/{2} csv" -f (Get-Date -Format 'HH:mm:ss'), $done, $total)
        if ($done -ge $total) { break }
        Start-Sleep -Seconds 60
    }
}
Write-Host 'all done'
Show-Summary
