# B2-EI 侧抑制筛查批跑（2026-11-20 签字）
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_b2_ei.ps1 -EiN 64 -EiW 1.0 -OutDir data_b2\ei\e0
# 单变量：除 --ei-* 外，全部参数与官方 e-prop B 臂逐位相同（run_m4_eprop.ps1）。
param(
    [int]$EiMode = 1,
    [int]$EiN = 64,
    [double]$EiW = 1.0,
    [double]$EiTau = 100,
    [double]$EiSigma = 0.25,
    [string]$Seeds = '303,404,707,808',
    [int]$Episodes = 320,
    [int]$EvalEpisodes = 20,
    [string]$OutDir = 'data_b2/ei/e0'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $dir | Out-Null

foreach ($s in $Seeds.Split(',')) {
    $tag = "m$($EiMode)_n$($EiN)_w$($EiW.ToString('0.0'))_sg$($EiSigma.ToString('0.00'))_s$s"
    $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
           '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes", '--seed', "$s",
           '--eprop', '--eprop-scope', 'all', '--eprop-alpha', '0.3', '--eprop-lr', '1e-5',
           '--trace-stride', '20',
           '--ei-mode', "$EiMode", '--ei-n', "$EiN", '--ei-w', "$EiW", '--ei-tau', "$EiTau", '--ei-sigma', "$EiSigma",
           '--out', (Join-Path $dir "$tag.csv"))
    Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $dir "log_$tag.txt") `
        -RedirectStandardError  (Join-Path $dir "err_$tag.txt")
    Write-Host "launched $tag"
}

$n = 0
while ($n -lt 480) {
    $running = (Get-Process benchmark2 -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($running -eq 0) { break }
    Start-Sleep -Seconds 5
    $n++
}
Write-Host ("waited ~{0}s" -f ($n * 5))
Write-Host "== medR(eval) per seed =="
Get-ChildItem (Join-Path $dir '*.csv') | Where-Object { $_.Name -notlike '*.csv.meta.json' -and $_.Extension -eq '.csv' } | Sort-Object Name | ForEach-Object {
    $ev = @(Import-Csv $_.FullName | Where-Object { $_.is_eval -eq '1' } | ForEach-Object { [double]$_.dist_ratio } | Sort-Object)
    if ($ev.Count -gt 0) {
        $med = [math]::Round(($ev[[int][math]::Floor(($ev.Count-1)/2)] + $ev[[int][math]::Ceiling(($ev.Count-1)/2)])/2, 4)
        Write-Host ("{0}: medR(eval)={1}" -f $_.Name, $med)
    }
}
