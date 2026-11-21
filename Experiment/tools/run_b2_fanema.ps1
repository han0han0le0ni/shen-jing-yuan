# B2-fan/EMA 筛查批跑（2026-11-21 签字）
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_b2_fanema.ps1 -DecodeEma 50 -OutDir data_b2/fanema/m2
#       powershell -ExecutionPolicy Bypass -File .\tools\run_b2_fanema.ps1 -IhFan 60 -OutDir data_b2/fanema/f2
# 单变量：除 --ih-fan / --decode-ema 外，全部参数与官方 e-prop B 臂逐位相同（run_b2_ei.ps1）。
param(
    [int]$IhFan = 20,
    [int]$DecodeEma = 0,
    [string]$Seeds = '303,404,707,808',
    [int]$Episodes = 320,
    [int]$EvalEpisodes = 20,
    [switch]$Trace,
    [string]$OutDir = 'data_b2/fanema/run'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $dir | Out-Null

foreach ($s in $Seeds.Split(',')) {
    $tag = "f$($IhFan)_e$($DecodeEma)_s$s"
    $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
           '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes", '--seed', "$s",
           '--eprop', '--eprop-scope', 'all', '--eprop-alpha', '0.3', '--eprop-lr', '1e-5',
           '--trace-stride', '20',
           '--ih-fan', "$IhFan", '--decode-ema', "$DecodeEma")
    if ($Trace) { $a += @('--trace-cmd', (Join-Path $dir "cmd_$tag.jsonl"), '--trace-stride', '1') }
    $a += @('--out', (Join-Path $dir "$tag.csv"))
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
Get-ChildItem (Join-Path $dir '*.csv') | Where-Object { $_.Extension -eq '.csv' } | Sort-Object Name | ForEach-Object {
    $ev = @(Import-Csv $_.FullName | Where-Object { $_.is_eval -eq '1' } | ForEach-Object { [double]$_.dist_ratio } | Sort-Object)
    if ($ev.Count -gt 0) {
        $med = [math]::Round(($ev[[int][math]::Floor(($ev.Count-1)/2)] + $ev[[int][math]::Ceiling(($ev.Count-1)/2)])/2, 4)
        $arr = @(Import-Csv $_.FullName | Where-Object { $_.is_eval -eq '1' -and $_.arrived -eq '1' }).Count
        Write-Host ("{0}: medR(eval)={1} arrived={2}/{3}" -f $_.Name, $med, $arr, $ev.Count)
    }
}
