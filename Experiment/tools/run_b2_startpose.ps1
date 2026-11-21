# B2-startpose 筛查批跑（2026-11-21 签字「a」）
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_b2_startpose.ps1 -StartJitter 0.3 -OutDir data_b2/startpose/p1
# 单变量：除 --start-jitter 外全部与官方 e-prop B 臂逐位相同。
param(
    [double]$StartJitter = 0.3,
    [string]$Seeds = '303,404,707,808',
    [int]$Episodes = 320,
    [int]$EvalEpisodes = 20,
    [string]$OutDir = 'data_b2/startpose/run'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $dir | Out-Null

foreach ($s in $Seeds.Split(',')) {
    $tag = "j$($StartJitter.ToString('0.00'))_s$s"
    $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
           '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes", '--seed', "$s",
           '--eprop', '--eprop-scope', 'all', '--eprop-alpha', '0.3', '--eprop-lr', '1e-5',
           '--trace-stride', '20',
           '--start-jitter', "$StartJitter",
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
Get-ChildItem (Join-Path $dir '*.csv') | Where-Object { $_.Extension -eq '.csv' } | Sort-Object Name | ForEach-Object {
    $rows = @(Import-Csv $_.FullName | Where-Object { $_.is_eval -eq '1' })
    $ev = @($rows | ForEach-Object { [double]$_.dist_ratio } | Sort-Object)
    if ($ev.Count -gt 0) {
        $med = [math]::Round(($ev[[int][math]::Floor(($ev.Count-1)/2)] + $ev[[int][math]::Ceiling(($ev.Count-1)/2)])/2, 4)
        $arr = @($rows | Where-Object { $_.arrived -eq '1' }).Count
        $near = @($rows | Where-Object { [math]::Sqrt([double]$_.target_x*[double]$_.target_x + [double]$_.target_y*[double]$_.target_y) -lt 0.5 })
        $nearArr = @($near | Where-Object { $_.arrived -eq '1' }).Count
        Write-Host ("{0}: medR={1} arrived={2}/{3} near(|t|<0.5)={4}/{5}" -f $_.Name, $med, $arr, $ev.Count, $nearArr, $near.Count)
    }
}
