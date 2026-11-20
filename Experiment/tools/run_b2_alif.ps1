# B2-ALIF A 路径 Gate 0/1 批跑（2026-11-19 签字）
# 用法：powershell -ExecutionPolicy Bypass -File .\tools\run_b2_alif.ps1 -Beta 4 -Seeds 303,404,707,808 -OutDir data_b2\alif\a_b4_screen
# 单变量：除 --alif-beta 外，全部参数与官方 e-prop B 臂逐位相同（run_m4_eprop.ps1）。
param(
    [double]$Beta = 4,
    [double]$TauA = 100,
    [int]$BPath = 0,
    [string]$Seeds = '303,404,707,808',
    [int]$Episodes = 320,
    [int]$EvalEpisodes = 20,
    [string]$OutDir = 'data_b2/alif/a_b4_screen'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$jobs = @()
foreach ($s in $Seeds.Split(',')) {
    $tag = "bp$($BPath)_b$($Beta.ToString('0'))_s$s"
    $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
           '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes", '--seed', "$s",
           '--eprop', '--eprop-scope', 'all', '--eprop-alpha', '0.3', '--eprop-lr', '1e-5',
           '--trace-stride', '20',
           '--alif-beta', "$Beta", '--alif-tau-a', "$TauA", '--alif-bpath', "$BPath",
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

Write-Host "== gate0 + medR per seed =="
Get-ChildItem (Join-Path $dir '*.gate0.json') | Sort-Object Name | ForEach-Object {
    $g = Get-Content $_.FullName -Raw | ConvertFrom-Json
    $csv = Join-Path $dir ($g.gate0_csv -replace '.*[\\/]', '')
    $med = if (Test-Path $csv) {
        $ev = @(Import-Csv $csv | Where-Object { $_.is_eval -eq '1' } | ForEach-Object { [double]$_.dist_ratio } | Sort-Object)
        if ($ev.Count -gt 0) { [math]::Round(($ev[[int][math]::Floor(($ev.Count-1)/2)] + $ev[[int][math]::Ceiling(($ev.Count-1)/2)])/2, 4) } else { 'n/a' }
    } else { 'n/a' }
    Write-Host ("{0}: beta={1} spikes/tick={2:N2} medR(eval)={3}" -f $_.Name, $g.alif_beta, $g.spikes_per_tick, $med)
}
