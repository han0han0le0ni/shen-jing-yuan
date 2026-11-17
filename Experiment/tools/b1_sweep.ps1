# B1 delay sweep (M1). Each (delay, condition) runs as an independent process,
# which is also an independent brain instance.
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\b1_sweep.ps1
#   powershell ... -Delays 10,50,100,200,500,1000 -Trials 300
param(
    [int[]]$Delays = @(10, 50, 100, 200, 500, 1000),
    [int]$Trials = 300,
    [string]$OutDir = 'data/b1_m1'
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$conditions = @(
    @{ name = 'k3';      extra = @() },
    @{ name = 'k1_200';  extra = @('--trace-k', '1', '--tau-traces', '200,200,200') },
    @{ name = 'notrace'; extra = @('--disable_trace') }
)

foreach ($d in $Delays) {
    foreach ($c in $conditions) {
        $tag = "sweep_$($c.name)_D$d"
        Write-Host "== $tag" -ForegroundColor Cyan
        & .\benchmark1.exe --trials $Trials --delay $d --learning-rate 0.02 @($c.extra) `
            --jsonl (Join-Path $OutDir "$tag.jsonl") `
            --tag $tag `
            --out (Join-Path $OutDir "$tag.csv") | Select-String 'Test Accuracy'
    }
}

Write-Host 'Sweep done. Report: py -3 tools/b1_report.py' -ForegroundColor Green
