# M0 build + acceptance test. Does not touch benchmark binaries.
# Usage: powershell -ExecutionPolicy Bypass -File build_m0.ps1
$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$Out  = Join-Path $env:TEMP 'neuro_m0'
New-Item -ItemType Directory -Force -Path $Out | Out-Null

foreach ($tool in @('gcc')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing compiler: $tool"
    }
}

$exe = Join-Path $Out 'test_m0.exe'
Write-Host '==> build test_m0' -ForegroundColor Cyan
& gcc -O2 -Wall -Wextra -std=c11 -I (Join-Path $Root 'core') `
    (Join-Path $Root 'core\test_m0.c') `
    (Join-Path $Root 'core\neuro_id.c') `
    (Join-Path $Root 'core\seed_bundle.c') `
    -o $exe -lm
if ($LASTEXITCODE -ne 0) { throw "build failed (exit=$LASTEXITCODE)" }

Write-Host '==> run test_m0' -ForegroundColor Cyan
& $exe
if ($LASTEXITCODE -ne 0) { throw "M0 test failed (exit=$LASTEXITCODE)" }

Write-Host '==> M0 OK' -ForegroundColor Green
