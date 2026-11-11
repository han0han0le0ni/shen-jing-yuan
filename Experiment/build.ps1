# Unified build entry for Benchmark-1 / Benchmark-2 / visualization DLL.
# Usage:
#   powershell -ExecutionPolicy Bypass -File build.ps1
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Backup -Smoke
# Requirements: MinGW-W64 gcc/g++ on PATH.

param(
    [switch]$Backup,
    [switch]$Smoke
)

$ErrorActionPreference = 'Continue'
$Root = $PSScriptRoot
$ObjDir = Join-Path $env:TEMP 'snn_audit\obj'
$ProbeDir = Join-Path $env:TEMP 'snn_audit'
New-Item -ItemType Directory -Force -Path $ObjDir | Out-Null

foreach ($tool in @('gcc', 'g++')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing compiler: $tool (install MinGW-W64 and add it to PATH)"
    }
}

function Invoke-Step {
    param([string]$Name, [string]$Tool, [string[]]$ToolArgs)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Tool @ToolArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit=$LASTEXITCODE)" }
}

if ($Backup) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $bak = Join-Path $Root "backup_$stamp"
    New-Item -ItemType Directory -Force -Path $bak | Out-Null
    foreach ($f in @('benchmark1.exe', 'benchmark2.exe', 'b2_core.dll', 'b2_core.o')) {
        $src = Join-Path $Root $f
        if (Test-Path $src) { Copy-Item $src (Join-Path $bak $f) -Force }
    }
    Write-Host "Backup: $bak"
}

$CFlags = @('-O2', '-Wall', '-Wextra')
$CxxFlags = @('-O2', '-Wall', '-Wextra')

Invoke-Step 'B1 core (C)' 'gcc' ($CFlags + @('-c', (Join-Path $Root 'brain\benchmark1_core.c'), '-o', (Join-Path $ObjDir 'b1_core.o')))
Invoke-Step 'B1 host (C++)' 'g++' ($CxxFlags + @('-c', (Join-Path $Root 'host\benchmark1_host.cpp'), '-o', (Join-Path $ObjDir 'b1_host.o')))
Invoke-Step 'B1 link' 'g++' @((Join-Path $ObjDir 'b1_core.o'), (Join-Path $ObjDir 'b1_host.o'), '-o', (Join-Path $Root 'benchmark1.exe'), '-lm')

Invoke-Step 'B2 core (C)' 'gcc' ($CFlags + @('-c', (Join-Path $Root 'brain_b2\benchmark2_core.c'), '-o', (Join-Path $ObjDir 'b2_core.o')))
Invoke-Step 'B2 host (C++)' 'g++' ($CxxFlags + @('-c', (Join-Path $Root 'host_b2\benchmark2_host.cpp'), '-o', (Join-Path $ObjDir 'b2_host.o')))
Invoke-Step 'B2 link' 'g++' @((Join-Path $ObjDir 'b2_core.o'), (Join-Path $ObjDir 'b2_host.o'), '-o', (Join-Path $Root 'benchmark2.exe'), '-lm')
Invoke-Step 'B2 shared lib (ctypes)' 'gcc' ($CFlags + @('-shared', (Join-Path $Root 'brain_b2\benchmark2_core.c'), '-o', (Join-Path $Root 'b2_core.dll'), '-lm'))

# Keep the legacy intermediate object and the audit probe DLL in sync.
Copy-Item (Join-Path $ObjDir 'b2_core.o') (Join-Path $Root 'b2_core.o') -Force
Copy-Item (Join-Path $Root 'b2_core.dll') (Join-Path $ProbeDir 'b2_core_audit.dll') -Force

$sourceFiles = @(
    'brain\benchmark1_core.c', 'brain\benchmark1_core.h', 'host\benchmark1_host.cpp',
    'brain_b2\benchmark2_core.c', 'brain_b2\benchmark2_core.h', 'host_b2\benchmark2_host.cpp'
)
$sourceHashes = @{}
foreach ($rel in $sourceFiles) {
    $sourceHashes[$rel] = (Get-FileHash (Join-Path $Root $rel) -Algorithm SHA256).Hash.ToLower()
}
$artifactHashes = @{}
foreach ($name in @('benchmark1.exe', 'benchmark2.exe', 'b2_core.dll', 'b2_core.o')) {
    $artifactHashes[$name] = (Get-FileHash (Join-Path $Root $name) -Algorithm SHA256).Hash.ToLower()
}
$buildInfo = [ordered]@{
    built_at = (Get-Date).ToString('s')
    compiler = ((& gcc --version | Select-Object -First 1) -join '')
    macros = 'defaults from headers: ELIGIBILITY_TRACE_ENABLE=1, REWARD_MODULATION_ENABLE=1'
    commands = @(
        'gcc -O2 -Wall -Wextra -c <core>.c -o <obj>.o',
        'g++ -O2 -Wall -Wextra -c <host>.cpp -o <obj>.o',
        'g++ <core>.o <host>.o -o <exe> -lm',
        'gcc -O2 -Wall -Wextra -shared brain_b2/benchmark2_core.c -o b2_core.dll -lm'
    )
    sources = $sourceHashes
    artifacts = $artifactHashes
}
$json = $buildInfo | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText((Join-Path $Root 'build_info.json'), $json,
    (New-Object System.Text.UTF8Encoding($false)))

if ($Smoke) {
    Push-Location $Root
    try {
        Write-Host '==> B2 smoke (3 episodes x 50 ticks)' -ForegroundColor Cyan
        $mini = Join-Path $ProbeDir 'smoke2.csv'
        & (Join-Path $Root 'benchmark2.exe') --episodes 3 --ticks 50 --out $mini | Select-Object -Last 8
        Write-Host "B2 smoke exit=$LASTEXITCODE (exit 2 = short run not passing, expected)"
        Write-Host '==> B1 --help' -ForegroundColor Cyan
        & (Join-Path $Root 'benchmark1.exe') --help | Select-Object -First 4
    }
    finally { Pop-Location }
}

Write-Host "`nBuild OK. Artifacts:" -ForegroundColor Green
Get-ChildItem (Join-Path $Root 'benchmark1.exe'), (Join-Path $Root 'benchmark2.exe'), (Join-Path $Root 'b2_core.dll') |
    Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
