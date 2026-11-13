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
    foreach ($f in @('benchmark1.exe', 'benchmark2.exe', 'b2_core.dll', 'b2_core.o', 'b1_probe.dll', 'b1_sym.dll', 'b1_t1.dll', 'b1_psel.dll', 'b1_fwd.dll')) {
        $src = Join-Path $Root $f
        if (Test-Path $src) { Copy-Item $src (Join-Path $bak $f) -Force }
    }
    Write-Host "Backup: $bak"
}

$CFlags = @('-O2', '-Wall', '-Wextra')
$CxxFlags = @('-O2', '-Wall', '-Wextra')

Invoke-Step 'M0 seed_bundle (C)' 'gcc' ($CFlags + @('-I', (Join-Path $Root 'core'), '-c', (Join-Path $Root 'core\seed_bundle.c'), '-o', (Join-Path $ObjDir 'seed_bundle.o')))
Invoke-Step 'B1 core (C)' 'gcc' ($CFlags + @('-I', (Join-Path $Root 'core'), '-c', (Join-Path $Root 'brain\benchmark1_core.c'), '-o', (Join-Path $ObjDir 'b1_core.o')))
Invoke-Step 'B1 host (C++)' 'g++' ($CxxFlags + @('-I', (Join-Path $Root 'core'), '-c', (Join-Path $Root 'host\benchmark1_host.cpp'), '-o', (Join-Path $ObjDir 'b1_host.o')))
Invoke-Step 'B1 link' 'g++' @((Join-Path $ObjDir 'b1_core.o'), (Join-Path $ObjDir 'b1_host.o'), (Join-Path $ObjDir 'seed_bundle.o'), '-o', (Join-Path $Root 'benchmark1.exe'), '-lm')

# T2 只读诊断适配层：只链接已编译的 b1_core.o / seed_bundle.o，
# 不重新编译核心源码，保证 benchmark1.exe 与诊断 DLL 使用完全相同的目标代码。
Invoke-Step 'B1 probe DLL (T2 read-only)' 'gcc' ($CFlags + @(
    '-I', (Join-Path $Root 'core'), '-I', (Join-Path $Root 'brain'),
    '-shared', (Join-Path $Root 'brain\benchmark1_probe.c'),
    (Join-Path $ObjDir 'b1_core.o'), (Join-Path $ObjDir 'seed_bundle.o'),
    '-o', (Join-Path $Root 'b1_probe.dll'), '-lm'))

# T1/T2 约束①：严格对称初始读出实验台。同样只链接既有目标文件，
# 核心源码零改动；独立 DLL 可单独删除，不影响 benchmark1.exe。
Invoke-Step 'B1 sym DLL (T1 symmetric readout)' 'gcc' ($CFlags + @(
    '-I', (Join-Path $Root 'core'), '-I', (Join-Path $Root 'brain'),
    '-shared', (Join-Path $Root 'brain\benchmark1_sym.c'),
    (Join-Path $ObjDir 'b1_core.o'), (Join-Path $ObjDir 'seed_bundle.o'),
    '-o', (Join-Path $Root 'b1_sym.dll'), '-lm'))

# T1 两必要条件析因实验台（判据 v1.4 §6.4 下一步）。同样只链接既有目标文件，
# 核心源码零改动；独立 DLL 可单独删除，不影响 benchmark1.exe 与前述证据链。
Invoke-Step 'B1 t1 DLL (T1 two-conditions factorial)' 'gcc' ($CFlags + @(
    '-I', (Join-Path $Root 'core'), '-I', (Join-Path $Root 'brain'),
    '-shared', (Join-Path $Root 'brain\benchmark1_t1.c'),
    (Join-Path $ObjDir 'b1_core.o'), (Join-Path $ObjDir 'seed_bundle.o'),
    '-o', (Join-Path $Root 'b1_t1.dll'), '-lm'))

# T1 通路选择性第三因子实验台（判据 v1.5 §6.5 / 附录 G.4 下一步）。
# 同样只链接既有目标文件，核心源码零改动；独立 DLL 可单独删除。
Invoke-Step 'B1 psel DLL (T1 pathway-selective third factor)' 'gcc' ($CFlags + @(
    '-I', (Join-Path $Root 'core'), '-I', (Join-Path $Root 'brain'),
    '-shared', (Join-Path $Root 'brain\benchmark1_psel.c'),
    (Join-Path $ObjDir 'b1_core.o'), (Join-Path $ObjDir 'seed_bundle.o'),
    '-o', (Join-Path $Root 'b1_psel.dll'), '-lm'))

# T1 可学习内部前向模型实验台（判据 v1.6 §6.6 / 附录 H.4 下一步）。
# 同样只链接既有目标文件，核心源码零改动；独立 DLL 可单独删除。
Invoke-Step 'B1 fwd DLL (T1 learnable internal forward model)' 'gcc' ($CFlags + @(
    '-I', (Join-Path $Root 'core'), '-I', (Join-Path $Root 'brain'),
    '-shared', (Join-Path $Root 'brain\benchmark1_fwd.c'),
    (Join-Path $ObjDir 'b1_core.o'), (Join-Path $ObjDir 'seed_bundle.o'),
    '-o', (Join-Path $Root 'b1_fwd.dll'), '-lm'))

Invoke-Step 'B2 core (C)' 'gcc' ($CFlags + @('-c', (Join-Path $Root 'brain_b2\benchmark2_core.c'), '-o', (Join-Path $ObjDir 'b2_core.o')))
Invoke-Step 'B2 host (C++)' 'g++' ($CxxFlags + @('-c', (Join-Path $Root 'host_b2\benchmark2_host.cpp'), '-o', (Join-Path $ObjDir 'b2_host.o')))
Invoke-Step 'B2 link' 'g++' @((Join-Path $ObjDir 'b2_core.o'), (Join-Path $ObjDir 'b2_host.o'), '-o', (Join-Path $Root 'benchmark2.exe'), '-lm')
Invoke-Step 'B2 shared lib (ctypes)' 'gcc' ($CFlags + @('-shared', (Join-Path $Root 'brain_b2\benchmark2_core.c'), '-o', (Join-Path $Root 'b2_core.dll'), '-lm'))

# Keep the legacy intermediate object and the audit probe DLL in sync.
Copy-Item (Join-Path $ObjDir 'b2_core.o') (Join-Path $Root 'b2_core.o') -Force
Copy-Item (Join-Path $Root 'b2_core.dll') (Join-Path $ProbeDir 'b2_core_audit.dll') -Force

$sourceFiles = @(
    'brain\benchmark1_core.c', 'brain\benchmark1_core.h', 'host\benchmark1_host.cpp',
    'brain\benchmark1_probe.c', 'brain\benchmark1_sym.c', 'brain\benchmark1_t1.c',
    'brain\benchmark1_psel.c', 'brain\benchmark1_fwd.c',
    'core\seed_bundle.c', 'core\seed_bundle.h',
    'brain_b2\benchmark2_core.c', 'brain_b2\benchmark2_core.h', 'host_b2\benchmark2_host.cpp'
)
$sourceHashes = @{}
foreach ($rel in $sourceFiles) {
    $sourceHashes[$rel] = (Get-FileHash (Join-Path $Root $rel) -Algorithm SHA256).Hash.ToLower()
}
$artifactHashes = @{}
foreach ($name in @('benchmark1.exe', 'benchmark2.exe', 'b2_core.dll', 'b2_core.o', 'b1_probe.dll', 'b1_sym.dll', 'b1_t1.dll', 'b1_psel.dll', 'b1_fwd.dll')) {
    $artifactHashes[$name] = (Get-FileHash (Join-Path $Root $name) -Algorithm SHA256).Hash.ToLower()
}
$buildInfo = [ordered]@{
    built_at = (Get-Date).ToString('s')
    compiler = ((& gcc --version | Select-Object -First 1) -join '')
    macros = 'defaults from headers: ELIGIBILITY_TRACE_ENABLE=1, REWARD_MODULATION_ENABLE=1'
    commands = @(
        'gcc -O2 -Wall -Wextra -I core -c core/seed_bundle.c -o <obj>/seed_bundle.o',
        'gcc -O2 -Wall -Wextra -I core -c <core>.c -o <obj>.o',
        'g++ -O2 -Wall -Wextra -I core -c <host>.cpp -o <obj>.o',
        'g++ <core>.o <host>.o <obj>/seed_bundle.o -o <exe> -lm',
        'gcc -O2 -Wall -Wextra -I core -I brain -shared brain/benchmark1_probe.c <obj>/b1_core.o <obj>/seed_bundle.o -o b1_probe.dll -lm',
        'gcc -O2 -Wall -Wextra -I core -I brain -shared brain/benchmark1_sym.c <obj>/b1_core.o <obj>/seed_bundle.o -o b1_sym.dll -lm',
        'gcc -O2 -Wall -Wextra -I core -I brain -shared brain/benchmark1_t1.c <obj>/b1_core.o <obj>/seed_bundle.o -o b1_t1.dll -lm',
        'gcc -O2 -Wall -Wextra -I core -I brain -shared brain/benchmark1_psel.c <obj>/b1_core.o <obj>/seed_bundle.o -o b1_psel.dll -lm',
        'gcc -O2 -Wall -Wextra -I core -I brain -shared brain/benchmark1_fwd.c <obj>/b1_core.o <obj>/seed_bundle.o -o b1_fwd.dll -lm',
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
Get-ChildItem (Join-Path $Root 'benchmark1.exe'), (Join-Path $Root 'benchmark2.exe'), (Join-Path $Root 'b2_core.dll'), (Join-Path $Root 'b1_probe.dll'), (Join-Path $Root 'b1_sym.dll'), (Join-Path $Root 'b1_t1.dll'), (Join-Path $Root 'b1_psel.dll'), (Join-Path $Root 'b1_fwd.dll') |
    Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
