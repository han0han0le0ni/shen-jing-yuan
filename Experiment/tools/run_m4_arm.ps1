# M4 通用单臂批跑：与 run_m4_eprop.ps1 的 e-prop 臂逐位同参，只额外允许透传任意 flag。
# 用途：探索臂（不改学习规则的那些）——例如 --wta-k 2、--hetero-thresh 4.0、--struct 1 ...
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\run_m4_arm.ps1 `
#       -OutDir data_b2\m4\armL -Tag armL -Seeds 101,202,404,707,808,1010 -ExtraArgs '--wta-k 2'
#   powershell -ExecutionPolicy Bypass -File .\tools\run_m4_arm.ps1 `
#       -OutDir data_b2\m4\armH2 -Tag armH2 -Seeds 101,202 -ExtraArgs '--hetero-thresh 4.0'
# 说明：
#   - 基础命令行与 run_m4_eprop.ps1 的 eprop 分支**完全一致**（同一个 preset/解码/预算/lr），
#     所以新臂与官方 B 臂只差 -ExtraArgs 里那几个 flag：单变量成立。
#   - -ExtraArgs 用空格切分，参数值里不要带空格。
#   - 并发上限 -Concurrency（默认 3）；-NoWait 只启动不等待（便于分批排期）。
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Seeds = '101',
    [string]$ExtraArgs = '',
    [string]$Tag = 'arm',
    [int]$Concurrency = 3,
    [int]$Episodes = 320,
    [int]$EvalEpisodes = 20,
    [int]$TraceStride = 20,
    [string]$EpropLr = '0.00001',
    [string]$EpropScope = 'all',
    [string]$Alpha = '0.3',
    [switch]$NoWait
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
if (-not (Test-Path $exe)) { throw "missing $exe" }
$dir = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$inv = [Globalization.CultureInfo]::InvariantCulture
$seedList = @($Seeds.Split(',') | Where-Object { $_.Trim() -ne '' } |
    ForEach-Object { [int]::Parse($_.Trim(), $inv) })
$extra = @($ExtraArgs.Split(' ') | Where-Object { $_ -ne '' })

$running = @()
foreach ($s in $seedList) {
    $seedTag = "${Tag}_s$s"
    $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
           '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes", '--seed', "$s",
           '--eprop', '--eprop-scope', $EpropScope, '--eprop-alpha', $Alpha, '--eprop-lr', $EpropLr) +
         $extra + @(
           '--save-init-weights', (Join-Path $dir ("w0_s{0}.txt" -f $s)),
           '--save-weights',      (Join-Path $dir ("w1_$seedTag.txt")),
           '--trace-cmd',         (Join-Path $dir ("cmd_$seedTag.jsonl")),
           '--trace-stride',      "$TraceStride",
           '--out', (Join-Path $dir ("$seedTag.csv")))
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Concurrency) {
        Start-Sleep -Seconds 5
        $running = @($running | Where-Object { -not $_.HasExited })
    }
    $p = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $dir ("log_$seedTag.txt")) `
        -RedirectStandardError  (Join-Path $dir ("err_$seedTag.txt"))
    $running += $p
    Write-Host ("launched $seedTag  extra=[{0}]" -f ($extra -join ' '))
}

if (-not $NoWait) {
    while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) { Start-Sleep -Seconds 10 }
    Write-Host "== done: $OutDir =="
    Get-ChildItem (Join-Path $dir '*.csv') | Select-Object Name, Length | Format-Table -AutoSize
}
