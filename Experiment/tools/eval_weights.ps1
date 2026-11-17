# 冻结权重的 held-out 评测：不做任何学习，只跑固定目标网格。
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\eval_weights.ps1 `
#       -Pattern 'data_b2\m4\ep_8s\w1_ep_lr1e05_s*.txt' -OutDir data_b2\m4\ho_eprop
#   powershell -ExecutionPolicy Bypass -File .\tools\eval_weights.ps1 `
#       -Pattern 'data_b2\v16_8s\w_final_s*.txt' -OutDir data_b2\m4\ho_base
# 要点：
#   - seed 必须与训练时一致（拓扑由 seed 生成，import 会校验条数）；
#   - --episodes == --eval-episodes ⇒ 全程 is_eval=1 ⇒ use_reward_modulation=0，权重纹丝不动；
#   - --target-mode cycle 用固定四目标 {1.5,0.5} {1.5,-0.5} {1.0,0.9} {1.0,-0.9}，
#     与训练时的随机目标**不重叠**，用于排除"eval 目标碰巧在训练目标附近"。
param(
    [string]$Pattern = 'data_b2/m4/ep_8s/w1_*.txt',
    [string]$OutDir = 'data_b2/m4/ho',
    [int]$Episodes = 40,
    [string]$TargetMode = 'cycle'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$files = Get-ChildItem $Pattern
if (-not $files) { throw "no weights match $Pattern" }

foreach ($f in $files) {
    if ($f.Name -notmatch '_s(\d+)\.txt$') { Write-Host ("skip (no seed in name): " + $f.Name); continue }
    $seed = $Matches[1]
    $label = ($f.BaseName -replace '[^0-9a-zA-Z_]', '_')
    # --no-speed-reward：关掉"碰到即换目标"，每个 episode 固定一个目标、跑满 1200 tick，
    # 这样 final_dist 就是对该固定目标的距离，四个目标各出现 Episodes/4 次。
    # 学习在 is_eval 下已冻结，speed 开关只影响结算方式，不影响动力学。
    $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
           '--seed', $seed,
           '--load-weights', $f.FullName,
           '--target-mode', $TargetMode, '--no-speed-reward',
           '--episodes', "$Episodes", '--eval-episodes', "$Episodes",
           '--out', (Join-Path $dir ("$label.csv")))
    Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $dir ("log_$label.txt")) `
        -RedirectStandardError  (Join-Path $dir ("err_$label.txt"))
    Write-Host ("launched $label (seed=$seed)")
}

$n = 0
while ($n -lt 120) {
    if ((Get-Process benchmark2 -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) { break }
    Start-Sleep -Seconds 5
    $n++
}

py (Join-Path $root 'tools\sum_eval.py') --dir $OutDir
