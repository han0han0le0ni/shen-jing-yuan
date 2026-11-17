# M4 e-prop 批跑：按 lr × seed 网格跑臂，导出权重与命令轨迹。
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\run_m4_eprop.ps1 `
#       -Lrs 0.000003,0.00001,0.00003 -OutDir data_b2\m4\lrcal
#   powershell -ExecutionPolicy Bypass -File .\tools\run_m4_eprop.ps1 `
#       -Seeds 101,202 -Lrs 0.00001 -Scope im -OutDir data_b2\m4\armD
# 说明：
#   - 逐 tick ADAM ⇒ 更新步数 N = episodes × ticks(≈1200)，总漂移 ~ lr·N，
#     所以 lr 必须与 episodes 一起理解；改 episodes 必须重标 lr。
#   - e-prop 臂按预注册关闭 error_gain 通道；-Base 打开 --error-gain 1 作为 R-STDP 对照。
#   - 同一 seed 的 --eprop-lr 之外的一切参数固定：单变量。
param(
    # 逗号分隔字符串（同 Lrs：-File 传参受区域设置影响，统一内部解析）
    [string]$Seeds = '202',
    # 逗号分隔字符串（避免 -File 传参时的区域设置把 3E-06 解析失败）；内部按不变文化转 double
    [string]$Lrs = '0.000003,0.00001,0.00003',
    [int]$Episodes = 80,
    [int]$EvalEpisodes = 20,
    [int]$TraceStride = 20,
    [string]$Scope = 'all',
    [double]$Alpha = 0.3,
    [string]$OutDir = 'data_b2/m4/lrcal',
    [string]$ErrorGain = '1.0',
    # 工作点覆盖（预注册臂 C1 = psp 1.5 / hidden-base 0）；留空 = 用 m2 默认
    [string]$Psp = '',
    [string]$HiddenBase = '',
    [switch]$Base,
    [switch]$SummaryOnly
)

$extra = @()
if ($Psp)        { $extra += @('--psp', $Psp) }
if ($HiddenBase) { $extra += @('--hidden-base', $HiddenBase) }

$inv = [Globalization.CultureInfo]::InvariantCulture
$lrList = @($Lrs.Split(',') | Where-Object { $_.Trim() -ne '' } |
    ForEach-Object { [double]::Parse($_.Trim(), $inv) })
$seedList = @($Seeds.Split(',') | Where-Object { $_.Trim() -ne '' } |
    ForEach-Object { [int]::Parse($_.Trim(), $inv) })

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root 'benchmark2.exe'
$dir = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $dir | Out-Null

if (-not $SummaryOnly) {
    $jobs = @()
    foreach ($s in $seedList) {
        if ($Base) {
            $tag = ("base_eg{0}_s{1}" -f $ErrorGain, $s) -replace '[^0-9a-zA-Z_]', ''
            $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
                   '--error-gain', $ErrorGain, '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
                   '--seed', "$s") + $extra + @(
                   '--save-init-weights', (Join-Path $dir ("w0_$tag.txt")),
                   '--save-weights',      (Join-Path $dir ("w1_$tag.txt")),
                   '--trace-cmd',         (Join-Path $dir ("cmd_s{0}.jsonl" -f $s)),
                   '--trace-stride', "$TraceStride",
                   '--out', (Join-Path $dir ("$tag.csv")))
            Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $dir ("log_$tag.txt")) `
                -RedirectStandardError  (Join-Path $dir ("err_$tag.txt"))
            Write-Host ("launched $tag")
            continue
        }
        foreach ($lr in $lrList) {
            $lrTag = ('lr{0:g}' -f $lr) -replace '[^0-9a-zA-Z]', ''
            $tag = "ep_${lrTag}_s$s"
            $a = @('--preset', 'm2', '--decode-mode', 'wta', '--wta-k', '1', '--reward-mode', 'potential',
                   '--episodes', "$Episodes", '--eval-episodes', "$EvalEpisodes",
                   '--seed', "$s") + $extra + @(
                   '--eprop', '--eprop-scope', $Scope, '--eprop-alpha', "$Alpha", '--eprop-lr', "$lr",
                   '--save-init-weights', (Join-Path $dir ("w0_s{0}.txt" -f $s)),
                   '--save-weights',      (Join-Path $dir ("w1_$tag.txt")),
                   '--trace-cmd',         (Join-Path $dir ("cmd_${tag}.jsonl")),
                   '--trace-stride', "$TraceStride",
                   '--out', (Join-Path $dir ("$tag.csv")))
            Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $dir ("log_$tag.txt")) `
                -RedirectStandardError  (Join-Path $dir ("err_$tag.txt"))
            Write-Host ("launched $tag")
        }
    }

    $n = 0
    while ($n -lt 240) {
        $running = (Get-Process benchmark2 -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($running -eq 0) { break }
        Start-Sleep -Seconds 5
        $n++
    }
    Write-Host ("waited ~{0}s" -f ($n * 5))
}

Write-Host "== weights =="
Get-ChildItem (Join-Path $dir 'w1_*.txt') -ErrorAction SilentlyContinue | ForEach-Object {
    $ref = Join-Path $dir 'w0_s202.txt'
    if (-not (Test-Path $ref)) { $ref = (Get-ChildItem (Join-Path $dir 'w0_*.txt') -ErrorAction SilentlyContinue | Select-Object -First 1).FullName }
    if ($ref -and (Test-Path $ref)) {
        Write-Host ("--- " + $_.Name)
        py (Join-Path $PSScriptRoot 'wdump_stats.py') --weights $_.FullName --ref $ref | Select-Object -Skip 1
    }
}
Write-Host "`n== align (cmd_align.py) =="
Get-ChildItem (Join-Path $dir 'cmd_*.jsonl') -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("--- " + $_.Name)
    py (Join-Path $PSScriptRoot 'cmd_align.py') --trace $_.FullName --block 0
}
