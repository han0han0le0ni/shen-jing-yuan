# M1 工作点修复扫参（预注册 audit/m1_workpoint_prereg_2026-11-16.md，负责人 2026-11-16 签字「可以」）
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\run_m1.ps1 -Launch
# 产出：data_b2/m1/<arm>_s<seed>.json（每个 (臂,seed) 一份 m0_workpoint_diag 的 JSON）
#       data_b2/m1/bitexact_s101.csv（320 ep 逐位校验对照，与 data_b2/m2/m2_s101.csv 比 SHA256）
param(
    [int[]]$Seeds = @(101, 202, 303),
    [switch]$Launch,
    [switch]$SkipBitExact
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path $PSScriptRoot -Parent
$OutDir = Join-Path $Root 'data_b2\m1'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# 六个臂（预注册 §2）；motor_base 全部保持 3.0（单变量）
$arms = @(
    @{ n = 'A0'; hb = 2.0; psp = 25.0 },
    @{ n = 'A1'; hb = 0.5; psp = 25.0 },
    @{ n = 'A2'; hb = 0.0; psp = 25.0 },
    @{ n = 'B0'; hb = 2.0; psp = 6.0 },
    @{ n = 'B1'; hb = 0.5; psp = 6.0 },
    @{ n = 'C1'; hb = 0.0; psp = 1.5 }
)

if (-not $Launch) {
    Write-Host 'Dry run. 将执行：'
    foreach ($a in $arms) { foreach ($s in $Seeds) { Write-Host ("  {0} hb={1} psp={2} seed={3}" -f $a.n, $a.hb, $a.psp, $s) } }
    Write-Host ("  bit-exact: 320 ep seed 101 -> {0}" -f (Join-Path $OutDir 'bitexact_s101.csv'))
    return
}

$jobs = @()

# 1) 逐位校验：320 ep / seed 101，与 data_b2/m2/m2_s101.csv 比 SHA256
if (-not $SkipBitExact) {
    $csv = Join-Path $OutDir 'bitexact_s101.csv'
    $jobs += Start-Job -ScriptBlock {
        param($root, $csv)
        Set-Location $root
        & (Join-Path $root 'benchmark2.exe') --preset m2 --episodes 320 --seed 101 --out $csv | Out-Null
        'done'
    } -ArgumentList $Root, $csv -Name 'bitexact'
}

# 2) 六臂 × 三 seed：每臂一个作业（内部串行跑 3 个 seed），控制并发内存
foreach ($a in $arms) {
    $jobs += Start-Job -ScriptBlock {
        param($root, $hb, $psp, $seeds, $outdir, $armname)
        Set-Location $root
        foreach ($s in $seeds) {
            $out = Join-Path $outdir ("{0}_s{1}.json" -f $armname, $s)
            & py tools/m0_workpoint_diag.py --seed $s --hidden-base $hb --psp $psp --out $out 2>$null | Out-Null
        }
        $armname
    } -ArgumentList $Root, $a.hb, $a.psp, $Seeds, $OutDir, $a.n -Name ("arm_" + $a.n)
}

Write-Host ("Launched {0} jobs. Waiting..." -f $jobs.Count)
$jobs | Wait-Job | Out-Null
$jobs | Receive-Job | Out-Null
$jobs | Remove-Job

if (-not $SkipBitExact) {
    $ref = Join-Path $Root 'data_b2\m2\m2_s101.csv'
    $new = Join-Path $OutDir 'bitexact_s101.csv'
    if ((Test-Path $ref) -and (Test-Path $new)) {
        $h1 = (Get-FileHash $ref -Algorithm SHA256).Hash
        $h2 = (Get-FileHash $new -Algorithm SHA256).Hash
        Write-Host ("bitexact ref={0}" -f $h1)
        Write-Host ("bitexact new={0}" -f $h2)
        Write-Host ("bitexact PASS={0}" -f ($h1 -eq $h2))
    }
}

Write-Host '==> summary'
& py (Join-Path $Root 'tools\m1_summary.py') --dir $OutDir
