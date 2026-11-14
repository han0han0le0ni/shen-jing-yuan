# T2h · 并行跑 M2 口径探针（直通/残差结构改动）
# 预注册：audit/b2_direct_prereg_2026-11-14.md
# 臂定义：s1(fan24,hm1,im1) / s2(fan24,hm0,im1) / s3(fan24,hm1,im0.5) / s4(fan24,hm0,im0.5)
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\tools\run_m2h.ps1 -Launch -SeedList "101,202,303" -ArmList "s3,s4"
#   powershell -ExecutionPolicy Bypass -File .\tools\run_m2h.ps1 -SeedList "101" -ArmList "s1"   # 等待模式
param([switch]$Launch,
      [string]$SeedList = '101,202,303,404,505,606,707,808,909,1010',
      [string]$ArmList = 's1,s2')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$seeds = @($SeedList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { [int]$_.Trim() })
$tags = @($ArmList.Split(',') | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $_.Trim() })

$all = @(
    @{ tag = 's1'; fan = 24; hm = '1.0'; im = '1.0' },
    @{ tag = 's2'; fan = 24; hm = '0';   im = '1.0' },
    @{ tag = 's3'; fan = 24; hm = '1.0'; im = '0.5' },
    @{ tag = 's4'; fan = 24; hm = '0';   im = '0.5' }
)
$arms = @($all | Where-Object { $tags -contains $_.tag })
if ($arms.Count -eq 0) { throw "no such arm: $ArmList" }

foreach ($arm in $arms) {
    $dir = Join-Path $root "data_b2\m2h\$($arm.tag)"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    foreach ($s in $seeds) {
        $out = Join-Path $dir "m2h_$($arm.tag)_s$s.csv"
        if (Test-Path $out) { continue }
        $args = @('--preset', 'm2', '--seed', "$s", '--episodes', '320', '--eval-episodes', '20',
                  '--direct-fan', "$($arm.fan)", '--hm-scale', "$($arm.hm)", '--im-scale', "$($arm.im)",
                  '--out', $out)
        Start-Process -FilePath (Join-Path $root 'benchmark2.exe') -ArgumentList $args `
            -RedirectStandardOutput (Join-Path $dir "log_$($arm.tag)_s$s.txt") `
            -RedirectStandardError  (Join-Path $dir "err_$($arm.tag)_s$s.txt") `
            -WindowStyle Hidden
    }
}

if ($Launch) { Write-Host 'launched'; exit 0 }

$total = $arms.Count * $seeds.Count
while ($true) {
    $done = 0
    foreach ($arm in $arms) {
        $dir = Join-Path $root "data_b2\m2h\$($arm.tag)"
        $done += (Get-ChildItem $dir -Filter *.csv -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    Write-Host ("{0}  {1}/{2} csv" -f (Get-Date -Format 'HH:mm:ss'), $done, $total)
    if ($done -ge $total) { break }
    Start-Sleep -Seconds 30
}
Write-Host 'all done'
