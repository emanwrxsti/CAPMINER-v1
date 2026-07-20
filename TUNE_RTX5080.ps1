$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$possibleExe = @(
    (Join-Path $PSScriptRoot 'capminer.exe'),
    (Join-Path $PSScriptRoot 'Release\capminer.exe'),
    (Join-Path $PSScriptRoot 'build\Release\capminer.exe')
)
$exe = $possibleExe | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $exe) {
    throw 'capminer.exe was not found. Put this script in the capminer source folder or beside capminer.exe.'
}

Write-Host "Using: $exe"
Write-Host 'Stop all other GPU workloads before tuning.'

# Raising the limit only removes a cap; the kernel may still draw less power.
try {
    & nvidia-smi -i 0 -pl 315 | Out-Host
} catch {
    Write-Warning 'Could not set 315 W. Open PowerShell as Administrator and run the script again.'
}

$threadValues = @(64, 128, 256, 512)
$blockValues  = @(4, 8, 12, 16, 24, 32)
$seconds = 5
$results = [System.Collections.Generic.List[object]]::new()

foreach ($threads in $threadValues) {
    foreach ($blocks in $blockValues) {
        Write-Host "`nTesting threads=$threads blocks-per-sm=$blocks ..." -ForegroundColor Cyan
        $output = & $exe `
            --algo alphanumeric `
            --benchmark `
            --devices 0 `
            --threads $threads `
            --blocks-per-sm $blocks `
            --bench-seconds $seconds `
            --no-opencl 2>&1 | Out-String

        $match = [regex]::Match($output, 'benchmark:\s*([0-9]+(?:\.[0-9]+)?)\s*GH/s', 'IgnoreCase')
        if ($match.Success) {
            $ghs = [double]$match.Groups[1].Value
            $results.Add([pscustomobject]@{
                Threads    = $threads
                BlocksPerSM = $blocks
                GHs         = $ghs
            })
            Write-Host ("Result: {0:N3} GH/s" -f $ghs) -ForegroundColor Green
        } else {
            Write-Warning 'No benchmark result was parsed. Full output follows:'
            Write-Host $output
        }
    }
}

$sorted = $results | Sort-Object GHs -Descending
$csv = Join-Path $PSScriptRoot 'rtx5080_tuning_results.csv'
$sorted | Export-Csv -NoTypeInformation -LiteralPath $csv

Write-Host "`nTop RTX 5080 configurations:" -ForegroundColor Yellow
$sorted | Select-Object -First 10 | Format-Table -AutoSize
Write-Host "Results saved to: $csv"

$best = $sorted | Select-Object -First 1
if ($best) {
    Write-Host "`nBest: THREADS=$($best.Threads), BLOCKS_PER_SM=$($best.BlocksPerSM), $($best.GHs) GH/s" -ForegroundColor Green
    Write-Host 'Copy those two values into START_MINING_315W.bat.'
}
