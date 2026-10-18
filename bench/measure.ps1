# Pinned, turbo-normalised A/B comparison of two huffman binaries.
#
# Why: on a laptop the CPU clock swings with turbo boost and thermal state, so
# raw MB/s from two runs minutes apart are not comparable. This script
#
#   1. pins every process to one physical core and raises its priority,
#   2. brackets each measurement with `freq_probe` runs on the same core,
#   3. rescales both throughputs to a common reference frequency,
#   4. interleaves A and B so both see the same thermal state,
#
# and reports the per-round speedup. Run it from an elevated shell if priority
# changes are refused; the affinity still applies.
#
# Usage:
#   .\measure.ps1 -A .\build\huffman.exe -B .\build-o2\huffman.exe `
#                 -ArgsA @('bench','--p-uniform','--seed','12345') `
#                 -ArgsB @('bench','--p-uniform','--seed','12345') `
#                 -PatternA 'best\s+([0-9.]+)\s+MB/s' -PatternB 'best\s+([0-9.]+)\s+MB/s'

[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]   $A,           # binary under test
    [Parameter(Mandatory)] [string]   $B,           # reference binary
    [string[]] $ArgsA = @(),
    [string[]] $ArgsB = @(),
    [string]   $PatternA = 'best\s+([0-9.]+)\s+MB/s',
    [string]   $PatternB = 'MB/s:\s*([0-9.]+)',
    [int]      $Core     = 2,
    [int]      $Rounds   = 6,
    [int]      $ProbeOps = 60000000,
    [string]   $Probe    = ''
)

$ErrorActionPreference = 'Stop'

if (-not $Probe) {
    $candidates = @(
        (Join-Path $PSScriptRoot 'freq_probe.exe'),
        (Join-Path $PSScriptRoot '..\build\huffman_freq_probe.exe')
    )
    $Probe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Probe) {
    throw 'freq_probe was not found; build the huffman_freq_probe target or pass -Probe'
}

$mask = [IntPtr]([int64]1 -shl $Core)

function Invoke-Pinned {
    param([string] $Exe, [string[]] $Argv)

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName               = $Exe
    $startInfo.Arguments              = ($Argv -join ' ')
    $startInfo.UseShellExecute        = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError  = $true

    $process = [System.Diagnostics.Process]::Start($startInfo)
    try { $process.ProcessorAffinity = $mask } catch { Write-Warning 'could not pin the process' }
    try { $process.PriorityClass = 'High' }    catch { Write-Warning 'could not raise the priority' }

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "$Exe failed: $stderr" }
    return $stdout
}

function Measure-Clock {
    # best of three: the fastest probe saw the least throttling
    $best = 0.0
    foreach ($unused in 1..3) {
        $match = [regex]::Match((Invoke-Pinned $Probe @($ProbeOps)), 'ops_per_ns\s+([0-9.]+)')
        if (-not $match.Success) { throw 'could not parse the frequency probe output' }
        $best = [math]::Max($best, [double] $match.Groups[1].Value)
    }
    return $best
}

function Measure-Throughput {
    param([string] $Exe, [string[]] $Argv, [string] $Pattern)

    $match = [regex]::Match((Invoke-Pinned $Exe $Argv), $Pattern)
    if (-not $match.Success) { throw "could not parse the throughput of $Exe" }
    return [double] $match.Groups[1].Value
}

Write-Host ("core {0} (mask 0x{1:X}), {2} rounds" -f $Core, [int64]$mask, $Rounds)

# Reference clock: the fastest probe ever seen, i.e. full turbo.
$reference = Measure-Clock
Write-Host ("reference clock: {0:N3} ops/ns`n" -f $reference)

$rows = @()
foreach ($round in 1..$Rounds) {
    $beforeA = Measure-Clock
    $valueA  = Measure-Throughput $A $ArgsA $PatternA
    $afterA  = Measure-Clock
    $valueB  = Measure-Throughput $B $ArgsB $PatternB

    $clockA    = ($beforeA + $afterA) / 2
    $normalA   = $valueA * $reference / $clockA
    $normalB   = $valueB * $reference / $afterA

    $rows += [pscustomobject]@{
        Round      = $round
        Clock_A    = [math]::Round($clockA, 3)
        Raw_A      = [math]::Round($valueA, 1)
        Norm_A     = [math]::Round($normalA, 1)
        Clock_B    = [math]::Round($afterA, 3)
        Raw_B      = [math]::Round($valueB, 1)
        Norm_B     = [math]::Round($normalB, 1)
        Speedup    = [math]::Round($normalA / $normalB, 2)
    }
}

$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

$speedups = $rows | ForEach-Object { $_.Speedup }
$best = ($rows | Measure-Object -Property Norm_B -Maximum).Maximum
$worst = ($rows | Measure-Object -Property Norm_B -Minimum).Minimum

Write-Host ("A ({0}): normalised {1:N1}..{2:N1} MB/s" -f (Split-Path $A -Leaf),
            ($rows | Measure-Object -Property Norm_A -Minimum).Minimum,
            ($rows | Measure-Object -Property Norm_A -Maximum).Maximum)
Write-Host ("B ({0}): normalised {1:N1}..{2:N1} MB/s" -f (Split-Path $B -Leaf), $worst, $best)
Write-Host ("per-round speedup (A over B): {0}" -f ($speedups -join ', '))
Write-Host ("mean speedup: {0:N2}x" -f ($speedups | Measure-Object -Average).Average)
