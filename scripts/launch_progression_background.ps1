<#
.SYNOPSIS
    Launches the HalfCheetah policy progression generation pipeline in a detached background terminal.
.DESCRIPTION
    Runs python -u scripts/run_halfcheetah_progression.py in an independent background process.
    Output is streamed continuously to logs/progression_generation.log.
    This allows the developer to proceed immediately with other phases while videos are generated.
.EXAMPLE
    pwsh -File .\scripts\launch_progression_background.ps1
    pwsh -File .\scripts\launch_progression_background.ps1 -SmokeTest
#>

param(
    [int]$TotalTimesteps = 10000,
    [int]$CheckpointFreq = 100,
    [int]$EvalFrames = 500,
    [int]$Fps = 30,
    [int]$Seed = 42,
    [string]$OutputDir = "videos",
    [string]$CheckpointDir = "runs/progression_run",
    [switch]$SkipTrain,
    [switch]$SmokeTest,
    [string]$Device = "cpu",
    [int]$TimelapseFactor = 1
)

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}

$ProjectRoot = (Get-Location).Path
$LogFile = Join-Path $ProjectRoot "logs\progression_generation.log"
$ErrLogFile = Join-Path $ProjectRoot "logs\progression_generation_err.log"

$ScriptArgs = @(
    "-u", "scripts/run_halfcheetah_progression.py",
    "--total-timesteps", "$TotalTimesteps",
    "--checkpoint-freq", "$CheckpointFreq",
    "--eval-frames", "$EvalFrames",
    "--fps", "$Fps",
    "--seed", "$Seed",
    "--output-dir", "$OutputDir",
    "--checkpoint-dir", "$CheckpointDir",
    "--device", "$Device",
    "--timelapse-factor", "$TimelapseFactor"
)

if ($SkipTrain) {
    $ScriptArgs += "--skip-train"
}
if ($SmokeTest) {
    $ScriptArgs += "--smoke-test"
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  HalfCheetah Progression Background Launcher (Issue #6)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
# Drop script name from args array since worker script invokes it
$ArgsOnly = $ScriptArgs[2..($ScriptArgs.Length - 1)]
$ArgsString = ($ArgsOnly | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join " "
$WorkerScript = Join-Path $PSScriptRoot "worker_progression.ps1"

$Process = Start-Process -FilePath "pwsh.exe" `
    -ArgumentList "-NoExit", "-Command", "& '$WorkerScript' -ArgsString '$ArgsString' -LogFile '$LogFile'" `
    -WorkingDirectory $ProjectRoot `
    -PassThru

Write-Host "Pipeline launched in separate terminal window!" -ForegroundColor Green
Write-Host "Process ID (PID) : $($Process.Id)" -ForegroundColor Green
Write-Host "Monitoring command:" -ForegroundColor Cyan
Write-Host "  Get-Content -Wait logs/progression_generation.log" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan
