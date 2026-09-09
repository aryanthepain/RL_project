param(
    [string]$ArgsString = "",
    [string]$LogFile = "logs/progression_generation.log"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  HalfCheetah Progression Background Worker (Issue #6)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Log destination: $LogFile" -ForegroundColor Yellow
Write-Host "Running command: python -u scripts/run_halfcheetah_progression.py $ArgsString"
Write-Host "------------------------------------------------------------------" -ForegroundColor DarkGray

Invoke-Expression "python -u scripts/run_halfcheetah_progression.py $ArgsString *>&1 | Tee-Object -FilePath '$LogFile'"

Write-Host "------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Worker completed successfully!" -ForegroundColor Green

$AlarmScript = Join-Path $env:USERPROFILE ".gemini\config\scripts\agent-alarm.ps1"
if (Test-Path $AlarmScript) {
    pwsh -File "$AlarmScript" -Type Success -Message "HalfCheetah progression videos and montage generation complete."
}

