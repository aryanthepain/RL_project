@echo off
title HalfCheetah Progression Pipeline (Issue #6)
cd /d "%~dp0\.."
echo ==================================================================
echo   HalfCheetah Policy Progression Pipeline (10,000 Steps)
echo ==================================================================
echo Repo Directory: %CD%
echo Output Log    : logs\progression_generation.log
echo Videos Dir    : videos\
echo.
if not exist logs mkdir logs
echo Starting execution at %DATE% %TIME%...
python -u scripts\run_halfcheetah_progression.py %* 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'logs\progression_generation.log'"
echo.
echo ==================================================================
echo Pipeline finished at %DATE% %TIME%
echo ==================================================================
powershell -NoProfile -Command "if (Test-Path '%USERPROFILE%\.gemini\config\scripts\agent-alarm.ps1') { pwsh -File '%USERPROFILE%\.gemini\config\scripts\agent-alarm.ps1' -Type Success -Message 'HalfCheetah progression videos and montage generation complete.' }"
echo.
echo Press any key to close this terminal...
pause > nul
