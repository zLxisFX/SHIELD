@echo off
setlocal

REM Run from repo root even if launched elsewhere
cd /d %~dp0\..

REM Try to activate conda env if possible
REM If this fails, user can run manually in Miniforge Prompt:
REM   conda activate shield
REM   python -m shield --mode demo ...

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] conda not found in PATH.
  echo Open "Miniforge Prompt", then run:
  echo   cd /d C:\SHIELD\SHIELD
  echo   conda activate shield
  echo   python -m shield --mode demo --scenario smoke_heat_day --archetype classroom --area 90
  pause
  exit /b 1
)

call conda activate shield >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Could not activate conda environment "shield".
  echo In Miniforge Prompt, run:
  echo   conda activate shield
  pause
  exit /b 1
)

python -m shield --mode demo --scenario smoke_heat_day --archetype classroom --area 90
echo.
echo Done. Check the outputs\ folder for exported plans.
pause
