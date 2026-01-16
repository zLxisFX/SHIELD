@echo off
setlocal

REM Build SHIELD.exe from the demo CLI (run_eval_sweep.py)
REM Run this from an activated (shield) environment.

cd /d "%~dp0\.."

echo [1/3] Using Python:
python -c "import sys; print(sys.executable)"

echo.
echo [2/3] Ensuring PyInstaller is installed...
python -m pip install --upgrade pyinstaller

echo.
echo [3/3] Building dist\SHIELD.exe ...
pyinstaller --noconfirm --clean --onefile --name SHIELD tools\run_eval_sweep.py

echo.
if exist "dist\SHIELD.exe" (
  echo SUCCESS: dist\SHIELD.exe
  echo.
  echo Try:
  echo   dist\SHIELD.exe --help
  echo   dist\SHIELD.exe --scenario smoke_heat_day --archetype classroom --area 90 --start-hour 0 --hepa-budget 2 --members 25 --horizon-hours 72
  exit /b 0
) else (
  echo ERROR: Build finished but dist\SHIELD.exe not found.
  exit /b 1
)

endlocal
