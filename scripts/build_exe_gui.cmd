@echo off
setlocal enabledelayedexpansion

REM Build a double-click GUI EXE (no console) using PyInstaller.
REM Output: dist\SHIELD-GUI.exe

cd /d %~dp0\..
set PY=python

echo [1/3] Using Python:
%PY% -c "import sys; print(sys.executable)"

echo.
echo [2/3] Ensuring PyInstaller is installed...
%PY% -m pip install --upgrade pyinstaller >nul
if errorlevel 1 (
  echo ERROR: Failed to install/upgrade pyinstaller.
  exit /b 1
)

echo.
echo [3/3] Building dist\SHIELD-GUI.exe ...

REM Clean old build artifacts
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Notes:
REM --windowed = no console
REM --paths adds your src/tools so imports work in frozen builds
REM Entry script is the GUI module we added.

%PY% -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name SHIELD-GUI ^
  --paths src ^
  --paths tools ^
  src\shield\app\gui.py

if errorlevel 1 (
  echo ERROR: PyInstaller build failed.
  exit /b 1
)

echo.
echo SUCCESS: dist\SHIELD-GUI.exe
echo Try:
echo   dist\SHIELD-GUI.exe
endlocal
