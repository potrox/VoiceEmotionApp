@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "VOICEEMOTIONAPP_RUNTIME=%~dp0.runtime"

call :find_python
if errorlevel 1 goto :python_error

%PYTHON_CMD% build.py
if errorlevel 1 goto :build_error
echo.
echo Build finished. Check the dist folder.
pause
exit /b 0

:find_python
set "PYTHON_CMD="
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.10"
    exit /b 0
)
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    exit /b 0
)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    exit /b 0
)
py -3.13 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.13"
    exit /b 0
)
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)
exit /b 1

:python_error
echo Python was not found.
echo Install Python 3.10 from python.org and enable Add Python to PATH.
pause
exit /b 1

:build_error
echo.
echo EXE build failed.
echo Recommended fix: extract the project to C:\VoiceEmotionApp and run build_exe.bat again.
echo If the error mentions Windows Long Path support, install Python 3.10 from python.org or enable Windows Long Path support.
pause
exit /b 1
