@echo off
setlocal
cd /d "%~dp0"
set VENV_DIR=%~dp0.venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    python -m venv "%VENV_DIR%"
)

"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
"%VENV_DIR%\Scripts\python.exe" -m pytest -q
pause
