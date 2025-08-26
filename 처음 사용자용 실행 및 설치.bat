@echo off
setlocal
cd /d "%~dp0"
set VENV_DIR=%~dp0.venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    echo Installing dependencies...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
)
set SETTINGS_SCRIPT=%~dp0Gui_settings.py
set /p MODIFY_SETTINGS="설정을 수정하시겠습니까? (Y/N): "
if /I "%MODIFY_SETTINGS%"=="Y" (
    "%VENV_DIR%\Scripts\python.exe" "%SETTINGS_SCRIPT%"
)

echo Starting application...
"%VENV_DIR%\Scripts\python.exe" -m agent.main
pause
