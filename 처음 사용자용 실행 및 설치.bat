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
    echo Downloading STT model (this may take a while)...
    "%VENV_DIR%\Scripts\python.exe" -c "from faster_whisper import WhisperModel; WhisperModel('small')"
    echo Downloading OCR model...
    "%VENV_DIR%\Scripts\python.exe" -c "import easyocr; easyocr.Reader(['ko','en'])"
)

echo Starting application...
"%VENV_DIR%\Scripts\python.exe" -m agent.main
pause
