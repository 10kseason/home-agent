@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Anchor to this script directory (absolute path)
pushd "%~dp0" || (echo [ERROR] Failed to cd to "%~dp0" & exit /b 1)
echo [INFO] Project root: %CD%

rem 1) Prefer venv interpreter
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  echo [INFO] Using venv: "%VENV_PY%"
  "%VENV_PY%" -V
  "%VENV_PY%" -m agent.main
  set "EC=%ERRORLEVEL%"
  popd
  endlocal & exit /b %EC%
)

rem 2) Fallback: Python launcher or python on PATH
set "PYLAUNCH="
where py >nul 2>nul && set "PYLAUNCH=py -3"
if not defined PYLAUNCH (
  where python >nul 2>nul && set "PYLAUNCH=python"
)
if not defined PYLAUNCH (
  echo [ERROR] No Python interpreter found on PATH.
  echo         Install Python 3.10+ or create .venv first.
  popd
  endlocal & exit /b 1
)

rem Quick dep check (avoid fragile IF .%var%==. hacks)
%PYLAUNCH% -c "import fastapi" 1>nul 2>nul
if errorlevel 1 (
  echo [ERROR] Missing deps (e.g., fastapi).
  echo         Create venv and install:
  echo         py -3 -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
  popd
  endlocal & exit /b 1
)

%PYLAUNCH% -m agent.main
set "EC=%ERRORLEVEL%"

popd
endlocal & exit /b %EC%
