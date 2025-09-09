@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Anchor to this script directory (absolute path)
pushd "%~dp0" || (echo [ERROR] Cannot cd to "%~dp0" & exit /b 1)
echo [INFO] Project root: %CD%

rem Locate a Python launcher on PATH
set "PYLAUNCH="
where py >nul 2>nul && set "PYLAUNCH=py -3"
if not defined PYLAUNCH (
  where python >nul 2>nul && set "PYLAUNCH=python"
)
if not defined PYLAUNCH (
  echo [ERROR] No Python interpreter found on PATH.
  echo         Install Python 3.10+ and re-run this installer.
  popd
  endlocal & exit /b 1
)

rem Create venv if needed
if not exist ".venv\" (
  echo [INFO] Creating venv at .venv
  %PYLAUNCH% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    popd
    endlocal & exit /b 1
  )
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [ERROR] Venv python not found at "%VENV_PY%"
  popd
  endlocal & exit /b 1
)

rem Install requirements if present
if exist "requirements.txt" (
  echo [INFO] Installing dependencies from requirements.txt
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install some requirements. See output above.
    popd
    endlocal & exit /b 1
  )
) else (
  echo [WARN] requirements.txt not found. Skipping dependency install.
)

echo.
echo [DONE] Setup complete.
echo [HINT] Run the server with: run-server.bat

popd
endlocal
exit /b 0
