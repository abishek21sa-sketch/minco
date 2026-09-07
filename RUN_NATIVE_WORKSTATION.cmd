@echo off
setlocal EnableExtensions

set "PRODUCT_ROOT=%~dp0MINCO"
set "PYTHON=%PRODUCT_ROOT%\.venv\Scripts\python.exe"
set "WORKSTATION_EXE=%PRODUCT_ROOT%\workstation\build\windows\x64\runner\Release\minco_workstation.exe"
set "BOOTSTRAP=%PRODUCT_ROOT%\scripts\bootstrap_flutter_windows.ps1"
set "ENSURE_RUNTIME=%PRODUCT_ROOT%\scripts\ensure_phase2_runtime.ps1"

if not exist "%PYTHON%" (
  echo MINCO Python runtime is missing: %PYTHON%
  exit /b 1
)

if not exist "%WORKSTATION_EXE%" (
  echo Native workstation build is missing.
  echo.
  echo Install Flutter Windows SDK and Visual Studio 2022 Desktop development with C++,
  echo then run:
  echo   powershell -ExecutionPolicy Bypass -File "%BOOTSTRAP%"
  echo.
  echo The governed browser Command Center remains available through RUN_APP.cmd.
  exit /b 2
)

echo Ensuring MINCO Phase-2 gRPC runtime on 127.0.0.1:50551...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ENSURE_RUNTIME%"
if errorlevel 1 (
  echo MINCO runtime failed its readiness check.
  exit /b 1
)

echo Starting native MINCO workstation...
start "MINCO Native Workstation" "%WORKSTATION_EXE%"
exit /b 0
