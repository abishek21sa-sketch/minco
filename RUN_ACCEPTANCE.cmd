@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo MINCO_FLOW_CVaR_RC3_PY314 - Python 3.14 Windows Acceptance
echo ============================================================

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Windows Python launcher ^(py.exe^) was not found.
  echo Install Python 3.14 with the Python Launcher and retry.
  exit /b 2
)

py -3.14 -c "import sys; assert sys.version_info[:2]==(3,14); print(sys.version)"
if errorlevel 1 (
  echo ERROR: Python 3.14 is not available through: py -3.14
  exit /b 3
)

cd /d "%~dp0MINCO"
if not exist ".venv\Scripts\python.exe" (
  echo Creating clean Python 3.14 virtual environment...
  py -3.14 -m venv .venv
  if errorlevel 1 goto :fail
)
set "PY=%CD%\.venv\Scripts\python.exe"
"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,14) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Existing .venv is not Python 3.14. Rebuilding it...
  rmdir /s /q .venv
  py -3.14 -m venv .venv
  if errorlevel 1 goto :fail
  set "PY=%CD%\.venv\Scripts\python.exe"
)

echo Updating pip/build tooling...
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo Installing acceptance dependencies...
"%PY%" -m pip install -e ".[dev,solver,dashboard]"
if errorlevel 1 goto :fail

echo Running acceptance gate...
"%PY%" scripts\rc3_windows_acceptance.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo RC3 WINDOWS ACCEPTANCE PASSED
echo ============================================================
exit /b 0

:fail
echo.
echo ============================================================
echo RC3 WINDOWS ACCEPTANCE FAILED - exit code %ERRORLEVEL%
echo Copy the full PowerShell/CMD output back to ChatGPT.
echo ============================================================
exit /b %ERRORLEVEL%
