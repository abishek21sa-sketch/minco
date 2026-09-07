@echo off
setlocal EnableExtensions
cd /d "%~dp0MINCO"
call scripts\product_bootstrap.cmd
if errorlevel 1 exit /b %errorlevel%
if not defined MINCO_PORT set "MINCO_PORT=8814"
echo MINCO GOVERNED API + COMMAND CENTER: http://127.0.0.1:%MINCO_PORT%/
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port %MINCO_PORT%
exit /b %errorlevel%
