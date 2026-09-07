@echo off
setlocal EnableExtensions
cd /d "%~dp0MINCO"
call scripts\product_bootstrap.cmd
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe scripts\product_runtime.py --accept
exit /b %errorlevel%
