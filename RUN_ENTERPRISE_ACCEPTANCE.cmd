@echo off
setlocal EnableExtensions
cd /d "%~dp0MINCO"
call scripts\product_bootstrap.cmd
if errorlevel 1 exit /b %errorlevel%
echo Running scaled research twin acceptance...
.venv\Scripts\python.exe scripts\run_scaled_research_acceptance.py
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe scripts\run_enterprise_acceptance.py %*
exit /b %errorlevel%
