$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\python.exe -m src.system.run_phase2_backend_acceptance
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
