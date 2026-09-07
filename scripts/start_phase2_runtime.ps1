$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  throw "MINCO .venv is missing. Create Python 3.13 environment first."
}

$env:MINCO_EVENT_STORE = if ($env:MINCO_EVENT_STORE) { $env:MINCO_EVENT_STORE } else { "duckdb" }
Write-Host "MINCO Phase 2 runtime" -ForegroundColor Cyan
Write-Host "Event store: $env:MINCO_EVENT_STORE"
Write-Host "gRPC: 127.0.0.1:50551"
& .\.venv\Scripts\python.exe -m src.runtime.grpc_server
