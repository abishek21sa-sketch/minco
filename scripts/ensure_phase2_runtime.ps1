[CmdletBinding()]
param(
  [int]$Port = 50551
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runtime = Join-Path $repoRoot "scripts\start_phase2_runtime.ps1"

if (-not (Test-Path -LiteralPath $python)) {
  throw "MINCO Python runtime is missing: $python"
}
if (-not (Test-Path -LiteralPath $runtime)) {
  throw "MINCO runtime launcher is missing: $runtime"
}

Set-Location $repoRoot

function Get-MincoListener {
  return Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
}

function Get-MincoProcess([int]$ProcessId) {
  return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
}

function Test-MincoRuntime([int]$ProcessId) {
  $process = Get-MincoProcess $ProcessId
  if (-not $process -or $process.Name -ne "python.exe") { return $false }
  if ($process.CommandLine -notmatch "src\.runtime\.grpc_server") { return $false }

  $probe = "from src.runtime.grpc_client import MincoRuntimeClient; c=MincoRuntimeClient('127.0.0.1:$Port'); c._call('GetScenarioCatalog', timeout=3); c.close(); print('MINCO_RUNTIME_OK')"
  & $python -c $probe 2>$null | Out-Null
  return $LASTEXITCODE -eq 0
}

$listener = Get-MincoListener
if ($listener) {
  if (Test-MincoRuntime $listener.OwningProcess) {
    Write-Host "MINCO runtime is healthy on 127.0.0.1:$Port (PID $($listener.OwningProcess)); reusing it."
    exit 0
  }

  $owner = Get-MincoProcess $listener.OwningProcess
  $expectedPython = [regex]::Escape($python)
  if ($owner -and $owner.Name -eq "python.exe" -and $owner.CommandLine -match $expectedPython -and $owner.CommandLine -match "src\.runtime\.grpc_server") {
    Write-Host "Restarting stale MINCO runtime PID $($listener.OwningProcess)."
    Stop-Process -Id $listener.OwningProcess -Force
    Start-Sleep -Seconds 1
  }
  else {
    throw "Port 127.0.0.1:$Port is owned by an unrelated process; refusing to stop it."
  }
}

Write-Host "Starting MINCO Phase-2 gRPC runtime on 127.0.0.1:$Port..."
$launcher = Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runtime) `
  -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru

$deadline = (Get-Date).AddSeconds(90)
do {
  Start-Sleep -Seconds 2
  $listener = Get-MincoListener
  if ($listener -and (Test-MincoRuntime $listener.OwningProcess)) {
    Write-Host "MINCO runtime is ready on 127.0.0.1:$Port (PID $($listener.OwningProcess))."
    exit 0
  }
  if ($launcher.HasExited -and -not $listener) {
    throw "MINCO runtime launcher exited before the gRPC contract became ready."
  }
} while ((Get-Date) -lt $deadline)

throw "MINCO runtime did not become ready within 90 seconds."
