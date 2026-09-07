$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo "workstation\build\windows\x64\runner\Release\minco_workstation.exe"
if (-not (Test-Path $exe)) {
  throw "Native workstation build missing. Run scripts\bootstrap_flutter_windows.ps1 first."
}
Start-Process $exe
