$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$workstation = Join-Path $repo "workstation"
Set-Location $workstation

if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
  throw "Flutter is not on PATH. Install the current Flutter Windows SDK, enable Windows desktop, reopen PowerShell, then rerun this script."
}

$doctorOutput = (& flutter doctor -v 2>&1 | Out-String)
if ($doctorOutput -match '(?im)^\[!\].*Visual Studio - develop Windows apps' -or $doctorOutput -match '(?i)Visual Studio.*not installed') {
  throw "Flutter is installed, but the Visual Studio C++ Windows toolchain is missing. Install Visual Studio 2022 with Desktop development with C++, MSVC v143, C++ CMake tools for Windows, and a Windows 10/11 SDK, then rerun this script."
}

$temp = Join-Path $env:TEMP ("minco_flutter_source_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temp | Out-Null
Copy-Item "lib" $temp -Recurse
Copy-Item "test" $temp -Recurse
Copy-Item "pubspec.yaml" $temp
Copy-Item "analysis_options.yaml" $temp

try {
  flutter config --enable-windows-desktop
  flutter create --platforms=windows --project-name minco_workstation .
  Remove-Item "lib" -Recurse -Force
  Remove-Item "test" -Recurse -Force
  Copy-Item (Join-Path $temp "lib") "." -Recurse
  Copy-Item (Join-Path $temp "test") "." -Recurse
  Copy-Item (Join-Path $temp "pubspec.yaml") "." -Force
  Copy-Item (Join-Path $temp "analysis_options.yaml") "." -Force
  flutter pub get
  flutter analyze
  flutter test
  flutter build windows --release
} finally {
  Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
}

$exe = Join-Path $workstation "build\windows\x64\runner\Release\minco_workstation.exe"
if (-not (Test-Path $exe)) {
  throw "Flutter build completed without the expected Windows executable: $exe"
}
Write-Host "MINCO workstation build passed: $exe" -ForegroundColor Green
