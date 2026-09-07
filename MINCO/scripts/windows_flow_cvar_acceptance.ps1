$ErrorActionPreference = "Stop"
Write-Host "MINCO FLOW-CVaR Windows Acceptance"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$pyCandidate = Join-Path $root ".venv\Scripts\python.exe"
$py = if (Test-Path $pyCandidate) { $pyCandidate } else { "python" }
$env:PYTHONPATH = $root
& $py -m pytest tests/phase1/test_flow_cvar_network_signature.py tests/phase1/test_flow_cvar_decision.py tests/test_flow_cvar_api.py tests/test_flow_cvar_ui_contract.py tests/test_fixed_first_stage_oracle.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/flow_cvar_signature_evidence.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/flow_cvar_product_evidence.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m py_compile src/minco4x/signature_algorithm.py src/api/main.py dashboard/app.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "FLOW_CVAR_WINDOWS_ACCEPTANCE=PASS"
