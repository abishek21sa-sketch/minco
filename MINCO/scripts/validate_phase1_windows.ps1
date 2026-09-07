$ErrorActionPreference = "Stop"
Write-Host "MINCO Phase 1 Mathematical Hospital Engine validation" -ForegroundColor Cyan
python --version
python -m pytest
python -m src.system.run_phase1_mathematical_engine_validation

$julia = Get-Command julia -ErrorAction SilentlyContinue
if ($null -eq $julia) {
    Write-Host "Julia runtime not found. Python/math validation passed; Julia/JuMP/Gurobi acceptance remains required." -ForegroundColor Yellow
    exit 2
}

julia julia/setup.jl
julia julia/scripts/run_reference.jl
julia julia/scripts/run_progressive_hedging.jl
Write-Host "MINCO Phase 1 acceptance PASSED." -ForegroundColor Green
