from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_julia_primary_optimizer_contains_required_stochastic_or_constructs():
    extensive = (ROOT / "julia/src/extensive_form.jl").read_text(encoding="utf-8")
    ph = (ROOT / "julia/src/progressive_hedging.jl").read_text(encoding="utf-8")
    rolling = (ROOT / "julia/src/rolling_horizon.jl").read_text(encoding="utf-8")
    for token in [
        "@variable(model, surge",
        "Bin",
        "flex",
        "scenario_loss",
        "cvar",
        "eta",
        "xi",
        "transfer_capacity",
        "unsafe",
        "boarding",
    ]:
        assert token in extensive
    assert "progressive_hedging" in ph
    assert "rho" in ph and "residual" in ph
    # Regression: PH objective-history weighting must not call LinearAlgebra.dot
    # unless LinearAlgebra is explicitly imported. Keep the implementation
    # dependency-free by using elementwise probability weighting + sum.
    assert "dot(data.probabilities, objectives)" not in ph
    assert "sum(data.probabilities .* objectives)" in ph
    assert "rolling_horizon_solve" in rolling
    project = (ROOT / "julia/Project.toml").read_text(encoding="utf-8")
    for package in ["JuMP", "MathOptInterface", "Gurobi", "HiGHS", "JSON3"]:
        assert f"{package} =" in project
    # Registry-integrity regression: package-name presence alone is insufficient.
    assert 'JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"' in project
    setup = (ROOT / "julia/setup.jl").read_text(encoding="utf-8")
    assert "Pkg.resolve()" in setup
    assert "Pkg.instantiate()" in setup
    assert "Pkg.precompile()" in setup
    reference = (ROOT / "julia/scripts/run_reference.jl").read_text(encoding="utf-8")
    ph_accept = (ROOT / "julia/scripts/run_progressive_hedging.jl").read_text(encoding="utf-8")
    assert "12.86" in reference and '"PASSED"' in reference
    assert "progressive_hedging" in ph_accept and "agreement" in ph_accept
