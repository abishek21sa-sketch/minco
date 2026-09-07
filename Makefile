.PHONY: install install-full install-dev smoke api dashboard run-pipeline command-center scalability scalability-plot test lint evidence docker-build solver-preflight forecast-baselines level2-validate level2-wp2 level2-wp3 level2-complete level3-wp1

install:
	python -m pip install -e .

install-full:
	python -m pip install -e ".[full]"

install-dev:
	python -m pip install -e ".[analytics,dashboard,dev]"

smoke:
	python -m src.system.run_smoke_pipeline

api:
	python -m uvicorn src.api.main:app --reload --port 8000

dashboard:
	python -m streamlit run dashboard/app.py

run-pipeline:
	python -m src.system.run_all_pipeline

command-center:
	python -m src.command_center.command_center_pipeline

SIZES ?= 10 25 50 100 200
scalability:
	python -m experiments.run_scalability_benchmark --sizes $(SIZES)

scalability-plot:
	python -m experiments.plot_scalability_results

test:
	pytest

lint:
	ruff check src/api src/config src/metrics src/validation src/system/run_smoke_pipeline.py tests scripts main.py

evidence:
	python scripts/generate_evidence_manifest.py

docker-build:
	docker build -t minco-api:0.4.0a1 .

solver-preflight:
	python -c "from src.validation.solver_readiness import check_gurobi_readiness; print(check_gurobi_readiness().to_dict())"

forecast-baselines:
	python -m src.validation.forecast_baseline_validation

level2-validate: test
	python -m src.system.run_level2_validation

level2-wp2:
	python -m src.system.run_level2_wp2_validation

level2-wp3:
	python -m src.system.run_level2_wp3_validation

level2-complete:
	python -m src.system.run_level2_completion_validation

level3-wp1:
	python -m src.system.run_level3_wp1_validation

phase1-math:
	python -m src.system.run_phase1_mathematical_engine_validation

phase1-tests:
	pytest tests/phase1

julia-setup:
	julia julia/setup.jl

julia-reference:
	julia julia/scripts/run_reference.jl

julia-ph:
	julia julia/scripts/run_progressive_hedging.jl
