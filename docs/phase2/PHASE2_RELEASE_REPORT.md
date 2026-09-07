# MINCO Phase 2 Release Report — 0.8.0a1

## Disposition

**CANDIDATE FOR WINDOWS ACCEPTANCE.** Phase 1 is locked: the licensed Windows run reported Julia/JuMP/Gurobi `OPTIMAL`, objective `12.86`, relative gap `0.0`, exact agreement with the independent HiGHS oracle, and Progressive Hedging acceptance passed.

Phase 2 adds the native hospital product architecture without discarding the validated stochastic engine.

## Implemented in this release

- Flutter/Dart native workstation source with five hospital-specific operational surfaces.
- gRPC + Protocol Buffers Python runtime; REST/FastAPI is no longer the primary product boundary.
- canonical hospital event schema with source provenance and strict field validation.
- synthetic historical replay generator plus time-indexed state reconstruction.
- DuckDB operational lake and ordered ZSTD Parquet export adapter.
- Polars lazy Parquet analytics adapter.
- patient-level discrete-event simulation with seeded common-random-number intervention comparison and patient-conservation checks.
- stochastic planning runtime wired to Julia/JuMP/Gurobi when requested; HiGHS remains a verification fallback.
- adaptive Anthropic Claude evidence-review routing; external API execution remains disabled by default.
- Windows scripts for computational runtime, Flutter bootstrap/build and workstation launch.

## Build-environment validation before packaging

- full Python suite: 93 passed, 3 skipped because `gurobipy` is unavailable in the packaging environment;
- Phase-2 suite: 9 passed;
- Phase-1 mathematical regression gate: 16/16 passed;
- Phase-2 core gate: 11/11 passed;
- Python bytecode compilation: passed;
- gRPC/Protobuf roundtrip: passed;
- DES conservation: passed;
- stochastic HiGHS plan feasibility: passed.

## Environmental gates intentionally not claimed here

The packaging environment does not contain the Windows Flutter SDK, Julia, Gurobi, DuckDB, or Polars. Therefore this report does **not** claim native Windows compilation or DuckDB/Polars execution in the packaging environment.

The Windows acceptance machine must run:

1. `python -m pip install -e ".[phase2,analytics,solver,dev,llm]"`
2. `python -m src.system.run_phase2_backend_acceptance`
3. `scripts\bootstrap_flutter_windows.ps1`
4. runtime + workstation smoke.

## Evidence boundary

All bundled operational events and platform results remain synthetic/replay evidence. Real-hospital integration and external clinical/operational validation remain pending.

## Exact artifact clean-extraction verification

The distribution ZIP was extracted into a separate clean directory and validated from that extracted tree:

- Phase-2 tests: **9 passed**;
- full Python suite: **93 passed, 3 skipped** (`gurobipy` unavailable in packaging environment);
- Phase-1 mathematical regression gate: **16/16 passed**;
- Phase-2 core platform gate: **11/11 passed**;
- package version: `0.8.0a1`;
- Python compilation: passed;
- archive hygiene before validation: no `.venv`, `.pyc`, `__pycache__`, audit database or Flutter build directory.

The remaining gates are specifically Windows/toolchain dependent and are listed above rather than being claimed as passed.
