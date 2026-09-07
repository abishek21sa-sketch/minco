# Phase 2 acceptance

Phase 2 requires environment-specific validation that the clean build environment cannot perform without the Flutter SDK and DuckDB/Polars packages.

## Backend/data gate

```powershell
python -m pip install -e ".[phase2,analytics,solver,dev,llm]"
python -m src.system.run_phase2_backend_acceptance
```

This verifies DuckDB ingestion, Parquet export, Polars replay analytics and an optimized workstation plan. The gate performs a Julia project preflight (`JuMP`, `Gurobi`, and `JSON3`) before selecting the primary solver. A Julia executable with an uninstantiated project is reported as unavailable and the plan uses the SciPy/HiGHS verification oracle with the readiness reason preserved in the report.

Latest canonical-package gate result (2026-09-04): `passed`; 459 events round-tripped through DuckDB, Parquet and Polars read all 459 events, and the plan was optimal. Julia was detected but reported `dependencies_unavailable`, so the plan used the SciPy/HiGHS verification oracle. The JSON evidence is written to `results/validation/phase2_backend_acceptance_report.json`.

## Native workstation gate

```powershell
.\scripts\bootstrap_flutter_windows.ps1
```

The script enables Flutter Windows desktop, regenerates only platform runner files, restores the governed MINCO Dart source, then runs:

- `flutter pub get`
- `flutter analyze`
- `flutter test`
- `flutter build windows --release`

## Runtime smoke

Terminal 1:

```powershell
.\scripts\start_phase2_runtime.ps1
```

Terminal 2:

```powershell
.\scripts\start_workstation_windows.ps1
```

The native workstation must show Runtime connected and allow a plan to be optimized and stress-tested.
