"""Scenario catalog and governed analytical evaluation service."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config.loader import load_healthcare_instance
from src.config.paths import AUDIT_DB_PATH, DATA_DIR
from src.contracts.common import EvidenceReference, ServiceMetadata
from src.contracts.metrics import OperationalMetrics
from src.contracts.scenario import (
    ScenarioDefinition,
    ScenarioEvaluationRequest,
    ScenarioEvaluationResponse,
    ScenarioParameters,
)
from src.storage.audit_repository import log_run_manifest, log_whatif_run
from src.validation.run_manifest import RUN_MANIFEST_DIR, write_run_manifest


class ScenarioNotFoundError(ValueError):
    pass


ScenarioRunner = Callable[[Any, ScenarioParameters, int], dict[str, Any]]


def _default_runner(instance: Any, parameters: ScenarioParameters, n_replications: int) -> dict[str, Any]:
    from dashboard.live_sandbox import run_live_whatif

    return run_live_whatif(
        instance=instance,
        icu_bed_delta=parameters.icu_bed_delta,
        transfer_capacity_multiplier=parameters.transfer_capacity_multiplier,
        demand_surge_multiplier=parameters.demand_surge_multiplier,
        n_replications=n_replications,
    )


BUILT_IN_SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        scenario_id="baseline",
        name="Baseline coordinated operation",
        description="Current synthetic reference network with no exogenous adjustment.",
        parameters=ScenarioParameters(),
    ),
    ScenarioDefinition(
        scenario_id="flu_surge",
        name="Seasonal demand surge",
        description="Forty-percent network-wide arrival increase.",
        parameters=ScenarioParameters(demand_surge_multiplier=1.40),
    ),
    ScenarioDefinition(
        scenario_id="mass_casualty",
        name="Mass-casualty pressure",
        description="Eighty-percent demand increase with twenty-percent transfer degradation.",
        parameters=ScenarioParameters(
            transfer_capacity_multiplier=0.80,
            demand_surge_multiplier=1.80,
        ),
    ),
    ScenarioDefinition(
        scenario_id="transfer_failure",
        name="Transfer network failure",
        description="Ninety-percent reduction in inter-hospital transfer availability.",
        parameters=ScenarioParameters(transfer_capacity_multiplier=0.10),
    ),
    ScenarioDefinition(
        scenario_id="icu_closure",
        name="ICU capacity closure",
        description="Fifteen ICU beds removed across the synthetic network.",
        parameters=ScenarioParameters(icu_bed_delta=-15),
    ),
)


class ScenarioService:
    """Stable service boundary around simulation and optimization execution."""

    service_name = "scenario_service"

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        *,
        runner: ScenarioRunner | None = None,
        db_path: Path = AUDIT_DB_PATH,
        manifest_dir: Path = RUN_MANIFEST_DIR,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.runner = runner or _default_runner
        self.db_path = Path(db_path)
        self.manifest_dir = Path(manifest_dir)
        self._catalog = {item.scenario_id: item for item in BUILT_IN_SCENARIOS}

    def list_scenarios(self) -> list[ScenarioDefinition]:
        return [item.model_copy(deep=True) for item in BUILT_IN_SCENARIOS]

    def get_scenario(self, scenario_id: str) -> ScenarioDefinition:
        try:
            return self._catalog[scenario_id].model_copy(deep=True)
        except KeyError as exc:
            raise ScenarioNotFoundError(f"Unknown scenario_id: {scenario_id}") from exc

    def resolve_scenario(self, request: ScenarioEvaluationRequest) -> ScenarioDefinition:
        if request.scenario_id == "custom":
            assert request.overrides is not None
            return ScenarioDefinition(
                scenario_id="custom",
                name="Custom operational scenario",
                description="Caller-supplied capacity, transfer, and demand adjustments.",
                parameters=request.overrides,
                built_in=False,
            )
        scenario = self.get_scenario(request.scenario_id)
        if request.overrides is not None:
            scenario = scenario.model_copy(
                update={
                    "parameters": request.overrides,
                    "description": f"{scenario.description} Full parameter override supplied by caller.",
                    "built_in": False,
                }
            )
        return scenario

    def analyze(
        self,
        scenario: ScenarioDefinition,
        n_replications: int,
    ) -> OperationalMetrics:
        instance = load_healthcare_instance(self.data_dir)
        result = self.runner(instance, scenario.parameters, n_replications)
        return OperationalMetrics.from_mapping(result)

    def evaluate(self, request: ScenarioEvaluationRequest) -> ScenarioEvaluationResponse:
        scenario = self.resolve_scenario(request)
        metrics = self.analyze(scenario, request.n_replications)
        parameters = scenario.parameters

        run_id = log_whatif_run(
            icu_bed_delta=parameters.icu_bed_delta,
            transfer_multiplier=parameters.transfer_capacity_multiplier,
            demand_multiplier=parameters.demand_surge_multiplier,
            n_replications=request.n_replications,
            result=metrics.model_dump(),
            surface=request.context.source,
            db_path=self.db_path,
        )
        manifest_path, manifest_hash = write_run_manifest(
            run_id=run_id,
            run_type="service_scenario_evaluation",
            parameters={
                "scenario": scenario.model_dump(mode="json"),
                "n_replications": request.n_replications,
                "execution_context": request.context.model_dump(mode="json"),
            },
            input_paths=[self.data_dir / "base_instance", self.data_dir / "transitions"],
            metrics=metrics.model_dump(),
            model_info={
                "simulation": "stochastic state-transition twin",
                "optimizer": "continuous network capacity LP",
                "solver": "Gurobi",
            },
            notes=[
                "Executed through the Level 3 scenario-service contract.",
                "Synthetic Meridian reference case; not real hospital evidence.",
            ],
            manifest_dir=self.manifest_dir,
        )
        log_run_manifest(run_id, manifest_path, manifest_hash, db_path=self.db_path)

        return ScenarioEvaluationResponse(
            run_id=run_id,
            scenario=scenario,
            metrics=metrics,
            metadata=ServiceMetadata(
                service=self.service_name,
                correlation_id=request.context.correlation_id,
                evidence=EvidenceReference(path=str(manifest_path), sha256=manifest_hash),
            ),
        )
