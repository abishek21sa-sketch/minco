from __future__ import annotations

from typing import Dict, Tuple

import gurobipy as gp
import pandas as pd

from src.config.dataclasses import HealthcareInstance
from src.markov.elective_response import build_c3_elective_response_table
from src.markov.forecast_engine import run_forecast
from src.markov.forecast_validation import validate_forecast_against_capacity
from src.optimizer.constraints import (
    add_extended_capacity_constraints,
    add_minimal_capacity_constraints,
    add_network_capacity_constraints,
)
from src.optimizer.objective import (
    set_extended_capacity_objective,
    set_minimal_capacity_objective,
    set_network_capacity_objective,
)
from src.optimizer.transfer_utils import (
    build_allowed_transfer_arcs,
    build_transfer_capacity_lookup,
    build_transfer_cost_lookup,
)
from src.optimizer.variables import (
    add_capacity_response_variables,
    add_extended_capacity_response_variables,
    add_network_capacity_response_variables,
    extract_variable_keys,
)


def _build_instance_without_baseline_c3_arrivals(
    instance: HealthcareInstance,
) -> HealthcareInstance:
    """
    Return a copy of the instance where baseline c3 arrivals are removed.

    This lets the extended and network optimizers treat elective c3 inflow
    as fully decision-controlled through e[h,t].
    """
    arrivals_df = instance.arrivals.df.copy()
    arrivals_wo_c3 = arrivals_df[arrivals_df["cohort"] != "c3"].copy()

    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=instance.capacities,
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=instance.transfer_lanes,
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=type(instance.arrivals)(df=arrivals_wo_c3),
        transitions=instance.transitions,
    )


def build_minimal_capacity_model(
    instance: HealthcareInstance,
    model_name: str = "minimal_capacity_response",
) -> Tuple[gp.Model, Dict[str, gp.tupledict], pd.DataFrame]:
    """
    Build the minimal capacity-response optimization model.
    """
    forecast_outputs = run_forecast(instance)
    validated_forecast_df = validate_forecast_against_capacity(
        resource_demand_df=forecast_outputs["resource_demand"],
        instance=instance,
    )

    days, hospitals, resources = extract_variable_keys(validated_forecast_df)

    model = gp.Model(model_name)

    variables = add_capacity_response_variables(
        model=model,
        days=days,
        hospitals=hospitals,
        resources=resources,
    )

    set_minimal_capacity_objective(
        model=model,
        variables=variables,
        validated_forecast_df=validated_forecast_df,
        costs_df=instance.costs.df,
    )

    add_minimal_capacity_constraints(
        model=model,
        variables=variables,
        validated_forecast_df=validated_forecast_df,
    )

    return model, variables, validated_forecast_df


def solve_minimal_capacity_model(
    instance: HealthcareInstance,
    model_name: str = "minimal_capacity_response",
    output_flag: int = 1,
) -> Tuple[gp.Model, Dict[str, gp.tupledict], pd.DataFrame]:
    """
    Build and solve the minimal capacity-response optimization model.
    """
    model, variables, validated_forecast_df = build_minimal_capacity_model(
        instance=instance,
        model_name=model_name,
    )

    model.setParam("OutputFlag", output_flag)
    model.optimize()

    return model, variables, validated_forecast_df


def build_extended_capacity_model(
    instance: HealthcareInstance,
    model_name: str = "extended_capacity_response",
) -> Tuple[gp.Model, Dict[str, gp.tupledict], pd.DataFrame, pd.DataFrame]:
    """
    Build the extended capacity-response model with elective control.

    Returns
    -------
    tuple
        (model, variables, validated_forecast_df, elective_response_df)
    """
    instance_without_c3 = _build_instance_without_baseline_c3_arrivals(instance)

    forecast_outputs = run_forecast(instance_without_c3)
    validated_forecast_df = validate_forecast_against_capacity(
        resource_demand_df=forecast_outputs["resource_demand"],
        instance=instance,
    )

    elective_response_df = build_c3_elective_response_table(instance)

    days, hospitals, resources = extract_variable_keys(validated_forecast_df)

    model = gp.Model(model_name)

    variables = add_extended_capacity_response_variables(
        model=model,
        days=days,
        hospitals=hospitals,
        resources=resources,
    )

    set_extended_capacity_objective(
        model=model,
        variables=variables,
        validated_forecast_df=validated_forecast_df,
        elective_bounds_df=instance.elective_bounds.df,
        costs_df=instance.costs.df,
    )

    add_extended_capacity_constraints(
        model=model,
        variables=variables,
        validated_forecast_df=validated_forecast_df,
        elective_bounds_df=instance.elective_bounds.df,
        elective_response_df=elective_response_df,
    )

    return model, variables, validated_forecast_df, elective_response_df


def solve_extended_capacity_model(
    instance: HealthcareInstance,
    model_name: str = "extended_capacity_response",
    output_flag: int = 1,
) -> Tuple[gp.Model, Dict[str, gp.tupledict], pd.DataFrame, pd.DataFrame]:
    """
    Build and solve the extended capacity-response model with elective control.
    """
    model, variables, validated_forecast_df, elective_response_df = (
        build_extended_capacity_model(
            instance=instance,
            model_name=model_name,
        )
    )

    model.setParam("OutputFlag", output_flag)
    model.optimize()

    return model, variables, validated_forecast_df, elective_response_df


def build_network_capacity_model(
    instance: HealthcareInstance,
    model_name: str = "network_capacity_response",
) -> Tuple[
    gp.Model,
    Dict[str, gp.tupledict],
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[str, str]],
    dict[tuple[str, str], float],
]:
    """
    Build the networked capacity-response model with:
    - elective control
    - ICU transfers

    Returns
    -------
    tuple
        (
            model,
            variables,
            validated_forecast_df,
            elective_response_df,
            arcs,
            transfer_cost_lookup,
        )
    """
    instance_without_c3 = _build_instance_without_baseline_c3_arrivals(instance)

    forecast_outputs = run_forecast(instance_without_c3)
    validated_forecast_df = validate_forecast_against_capacity(
        resource_demand_df=forecast_outputs["resource_demand"],
        instance=instance,
    )

    elective_response_df = build_c3_elective_response_table(instance)

    arcs = build_allowed_transfer_arcs(instance)
    transfer_cost_lookup = build_transfer_cost_lookup(
        instance=instance,
        include_generic_cost=True,
    )
    transfer_capacity_lookup = build_transfer_capacity_lookup(instance)

    days, hospitals, resources = extract_variable_keys(validated_forecast_df)

    model = gp.Model(model_name)

    variables = add_network_capacity_response_variables(
        model=model,
        days=days,
        hospitals=hospitals,
        resources=resources,
        arcs=arcs,
    )

    set_network_capacity_objective(
        model=model,
        variables=variables,
        validated_forecast_df=validated_forecast_df,
        elective_bounds_df=instance.elective_bounds.df,
        transfer_cost_lookup=transfer_cost_lookup,
        days=days,
        costs_df=instance.costs.df,
    )

    add_network_capacity_constraints(
        model=model,
        variables=variables,
        validated_forecast_df=validated_forecast_df,
        elective_bounds_df=instance.elective_bounds.df,
        elective_response_df=elective_response_df,
        arcs=arcs,
        transfer_capacity_lookup=transfer_capacity_lookup,
    )

    return (
        model,
        variables,
        validated_forecast_df,
        elective_response_df,
        arcs,
        transfer_cost_lookup,
    )


def solve_network_capacity_model(
    instance: HealthcareInstance,
    model_name: str = "network_capacity_response",
    output_flag: int = 1,
) -> Tuple[
    gp.Model,
    Dict[str, gp.tupledict],
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[str, str]],
    dict[tuple[str, str], float],
]:
    """
    Build and solve the networked capacity-response model.
    """
    (
        model,
        variables,
        validated_forecast_df,
        elective_response_df,
        arcs,
        transfer_cost_lookup,
    ) = build_network_capacity_model(
        instance=instance,
        model_name=model_name,
    )

    model.setParam("OutputFlag", output_flag)
    model.optimize()

    return (
        model,
        variables,
        validated_forecast_df,
        elective_response_df,
        arcs,
        transfer_cost_lookup,
    )