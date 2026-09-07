from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from datetime import datetime

import pandas as pd
import streamlit as st

from src.agent.rule_policy import AgentState
from src.agent.supervisor_agent import explain_supervisor_decision

from live_sandbox import load_base_instance, parse_whatif_question, run_live_whatif
from src.agent.agent_pipeline import run_agent_pipeline
from src.analysis.stress_classification import classify_operating_regime, classify_stress_label
from src.decision_math.flow_cvar_decision import build_flow_cvar_decision
from src.services.control_tower_service import ControlTowerService
from src.services.operational_history_service import OperationalHistoryService
from src.services.release_readiness_service import ReleaseReadinessService

st.set_page_config(
    page_title="MINCO - Healthcare Operations Command Center",
    page_icon="🏥",
    layout="wide",
)

RESULTS_DIR = ROOT_DIR / "results"

HOSPITAL_NAMES = {
    "H1": "Meridian University Medical Center",
    "H2": "Meridian North Regional",
    "H3": "Meridian Community Hospital",
}


def hospital_label(hospital_id: str | None) -> str:
    if hospital_id is None:
        return "Unknown"
    return HOSPITAL_NAMES.get(hospital_id, hospital_id)


@st.cache_resource(show_spinner=False)
def get_base_instance():
    return load_base_instance(ROOT_DIR / "data")


@st.cache_data(show_spinner=False)
def cached_live_whatif(
    icu_bed_delta: int,
    transfer_capacity_multiplier: float,
    demand_surge_multiplier: float,
    n_replications: int,
):
    instance = get_base_instance()
    return run_live_whatif(
        instance=instance,
        icu_bed_delta=icu_bed_delta,
        transfer_capacity_multiplier=transfer_capacity_multiplier,
        demand_surge_multiplier=demand_surge_multiplier,
        n_replications=n_replications,
    )


# ============================================================
# DATA LOADING
# ============================================================

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_all_data() -> dict[str, pd.DataFrame]:
    return {
        "agent_policy_dataset": _safe_read_csv(RESULTS_DIR / "agent_policy_dataset.csv"),
        "scenario_suite_summary": _safe_read_csv(RESULTS_DIR / "scenario_suite_summary.csv"),
        "scenario_suite_replications": _safe_read_csv(RESULTS_DIR / "scenario_suite_replications.csv"),
        "policy_comparison_summary": _safe_read_csv(RESULTS_DIR / "policy_comparison_combined_summary.csv"),
        "policy_comparison_replications": _safe_read_csv(RESULTS_DIR / "policy_comparison_combined_replications.csv"),
        "hospital_bottleneck_summary": _safe_read_csv(RESULTS_DIR / "hospital_bottleneck_summary.csv"),
        "hospital_bottleneck_decomposition": _safe_read_csv(RESULTS_DIR / "hospital_bottleneck_decomposition.csv"),
        "hospital_time_series_outputs": _safe_read_csv(RESULTS_DIR / "hospital_time_series_outputs.csv"),
        "hospital_time_series_summary": _safe_read_csv(RESULTS_DIR / "hospital_time_series_summary.csv"),
    }


@st.cache_data(show_spinner=False, ttl=300)
def cached_control_tower_snapshot() -> dict:
    """Load the enterprise read model once per five-minute operator session."""
    return ControlTowerService(
        operational_history=OperationalHistoryService()
    ).snapshot().model_dump(mode="json")


@st.cache_data(show_spinner=False, ttl=300)
def cached_release_readiness() -> dict:
    """Load the evidence-backed deployment scorecard for the operator session."""
    return ReleaseReadinessService(results_dir=RESULTS_DIR).snapshot().model_dump(mode="json")


# ============================================================
# HELPERS
# ============================================================

def choose_scenario(data: dict[str, pd.DataFrame]) -> str:
    df = data["scenario_suite_summary"]
    if df.empty:
        return "baseline"

    scenarios = df["scenario"].dropna().unique().tolist()
    default_idx = scenarios.index("baseline") if "baseline" in scenarios else 0

    return st.sidebar.selectbox(
        "Scenario",
        options=scenarios,
        index=default_idx,
    )


def get_scenario_rows(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[df["scenario"] == scenario].copy()


def get_reference_row(df: pd.DataFrame, scenario: str) -> pd.Series | None:
    rows = get_scenario_rows(df, scenario)
    if rows.empty:
        return None

    opt = rows[rows["policy_name"] == "optimized_network"]
    if not opt.empty:
        return opt.iloc[0]

    return rows.iloc[0]


def get_real_bottleneck_summary(
    hospital_summary_df: pd.DataFrame,
    scenario: str,
    policy_name: str,
) -> pd.DataFrame:
    if hospital_summary_df.empty:
        return pd.DataFrame()

    sdf = hospital_summary_df[
        (hospital_summary_df["scenario"] == scenario)
        & (hospital_summary_df["policy_name"] == policy_name)
    ].copy()

    if sdf.empty:
        return pd.DataFrame()

    score_col = "bottleneck_score_mean"
    total_score = float(sdf[score_col].sum()) if score_col in sdf.columns else 0.0

    if total_score > 0:
        sdf["contribution_share"] = (sdf[score_col] / total_score).round(3)
    else:
        sdf["contribution_share"] = 0.0

    def severity_label(share: float) -> str:
        if share > 0.45:
            return "High"
        if share > 0.25:
            return "Moderate"
        return "Low"

    sdf["severity"] = sdf["contribution_share"].apply(severity_label)

    return sdf.sort_values(score_col, ascending=False).reset_index(drop=True)


def get_real_bottleneck_decomposition(
    decomposition_df: pd.DataFrame,
    scenario: str,
    policy_name: str,
    hospital_id: str,
) -> pd.DataFrame:
    if decomposition_df.empty:
        return pd.DataFrame()

    sdf = decomposition_df[
        (decomposition_df["scenario"] == scenario)
        & (decomposition_df["policy_name"] == policy_name)
        & (decomposition_df["hospital_id"] == hospital_id)
    ].copy()

    if sdf.empty:
        return pd.DataFrame()

    return sdf.reset_index(drop=True)


def build_component_share_table(decomp_row: pd.Series) -> pd.DataFrame:
    total_score = float(decomp_row.get("bottleneck_score_mean", 0.0))

    component_map = {
        "Utilization": float(decomp_row.get("util_component_mean", 0.0)),
        "Unsafe excess": float(decomp_row.get("unsafe_component_mean", 0.0)),
        "Overflow": float(decomp_row.get("overflow_component_mean", 0.0)),
        "Unsafe events": float(decomp_row.get("event_component_mean", 0.0)),
        "Transfer": float(decomp_row.get("transfer_component_mean", 0.0)),
    }

    rows = []
    for comp, value in component_map.items():
        share = (value / total_score) if total_score > 0 else 0.0
        rows.append(
            {
                "component": comp,
                "value": round(value, 3),
                "share": round(share, 3),
            }
        )

    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


def get_hospital_time_series(
    hospital_ts_df: pd.DataFrame,
    scenario: str,
    policy_name: str,
    hospital_id: str,
) -> pd.DataFrame:
    if hospital_ts_df.empty:
        return pd.DataFrame()

    sdf = hospital_ts_df[
        (hospital_ts_df["scenario"] == scenario)
        & (hospital_ts_df["policy_name"] == policy_name)
        & (hospital_ts_df["hospital_id"] == hospital_id)
    ].copy()

    if sdf.empty:
        return pd.DataFrame()

    # Aggregate across replications to get clean per-day temporal profile
    grouped = (
        sdf.groupby(["day", "hospital_id"], as_index=False)
        .agg(
            mean_bottleneck_score=("bottleneck_score", "mean"),
            max_bottleneck_score=("bottleneck_score", "max"),
            mean_unsafe_excess=("unsafe_excess", "mean"),
            mean_overflow_excess=("overflow_excess", "mean"),
            mean_surge_gap=("surge_gap", "mean"),
            mean_utilization=("max_utilization", "mean"),
            mean_transfer_activity=("transfer_activity", "mean"),
            mean_num_unsafe_rows=("num_unsafe_rows", "mean"),
        )
    )

    return grouped.sort_values("day").reset_index(drop=True)


def get_hospital_time_series_summary(
    hospital_ts_summary_df: pd.DataFrame,
    scenario: str,
    policy_name: str,
    hospital_id: str,
) -> pd.DataFrame:
    if hospital_ts_summary_df.empty:
        return pd.DataFrame()

    sdf = hospital_ts_summary_df[
        (hospital_ts_summary_df["scenario"] == scenario)
        & (hospital_ts_summary_df["policy_name"] == policy_name)
        & (hospital_ts_summary_df["hospital_id"] == hospital_id)
    ].copy()

    if sdf.empty:
        return pd.DataFrame()

    return sdf.reset_index(drop=True)


# ============================================================
# HEADER
# ============================================================

def render_header() -> None:
    st.title("MINCO - Healthcare Operations Command Center")
    st.caption("Synthetic Meridian reference network: forecasting, uncertainty-aware optimization, and stochastic policy evaluation")

    report_path = RESULTS_DIR / "system_pipeline" / "full_pipeline_decision_report.json"
    if report_path.exists():
        last_run = datetime.fromtimestamp(report_path.stat().st_mtime)
        last_run_str = last_run.strftime("%Y-%m-%d %H:%M")
        st.caption(f"Last decision cycle: {last_run_str}")


def render_missing_data_notice(data: dict[str, pd.DataFrame]) -> None:
    missing = [name for name, df in data.items() if df.empty]
    if missing:
        st.warning(
            "Some result files are missing or empty. Run the experiment pipeline first for full functionality.\n\n"
            f"Missing/empty: {', '.join(missing)}"
        )


# ============================================================
# SYSTEM STATUS
# ============================================================

def render_system_status(summary_df: pd.DataFrame, scenario: str) -> None:
    row = get_reference_row(summary_df, scenario)
    if row is None:
        return

    st.subheader("System Status")

    max_util = float(row["max_utilization_ratio_mean"])
    overflow = float(row["total_overflow_excess_mean"])
    unsafe_rows = float(row["num_unsafe_rows_mean"])

    icu_status, overflow_status, bottleneck = classify_stress_label(
        max_util=max_util,
        overflow=overflow,
        unsafe_rows=unsafe_rows,
    )

    c1, c2, c3 = st.columns(3)

    if icu_status == "Critical":
        c1.error(f"ICU Status: {icu_status}")
    elif icu_status == "Stressed":
        c1.warning(f"ICU Status: {icu_status}")
    else:
        c1.success(f"ICU Status: {icu_status}")

    if overflow_status == "High":
        c2.error(f"Overflow Risk: {overflow_status}")
    elif overflow_status == "Moderate":
        c2.warning(f"Overflow Risk: {overflow_status}")
    else:
        c2.success(f"Overflow Risk: {overflow_status}")

    if "Severe" in bottleneck or "H3" in bottleneck:
        c3.error(f"Primary Bottleneck: {bottleneck}")
    elif "Local" in bottleneck:
        c3.warning(f"Primary Bottleneck: {bottleneck}")
    else:
        c3.info(f"Primary Bottleneck: {bottleneck}")


# ============================================================
# KPI OVERVIEW
# ============================================================

def render_kpi_cards(summary_df: pd.DataFrame, scenario: str) -> None:
    row = get_reference_row(summary_df, scenario)
    if row is None:
        st.info("No KPI summary available for the selected scenario.")
        return

    st.subheader("KPI Overview")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("ICU / Ward Unsafe Excess", f"{row['total_unsafe_excess_mean']:.2f}")
    c2.metric("Max Utilization", f"{row['max_utilization_ratio_mean']:.3f}")
    c3.metric("Unsafe Events", f"{row['num_unsafe_rows_mean']:.2f}")
    c4.metric("Blocked Arrivals", f"{row['total_blocked_arrivals_mean']:.2f}")
    c5.metric("Overflow", f"{row['total_overflow_excess_mean']:.2f}")
    c6.metric("Surge Gap", f"{row['total_surge_gap_mean']:.2f}")


# ============================================================
# TAB 1: SYSTEM FORECAST
# ============================================================

def render_forecast_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("Forecast / Scenario Summary")

    df = data["scenario_suite_summary"]
    if df.empty:
        st.info("Scenario summary file not available.")
        return

    sdf = get_scenario_rows(df, scenario)
    if sdf.empty:
        st.info("No rows found for the selected scenario.")
        return

    display_cols = [
        "policy_name",
        "total_unsafe_excess_mean",
        "max_utilization_ratio_mean",
        "num_unsafe_rows_mean",
        "total_blocked_arrivals_mean",
        "total_overflow_excess_mean",
        "total_surge_gap_mean",
    ]

    styled = (
        sdf[display_cols]
        .sort_values("total_unsafe_excess_mean")
        .style
        .highlight_min(subset=["total_unsafe_excess_mean"], color="lightgreen")
        .format(
            {
                "total_unsafe_excess_mean": "{:.2f}",
                "max_utilization_ratio_mean": "{:.3f}",
                "num_unsafe_rows_mean": "{:.2f}",
                "total_blocked_arrivals_mean": "{:.2f}",
                "total_overflow_excess_mean": "{:.2f}",
                "total_surge_gap_mean": "{:.2f}",
            }
        )
    )

    st.dataframe(styled, use_container_width=True)

    best = sdf.sort_values("total_unsafe_excess_mean").iloc[0]
    worst = sdf.sort_values("total_unsafe_excess_mean", ascending=False).iloc[0]

    st.caption(
        f"For `{scenario}`, the lowest mean unsafe excess is under `{best['policy_name']}` "
        f"at `{best['total_unsafe_excess_mean']:.2f}`, while the highest is under "
        f"`{worst['policy_name']}` at `{worst['total_unsafe_excess_mean']:.2f}`."
    )


# ============================================================
# TAB 2: DECISION ENGINE
# ============================================================

def render_decision_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("Decision Engine")

    dataset_df = data["agent_policy_dataset"]
    summary_df = data["scenario_suite_summary"]

    if dataset_df.empty:
        st.info("Agent policy dataset not available.")
        return

    sdf = dataset_df[dataset_df["scenario"] == scenario].copy()
    if sdf.empty:
        st.info("No agent dataset rows found for the selected scenario.")
        return

    default = sdf.sort_values("unsafe_excess").iloc[0]

    with st.expander("Control Inputs", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        max_util = c1.number_input(
            "Forecast max utilization",
            min_value=0.0,
            value=float(default["max_utilization"]),
            step=0.01,
        )

        unsafe_excess = c2.number_input(
            "Forecast unsafe excess",
            min_value=0.0,
            value=float(default["unsafe_excess"]),
            step=0.1,
        )

        unsafe_rows = c3.number_input(
            "Forecast unsafe rows",
            min_value=0.0,
            value=float(default["unsafe_rows"]),
            step=0.1,
        )

        transfer_available = c4.checkbox(
            "Transfers available",
            value=(scenario != "transfer_disabled"),
        )

    state = AgentState(
        scenario=scenario,
        forecast_max_utilization=max_util,
        forecast_unsafe_excess=unsafe_excess,
        forecast_unsafe_rows=unsafe_rows,
        transfer_available=transfer_available,
    )

    result = explain_supervisor_decision(state, dataset_df)
    final_policy = result["hybrid_result"]["final_policy"]

    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown("#### Recommended Policy")
        st.success(f"Final recommended policy: `{final_policy}`")

        st.markdown("#### Immediate Actions")

        if final_policy == "optimized_network":
            actions = [
                "Transfer patients across hospitals to reduce localized ICU overload.",
                "Throttle elective admissions in the most stressed units.",
                "Activate surge beds where network pressure remains elevated.",
            ]
        elif final_policy == "myopic_milp":
            actions = [
                "Use day-by-day optimization with moderate elective throttling.",
                "Apply limited or targeted transfers only where immediately beneficial.",
                "Monitor stressed hospitals closely for short-horizon overload.",
            ]
        elif final_policy == "no_transfer":
            actions = [
                "Activate local surge capacity immediately.",
                "Cancel or defer electives in the most constrained hospitals.",
                "Handle overflow locally because transfers are unavailable or ineffective.",
            ]
        elif final_policy == "local_only":
            actions = [
                "Operate locally without escalation to heavier network coordination.",
                "Accept moderate overload risk while preserving access.",
                "Monitor utilization and overflow indicators frequently.",
            ]
        else:
            actions = [
                "Apply the selected policy and monitor system conditions closely.",
                "Review overflow and unsafe capacity indicators.",
                "Escalate if utilization worsens.",
            ]

        for i, action in enumerate(actions, 1):
            st.write(f"{i}. {action}")

        st.markdown("#### Why this decision")
        if final_policy == "optimized_network":
            st.write(
                "This recommendation favors full network coordination because system-wide "
                "redistribution is expected to reduce unsafe overload more effectively than local action."
            )
        elif final_policy == "myopic_milp":
            st.write(
                "This recommendation favors short-term optimization because it preserves more access "
                "while still controlling immediate overload."
            )
        elif final_policy == "no_transfer":
            st.write(
                "This recommendation avoids inter-hospital transfers because they are unavailable "
                "or expected to be ineffective under the selected condition."
            )
        elif final_policy == "local_only":
            st.write(
                "This recommendation keeps the response local because the current condition does not justify "
                "a heavier coordinated intervention."
            )
        else:
            st.write("The decision is based on the hybrid agent reconciliation layer.")

    with right:
        st.markdown("#### Δ vs Baseline")

        baseline_rows = summary_df[
            (summary_df["scenario"] == "baseline")
            & (summary_df["policy_name"] == final_policy)
        ].copy()

        current_rows = summary_df[
            (summary_df["scenario"] == scenario)
            & (summary_df["policy_name"] == final_policy)
        ].copy()

        if baseline_rows.empty or current_rows.empty:
            st.info("Baseline comparison not available for the selected final policy.")
        else:
            b = baseline_rows.iloc[0]
            c = current_rows.iloc[0]

            st.metric(
                "Unsafe Excess Δ",
                f"{c['total_unsafe_excess_mean'] - b['total_unsafe_excess_mean']:+.2f}",
            )
            st.metric(
                "Max Utilization Δ",
                f"{c['max_utilization_ratio_mean'] - b['max_utilization_ratio_mean']:+.3f}",
            )
            st.metric(
                "Blocked Arrivals Δ",
                f"{c['total_blocked_arrivals_mean'] - b['total_blocked_arrivals_mean']:+.2f}",
            )

    with st.expander("Supervisor Explanation"):
        st.code(result["explanation"])

    with st.expander("Ranked Policy Options"):
        st.dataframe(
            result["ranking"][
                ["policy", "unsafe_excess", "unsafe_rows", "blocked_arrivals", "max_utilization", "score"]
            ],
            use_container_width=True,
        )


# ============================================================
# TAB 3: SCENARIO LAB
# ============================================================

def render_scenario_lab_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("Scenario Lab")

    df = data["scenario_suite_summary"]
    if df.empty:
        st.info("Scenario suite summary not available.")
        return

    metric = st.selectbox(
        "Metric",
        options=[
            "total_unsafe_excess_mean",
            "max_utilization_ratio_mean",
            "num_unsafe_rows_mean",
            "total_blocked_arrivals_mean",
            "total_overflow_excess_mean",
        ],
        index=0,
    )

    pivot = df.pivot(index="scenario", columns="policy_name", values=metric)
    st.dataframe(pivot, use_container_width=True)

    if scenario in pivot.index:
        st.caption(f"Selected scenario row: `{scenario}`")
        st.dataframe(pivot.loc[[scenario]], use_container_width=True)

    if scenario in pivot.index:
        best_policy = pivot.loc[scenario].idxmin()
        best_value = pivot.loc[scenario].min()
        st.caption(
            f"For metric `{metric}`, the best policy in `{scenario}` is "
            f"`{best_policy}` with value `{best_value:.3f}`."
        )

    st.markdown("---")
    st.markdown("#### Live What-If Sandbox")
    st.caption(
        "This re-solves the network optimization model live and re-runs stochastic "
        "simulation against your changes -- it is a real solve, not a lookup."
    )

    s1, s2, s3 = st.columns(3)
    icu_bed_delta = s1.slider("Add ICU beds (network-wide)", min_value=-10, max_value=40, value=0, step=2)
    transfer_pct = s2.slider("Transfer capacity change (%)", min_value=-50, max_value=100, value=0, step=5)
    demand_pct = s3.slider("Demand surge (%)", min_value=-25, max_value=75, value=0, step=5)

    precision = st.radio(
        "Precision",
        options=["Quick preview (10 reps, ~5s)", "Full precision (50 reps, matches baseline exactly)"],
        index=0,
        horizontal=True,
    )
    n_reps = 10 if precision.startswith("Quick") else 50

    run_clicked = st.button("Run live optimization", type="primary")

    if run_clicked:
        with st.spinner("Solving the network optimization model and running stochastic replications..."):
            result = cached_live_whatif(
                icu_bed_delta=icu_bed_delta,
                transfer_capacity_multiplier=1.0 + transfer_pct / 100.0,
                demand_surge_multiplier=1.0 + demand_pct / 100.0,
                n_replications=n_reps,
            )

        baseline_row = get_reference_row(data["scenario_suite_summary"], scenario)

        r1, r2, r3 = st.columns(3)

        unsafe_val = result.get("total_unsafe_excess", 0.0)
        blocked_val = result.get("total_blocked_arrivals", 0.0)
        util_val = result.get("max_utilization_ratio", 0.0)

        if baseline_row is not None:
            r1.metric(
                "Unsafe Excess",
                f"{unsafe_val:.2f}",
                delta=round(unsafe_val - float(baseline_row["total_unsafe_excess_mean"]), 2),
                delta_color="inverse",
            )
            r2.metric(
                "Blocked Arrivals",
                f"{blocked_val:.2f}",
                delta=round(blocked_val - float(baseline_row["total_blocked_arrivals_mean"]), 2),
                delta_color="inverse",
            )
            r3.metric(
                "Max Utilization",
                f"{util_val:.3f}",
                delta=round(util_val - float(baseline_row["max_utilization_ratio_mean"]), 3),
                delta_color="inverse",
            )
            st.caption(f"Deltas are versus the precomputed `{scenario}` baseline (optimized network policy).")
        else:
            r1.metric("Unsafe Excess", f"{unsafe_val:.2f}")
            r2.metric("Blocked Arrivals", f"{blocked_val:.2f}")
            r3.metric("Max Utilization", f"{util_val:.3f}")

        st.caption(
            f"Solved in {result.get('solve_runtime_seconds', 0.0):.2f}s across "
            f"{int(result.get('n_replications', 0))} stochastic replications."
        )

        if n_reps < 50:
            st.caption(
                "Quick preview uses fewer replications, so numbers will be close but not "
                "identical to the 50-rep baseline. Switch to Full precision for an exact comparison."
            )


# ============================================================
# TAB 4: POLICY BENCHMARK
# ============================================================

def render_policy_benchmark_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("Policy Benchmark")

    df = data["policy_comparison_summary"]
    if df.empty:
        st.info("Policy comparison summary not available.")
        return

    sdf = df[df["scenario"] == scenario].copy()
    if sdf.empty:
        st.info("No policy comparison rows found for selected scenario.")
        return

    metric = st.selectbox(
        "Comparison metric",
        options=[
            "total_unsafe_excess_mean",
            "max_utilization_ratio_mean",
            "num_unsafe_rows_mean",
        ],
        index=0,
        key="policy_metric",
    )

    display_cols = ["policy_name", metric]
    st.dataframe(
        sdf[display_cols].sort_values(metric),
        use_container_width=True,
    )

    best = sdf.sort_values(metric).iloc[0]
    worst = sdf.sort_values(metric, ascending=False).iloc[0]

    st.success(
        f"Best policy for `{metric}` in `{scenario}`: "
        f"`{best['policy_name']}` ({best[metric]:.3f})"
    )
    st.caption(
        f"`{best['policy_name']}` performs best on `{metric}`, while "
        f"`{worst['policy_name']}` performs worst for this scenario."
    )


# ============================================================
# TAB 5: AI COPILOT
# ============================================================

def render_ai_copilot_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("AI Copilot (Decision Intelligence)")

    dataset_df = data["agent_policy_dataset"]
    summary_df = data["scenario_suite_summary"]
    hospital_summary_df = data["hospital_bottleneck_summary"]
    decomposition_df = data["hospital_bottleneck_decomposition"]
    hospital_ts_df = data["hospital_time_series_outputs"]
    hospital_ts_summary_df = data["hospital_time_series_summary"]

    if dataset_df.empty or summary_df.empty:
        st.info("Required data not available.")
        return

    question = st.text_area(
        "Ask an operational question",
        value="Why is the system stressed in this scenario?",
        height=80,
    )

    instance_for_parsing = get_base_instance()
    total_icu_capacity = float(
        instance_for_parsing.capacities.df[
            instance_for_parsing.capacities.df["resource"].astype(str).str.upper() == "ICU"
        ]["base_capacity"].sum()
    )

    whatif_params = parse_whatif_question(question, total_icu_capacity=total_icu_capacity)

    if whatif_params is not None:
        st.markdown("### 🔴 Live What-If Answer")
        st.caption(
            f"Detected a what-if question -- interpreted as ICU beds "
            f"{whatif_params['icu_bed_delta']:+d}, transfer capacity x"
            f"{whatif_params['transfer_capacity_multiplier']:.2f}, demand x"
            f"{whatif_params['demand_surge_multiplier']:.2f}."
        )

        if st.button("Run live optimization for this question", key="copilot_whatif_button"):
            with st.spinner("Solving the network optimization model..."):
                whatif_result = cached_live_whatif(
                    icu_bed_delta=whatif_params["icu_bed_delta"],
                    transfer_capacity_multiplier=whatif_params["transfer_capacity_multiplier"],
                    demand_surge_multiplier=whatif_params["demand_surge_multiplier"],
                    n_replications=10,
                )

            baseline_row_for_copilot = get_reference_row(summary_df, scenario)
            wc1, wc2, wc3 = st.columns(3)

            wc_unsafe = whatif_result.get("total_unsafe_excess", 0.0)
            wc_blocked = whatif_result.get("total_blocked_arrivals", 0.0)
            wc_util = whatif_result.get("max_utilization_ratio", 0.0)

            if baseline_row_for_copilot is not None:
                wc1.metric(
                    "Unsafe Excess",
                    f"{wc_unsafe:.2f}",
                    delta=round(wc_unsafe - float(baseline_row_for_copilot["total_unsafe_excess_mean"]), 2),
                    delta_color="inverse",
                )
                wc2.metric(
                    "Blocked Arrivals",
                    f"{wc_blocked:.2f}",
                    delta=round(wc_blocked - float(baseline_row_for_copilot["total_blocked_arrivals_mean"]), 2),
                    delta_color="inverse",
                )
                wc3.metric(
                    "Max Utilization",
                    f"{wc_util:.3f}",
                    delta=round(wc_util - float(baseline_row_for_copilot["max_utilization_ratio_mean"]), 3),
                    delta_color="inverse",
                )
            else:
                wc1.metric("Unsafe Excess", f"{wc_unsafe:.2f}")
                wc2.metric("Blocked Arrivals", f"{wc_blocked:.2f}")
                wc3.metric("Max Utilization", f"{wc_util:.3f}")

            st.caption(
                f"Quick preview ({int(whatif_result.get('n_replications', 0))} reps, "
                f"{whatif_result.get('solve_runtime_seconds', 0.0):.1f}s) -- "
                f"use the Scenario Lab tab for a full-precision (50-rep) version of this same what-if."
            )

        st.markdown("---")
        st.caption("General system diagnosis for the current scenario follows below.")

    sdf = dataset_df[dataset_df["scenario"] == scenario].copy()
    sum_df = summary_df[summary_df["scenario"] == scenario].copy()

    if sdf.empty or sum_df.empty:
        st.info("No data for this scenario.")
        return

    best = sdf.sort_values("unsafe_excess").iloc[0]
    worst = sdf.sort_values("unsafe_excess", ascending=False).iloc[0]

    # Dashboard model reference row
    ref = sum_df[sum_df["policy_name"] == "optimized_network"]
    ref = ref.iloc[0] if not ref.empty else sum_df.iloc[0]

    max_util = float(ref["max_utilization_ratio_mean"])
    unsafe = float(ref["total_unsafe_excess_mean"])
    unsafe_rows = float(ref["num_unsafe_rows_mean"])
    overflow = float(ref["total_overflow_excess_mean"])

    regime = classify_operating_regime(max_util)

    if max_util > 1.7:
        diagnosis = "critical ICU saturation"
    elif max_util > 1.2:
        diagnosis = "system under capacity stress"
    else:
        diagnosis = "system operating within safe limits"

    if unsafe_rows > 4:
        bottleneck = "repeated ICU capacity violations across the network"
    elif overflow > 5:
        bottleneck = "overflow due to insufficient downstream capacity"
    elif max_util > 1.3:
        bottleneck = "localized ICU congestion"
    else:
        bottleneck = "no dominant bottleneck"

    policy_reasoning = {
        "optimized_network": "uses inter-hospital transfers to rebalance load and minimize global overload",
        "myopic_milp": "applies short-horizon optimization to control overload without aggressive intervention",
        "no_transfer": "forces local handling, increasing overload risk but avoiding transfer costs",
        "local_only": "maintains local autonomy with minimal coordination",
        "no_control": "takes no corrective action, leading to highest overload",
    }

    best_policy_reason = policy_reasoning.get(best["policy"], "data-driven selection")
    worst_policy_reason = policy_reasoning.get(worst["policy"], "lack of intervention")

    sdf = sdf.copy()
    best_unsafe = float(best["unsafe_excess"])
    sdf["regret"] = (sdf["unsafe_excess"] - best_unsafe).round(2)

    unsafe_delta = float(worst["unsafe_excess"]) - float(best["unsafe_excess"])
    blocked_delta = float(worst["blocked_arrivals"]) - float(best["blocked_arrivals"])
    util_delta = float(worst["max_utilization"]) - float(best["max_utilization"])

    counterfactuals = []
    if unsafe > 6:
        counterfactuals.append("Reducing elective admissions would likely lower unsafe capacity violations.")
    if max_util > 1.5:
        counterfactuals.append("Increasing transfers or activating surge capacity would reduce peak ICU utilization.")
    if overflow > 5:
        counterfactuals.append("Expanding downstream ward capacity would reduce ICU overflow pressure.")
    if not counterfactuals:
        counterfactuals.append("No major intervention appears necessary; the system is near stable operation.")

    # REAL bottleneck outputs based on best policy
    chosen_policy = best["policy"]
    real_bottleneck_df = get_real_bottleneck_summary(
        hospital_summary_df=hospital_summary_df,
        scenario=scenario,
        policy_name=chosen_policy,
    )

    top_hospital_id = None
    top_hospital_share = 0.0
    top_hospital_severity = "Unknown"

    if not real_bottleneck_df.empty:
        top_hospital_row = real_bottleneck_df.iloc[0]
        top_hospital_id = str(top_hospital_row["hospital_id"])
        top_hospital_share = float(top_hospital_row["contribution_share"])
        top_hospital_severity = str(top_hospital_row["severity"])

    real_decomp_df = (
        get_real_bottleneck_decomposition(
            decomposition_df=decomposition_df,
            scenario=scenario,
            policy_name=chosen_policy,
            hospital_id=top_hospital_id,
        )
        if top_hospital_id is not None
        else pd.DataFrame()
    )

    component_share_df = (
        build_component_share_table(real_decomp_df.iloc[0])
        if not real_decomp_df.empty
        else pd.DataFrame()
    )

    temporal_df = (
        get_hospital_time_series(
            hospital_ts_df=hospital_ts_df,
            scenario=scenario,
            policy_name=chosen_policy,
            hospital_id=top_hospital_id,
        )
        if top_hospital_id is not None
        else pd.DataFrame()
    )

    temporal_summary_df = (
        get_hospital_time_series_summary(
            hospital_ts_summary_df=hospital_ts_summary_df,
            scenario=scenario,
            policy_name=chosen_policy,
            hospital_id=top_hospital_id,
        )
        if top_hospital_id is not None
        else pd.DataFrame()
    )

    # Top compact summary
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Operating Regime", regime)
    s2.metric("Max Utilization", f"{max_util:.2f}")
    s3.metric("Unsafe Excess", f"{unsafe:.2f}")
    s4.metric("Overflow", f"{overflow:.2f}")

    # Executive summary
    st.markdown("### 🔷 Executive Decision Summary")
    if max_util < 1.2:
        st.success("System is stable → no aggressive intervention required.")
    elif max_util < 1.5:
        st.warning("System is under stress → moderate intervention recommended.")
    else:
        st.error("System is critical → aggressive coordination required.")

    st.markdown(
        f"Recommended strategy: **{chosen_policy}** "
        f"(reduces unsafe excess to **{best['unsafe_excess']:.2f}**)."
    )

    # Diagnosis and bottleneck
    left, right = st.columns([1.0, 1.2])

    with left:
        st.markdown("#### System Diagnosis")
        st.write(f"Diagnosis: {diagnosis}")
        st.write(f"Primary network-level driver: {bottleneck}")

        st.markdown("#### System Dynamics Insight")
        st.write(
            f"The system is currently in a **{regime}**, where patient inflow into critical states "
            "is not being fully offset by downstream transitions (ICU → ward → discharge). "
            "This creates accumulation pressure and drives unsafe capacity conditions."
        )
        st.write(
            f"At a max utilization of **{max_util:.2f}**, the Markov progression dynamics suggest "
            "that transition rates out of ICU are insufficient relative to inflow, causing congestion propagation."
        )

        twin_horizon = compute_digital_twin_horizon(
            hospital_ts_df=hospital_ts_df,
            scenario=scenario,
            policy_name=chosen_policy,
        )
        if not twin_horizon.empty:
            twin_peak_day = int(twin_horizon.loc[twin_horizon["worst"].idxmax(), "day"])
            twin_peak_value = float(twin_horizon["worst"].max())
            st.write(
                f"The 7-day digital twin projects worst-case network bottleneck risk peaking on "
                f"**day {twin_peak_day}** (bottleneck score {twin_peak_value:.2f}) under the "
                f"**{chosen_policy}** policy -- see the Stochastic Twin tab for the full trajectory."
            )

        st.markdown("#### Question")
        st.write(question)

    with right:
        st.markdown("#### Bottleneck Attribution (Real Outputs)")
        if real_bottleneck_df.empty:
            st.info("Hospital bottleneck summary not available for this scenario/policy.")
        else:
            display_cols = [
                "hospital_id",
                "bottleneck_score_mean",
                "unsafe_excess_mean",
                "overflow_excess_mean",
                "max_utilization_mean",
                "num_unsafe_rows_mean",
                "contribution_share",
                "severity",
            ]
            display_cols = [c for c in display_cols if c in real_bottleneck_df.columns]
            st.dataframe(real_bottleneck_df[display_cols], use_container_width=True)

            st.error(
                f"Primary bottleneck: {hospital_label(top_hospital_id)} "
                f"({top_hospital_share * 100:.1f}% of bottleneck score, severity: {top_hospital_severity})"
            )

    # Policy insights
    st.markdown("#### Policy Insights")
    p1, p2 = st.columns(2)
    with p1:
        st.success(
            f"Best policy: {best['policy']} | unsafe_excess={best['unsafe_excess']:.2f} | regret=0.00"
        )
        st.write(best_policy_reason)
    with p2:
        worst_regret = float(sdf[sdf["policy"] == worst["policy"]]["regret"].iloc[0])
        st.error(
            f"Worst policy: {worst['policy']} | unsafe_excess={worst['unsafe_excess']:.2f} | regret={worst_regret:.2f}"
        )
        st.write(worst_policy_reason)

    st.info(
        "Recommended action: prioritize transfer-enabled coordination while monitoring ICU utilization closely."
    )

    st.markdown("#### Optimization Perspective")
    st.write(
        f"The selected policy **{best['policy']}** minimizes system-wide overload by balancing capacity "
        "constraints across the hospital network."
    )
    st.write(
        "From the MILP perspective, this policy reduces the marginal cost of overload by using available "
        "capacity across hospitals and avoiding binding ICU constraints that would otherwise lead to unsafe states."
    )

    # Real decomposition
    st.markdown("#### Bottleneck Decomposition (Real Outputs)")
    if real_decomp_df.empty or component_share_df.empty:
        st.info("Hospital bottleneck decomposition not available for the selected scenario/policy.")
    else:
        dd1, dd2 = st.columns([1.1, 0.9])

        with dd1:
            st.dataframe(component_share_df, use_container_width=True)

        with dd2:
            top_component = component_share_df.iloc[0]
            st.metric("Top component", str(top_component["component"]))
            st.metric("Top component share", f"{top_component['share'] * 100:.1f}%")
            st.caption(
                f"For {hospital_label(top_hospital_id)}, the dominant bottleneck driver is "
                f"`{top_component['component']}`."
            )

    # Temporal bottleneck view
    st.markdown("#### Temporal Bottleneck View")
    if temporal_df.empty:
        st.info("Temporal hospital outputs not available for the selected scenario/policy.")
    else:
        tv1, tv2 = st.columns([1.15, 0.85])

        with tv1:
            display_temporal_cols = [
                "day",
                "mean_bottleneck_score",
                "max_bottleneck_score",
                "mean_unsafe_excess",
                "mean_overflow_excess",
                "mean_utilization",
                "mean_transfer_activity",
            ]
            display_temporal_cols = [c for c in display_temporal_cols if c in temporal_df.columns]
            st.dataframe(temporal_df[display_temporal_cols], use_container_width=True)

        with tv2:
            if not temporal_summary_df.empty:
                ts = temporal_summary_df.iloc[0]
                st.metric("Peak bottleneck day", int(ts["peak_bottleneck_day"]))
                st.metric("Peak bottleneck score", f"{ts['peak_bottleneck_score']:.2f}")
                st.metric("Days above safe utilization", int(ts["days_above_safe_utilization"]))
                st.metric("Mean daily bottleneck", f"{ts['mean_daily_bottleneck_score']:.2f}")
            else:
                st.info("Temporal bottleneck summary not available.")

        if not temporal_df.empty:
            peak_row = temporal_df.sort_values("mean_bottleneck_score", ascending=False).iloc[0]
            st.caption(
                f"For {hospital_label(top_hospital_id)}, the average bottleneck profile peaks around day "
                f"`{int(peak_row['day'])}` with mean bottleneck score "
                f"`{peak_row['mean_bottleneck_score']:.2f}`."
            )

    # Marginal impact
    st.markdown("#### Marginal Impact of Better Control")
    st.markdown(
        f"Switching from **{worst['policy']} → {best['policy']}** results in:"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Δ Unsafe Excess", f"{unsafe_delta:.2f}")
    m2.metric("Δ Blocked Arrivals", f"{blocked_delta:.2f}")
    m3.metric("Δ Max Utilization", f"{util_delta:.3f}")

    # Bottom two-column section
    b1, b2 = st.columns([1.1, 0.9])

    with b1:
        st.markdown("#### Policy Regret Analysis")
        regret_table = sdf[["policy", "unsafe_excess", "regret"]].sort_values(["regret", "unsafe_excess"])
        st.dataframe(regret_table, use_container_width=True)
        st.caption(
            "Regret is measured relative to the best policy in the selected scenario."
        )

    with b2:
        st.markdown("#### Tradeoff Structure")
        st.markdown(
            """
- More transfers → lower overload but higher coordination burden  
- Less intervention → preserves access but increases overload risk  
- Surge activation → lowers peak stress but adds operational burden
            """
        )

        st.markdown("#### Counterfactual Analysis")
        for cf in counterfactuals:
            st.write(f"- {cf}")

        st.markdown("#### Model Basis")
        st.caption(
            "Grounded in outputs from the Markov progression model, rolling-horizon MILP, "
            "policy simulations, hospital bottleneck summary/decomposition, hospital time-series outputs, "
            "and the hybrid decision agent."
        )


# ============================================================
# TAB 6: DIGITAL TWIN (NEXT 7 DAYS)
# ============================================================

def compute_digital_twin_horizon(
    hospital_ts_df: pd.DataFrame,
    scenario: str,
    policy_name: str = "optimized_network",
    metric: str = "bottleneck_score",
) -> pd.DataFrame:
    """
    Build a day-by-day best/expected/worst-case horizon for the network by
    aggregating the per-hospital, per-replication time series to a network-wide
    value per day per replication, then taking percentile bands across replications.
    """
    df = hospital_ts_df[
        (hospital_ts_df["scenario"] == scenario)
        & (hospital_ts_df["policy_name"] == policy_name)
    ].copy()

    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    network_by_rep_day = (
        df.groupby(["day", "replication"], as_index=False)[metric].sum()
    )

    horizon = (
        network_by_rep_day.groupby("day")[metric]
        .agg(
            best=lambda s: s.quantile(0.10),
            expected="mean",
            worst=lambda s: s.quantile(0.90),
        )
        .reset_index()
        .sort_values("day")
    )

    return horizon


def render_digital_twin_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("Stochastic State-Transition Twin: Next 7 Days")
    st.caption(
        "Forward-looking projection from the stochastic twin -- best, expected, and worst "
        "case network bottleneck trajectory under the recommended policy."
    )

    hospital_ts_df = data["hospital_time_series_outputs"]
    if hospital_ts_df.empty:
        st.info("Hospital time-series data not available.")
        return

    horizon = compute_digital_twin_horizon(
        hospital_ts_df=hospital_ts_df,
        scenario=scenario,
        policy_name="optimized_network",
        metric="bottleneck_score",
    )

    if horizon.empty:
        st.info("No horizon data available for this scenario.")
        return

    chart_df = horizon.set_index("day")[["best", "expected", "worst"]]
    chart_df.columns = ["Best case", "Expected case", "Worst case"]
    st.line_chart(chart_df)

    peak_day = int(horizon.loc[horizon["worst"].idxmax(), "day"])
    peak_worst = float(horizon["worst"].max())
    st.caption(
        f"Worst-case network bottleneck peaks on day {peak_day} "
        f"(bottleneck score {peak_worst:.2f}) under the `optimized_network` policy in the "
        f"`{scenario}` scenario."
    )

    display_horizon = horizon.rename(
        columns={"best": "Best case", "expected": "Expected case", "worst": "Worst case"}
    )
    st.dataframe(display_horizon, use_container_width=True)


# ============================================================
# TAB 7: EVENT SIMULATOR
# ============================================================

EVENT_PRESETS = {
    "Flu Surge": {
        "description": "Seasonal flu surge increases ED arrivals network-wide.",
        "icu_bed_delta": 0,
        "transfer_capacity_multiplier": 1.0,
        "demand_surge_multiplier": 1.40,
    },
    "Mass Casualty Event": {
        "description": "Acute mass casualty incident drives a sharp arrival spike with transfer congestion.",
        "icu_bed_delta": 0,
        "transfer_capacity_multiplier": 0.80,
        "demand_surge_multiplier": 1.80,
    },
    "Transfer Failure": {
        "description": "Inter-hospital transfer routes are largely unavailable (e.g. ambulance system outage).",
        "icu_bed_delta": 0,
        "transfer_capacity_multiplier": 0.10,
        "demand_surge_multiplier": 1.0,
    },
    "ICU Closure": {
        "description": "A wing closure removes ICU beds from the network (e.g. infection control, renovation).",
        "icu_bed_delta": -15,
        "transfer_capacity_multiplier": 1.0,
        "demand_surge_multiplier": 1.0,
    },
}


def render_event_simulator_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("Event Simulator")
    st.caption(
        "Trigger a named disruption event and watch the network react -- this runs the same "
        "live solve as the Scenario Lab and Copilot, with realistic preset parameters."
    )

    event_name = st.selectbox("Disruption event", options=list(EVENT_PRESETS.keys()))
    preset = EVENT_PRESETS[event_name]
    st.caption(preset["description"])
    st.caption(
        f"Parameters: ICU beds {preset['icu_bed_delta']:+d}, "
        f"transfer capacity x{preset['transfer_capacity_multiplier']:.2f}, "
        f"demand x{preset['demand_surge_multiplier']:.2f}."
    )

    precision = st.radio(
        "Precision",
        options=["Quick preview (10 reps, ~5s)", "Full precision (50 reps, matches baseline exactly)"],
        index=0,
        horizontal=True,
        key="event_precision",
    )
    n_reps = 10 if precision.startswith("Quick") else 50

    if st.button("Trigger event", type="primary"):
        with st.spinner(f"Simulating {event_name}..."):
            result = cached_live_whatif(
                icu_bed_delta=preset["icu_bed_delta"],
                transfer_capacity_multiplier=preset["transfer_capacity_multiplier"],
                demand_surge_multiplier=preset["demand_surge_multiplier"],
                n_replications=n_reps,
            )

        baseline_row = get_reference_row(data["scenario_suite_summary"], scenario)

        e1, e2, e3 = st.columns(3)
        unsafe_val = result.get("total_unsafe_excess", 0.0)
        blocked_val = result.get("total_blocked_arrivals", 0.0)
        util_val = result.get("max_utilization_ratio", 0.0)

        if baseline_row is not None:
            e1.metric(
                "Unsafe Excess",
                f"{unsafe_val:.2f}",
                delta=round(unsafe_val - float(baseline_row["total_unsafe_excess_mean"]), 2),
                delta_color="inverse",
            )
            e2.metric(
                "Blocked Arrivals",
                f"{blocked_val:.2f}",
                delta=round(blocked_val - float(baseline_row["total_blocked_arrivals_mean"]), 2),
                delta_color="inverse",
            )
            e3.metric(
                "Max Utilization",
                f"{util_val:.3f}",
                delta=round(util_val - float(baseline_row["max_utilization_ratio_mean"]), 3),
                delta_color="inverse",
            )
        else:
            e1.metric("Unsafe Excess", f"{unsafe_val:.2f}")
            e2.metric("Blocked Arrivals", f"{blocked_val:.2f}")
            e3.metric("Max Utilization", f"{util_val:.3f}")

        icu_status, overflow_status, bottleneck = classify_stress_label(
            max_util=util_val,
            overflow=result.get("total_overflow_excess", 0.0),
            unsafe_rows=result.get("num_unsafe_rows", 0.0),
        )

        st.markdown("#### System Response")
        if icu_status == "Critical":
            st.error(
                f"ICU Status: {icu_status} -- network transfers are already fully engaged; "
                "recommend activating surge capacity."
            )
        elif icu_status == "Stressed":
            st.warning(
                f"ICU Status: {icu_status} -- recommend continued transfer-enabled coordination, monitor closely."
            )
        else:
            st.success(
                f"ICU Status: {icu_status} -- the optimized network policy is absorbing this event "
                "without escalation."
            )

        st.caption(
            f"{int(result.get('n_replications', 0))} reps, "
            f"{result.get('solve_runtime_seconds', 0.0):.1f}s, under the optimized_network policy."
        )


# ============================================================
# TAB 8: AGENT PIPELINE
# ============================================================

def render_agent_pipeline_tab(data: dict[str, pd.DataFrame], scenario: str) -> None:
    st.subheader("Agent Pipeline")
    st.caption(
        "Runs the full Forecast -> Optimization -> Risk -> Explanation agent chain live, "
        "end to end, showing each agent's handoff to the next. Each agent wraps existing "
        "platform logic (stochastic twin, live solver, stress classifier) -- this is the "
        "orchestration layer, not separate reimplementations."
    )

    precision = st.radio(
        "Precision",
        options=["Quick preview (10 reps, ~5s)", "Full precision (50 reps, matches baseline exactly)"],
        index=0,
        horizontal=True,
        key="agent_precision",
    )
    n_reps = 10 if precision.startswith("Quick") else 50

    if st.button("Run agent pipeline", type="primary"):
        hospital_ts_df = data["hospital_time_series_outputs"]
        horizon = compute_digital_twin_horizon(
            hospital_ts_df=hospital_ts_df,
            scenario=scenario,
            policy_name="optimized_network",
        )

        if horizon.empty:
            twin_day7 = {"day": None, "expected": 0.0, "worst": 0.0}
        else:
            twin_day7 = horizon.loc[horizon["day"].idxmax()].to_dict()

        with st.spinner("Running agent pipeline..."):
            trace = run_agent_pipeline(
                twin_horizon_final_day=twin_day7,
                classify_fn=classify_stress_label,
                whatif_fn=cached_live_whatif,
                whatif_kwargs={
                    "icu_bed_delta": 0,
                    "transfer_capacity_multiplier": 1.0,
                    "demand_surge_multiplier": 1.0,
                    "n_replications": n_reps,
                },
            )

        for i, step in enumerate(trace):
            st.markdown(f"#### {i + 1}. {step.agent_name}")
            st.write(step.summary)
            if i < len(trace) - 1:
                st.markdown("<div style='text-align:center; color:#999;'>&#8595;</div>", unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================


def render_flow_cvar_tab() -> None:
    st.subheader("FLOW-CVaR Capacity Council")
    st.caption("Markov-aware tail-risk capacity planning on MINCO's synthetic reference benchmark. Operator approval is mandatory.")
    c1, c2, c3 = st.columns(3)
    alpha = c1.select_slider("CVaR confidence", options=[0.70, 0.80, 0.90, 0.95], value=0.80, key="flow_alpha")
    weight = c2.slider("Tail-risk weight", 0.0, 5.0, 1.5, 0.25, key="flow_weight")
    severe = c3.slider("Rare-surge H1 demand", 10.0, 30.0, 22.0, 1.0, key="flow_demand")
    payload = build_flow_cvar_decision(cvar_alpha=float(alpha), cvar_weight=float(weight), severe_demand=float(severe))
    dec = payload["decision"]
    base = payload["no_cvar_baseline"]
    g1, g2 = st.columns([1, 3])
    if payload["gate"] == "AUTHORIZED":
        g1.success("Evidence gate: AUTHORIZED")
    else:
        g1.error("Evidence gate: BLOCKED")
    g2.code(payload["decision_id"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FLOW-CVaR tail loss", f'{dec["cvar_loss"]:.2f}')
    m2.metric("No-CVaR tail loss", f'{base["cvar_loss"]:.2f}', delta=f'{dec["cvar_loss"]-base["cvar_loss"]:.2f}')
    m3.metric("Surge activations", int(dec["surge_activations"]))
    m4.metric("Flex staff blocks", int(dec["flex_staff_blocks"]))
    st.markdown("### Markov patient-flow forecast")
    st.dataframe(pd.DataFrame(payload["flow_forecast"]["transition_matrix"], index=payload["flow_forecast"]["states"], columns=payload["flow_forecast"]["states"]), use_container_width=True)
    st.json(payload["flow_forecast"]["forecast_hospital_census"])
    st.markdown("### Recommended first-stage actions")
    st.dataframe(pd.DataFrame(payload["actions"]), use_container_width=True, hide_index=True)
    st.markdown("### Scenario transfer / diversion actions")
    st.dataframe(pd.DataFrame(payload["scenario_actions"]), use_container_width=True, hide_index=True)
    st.markdown("### Tail-risk counterfactual")
    st.dataframe(pd.DataFrame([
        {"policy":"FLOW-CVaR","cvar_loss":dec["cvar_loss"],"expected_loss":dec["expected_recourse_loss"],"tail_loss":dec["tail_scenario_loss"],"surge":dec["surge_activations"],"flex":dec["flex_staff_blocks"]},
        {"policy":"No CVaR","cvar_loss":base["cvar_loss"],"expected_loss":base["expected_recourse_loss"],"tail_loss":base["tail_scenario_loss"],"surge":base["surge_activations"],"flex":base["flex_staff_blocks"]},
    ]), use_container_width=True, hide_index=True)
    st.markdown("### Evidence gate")
    st.dataframe(pd.DataFrame([{"check":k,"passed":v} for k,v in payload["checks"].items()]), use_container_width=True, hide_index=True)
    st.warning(payload["operator_note"])
    st.caption(f'Evidence class: {payload["evidence_class"]}. {dec["bounded_claim"]}')


def render_control_tower_tab() -> None:
    """Render the executive/operator landing surface over the existing services."""
    payload = cached_control_tower_snapshot()
    release = cached_release_readiness()
    governance = payload["governance"]
    network = payload["network"]
    risk = payload["risk"]
    decision = payload["decision"]
    history = payload["operational_history"]

    st.subheader("Enterprise Control Tower")
    st.caption(
        "One operator view of network state, projected flow, tail risk, recommended actions, "
        "solver evidence, and human-gated authorization."
    )

    banner_left, banner_right = st.columns([1, 3])
    if payload["governance_state"] == "BLOCKED":
        banner_left.error("GOVERNANCE: BLOCKED")
    else:
        banner_left.warning("GOVERNANCE: HUMAN REVIEW")
    banner_right.info(governance["authorization_scope"])
    st.caption(
        f"Operating mode: {payload['operating_mode']} | Alert level: {payload['overall_alert_level']} | "
        f"Snapshot: {payload['snapshot_id']}"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hospitals in inventory", int(network["hospital_count"]))
    m2.metric("Projected H1/H2 census", f"{sum(network['projected_census'].values()):.1f}")
    m3.metric("FLOW-CVaR tail loss", f"{risk['cvar_loss']:.2f}")
    m4.metric("Recommended actions", int(decision["selected_action_count"]))

    st.markdown("### Network operating picture")
    hospital_rows = []
    modeled = set(network["modeled_hospitals"])
    for hospital_id in network["state_inventory_hospitals"]:
        hospital_rows.append(
            {
                "hospital": hospital_label(hospital_id),
                "id": hospital_id,
                "model scope": "FLOW-CVaR modeled" if hospital_id in modeled else "Inventory only",
                "projected census": network["projected_census"].get(hospital_id),
            }
        )
    st.dataframe(pd.DataFrame(hospital_rows), use_container_width=True, hide_index=True)
    st.caption(network["model_scope_note"])

    left, right = st.columns(2)
    with left:
        st.markdown("### Tail-risk decision view")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "measure": "Expected recourse loss",
                        "FLOW-CVaR": risk["expected_recourse_loss"],
                        "No-CVaR baseline": payload["risk"]["no_cvar_cvar_loss"],
                    },
                    {
                        "measure": "CVaR tail loss",
                        "FLOW-CVaR": risk["cvar_loss"],
                        "No-CVaR baseline": risk["no_cvar_cvar_loss"],
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"alpha={risk['cvar_alpha']:.2f}, lambda={risk['cvar_weight']:.2f}, "
            f"{risk['scenario_count']} governed scenarios."
        )
    with right:
        st.markdown("### Recommended capacity actions")
        action_rows = decision["first_stage_actions"] + decision["scenario_recourse_actions"]
        st.dataframe(
            pd.DataFrame(action_rows) if action_rows else pd.DataFrame([{"action": "NO_ACTION"}]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Decision ID: {decision['decision_id']}")

    with st.expander("Evidence and governance chain", expanded=True):
        st.dataframe(
            pd.DataFrame(
                [
                    {"check": name, "passed": passed}
                    for name, passed in governance["gate_checks"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.write(
            {
                "release_readiness": governance["release_readiness"],
                "status_freshness": payload["evidence"]["data_freshness"]["status"],
                "reference_case": payload["reference_case"],
                "human_review_required": governance["human_review_required"],
                "autonomous_execution_permitted": governance["autonomous_execution_permitted"],
            }
        )
        st.warning(governance["operator_action"])
        st.caption(governance["claim_boundary"])

        st.markdown("#### Human review workflow")
        review_workflow = governance["review_workflow"]
        st.write(
            {
                "review_endpoint": review_workflow["review_endpoint"],
                "history_endpoint": review_workflow["history_endpoint"],
                "integrity_endpoint": review_workflow["integrity_endpoint"],
                "integrity_status": review_workflow["integrity_status"],
                "integrity_review_count": review_workflow["integrity_review_count"],
                "allowed_decisions": review_workflow["allowed_decisions"],
                "immutable_event_log": review_workflow["immutable_event_log"],
                "autonomous_execution_permitted": review_workflow["autonomous_execution_permitted"],
                "recent_review_count": review_workflow["recent_review_count"],
            }
        )
        if review_workflow["latest_review"]:
            st.info(f"Latest review: {review_workflow['latest_review']}")
        st.caption(review_workflow["effect"])

        st.markdown("#### Release-readiness scorecard")
        scorecard_left, scorecard_right = st.columns([1, 2])
        if release["status"] == "BLOCKED":
            scorecard_left.error("RELEASE: BLOCKED")
        else:
            scorecard_left.warning("RELEASE: CONDITIONAL")
        scorecard_right.write(
            {
                "status": release["status"],
                "release_readiness": release["release_readiness"],
                "blocking_reason_count": len(release["blocking_reasons"]),
                "next_action_count": len(release["next_actions"]),
                "security_mode": release["security_controls"]["security_mode"],
            }
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"check": name, "passed": passed}
                    for name, passed in release["checks"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if release["blocking_reasons"]:
            st.warning("; ".join(release["blocking_reasons"]))
        st.caption("GET /v1/release-readiness | " + release["claim_boundary"])

    with st.expander("Scenario catalog", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "scenario": item["name"],
                        "id": item["scenario_id"],
                        **item["parameters"],
                    }
                    for item in payload["scenario_catalog"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Operational event history", expanded=False):
        st.write(
            {
                "data_mode": history["data_mode"],
                "data_stack": history["data_stack"],
                "event_count": history["event_count"],
                "latest_event_timestamp": history["latest_event_timestamp"],
                "freshness_status": history["freshness_status"],
                "source_modes": history["source_modes"],
                "live_feed_connected": history["live_feed_connected"],
            }
        )
        if history["event_type_counts"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"event_type": name, "count": count}
                        for name, count in history["event_type_counts"].items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption("Operational event history is replay/provenance evidence; it is not a live ADT/EHR feed.")

def main() -> None:
    data = load_all_data()

    render_header()
    render_missing_data_notice(data)

    scenario = choose_scenario(data)

    render_system_status(data["scenario_suite_summary"], scenario)
    render_kpi_cards(data["scenario_suite_summary"], scenario)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        [
            "System Forecast",
            "Decision Engine",
            "Scenario Lab",
            "Policy Benchmark",
            "AI Copilot",
            "Stochastic Twin",
            "Event Simulator",
            "Agent Pipeline",
            "FLOW-CVaR Council",
            "Enterprise Control Tower",
        ]
    )

    with tab1:
        render_forecast_tab(data, scenario)

    with tab2:
        render_decision_tab(data, scenario)

    with tab3:
        render_scenario_lab_tab(data, scenario)

    with tab4:
        render_policy_benchmark_tab(data, scenario)

    with tab5:
        render_ai_copilot_tab(data, scenario)

    with tab6:
        render_digital_twin_tab(data, scenario)

    with tab7:
        render_event_simulator_tab(data, scenario)

    with tab8:
        render_agent_pipeline_tab(data, scenario)

    with tab9:
        render_flow_cvar_tab()

    with tab10:
        render_control_tower_tab()


if __name__ == "__main__":
    main()
