from __future__ import annotations

from src.rl.state_encoder import (
    bucket_blocked_arrivals,
    bucket_overflow,
    bucket_regime,
    bucket_surge_gap,
    bucket_timestep,
    bucket_unsafe_excess,
    bucket_unsafe_rows,
    bucket_utilization,
    encode_state_to_dict,
    encode_state_to_key,
    encode_state_to_key_small,
    encode_state_to_tuple,
    encode_state_to_tuple_small,
    safe_float,
    safe_str,
)


def test_safe_float_falls_back_on_none_and_bad_input() -> None:
    assert safe_float(None) == 0.0
    assert safe_float(None, default=1.5) == 1.5
    assert safe_float("not a number") == 0.0
    assert safe_float("3.5") == 3.5
    assert safe_float(3) == 3.0


def test_safe_str_falls_back_on_none() -> None:
    assert safe_str(None) == ""
    assert safe_str(None, default="x") == "x"
    assert safe_str(7) == "7"


def test_bucket_regime_normalizes_case_and_rejects_unknown() -> None:
    assert bucket_regime("Surge") == "surge"
    assert bucket_regime(" crisis ") == "crisis"
    assert bucket_regime("normal") == "normal"
    assert bucket_regime("unknown_regime") == "normal"
    assert bucket_regime(None) == "normal"


def test_bucket_unsafe_excess_boundaries() -> None:
    assert bucket_unsafe_excess(0) == "none"
    assert bucket_unsafe_excess(-5) == "none"
    assert bucket_unsafe_excess(3) == "low"
    assert bucket_unsafe_excess(3.01) == "moderate"
    assert bucket_unsafe_excess(8) == "moderate"
    assert bucket_unsafe_excess(8.01) == "high"


def test_bucket_blocked_arrivals_boundaries() -> None:
    assert bucket_blocked_arrivals(0) == "none"
    assert bucket_blocked_arrivals(5) == "low"
    assert bucket_blocked_arrivals(5.01) == "moderate"
    assert bucket_blocked_arrivals(12) == "moderate"
    assert bucket_blocked_arrivals(12.01) == "high"


def test_bucket_utilization_boundaries() -> None:
    assert bucket_utilization(0.99) == "stable"
    assert bucket_utilization(1.0) == "elevated"
    assert bucket_utilization(1.099) == "elevated"
    assert bucket_utilization(1.1) == "stressed"
    assert bucket_utilization(1.299) == "stressed"
    assert bucket_utilization(1.3) == "critical"


def test_bucket_unsafe_rows_boundaries() -> None:
    assert bucket_unsafe_rows(0) == "none"
    assert bucket_unsafe_rows(2) == "few"
    assert bucket_unsafe_rows(2.01) == "moderate"
    assert bucket_unsafe_rows(5) == "moderate"
    assert bucket_unsafe_rows(5.01) == "many"


def test_bucket_overflow_boundaries() -> None:
    assert bucket_overflow(0) == "none"
    assert bucket_overflow(2) == "low"
    assert bucket_overflow(2.01) == "moderate"
    assert bucket_overflow(6) == "moderate"
    assert bucket_overflow(6.01) == "high"


def test_bucket_surge_gap_boundaries() -> None:
    assert bucket_surge_gap(0) == "none"
    assert bucket_surge_gap(1) == "low"
    assert bucket_surge_gap(1.01) == "moderate"
    assert bucket_surge_gap(3) == "moderate"
    assert bucket_surge_gap(3.01) == "high"


def test_bucket_timestep_boundaries() -> None:
    assert bucket_timestep(0) == "early"
    assert bucket_timestep(1) == "early"
    assert bucket_timestep(1.9) == "early"  # int() truncates toward zero, not rounds
    assert bucket_timestep(2) == "mid"
    assert bucket_timestep(4) == "mid"
    assert bucket_timestep(5) == "late"


def _example_state() -> dict:
    return {
        "timestep": 1,
        "regime": "crisis",
        "prev_unsafe_excess": 7.45,
        "prev_blocked_arrivals": 12.4,
        "prev_max_utilization": 1.085,
        "prev_unsafe_rows": 2.8,
        "prev_overflow_excess": 3.5,
        "prev_surge_gap": 0.4,
    }


def test_encode_state_to_dict_matches_individual_bucketizers() -> None:
    state = _example_state()
    encoded = encode_state_to_dict(state)
    assert encoded == {
        "regime": "crisis",
        "unsafe_excess_bin": "moderate",
        "blocked_arrivals_bin": "high",
        "utilization_bin": "elevated",
        "unsafe_rows_bin": "moderate",
        "overflow_bin": "moderate",
        "surge_gap_bin": "low",
        "timestep_bin": "early",
    }


def test_encode_state_to_tuple_orders_fields_consistently() -> None:
    state = _example_state()
    d = encode_state_to_dict(state)
    t = encode_state_to_tuple(state)
    assert t == (
        d["regime"], d["unsafe_excess_bin"], d["blocked_arrivals_bin"],
        d["utilization_bin"], d["unsafe_rows_bin"], d["overflow_bin"],
        d["surge_gap_bin"], d["timestep_bin"],
    )


def test_encode_state_to_key_is_pipe_joined_tuple() -> None:
    state = _example_state()
    assert encode_state_to_key(state) == "|".join(encode_state_to_tuple(state))


def test_encode_state_to_tuple_small_is_prefix_of_full_tuple() -> None:
    state = _example_state()
    full = encode_state_to_tuple(state)
    small = encode_state_to_tuple_small(state)
    assert small == full[:4]
    assert encode_state_to_key_small(state) == "|".join(small)


def test_missing_fields_fall_back_to_defaults_without_raising() -> None:
    # An empty raw state must not crash the encoder -- every field has a safe default.
    encoded = encode_state_to_dict({})
    assert encoded == {
        "regime": "normal",
        "unsafe_excess_bin": "none",
        "blocked_arrivals_bin": "none",
        "utilization_bin": "stable",
        "unsafe_rows_bin": "none",
        "overflow_bin": "none",
        "surge_gap_bin": "none",
        "timestep_bin": "early",
    }
