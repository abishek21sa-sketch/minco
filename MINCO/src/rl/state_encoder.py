from __future__ import annotations

from typing import Any


# ============================================================
# Safe helpers
# ============================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        return str(x)
    except Exception:
        return default


# ============================================================
# Bucketizers
# ============================================================

def bucket_regime(regime: str) -> str:
    r = safe_str(regime).strip().lower()
    if r in {"normal", "surge", "crisis"}:
        return r
    return "normal"


def bucket_unsafe_excess(value: float) -> str:
    x = safe_float(value)
    if x <= 0:
        return "none"
    if x <= 3:
        return "low"
    if x <= 8:
        return "moderate"
    return "high"


def bucket_blocked_arrivals(value: float) -> str:
    x = safe_float(value)
    if x <= 0:
        return "none"
    if x <= 5:
        return "low"
    if x <= 12:
        return "moderate"
    return "high"


def bucket_utilization(value: float) -> str:
    x = safe_float(value)
    if x < 1.0:
        return "stable"
    if x < 1.1:
        return "elevated"
    if x < 1.3:
        return "stressed"
    return "critical"


def bucket_unsafe_rows(value: float) -> str:
    x = safe_float(value)
    if x <= 0:
        return "none"
    if x <= 2:
        return "few"
    if x <= 5:
        return "moderate"
    return "many"


def bucket_overflow(value: float) -> str:
    x = safe_float(value)
    if x <= 0:
        return "none"
    if x <= 2:
        return "low"
    if x <= 6:
        return "moderate"
    return "high"


def bucket_surge_gap(value: float) -> str:
    x = safe_float(value)
    if x <= 0:
        return "none"
    if x <= 1:
        return "low"
    if x <= 3:
        return "moderate"
    return "high"


def bucket_timestep(value: float) -> str:
    x = int(safe_float(value))
    if x <= 1:
        return "early"
    if x <= 4:
        return "mid"
    return "late"


# ============================================================
# Core encoders
# ============================================================

def encode_state_to_dict(raw_state: dict[str, Any]) -> dict[str, str]:
    """
    Convert raw env state into a compact discrete feature dictionary.
    """
    return {
        "regime": bucket_regime(raw_state.get("regime")),
        "unsafe_excess_bin": bucket_unsafe_excess(raw_state.get("prev_unsafe_excess")),
        "blocked_arrivals_bin": bucket_blocked_arrivals(raw_state.get("prev_blocked_arrivals")),
        "utilization_bin": bucket_utilization(raw_state.get("prev_max_utilization")),
        "unsafe_rows_bin": bucket_unsafe_rows(raw_state.get("prev_unsafe_rows")),
        "overflow_bin": bucket_overflow(raw_state.get("prev_overflow_excess")),
        "surge_gap_bin": bucket_surge_gap(raw_state.get("prev_surge_gap")),
        "timestep_bin": bucket_timestep(raw_state.get("timestep")),
    }


def encode_state_to_tuple(raw_state: dict[str, Any]) -> tuple[str, ...]:
    """
    Main state representation for tabular Q-learning.
    """
    s = encode_state_to_dict(raw_state)
    return (
        s["regime"],
        s["unsafe_excess_bin"],
        s["blocked_arrivals_bin"],
        s["utilization_bin"],
        s["unsafe_rows_bin"],
        s["overflow_bin"],
        s["surge_gap_bin"],
        s["timestep_bin"],
    )


def encode_state_to_key(raw_state: dict[str, Any]) -> str:
    """
    String key form for dictionaries / logging / CSV exports.
    """
    return "|".join(encode_state_to_tuple(raw_state))


# ============================================================
# Optional reduced encoders
# ============================================================

def encode_state_to_tuple_small(raw_state: dict[str, Any]) -> tuple[str, ...]:
    """
    Smaller state space if tabular Q-learning gets too sparse.
    """
    return (
        bucket_regime(raw_state.get("regime")),
        bucket_unsafe_excess(raw_state.get("prev_unsafe_excess")),
        bucket_blocked_arrivals(raw_state.get("prev_blocked_arrivals")),
        bucket_utilization(raw_state.get("prev_max_utilization")),
    )


def encode_state_to_key_small(raw_state: dict[str, Any]) -> str:
    return "|".join(encode_state_to_tuple_small(raw_state))


# ============================================================
# Diagnostic main
# ============================================================

def main() -> None:
    example_state = {
        "timestep": 1,
        "regime": "crisis",
        "regime_code": 2,
        "prev_unsafe_excess": 7.45,
        "prev_blocked_arrivals": 12.4,
        "prev_max_utilization": 1.085,
        "prev_unsafe_rows": 2.8,
        "prev_overflow_excess": 3.5,
        "prev_surge_gap": 0.4,
    }

    print("\nSTATE ENCODER")
    print("=" * 60)
    print("\nRaw state:")
    print(example_state)

    print("\nEncoded dict:")
    print(encode_state_to_dict(example_state))

    print("\nEncoded tuple:")
    print(encode_state_to_tuple(example_state))

    print("\nEncoded key:")
    print(encode_state_to_key(example_state))

    print("\nEncoded tuple (small):")
    print(encode_state_to_tuple_small(example_state))

    print("\nEncoded key (small):")
    print(encode_state_to_key_small(example_state))


if __name__ == "__main__":
    main()