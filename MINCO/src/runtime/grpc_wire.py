from __future__ import annotations

import json
from typing import Any, Callable

from google.protobuf.wrappers_pb2 import StringValue

SERVICE = "minco.runtime.v1.MincoRuntime"
METHODS = {
    "GetStatus": f"/{SERVICE}/GetStatus",
    "GetNetworkState": f"/{SERVICE}/GetNetworkState",
    "GetScenarioCatalog": f"/{SERVICE}/GetScenarioCatalog",
    "OptimizePlan": f"/{SERVICE}/OptimizePlan",
    "EvaluateIntervention": f"/{SERVICE}/EvaluateIntervention",
    "ReviewDecision": f"/{SERVICE}/ReviewDecision",
}


def encode_payload(payload: Any) -> StringValue:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return StringValue(value=json.dumps(payload, sort_keys=True, default=str))


def decode_payload(message: StringValue) -> dict[str, Any]:
    if not message.value:
        return {}
    value = json.loads(message.value)
    if not isinstance(value, dict):
        raise ValueError("MINCO gRPC payload must be a JSON object")
    return value


def serialize_string_value(message: StringValue) -> bytes:
    return message.SerializeToString()


def deserialize_string_value(payload: bytes) -> StringValue:
    return StringValue.FromString(payload)
