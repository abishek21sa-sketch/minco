from __future__ import annotations

from typing import Any

import grpc
from google.protobuf.wrappers_pb2 import StringValue

from src.runtime.grpc_wire import METHODS, decode_payload, encode_payload


class MincoRuntimeClient:
    def __init__(self, target: str = "127.0.0.1:50551") -> None:
        self.channel = grpc.insecure_channel(target)

    def close(self) -> None:
        self.channel.close()

    def _call(self, name: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
        call = self.channel.unary_unary(
            METHODS[name],
            request_serializer=StringValue.SerializeToString,
            response_deserializer=StringValue.FromString,
        )
        response = call(encode_payload(payload or {}), timeout=timeout)
        return decode_payload(response)

    def status(self): return self._call("GetStatus")
    def network_state(self): return self._call("GetNetworkState")
    def optimize_plan(self, payload: dict[str, Any]): return self._call("OptimizePlan", payload, timeout=180)
    def evaluate_intervention(self, payload: dict[str, Any]): return self._call("EvaluateIntervention", payload, timeout=180)
    def review_decision(self, payload: dict[str, Any]): return self._call("ReviewDecision", payload, timeout=180)
