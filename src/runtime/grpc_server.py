from __future__ import annotations

from concurrent import futures
import json
from pathlib import Path
import os
from typing import Any, Callable

import grpc
from google.protobuf.wrappers_pb2 import StringValue

from src.runtime.contracts import InterventionRequest, PlanRequest, ReviewRequest
from src.runtime.grpc_wire import SERVICE, decode_payload, encode_payload
from src.runtime.workstation_service import WorkstationService


class RuntimeGrpcAdapter:
    def __init__(self, service: WorkstationService | None = None) -> None:
        self.service = service or WorkstationService()

    def _wrap(self, fn: Callable[[dict[str, Any]], Any]):
        def handler(request: StringValue, context) -> StringValue:
            try:
                return encode_payload(fn(decode_payload(request)))
            except Exception as exc:  # boundary converts typed failures to gRPC status
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return encode_payload({"error": str(exc)})
        return handler

    def handlers(self) -> dict[str, grpc.RpcMethodHandler]:
        return {
            "GetStatus": grpc.unary_unary_rpc_method_handler(
                self._wrap(lambda _: self.service.status()),
                request_deserializer=StringValue.FromString,
                response_serializer=StringValue.SerializeToString,
            ),
            "GetNetworkState": grpc.unary_unary_rpc_method_handler(
                self._wrap(lambda _: self.service.network_state()),
                request_deserializer=StringValue.FromString,
                response_serializer=StringValue.SerializeToString,
            ),
            "GetScenarioCatalog": grpc.unary_unary_rpc_method_handler(
                self._wrap(lambda _: self.service.scenario_catalog()),
                request_deserializer=StringValue.FromString,
                response_serializer=StringValue.SerializeToString,
            ),
            "OptimizePlan": grpc.unary_unary_rpc_method_handler(
                self._wrap(lambda p: self.service.plan(PlanRequest.model_validate(p))),
                request_deserializer=StringValue.FromString,
                response_serializer=StringValue.SerializeToString,
            ),
            "EvaluateIntervention": grpc.unary_unary_rpc_method_handler(
                self._wrap(lambda p: self.service.evaluate_intervention(InterventionRequest.model_validate(p))),
                request_deserializer=StringValue.FromString,
                response_serializer=StringValue.SerializeToString,
            ),
            "ReviewDecision": grpc.unary_unary_rpc_method_handler(
                self._wrap(lambda p: self.service.review(ReviewRequest.model_validate(p))),
                request_deserializer=StringValue.FromString,
                response_serializer=StringValue.SerializeToString,
            ),
        }


def create_server(*, service: WorkstationService | None = None, max_workers: int = 8) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    adapter = RuntimeGrpcAdapter(service)
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, adapter.handlers()),))
    return server


def serve(host: str = "127.0.0.1", port: int = 50551) -> None:
    server = create_server()
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound == 0:
        raise RuntimeError(f"Unable to bind MINCO gRPC runtime to {host}:{port}")
    server.start()
    print(f"MINCO gRPC runtime listening on {host}:{port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2)


if __name__ == "__main__":
    serve(host=os.getenv("MINCO_GRPC_HOST", "127.0.0.1"), port=int(os.getenv("MINCO_GRPC_PORT", "50551")))
