from pathlib import Path


def test_grpc_contract_uses_protobuf_and_no_rest_dependency():
    proto = Path("proto/minco_runtime.proto").read_text(encoding="utf-8")
    assert 'syntax = "proto3"' in proto
    assert "service MincoRuntime" in proto
    for method in [
        "GetStatus",
        "GetNetworkState",
        "OptimizePlan",
        "EvaluateIntervention",
        "ReviewDecision",
    ]:
        assert f"rpc {method}" in proto
    server = Path("src/runtime/grpc_server.py").read_text(encoding="utf-8")
    assert "grpc.server" in server
    assert "FastAPI" not in server
