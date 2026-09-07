from src.data_runtime.memory_store import MemoryEventStore
from src.runtime.grpc_client import MincoRuntimeClient
from src.runtime.grpc_server import create_server
from src.runtime.workstation_service import WorkstationService


def test_grpc_protobuf_roundtrip_without_rest():
    server = create_server(service=WorkstationService(store=MemoryEventStore()), max_workers=2)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    server.start()
    client = MincoRuntimeClient(f"127.0.0.1:{port}")
    try:
        status = client.status()
        assert status["product"].startswith("MINCO Native")
        state = client.network_state()
        assert len(state["hospitals"]) == 3
        plan = client.optimize_plan(
            {
                "planning_horizon_periods": 2,
                "n_scenarios": 4,
                "risk_alpha": 0.9,
                "risk_weight": 0.3,
                "risk_posture": "balanced",
                "use_primary_julia_solver": False,
            }
        )
        assert plan["status"] == "OPTIMAL"
    finally:
        client.close()
        server.stop(0)
