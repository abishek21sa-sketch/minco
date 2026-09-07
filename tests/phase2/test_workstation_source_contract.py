from pathlib import Path


def test_flutter_workstation_is_native_clinical_workflow_not_generic_dashboard():
    root = Path("workstation")
    required = [
        root / "pubspec.yaml",
        root / "lib/main.dart",
        root / "lib/services/minco_grpc_client.dart",
        root / "lib/screens/census_board.dart",
        root / "lib/screens/flow_theatre.dart",
        root / "lib/screens/monte_carlo_room.dart",
        root / "lib/screens/intervention_composer.dart",
        root / "lib/screens/decision_review.dart",
    ]
    assert all(p.exists() for p in required)
    merged = "\n".join(p.read_text(encoding="utf-8") for p in required)
    for signature in [
        "Census Board",
        "Patient Flow Theatre",
        "Monte Carlo Room",
        "Intervention Composer",
        "Decision Review Board",
    ]:
        assert signature in merged
    assert "ClientChannel" in merged
    assert "/minco.runtime.v1.MincoRuntime/" in merged
    assert "Mission Control" not in merged
