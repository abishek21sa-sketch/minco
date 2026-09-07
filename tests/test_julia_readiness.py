from __future__ import annotations

from types import SimpleNamespace

from src.optimization_julia import bridge


def test_julia_readiness_is_false_when_executable_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(bridge.shutil, "which", lambda name: None)

    readiness = bridge.check_julia_readiness()

    assert readiness.ready is False
    assert readiness.status == "executable_unavailable"
    assert readiness.dependencies_available is False


def test_julia_readiness_checks_project_dependencies(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "julia.exe")

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="MINCO_JULIA_READY\n", stderr="")

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    readiness = bridge.check_julia_readiness()

    assert readiness.ready is True
    assert readiness.status == "ready"
    assert any(arg.startswith("--project=") for arg in calls[0][0][0])
    assert "JuMP" in calls[0][0][0][-1]


def test_julia_readiness_preserves_remediation_when_packages_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(bridge.shutil, "which", lambda name: "julia.exe")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Package JuMP is required but does not seem to be installed",
        ),
    )

    readiness = bridge.check_julia_readiness()

    assert readiness.ready is False
    assert readiness.status == "dependencies_unavailable"
    assert "julia julia/setup.jl" in readiness.message
