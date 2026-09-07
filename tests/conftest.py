from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def copied_data_dir(tmp_path: Path, project_root: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(project_root / "data" / "base_instance", target / "base_instance")
    shutil.copytree(project_root / "data" / "transitions", target / "transitions")
    return target
