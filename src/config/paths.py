from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("MINCO_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA_DIR = Path(os.getenv("MINCO_DATA_DIR", PROJECT_ROOT / "data")).resolve()
RESULTS_DIR = Path(os.getenv("MINCO_RESULTS_DIR", PROJECT_ROOT / "results")).resolve()
CONFIG_DIR = Path(os.getenv("MINCO_CONFIG_DIR", PROJECT_ROOT / "configs")).resolve()
AUDIT_DB_PATH = Path(
    os.getenv("MINCO_AUDIT_DB_PATH", RESULTS_DIR / "ops_platform" / "audit.db")
).resolve()
