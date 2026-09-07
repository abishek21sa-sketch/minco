from __future__ import annotations

import importlib
import json
import platform
from typing import Any

REQUIRED = ["numpy", "pandas", "fastapi", "pydantic"]
OPTIONAL = ["streamlit", "sklearn", "matplotlib", "gurobipy"]


def inspect_module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    report = {
        "python": platform.python_version(),
        "required": {name: inspect_module(name) for name in REQUIRED},
        "optional": {name: inspect_module(name) for name in OPTIONAL},
    }
    print(json.dumps(report, indent=2))
    return 0 if all(item["available"] for item in report["required"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
