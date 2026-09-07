"""One-command acceptance gate for closing MINCO Level 2."""
from __future__ import annotations

import json

from src.validation.level2_completion_gate import run_level2_completion_gate


def main() -> None:
    report = run_level2_completion_gate()
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
