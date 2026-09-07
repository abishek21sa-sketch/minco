from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="MINCO command-line entry point")
    parser.add_argument("command", choices=["smoke", "api", "dashboard", "full-pipeline"])
    args, remaining = parser.parse_known_args()

    if args.command == "smoke":
        command = [sys.executable, "-m", "src.system.run_smoke_pipeline", *remaining]
    elif args.command == "api":
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            *remaining,
        ]
    elif args.command == "dashboard":
        command = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", *remaining]
    else:
        command = [sys.executable, "-m", "src.system.run_all_pipeline", *remaining]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
