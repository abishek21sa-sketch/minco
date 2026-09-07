from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("archive/manifests/evidence_manifest.csv")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    included = []
    for folder in ["data/base_instance", "data/transitions", "configs", "results"]:
        base = root / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() not in {".pkl", ".pyc", ".db", ".png"}:
                included.append(path)
    rows = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(included)
    ]
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
