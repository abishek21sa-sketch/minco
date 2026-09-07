from __future__ import annotations

import json
from pathlib import Path

from src.validation.run_manifest import sha256_file, write_run_manifest


def test_run_manifest_hashes_inputs_and_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.json"
    input_path.write_text("a,b\n1,2\n", encoding="utf-8")
    output_path.write_text('{"status": "ok"}', encoding="utf-8")

    manifest_path, manifest_hash = write_run_manifest(
        run_id="test_run_manifest",
        run_type="unit_test",
        parameters={"replications": 3},
        input_paths=[input_path],
        output_paths=[output_path],
        metrics={"objective": 1.25},
        manifest_dir=tmp_path / "manifests",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test_run_manifest"
    assert payload["inputs"][0]["sha256"] == sha256_file(input_path)
    assert payload["outputs"][0]["sha256"] == sha256_file(output_path)
    assert manifest_hash == sha256_file(manifest_path)
