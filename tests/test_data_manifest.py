import json

import pytest

from scripts.data_manifest import verify_manifest, write_manifest


def test_data_manifest_records_and_verifies_processed_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    (data_dir / "prices.parquet").write_bytes(b"prices")
    nested = data_dir / "sec"
    nested.mkdir()
    (nested / "facts.parquet").write_bytes(b"facts")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_RUN_ID", "456")

    output = data_dir / "data_manifest.json"
    manifest = write_manifest(data_dir, output)
    verify_manifest(data_dir, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert manifest == saved
    assert saved["git_commit"] == "abc123"
    assert saved["workflow_run_id"] == "456"
    assert [entry["path"] for entry in saved["files"]] == [
        "prices.parquet",
        "sec/facts.parquet",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in saved["files"])


def test_data_manifest_detects_changed_missing_and_unexpected_files(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    first = data_dir / "first.parquet"
    second = data_dir / "second.parquet"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output = data_dir / "data_manifest.json"
    write_manifest(data_dir, output)

    first.write_bytes(b"changed")
    second.unlink()
    (data_dir / "extra.parquet").write_bytes(b"extra")

    with pytest.raises(ValueError) as error:
        verify_manifest(data_dir, output)

    message = str(error.value)
    assert "checksum mismatch: first.parquet" in message
    assert "missing: second.parquet" in message
    assert "unexpected: extra.parquet" in message


def test_data_manifest_requires_at_least_one_file(tmp_path):
    with pytest.raises(ValueError, match="No processed data files"):
        write_manifest(tmp_path, tmp_path / "data_manifest.json")
