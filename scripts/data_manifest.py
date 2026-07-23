"""Create and verify checksummed manifests for processed datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MANIFEST_NAME = "data_manifest.json"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    data_dir: Path, *, manifest_name: str = DEFAULT_MANIFEST_NAME
) -> dict:
    """Describe every processed file with its size and content digest."""
    files = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name == manifest_name:
            continue
        files.append(
            {
                "path": path.relative_to(data_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"No processed data files found under {data_dir}.")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.getenv("GITHUB_SHA", "local"),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "files": files,
    }


def write_manifest(data_dir: Path, output: Path) -> dict:
    """Create and save a processed-data manifest."""
    manifest = build_manifest(data_dir, manifest_name=output.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(data_dir: Path, manifest_path: Path) -> None:
    """Raise a readable error when a snapshot differs from its manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest.get("files", [])}
    actual = {
        path.relative_to(data_dir).as_posix(): path
        for path in data_dir.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(
        relative
        for relative in set(expected) & set(actual)
        if actual[relative].stat().st_size != expected[relative]["bytes"]
        or sha256_file(actual[relative]) != expected[relative]["sha256"]
    )
    if missing or unexpected or changed:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if unexpected:
            parts.append("unexpected: " + ", ".join(unexpected))
        if changed:
            parts.append("checksum mismatch: " + ", ".join(changed))
        raise ValueError(
            "Processed data failed manifest verification (" + "; ".join(parts) + ")."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.data_dir / DEFAULT_MANIFEST_NAME
    if args.action == "create":
        manifest = write_manifest(args.data_dir, manifest_path)
        print(f"Wrote {manifest_path} with {len(manifest['files'])} files.")
    else:
        verify_manifest(args.data_dir, manifest_path)
        print(f"Verified processed data against {manifest_path}.")


if __name__ == "__main__":
    main()
