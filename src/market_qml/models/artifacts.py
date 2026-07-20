"""Traceable, private model artifact bundles."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any

import pandas as pd


def resolve_git_sha() -> str:
    """Return the CI commit SHA, falling back to the current Git checkout."""
    if sha := os.getenv("GITHUB_SHA"):
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_model_artifact(
    *,
    root: str | Path,
    model_name: str,
    split_id: int,
    model: Any,
    preprocessor: Any,
    pca: Any | None,
    result: Any,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    target_column: str,
    run_config: dict[str, Any],
    git_sha: str,
) -> dict[str, Any]:
    """Persist one fitted split and return its manifest record."""
    artifact_id = f"{model_name}-split-{split_id:03d}"
    directory = Path(root) / model_name / f"split_{split_id:03d}"
    directory.mkdir(parents=True, exist_ok=True)

    files = {
        "model": _pickle(directory / "model.pkl", model),
        "preprocessor": _pickle(directory / "preprocessor.pkl", preprocessor),
    }
    if pca is not None:
        files["pca"] = _pickle(directory / "pca.pkl", pca)

    model_config = _model_config(model=model, result=result)
    config_path = directory / "config_snapshot.json"
    _write_json(
        config_path,
        {
            "run": run_config,
            "model": model_config,
            "target_column": target_column,
        },
    )
    files["config_snapshot"] = _file_record(config_path)

    record = {
        "artifact_id": artifact_id,
        "model_name": model_name,
        "split_id": split_id,
        "git_sha": git_sha,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": target_column,
        "train_range": _date_range(train_metadata),
        "validation_range": _date_range(validation_metadata),
        "files": files,
    }
    if weights := _qml_weights(model):
        record["qml_parameters"] = weights
    _write_json(directory / "metadata.json", record)
    return record


def save_artifact_manifest(root: str | Path, records: list[dict[str, Any]]) -> Path:
    """Save the index mapping prediction artifact IDs to fitted bundles."""
    path = Path(root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"schema_version": 1, "artifacts": records})
    return path


def _pickle(path: Path, value: Any) -> dict[str, Any]:
    with path.open("wb") as file:
        pickle.dump(value, file)
    return _file_record(path)


def _file_record(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _model_config(*, model: Any, result: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if hasattr(model, "get_params"):
        config["estimator_parameters"] = model.get_params(deep=True)
    if hasattr(result, "parameters"):
        config["selected_parameters"] = result.parameters
    if hasattr(result, "config"):
        config["qml_config"] = result.config
    return config


def _qml_weights(model: Any) -> dict[str, Any] | None:
    weights = getattr(model, "weights_", None)
    if weights is None:
        return None
    return {"weights": weights.tolist()}


def _date_range(metadata: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(metadata["date"], errors="coerce").dropna()
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "rows": len(metadata),
    }
