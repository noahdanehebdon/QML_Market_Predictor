"""Persistence boundary for QML PCA outputs and fitted artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from market_qml.qml.pca_types import GroupedPCAArtifact, PCAArtifact


def save_frame(frame: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def save_artifacts(
    artifacts: dict[int, PCAArtifact | GroupedPCAArtifact],
    artifact_dir: str | Path,
) -> list[Path]:
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for split_id, artifact in sorted(artifacts.items()):
        path = directory / f"pca_split_{split_id:03d}.pkl"
        with path.open("wb") as handle:
            pickle.dump(artifact, handle)
        paths.append(path)
    return paths


def load_artifact(path: str | Path) -> PCAArtifact | GroupedPCAArtifact:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, PCAArtifact | GroupedPCAArtifact):
        raise TypeError(f"Unexpected PCA artifact type: {type(artifact)!r}")
    return artifact
