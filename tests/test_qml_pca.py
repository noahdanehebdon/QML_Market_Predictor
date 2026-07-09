import pandas as pd
import pytest

from market_qml.qml.pca import (
    PCAArtifact,
    build_qml_pca_features,
    load_qml_pca_artifact,
    save_qml_pca_artifacts,
    save_qml_pca_diagnostics,
    save_qml_pca_features,
)


def _features() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    for symbol, offset in [("AAPL", 1.0), ("MSFT", 2.0), ("NVDA", 3.0)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "feature_a": offset + index,
                    "feature_b": offset * 2 + index * 0.5,
                    "feature_c": (-offset) + index * 0.25,
                }
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    for symbol, offset in [("AAPL", 1), ("MSFT", 2), ("NVDA", 3)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "forward_return_5d": 0.01 * (offset + index),
                    "forward_excess_return_5d": 0.005 * (offset - index),
                    "outperform_spy_5d": int((offset + index) % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def _splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_id": [0],
            "train_start_date": [pd.Timestamp("2024-01-01")],
            "train_end_date": [pd.Timestamp("2024-01-03")],
            "validation_start_date": [pd.Timestamp("2024-01-04")],
            "validation_end_date": [pd.Timestamp("2024-01-05")],
            "train_days": [3],
            "validation_days": [2],
            "train_rows": [9],
            "validation_rows": [6],
        }
    )


def test_build_qml_pca_features_fits_train_only_and_outputs_configured_components():
    result = build_qml_pca_features(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        n_components=2,
    )

    component_columns = ["pca_00", "pca_01"]

    assert list(result.features.columns) == [
        "symbol",
        "date",
        "split_id",
        "sample_role",
        "target",
        *component_columns,
    ]
    assert set(result.features["sample_role"]) == {"train", "validation"}
    assert result.features[component_columns].shape[1] == 2
    assert set(result.artifacts) == {0}
    assert isinstance(result.artifacts[0], PCAArtifact)
    assert result.artifacts[0].pca.n_components_ == 2
    assert result.diagnostics["component"].tolist() == ["pca_00", "pca_01"]
    assert result.diagnostics["n_components"].unique().tolist() == [2]
    assert result.diagnostics["n_original_features"].unique().tolist() == [3]


def test_build_qml_pca_features_preprocessor_ignores_validation_distribution():
    features = _features()
    validation_mask = features["date"] >= pd.Timestamp("2024-01-04")
    features.loc[validation_mask, "feature_a"] = 9999.0

    result = build_qml_pca_features(
        features=features,
        labels=_labels(),
        splits=_splits(),
        n_components=2,
    )
    artifact = result.artifacts[0]

    assert artifact.preprocessor.means["feature_a"] != pytest.approx(9999.0)
    assert artifact.preprocessor.means["feature_a"] == pytest.approx(3.0)


def test_build_qml_pca_features_rejects_invalid_component_counts():
    with pytest.raises(ValueError, match="positive"):
        build_qml_pca_features(
            features=_features(),
            labels=_labels(),
            splits=_splits(),
            n_components=0,
        )

    with pytest.raises(ValueError, match="feature columns"):
        build_qml_pca_features(
            features=_features(),
            labels=_labels(),
            splits=_splits(),
            n_components=4,
        )


def test_qml_pca_outputs_can_be_saved(tmp_path):
    result = build_qml_pca_features(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        n_components=2,
    )
    feature_path = tmp_path / "qml_pca_features.parquet"
    diagnostics_path = tmp_path / "qml_pca_explained_variance.parquet"
    artifact_dir = tmp_path / "artifacts"

    save_qml_pca_features(result.features, feature_path)
    save_qml_pca_diagnostics(result.diagnostics, diagnostics_path)
    artifact_paths = save_qml_pca_artifacts(result.artifacts, artifact_dir)

    saved_features = pd.read_parquet(feature_path)
    saved_diagnostics = pd.read_parquet(diagnostics_path)
    loaded_artifact = load_qml_pca_artifact(artifact_paths[0])

    assert feature_path.exists()
    assert diagnostics_path.exists()
    assert len(artifact_paths) == 1
    assert list(saved_features.columns) == list(result.features.columns)
    assert list(saved_diagnostics.columns) == list(result.diagnostics.columns)
    assert loaded_artifact.split_id == 0
    assert loaded_artifact.pca.n_components_ == 2
