import pandas as pd
import pytest

from market_qml.qml.pca import (
    GroupedPCAArtifact,
    PCAArtifact,
    allocate_group_components,
    build_grouped_qml_pca_features,
    build_qml_pca_features,
    infer_feature_groups,
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
    assert result.diagnostics["group"].unique().tolist() == ["global"]
    assert result.diagnostics["component"].tolist() == ["global_pca_00", "global_pca_01"]
    assert result.diagnostics["n_components"].unique().tolist() == [2]
    assert result.diagnostics["n_original_features"].unique().tolist() == [3]


def test_build_grouped_qml_pca_features_outputs_grouped_components():
    features = _features().rename(
        columns={
            "feature_a": "return_1d",
            "feature_b": "realized_vol_5d",
            "feature_c": "avg_dollar_volume_5d",
        }
    )

    result = build_grouped_qml_pca_features(
        features=features,
        labels=_labels(),
        splits=_splits(),
        n_components=3,
    )

    component_columns = [
        "returns_momentum_pca_00",
        "volatility_pca_00",
        "volume_liquidity_pca_00",
    ]

    assert list(result.features.columns) == [
        "symbol",
        "date",
        "split_id",
        "sample_role",
        "target",
        *component_columns,
    ]
    assert set(result.features["target"]) <= {0, 1}
    assert result.diagnostics["group"].tolist() == [
        "returns_momentum",
        "volatility",
        "volume_liquidity",
    ]
    assert isinstance(result.artifacts[0], GroupedPCAArtifact)
    assert result.artifacts[0].component_columns == component_columns


def test_build_grouped_qml_pca_features_supports_regression_ranking_target():
    result = build_grouped_qml_pca_features(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        n_components=2,
        target_column="forward_excess_return_5d",
    )

    assert result.artifacts[0].target_column == "forward_excess_return_5d"
    assert result.features["target"].dtype.kind == "f"
    assert result.features["target"].nunique() > 2


def test_group_component_allocation_respects_budget_and_capacity():
    groups = infer_feature_groups(
        [
            "open",
            "close",
            "return_1d",
            "momentum_20d",
            "realized_vol_5d",
            "avg_dollar_volume_20d",
            "treasury_10y",
            "assets",
        ]
    )

    allocation = allocate_group_components(
        group_columns=groups,
        n_components=5,
        train_rows=20,
    )

    assert sum(allocation.values()) == 5
    assert allocation.get("raw_price", 0) <= 2
    assert allocation.get("returns_momentum", 0) <= 2
    assert all(count <= len(groups[group]) for group, count in allocation.items())
    assert set(allocation) <= {group for group, columns in groups.items() if columns}


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
    assert isinstance(loaded_artifact, PCAArtifact)
    assert loaded_artifact.pca.n_components_ == 2


def test_grouped_qml_pca_outputs_can_be_saved(tmp_path):
    result = build_grouped_qml_pca_features(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        n_components=2,
    )
    feature_path = tmp_path / "qml_grouped_pca_features.parquet"
    diagnostics_path = tmp_path / "qml_grouped_pca_explained_variance.parquet"
    artifact_dir = tmp_path / "grouped_artifacts"

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
    assert isinstance(loaded_artifact, GroupedPCAArtifact)
    assert loaded_artifact.split_id == 0
