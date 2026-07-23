import numpy as np
import pandas as pd

from market_qml.qml.selected_features import build_selected_qml_features


def _inputs():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    feature_rows = []
    label_rows = []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(["A", "B", "C", "SPY"]):
            signal = symbol_index - 1.5 + date_index * 0.01
            feature_row = {"symbol": symbol, "date": date}
            for feature_index in range(12):
                feature_row[f"feature_{feature_index:02d}"] = signal * (
                    1.0 if feature_index == 0 else 0.05 * feature_index
                ) + np.sin(date_index + feature_index)
            feature_rows.append(feature_row)
            excess = signal * 0.01
            label_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "forward_return_5d": excess + 0.002,
                    "forward_excess_return_5d": excess,
                    "outperform_spy_5d": int(excess > 0),
                }
            )
    splits = pd.DataFrame(
        {
            "split_id": [0],
            "train_start_date": [dates[0]],
            "train_end_date": [dates[19]],
            "validation_start_date": [dates[20]],
            "validation_end_date": [dates[24]],
        }
    )
    diagnostics = pd.DataFrame({"split_id": [0], "rank": [1], "feature_count": [10]})
    return pd.DataFrame(feature_rows), pd.DataFrame(label_rows), splits, diagnostics


def test_selected_qml_features_use_classical_budget_and_outer_training_only():
    features, labels, splits, diagnostics = _inputs()
    result = build_selected_qml_features(
        features=features,
        labels=labels,
        splits=splits,
        selection_diagnostics=diagnostics,
    )

    assert len(result.manifest) == 8
    assert result.manifest["qubit"].tolist() == list(range(8))
    assert set(result.manifest["classical_candidate_feature_count"]) == {10}
    assert len([c for c in result.features if c.startswith("selected_feature_")]) == 8
    assert set(result.features["sample_role"]) == {"train", "validation"}

    changed_labels = labels.copy()
    changed_labels.loc[
        changed_labels["date"] >= splits.iloc[0].validation_start_date,
        "forward_excess_return_5d",
    ] *= -1000
    changed = build_selected_qml_features(
        features=features,
        labels=changed_labels,
        splits=splits,
        selection_diagnostics=diagnostics,
    )
    assert (
        result.manifest["source_feature"].tolist()
        == changed.manifest["source_feature"].tolist()
    )
