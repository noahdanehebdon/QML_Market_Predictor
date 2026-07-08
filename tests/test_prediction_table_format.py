import pandas as pd
import pytest

from market_qml.models.predictions import (
    REQUIRED_PREDICTION_COLUMNS,
    build_prediction_table,
    save_predictions,
)


def test_build_prediction_table_outputs_standard_columns():
    metadata = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "forward_return_5d": [0.02, -0.01],
            "forward_excess_return_5d": [0.01, -0.02],
        }
    )

    result = build_prediction_table(
        metadata=metadata,
        y_true=pd.Series([1, 0]),
        y_score=[0.8, 0.2],
        model_name="demo_model",
        split_id=3,
    )

    assert list(result.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result["symbol"].tolist() == ["AAPL", "MSFT"]
    assert result["forward_return"].tolist() == [-0.01, 0.02]
    assert result["forward_excess_return"].tolist() == [-0.02, 0.01]
    assert result["model_name"].unique().tolist() == ["demo_model"]
    assert result["split_id"].unique().tolist() == [3]


def test_build_prediction_table_rejects_missing_return_metadata():
    metadata = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": pd.to_datetime(["2024-01-01"]),
        }
    )

    with pytest.raises(ValueError, match="forward return"):
        build_prediction_table(
            metadata=metadata,
            y_true=pd.Series([1]),
            y_score=[0.8],
            model_name="demo_model",
            split_id=0,
        )


def test_save_predictions_rejects_incomplete_tables(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        save_predictions(
            pd.DataFrame({"symbol": ["AAPL"]}),
            tmp_path / "predictions.parquet",
        )
