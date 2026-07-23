import pandas as pd
import pytest

from market_qml.qml.definitive_comparison import (
    build_definitive_comparison,
    save_definitive_comparison,
)


def _predictions(models=("logistic_regression", "vqc")):
    rows = []
    for split_id in (0, 1):
        for day in range(8):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=split_id * 10 + day)
            for symbol_index, symbol in enumerate(("A", "B", "C", "D")):
                target = (day + symbol_index) % 2
                for model in models:
                    confidence = 0.8 if model == "vqc" else 0.65
                    score = confidence if target else 1 - confidence
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": date,
                            "y_true": target,
                            "y_score": score,
                            "forward_return": 0.02 if target else -0.01,
                            "forward_excess_return": 0.015 if target else -0.015,
                            "model_name": model,
                            "split_id": split_id,
                        }
                    )
    return pd.DataFrame(rows)


def test_unaccessed_locked_test_prevents_quantum_advantage_and_saves_outputs(tmp_path):
    predictions = _predictions()
    result = build_definitive_comparison(
        predictions,
        predictions,
        locked_test_manifest={"locked_test_accessed": False},
        bootstrap_iterations=20,
    )

    assert set(result.aggregate_metrics["lane"]) == {"equal_input", "best_available"}
    assert result.conclusion["quantum_advantage_demonstrated"] is False
    assert result.conclusion["strongest_classical_system"] == "logistic_regression"

    save_definitive_comparison(result, tmp_path / "private", tmp_path / "public")
    assert (tmp_path / "private" / "paired_comparisons.parquet").exists()
    public = (tmp_path / "public" / "definitive_summary.md").read_text()
    assert "No statistically and practically defensible" in public


def test_equal_input_lane_rejects_different_outer_rows():
    predictions = _predictions()
    remove = predictions.index[
        (predictions["model_name"] == "vqc") & (predictions["split_id"] == 0)
    ][0]

    with pytest.raises(ValueError, match="identical outer rows"):
        build_definitive_comparison(
            predictions.drop(index=remove),
            predictions,
            bootstrap_iterations=10,
        )
