import sys

import pandas as pd
import yaml

from scripts import build_walk_forward_splits


def test_main_builds_splits_from_non_default_target_horizon(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=40)
    features = pd.DataFrame(
        {
            "symbol": ["AAPL"] * len(dates),
            "date": dates,
            "signal": range(len(dates)),
        }
    )
    labels = pd.DataFrame(
        {
            "symbol": ["AAPL"] * len(dates),
            "date": dates,
            "outperform_spy_10d": [index % 2 for index in range(len(dates))],
        }
    )
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.parquet"
    config_path = tmp_path / "backtest.yaml"
    output_path = tmp_path / "splits.parquet"
    features.to_parquet(feature_path, index=False)
    labels.to_parquet(label_path, index=False)
    config_path.write_text(
        yaml.safe_dump(
            {
                "walk_forward": {
                    "train_window_days": 4,
                    "validation_window_days": 2,
                    "locked_test_days": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_walk_forward_splits",
            "--features",
            str(feature_path),
            "--labels",
            str(label_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--target-horizon-days",
            "10",
            "--purge-days",
            "10",
            "--embargo-days",
            "10",
        ],
    )

    build_walk_forward_splits.main()

    result = pd.read_parquet(output_path)
    assert output_path.exists()
    assert not result.empty
    assert result["purge_days"].eq(10).all()
    assert result["embargo_days"].eq(10).all()
    assert not result["locked_test_accessed"].any()
