import numpy as np
import pandas as pd

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.preprocessing import fit_transform_train_validation
from market_qml.models.xgboost_baselines import (
    train_xgboost_classifier,
    train_xgboost_ranker,
)


def _data(classification=True):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=15)
    symbols = [f"S{i}" for i in range(8)]

    def part(selected):
        meta = pd.DataFrame(
            [(s, d) for d in selected for s in symbols], columns=["symbol", "date"]
        )
        signal = np.tile(np.arange(8), len(selected)) + rng.normal(0, 0.1, len(meta))
        X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(meta))})
        target = pd.Series((signal >= 4).astype(int) if classification else signal)
        meta["forward_return_5d"] = signal / 100
        meta["forward_excess_return_5d"] = signal / 100
        return ModelingDataset(X, target, meta)

    return fit_transform_train_validation(
        TrainValidationDatasets(part(dates[:12]), part(dates[12:]))
    )


def test_xgboost_classifier_uses_early_stopping_and_date_weights():
    result = train_xgboost_classifier(_data(), split_id=2)
    assert result.parameters["date_aware_weights"] is True
    assert result.selection_diagnostics.iloc[0]["objective"] == "logloss"
    assert result.predictions["model_name"].unique().tolist() == ["xgboost_classifier"]


def test_xgboost_ranker_groups_queries_by_date():
    result = train_xgboost_ranker(_data(classification=False), split_id=3)
    assert result.parameters["grouped_by"] == "date"
    assert result.selection_diagnostics.iloc[0]["objective"] == "rank:ndcg"
    assert result.predictions.groupby("date").size().eq(8).all()
