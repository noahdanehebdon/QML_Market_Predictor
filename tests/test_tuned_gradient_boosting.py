import numpy as np
import pandas as pd

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.preprocessing import fit_transform_train_validation
from market_qml.models.tuned_gradient_boosting import (
    MODEL_NAME,
    train_tuned_gradient_boosting_regressor,
)


def _data():
    rng = np.random.default_rng(42)
    train_rows = 100
    validation_rows = 20
    train_X = pd.DataFrame(
        {
            "signal": np.linspace(-2, 2, train_rows),
            "noise_1": rng.normal(size=train_rows),
            "noise_2": rng.normal(size=train_rows),
        }
    )
    validation_X = pd.DataFrame(
        {
            "signal": np.linspace(-1.5, 1.5, validation_rows),
            "noise_1": rng.normal(size=validation_rows),
            "noise_2": rng.normal(size=validation_rows),
        }
    )
    train = ModelingDataset(
        X=train_X,
        y=pd.Series(train_X["signal"] * 0.02),
        metadata=pd.DataFrame(
            {
                "symbol": ["AAPL"] * train_rows,
                "date": pd.date_range("2020-01-01", periods=train_rows),
                "forward_return_5d": train_X["signal"] * 0.02,
                "forward_excess_return_5d": train_X["signal"] * 0.02,
            }
        ),
    )
    validation = ModelingDataset(
        X=validation_X,
        y=pd.Series(validation_X["signal"] * 0.02),
        metadata=pd.DataFrame(
            {
                "symbol": ["AAPL"] * validation_rows,
                "date": pd.date_range("2021-01-01", periods=validation_rows),
                "forward_return_5d": validation_X["signal"] * 0.02,
                "forward_excess_return_5d": validation_X["signal"] * 0.02,
            }
        ),
    )
    return fit_transform_train_validation(
        TrainValidationDatasets(train=train, validation=validation)
    )


def test_tuned_gradient_boosting_selects_using_inner_chronological_validation():
    result = train_tuned_gradient_boosting_regressor(
        _data(),
        feature_counts=(1, 3),
        learning_rates=(0.05,),
        max_leaf_nodes_values=(7,),
        l2_values=(0.0,),
        min_samples_leaf=2,
        max_iter=30,
    )

    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.parameters["selected_features"] == ["signal"]
    assert set(result.selection_diagnostics["feature_count"]) == {1, 3}
    assert result.selection_diagnostics["inner_validation_start"].min() > pd.Timestamp(
        "2020-01-01"
    )
