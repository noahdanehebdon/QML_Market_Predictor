import pandas as pd

from market_qml.qml.regime_analysis import analyze_predictions_by_regime, save_regime_analysis


def _predictions():
    rows = []
    dates = pd.date_range("2024-01-01", periods=8)
    for model_index, model in enumerate(["qcnn", "vqc", "qsvm", "logistic_regression"]):
        for index, date in enumerate(dates):
            target = index % 2
            score = (0.8 if target else 0.2) if model == "qcnn" else 0.5 + model_index * 0.01
            rows.append({"symbol": f"S{index}", "date": date, "y_true": target,
                         "y_score": score, "model_name": model, "split_id": index // 4,
                         "forward_return": .02 if target else -.01,
                         "forward_excess_return": .01 if target else -.02})
    return pd.DataFrame(rows)


def _regimes():
    dates = pd.date_range("2024-01-01", periods=8)
    return pd.DataFrame({"date": dates,
                         "volatility_regime": ["low_volatility"] * 4 + ["high_volatility"] * 4,
                         "rate_regime": ["rising_rates", "falling_rates"] * 4,
                         "yield_curve_regime": ["normal_curve"] * 6 + ["inverted_curve"] * 2})


def test_regime_analysis_slices_models_and_compares_qcnn():
    result = analyze_predictions_by_regime(_predictions(), _regimes(), minimum_rows=2)
    assert set(result.metrics.regime_type) == {"volatility_regime", "rate_regime", "yield_curve_regime"}
    assert set(result.metrics.model_name) == {"qcnn", "vqc", "qsvm", "logistic_regression"}
    low_qcnn = result.metrics.query("regime == 'low_volatility' and model_name == 'qcnn'").iloc[0]
    assert low_qcnn.roc_auc == 1.0
    assert not result.model_differences.empty
    assert result.model_differences.roc_auc_difference.max() > 0


def test_small_or_single_class_slices_have_no_auc():
    result = analyze_predictions_by_regime(_predictions(), _regimes(), minimum_rows=3)
    inverted = result.metrics.query("regime == 'inverted_curve'")
    assert inverted.roc_auc.isna().all()
    assert (~inverted.meets_minimum_rows).all()


def test_regime_join_is_date_keyed_and_outputs_save(tmp_path):
    shuffled = _regimes().sample(frac=1, random_state=3)
    result = analyze_predictions_by_regime(_predictions(), shuffled, minimum_rows=2)
    assert result.joined_predictions.volatility_regime.notna().all()
    paths = save_regime_analysis(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    assert "QCNN pattern" in paths["report"].read_text()


def test_missing_regime_dates_remain_explicitly_unlabeled():
    result = analyze_predictions_by_regime(_predictions(), _regimes().iloc[:-1], minimum_rows=2)
    assert result.joined_predictions.volatility_regime.isna().sum() == 4
