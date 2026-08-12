import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning

from market_qml.models.ensembles import (
    build_chronological_ensembles,
    eligible_regime_weights,
)


def _predictions():
    rows = []
    for split in range(3):
        for day in range(3):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=split * 3 + day)
            for symbol_index, symbol in enumerate(["A", "B", "C", "D"]):
                truth = symbol_index >= 2
                for model, offset in [("classifier_a", 0.0), ("classifier_b", 0.08)]:
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": date,
                            "y_true": int(truth),
                            "y_score": np.clip(0.15 + 0.7 * truth + offset, 0.01, 0.99),
                            "forward_return": symbol_index / 100,
                            "forward_excess_return": symbol_index / 100,
                            "model_name": model,
                            "split_id": split,
                        }
                    )
    return pd.DataFrame(rows)


def test_ensembles_use_only_prior_fold_history_and_emit_sensitivity():
    result = build_chronological_ensembles(_predictions(), min_history_rows=8)
    diagnostics = result.diagnostics
    first = diagnostics.loc[diagnostics["split_id"].eq(0)]
    later = diagnostics.loc[diagnostics["split_id"].eq(1)]
    assert first["weight_source"].eq("equal_weight_insufficient_history").all()
    assert later["weight_source"].eq("chronological_constrained").all()
    assert (
        diagnostics.groupby(["task", "split_id"])["weight"].sum().round(8).eq(1).all()
    )
    assert set(result.predictions["model_name"]) == {
        "classification_simple_average_ensemble",
        "classification_rank_average_ensemble",
        "classification_constrained_stack_ensemble",
    }
    assert result.sensitivity["removed_model"].nunique() == 2


def test_future_fold_changes_cannot_change_earlier_ensemble_scores():
    original = _predictions()
    changed = original.copy()
    changed.loc[changed["split_id"].eq(2), "y_score"] = (
        1 - changed.loc[changed["split_id"].eq(2), "y_score"]
    )
    left = build_chronological_ensembles(original, min_history_rows=8).predictions
    right = build_chronological_ensembles(changed, min_history_rows=8).predictions
    keys = ["model_name", "split_id", "symbol", "date"]
    earlier_left = left.loc[left["split_id"].lt(2)].sort_values(keys)
    earlier_right = right.loc[right["split_id"].lt(2)].sort_values(keys)
    assert np.allclose(earlier_left["y_score"], earlier_right["y_score"])


def test_regime_conditioning_requires_sufficient_samples():
    regimes = pd.DataFrame({"regime": ["high"] * 99 + ["low"] * 100})
    result = eligible_regime_weights(regimes, regime_column="regime", min_rows=100)
    eligible = result.set_index("regime")["eligible_for_conditioning"]
    assert not eligible["high"]
    assert eligible["low"]


def test_constant_removal_scores_do_not_emit_correlation_warnings():
    predictions = _predictions()
    predictions.loc[predictions["model_name"].eq("classifier_b"), "y_score"] = 0.5

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = build_chronological_ensembles(predictions, min_history_rows=8)

    assert not any(
        isinstance(warning.message, ConstantInputWarning) for warning in captured
    )
    assert result.sensitivity["score_correlation"].isna().any()
