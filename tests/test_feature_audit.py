import warnings

import numpy as np
import pandas as pd

from market_qml.features.audit import audit_features, feature_family


def _inputs():
    dates = pd.date_range("2024-01-01", periods=8)
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    rows, labels = [], []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            signal = float(symbol_index)
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "return_5d": signal,
                    "return_10d": signal * 2,
                    "treasury_10y": 4 + date_index / 10,
                    "sec_recent_filing_30d": symbol_index == 0,
                    "sparse_feature": np.nan if symbol_index < 2 else signal,
                }
            )
            labels.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "forward_excess_return_5d": signal,
                    "outperform_spy_5d": symbol_index >= 2,
                }
            )
    splits = pd.DataFrame(
        {
            "split_id": [0, 1],
            "train_start_date": [dates[0], dates[1]],
            "train_end_date": [dates[2], dates[3]],
            "validation_start_date": [dates[3], dates[4]],
            "validation_end_date": [dates[4], dates[5]],
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(labels), splits


def test_audit_profiles_quality_stability_redundancy_and_ablations():
    features, labels, splits = _inputs()
    result = audit_features(features, labels, splits)

    sparse = result.quality.loc[result.quality["feature"].eq("sparse_feature")]
    assert sparse["train_missing_rate"].eq(0.5).all()
    assert not sparse["revision_tracking_available"].any()
    assert sparse["train_revision_count"].isna().all()
    stable = result.stability.set_index("feature")
    assert stable.loc["return_5d", "stable_evidence"]
    assert stable.loc["return_5d", "validation_median_ic"] == 1.0
    pairs = set(map(frozenset, result.redundancy[["feature_a", "feature_b"]].values))
    assert frozenset({"return_5d", "return_10d"}) in pairs
    assert {"technical", "macro", "filing_events"} <= set(result.ablations["family"])


def test_validation_sign_is_reported_separately_from_training():
    features, labels, splits = _inputs()
    validation_dates = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-06"])
    mask = labels["date"].isin(validation_dates)
    labels.loc[mask, "forward_excess_return_5d"] *= -1

    result = audit_features(features, labels, splits)
    rows = result.predictive.loc[result.predictive["feature"].eq("return_5d")]

    assert rows.loc[rows["period"].eq("train"), "rank_ic"].iloc[0] == 1.0
    assert rows.loc[rows["period"].eq("validation"), "rank_ic"].iloc[0] == -1.0
    assert not result.stability.set_index("feature").loc["return_5d", "stable_evidence"]


def test_feature_family_is_explicit_and_deterministic():
    assert feature_family("return_20d") == "technical"
    assert feature_family("spy_return_5d") == "benchmark"
    assert feature_family("treasury_10y") == "macro"
    assert feature_family("sec_recent_filing_30d") == "filing_events"
    assert feature_family("revenue_growth") == "fundamentals"


def test_audit_reports_non_finite_values_without_quantile_warnings():
    features, labels, splits = _inputs()
    features.loc[0, "return_5d"] = np.inf
    features.loc[13, "return_5d"] = -np.inf

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = audit_features(features, labels, splits)

    assert not [
        warning
        for warning in captured
        if "invalid value encountered" in str(warning.message)
    ]
    quality = result.quality.loc[result.quality["feature"].eq("return_5d")]
    assert quality["train_non_finite_rate"].gt(0).any()
    assert quality["validation_non_finite_rate"].gt(0).any()
    assert quality["distribution_psi"].dropna().map(np.isfinite).all()
