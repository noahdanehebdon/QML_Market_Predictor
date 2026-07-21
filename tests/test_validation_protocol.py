import json

import pandas as pd
import pytest

from market_qml.backtest.validation import (
    log_locked_test_access,
    paired_model_comparisons,
    partition_locked_test,
    prediction_date_block_metrics,
)


def test_partition_locked_test_is_non_overlapping_and_embargoed():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20), "value": range(20)})

    development, locked, manifest = partition_locked_test(
        data, locked_test_days=5, embargo_days=2
    )

    assert development.date.max() == pd.Timestamp("2024-01-13")
    assert locked.date.min() == pd.Timestamp("2024-01-16")
    assert set(development.date).isdisjoint(locked.date)
    assert manifest["locked_test_accessed"] is False


def test_locked_test_access_requires_reason_and_writes_audit(tmp_path):
    manifest = {"protocol_version": "locked-test-v1", "locked_test_accessed": False}
    with pytest.raises(ValueError, match="reason"):
        log_locked_test_access(manifest, reason="", audit_path=tmp_path / "audit.json")

    path = tmp_path / "audit.json"
    record = log_locked_test_access(manifest, reason="final milestone evaluation", audit_path=path)
    assert record["locked_test_accessed"] is True
    assert json.loads(path.read_text())["access_reason"] == "final milestone evaluation"


def test_paired_comparisons_report_effect_uncertainty_and_adjusted_test():
    rows = []
    for split_id, baseline in enumerate([0.50, 0.51, 0.49, 0.50]):
        rows.extend(
            [
                {"model_name": "baseline", "split_id": split_id, "roc_auc": baseline},
                {"model_name": "better", "split_id": split_id, "roc_auc": baseline + 0.05},
                {"model_name": "same", "split_id": split_id, "roc_auc": baseline + 0.001},
            ]
        )

    result = paired_model_comparisons(
        pd.DataFrame(rows), metric="roc_auc", baseline_model="baseline",
        bootstrap_iterations=500, practical_threshold=0.02,
    ).set_index("candidate_model")

    assert result.loc["better", "mean_difference"] == pytest.approx(0.05)
    assert bool(result.loc["better", "practically_meaningful"])
    assert not bool(result.loc["same", "practically_meaningful"])
    assert 0 <= result["holm_adjusted_p_value"].max() <= 1


def test_prediction_metrics_use_non_overlapping_date_blocks():
    rows = []
    for model, offset in [("a", 0.0), ("b", 0.1)]:
        for day in range(8):
            rows.append({"model_name": model, "split_id": 0,
                         "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                         "y_true": day % 2, "y_score": 0.2 + 0.6 * (day % 2) + offset})
    result = prediction_date_block_metrics(pd.DataFrame(rows), block_days=4)
    assert result.groupby("model_name").size().eq(2).all()
    assert (result.block_end_date - result.block_start_date).dt.days.le(3).all()
