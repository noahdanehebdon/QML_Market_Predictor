import pandas as pd

from market_qml.reporting.baseline_evidence import compare_to_naive_baselines


def test_complex_model_requires_practical_and_statistical_margin():
    rows = []
    for split in range(4):
        for model, scores in [("momentum_rank", [0, 1, 2, 3]), ("xgboost_ranker", [0, 1, 2, 3])]:
            for index, score in enumerate(scores):
                rows.append({"model_name": model, "split_id": split, "date": pd.Timestamp("2024-01-01"), "y_score": score, "forward_excess_return": index if model == "xgboost_ranker" else [1, 0, 3, 2][index]})
    result = compare_to_naive_baselines(pd.DataFrame(rows), practical_margin=0.01, bootstrap_samples=100)
    assert result.iloc[0]["naive_model"] == "momentum_rank"
    assert result.iloc[0]["beats_naive"]
