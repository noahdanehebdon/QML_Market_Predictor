import pandas as pd

from market_qml.features.cross_sectional import add_cross_sectional_features


def test_add_cross_sectional_features_ranks_within_each_date_and_flags_missing():
    features = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "date": [pd.Timestamp("2024-01-01")] * 3
            + [pd.Timestamp("2024-01-02")] * 3,
            "return_5d": [1.0, 2.0, None, 30.0, 10.0, 20.0],
        }
    )

    result = add_cross_sectional_features(features)

    ranks = result["return_5d_xs_rank"]
    assert ranks.dropna().tolist() == [0.5, 1.0, 1.0, 1 / 3, 2 / 3]
    assert pd.isna(ranks.iloc[2])
    assert result["return_5d_missing"].tolist() == [0, 0, 1, 0, 0, 0]
