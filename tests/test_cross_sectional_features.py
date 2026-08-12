import pandas as pd

from market_qml.features.cross_sectional import add_cross_sectional_features


def test_add_cross_sectional_features_ranks_within_each_date_and_flags_missing():
    features = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "date": [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3,
            "return_5d": [1.0, 2.0, None, 30.0, 10.0, 20.0],
        }
    )

    result = add_cross_sectional_features(features)

    ranks = result["return_5d_xs_rank"]
    assert ranks.dropna().tolist() == [0.5, 1.0, 1.0, 1 / 3, 2 / 3]
    assert pd.isna(ranks.iloc[2])
    assert result["return_5d_missing"].tolist() == [0, 0, 1, 0, 0, 0]


def test_cross_sectional_features_add_robust_and_sector_neutral_context():
    features = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "sector": ["tech", "tech", "health", "health"],
            "return_5d": [1.0, 3.0, 10.0, 20.0],
        }
    )

    result = add_cross_sectional_features(features)

    assert result["return_5d_xs_robust_z"].between(-5, 5).all()
    assert result["return_5d_sector_rank"].tolist() == [0.5, 1.0, 0.5, 1.0]
