import pandas as pd

from market_qml.features.conditional import add_conditional_features


def test_conditional_features_create_only_available_interactions():
    features = pd.DataFrame(
        {
            "rolling_beta_20d_vs_spy": [1.2],
            "treasury_10y_change_20d": [0.5],
            "debt_ratio": [0.4],
            "amihud_illiquidity_20d": [0.01],
            "realized_vol_20d": [0.2],
        }
    )
    result = add_conditional_features(features)
    assert result.loc[0, "rate_beta_interaction_20d"] == 0.6
    assert result.loc[0, "rate_debt_interaction_20d"] == 0.2
    assert result.loc[0, "stress_liquidity_interaction_20d"] == 0.002
    assert "inflation_margin_interaction_63d" not in result


def test_conditional_features_do_not_mutate_inputs():
    features = pd.DataFrame({"debt_ratio": [0.4], "treasury_10y_change_20d": [0.5]})
    original = features.copy()
    add_conditional_features(features)
    pd.testing.assert_frame_equal(features, original)
