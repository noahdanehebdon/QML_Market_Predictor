import pandas as pd
import pytest

from market_qml.features.macro import (
    add_macro_features,
    build_macro_feature_table,
    merge_macro_features,
)


def _macro_daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treasury_10y": [4.0, 4.1, 4.3],
            "treasury_2y": [3.8, 3.9, 4.0],
            "fed_funds": [5.25, 5.25, 5.30],
            "cpi_all_items_sa": [300.0, 303.0, 306.0],
            "unemployment_rate": [3.8, 3.9, 3.7],
            "industrial_production": [100.0, 101.0, 103.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def _market_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "SPY", "SPY"],
            "date": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-03", "2024-01-04"]
            ),
            "close": [100.0, 101.0, 400.0, 401.0],
            "return_1d": [0.01, 0.01, 0.005, 0.0025],
        }
    )


def test_add_macro_features_creates_levels_spreads_and_changes():
    result = add_macro_features(
        _macro_daily(),
        rate_change_windows=[1],
        macro_change_windows=[1],
    )

    assert list(result["date"]) == list(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    assert result.loc[0, "yield_spread_10y_2y"] == pytest.approx(0.2)
    assert result.loc[1, "treasury_10y_change_1d"] == pytest.approx(0.1)
    assert result.loc[2, "yield_spread_10y_2y_change_1d"] == pytest.approx(0.1)
    assert result.loc[2, "fed_funds_change_1d"] == pytest.approx(0.05)
    assert result.loc[1, "cpi_inflation_1d"] == pytest.approx(0.01)
    assert result.loc[2, "unemployment_rate_change_1d"] == pytest.approx(-0.2)
    assert result.loc[2, "industrial_production_growth_1d"] == pytest.approx(2 / 101)


def test_merge_macro_features_preserves_symbol_rows_and_merges_by_date():
    macro_features = add_macro_features(
        _macro_daily(),
        rate_change_windows=[1],
        macro_change_windows=[1],
    )

    result = merge_macro_features(_market_features(), macro_features)

    assert len(result) == 4
    assert result[result["date"] == pd.Timestamp("2024-01-03")][
        "treasury_10y"
    ].tolist() == [4.1, 4.1]
    assert "yield_spread_10y_2y" in result.columns


def test_macro_features_do_not_use_future_values():
    result = add_macro_features(
        _macro_daily(),
        rate_change_windows=[1],
        macro_change_windows=[1],
    )

    assert pd.isna(result.loc[0, "treasury_10y_change_1d"])
    assert result.loc[1, "treasury_10y_change_1d"] == pytest.approx(0.1)


def test_add_macro_features_requires_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        add_macro_features(pd.DataFrame({"treasury_10y": [4.0]}))


def test_build_macro_feature_table_saves_output(tmp_path):
    feature_path = tmp_path / "benchmark_relative_features.parquet"
    macro_path = tmp_path / "macro_daily.parquet"
    output_path = tmp_path / "macro_features.parquet"

    _market_features().to_parquet(feature_path, index=False)
    _macro_daily().to_parquet(macro_path)

    result = build_macro_feature_table(
        feature_path=feature_path,
        macro_daily_path=macro_path,
        output_path=output_path,
        rate_change_windows=[1],
        macro_change_windows=[1],
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert "cpi_inflation_1d" in saved.columns
