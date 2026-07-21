import pandas as pd

from market_qml.labels.forward_returns import build_multi_horizon_target_table
from market_qml.labels.target_research import research_target_candidates, target_catalog


def _prices(days: int = 18) -> pd.DataFrame:
    rows = []
    for symbol, sector, slope in [
        ("AAA", "tech", 1.5),
        ("BBB", "tech", 0.7),
        ("CCC", "health", -0.2),
        ("SPY", "benchmark", 0.4),
    ]:
        for day in range(days):
            rows.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "close": 100 + slope * day + (day % 3) * 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_multi_horizon_targets_include_rank_and_sector_variants():
    result = build_multi_horizon_target_table(
        _prices(), horizons=[2, 4], sector_column="sector"
    )

    assert set(result["label_horizon_days"]) == {2, 4}
    assert "cross_sectional_rank_2d" in result
    assert "sector_relative_return_4d" in result


def test_target_catalog_documents_horizon_specific_rules():
    labels = build_multi_horizon_target_table(_prices(), horizons=[2])
    catalog = target_catalog(labels)

    assert catalog["purge_days"].eq(2).all()
    assert catalog["benchmark"].eq("SPY").all()
    assert catalog["timing_rule"].str.contains(r"t\+2").all()
    assert catalog["missing_label_rule"].str.contains("future price").all()


def test_research_excludes_locked_test_and_selects_roles():
    labels = build_multi_horizon_target_table(
        _prices(30), horizons=[2, 4], sector_column="sector"
    )
    diagnostics, selection, manifest = research_target_candidates(
        labels, locked_test_days=5, embargo_days=2, period_frequency="W"
    )

    assert not diagnostics.empty
    assert set(selection["selected_role"]) == {"classification", "ranking"}
    assert diagnostics.groupby("target_name")["purge_days"].first().isin([2, 4]).all()
    assert manifest["selection_scope"] == "development_only"
    assert manifest["selection_validation"] == "nested_chronological_inner_outer"
    assert {"inner_validation", "outer_validation"} <= set(
        diagnostics["validation_role"]
    )
    assert manifest["locked_test_rows_inspected"] == 0
    assert manifest["locked_test_accessed"] is False
