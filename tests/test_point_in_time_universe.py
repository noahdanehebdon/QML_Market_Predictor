import pandas as pd

from market_qml.universe import (
    UniverseRules,
    build_point_in_time_universe,
    universe_diagnostics,
)


def _prices() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=7)
    for symbol, close, volume in [
        ("AAA", 10.0, 200.0),
        ("BBB", 20.0, 100.0),
        ("CCC", 2.0, 1_000.0),
        ("SPY", 100.0, 10_000.0),
    ]:
        for date in dates:
            rows.append(
                {"symbol": symbol, "date": date, "close": close, "volume": volume}
            )
    return pd.DataFrame(rows)


def _assets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": symbol, "effective_date": "2024-01-01", "asset_class": "us_equity", "status": "active", "tradable": True}
            for symbol in ["AAA", "BBB", "CCC", "SPY"]
        ]
        + [
            {"symbol": "BBB", "effective_date": "2024-01-08", "asset_class": "us_equity", "status": "inactive", "tradable": False}
        ]
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAA", "effective_date": "2024-01-01", "sector": "tech", "industry": "software", "market_cap": 100.0},
            {"symbol": "BBB", "effective_date": "2024-01-01", "sector": "finance", "industry": "banks", "market_cap": 200.0},
            {"symbol": "CCC", "effective_date": "2024-01-01", "sector": "health", "industry": "biotech", "market_cap": 50.0},
        ]
    )


RULES = UniverseRules(
    min_price=5,
    min_median_dollar_volume=1_000,
    liquidity_window=2,
    min_history_days=2,
    min_names=2,
    min_sectors=2,
    min_sector_names=1,
)


def test_membership_uses_only_effective_dated_and_trailing_information():
    result = build_point_in_time_universe(
        _prices(), _assets(), metadata_history=_metadata(), rules=RULES
    )

    first = result[result["date"].eq(pd.Timestamp("2024-01-01"))]
    assert not first["is_member"].any()
    third = result[result["date"].eq(pd.Timestamp("2024-01-03"))]
    assert set(third.loc[third["is_member"], "symbol"]) == {"AAA", "BBB"}
    assert not third.loc[third["symbol"].eq("CCC"), "eligible_price"].item()
    assert not third.loc[third["symbol"].eq("SPY"), "is_member"].item()
    after_inactive = result[
        result["date"].eq(pd.Timestamp("2024-01-08")) & result["symbol"].eq("BBB")
    ].iloc[0]
    assert not after_inactive["is_member"]
    assert after_inactive["status"] == "inactive"


def test_future_asset_changes_do_not_modify_past_membership():
    baseline = build_point_in_time_universe(_prices(), _assets(), rules=RULES)
    changed = _assets().copy()
    changed.loc[
        changed["effective_date"].eq("2024-01-08") & changed["symbol"].eq("BBB"),
        "effective_date",
    ] = "2024-01-09"
    revised = build_point_in_time_universe(_prices(), changed, rules=RULES)

    cutoff = pd.Timestamp("2024-01-05")
    columns = ["symbol", "date", "is_member", "eligible_tradability"]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["date"].le(cutoff), columns].reset_index(drop=True),
        revised.loc[revised["date"].le(cutoff), columns].reset_index(drop=True),
    )


def test_diagnostics_report_coverage_entries_exits_and_controls():
    membership = build_point_in_time_universe(
        _prices(), _assets(), metadata_history=_metadata(), rules=RULES
    )
    coverage, transitions, summary = universe_diagnostics(membership, rules=RULES)

    assert summary["entries"] == 2
    assert summary["exits"] == 1
    assert set(transitions.loc[transitions["entered"], "symbol"]) == {"AAA", "BBB"}
    assert transitions.loc[transitions["exited"], "symbol"].tolist() == ["BBB"]
    stable = coverage.loc[coverage["member_count"].eq(2)].iloc[0]
    assert not stable["stable_deciles"]
    assert stable["stable_sector_controls"]
    assert stable["sector_count"] == 2


def test_missing_asset_snapshot_never_backfills_tradability():
    assets = _assets()
    assets.loc[assets["symbol"].eq("AAA"), "effective_date"] = "2024-01-04"
    result = build_point_in_time_universe(_prices(), assets, rules=RULES)

    before_snapshot = result[
        result["symbol"].eq("AAA") & result["date"].lt(pd.Timestamp("2024-01-04"))
    ]
    assert not before_snapshot["eligible_tradability"].any()
    assert before_snapshot["effective_date"].isna().all()
