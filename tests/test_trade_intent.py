import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from market_qml.execution.trade_intent import (
    PortfolioPolicy,
    build_trade_intent,
    load_promotion_manifest,
    save_trade_intent,
)
from scripts.generate_trade_intent import load_policy


def _promotion(**overrides):
    promotion = {
        "schema_version": 1,
        "artifact_id": "gradient-boosting-split-004",
        "model_name": "gradient_boosting",
        "model_sha256": "a" * 64,
        "preprocessor_sha256": "b" * 64,
        "feature_version": "canonical-v1",
        "promoted_at_utc": "2026-07-22T18:00:00+00:00",
        "selection_scope": "development_validation",
    }
    promotion.update(overrides)
    return promotion


def _signals():
    return pd.DataFrame(
        {
            "date": ["2026-07-23"] * 5,
            "symbol": ["SPY", "MSFT", "AAPL", "NVDA", "GOOG"],
            "predicted_outperformance_probability": [0.50, 0.70, 0.80, 0.60, 0.65],
            "reference_price": [600.0, 500.0, 200.0, 180.0, 190.0],
            "is_benchmark": [True, False, False, False, False],
            "model_name": ["gradient_boosting"] * 5,
            "artifact_id": ["gradient-boosting-split-004"] * 5,
            "feature_version": ["canonical-v1"] * 5,
        }
    )


def _policy(**overrides):
    values = {
        "selected_count": 3,
        "max_position_weight": 0.30,
        "cash_reserve_weight": 0.10,
        "max_turnover": 1.0,
        "minimum_trade_notional": 10.0,
        "rebalance_frequency_trading_days": 5,
        "max_signal_age_days": 3,
    }
    values.update(overrides)
    return PortfolioPolicy(**values)


AS_OF = datetime(2026, 7, 23, 14, tzinfo=timezone.utc)


def test_build_trade_intent_is_deterministic_and_auditable():
    positions = pd.DataFrame({"symbol": ["AAPL"], "market_value": [10_000.0]})

    first = build_trade_intent(
        _signals(),
        positions,
        promotion=_promotion(),
        account_equity=100_000,
        as_of=AS_OF,
        policy=_policy(),
    )
    second = build_trade_intent(
        _signals().sample(frac=1, random_state=4),
        positions,
        promotion=_promotion(),
        account_equity=100_000,
        as_of=AS_OF,
        policy=_policy(),
    )

    assert first == second
    assert first["run_id"] == second["run_id"]
    assert first["broker_submission_allowed"] is False
    assert first["model"]["artifact_id"] == "gradient-boosting-split-004"
    assert first["portfolio"]["cash_reserve_weight"] == 0.1
    assert first["portfolio"]["turnover"] == 0.4
    assert [trade["symbol"] for trade in first["trades"]] == ["AAPL", "GOOG", "MSFT"]


@pytest.mark.parametrize(
    "mutator,error",
    [
        (lambda data: data.iloc[0:0], "empty"),
        (lambda data: data.drop(columns="reference_price"), "required columns"),
        (lambda data: data.assign(reference_price=np.nan), "nonfinite"),
        (lambda data: data.assign(date="2026-07-01"), "stale"),
        (lambda data: pd.concat([data, data.iloc[[0]]]), "unique"),
    ],
)
def test_trade_intent_rejects_bad_signals(mutator, error):
    with pytest.raises(ValueError, match=error):
        build_trade_intent(
            mutator(_signals()),
            pd.DataFrame(columns=["symbol", "market_value"]),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )


def test_trade_intent_rejects_locked_test_promotion():
    with pytest.raises(ValueError, match="locked test"):
        build_trade_intent(
            _signals(),
            pd.DataFrame(),
            promotion=_promotion(selection_scope="locked_test"),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )


def test_trade_intent_rejects_future_promotion_and_mismatched_lineage():
    with pytest.raises(ValueError, match="after the decision"):
        build_trade_intent(
            _signals(),
            pd.DataFrame(),
            promotion=_promotion(promoted_at_utc="2026-07-24T00:00:00+00:00"),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )

    with pytest.raises(ValueError, match="does not match"):
        build_trade_intent(
            _signals().assign(model_name="other_model"),
            pd.DataFrame(),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )


def test_trade_intent_rejects_short_missing_price_and_excess_turnover():
    with pytest.raises(ValueError, match="Short positions"):
        build_trade_intent(
            _signals(),
            pd.DataFrame({"symbol": ["AAPL"], "market_value": [-1.0]}),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )

    with pytest.raises(ValueError, match="missing reference prices"):
        build_trade_intent(
            _signals(),
            pd.DataFrame({"symbol": ["TSLA"], "market_value": [1_000.0]}),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )

    with pytest.raises(ValueError, match="exceeds max_turnover"):
        build_trade_intent(
            _signals(),
            pd.DataFrame(columns=["symbol", "market_value"]),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(max_turnover=0.1),
        )


def test_minimum_notional_suppresses_trade_but_preserves_target():
    intent = build_trade_intent(
        _signals(),
        pd.DataFrame({"symbol": ["AAPL"], "market_value": [29_995.0]}),
        promotion=_promotion(),
        account_equity=100_000,
        as_of=AS_OF,
        policy=_policy(minimum_trade_notional=10.0),
    )

    assert "AAPL" not in {trade["symbol"] for trade in intent["trades"]}
    aapl = next(
        item for item in intent["portfolio"]["holdings"] if item["symbol"] == "AAPL"
    )
    assert aapl["target_notional"] == 30_000.0


def test_save_is_immutable_and_json_is_finite(tmp_path):
    intent = build_trade_intent(
        _signals(),
        pd.DataFrame(),
        promotion=_promotion(),
        account_equity=100_000,
        as_of=AS_OF,
        policy=_policy(),
    )
    output = tmp_path / "intent.json"

    save_trade_intent(intent, output)

    assert json.loads(output.read_text(encoding="utf-8")) == intent
    with pytest.raises(FileExistsError):
        save_trade_intent(intent, output)


def test_load_promotion_and_policy(tmp_path):
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(json.dumps(_promotion()), encoding="utf-8")
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        "portfolio_policy:\n  selected_count: 3\n  max_position_weight: 0.3\n",
        encoding="utf-8",
    )

    assert load_promotion_manifest(promotion_path)["artifact_id"].startswith("gradient")
    assert load_policy(config_path).selected_count == 3


@pytest.mark.parametrize(
    "promotion,error",
    [
        ({"schema_version": 1}, "missing required fields"),
        (_promotion(schema_version=2), "schema_version"),
        (_promotion(model_name=""), "non-empty"),
        (_promotion(model_sha256="bad"), "SHA-256"),
    ],
)
def test_invalid_promotion_manifests_fail_closed(promotion, error):
    with pytest.raises(ValueError, match=error):
        build_trade_intent(
            _signals(),
            pd.DataFrame(),
            promotion=promotion,
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )


@pytest.mark.parametrize(
    "policy,error",
    [
        (_policy(selected_count=0), "selected_count"),
        (_policy(cash_reserve_weight=1.0), "cash_reserve_weight"),
        (_policy(minimum_trade_notional=-1), "minimum_trade_notional"),
        (_policy(rebalance_frequency_trading_days=0), "rebalance_frequency"),
        (_policy(max_signal_age_days=-1), "max_signal_age_days"),
        (_policy(selected_count=2), "max_position_weight"),
    ],
)
def test_invalid_portfolio_policies_fail_closed(policy, error):
    with pytest.raises(ValueError, match=error):
        build_trade_intent(
            _signals(),
            pd.DataFrame(),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=policy,
        )


@pytest.mark.parametrize(
    "positions,error",
    [
        (pd.DataFrame({"symbol": ["AAPL"]}), "required columns"),
        (
            pd.DataFrame({"symbol": ["AAPL", "AAPL"], "market_value": [1.0, 1.0]}),
            "unique",
        ),
        (
            pd.DataFrame({"symbol": ["AAPL"], "market_value": [np.inf]}),
            "nonfinite",
        ),
        (
            pd.DataFrame({"symbol": ["AAPL"], "market_value": [100_001.0]}),
            "exceeds account equity",
        ),
    ],
)
def test_invalid_positions_fail_closed(positions, error):
    with pytest.raises(ValueError, match=error):
        build_trade_intent(
            _signals(),
            positions,
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )


def test_invalid_account_timestamp_and_signal_values_fail_closed():
    with pytest.raises(ValueError, match="account_equity must be positive"):
        build_trade_intent(
            _signals(),
            pd.DataFrame(),
            promotion=_promotion(),
            account_equity=0,
            as_of=AS_OF,
            policy=_policy(),
        )
    with pytest.raises(ValueError, match="include a timezone"):
        build_trade_intent(
            _signals(),
            pd.DataFrame(),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=datetime(2026, 7, 23),
            policy=_policy(),
        )
    with pytest.raises(ValueError, match="probabilities"):
        build_trade_intent(
            _signals().assign(predicted_outperformance_probability=2.0),
            pd.DataFrame(),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )
    with pytest.raises(ValueError, match="prices must be positive"):
        build_trade_intent(
            _signals().assign(reference_price=0.0),
            pd.DataFrame(),
            promotion=_promotion(),
            account_equity=100_000,
            as_of=AS_OF,
            policy=_policy(),
        )
