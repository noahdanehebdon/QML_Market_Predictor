from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from market_qml.execution.broker import (
    ALPACA_PAPER_BASE_URL,
    AlpacaPaperBroker,
    BrokerError,
    validate_paper_base_url,
)
from market_qml.execution.paper_execution import (
    PaperExecutionPolicy,
    PreTradeError,
    cancel_stale_paper_orders,
    deterministic_client_order_id,
    execute_paper_intent,
)

NOW = datetime(2026, 7, 23, 14, tzinfo=timezone.utc)


def _execution_policy() -> PaperExecutionPolicy:
    return PaperExecutionPolicy(max_position_weight=0.30)


def _intent() -> dict[str, Any]:
    holdings = [
        {
            "symbol": symbol,
            "current_weight": 0.0,
            "target_weight": 0.3,
            "current_notional": 0.0,
            "target_notional": 30_000.0,
        }
        for symbol in ("AAPL", "MSFT", "NVDA")
    ]
    trades = [
        {
            "symbol": symbol,
            "side": "buy",
            "notional": 30_000.0,
            "reference_price": price,
            "estimated_quantity": 30_000.0 / price,
            "score": score,
            "reason": "selected_by_promoted_model",
        }
        for symbol, price, score in (
            ("AAPL", 200.0, 0.8),
            ("MSFT", 500.0, 0.7),
            ("NVDA", 180.0, 0.6),
        )
    ]
    return {
        "schema_version": 1,
        "intent_type": "paper_trade_dry_run",
        "broker_submission_allowed": False,
        "as_of_utc": NOW.isoformat(),
        "signal_date": "2026-07-23",
        "account_equity": 100_000.0,
        "model": {
            "artifact_id": "gradient-boosting-split-004",
            "feature_version": "canonical-v1",
            "model_name": "gradient_boosting",
            "model_sha256": "a" * 64,
            "preprocessor_sha256": "b" * 64,
            "promoted_at_utc": "2026-07-22T18:00:00+00:00",
            "selection_scope": "development_validation",
        },
        "policy": {},
        "portfolio": {
            "cash_reserve_weight": 0.1,
            "invested_weight": 0.9,
            "turnover": 0.45,
            "holdings": holdings,
        },
        "trades": trades,
        "run_id": "c" * 64,
    }


class FakeBroker:
    def __init__(self) -> None:
        self.account = {
            "status": "ACTIVE",
            "trading_blocked": False,
            "equity": "100000",
            "last_equity": "100000",
            "cash": "100000",
            "buying_power": "100000",
        }
        self.positions: list[dict[str, Any]] = []
        self.open_orders: list[dict[str, Any]] = []
        self.recent_orders: list[dict[str, Any]] = []
        self.assets = {
            symbol: {
                "symbol": symbol,
                "class": "us_equity",
                "status": "active",
                "tradable": True,
                "fractionable": True,
            }
            for symbol in ("AAPL", "MSFT", "NVDA")
        }
        self.submissions: list[dict[str, Any]] = []
        self.cancellations: list[str] = []

    def get_account(self):
        return self.account

    def list_positions(self):
        return self.positions

    def list_orders(self, *, status="open", after=None):
        return self.open_orders if status == "open" else self.recent_orders

    def get_asset(self, symbol):
        return self.assets[symbol]

    def get_calendar(self, start, end):
        return [{"date": "2026-07-23", "open": "09:30", "close": "16:00"}]

    def submit_order(self, order):
        self.submissions.append(dict(order))
        return {"status": "accepted", "id": "private-broker-order-id"}

    def cancel_order(self, order_id):
        self.cancellations.append(order_id)


def test_dry_run_fetches_state_but_never_submits():
    broker = FakeBroker()

    report = execute_paper_intent(
        broker, _intent(), now=NOW, policy=_execution_policy()
    )

    assert report["mode"] == "dry_run"
    assert report["status"] == "approved"
    assert report["paper_only"] is True
    assert len(report["orders"]) == 3
    assert broker.submissions == []
    assert all(order["type"] == "limit" for order in report["orders"])
    assert all(order["time_in_force"] == "day" for order in report["orders"])
    assert all(order["extended_hours"] is False for order in report["orders"])


def test_daily_loss_limit_blocks_execution():
    broker = FakeBroker()
    broker.account["equity"] = "99000"
    broker.account["last_equity"] = "102000"

    with pytest.raises(PreTradeError) as error:
        execute_paper_intent(broker, _intent(), now=NOW, policy=_execution_policy())

    assert error.value.code == "daily_loss_limit_breached"
    assert broker.submissions == []


def test_submission_requires_all_gates_and_submits_sells_first():
    broker = FakeBroker()
    intent = _intent()
    intent["trades"].insert(
        0,
        {
            "symbol": "OLD",
            "side": "sell",
            "notional": 1_000.0,
            "reference_price": 100.0,
            "estimated_quantity": 10.0,
        },
    )
    intent["portfolio"]["holdings"].append({"symbol": "OLD", "target_weight": 0.0})
    broker.assets["OLD"] = {
        "class": "us_equity",
        "status": "active",
        "tradable": True,
        "fractionable": False,
    }
    broker.positions = [{"symbol": "OLD", "qty": "10", "side": "long"}]

    with pytest.raises(PreTradeError, match="kill switch"):
        execute_paper_intent(
            broker,
            intent,
            now=NOW,
            policy=_execution_policy(),
            submit=True,
            submission_enabled=True,
            kill_switch_active=True,
        )
    with pytest.raises(PreTradeError, match="not enabled"):
        execute_paper_intent(
            broker,
            intent,
            now=NOW,
            policy=_execution_policy(),
            submit=True,
            submission_enabled=False,
            kill_switch_active=False,
        )

    report = execute_paper_intent(
        broker,
        intent,
        now=NOW,
        policy=_execution_policy(),
        submit=True,
        submission_enabled=True,
        kill_switch_active=False,
    )

    assert report["mode"] == "paper_submit"
    assert [order["symbol"] for order in broker.submissions] == [
        "OLD",
        "AAPL",
        "MSFT",
        "NVDA",
    ]
    assert all(
        "private-broker-order-id" not in str(item) for item in report["submitted"]
    )


def test_retry_skips_existing_client_id_without_submission():
    broker = FakeBroker()
    intent = _intent()
    duplicate_id = deterministic_client_order_id(
        strategy_id="market-qml",
        run_id=intent["run_id"],
        signal_date=intent["signal_date"],
        symbol="AAPL",
        side="buy",
    )
    broker.recent_orders = [
        {"id": "broker-id", "client_order_id": duplicate_id, "status": "filled"}
    ]

    report = execute_paper_intent(
        broker,
        intent,
        now=NOW,
        policy=_execution_policy(),
        submit=True,
        submission_enabled=True,
        kill_switch_active=False,
    )

    assert {order["symbol"] for order in broker.submissions} == {"MSFT", "NVDA"}
    duplicate = next(item for item in report["decisions"] if item["symbol"] == "AAPL")
    assert duplicate == {
        "symbol": "AAPL",
        "side": "buy",
        "status": "skipped",
        "reason": "duplicate_client_order_id",
    }


def test_broker_failure_stops_remaining_orders_and_reports_partial_state():
    class FailingBroker(FakeBroker):
        def submit_order(self, order):
            if order["symbol"] == "MSFT":
                raise BrokerError("sanitized failure")
            return super().submit_order(order)

    broker = FailingBroker()

    report = execute_paper_intent(
        broker,
        _intent(),
        now=NOW,
        policy=_execution_policy(),
        submit=True,
        submission_enabled=True,
        kill_switch_active=False,
    )

    assert report["status"] == "submission_failed"
    assert report["error"]["code"] == "broker_submission_failed"
    assert [item["symbol"] for item in report["submitted"]] == ["AAPL"]
    assert [item["symbol"] for item in broker.submissions] == ["AAPL"]


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            lambda broker: broker.assets["AAPL"].update(tradable=False),
            "asset_not_tradable",
        ),
        (
            lambda broker: broker.open_orders.append(
                {"id": "1", "symbol": "AAPL", "status": "new"}
            ),
            "conflicting_open_order",
        ),
        (
            lambda broker: broker.assets["NVDA"].update(fractionable=False),
            "asset_not_fractionable",
        ),
    ],
)
def test_trade_rejections_are_machine_readable_and_atomic(change, reason):
    broker = FakeBroker()
    change(broker)

    report = execute_paper_intent(
        broker,
        _intent(),
        now=NOW,
        policy=_execution_policy(),
        submit=True,
        submission_enabled=True,
        kill_switch_active=False,
    )

    assert report["status"] == "blocked"
    assert reason in {item["reason"] for item in report["decisions"]}
    assert broker.submissions == []


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.alpaca.markets",
        "http://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.example",
        "https://paper-api.alpaca.markets/v2",
        "https://paper-api.alpaca.markets?next=https://api.alpaca.markets",
    ],
)
def test_live_or_modified_hosts_are_rejected(base_url):
    with pytest.raises(ValueError, match="allows only"):
        validate_paper_base_url(base_url)
    with pytest.raises(ValueError, match="allows only"):
        AlpacaPaperBroker(
            api_key="paper-key", secret_key="paper-secret", base_url=base_url
        )


def test_canonical_paper_host_and_credentials_are_accepted(monkeypatch):
    assert validate_paper_base_url(f"{ALPACA_PAPER_BASE_URL}/") == ALPACA_PAPER_BASE_URL
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "paper-secret")

    broker = AlpacaPaperBroker.from_environment()

    assert broker.base_url == ALPACA_PAPER_BASE_URL


def test_stale_project_orders_are_cancelled_but_unrelated_orders_are_untouched():
    broker = FakeBroker()
    stale = (NOW - timedelta(minutes=16)).isoformat()
    fresh = (NOW - timedelta(minutes=5)).isoformat()
    broker.open_orders = [
        {"id": "stale-id", "client_order_id": "mqml-old", "submitted_at": stale},
        {"id": "fresh-id", "client_order_id": "mqml-fresh", "submitted_at": fresh},
        {"id": "manual-id", "client_order_id": "manual-order", "submitted_at": stale},
    ]

    result = cancel_stale_paper_orders(
        broker,
        now=NOW,
        cancel_after_minutes=15,
        submission_enabled=True,
        kill_switch_active=False,
    )

    assert broker.cancellations == ["stale-id"]
    assert result == [{"client_order_id": "mqml-old", "status": "cancel_requested"}]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = {"X-Request-ID": "safe-request-id"}

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_adapter_uses_paper_routes_and_sanitizes_errors():
    session = RecordingSession(
        [
            FakeResponse(payload={"status": "ACTIVE"}),
            FakeResponse(
                status_code=403, payload={"secret": "private"}, text="private"
            ),
        ]
    )
    broker = AlpacaPaperBroker(
        api_key="paper-key",
        secret_key="paper-secret",
        session=session,
    )

    assert broker.get_account() == {"status": "ACTIVE"}
    with pytest.raises(BrokerError) as captured:
        broker.submit_order({"symbol": "AAPL"})

    assert session.calls[0][1] == f"{ALPACA_PAPER_BASE_URL}/v2/account"
    assert session.calls[1][1] == f"{ALPACA_PAPER_BASE_URL}/v2/orders"
    assert "paper-secret" not in str(captured.value)
    assert "private" not in str(captured.value)
    assert "safe-request-id" in str(captured.value)


def test_adapter_covers_required_paper_endpoints():
    session = RecordingSession(
        [
            FakeResponse(payload=[]),
            FakeResponse(payload=[]),
            FakeResponse(payload={"symbol": "AAPL"}),
            FakeResponse(payload=[]),
            FakeResponse(payload={"status": "accepted"}),
            FakeResponse(status_code=204),
        ]
    )
    broker = AlpacaPaperBroker(
        api_key="paper-key", secret_key="paper-secret", session=session
    )

    assert broker.list_positions() == []
    assert broker.list_orders(status="all", after=NOW.isoformat()) == []
    assert broker.get_asset("aapl") == {"symbol": "AAPL"}
    assert broker.get_calendar("2026-07-23", "2026-07-23") == []
    assert broker.submit_order({"symbol": "AAPL"}) == {"status": "accepted"}
    broker.cancel_order("paper-order-id")

    assert [call[0] for call in session.calls] == [
        "GET",
        "GET",
        "GET",
        "GET",
        "POST",
        "DELETE",
    ]
    assert session.calls[1][2]["params"]["status"] == "all"
    assert session.calls[2][1].endswith("/v2/assets/AAPL")
    assert session.calls[-1][1].endswith("/v2/orders/paper-order-id")


def test_adapter_rejects_missing_credentials_invalid_status_and_unsafe_paths(
    monkeypatch,
):
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing paper credentials"):
        AlpacaPaperBroker.from_environment()
    with pytest.raises(ValueError, match="non-empty"):
        AlpacaPaperBroker(api_key="", secret_key="secret")

    broker = AlpacaPaperBroker(api_key="key", secret_key="secret")
    with pytest.raises(ValueError, match="status"):
        broker.list_orders(status="invalid")
    with pytest.raises(ValueError, match="unsafe"):
        broker.get_asset("../AAPL")
    with pytest.raises(ValueError, match="unsafe"):
        broker.cancel_order("bad/id")


@pytest.mark.parametrize(
    "change,code",
    [
        (
            lambda broker, intent: broker.account.update(trading_blocked=True),
            "account_trading_blocked",
        ),
        (
            lambda broker, intent: broker.account.update(status="INACTIVE"),
            "account_inactive",
        ),
        (
            lambda broker, intent: broker.account.update(equity="90000"),
            "account_equity_drift",
        ),
        (
            lambda broker, intent: broker.account.update(buying_power="1"),
            "insufficient_buying_power",
        ),
        (
            lambda broker, intent: broker.account.update(cash="1"),
            "cash_reserve_breached",
        ),
        (
            lambda broker, intent: intent.update(
                as_of_utc=(NOW - timedelta(hours=1)).isoformat()
            ),
            "stale_intent",
        ),
        (
            lambda broker, intent: intent["portfolio"].update(invested_weight=0.95),
            "gross_exposure_exceeded",
        ),
        (
            lambda broker, intent: broker.positions.append(
                {"symbol": "OLD", "qty": "-1", "side": "short"}
            ),
            "short_position",
        ),
    ],
)
def test_global_risk_failures_stop_before_submission(change, code):
    broker = FakeBroker()
    intent = _intent()
    change(broker, intent)

    with pytest.raises(PreTradeError) as captured:
        execute_paper_intent(
            broker,
            intent,
            now=NOW,
            policy=_execution_policy(),
            submit=True,
            submission_enabled=True,
            kill_switch_active=False,
        )

    assert captured.value.code == code
    assert broker.submissions == []


def test_market_calendar_and_order_count_fail_closed():
    class ClosedBroker(FakeBroker):
        def get_calendar(self, start, end):
            return []

    with pytest.raises(PreTradeError) as captured:
        execute_paper_intent(
            ClosedBroker(), _intent(), now=NOW, policy=_execution_policy()
        )
    assert captured.value.code == "market_closed"

    with pytest.raises(PreTradeError) as captured:
        execute_paper_intent(
            FakeBroker(),
            _intent(),
            now=NOW,
            policy=PaperExecutionPolicy(max_position_weight=0.30, max_order_count=2),
        )
    assert captured.value.code == "max_order_count_exceeded"
