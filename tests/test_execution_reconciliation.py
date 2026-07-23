from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from market_qml.execution.broker import BrokerError
from market_qml.execution.journal import ExecutionJournal
from market_qml.execution.paper_execution import (
    PaperExecutionPolicy,
    PreTradeError,
    deterministic_client_order_id,
    execute_paper_intent,
)
from market_qml.execution.reconciliation import (
    ReconciliationError,
    enforce_rebalance_cadence,
    reconcile_paper_execution,
    record_submission_report,
    register_execution_plan,
    save_reconciliation_report,
)

NOW = datetime(2026, 7, 23, 14, tzinfo=timezone.utc)


def _intent() -> dict[str, Any]:
    trades = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "notional": 20_000.0,
            "reference_price": 200.0,
            "estimated_quantity": 100.0,
        },
        {
            "symbol": "MSFT",
            "side": "buy",
            "notional": 20_000.0,
            "reference_price": 500.0,
            "estimated_quantity": 40.0,
        },
    ]
    return {
        "schema_version": 1,
        "intent_type": "paper_trade_dry_run",
        "broker_submission_allowed": False,
        "as_of_utc": NOW.isoformat(),
        "signal_date": "2026-07-23",
        "account_equity": 100_000.0,
        "model": {
            "artifact_id": "model-split-004",
            "feature_version": "canonical-v1",
            "model_name": "model",
            "model_sha256": "a" * 64,
            "preprocessor_sha256": "b" * 64,
            "promoted_at_utc": "2026-07-22T18:00:00+00:00",
            "selection_scope": "development_validation",
        },
        "policy": {},
        "portfolio": {
            "cash_reserve_weight": 0.6,
            "invested_weight": 0.4,
            "turnover": 0.2,
            "holdings": [
                {
                    "symbol": "AAPL",
                    "current_weight": 0.0,
                    "target_weight": 0.2,
                    "current_notional": 0.0,
                    "target_notional": 20_000.0,
                },
                {
                    "symbol": "MSFT",
                    "current_weight": 0.0,
                    "target_weight": 0.2,
                    "current_notional": 0.0,
                    "target_notional": 20_000.0,
                },
            ],
        },
        "trades": trades,
        "run_id": "c" * 64,
    }


def _client_id(symbol: str) -> str:
    intent = _intent()
    return deterministic_client_order_id(
        strategy_id="market-qml",
        run_id=intent["run_id"],
        signal_date=intent["signal_date"],
        symbol=symbol,
        side="buy",
    )


class FakeBroker:
    def __init__(self) -> None:
        self.account = {
            "status": "ACTIVE",
            "trading_blocked": False,
            "equity": "100000",
            "cash": "100000",
            "buying_power": "100000",
        }
        self.orders: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        self.submissions: list[dict[str, Any]] = []
        self.cancellations: list[str] = []

    def get_account(self):
        return self.account

    def list_positions(self):
        return self.positions

    def list_orders(self, *, status="open", after=None):
        if status == "open":
            return [
                order
                for order in self.orders
                if order.get("status")
                not in {"filled", "canceled", "expired", "replaced", "rejected"}
            ]
        return self.orders

    def get_asset(self, symbol):
        return {
            "symbol": symbol,
            "class": "us_equity",
            "status": "active",
            "tradable": True,
            "fractionable": True,
        }

    def get_calendar(self, start, end):
        return [{"date": "2026-07-23", "open": "09:30", "close": "16:00"}]

    def submit_order(self, order):
        self.submissions.append(dict(order))
        return {
            "id": f"broker-{order['symbol']}",
            "client_order_id": order["client_order_id"],
            "status": "accepted",
        }

    def cancel_order(self, order_id):
        self.cancellations.append(order_id)


def _register_orders(journal, intent):
    for symbol, qty, price in (("AAPL", 100.0, 200.5), ("MSFT", 40.0, 501.25)):
        register_execution_plan(
            journal,
            intent,
            {
                "client_order_id": _client_id(symbol),
                "symbol": symbol,
                "side": "buy",
                "qty": qty,
                "limit_price": price,
            },
            recorded_at=NOW,
        )


@pytest.mark.parametrize(
    "status",
    [
        "submitted",
        "new",
        "partially_filled",
        "filled",
        "canceled",
        "expired",
        "replaced",
        "rejected",
    ],
)
def test_journal_supports_every_required_order_state(tmp_path, status):
    with ExecutionJournal(tmp_path / f"{status}.sqlite") as journal:
        intent = _intent()
        _register_orders(journal, intent)
        update = {
            "client_order_id": _client_id("AAPL"),
            "id": "broker-aapl",
            "status": status,
            "filled_qty": 50 if status == "partially_filled" else 0,
            "filled_avg_price": 201 if status == "partially_filled" else None,
            "updated_at": NOW.isoformat(),
            "rejection_reason": "paper rejection" if status == "rejected" else None,
        }

        changed = journal.apply_order_update(update, observed_at=NOW, source="test")

        order = next(
            item
            for item in journal.orders_for_run(intent["run_id"])
            if item["symbol"] == "AAPL"
        )
        assert order["status"] == status
        assert order["broker_order_id"] == "broker-aapl"
        assert (
            changed is (status != "submitted") or order["broker_order_id"] is not None
        )


def test_partial_fill_progress_and_terminal_state_do_not_regress(tmp_path):
    with ExecutionJournal(tmp_path / "journal.sqlite") as journal:
        intent = _intent()
        _register_orders(journal, intent)
        for status, filled in (("new", 0), ("partial_fill", 25), ("fill", 100)):
            journal.apply_order_update(
                {
                    "client_order_id": _client_id("AAPL"),
                    "status": status,
                    "filled_qty": filled,
                    "filled_avg_price": 201,
                    "updated_at": NOW.isoformat(),
                },
                observed_at=NOW,
                source="trade_update",
            )

        regressed = journal.apply_order_update(
            {
                "client_order_id": _client_id("AAPL"),
                "status": "new",
                "filled_qty": 0,
            },
            observed_at=NOW,
            source="stale_rest",
        )

        order = journal.orders_for_run(intent["run_id"])[0]
        assert order["status"] == "filled"
        assert order["filled_qty"] == 100
        assert regressed is False
        assert [
            event["status"] for event in journal.events_for_order(_client_id("AAPL"))
        ] == [
            "new",
            "partially_filled",
            "filled",
        ]


def test_submission_is_journaled_before_post_and_restart_prevents_duplicate(tmp_path):
    path = tmp_path / "journal.sqlite"
    intent = _intent()
    broker = FakeBroker()
    policy = PaperExecutionPolicy(cash_reserve_weight=0.6)
    with ExecutionJournal(path) as journal:
        report = execute_paper_intent(
            broker,
            intent,
            now=NOW,
            policy=policy,
            submit=True,
            submission_enabled=True,
            kill_switch_active=False,
            known_client_order_ids=journal.known_client_order_ids(),
            planned_order_callback=lambda order: register_execution_plan(
                journal, intent, order, recorded_at=NOW
            ),
        )
        record_submission_report(journal, report, observed_at=NOW)
        assert len(journal.known_client_order_ids()) == 2

    restarted_broker = FakeBroker()
    with ExecutionJournal(path) as restarted:
        retry = execute_paper_intent(
            restarted_broker,
            intent,
            now=NOW,
            policy=policy,
            submit=True,
            submission_enabled=True,
            kill_switch_active=False,
            known_client_order_ids=restarted.known_client_order_ids(),
        )

    assert restarted_broker.submissions == []
    assert {item["reason"] for item in retry["decisions"]} == {
        "duplicate_client_order_id"
    }


def test_failure_journals_ambiguous_order_but_not_never_attempted_orders(tmp_path):
    class FailingBroker(FakeBroker):
        def submit_order(self, order):
            if order["symbol"] == "MSFT":
                raise BrokerError("ambiguous submission failure")
            return super().submit_order(order)

    intent = _intent()
    intent["trades"].append(
        {
            "symbol": "NVDA",
            "side": "buy",
            "notional": 10_000.0,
            "reference_price": 100.0,
            "estimated_quantity": 100.0,
        }
    )
    intent["portfolio"]["holdings"].append(
        {
            "symbol": "NVDA",
            "current_weight": 0.0,
            "target_weight": 0.1,
            "current_notional": 0.0,
            "target_notional": 10_000.0,
        }
    )
    intent["portfolio"]["invested_weight"] = 0.5
    path = tmp_path / "journal.sqlite"
    policy = PaperExecutionPolicy(cash_reserve_weight=0.4)
    with ExecutionJournal(path) as journal:
        report = execute_paper_intent(
            FailingBroker(),
            intent,
            now=NOW,
            policy=policy,
            submit=True,
            submission_enabled=True,
            kill_switch_active=False,
            planned_order_callback=lambda order: register_execution_plan(
                journal, intent, order, recorded_at=NOW
            ),
        )
        assert report["status"] == "submission_failed"
        assert len(journal.known_client_order_ids()) == 2

    retry_broker = FakeBroker()
    with ExecutionJournal(path) as restarted:
        retry = execute_paper_intent(
            retry_broker,
            intent,
            now=NOW,
            policy=policy,
            submit=True,
            submission_enabled=True,
            kill_switch_active=False,
            known_client_order_ids=restarted.known_client_order_ids(),
        )

    assert [order["symbol"] for order in retry_broker.submissions] == ["NVDA"]
    assert retry["status"] == "approved"


def test_journal_enforces_five_trading_day_rebalance_cadence(tmp_path):
    intent = _intent()
    with ExecutionJournal(tmp_path / "journal.sqlite") as journal:
        _register_orders(journal, intent)
        journal.apply_order_update(
            {
                "client_order_id": _client_id("AAPL"),
                "id": "broker-aapl",
                "status": "new",
                "filled_qty": 0,
                "updated_at": NOW.isoformat(),
            },
            observed_at=NOW,
            source="test",
        )
        next_intent = _intent()
        next_intent["run_id"] = "d" * 64
        next_intent["signal_date"] = "2026-07-24"
        next_intent["policy"] = {"rebalance_frequency_trading_days": 5}

        with pytest.raises(PreTradeError) as captured:
            enforce_rebalance_cadence(FakeBroker(), journal, next_intent)

        assert captured.value.code == "rebalance_too_soon"

        class FiveDayCalendarBroker(FakeBroker):
            def get_calendar(self, start, end):
                return [
                    {"date": day}
                    for day in (
                        "2026-07-24",
                        "2026-07-27",
                        "2026-07-28",
                        "2026-07-29",
                        "2026-07-30",
                    )
                ]

        next_intent["signal_date"] = "2026-07-30"
        enforce_rebalance_cadence(FiveDayCalendarBroker(), journal, next_intent)


def test_reconciliation_reports_partial_fills_slippage_rejections_and_positions(
    tmp_path,
):
    intent = _intent()
    broker = FakeBroker()
    broker.orders = [
        {
            "id": "broker-aapl",
            "client_order_id": _client_id("AAPL"),
            "symbol": "AAPL",
            "side": "buy",
            "status": "partially_filled",
            "qty": "100",
            "filled_qty": "50",
            "filled_avg_price": "202",
            "submitted_at": (NOW - timedelta(minutes=5)).isoformat(),
            "updated_at": NOW.isoformat(),
        },
        {
            "id": "broker-msft",
            "client_order_id": _client_id("MSFT"),
            "symbol": "MSFT",
            "side": "buy",
            "status": "rejected",
            "qty": "40",
            "filled_qty": "0",
            "rejection_reason": "insufficient buying power",
            "submitted_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        },
    ]
    broker.positions = [{"symbol": "AAPL", "qty": "50", "market_value": "10100"}]
    with ExecutionJournal(tmp_path / "journal.sqlite") as journal:
        _register_orders(journal, intent)

        report = reconcile_paper_execution(broker, journal, intent, now=NOW)

    aapl = next(item for item in report["orders"] if item["symbol"] == "AAPL")
    msft = next(item for item in report["orders"] if item["symbol"] == "MSFT")
    assert report["status"] == "attention_required"
    assert aapl["fill_percentage"] == 0.5
    assert aapl["adverse_slippage_bps"] == pytest.approx(100.0)
    assert msft["failure_reason"] == "insufficient buying power"
    assert report["summary"]["fill_percentage"] == pytest.approx(10_000 / 40_000)
    assert report["summary"]["absolute_residual_notional"] == 29_900
    assert "Residual target notional" in report["markdown"]


def test_stream_disconnect_falls_back_to_rest_and_rest_retry_recovers(tmp_path):
    intent = _intent()

    def disconnected_updates():
        yield {
            "event": "partial_fill",
            "order": {
                "client_order_id": _client_id("AAPL"),
                "filled_qty": "10",
                "filled_avg_price": "201",
                "updated_at": NOW.isoformat(),
            },
        }
        raise ConnectionError("stream disconnected")

    class FlakyBroker(FakeBroker):
        failures = 1

        def list_orders(self, *, status="open", after=None):
            if self.failures:
                self.failures -= 1
                raise BrokerError("temporary REST failure")
            return super().list_orders(status=status, after=after)

    broker = FlakyBroker()
    with ExecutionJournal(tmp_path / "journal.sqlite") as journal:
        _register_orders(journal, intent)
        report = reconcile_paper_execution(
            broker,
            journal,
            intent,
            now=NOW,
            trade_updates=disconnected_updates(),
        )

        assert "trade_update_stream_disconnected_rest_resync_used" in report["warnings"]
        assert "rest_attempt_1_failed" in report["warnings"]
        order = next(
            item
            for item in journal.orders_for_run(intent["run_id"])
            if item["symbol"] == "AAPL"
        )
        assert order["status"] == "partially_filled"


def test_total_rest_failure_preserves_journal_for_later_recovery(tmp_path):
    intent = _intent()

    class DownBroker(FakeBroker):
        def list_orders(self, *, status="open", after=None):
            raise BrokerError("down")

    path = tmp_path / "journal.sqlite"
    with ExecutionJournal(path) as journal:
        _register_orders(journal, intent)
        with pytest.raises(ReconciliationError, match="authoritative"):
            reconcile_paper_execution(
                DownBroker(), journal, intent, now=NOW, rest_attempts=2
            )

    with ExecutionJournal(path) as recovered:
        report = reconcile_paper_execution(FakeBroker(), recovered, intent, now=NOW)
        assert report["summary"]["order_count"] == 2


def test_stale_cancellation_is_recorded_and_reports_are_private_and_immutable(tmp_path):
    intent = _intent()
    broker = FakeBroker()
    broker.orders = [
        {
            "id": "broker-aapl",
            "client_order_id": _client_id("AAPL"),
            "symbol": "AAPL",
            "status": "new",
            "filled_qty": "0",
            "submitted_at": (NOW - timedelta(minutes=16)).isoformat(),
            "updated_at": NOW.isoformat(),
        }
    ]
    with ExecutionJournal(tmp_path / "private" / "journal.sqlite") as journal:
        _register_orders(journal, intent)
        report = reconcile_paper_execution(
            broker,
            journal,
            intent,
            now=NOW,
            cancel_stale=True,
            cancel_after_minutes=15,
            submission_enabled=True,
            kill_switch_active=False,
        )
        aapl = next(
            item
            for item in journal.orders_for_run(intent["run_id"])
            if item["symbol"] == "AAPL"
        )
        assert aapl["cancel_requested_at_utc"] == NOW.isoformat()

    json_path = tmp_path / "private" / "daily.json"
    markdown_path = tmp_path / "private" / "daily.md"
    save_reconciliation_report(report, json_path=json_path, markdown_path=markdown_path)

    serialized = json_path.read_text(encoding="utf-8")
    assert "account_id" not in serialized
    assert "secret" not in serialized
    assert broker.cancellations == ["broker-aapl"]
    with pytest.raises(FileExistsError):
        save_reconciliation_report(
            report, json_path=json_path, markdown_path=markdown_path
        )
