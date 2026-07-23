"""REST-authoritative paper order and position reconciliation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from market_qml.execution.broker import BrokerError, PaperBroker
from market_qml.execution.journal import ExecutionJournal
from market_qml.execution.paper_execution import (
    PreTradeError,
    cancel_stale_paper_orders,
    deterministic_client_order_id,
)


class ReconciliationError(RuntimeError):
    """Reconciliation could not establish authoritative broker state."""


def enforce_rebalance_cadence(
    broker: PaperBroker,
    journal: ExecutionJournal,
    intent: dict[str, Any],
) -> None:
    """Reject a new submission before its configured trading-day interval."""
    previous = journal.latest_submitted_signal_date()
    if previous is None:
        return
    current = str(intent["signal_date"])
    frequency = int(intent["policy"].get("rebalance_frequency_trading_days", 5))
    if frequency <= 0:
        raise PreTradeError(
            "invalid_rebalance_frequency", "Rebalance frequency is invalid."
        )
    if current <= previous:
        raise PreTradeError(
            "rebalance_too_soon",
            "A paper rebalance has already been submitted for this or a later date.",
        )
    calendar = broker.get_calendar(previous, current)
    elapsed = sum(previous < str(day.get("date", "")) <= current for day in calendar)
    if elapsed < frequency:
        raise PreTradeError(
            "rebalance_too_soon",
            f"Only {elapsed} trading days have elapsed; {frequency} are required.",
        )


def register_execution_plan(
    journal: ExecutionJournal,
    intent: dict[str, Any],
    order: dict[str, Any],
    *,
    recorded_at: datetime,
) -> None:
    """Durably record a planned order before its broker POST occurs."""
    journal.register_intent(intent, recorded_at)
    journal.record_planned_order(
        run_id=intent["run_id"],
        client_order_id=str(order["client_order_id"]),
        symbol=str(order["symbol"]),
        side=str(order["side"]),
        requested_qty=_finite(order["qty"], "order.qty"),
        requested_limit_price=_finite(order["limit_price"], "order.limit_price"),
        recorded_at=recorded_at,
    )


def record_submission_report(
    journal: ExecutionJournal,
    report: dict[str, Any],
    *,
    observed_at: datetime,
) -> None:
    """Record sanitized immediate submission acknowledgements."""
    for submitted in report.get("submitted", []):
        journal.apply_order_update(
            {
                "client_order_id": submitted["client_order_id"],
                "id": submitted.get("broker_order_id"),
                "status": submitted.get("status", "submitted"),
                "filled_qty": 0,
                "updated_at": _timestamp(observed_at),
            },
            observed_at=observed_at,
            source="submission_response",
        )


def reconcile_paper_execution(
    broker: PaperBroker,
    journal: ExecutionJournal,
    intent: dict[str, Any],
    *,
    now: datetime,
    strategy_id: str = "market-qml",
    trade_updates: Iterable[dict[str, Any]] | None = None,
    rest_attempts: int = 3,
    cancel_stale: bool = False,
    cancel_after_minutes: int = 15,
    submission_enabled: bool = False,
    kill_switch_active: bool = True,
) -> dict[str, Any]:
    """Reconcile journal state with Alpaca orders and positions."""
    checked_now = _aware_utc(now)
    if rest_attempts <= 0:
        raise ValueError("rest_attempts must be positive.")
    journal.register_intent(intent, checked_now)
    warnings = _consume_trade_updates(journal, trade_updates, checked_now)

    cancellations = []
    if cancel_stale:
        cancellations = cancel_stale_paper_orders(
            broker,
            now=checked_now,
            cancel_after_minutes=cancel_after_minutes,
            submission_enabled=submission_enabled,
            kill_switch_active=kill_switch_active,
        )
        for item in cancellations:
            journal.record_cancel_request(item["client_order_id"], checked_now)

    open_orders, recent_orders, positions, retry_warnings = _fetch_broker_state(
        broker, intent["as_of_utc"], rest_attempts
    )
    warnings.extend(retry_warnings)
    broker_orders = _unique_orders([*open_orders, *recent_orders])
    known_ids = journal.known_client_order_ids()
    for order in broker_orders:
        client_order_id = str(order.get("client_order_id", ""))
        if client_order_id not in known_ids:
            _backfill_observed_order(journal, intent, order, strategy_id, checked_now)
            known_ids = journal.known_client_order_ids()
        if client_order_id not in known_ids:
            continue
        journal.apply_order_update(
            order,
            observed_at=checked_now,
            source="rest_reconciliation",
        )
    journal.replace_position_snapshot(intent["run_id"], positions, checked_now)

    order_rows = _order_report(journal.orders_for_run(intent["run_id"]), intent)
    position_rows = _position_report(positions, intent)
    report = {
        "schema_version": 1,
        "paper_only": True,
        "run_id": intent["run_id"],
        "signal_date": intent["signal_date"],
        "reconciled_at_utc": checked_now.isoformat(),
        "status": _reconciliation_status(order_rows, position_rows),
        "summary": _summary(order_rows, position_rows),
        "orders": order_rows,
        "positions": position_rows,
        "cancellations": cancellations,
        "warnings": warnings,
    }
    report["markdown"] = render_reconciliation_markdown(report)
    return report


def save_reconciliation_report(
    report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path
) -> None:
    """Write private daily reconciliation outputs without overwriting them."""
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = {key: value for key, value in report.items() if key != "markdown"}
    with json_destination.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(serialized, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    with markdown_destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(str(report["markdown"]))


def render_reconciliation_markdown(report: dict[str, Any]) -> str:
    """Render a concise private daily execution report."""
    summary = report["summary"]
    lines = [
        "# Paper Execution Reconciliation",
        "",
        f"**Signal date:** {report['signal_date']}",
        f"**Status:** `{report['status']}`",
        f"**Fill percentage:** {summary['fill_percentage']:.2%}",
        f"**Average adverse slippage:** {summary['average_adverse_slippage_bps']:.2f} bps",
        f"**Residual target notional:** ${summary['absolute_residual_notional']:.2f}",
        "",
        "## Orders",
        "",
        "| Symbol | Side | Status | Requested | Filled | Fill % | Slippage (bps) | Failure |",
        "|:---|:---|:---|---:|---:|---:|---:|:---|",
    ]
    for order in report["orders"]:
        slippage = order["adverse_slippage_bps"]
        slippage_text = "—" if slippage is None else f"{slippage:.2f}"
        lines.append(
            f"| {order['symbol']} | {order['side']} | {order['status']} | "
            f"{order['requested_qty']:.6f} | {order['filled_qty']:.6f} | "
            f"{order['fill_percentage']:.2%} | {slippage_text} | "
            f"{order['failure_reason'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Position Differences",
            "",
            "| Symbol | Target notional | Actual notional | Residual |",
            "|:---|---:|---:|---:|",
        ]
    )
    for position in report["positions"]:
        lines.append(
            f"| {position['symbol']} | ${position['target_notional']:.2f} | "
            f"${position['actual_notional']:.2f} | "
            f"${position['residual_notional']:.2f} |"
        )
    if report["warnings"]:
        lines.extend(["", "## Recovery Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in report["warnings"])
    lines.append("")
    return "\n".join(lines)


def _backfill_observed_order(
    journal: ExecutionJournal,
    intent: dict[str, Any],
    broker_order: dict[str, Any],
    strategy_id: str,
    recorded_at: datetime,
) -> None:
    observed_id = str(broker_order.get("client_order_id", ""))
    for trade in intent["trades"]:
        client_order_id = deterministic_client_order_id(
            strategy_id=strategy_id,
            run_id=intent["run_id"],
            signal_date=intent["signal_date"],
            symbol=trade["symbol"],
            side=trade["side"],
        )
        if client_order_id != observed_id:
            continue
        journal.record_planned_order(
            run_id=intent["run_id"],
            client_order_id=client_order_id,
            symbol=trade["symbol"],
            side=trade["side"],
            requested_qty=_finite(trade["estimated_quantity"], "estimated_quantity"),
            requested_limit_price=_finite(trade["reference_price"], "reference_price"),
            recorded_at=recorded_at,
        )
        return


def _consume_trade_updates(
    journal: ExecutionJournal,
    trade_updates: Iterable[dict[str, Any]] | None,
    observed_at: datetime,
) -> list[str]:
    if trade_updates is None:
        return []
    warnings: list[str] = []
    known_ids = journal.known_client_order_ids()
    try:
        for event in trade_updates:
            order = event.get("order", event)
            client_order_id = str(order.get("client_order_id", ""))
            if client_order_id not in known_ids:
                continue
            update = dict(order)
            update["status"] = event.get("event", update.get("status"))
            journal.apply_order_update(
                update, observed_at=observed_at, source="trade_update"
            )
    except (ConnectionError, TimeoutError, BrokerError):
        warnings.append("trade_update_stream_disconnected_rest_resync_used")
    return warnings


def _fetch_broker_state(
    broker: PaperBroker, after: str, attempts: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    warnings: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            return (
                broker.list_orders(status="open"),
                broker.list_orders(status="all", after=after),
                broker.list_positions(),
                warnings,
            )
        except BrokerError as error:
            warnings.append(f"rest_attempt_{attempt}_failed")
            if attempt == attempts:
                raise ReconciliationError(
                    "Unable to establish authoritative Alpaca paper state."
                ) from error
    raise AssertionError("unreachable")


def _order_report(
    orders: list[dict[str, Any]], intent: dict[str, Any]
) -> list[dict[str, Any]]:
    reference = {
        (str(trade["symbol"]).upper(), str(trade["side"])): _finite(
            trade["reference_price"], "reference_price"
        )
        for trade in intent["trades"]
    }
    rows = []
    for order in orders:
        requested = float(order["requested_qty"])
        filled = float(order["filled_qty"])
        average = order["average_fill_price"]
        reference_price = reference[(order["symbol"], order["side"])]
        slippage = None
        if average is not None and filled > 0:
            direction = 1 if order["side"] == "buy" else -1
            slippage = (
                direction
                * (float(average) - reference_price)
                / reference_price
                * 10_000
            )
        rows.append(
            {
                "client_order_id": order["client_order_id"],
                "broker_order_id": order["broker_order_id"],
                "symbol": order["symbol"],
                "side": order["side"],
                "status": order["status"],
                "requested_qty": requested,
                "requested_limit_price": float(order["requested_limit_price"]),
                "reference_price": reference_price,
                "filled_qty": filled,
                "average_fill_price": average,
                "fill_percentage": min(1.0, filled / requested),
                "adverse_slippage_bps": slippage,
                "failure_reason": order["rejection_reason"],
                "cancel_requested_at_utc": order["cancel_requested_at_utc"],
            }
        )
    return rows


def _position_report(
    positions: list[dict[str, Any]], intent: dict[str, Any]
) -> list[dict[str, Any]]:
    actual = {
        str(position.get("symbol", "")).upper(): {
            "quantity": _finite(position.get("qty", 0), "position.qty"),
            "notional": _finite(
                position.get("market_value", 0), "position.market_value"
            ),
        }
        for position in positions
    }
    targets = {
        str(holding["symbol"]).upper(): _finite(
            holding["target_notional"], "holding.target_notional"
        )
        for holding in intent["portfolio"]["holdings"]
    }
    rows = []
    for symbol in sorted(set(actual) | set(targets)):
        target = targets.get(symbol, 0.0)
        actual_notional = actual.get(symbol, {}).get("notional", 0.0)
        rows.append(
            {
                "symbol": symbol,
                "target_notional": target,
                "actual_quantity": actual.get(symbol, {}).get("quantity", 0.0),
                "actual_notional": actual_notional,
                "residual_notional": target - actual_notional,
            }
        )
    return rows


def _summary(
    orders: list[dict[str, Any]], positions: list[dict[str, Any]]
) -> dict[str, Any]:
    requested = sum(
        order["requested_qty"] * order["reference_price"] for order in orders
    )
    filled = sum(order["filled_qty"] * order["reference_price"] for order in orders)
    slippage = [
        order["adverse_slippage_bps"]
        for order in orders
        if order["adverse_slippage_bps"] is not None
    ]
    return {
        "order_count": len(orders),
        "terminal_order_count": sum(
            order["status"] in {"filled", "canceled", "expired", "replaced", "rejected"}
            for order in orders
        ),
        "fill_percentage": 0.0 if requested == 0 else min(1.0, filled / requested),
        "average_adverse_slippage_bps": 0.0
        if not slippage
        else sum(slippage) / len(slippage),
        "absolute_residual_notional": sum(
            abs(position["residual_notional"]) for position in positions
        ),
        "failure_count": sum(order["failure_reason"] is not None for order in orders),
    }


def _reconciliation_status(
    orders: list[dict[str, Any]], positions: list[dict[str, Any]]
) -> str:
    if any(order["status"] == "rejected" for order in orders):
        return "attention_required"
    if any(abs(position["residual_notional"]) > 1.0 for position in positions):
        return "in_progress"
    if orders and all(
        order["status"] in {"filled", "canceled", "expired", "replaced"}
        for order in orders
    ):
        return "reconciled"
    return "in_progress"


def _unique_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {}
    for index, order in enumerate(orders):
        key = str(order.get("id") or order.get("client_order_id") or index)
        unique[key] = order
    return list(unique.values())


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric.") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative.")
    return result


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Reconciliation timestamps must include a timezone.")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _aware_utc(value).isoformat()
