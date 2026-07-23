"""Fail-closed pre-trade validation and Alpaca paper order submission."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from market_qml.execution.broker import BrokerError, PaperBroker

NEW_YORK = ZoneInfo("America/New_York")
CLIENT_ORDER_PREFIX = "mqml"
TERMINAL_ORDER_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}


class PreTradeError(RuntimeError):
    """A machine-readable pre-trade validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PaperExecutionPolicy:
    """Independent limits applied again immediately before submission."""

    strategy_id: str = "market-qml"
    max_intent_age_minutes: int = 30
    max_order_count: int = 20
    max_gross_exposure: float = 0.90
    max_position_weight: float = 0.20
    max_turnover: float = 1.0
    minimum_order_notional: float = 10.0
    cash_reserve_weight: float = 0.10
    max_equity_drift_fraction: float = 0.02
    max_daily_loss_fraction: float = 0.02
    limit_price_buffer_bps: float = 25.0
    cancel_after_minutes: int = 15


def execute_paper_intent(
    broker: PaperBroker,
    intent: dict[str, Any],
    *,
    now: datetime,
    policy: PaperExecutionPolicy | None = None,
    submit: bool = False,
    submission_enabled: bool = False,
    kill_switch_active: bool = True,
    known_client_order_ids: set[str] | None = None,
    planned_order_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate an intent and optionally submit guarded paper limit orders."""
    policy = policy or PaperExecutionPolicy()
    _validate_policy(policy)
    checked_now = _aware_utc(now)
    _validate_submission_gates(
        submit=submit,
        submission_enabled=submission_enabled,
        kill_switch_active=kill_switch_active,
    )
    _validate_intent(intent, checked_now, policy)

    account = broker.get_account()
    positions = broker.list_positions()
    open_orders = broker.list_orders(status="open")
    recent_orders = broker.list_orders(status="all", after=intent["as_of_utc"])
    orders = _unique_orders([*open_orders, *recent_orders])
    calendar = broker.get_calendar(intent["signal_date"], intent["signal_date"])
    _validate_account(account, policy, intent)
    _validate_market_window(calendar, checked_now, intent["signal_date"])
    _validate_positions(positions)

    existing_client_ids = {
        str(order.get("client_order_id", "")) for order in orders
    } | (known_client_order_ids or set())
    open_symbols = {
        str(order.get("symbol", "")).upper()
        for order in orders
        if str(order.get("status", "")).lower() not in TERMINAL_ORDER_STATUSES
    }
    position_quantities = {
        str(position.get("symbol", "")).upper(): _number(
            position.get("qty"), "position.qty"
        )
        for position in positions
    }

    planned = []
    decisions = []
    for trade in sorted(
        intent["trades"], key=lambda item: (item["side"] != "sell", item["symbol"])
    ):
        symbol = str(trade["symbol"]).upper()
        client_order_id = deterministic_client_order_id(
            strategy_id=policy.strategy_id,
            run_id=intent["run_id"],
            signal_date=intent["signal_date"],
            symbol=symbol,
            side=trade["side"],
        )
        if client_order_id in existing_client_ids:
            decisions.append(
                _decision(symbol, trade["side"], "skipped", "duplicate_client_order_id")
            )
            continue
        if symbol in open_symbols:
            decisions.append(
                _decision(symbol, trade["side"], "rejected", "conflicting_open_order")
            )
            continue
        asset = broker.get_asset(symbol)
        rejection = _asset_rejection(asset)
        if rejection is not None:
            decisions.append(_decision(symbol, trade["side"], "rejected", rejection))
            continue
        quantity = _number(trade["estimated_quantity"], "estimated_quantity")
        if not _whole(quantity) and not bool(asset.get("fractionable", False)):
            decisions.append(
                _decision(symbol, trade["side"], "rejected", "asset_not_fractionable")
            )
            continue
        if (
            trade["side"] == "sell"
            and quantity > position_quantities.get(symbol, 0.0) + 1e-9
        ):
            decisions.append(
                _decision(symbol, trade["side"], "rejected", "insufficient_position")
            )
            continue
        order = _order_payload(trade, client_order_id, policy)
        planned.append(order)
        decisions.append(
            _decision(symbol, trade["side"], "approved", "risk_checks_passed")
        )

    if len(planned) > policy.max_order_count:
        raise PreTradeError(
            "max_order_count_exceeded",
            f"Approved order count {len(planned)} exceeds {policy.max_order_count}.",
        )
    blocked = any(decision["status"] == "rejected" for decision in decisions)

    submitted = []
    submission_error = None
    if submit and not blocked:
        for order in planned:
            if planned_order_callback is not None:
                planned_order_callback(order)
            try:
                response = broker.submit_order(order)
            except BrokerError:
                submission_error = {
                    "code": "broker_submission_failed",
                    "client_order_id": order["client_order_id"],
                    "broker_order_id": str(response.get("id", "")),
                    "symbol": order["symbol"],
                }
                break
            submitted.append(
                {
                    "client_order_id": order["client_order_id"],
                    "symbol": order["symbol"],
                    "status": str(response.get("status", "submitted")),
                }
            )

    status = "blocked" if blocked else "approved"
    if submission_error is not None:
        status = "submission_failed"
    report = {
        "schema_version": 1,
        "mode": "paper_submit" if submit else "dry_run",
        "status": status,
        "paper_only": True,
        "run_id": intent["run_id"],
        "checked_at_utc": checked_now.isoformat(),
        "policy": asdict(policy),
        "decisions": decisions,
        "orders": planned,
        "submitted": submitted,
    }
    if submission_error is not None:
        report["error"] = submission_error
    return report


def cancel_stale_paper_orders(
    broker: PaperBroker,
    *,
    now: datetime,
    cancel_after_minutes: int,
    submission_enabled: bool,
    kill_switch_active: bool,
) -> list[dict[str, str]]:
    """Cancel only stale orders created by this project after explicit gates pass."""
    _validate_submission_gates(
        submit=True,
        submission_enabled=submission_enabled,
        kill_switch_active=kill_switch_active,
    )
    if cancel_after_minutes <= 0:
        raise ValueError("cancel_after_minutes must be positive.")
    checked_now = _aware_utc(now)
    results = []
    for order in broker.list_orders(status="open"):
        client_order_id = str(order.get("client_order_id", ""))
        if not client_order_id.startswith(f"{CLIENT_ORDER_PREFIX}-"):
            continue
        submitted_at = _parse_timestamp(order.get("submitted_at"), "submitted_at")
        if checked_now - submitted_at < timedelta(minutes=cancel_after_minutes):
            continue
        order_id = str(order.get("id", ""))
        if not order_id:
            raise PreTradeError(
                "missing_order_id", "A stale project order has no order ID."
            )
        broker.cancel_order(order_id)
        results.append(
            {"client_order_id": client_order_id, "status": "cancel_requested"}
        )
    return results


def deterministic_client_order_id(
    *, strategy_id: str, run_id: str, signal_date: str, symbol: str, side: str
) -> str:
    """Build a stable Alpaca client ID without exposing account information."""
    source = "|".join((strategy_id, run_id, signal_date, symbol.upper(), side.lower()))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    return f"{CLIENT_ORDER_PREFIX}-{signal_date.replace('-', '')}-{digest}"


def _validate_submission_gates(
    *, submit: bool, submission_enabled: bool, kill_switch_active: bool
) -> None:
    if not submit:
        return
    if kill_switch_active:
        raise PreTradeError(
            "kill_switch_active", "Paper submission kill switch is active."
        )
    if not submission_enabled:
        raise PreTradeError(
            "paper_submission_disabled",
            "Paper submission is not enabled in the environment.",
        )


def _validate_intent(
    intent: dict[str, Any], now: datetime, policy: PaperExecutionPolicy
) -> None:
    required = {
        "schema_version",
        "intent_type",
        "broker_submission_allowed",
        "as_of_utc",
        "signal_date",
        "account_equity",
        "model",
        "policy",
        "portfolio",
        "trades",
        "run_id",
    }
    missing = required - set(intent)
    if missing:
        raise PreTradeError(
            "invalid_intent", "Intent is missing fields: " + ", ".join(sorted(missing))
        )
    if intent["schema_version"] != 1 or intent["intent_type"] != "paper_trade_dry_run":
        raise PreTradeError(
            "invalid_intent", "Unsupported trade-intent schema or type."
        )
    if intent["broker_submission_allowed"] is not False:
        raise PreTradeError(
            "invalid_intent", "Only broker-independent dry-run intents are accepted."
        )
    as_of = _parse_timestamp(intent["as_of_utc"], "intent.as_of_utc")
    age = now - as_of
    if age < timedelta(0) or age > timedelta(minutes=policy.max_intent_age_minutes):
        raise PreTradeError("stale_intent", "Trade intent is stale or from the future.")
    model = intent["model"]
    if not isinstance(model, dict) or model.get("selection_scope") not in {
        "development_only",
        "development_validation",
        "production_candidate",
    }:
        raise PreTradeError(
            "unapproved_model", "Intent does not reference an approved model."
        )
    portfolio = intent["portfolio"]
    gross = _number(portfolio.get("invested_weight"), "portfolio.invested_weight")
    turnover = _number(portfolio.get("turnover"), "portfolio.turnover")
    if gross > policy.max_gross_exposure + 1e-12:
        raise PreTradeError(
            "gross_exposure_exceeded", "Intent exceeds gross exposure limit."
        )
    if turnover > policy.max_turnover + 1e-12:
        raise PreTradeError("turnover_exceeded", "Intent exceeds turnover limit.")
    holdings = portfolio.get("holdings")
    if not isinstance(holdings, list):
        raise PreTradeError("invalid_intent", "Portfolio holdings must be a list.")
    if any(
        _number(item.get("target_weight"), "holding.target_weight")
        > policy.max_position_weight + 1e-12
        for item in holdings
    ):
        raise PreTradeError("position_limit_exceeded", "Intent exceeds position limit.")
    trades = intent["trades"]
    if not isinstance(trades, list):
        raise PreTradeError("invalid_intent", "Intent trades must be a list.")
    for trade in trades:
        if trade.get("side") not in {"buy", "sell"}:
            raise PreTradeError(
                "unsupported_side", "Only long-only buys and sells are allowed."
            )
        if (
            _number(trade.get("notional"), "trade.notional")
            < policy.minimum_order_notional
        ):
            raise PreTradeError(
                "order_below_minimum", "Intent contains a sub-minimum order."
            )


def _validate_account(
    account: dict[str, Any], policy: PaperExecutionPolicy, intent: dict[str, Any]
) -> None:
    if bool(account.get("trading_blocked", True)):
        raise PreTradeError(
            "account_trading_blocked", "Paper account is trading-blocked."
        )
    if str(account.get("status", "")).upper() != "ACTIVE":
        raise PreTradeError("account_inactive", "Paper account is not active.")
    equity = _number(account.get("equity"), "account.equity")
    cash = _number(account.get("cash"), "account.cash")
    buying_power = _number(account.get("buying_power"), "account.buying_power")
    if equity <= 0 or cash < 0 or buying_power < 0:
        raise PreTradeError("invalid_account", "Paper account balances are invalid.")
    intended_equity = _number(intent["account_equity"], "intent.account_equity")
    equity_drift = abs(equity - intended_equity) / intended_equity
    if equity_drift > policy.max_equity_drift_fraction:
        raise PreTradeError(
            "account_equity_drift", "Paper account equity has drifted from the intent."
        )
    previous_equity = _number(account.get("last_equity"), "account.last_equity")
    if previous_equity <= 0:
        raise PreTradeError("invalid_account", "Previous paper equity is invalid.")
    daily_loss = max(0.0, (previous_equity - equity) / previous_equity)
    if daily_loss > policy.max_daily_loss_fraction:
        raise PreTradeError(
            "daily_loss_limit_breached",
            "Paper account daily loss exceeds the configured kill threshold.",
        )
    buy_notional = sum(
        _number(trade["notional"], "trade.notional")
        for trade in intent["trades"]
        if trade["side"] == "buy"
    )
    reserve = equity * policy.cash_reserve_weight
    if buy_notional > buying_power + 1e-9:
        raise PreTradeError(
            "insufficient_buying_power", "Insufficient paper buying power."
        )
    if buy_notional > max(0.0, cash - reserve) + 1e-9:
        raise PreTradeError(
            "cash_reserve_breached", "Orders would breach the cash reserve."
        )


def _validate_market_window(
    calendar: list[dict[str, Any]], now: datetime, signal_date: str
) -> None:
    row = next((item for item in calendar if item.get("date") == signal_date), None)
    if row is None:
        raise PreTradeError("market_closed", "Signal date is not a trading day.")
    local_now = now.astimezone(NEW_YORK)
    if local_now.date().isoformat() != signal_date:
        raise PreTradeError(
            "outside_execution_date", "Execution date differs from signal date."
        )
    open_time = _market_time(row.get("open"), "calendar.open")
    close_time = _market_time(row.get("close"), "calendar.close")
    if not open_time <= local_now.time().replace(tzinfo=None) < close_time:
        raise PreTradeError(
            "outside_market_hours", "Execution is outside regular market hours."
        )


def _validate_positions(positions: list[dict[str, Any]]) -> None:
    for position in positions:
        if str(position.get("asset_class", "us_equity")) != "us_equity":
            raise PreTradeError(
                "unsupported_asset_class", "Only US equity positions are allowed."
            )
        if (
            _number(position.get("qty"), "position.qty") < 0
            or str(position.get("side", "long")).lower() == "short"
        ):
            raise PreTradeError("short_position", "Short positions are not supported.")


def _asset_rejection(asset: dict[str, Any]) -> str | None:
    if str(asset.get("class", "")) != "us_equity":
        return "unsupported_asset_class"
    if str(asset.get("status", "")) != "active":
        return "asset_inactive"
    if not bool(asset.get("tradable", False)):
        return "asset_not_tradable"
    return None


def _order_payload(
    trade: dict[str, Any], client_order_id: str, policy: PaperExecutionPolicy
) -> dict[str, Any]:
    price = _number(trade["reference_price"], "reference_price")
    if price <= 0:
        raise PreTradeError(
            "invalid_reference_price", "Reference price must be positive."
        )
    multiplier = 1 + policy.limit_price_buffer_bps / 10_000
    rounding = ROUND_UP
    if trade["side"] == "sell":
        multiplier = 1 - policy.limit_price_buffer_bps / 10_000
        rounding = ROUND_DOWN
    tick = Decimal("0.01") if price >= 1 else Decimal("0.0001")
    limit_price = (Decimal(str(price)) * Decimal(str(multiplier))).quantize(
        tick, rounding=rounding
    )
    quantity = Decimal(str(_number(trade["estimated_quantity"], "estimated_quantity")))
    if quantity <= 0:
        raise PreTradeError("invalid_quantity", "Estimated quantity must be positive.")
    return {
        "symbol": str(trade["symbol"]).upper(),
        "qty": format(quantity.normalize(), "f"),
        "side": trade["side"],
        "type": "limit",
        "time_in_force": "day",
        "limit_price": format(limit_price, "f"),
        "extended_hours": False,
        "client_order_id": client_order_id,
    }


def _validate_policy(policy: PaperExecutionPolicy) -> None:
    if not policy.strategy_id.strip():
        raise ValueError("strategy_id must be non-empty.")
    for field in ("max_intent_age_minutes", "max_order_count", "cancel_after_minutes"):
        if getattr(policy, field) <= 0:
            raise ValueError(f"{field} must be positive.")
    for field in (
        "max_gross_exposure",
        "max_position_weight",
        "max_turnover",
        "cash_reserve_weight",
        "max_equity_drift_fraction",
        "max_daily_loss_fraction",
    ):
        value = _number(getattr(policy, field), field)
        if not 0 <= value <= 1:
            raise ValueError(f"{field} must be between 0 and 1.")
    if (
        policy.minimum_order_notional < 0
        or not 0 <= policy.limit_price_buffer_bps <= 1_000
    ):
        raise ValueError("Order notional and price buffer limits must be non-negative.")


def _decision(symbol: str, side: str, status: str, reason: str) -> dict[str, str]:
    return {"symbol": symbol, "side": side, "status": status, "reason": reason}


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise PreTradeError("invalid_timestamp", f"{field} must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PreTradeError("invalid_timestamp", f"{field} is invalid.") from error
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Execution timestamps must include a timezone.")
    return value.astimezone(timezone.utc)


def _market_time(value: Any, field: str) -> time:
    if not isinstance(value, str):
        raise PreTradeError("invalid_calendar", f"{field} is missing.")
    try:
        return time.fromisoformat(value)
    except ValueError as error:
        raise PreTradeError("invalid_calendar", f"{field} is invalid.") from error


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PreTradeError(
            "invalid_numeric_value", f"{field} must be numeric."
        ) from error
    if not math.isfinite(result):
        raise PreTradeError("invalid_numeric_value", f"{field} must be finite.")
    return result


def _whole(value: float) -> bool:
    return math.isclose(value, round(value), abs_tol=1e-9)


def _unique_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for index, order in enumerate(orders):
        key = str(order.get("id") or order.get("client_order_id") or index)
        unique[key] = order
    return list(unique.values())
