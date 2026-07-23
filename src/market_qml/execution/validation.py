"""Shadow execution evidence and manual paper-promotion gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationCriteria:
    """Objective minimum evidence required for staged promotion."""

    minimum_shadow_observations: int = 20
    minimum_shadow_calendar_days: int = 28
    minimum_paper_rebalances: int = 20
    minimum_paper_calendar_days: int = 56
    maximum_operational_error_rate: float = 0.02
    maximum_stale_signal_rate: float = 0.0
    maximum_rejection_rate: float = 0.02
    maximum_tracking_error_fraction: float = 0.05
    maximum_adverse_slippage_bps: float = 50.0
    maximum_drawdown_fraction: float = 0.15
    maximum_gross_exposure: float = 0.90


def create_shadow_record(
    intent: dict[str, Any], *, observed_at: datetime
) -> dict[str, Any]:
    """Create a no-network proposed-order record with no submission capability."""
    checked_at = _aware_utc(observed_at)
    required = {
        "run_id",
        "signal_date",
        "as_of_utc",
        "model",
        "portfolio",
        "trades",
        "account_equity",
    }
    missing = required - set(intent)
    if missing:
        raise ValueError("Intent is missing fields: " + ", ".join(sorted(missing)))
    intent_as_of = _parse_timestamp(intent["as_of_utc"], "intent.as_of_utc")
    signal_date = datetime.fromisoformat(str(intent["signal_date"])).date()
    age_days = (checked_at.date() - signal_date).days
    if age_days < 0:
        raise ValueError("Shadow intent signal date is in the future.")
    trades = []
    for trade in intent["trades"]:
        trades.append(
            {
                "symbol": str(trade["symbol"]).upper(),
                "side": str(trade["side"]),
                "notional": _finite(trade["notional"], "trade.notional"),
                "estimated_quantity": _finite(
                    trade["estimated_quantity"], "trade.estimated_quantity"
                ),
                "reference_price": _finite(
                    trade["reference_price"], "trade.reference_price"
                ),
            }
        )
    return {
        "schema_version": 1,
        "mode": "shadow",
        "submission_capability": "none",
        "paper_only": True,
        "status": "approved",
        "run_id": intent["run_id"],
        "signal_date": str(intent["signal_date"]),
        "intent_as_of_utc": intent_as_of.isoformat(),
        "observed_at_utc": checked_at.isoformat(),
        "signal_age_days": age_days,
        "model": {
            "model_name": intent["model"]["model_name"],
            "artifact_id": intent["model"]["artifact_id"],
            "feature_version": intent["model"]["feature_version"],
        },
        "account_equity": _finite(intent["account_equity"], "account_equity"),
        "portfolio_turnover": _finite(
            intent["portfolio"]["turnover"], "portfolio.turnover"
        ),
        "gross_exposure": _finite(
            intent["portfolio"]["invested_weight"], "portfolio.invested_weight"
        ),
        "proposed_orders": sorted(
            trades, key=lambda item: (item["side"], item["symbol"])
        ),
    }


def save_shadow_record(record: dict[str, Any], path: str | Path) -> None:
    """Archive a shadow observation once without importing any broker client."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(record, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def build_validation_report(
    shadow_records: list[dict[str, Any]],
    reconciliation_reports: list[dict[str, Any]],
    valuations: pd.DataFrame,
    *,
    backtest_summary: dict[str, Any],
    generated_at: datetime,
    criteria: ValidationCriteria | None = None,
) -> dict[str, Any]:
    """Compare shadow intent, paper execution, valuation, and backtest evidence."""
    criteria = criteria or ValidationCriteria()
    _validate_criteria(criteria)
    valuation_metrics = _valuation_metrics(valuations)
    shadow_metrics = _shadow_metrics(shadow_records)
    paper_metrics = _paper_metrics(reconciliation_reports, shadow_records)
    operational = _operational_metrics(shadow_records, reconciliation_reports)
    comparison = _backtest_comparison(
        backtest_summary, shadow_metrics, paper_metrics, valuation_metrics
    )
    metrics = {
        "shadow": shadow_metrics,
        "paper": paper_metrics,
        "operational": operational,
        "valuation": valuation_metrics,
        "backtest_comparison": comparison,
    }
    gates = {
        "shadow_to_paper": _shadow_gates(metrics, criteria),
        "paper_to_live_review": _paper_gates(metrics, criteria),
    }
    for gate in gates.values():
        gate["eligible"] = all(check["passed"] for check in gate["checks"])
        gate["manual_approval_required"] = True
    report = {
        "schema_version": 1,
        "report_type": "shadow_paper_validation",
        "paper_only": True,
        "live_trading_supported": False,
        "generated_at_utc": _aware_utc(generated_at).isoformat(),
        "criteria": asdict(criteria),
        "metrics": metrics,
        "promotion_gates": gates,
        "limitations": [
            "Paper fills are simulated and do not model market impact, information leakage, latency slippage, or queue position.",
            "Eligibility permits only a manually approved paper stage; it is not evidence that live trading is appropriate.",
        ],
    }
    report["report_digest"] = validation_report_digest(report)
    return report


def validation_report_digest(report: dict[str, Any]) -> str:
    """Hash canonical report content without its self-referential digest."""
    content = {key: value for key, value in report.items() if key != "report_digest"}
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_paper_promotion_approval(
    report: dict[str, Any], approval: dict[str, Any]
) -> None:
    """Require a named human approval bound to eligible shadow evidence."""
    if not isinstance(report, dict) or not isinstance(approval, dict):
        raise ValueError("Validation report and approval must be JSON objects.")
    gate = report.get("promotion_gates", {}).get("shadow_to_paper", {})
    if not gate.get("eligible", False):
        raise ValueError("Shadow evidence is not eligible for paper promotion.")
    required = {
        "schema_version",
        "stage",
        "decision",
        "approved_by",
        "approved_at_utc",
        "report_digest",
        "acknowledgements",
    }
    missing = required - set(approval)
    if missing:
        raise ValueError("Approval is missing fields: " + ", ".join(sorted(missing)))
    if approval["schema_version"] != 1:
        raise ValueError("Unsupported approval schema_version.")
    if approval["stage"] != "shadow_to_paper" or approval["decision"] != "approved":
        raise ValueError("Approval must explicitly approve shadow_to_paper promotion.")
    if (
        not isinstance(approval["approved_by"], str)
        or not approval["approved_by"].strip()
    ):
        raise ValueError("Approval must identify the human reviewer.")
    _parse_timestamp(approval["approved_at_utc"], "approved_at_utc")
    expected = validation_report_digest(report)
    if report.get("report_digest") != expected or approval["report_digest"] != expected:
        raise ValueError("Approval is not bound to this validation report.")
    acknowledgements = approval["acknowledgements"]
    required_acknowledgements = {
        "paper_fills_are_simulated",
        "kill_switch_tested",
        "no_live_trading_authorized",
    }
    if not isinstance(acknowledgements, dict) or any(
        acknowledgements.get(item) is not True for item in required_acknowledgements
    ):
        raise ValueError("Approval acknowledgements are incomplete.")


def save_validation_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def _shadow_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(str(record["signal_date"]) for record in records)
    return {
        "observation_count": len(records),
        "calendar_days": _calendar_span(dates),
        "average_intended_turnover": _average(
            [record.get("portfolio_turnover", 0) for record in records]
        ),
        "average_gross_exposure": _average(
            [record.get("gross_exposure", 0) for record in records]
        ),
        "maximum_gross_exposure": _maximum(
            [record.get("gross_exposure", 0) for record in records]
        ),
        "stale_signal_count": sum(
            record.get("signal_age_days", 0) > 0 for record in records
        ),
    }


def _paper_metrics(
    reports: list[dict[str, Any]], shadow_records: list[dict[str, Any]]
) -> dict[str, Any]:
    equity_by_run = {
        record["run_id"]: _finite(record["account_equity"], "account_equity")
        for record in shadow_records
    }
    orders = [order for report in reports for order in report.get("orders", [])]
    executed_turnovers = []
    tracking_errors = []
    for report in reports:
        equity = equity_by_run.get(report.get("run_id"))
        if equity:
            executed = sum(
                _finite(order.get("filled_qty", 0), "filled_qty")
                * _finite(
                    order.get("average_fill_price") or order.get("reference_price", 0),
                    "fill_price",
                )
                for order in report.get("orders", [])
            )
            executed_turnovers.append(executed / equity)
        target = sum(
            abs(_finite(position.get("target_notional", 0), "target_notional"))
            for position in report.get("positions", [])
        )
        residual = sum(
            abs(
                _finite(
                    position.get("residual_notional", 0),
                    "residual_notional",
                    signed=True,
                )
            )
            for position in report.get("positions", [])
        )
        tracking_errors.append(0.0 if target == 0 else residual / target)
    slippage = [
        _finite(
            order["adverse_slippage_bps"],
            "adverse_slippage_bps",
            signed=True,
        )
        for order in orders
        if order.get("adverse_slippage_bps") is not None
    ]
    quoted_slippage = []
    for order in orders:
        reference = _finite(order.get("reference_price", 0), "reference_price")
        limit_price = _finite(
            order.get("requested_limit_price", reference), "requested_limit_price"
        )
        if reference > 0:
            direction = 1 if order.get("side") == "buy" else -1
            quoted_slippage.append(
                direction * (limit_price - reference) / reference * 10_000
            )
    dates = sorted(str(report["signal_date"]) for report in reports)
    return {
        "rebalance_count": len({report.get("run_id") for report in reports}),
        "calendar_days": _calendar_span(dates),
        "average_executed_turnover": _average(executed_turnovers),
        "average_quoted_slippage_bps": _average(quoted_slippage),
        "average_paper_fill_slippage_bps": _average(slippage),
        "average_tracking_error_fraction": _average(tracking_errors),
        "missed_order_count": sum(
            order.get("status") in {"canceled", "expired"}
            and order.get("filled_qty", 0) == 0
            for order in orders
        ),
        "canceled_order_count": sum(
            order.get("status") == "canceled" for order in orders
        ),
        "rejected_order_count": sum(
            order.get("status") == "rejected" for order in orders
        ),
        "order_count": len(orders),
    }


def _operational_metrics(
    shadow_records: list[dict[str, Any]], reports: list[dict[str, Any]]
) -> dict[str, Any]:
    cycles = len(shadow_records) + len(reports)
    failures = sum(record.get("status") != "approved" for record in shadow_records)
    failures += sum(
        report.get("status") == "attention_required" or bool(report.get("warnings"))
        for report in reports
    )
    stale = sum(record.get("signal_age_days", 0) > 0 for record in shadow_records)
    return {
        "cycle_count": cycles,
        "failure_count": failures,
        "operational_error_rate": 0.0 if cycles == 0 else failures / cycles,
        "stale_signal_rate": 0.0 if not shadow_records else stale / len(shadow_records),
    }


def _valuation_metrics(valuations: pd.DataFrame) -> dict[str, Any]:
    required = {"date", "equity", "gross_exposure"}
    missing = required - set(valuations.columns)
    if missing:
        raise ValueError(
            "Valuations are missing columns: " + ", ".join(sorted(missing))
        )
    if valuations.empty:
        return {
            "observation_count": 0,
            "cumulative_return": 0.0,
            "maximum_drawdown": 0.0,
            "average_gross_exposure": 0.0,
            "maximum_gross_exposure": 0.0,
        }
    data = valuations.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    data["gross_exposure"] = pd.to_numeric(data["gross_exposure"], errors="coerce")
    if data[["date", "equity", "gross_exposure"]].isna().any().any():
        raise ValueError("Valuations contain invalid or missing values.")
    if not data["equity"].map(math.isfinite).all() or not data["equity"].gt(0).all():
        raise ValueError("Valuation equity must be finite and positive.")
    if (
        not data["gross_exposure"].map(math.isfinite).all()
        or not data["gross_exposure"].ge(0).all()
    ):
        raise ValueError("Valuation exposure must be finite and non-negative.")
    data = data.sort_values("date")
    equity = data["equity"]
    drawdown = equity / equity.cummax() - 1
    return {
        "observation_count": len(data),
        "cumulative_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "maximum_drawdown": abs(float(drawdown.min())),
        "average_gross_exposure": float(data["gross_exposure"].mean()),
        "maximum_gross_exposure": float(data["gross_exposure"].max()),
    }


def _backtest_comparison(
    backtest: dict[str, Any],
    shadow: dict[str, Any],
    paper: dict[str, Any],
    valuation: dict[str, Any],
) -> dict[str, Any]:
    assumed_cost = _finite(
        backtest.get("transaction_cost_bps", 0), "transaction_cost_bps"
    )
    return {
        "backtest_transaction_cost_bps": assumed_cost,
        "paper_fill_slippage_minus_assumption_bps": paper[
            "average_paper_fill_slippage_bps"
        ]
        - assumed_cost,
        "backtest_average_turnover": _finite(
            backtest.get("average_turnover", 0), "average_turnover"
        ),
        "shadow_turnover_difference": shadow["average_intended_turnover"]
        - _finite(backtest.get("average_turnover", 0), "average_turnover"),
        "paper_turnover_difference": paper["average_executed_turnover"]
        - _finite(backtest.get("average_turnover", 0), "average_turnover"),
        "backtest_cumulative_net_return": _finite(
            backtest.get("cumulative_net_return", 0),
            "cumulative_net_return",
            signed=True,
        ),
        "paper_return_difference": valuation["cumulative_return"]
        - _finite(
            backtest.get("cumulative_net_return", 0),
            "cumulative_net_return",
            signed=True,
        ),
        "backtest_maximum_drawdown": abs(
            _finite(
                backtest.get("net_max_drawdown", 0), "net_max_drawdown", signed=True
            )
        ),
        "paper_drawdown_difference": valuation["maximum_drawdown"]
        - abs(
            _finite(
                backtest.get("net_max_drawdown", 0), "net_max_drawdown", signed=True
            )
        ),
    }


def _shadow_gates(
    metrics: dict[str, Any], criteria: ValidationCriteria
) -> dict[str, Any]:
    shadow = metrics["shadow"]
    operational = metrics["operational"]
    checks = [
        _check(
            "minimum_shadow_observations",
            shadow["observation_count"],
            criteria.minimum_shadow_observations,
            ">=",
        ),
        _check(
            "minimum_shadow_calendar_days",
            shadow["calendar_days"],
            criteria.minimum_shadow_calendar_days,
            ">=",
        ),
        _check(
            "maximum_operational_error_rate",
            operational["operational_error_rate"],
            criteria.maximum_operational_error_rate,
            "<=",
        ),
        _check(
            "maximum_stale_signal_rate",
            operational["stale_signal_rate"],
            criteria.maximum_stale_signal_rate,
            "<=",
        ),
        _check(
            "maximum_gross_exposure",
            shadow["maximum_gross_exposure"],
            criteria.maximum_gross_exposure,
            "<=",
        ),
    ]
    return {"checks": checks}


def _paper_gates(
    metrics: dict[str, Any], criteria: ValidationCriteria
) -> dict[str, Any]:
    paper = metrics["paper"]
    operational = metrics["operational"]
    valuation = metrics["valuation"]
    rejection_rate = (
        0.0
        if paper["order_count"] == 0
        else paper["rejected_order_count"] / paper["order_count"]
    )
    checks = [
        _check(
            "minimum_paper_rebalances",
            paper["rebalance_count"],
            criteria.minimum_paper_rebalances,
            ">=",
        ),
        _check(
            "minimum_paper_calendar_days",
            paper["calendar_days"],
            criteria.minimum_paper_calendar_days,
            ">=",
        ),
        _check(
            "maximum_operational_error_rate",
            operational["operational_error_rate"],
            criteria.maximum_operational_error_rate,
            "<=",
        ),
        _check(
            "maximum_rejection_rate",
            rejection_rate,
            criteria.maximum_rejection_rate,
            "<=",
        ),
        _check(
            "maximum_tracking_error_fraction",
            paper["average_tracking_error_fraction"],
            criteria.maximum_tracking_error_fraction,
            "<=",
        ),
        _check(
            "maximum_adverse_slippage_bps",
            paper["average_paper_fill_slippage_bps"],
            criteria.maximum_adverse_slippage_bps,
            "<=",
        ),
        _check(
            "maximum_drawdown_fraction",
            valuation["maximum_drawdown"],
            criteria.maximum_drawdown_fraction,
            "<=",
        ),
        _check(
            "maximum_gross_exposure",
            valuation["maximum_gross_exposure"],
            criteria.maximum_gross_exposure,
            "<=",
        ),
    ]
    return {"checks": checks}


def _check(name: str, actual: float, threshold: float, operator: str) -> dict[str, Any]:
    passed = actual >= threshold if operator == ">=" else actual <= threshold
    return {
        "name": name,
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _validate_criteria(criteria: ValidationCriteria) -> None:
    for field in (
        "minimum_shadow_observations",
        "minimum_shadow_calendar_days",
        "minimum_paper_rebalances",
        "minimum_paper_calendar_days",
    ):
        if getattr(criteria, field) <= 0:
            raise ValueError(f"{field} must be positive.")
    for field in (
        "maximum_operational_error_rate",
        "maximum_stale_signal_rate",
        "maximum_rejection_rate",
        "maximum_tracking_error_fraction",
        "maximum_drawdown_fraction",
        "maximum_gross_exposure",
    ):
        value = getattr(criteria, field)
        if not 0 <= value <= 1:
            raise ValueError(f"{field} must be between 0 and 1.")
    if criteria.maximum_adverse_slippage_bps < 0:
        raise ValueError("maximum_adverse_slippage_bps must be non-negative.")


def _calendar_span(dates: list[str]) -> int:
    if not dates:
        return 0
    return (
        datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])
    ).days + 1


def _average(values: list[Any]) -> float:
    numbers = [_finite(value, "metric") for value in values]
    return 0.0 if not numbers else sum(numbers) / len(numbers)


def _maximum(values: list[Any]) -> float:
    numbers = [_finite(value, "metric") for value in values]
    return 0.0 if not numbers else max(numbers)


def _finite(value: Any, field: str, *, signed: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric.") from error
    if not math.isfinite(result) or (result < 0 and not signed):
        qualifier = "finite" if signed else "finite and non-negative"
        raise ValueError(f"{field} must be {qualifier}.")
    return result


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} is invalid.") from error
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Validation timestamps must include a timezone.")
    return value.astimezone(timezone.utc)
