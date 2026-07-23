"""Deterministic, broker-independent target portfolio generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROMOTION_SCHEMA_VERSION = 1
TRADE_INTENT_SCHEMA_VERSION = 1
REQUIRED_PROMOTION_FIELDS = {
    "artifact_id",
    "model_name",
    "model_sha256",
    "preprocessor_sha256",
    "feature_version",
    "promoted_at_utc",
    "selection_scope",
}
REQUIRED_SIGNAL_COLUMNS = {
    "artifact_id",
    "date",
    "feature_version",
    "model_name",
    "symbol",
    "predicted_outperformance_probability",
    "reference_price",
}


@dataclass(frozen=True)
class PortfolioPolicy:
    """Long-only target portfolio constraints."""

    selected_count: int = 5
    max_position_weight: float = 0.20
    cash_reserve_weight: float = 0.10
    max_turnover: float = 1.0
    minimum_trade_notional: float = 10.0
    rebalance_frequency_trading_days: int = 5
    max_signal_age_days: int = 3


def load_promotion_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate an explicit model-promotion record."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Promotion manifest must be a JSON object.")
    _validate_promotion(manifest)
    return manifest


def build_trade_intent(
    signals: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    promotion: dict[str, Any],
    account_equity: float,
    as_of: datetime,
    policy: PortfolioPolicy | None = None,
) -> dict[str, Any]:
    """Build a reproducible long-only trade intent without broker access."""
    policy = policy or PortfolioPolicy()
    _validate_policy(policy)
    _validate_promotion(promotion)
    equity = _finite_positive(account_equity, "account_equity")
    normalized_as_of = _utc_datetime(as_of, "as_of")
    promoted_at = _utc_datetime(promotion["promoted_at_utc"], "promoted_at_utc")
    if promoted_at > normalized_as_of:
        raise ValueError(
            "The model promotion timestamp is after the decision timestamp."
        )
    normalized_signals = _validate_signals(signals, normalized_as_of, policy, promotion)
    current = _validate_positions(positions, normalized_signals, equity)

    candidates = normalized_signals.loc[~normalized_signals["is_benchmark"]].copy()
    if len(candidates) < policy.selected_count:
        raise ValueError(
            f"Only {len(candidates)} eligible signals are available; "
            f"selected_count is {policy.selected_count}."
        )
    selected = candidates.sort_values(
        ["predicted_outperformance_probability", "symbol"],
        ascending=[False, True],
    ).head(policy.selected_count)

    investable_weight = 1.0 - policy.cash_reserve_weight
    target_weight = investable_weight / policy.selected_count
    if target_weight > policy.max_position_weight + 1e-12:
        raise ValueError(
            "Equal-weight target exceeds max_position_weight; increase selected_count "
            "or reduce the investable allocation."
        )
    target_weights = dict.fromkeys(selected["symbol"], target_weight)
    turnover = _turnover(current, target_weights)
    if turnover > policy.max_turnover + 1e-12:
        raise ValueError(
            f"Proposed turnover {turnover:.6f} exceeds max_turnover "
            f"{policy.max_turnover:.6f}."
        )

    score_by_symbol = selected.set_index("symbol")[
        "predicted_outperformance_probability"
    ].to_dict()
    price_by_symbol = normalized_signals.set_index("symbol")[
        "reference_price"
    ].to_dict()
    symbols = sorted(set(current) | set(target_weights))
    holdings = []
    trades = []
    for symbol in symbols:
        current_weight = current.get(symbol, 0.0)
        desired_weight = target_weights.get(symbol, 0.0)
        delta_notional = (desired_weight - current_weight) * equity
        price = price_by_symbol.get(symbol)
        holdings.append(
            {
                "symbol": symbol,
                "current_weight": current_weight,
                "target_weight": desired_weight,
                "current_notional": current_weight * equity,
                "target_notional": desired_weight * equity,
            }
        )
        if abs(delta_notional) + 1e-12 < policy.minimum_trade_notional:
            continue
        if price is None:
            raise ValueError(
                f"A reference price is required to trade current symbol {symbol}."
            )
        side = "buy" if delta_notional > 0 else "sell"
        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "notional": abs(delta_notional),
                "reference_price": price,
                "estimated_quantity": abs(delta_notional) / price,
                "score": score_by_symbol.get(symbol),
                "reason": (
                    "selected_by_promoted_model"
                    if side == "buy"
                    else "not_in_target_portfolio"
                ),
            }
        )

    signal_date = normalized_signals["date"].iloc[0].date().isoformat()
    payload: dict[str, Any] = {
        "schema_version": TRADE_INTENT_SCHEMA_VERSION,
        "intent_type": "paper_trade_dry_run",
        "broker_submission_allowed": False,
        "as_of_utc": normalized_as_of.isoformat(),
        "signal_date": signal_date,
        "account_equity": equity,
        "model": {
            field: promotion[field] for field in sorted(REQUIRED_PROMOTION_FIELDS)
        },
        "policy": asdict(policy),
        "portfolio": {
            "cash_reserve_weight": policy.cash_reserve_weight,
            "invested_weight": investable_weight,
            "turnover": turnover,
            "holdings": holdings,
        },
        "trades": trades,
    }
    payload["run_id"] = _payload_hash(payload)
    rounded = _round_floats(payload)
    if not isinstance(rounded, dict):  # pragma: no cover - payload is always a dict
        raise TypeError("Trade-intent payload must be a mapping.")
    return rounded


def save_trade_intent(intent: dict[str, Any], path: str | Path) -> None:
    """Write an intent once; existing intent files are never overwritten."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(intent, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def _validate_promotion(promotion: dict[str, Any]) -> None:
    missing = REQUIRED_PROMOTION_FIELDS - set(promotion)
    if missing:
        raise ValueError(
            "Promotion manifest is missing required fields: "
            + ", ".join(sorted(missing))
        )
    if promotion.get("schema_version") != PROMOTION_SCHEMA_VERSION:
        raise ValueError("Unsupported promotion manifest schema_version.")
    for field in REQUIRED_PROMOTION_FIELDS:
        if not isinstance(promotion[field], str) or not promotion[field].strip():
            raise ValueError(f"Promotion field '{field}' must be a non-empty string.")
    allowed_scopes = {
        "development_only",
        "development_validation",
        "production_candidate",
    }
    if promotion["selection_scope"] not in allowed_scopes:
        raise ValueError(
            "Promotion selection_scope must be development-only and cannot reference "
            "the locked test."
        )
    _utc_datetime(promotion["promoted_at_utc"], "promoted_at_utc")
    for field in ("model_sha256", "preprocessor_sha256"):
        digest = promotion[field].lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Promotion field '{field}' must be a SHA-256 digest.")


def _validate_policy(policy: PortfolioPolicy) -> None:
    if policy.selected_count <= 0:
        raise ValueError("selected_count must be positive.")
    for field in ("max_position_weight", "cash_reserve_weight", "max_turnover"):
        value = _finite_float(getattr(policy, field), field)
        if not 0 <= value <= 1:
            raise ValueError(f"{field} must be between 0 and 1.")
    if policy.cash_reserve_weight >= 1:
        raise ValueError("cash_reserve_weight must be less than 1.")
    if policy.minimum_trade_notional < 0:
        raise ValueError("minimum_trade_notional must be non-negative.")
    if policy.rebalance_frequency_trading_days <= 0:
        raise ValueError("rebalance_frequency_trading_days must be positive.")
    if policy.max_signal_age_days < 0:
        raise ValueError("max_signal_age_days must be non-negative.")


def _validate_signals(
    signals: pd.DataFrame,
    as_of: datetime,
    policy: PortfolioPolicy,
    promotion: dict[str, Any],
) -> pd.DataFrame:
    missing = REQUIRED_SIGNAL_COLUMNS - set(signals.columns)
    if missing:
        raise ValueError(
            "Signals are missing required columns: " + ", ".join(sorted(missing))
        )
    if signals.empty:
        raise ValueError("Signals are empty.")
    data = signals.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    if data["symbol"].eq("").any() or data["symbol"].duplicated().any():
        raise ValueError("Signals must contain unique, non-empty symbols.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data["date"].isna().any() or data["date"].nunique() != 1:
        raise ValueError("Signals must contain one valid signal date.")
    signal_date = data["date"].iloc[0].date()
    age = (as_of.date() - signal_date).days
    if age < 0 or age > policy.max_signal_age_days:
        raise ValueError(
            f"Signal date {signal_date.isoformat()} is stale or in the future for "
            f"as_of {as_of.date().isoformat()}."
        )
    for column in ("predicted_outperformance_probability", "reference_price"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not data[column].map(math.isfinite).all():
            raise ValueError(f"Signals contain nonfinite {column} values.")
    if not data["predicted_outperformance_probability"].between(0, 1).all():
        raise ValueError("Signal probabilities must be between 0 and 1.")
    if not data["reference_price"].gt(0).all():
        raise ValueError("Reference prices must be positive.")
    if "is_benchmark" not in data:
        data["is_benchmark"] = False
    else:
        data["is_benchmark"] = data["is_benchmark"].map(_boolean)
    _validate_signal_lineage(data, promotion)
    return data.sort_values("symbol").reset_index(drop=True)


def _validate_signal_lineage(signals: pd.DataFrame, promotion: dict[str, Any]) -> None:
    for column, promotion_field in (
        ("model_name", "model_name"),
        ("artifact_id", "artifact_id"),
        ("feature_version", "feature_version"),
    ):
        values = set(signals[column].astype(str))
        if values != {promotion[promotion_field]}:
            raise ValueError(
                f"Signal {column} does not match the promoted {promotion_field}."
            )


def _validate_positions(
    positions: pd.DataFrame, signals: pd.DataFrame, equity: float
) -> dict[str, float]:
    if positions.empty:
        return {}
    required = {"symbol", "market_value"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(
            "Positions are missing required columns: " + ", ".join(sorted(missing))
        )
    data = positions.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    if data["symbol"].eq("").any() or data["symbol"].duplicated().any():
        raise ValueError("Positions must contain unique, non-empty symbols.")
    data["market_value"] = pd.to_numeric(data["market_value"], errors="coerce")
    if not data["market_value"].map(math.isfinite).all():
        raise ValueError("Positions contain nonfinite market values.")
    if data["market_value"].lt(0).any():
        raise ValueError("Short positions are not supported.")
    if data["market_value"].sum() > equity + 1e-8:
        raise ValueError("Position market value exceeds account equity.")
    missing_prices = set(data.loc[data["market_value"].gt(0), "symbol"]) - set(
        signals["symbol"]
    )
    if missing_prices:
        raise ValueError(
            "Signals are missing reference prices for current positions: "
            + ", ".join(sorted(missing_prices))
        )
    return dict(zip(data["symbol"], data["market_value"] / equity, strict=True))


def _turnover(current: dict[str, float], target: dict[str, float]) -> float:
    symbols = set(current) | set(target)
    return 0.5 * sum(
        abs(target.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in symbols
    )


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _finite_positive(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive.")
    return result


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite and numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite and numeric.")
    return result


def _utc_datetime(value: datetime | str, field: str) -> datetime:
    try:
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value.replace("Z", "+00:00"))
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError("is_benchmark must contain boolean values.")
