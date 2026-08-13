# mypy: disable-error-code="import-untyped,no-untyped-def,no-untyped-call"
"""Fail-closed quality contracts for private market-data snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityResult:
    checks: pd.DataFrame
    quarantine: pd.DataFrame

    @property
    def passed(self) -> bool:
        critical = self.checks.loc[self.checks["severity"].eq("critical")]
        return bool(critical["passed"].all())


def validate_snapshot(
    *,
    prices: pd.DataFrame,
    assets: pd.DataFrame,
    submissions: pd.DataFrame,
    fundamentals: pd.DataFrame,
    macro_raw: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> QualityResult:
    """Validate schemas, point-in-time ordering, values, and refresh provenance."""
    now = now or pd.Timestamp.now(tz="UTC")
    now = _utc(now)
    checks: list[dict[str, object]] = []
    quarantine: list[pd.DataFrame] = []

    _required(
        checks,
        "prices",
        prices,
        {
            "symbol",
            "timestamp",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ingested_at_utc",
        },
    )
    if {"symbol", "timestamp"}.issubset(prices):
        _check(
            checks,
            "prices.unique_key",
            not prices.duplicated(["symbol", "timestamp"]).any(),
            "critical",
        )
    if {"open", "high", "low", "close", "volume"}.issubset(prices):
        numeric = prices[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = (
            numeric.notna().all(axis=1)
            & numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & numeric["volume"].ge(0)
            & numeric["high"].ge(numeric[["open", "close", "low"]].max(axis=1))
            & numeric["low"].le(numeric[["open", "close", "high"]].min(axis=1))
        )
        _check(
            checks,
            "prices.valid_ohlcv",
            bool(valid.all()),
            "critical",
            int((~valid).sum()),
        )
        ordered = prices.assign(_close=numeric["close"]).sort_values(
            ["symbol", "timestamp"]
        )
        returns = ordered.groupby("symbol", sort=False)["_close"].pct_change(
            fill_method=None
        )
        suspicious = returns.abs().gt(1.0)
        if suspicious.any():
            quarantine.append(
                _quarantine(
                    ordered.loc[suspicious],
                    "prices",
                    "absolute_daily_return_over_100pct",
                )
            )
        _check(
            checks,
            "prices.extreme_returns_quarantined",
            True,
            "warning",
            int(suspicious.sum()),
        )
    _not_future(checks, "prices.timestamp_not_future", prices.get("timestamp"), now)
    _provenance(checks, "prices", prices)

    _required(
        checks,
        "assets",
        assets,
        {
            "symbol",
            "effective_date",
            "security_type",
            "status",
            "tradable",
            "ingested_at_utc",
        },
    )
    if {"symbol", "effective_date"}.issubset(assets):
        _check(
            checks,
            "assets.unique_key",
            not assets.duplicated(["symbol", "effective_date"]).any(),
            "critical",
        )
    _not_future(
        checks, "assets.effective_date_not_future", assets.get("effective_date"), now
    )
    _provenance(checks, "assets", assets)

    _validate_sec(checks, quarantine, submissions, "submissions", "report_date", now)
    _validate_sec(checks, quarantine, fundamentals, "fundamentals", "end_date", now)

    _required(
        checks, "macro_raw", macro_raw, {"date", "series_id", "value", "retrieved_at"}
    )
    if {"date", "series_id"}.issubset(macro_raw):
        _check(
            checks,
            "macro_raw.unique_key",
            not macro_raw.duplicated(["date", "series_id"]).any(),
            "critical",
        )
    _not_future(checks, "macro_raw.observation_not_future", macro_raw.get("date"), now)
    _check(
        checks, "macro_raw.finite_values", _finite(macro_raw.get("value")), "critical"
    )
    _provenance(checks, "macro_raw", macro_raw, column="retrieved_at")

    return QualityResult(
        checks=pd.DataFrame(checks),
        quarantine=pd.concat(quarantine, ignore_index=True)
        if quarantine
        else pd.DataFrame(columns=["dataset", "reason"]),
    )


def _validate_sec(checks, quarantine, frame, name, period_column, now):
    _required(
        checks,
        name,
        frame,
        {
            "symbol",
            "filing_date",
            "accession_number",
            period_column,
            "ingested_at_utc",
            "earliest_tradable_date",
        },
    )
    key = ["symbol", "accession_number"]
    if name == "fundamentals":
        key += ["concept", "unit", "end_date"]
    if set(key).issubset(frame):
        duplicate = frame.duplicated(key, keep=False)
        _check(
            checks,
            f"{name}.unique_point_in_time_key",
            not duplicate.any(),
            "critical",
            int(duplicate.sum()),
        )
    if {"filing_date", period_column}.issubset(frame):
        filing = pd.to_datetime(frame["filing_date"], errors="coerce", utc=True)
        period = pd.to_datetime(frame[period_column], errors="coerce", utc=True)
        invalid = filing.notna() & period.notna() & filing.lt(period)
        if invalid.any():
            quarantine.append(
                _quarantine(
                    frame.loc[invalid], name, "filing_precedes_reporting_period"
                )
            )
        _check(
            checks,
            f"{name}.availability_order",
            not invalid.any(),
            "critical",
            int(invalid.sum()),
        )
    _not_future(checks, f"{name}.filing_not_future", frame.get("filing_date"), now)
    if {"filing_date", "earliest_tradable_date"}.issubset(frame):
        filing = pd.to_datetime(frame["filing_date"], errors="coerce", utc=True)
        tradable = pd.to_datetime(
            frame["earliest_tradable_date"], errors="coerce", utc=True
        )
        _check(
            checks,
            f"{name}.next_session_availability",
            bool((tradable > filing).all()),
            "critical",
        )
    _provenance(checks, name, frame)


def _required(checks, name, frame, columns):
    missing = sorted(columns - set(frame))
    _check(
        checks,
        f"{name}.required_columns",
        not missing,
        "critical",
        detail=", ".join(missing),
    )


def _not_future(checks, name, values, now):
    passed = (
        values is not None
        and not pd.to_datetime(values, errors="coerce", utc=True)
        .gt(now + pd.Timedelta(days=1))
        .any()
    )
    _check(checks, name, passed, "critical")


def _provenance(checks, name, frame, column="ingested_at_utc"):
    parsed = (
        pd.to_datetime(frame[column], errors="coerce", utc=True)
        if column in frame
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    passed = column in frame and parsed.notna().any()
    _check(checks, f"{name}.provenance", passed, "critical")
    _check(
        checks,
        f"{name}.provenance_coverage",
        bool(len(parsed) and parsed.notna().all()),
        "warning",
        int(parsed.isna().sum()),
        "Legacy rows may predate provenance capture.",
    )


def _finite(values) -> bool:
    if values is None:
        return False
    numeric = pd.to_numeric(values, errors="coerce")
    return bool(numeric.notna().all() and np.isfinite(numeric).all())


def _check(checks, name, passed, severity, affected_rows=0, detail=""):
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "affected_rows": affected_rows,
            "detail": detail,
        }
    )


def _quarantine(frame, dataset, reason):
    result = pd.DataFrame(index=frame.index)
    result["dataset"] = dataset
    result["reason"] = reason
    result["symbol"] = frame["symbol"].astype(str) if "symbol" in frame else pd.NA
    date_column = next(
        (column for column in ["date", "timestamp", "filing_date"] if column in frame),
        None,
    )
    result["observation_time"] = (
        frame[date_column].astype(str) if date_column else pd.NA
    )
    return result.reset_index(drop=True)


def _utc(value):
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
