"""Private durable SQLite journal for paper execution state."""

from __future__ import annotations

import math
import os
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_ORDER_STATUSES = {
    "submitted",
    "new",
    "partially_filled",
    "filled",
    "canceled",
    "expired",
    "replaced",
    "rejected",
}
TERMINAL_ORDER_STATUSES = {"filled", "canceled", "expired", "replaced", "rejected"}
STATUS_RANK = {
    "submitted": 0,
    "new": 1,
    "partially_filled": 2,
    "filled": 3,
    "canceled": 3,
    "expired": 3,
    "replaced": 3,
    "rejected": 3,
}


class ExecutionJournal:
    """Persist sanitized intent, order, event, and position state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        _restrict_permissions(self.path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ExecutionJournal:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def register_intent(self, intent: dict[str, Any], recorded_at: datetime) -> None:
        """Record non-account intent lineage idempotently."""
        model = intent["model"]
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO runs (
                    run_id, signal_date, intent_as_of_utc, model_name,
                    artifact_id, feature_version, recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO NOTHING
                """,
                (
                    intent["run_id"],
                    intent["signal_date"],
                    intent["as_of_utc"],
                    model["model_name"],
                    model["artifact_id"],
                    model["feature_version"],
                    _timestamp(recorded_at),
                ),
            )

    def record_planned_order(
        self,
        *,
        run_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        requested_qty: float,
        requested_limit_price: float,
        recorded_at: datetime,
    ) -> None:
        """Record a deterministic order plan without overwriting broker state."""
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO orders (
                    client_order_id, run_id, symbol, side, requested_qty,
                    requested_limit_price, status, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'submitted', ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    requested_qty = excluded.requested_qty,
                    requested_limit_price = excluded.requested_limit_price
                """,
                (
                    client_order_id,
                    run_id,
                    symbol,
                    side,
                    requested_qty,
                    requested_limit_price,
                    _timestamp(recorded_at),
                ),
            )

    def apply_order_update(
        self,
        update: dict[str, Any],
        *,
        observed_at: datetime,
        source: str,
    ) -> bool:
        """Apply a sanitized broker state transition; ignore stale regressions."""
        client_order_id = str(update.get("client_order_id", ""))
        if not client_order_id:
            raise ValueError("Order update is missing client_order_id.")
        status = _normalize_status(update.get("status"))
        current = self._connection.execute(
            """
            SELECT status, filled_qty, broker_order_id
            FROM orders WHERE client_order_id = ?
            """,
            (client_order_id,),
        ).fetchone()
        if current is None:
            raise ValueError("Order update does not match a registered intent order.")
        current_status = str(current["status"])
        filled_qty = _nonnegative(update.get("filled_qty", 0), "filled_qty")
        if current_status in TERMINAL_ORDER_STATUSES and current_status != status:
            return False
        if STATUS_RANK[status] < STATUS_RANK[current_status]:
            return False
        broker_order_id = str(update.get("id", "")) or None
        if (
            status == current_status
            and filled_qty == float(current["filled_qty"])
            and (
                broker_order_id is None or broker_order_id == current["broker_order_id"]
            )
        ):
            return False
        average_fill_price = _optional_nonnegative(
            update.get("filled_avg_price"), "filled_avg_price"
        )
        submitted_at = _optional_timestamp(update.get("submitted_at"))
        rejection_reason = (
            str(update.get("rejection_reason", "")) or None
            if status == "rejected"
            else None
        )
        event_at = _optional_timestamp(update.get("updated_at")) or _timestamp(
            observed_at
        )
        with self._connection:
            self._connection.execute(
                """
                UPDATE orders SET
                    broker_order_id = COALESCE(?, broker_order_id),
                    status = ?, filled_qty = ?, average_fill_price = ?,
                    submitted_at_utc = COALESCE(?, submitted_at_utc),
                    updated_at_utc = ?, rejection_reason = ?
                WHERE client_order_id = ?
                """,
                (
                    broker_order_id,
                    status,
                    filled_qty,
                    average_fill_price,
                    submitted_at,
                    event_at,
                    rejection_reason,
                    client_order_id,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO order_events (
                    client_order_id, event_at_utc, status, filled_qty,
                    average_fill_price, source, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_order_id,
                    event_at,
                    status,
                    filled_qty,
                    average_fill_price,
                    source,
                    rejection_reason,
                ),
            )
        return True

    def record_cancel_request(
        self, client_order_id: str, requested_at: datetime
    ) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE orders SET cancel_requested_at_utc = ? WHERE client_order_id = ?",
                (_timestamp(requested_at), client_order_id),
            )

    def replace_position_snapshot(
        self,
        run_id: str,
        positions: list[dict[str, Any]],
        captured_at: datetime,
    ) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM position_snapshots WHERE run_id = ?",
                (run_id,),
            )
            for position in positions:
                self._connection.execute(
                    """
                    INSERT INTO position_snapshots (
                        run_id, captured_at_utc, symbol, quantity, market_value
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        _timestamp(captured_at),
                        str(position.get("symbol", "")).upper(),
                        _nonnegative(position.get("qty", 0), "position.qty"),
                        _nonnegative(
                            position.get("market_value", 0), "position.market_value"
                        ),
                    ),
                )

    def known_client_order_ids(self) -> set[str]:
        rows = self._connection.execute("SELECT client_order_id FROM orders").fetchall()
        return {str(row["client_order_id"]) for row in rows}

    def latest_submitted_signal_date(self) -> str | None:
        """Return the latest signal date with broker-confirmed order activity."""
        row = self._connection.execute(
            """
            SELECT MAX(r.signal_date) AS signal_date
            FROM runs r JOIN orders o ON o.run_id = r.run_id
            WHERE o.broker_order_id IS NOT NULL OR o.filled_qty > 0
            """
        ).fetchone()
        if row is None or row["signal_date"] is None:
            return None
        return str(row["signal_date"])

    def orders_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM orders WHERE run_id = ? ORDER BY symbol, side",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def events_for_order(self, client_order_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT event_at_utc, status, filled_qty, average_fill_price, source, reason
            FROM order_events WHERE client_order_id = ? ORDER BY id
            """,
            (client_order_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    signal_date TEXT NOT NULL,
                    intent_as_of_utc TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    broker_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
                    requested_qty REAL NOT NULL CHECK(requested_qty > 0),
                    requested_limit_price REAL NOT NULL CHECK(requested_limit_price > 0),
                    status TEXT NOT NULL,
                    filled_qty REAL NOT NULL DEFAULT 0,
                    average_fill_price REAL,
                    submitted_at_utc TEXT,
                    updated_at_utc TEXT NOT NULL,
                    rejection_reason TEXT,
                    cancel_requested_at_utc TEXT
                );
                CREATE TABLE IF NOT EXISTS order_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_order_id TEXT NOT NULL REFERENCES orders(client_order_id),
                    event_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    average_fill_price REAL,
                    source TEXT NOT NULL,
                    reason TEXT
                );
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    captured_at_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    market_value REAL NOT NULL,
                    UNIQUE(run_id, symbol)
                );
                """
            )


def _normalize_status(value: Any) -> str:
    aliases = {
        "accepted": "submitted",
        "pending_new": "submitted",
        "new": "new",
        "partial_fill": "partially_filled",
        "fill": "filled",
        "cancelled": "canceled",
    }
    status = aliases.get(str(value).lower(), str(value).lower())
    if status not in SUPPORTED_ORDER_STATUSES:
        raise ValueError(f"Unsupported order status: {status}")
    return status


def _nonnegative(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric.") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative.")
    return result


def _optional_nonnegative(value: Any, field: str) -> float | None:
    if value in {None, ""}:
        return None
    return _nonnegative(value, field)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Journal timestamps must include a timezone.")
    return value.astimezone(timezone.utc).isoformat()


def _optional_timestamp(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    if not isinstance(value, str):
        raise ValueError("Broker timestamp must be an ISO-8601 string.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _timestamp(parsed)


def _restrict_permissions(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        # Windows ACLs and some network filesystems may not support POSIX mode bits.
        pass
