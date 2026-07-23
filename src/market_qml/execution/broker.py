"""Broker protocol and a strictly paper-only Alpaca REST adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
PAPER_API_KEY_ENV = "ALPACA_PAPER_API_KEY"
PAPER_SECRET_KEY_ENV = "ALPACA_PAPER_SECRET_KEY"


class BrokerError(RuntimeError):
    """A sanitized broker communication failure."""


class PaperBroker(Protocol):
    """Minimum broker surface needed for guarded paper execution."""

    def get_account(self) -> dict[str, Any]: ...

    def list_positions(self) -> list[dict[str, Any]]: ...

    def list_orders(
        self, *, status: str = "open", after: str | None = None
    ) -> list[dict[str, Any]]: ...

    def get_asset(self, symbol: str) -> dict[str, Any]: ...

    def get_calendar(self, start: str, end: str) -> list[dict[str, Any]]: ...

    def submit_order(self, order: Mapping[str, Any]) -> dict[str, Any]: ...

    def cancel_order(self, order_id: str) -> None: ...


class AlpacaPaperBroker:
    """Small Alpaca Trading API client that rejects every non-paper host."""

    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str,
        base_url: str = ALPACA_PAPER_BASE_URL,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = validate_paper_base_url(base_url)
        if not api_key.strip() or not secret_key.strip():
            raise ValueError("Alpaca paper credentials must be non-empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(
        cls,
        *,
        base_url: str = ALPACA_PAPER_BASE_URL,
        session: requests.Session | None = None,
    ) -> AlpacaPaperBroker:
        """Create a client from paper-specific environment variables."""
        api_key = os.getenv(PAPER_API_KEY_ENV)
        secret_key = os.getenv(PAPER_SECRET_KEY_ENV)
        if not api_key or not secret_key:
            raise RuntimeError(
                f"Missing paper credentials. Set {PAPER_API_KEY_ENV} and "
                f"{PAPER_SECRET_KEY_ENV}."
            )
        return cls(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            session=session,
        )

    def get_account(self) -> dict[str, Any]:
        return self._json("GET", "/v2/account")

    def list_positions(self) -> list[dict[str, Any]]:
        return self._json_list("GET", "/v2/positions")

    def list_orders(
        self, *, status: str = "open", after: str | None = None
    ) -> list[dict[str, Any]]:
        if status not in {"open", "closed", "all"}:
            raise ValueError("Order status must be open, closed, or all.")
        params = {"status": status, "limit": 500}
        if after is not None:
            params["after"] = after
        return self._json_list("GET", "/v2/orders", params=params)

    def get_asset(self, symbol: str) -> dict[str, Any]:
        normalized = _safe_path_value(symbol, "symbol")
        return self._json("GET", f"/v2/assets/{normalized}")

    def get_calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        return self._json_list(
            "GET", "/v2/calendar", params={"start": start, "end": end}
        )

    def submit_order(self, order: Mapping[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/v2/orders", json=dict(order))

    def cancel_order(self, order_id: str) -> None:
        normalized = _safe_path_value(order_id, "order_id")
        self._request("DELETE", f"/v2/orders/{normalized}", expected_status={204})

    def _json_list(self, method: str, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        payload = self._request(method, path, expected_status={200}, **kwargs)
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise BrokerError(f"Alpaca paper {method} {path} returned invalid JSON.")
        return payload

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = self._request(method, path, expected_status={200}, **kwargs)
        if not isinstance(payload, dict):
            raise BrokerError(f"Alpaca paper {method} {path} returned invalid JSON.")
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: set[int],
        **kwargs: Any,
    ) -> Any:
        try:
            response = self._session.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers,
                timeout=self._timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as error:
            raise BrokerError(
                f"Alpaca paper {method} {path} failed before receiving a response."
            ) from error
        if response.status_code not in expected_status:
            request_id = response.headers.get("X-Request-ID", "unavailable")
            raise BrokerError(
                f"Alpaca paper {method} {path} failed with status "
                f"{response.status_code}; request_id={request_id}."
            )
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except requests.JSONDecodeError as error:
            raise BrokerError(
                f"Alpaca paper {method} {path} returned invalid JSON."
            ) from error


def validate_paper_base_url(base_url: str) -> str:
    """Return the canonical paper URL or reject the configuration."""
    candidate = base_url.strip().rstrip("/")
    parsed = urlparse(candidate)
    if (
        candidate != ALPACA_PAPER_BASE_URL
        or parsed.scheme != "https"
        or parsed.hostname != "paper-api.alpaca.markets"
        or parsed.port is not None
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"Paper execution allows only {ALPACA_PAPER_BASE_URL}; configured host was rejected."
        )
    return ALPACA_PAPER_BASE_URL


def _safe_path_value(value: str, field: str) -> str:
    normalized = value.strip().upper() if field == "symbol" else value.strip()
    if not normalized or any(character in normalized for character in "/?#"):
        raise ValueError(f"{field} contains unsafe path characters.")
    return normalized
