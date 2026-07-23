"""Explicitly enabled smoke test for the real Alpaca paper order/cancel path."""

from __future__ import annotations

import os
import uuid

import pytest

from market_qml.execution.broker import AlpacaPaperBroker

RUN_INTEGRATION = os.getenv("MARKET_QML_RUN_ALPACA_PAPER_INTEGRATION") == "true"


@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="requires MARKET_QML_RUN_ALPACA_PAPER_INTEGRATION=true",
)
def test_submit_and_cancel_minimal_paper_limit_order():
    broker = AlpacaPaperBroker.from_environment()
    order = broker.submit_order(
        {
            "symbol": "AAPL",
            "qty": "0.001",
            "side": "buy",
            "type": "limit",
            "time_in_force": "day",
            "limit_price": "1.00",
            "extended_hours": False,
            "client_order_id": f"mqml-integration-{uuid.uuid4().hex[:20]}",
        }
    )
    broker.cancel_order(str(order["id"]))
