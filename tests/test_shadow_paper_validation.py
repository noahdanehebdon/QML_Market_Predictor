from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_qml.execution.validation import (
    build_validation_report,
    create_shadow_record,
    save_shadow_record,
    validate_paper_promotion_approval,
)
from scripts import execute_alpaca_paper

START = datetime(2026, 1, 1, 14, tzinfo=timezone.utc)


def _intent(index: int) -> dict[str, Any]:
    observed = START + timedelta(days=index * 2)
    return {
        "run_id": f"run-{index}",
        "signal_date": observed.date().isoformat(),
        "as_of_utc": observed.isoformat(),
        "account_equity": 100_000.0,
        "model": {
            "model_name": "gradient_boosting",
            "artifact_id": "model-1",
            "feature_version": "canonical-v1",
        },
        "portfolio": {"turnover": 0.2, "invested_weight": 0.8},
        "trades": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "notional": 20_000.0,
                "estimated_quantity": 100.0,
                "reference_price": 200.0,
            }
        ],
    }


def _shadow_records() -> list[dict[str, Any]]:
    return [
        create_shadow_record(
            _intent(index), observed_at=START + timedelta(days=index * 2)
        )
        for index in range(20)
    ]


def _reconciliation_reports() -> list[dict[str, Any]]:
    reports = []
    for index in range(20):
        signal_date = (START + timedelta(days=index * 3)).date().isoformat()
        reports.append(
            {
                "run_id": f"run-{index}",
                "signal_date": signal_date,
                "status": "reconciled",
                "warnings": [],
                "orders": [
                    {
                        "side": "buy",
                        "status": "filled",
                        "filled_qty": 100.0,
                        "reference_price": 200.0,
                        "requested_limit_price": 200.5,
                        "average_fill_price": 200.2,
                        "adverse_slippage_bps": 10.0,
                    }
                ],
                "positions": [{"target_notional": 20_000.0, "residual_notional": 0.0}],
            }
        )
    return reports


def _valuations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60),
            "equity": [100_000.0 + index * 100 for index in range(60)],
            "gross_exposure": [0.8] * 60,
        }
    )


def _report() -> dict[str, Any]:
    return build_validation_report(
        _shadow_records(),
        _reconciliation_reports(),
        _valuations(),
        backtest_summary={
            "transaction_cost_bps": 8.0,
            "average_turnover": 0.18,
            "cumulative_net_return": 0.08,
            "net_max_drawdown": -0.10,
        },
        generated_at=START + timedelta(days=60),
    )


def _approval(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "shadow_to_paper",
        "decision": "approved",
        "approved_by": "Human Reviewer",
        "approved_at_utc": "2026-03-02T14:00:00+00:00",
        "report_digest": report["report_digest"],
        "acknowledgements": {
            "paper_fills_are_simulated": True,
            "kill_switch_tested": True,
            "no_live_trading_authorized": True,
        },
    }


def test_shadow_record_has_no_submission_capability_and_is_immutable(tmp_path: Path):
    record = create_shadow_record(_intent(0), observed_at=START)
    path = tmp_path / "shadow.json"

    save_shadow_record(record, path)

    assert record["submission_capability"] == "none"
    assert record["paper_only"] is True
    with pytest.raises(FileExistsError):
        save_shadow_record(record, path)

    source = Path("scripts/generate_shadow_execution.py").read_text(encoding="utf-8")
    assert "AlpacaPaperBroker" not in source
    assert "submit_order" not in source


def test_report_compares_shadow_paper_and_backtest_and_passes_criteria():
    report = _report()
    metrics = report["metrics"]

    assert metrics["shadow"]["average_intended_turnover"] == pytest.approx(0.2)
    assert metrics["paper"]["average_executed_turnover"] == pytest.approx(0.2002)
    assert metrics["paper"]["average_quoted_slippage_bps"] == pytest.approx(25.0)
    assert metrics["paper"]["average_paper_fill_slippage_bps"] == pytest.approx(10.0)
    assert metrics["paper"]["average_tracking_error_fraction"] == 0.0
    assert metrics["valuation"]["cumulative_return"] == pytest.approx(0.059)
    assert (
        metrics["backtest_comparison"]["paper_fill_slippage_minus_assumption_bps"]
        == 2.0
    )
    assert report["promotion_gates"]["shadow_to_paper"]["eligible"] is True
    assert report["promotion_gates"]["paper_to_live_review"]["eligible"] is True
    assert report["live_trading_supported"] is False


def test_manual_approval_is_bound_to_eligible_evidence():
    report = _report()
    approval = _approval(report)

    validate_paper_promotion_approval(report, approval)

    tampered = deepcopy(report)
    tampered["metrics"]["shadow"]["observation_count"] = 0
    with pytest.raises(ValueError, match="bound"):
        validate_paper_promotion_approval(tampered, approval)

    incomplete = deepcopy(approval)
    incomplete["acknowledgements"]["kill_switch_tested"] = False
    with pytest.raises(ValueError, match="acknowledgements"):
        validate_paper_promotion_approval(report, incomplete)


def test_operational_failure_and_risk_breaches_block_promotion():
    reports = _reconciliation_reports()
    reports[0]["status"] = "attention_required"
    reports[0]["warnings"] = ["rejected order"]
    reports[0]["orders"][0]["status"] = "rejected"
    valuations = _valuations()
    valuations.loc[30, "equity"] = 70_000.0
    valuations.loc[30, "gross_exposure"] = 0.95

    report = build_validation_report(
        _shadow_records(),
        reports,
        valuations,
        backtest_summary={},
        generated_at=START + timedelta(days=60),
    )

    assert report["metrics"]["paper"]["rejected_order_count"] == 1
    assert report["metrics"]["operational"]["failure_count"] == 1
    assert report["promotion_gates"]["paper_to_live_review"]["eligible"] is False


def test_negative_valuation_exposure_is_rejected():
    valuations = _valuations()
    valuations.loc[0, "gross_exposure"] = -0.1

    with pytest.raises(ValueError, match="non-negative"):
        build_validation_report(
            [], [], valuations, backtest_summary={}, generated_at=START
        )


def test_paper_cli_requires_manual_promotion_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    intent = tmp_path / "intent.json"
    output = tmp_path / "blocked.json"
    intent.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_alpaca_paper",
            "--intent",
            str(intent),
            "--output",
            str(output),
            "--journal",
            str(tmp_path / "journal.sqlite3"),
            "--submit-paper",
        ],
    )

    with pytest.raises(SystemExit) as error:
        execute_alpaca_paper.main()

    assert error.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["error"]["code"] == "manual_promotion_required"
