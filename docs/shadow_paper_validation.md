# Shadow and Paper Validation

This workflow collects evidence before paper-order submission is enabled. It does not
implement or authorize live trading. Paper submission remains disabled by default.

## Archive shadow observations

Generate the normal private trade intent, then record its proposed orders:

```bash
python -m scripts.generate_shadow_execution \
  --intent private/trade_intent.json \
  --output private/shadow/2026-07-23.json
```

The shadow command imports no broker client and has no order-submission path. Its
output declares `submission_capability: none` and is created exclusively, so an
existing observation cannot be silently overwritten.

## Build the daily evidence report

After reconciliation and account valuation are available, build a report:

```bash
python -m scripts.build_shadow_paper_validation \
  --shadow-record private/shadow/2026-07-23.json \
  --reconciliation-report private/reconciliation/2026-07-23.json \
  --valuations private/paper_valuations.csv \
  --backtest-summary private/backtest_summary.json \
  --output private/validation/2026-07-23.json
```

Repeat the record arguments for the complete observation window. The report tracks
intended/executed turnover, assumed/quoted/paper-fill slippage, target tracking error,
return, drawdown, gross exposure, stale signals, missed/canceled/rejected orders, and
operational failures. It also shows differences from the supplied backtest.

Defaults in `configs/validation.yaml` require 20 shadow observations across 28 days
and 20 paper rebalances across 56 days. Risk and operational thresholds must pass.
Eligibility only permits human review; promotion is never automatic.

## Approve shadow-to-paper manually

Copy `configs/paper_promotion_approval.example.json` privately. Enter the reviewer and
UTC timestamp, copy the exact eligible report digest, and affirm all acknowledgements.
A changed report invalidates the approval.

Paper submission also requires the journal, environment enablement, inactive kill
switch, and existing account/order safeguards:

```bash
python -m scripts.execute_alpaca_paper \
  --intent private/trade_intent.json \
  --validation-report private/validation/2026-07-23.json \
  --promotion-approval private/paper_promotion_approval.json \
  --journal private/execution.sqlite3 \
  --output private/paper_execution.json \
  --submit-paper
```

## Guardrails and rollback

- Keep `MARKET_QML_PAPER_KILL_SWITCH=active` except during an intentionally approved
  paper run. Activating it blocks new submissions and cancellation actions.
- Defaults block a run above 90% gross exposure or after a 2% daily account loss, as
  well as for stale signals, excess turnover, account drift, duplicates, and unsuitable
  assets.
- To roll back, activate the kill switch, remove `MARKET_QML_ENABLE_PAPER_ORDERS`,
  preserve and reconcile the journal/reports, inspect open paper orders, and cancel
  them manually if appropriate. Do not delete evidence.
- Resume only after investigation, a fresh passing report, and a new bound approval.

Paper fills are simulated and omit live-market effects including market impact, queue
position, some latency slippage, and information leakage. Paper results cannot
establish expected live performance.

## Review required before any future live proposal

A separate design review must cover permissions and credentials, legal and tax
obligations, capital/loss limits, incident ownership, monitoring, independent model
review, liquidity and impact tests, recovery drills, and an explicit rollback owner.
This repository intentionally has no live-order command or live promotion gate.
