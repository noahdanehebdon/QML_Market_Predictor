# Alpaca Paper Execution

This workflow validates an immutable trade intent against current Alpaca paper-account
state and can submit paper orders only after several explicit gates. It contains no live
endpoint and rejects any configured host other than
`https://paper-api.alpaca.markets`.

Alpaca paper trading has separate credentials from live trading. Create paper keys in
Alpaca and store them only in the local environment or GitHub environment secrets:

```dotenv
ALPACA_PAPER_API_KEY=...
ALPACA_PAPER_SECRET_KEY=...
MARKET_QML_ENABLE_PAPER_ORDERS=false
MARKET_QML_PAPER_KILL_SWITCH=active
```

Do not reuse the ingestion variables `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Never
commit `.env`, trade intents, execution reports, account data, positions, or order
journals. The examples below write under `reports/`, which Git ignores.

## Dry-run first

Dry-run is the default. It reads the account, positions, orders, assets, and market
calendar, performs every pre-trade check, and produces proposed day limit orders. It
does not call the order-submission endpoint.

```powershell
python -m scripts.execute_alpaca_paper `
  --intent reports/private/trade_intents/2026-07-23.json `
  --config configs/paper_execution.yaml `
  --output reports/private/execution/2026-07-23-dry-run.json
```

The output is created once and cannot overwrite an earlier report. It records approved,
skipped, or rejected decisions with machine-readable reasons but omits credentials,
account IDs, and raw broker responses.

## Explicit paper submission

Paper submission requires all three conditions:

1. pass `--submit-paper`;
2. set `MARKET_QML_ENABLE_PAPER_ORDERS=true`;
3. set `MARKET_QML_PAPER_KILL_SWITCH=inactive`.

The default environment therefore fails closed. To stop submissions immediately, set
the kill switch to `active` or remove either environment variable.

```powershell
$env:MARKET_QML_ENABLE_PAPER_ORDERS = "true"
$env:MARKET_QML_PAPER_KILL_SWITCH = "inactive"
python -m scripts.execute_alpaca_paper `
  --intent reports/private/trade_intents/2026-07-23.json `
  --config configs/paper_execution.yaml `
  --output reports/private/execution/2026-07-23-submit.json `
  --submit-paper
```

The engine rejects stale intents, unapproved lineage, non-trading dates, times outside
regular market hours, blocked/inactive accounts, shorts, non-equity or untradable
assets, fractional orders on ineligible assets, conflicting orders, insufficient
buying power, cash-reserve breaches, equity drift, and configured portfolio limits.
Sells are submitted before buys. Deterministic client order IDs make retries skip
previously submitted orders, including completed orders returned by Alpaca.

All orders are simple, regular-hours, day limit orders with `extended_hours=false`.
The default limit buffer is 25 basis points around the intent reference price. This is
a test execution rule, not a claim of execution quality or investment advice.

## Bounded stale-order cancellation

Passing `--cancel-stale-paper` together with `--submit-paper` requests cancellation of
open project-created orders older than `cancel_after_minutes` before any new orders are
submitted. The same environment and kill-switch gates apply. Orders without this
project's deterministic `mqml-` prefix are never canceled.

## Optional live paper integration test

Normal tests mock every broker request. The real paper smoke test is skipped unless all
of the following are supplied deliberately:

```powershell
$env:MARKET_QML_RUN_ALPACA_PAPER_INTEGRATION = "true"
$env:ALPACA_PAPER_API_KEY = "..."
$env:ALPACA_PAPER_SECRET_KEY = "..."
python -m pytest tests/test_alpaca_paper_integration.py -v
```

The test submits a minimal far-from-market AAPL paper limit order and immediately
requests cancellation. It never targets the live API.
