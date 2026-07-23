# Paper Execution Reconciliation

The reconciliation layer makes Alpaca paper execution restart-safe and auditable. It
uses a private SQLite journal for sanitized lifecycle state and treats Alpaca REST
orders, open positions, and market values as the authority for what actually happened.

## Private journal

Provide a journal whenever paper submission is enabled:

```powershell
python -m scripts.execute_alpaca_paper `
  --intent reports/private/trade_intents/2026-07-23.json `
  --journal reports/private/execution/paper_journal.sqlite `
  --output reports/private/execution/2026-07-23-submit.json `
  --submit-paper
```

The command commits each deterministic client order ID to SQLite before its Alpaca
`POST /v2/orders` request. It records the broker order ID immediately after a successful
response. On restart, every journaled client ID is supplied to duplicate prevention,
including when Alpaca's recent-order response is temporarily unavailable.

The journal contains model/run lineage, symbols, sides, requested quantities and limit
prices, broker order IDs, timestamps, states, fills, rejection reasons, cancellation
requests, and sanitized position snapshots. It never stores credentials, account IDs,
or raw broker payloads. SQLite files and `reports/` are ignored by Git.

The journal also enforces the configured five-trading-day rebalance interval across
distinct intents. It obtains actual trading dates from Alpaca's calendar, including
holidays, rather than counting ordinary weekdays.

## Reconcile orders, fills, and positions

```powershell
python -m scripts.reconcile_alpaca_paper `
  --intent reports/private/trade_intents/2026-07-23.json `
  --journal reports/private/execution/paper_journal.sqlite `
  --json-output reports/private/reconciliation/2026-07-23T1500.json `
  --markdown-output reports/private/reconciliation/2026-07-23T1500.md
```

Run reconciliation after submission and again near the cancellation deadline or market
close. Outputs use exclusive creation so earlier observations are never overwritten.
Each report includes:

- intended, submitted, and filled quantities plus notional-weighted fill percentage;
- partial-fill percentage and terminal state;
- average fill price and adverse slippage versus the intent reference price;
- target versus actual position notionals and residual exposure;
- cancellations, rejection reasons, and recovery warnings.

Supported states are `submitted`, `new`, `partially_filled`, `filled`, `canceled`,
`expired`, `replaced`, and `rejected`. Terminal states cannot regress when delayed REST
or stream messages arrive out of order.

## Recovery behavior

Optional trade-update events can reduce reporting latency, but the websocket is not the
source of truth. A disconnect produces a warning and reconciliation continues from REST.
Temporary REST failures are retried. If authoritative state still cannot be established,
the cycle fails without changing prior durable state; the same journal can be reopened
and reconciled later.

Use `--cancel-stale-paper` only with the paper enable flag set and the kill switch
inactive. It requests cancellation only for stale project-prefixed orders and records
the request in the journal. A cancellation request is not treated as complete until a
later REST cycle reports a terminal state.

This remains simulated paper execution. It does not authorize live-money trading.
