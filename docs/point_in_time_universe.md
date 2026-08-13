# Point-in-time equity universe

The production universe is a prospective, effective-dated record. Each nightly run
stores the current Alpaca asset state, preserves previous snapshots, refreshes prices
for a deterministic broad candidate pool, and computes membership using information
available on or before each observation date.

## Eligibility

A symbol is eligible when it is an active, tradable US equity; has at least 252
observed trading days; closes at or above $5; and has 20-day trailing median dollar
volume of at least $5 million. SPY is retained as the benchmark but excluded from
membership. The candidate pool is not membership: its stable hash selection avoids
ranking historical candidates with future liquidity, while the dated trailing rules
make the final decision.

Sector, industry, and market capitalization inputs must themselves be effective-dated.
When a separately licensed metadata history is supplied, the builder attaches only
the latest record effective on that date and derives cross-sectional small, mid, and
large buckets. Current classifications must never be backfilled into historical dates.

## Run locally

```bash
python -m scripts.snapshot_alpaca_assets
python -m scripts.ingest_alpaca_prices
python -m scripts.build_point_in_time_universe --confirm-provider-permissions
python -m scripts.run_walk_forward_backtest \
  --universe-membership data/processed/universe_membership.parquet
```

Asset snapshots default to Alpaca's paper Trading API host because paper and live
credentials are not interchangeable. Live-account users can set
`ALPACA_TRADING_BASE_URL=https://api.alpaca.markets` explicitly.

The confirmation flag records that the operator checked the current provider account
and plan for private research use. It does not grant redistribution rights.

## Diagnostics and limitations

The builder emits daily observed/member coverage, entries, exits, membership turnover,
sector and size coverage, and flags for dates capable of stable deciles and sector
controls. Every row also carries its effective-dated or legacy membership basis and a
single deterministic exclusion reason. Diagnostics quantify asset-state coverage,
metadata coverage, legacy-period exposure, and exits associated with inactive or
untradable states. Its manifest records the rules and known limitations.

Alpaca's asset endpoint is a current security master, not a historical one. Consequently:

- membership is valid prospectively from the first stored asset snapshot;
- no future asset state is backfilled into earlier dates;
- delisted or inactive names already observed remain in price and asset history, but
  names delisted before snapshot collection cannot be reconstructed from Alpaca;
- pre-snapshot experiments using the legacy static symbol list retain survivorship bias;
- OTC availability depends on separate account permissions.

These constraints are documented rather than hidden or imputed. A licensed historical
security master can later be supplied through the same effective-dated schema.

## Data rights and storage

Provider-derived prices, asset snapshots, memberships, reports, and manifests remain
under ignored `data/` or `reports/` paths and are uploaded only to the repository's
private R2 bucket. GitHub Actions does not publish them as artifacts, and they must not
be committed or redistributed. Re-check the current provider agreement before changing
the use case, account plan, collaborators, or storage policy.

Provider references:

- Alpaca Trading API, Assets: https://docs.alpaca.markets/us/reference/getassets
- Alpaca Broker API FAQ, asset status and tradability:
  https://docs.alpaca.markets/us/docs/broker-api-faq
- Alpaca Market Data FAQ, permissions and OTC availability:
  https://docs.alpaca.markets/us/docs/market-data-faq
- Alpaca customer agreement:
  https://files.alpaca.markets/disclosures/library/AcctAppMarginAndCustAgmt.pdf
