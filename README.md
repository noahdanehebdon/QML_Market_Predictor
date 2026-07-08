# QML_Market_Predictor

A market prediction platform comparing classical ML, standard QML models, and a QCNN architecture for regime-aware equity outperformance prediction.

## Project Status

Milestones 1 and 2 of 5 are complete.

Milestone 1 implements the core data-ingestion layer for market prices, macroeconomic data, SEC filings metadata, and SEC company fundamentals. Milestone 2 builds the canonical modeling feature table, forward-return labels, modeling dataset constructor, and leakage-focused tests. Generated raw, processed, feature, and label data files are stored locally and excluded from version control.

## Local Environment

Create a local `.env` file from `.env.example` and fill in your own API credentials:

```text
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
BLS_API_KEY=
SEC_USER_AGENT=
```

Do not commit `.env`. It is ignored by Git and should contain only local secrets.

The SEC requires a descriptive User-Agent for automated requests. Use a value that identifies the project and provides contact information, such as:

```text
SEC_USER_AGENT=QML Market Predictor contact@example.com
```

## GitHub Secrets

For GitHub Actions or other hosted workflows, add the same secret names in the repository settings under GitHub Secrets:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `BLS_API_KEY`
- `SEC_USER_AGENT`

## Milestone 1: Data Ingestion

Milestone 1 implements ingestion and normalization for:

- Alpaca daily OHLCV equity bars
- Federal Reserve Board Data Download Program macroeconomic series
- Bureau of Labor Statistics CPI and unemployment series
- SEC ticker-to-CIK lookup
- SEC company submissions metadata
- SEC XBRL companyfacts fundamentals

Key scripts:

```powershell
python -m scripts.ingest_alpaca_prices
python -m scripts.pull_macro
python -m scripts.build_macro_daily
python -m scripts.build_sec_ticker_cik_lookup
python -m scripts.ingest_sec_submissions
python -m scripts.ingest_sec_company_facts
```

Generated outputs are written under:

```text
data/raw/
data/processed/
```

These generated data files are excluded from version control.

## Milestone 2: Features and Labels

Milestone 2 implements:

- Technical price, return, volatility, volume, and liquidity features
- Benchmark-relative features versus `SPY`
- Daily market-aligned macro features
- SEC fundamental features aligned by filing date
- SEC filing event features aligned by filing date
- Canonical feature table construction
- Forward return label construction
- Modeling dataset construction
- Feature leakage tests

Key scripts:

```powershell
python -m scripts.build_macro_daily --prices data/processed/prices.parquet
python -m scripts.build_price_return_features
python -m scripts.build_price_volatility_features
python -m scripts.build_price_volume_features
python -m scripts.build_benchmark_relative_features
python -m scripts.build_macro_features
python -m scripts.build_fundamental_features
python -m scripts.build_filing_event_features
python -m scripts.build_feature_table
python -m scripts.build_forward_return_labels
```

Generated outputs are written under:

```text
data/features/
data/labels/
```

The canonical modeling feature table is:

```text
data/features/feature_table.parquet
```

The default forward-return label table is:

```text
data/labels/forward_return_labels.parquet
```

See [docs/features.md](docs/features.md) for the feature set, leakage precautions, and missing-value handling.

## Testing

Run the full test suite with:

```powershell
python -m pytest
```

The ingestion test suite includes mock API responses for Alpaca, BLS, Federal Reserve DDP, and SEC EDGAR so tests can run without live API calls or real API keys.

## Data Sources and Disclaimers

This project is intended for research, education, feature engineering, backtesting, and model-development purposes only. It does not provide financial, investment, trading, legal, accounting, or tax advice. Model outputs, forecasts, signals, metrics, and backtests should not be interpreted as recommendations to buy, sell, or hold any security or financial instrument. Past performance and backtested performance do not guarantee future results.

### Federal Reserve Board Data

Selected interest-rate and industrial-production series may be retrieved from public data releases of the Board of Governors of the Federal Reserve System, including H.15 Selected Interest Rates and G.17 Industrial Production and Capacity Utilization.

Federal Reserve Board data are used as inputs to derived macroeconomic features for local research and modeling workflows. This project is not sponsored, endorsed, certified, or approved by the Board of Governors of the Federal Reserve System. The Board of Governors of the Federal Reserve System does not endorse this project, its models, its outputs, or any investment-related interpretation derived from the data.

Source: Board of Governors of the Federal Reserve System.

### Bureau of Labor Statistics Data

CPI and unemployment data may be retrieved from the U.S. Bureau of Labor Statistics Public Data API.

BLS data are used as inputs to derived macroeconomic features for local research and modeling workflows. Users should record and cite the date on which BLS data were accessed or retrieved.

“BLS.gov cannot vouch for the data or analyses derived from these data after the data have been retrieved from BLS.gov.”

This project is not sponsored, endorsed, certified, or approved by the U.S. Bureau of Labor Statistics. The BLS.gov logo is not used in this project.

Source: U.S. Bureau of Labor Statistics.

### Alpaca Market Data

Equity market data may be retrieved from Alpaca’s Market Data API using the user’s own Alpaca API credentials, account permissions, market-data subscription, and applicable exchange data permissions.

This project does not redistribute Alpaca market data. Raw and processed Alpaca-derived market data files are generated locally by the user and are excluded from version control. Users are solely responsible for complying with Alpaca’s Terms & Conditions, Customer Agreement, applicable market-data subscription terms, exchange data agreements, and any restrictions associated with their Alpaca account or data plan.

Users should not reproduce, distribute, sell, publicly display, commercially exploit, or otherwise redistribute Alpaca market data unless they have the required permissions. Alpaca market data are used only as inputs to local research, backtesting, and model-development workflows.

This project is not sponsored, endorsed, certified, or approved by Alpaca, Alpaca Securities LLC, AlpacaDB, Inc., any exchange, or any third-party market-data provider. Nothing in this project should be interpreted as investment advice, a trading recommendation, or a representation by Alpaca or any third-party market-data provider.

### SEC EDGAR Data

SEC company ticker mappings, submissions metadata, and XBRL companyfacts may be retrieved from SEC EDGAR public APIs.

SEC EDGAR data are used as inputs to local research, feature engineering, and modeling workflows. Government-created content on SEC.gov and public EDGAR filing content are generally free to access and reuse. Users are responsible for following SEC access guidance, including efficient scripting, downloading only what is needed, respecting SEC fair-access limits, and declaring a descriptive User-Agent header through the `SEC_USER_AGENT` environment variable.

This project is not sponsored, endorsed, certified, or approved by the U.S. Securities and Exchange Commission. The SEC does not endorse this project, its models, its outputs, or any investment-related interpretation derived from EDGAR data.

Source: U.S. Securities and Exchange Commission EDGAR.

## Next Steps 

The best next confidence upgrades would be:
- Add GitHub Actions so tests run automatically on every PR.
- Add a small end-to-end smoke workflow from raw/sample data to final report.
- Add “golden output” tests for a tiny known dataset.
- Add report validation tests: expected models present, expected task types, expected metric columns.
- Add more leakage tests around labels, macro release timing, SEC filing timing, and split boundaries.
- Add documented commands for reproducing the current backtest outputs.
