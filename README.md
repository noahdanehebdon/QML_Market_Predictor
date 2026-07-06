# QML_Market_Predictor

A market prediction platform comparing classical ML, standard QML models, and a QCNN architecture for regime-aware equity outperformance prediction.

## Project Status

Milestone 1 of 5 is complete.

Milestone 1 implements the core data-ingestion layer for market prices, macroeconomic data, SEC filings metadata, and SEC company fundamentals. Generated raw and processed data files are stored locally and excluded from version control.

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

## Testing

Run the full test suite with:

```powershell
python -m pytest
```

The ingestion test suite includes mock API responses for Alpaca, BLS, Federal Reserve DDP, and SEC EDGAR so tests can run without live API calls or real API keys.

## Data Sources and Usage Notes

This project pulls macroeconomic, market, filing, and fundamentals data from third-party and public data sources. Raw and processed data files are generated locally and are excluded from version control.

This repository is for research and model-development workflows. Nothing in this repository constitutes financial, investment, legal, or tax advice.

### Federal Reserve Board Data

Selected interest-rate and industrial-production series are retrieved from Federal Reserve Board public data releases through the Federal Reserve Board Data Download Program, including H.15 Selected Interest Rates and G.17 Industrial Production and Capacity Utilization.

Federal Reserve Board data are used as inputs to derived macroeconomic features for modeling. This project is not sponsored, endorsed, or certified by the Board of Governors of the Federal Reserve System. The Federal Reserve Board does not endorse this project, its models, its outputs, or any investment-related interpretation derived from the data.

Source: Board of Governors of the Federal Reserve System.

### Bureau of Labor Statistics Data

CPI and unemployment data are retrieved from the U.S. Bureau of Labor Statistics Public Data API.

BLS data are used as inputs to derived macroeconomic features for modeling. BLS.gov cannot vouch for the data or analyses derived from these data after the data have been retrieved from BLS.gov. This project is not sponsored, endorsed, or certified by the U.S. Bureau of Labor Statistics.

Source: U.S. Bureau of Labor Statistics.

### Alpaca Market Data

Equity market data may be retrieved from Alpaca’s Market Data API using the user’s own Alpaca API credentials.

This project does not redistribute Alpaca market data. Raw and processed Alpaca data files are generated locally and are excluded from version control. Users are responsible for complying with Alpaca’s Terms & Conditions, Customer Agreement, applicable market-data subscription terms, and any exchange data agreements that apply to their account or data plan.

Alpaca market data are used only as inputs to local research, backtesting, and model-development workflows. This project is not sponsored, endorsed, or certified by Alpaca.

### SEC EDGAR Data

SEC company ticker mappings, submissions metadata, and XBRL companyfacts are retrieved from SEC EDGAR public APIs.

SEC data are used as inputs to local research, feature engineering, and modeling workflows. This project is not sponsored, endorsed, or certified by the U.S. Securities and Exchange Commission. Users are responsible for following SEC fair access guidelines, including use of a descriptive `SEC_USER_AGENT`.

Source: U.S. Securities and Exchange Commission EDGAR.
