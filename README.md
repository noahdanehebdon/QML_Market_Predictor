# QML_Market_Predictor

A market prediction platform comparing classical ML, standard QML models, and a QCNN architecture for regime-aware equity outperformance prediction.

For a standalone explanation of the quantum experiments, circuit designs,
results, and limitations, see [docs/qml_experiments.md](docs/qml_experiments.md).

## Project Status

Milestones 1 through 3 of 5 are complete. Milestone 4 is in progress through
the variational quantum classifier (VQC) baseline.

Milestone 1 implements the core data-ingestion layer for market prices,
macroeconomic data, SEC filings metadata, and SEC company fundamentals.
Milestone 2 builds the canonical modeling feature table, forward-return labels,
modeling dataset constructor, and leakage-focused tests. Milestone 3 adds
leakage-safe preprocessing, classical classification and ranking baselines,
walk-forward evaluation, portfolio simulation, risk metrics, and experiment
tracking. Milestone 4 currently includes train-only PCA compression, reproducible
QML sampling, shared model interfaces, angle encoding, and a statevector-simulated
VQC. Generated data and model artifacts are stored locally and excluded from
version control.

## Local Environment

Create and activate a Python 3.10 or newer virtual environment, then install the
project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

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
python -m scripts.build_market_regimes
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

Run the same lint check enforced in CI before opening a pull request:

```powershell
python -m ruff check .
```

Apply Ruff's automatic fixes and optional formatter with:

```powershell
python -m ruff check --fix .
python -m ruff format .
```

Run the full test suite with:

```powershell
python -m pytest
```

Pytest is configured for the repository's `src` layout, so this command also
works directly from a checkout before an editable install.

## Command-Line Interface

Installing the project creates seven workflow commands. Each reads
`configs/cli.yaml`, whose steps can be edited to select modules and arguments:

```powershell
ingest-prices
ingest-macro
ingest-sec
build-features
train
backtest
report
```

Use a different pipeline configuration or preview a command without running it:

```powershell
backtest --config configs/cli.yaml --dry-run
train --config configs/cli.yaml
```

The ingestion commands require the credentials described in Local Environment.
Run commands from the repository root so configured relative paths resolve
against the project directories.

### Nightly data refresh

The `Nightly data refresh` GitHub Actions workflow runs at 23:00 UTC on weekdays
(after the US equity market close in both standard and daylight time) and can
also be started manually from the repository's **Actions** tab. Configure these
repository secrets before running it:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `BLS_API_KEY`
- `SEC_USER_AGENT` (for example, `QML Market Predictor you@example.com`)

The workflow refreshes Alpaca prices, BLS/Federal Reserve macro data, and SEC
ticker, submissions, and company-facts data. SEC requests are paced at no more
than five per second and retry temporary failures. The workflow builds the
market-aligned macro table and uploads only `data/processed/` as a seven-day
GitHub Actions artifact; raw provider responses are not uploaded. Generated
data remains excluded by `.gitignore`, and the workflow never commits data
files to the repository. A failed run emits a GitHub error annotation and
identifies the failed step in its log.

After training the default logistic-regression model, generate the latest
cross-sectional signal report with:

```powershell
report
```

This reads the latest canonical feature date, applies the saved train-fitted
preprocessor and model, and writes `reports/daily_signal.md` plus
`reports/daily_signal.csv`. The report ranks predicted benchmark outperformance
for research purposes only; it is not financial advice.

After running the controlled QML comparison and regime analysis, build the
unified classical/QML report with:

```powershell
python -m scripts.build_model_comparison_report
```

The command writes `reports/model_comparison.md` and companion CSV tables. It
compares classification, ranking, transaction-cost-aware portfolio, and
regime-specific results on aligned validation rows, identifies overall and
model-family leaders, and documents the limits of the comparison.

Generate the reproducible chart set after building the comparison report:

```powershell
python -m scripts.build_backtest_charts
```

The command saves cumulative return, drawdown, rolling Sharpe, model comparison,
and regime-specific figures under `reports/figures/`. It also creates
`reports/backtest_charts.md`, which embeds every figure for inclusion in the
final project report.

### Local results dashboard

Install the optional dashboard dependency and launch Streamlit from the
repository root:

```powershell
python -m pip install -e ".[dashboard]"
python -m streamlit run scripts/dashboard.py
```

The dashboard reads existing artifacts under `reports/` and shows the latest
signal report, classical model comparison, cumulative net returns, drawdowns,
top-ranked stocks, and QML experiment summaries. If `reports/daily_signal.csv`
does not exist, generate it first with `report`; the top-stock view falls back
to the latest saved backtest predictions in the meantime.

## Milestone 3: Classical Backtesting

Milestone 3 provides chronological walk-forward splits, train-only preprocessing,
classification and regression/ranking baselines, standard prediction tables,
classification and ranking metrics, portfolio simulation, transaction costs,
risk metrics, MLflow tracking, and task-aware baseline reports.

Run a small reproducible backtest with:

```powershell
python -m scripts.run_walk_forward_backtest --max-splits 1 --disable-mlflow
```

The default label workflow also produces a volatility-normalized continuous
target and a 0.5% neutral-zone classification target. Walk-forward splits use a
five-trading-day purge configured in `configs/backtest.yaml`, preventing labels
at the end of training from overlapping the validation period. The canonical
feature table includes same-date cross-sectional ranks, and the normalized
target can be evaluated with:

```powershell
python -m scripts.run_walk_forward_backtest `
  --models vol_normalized_gradient_boosting_regressor `
  --disable-mlflow
```

For the stronger classical selection baseline, run:

```powershell
python -m scripts.run_walk_forward_backtest `
  --models gradient_boosting_regressor tuned_gradient_boosting_regressor `
  --transaction-cost-bps 5 `
  --disable-mlflow
```

The tuned model ranks candidate feature counts and boosting parameters on the
chronological tail of each outer training window. It never selects against the
outer validation period, and writes every trial to
`selection_diagnostics.parquet`.

## Milestone 4: QML Experiments

The current VQC uses PCA-compressed inputs and RY angle encoding. A local exact
statevector simulator executes trainable RY layers separated by ring-CNOT
entanglers, and SPSA optimizes the circuit parameters. The walk-forward runner
saves standard predictions together with per-iteration training loss and
validation diagnostics.

Run a one-split VQC smoke backtest with:

```powershell
python -m scripts.run_walk_forward_backtest --models vqc --max-splits 1 --disable-mlflow
```

Tune VQC depth, learning rate, and optimizer with:

```powershell
python -m scripts.tune_vqc
```

See [docs/vqc_tuning.md](docs/vqc_tuning.md) for the tuning methodology,
generated artifacts, current best tested configuration, and limitations.

Build simulator-backed quantum kernel feature-map states with:

```powershell
python -m scripts.build_quantum_feature_map --split-id 0
```

See [docs/quantum_feature_map.md](docs/quantum_feature_map.md) for the circuit,
fidelity-kernel definition, simulator backend, and saved outputs.

Train the reduced-sample quantum kernel SVM with:

```powershell
python -m scripts.train_qsvm_baseline
```

See [docs/qsvm.md](docs/qsvm.md) for the hybrid quantum/classical training flow,
quadratic kernel-scaling constraint, diagnostics, and initial result.

The reusable eight-qubit QCNN convolution and pooling architecture is documented
in [docs/qcnn_blocks.md](docs/qcnn_blocks.md), including its `8 → 4 → 2`
active-qubit flow and complete parameter layout.

Train the first complete QCNN classifier with:

```powershell
python -m scripts.train_qcnn_classifier
```

See [docs/qcnn_classifier.md](docs/qcnn_classifier.md) for angle encoding,
two-qubit expectation readout, SPSA training, saved metrics, and initial results.

Run the reproducible QCNN stability grid with:

```powershell
python -m scripts.analyze_qcnn_stability
```

See [docs/qcnn_stability.md](docs/qcnn_stability.md) for gradient diagnostics,
failure thresholds, the selected stable configuration, and known limitations.

Run the controlled six-split QML/classical comparison with:

```powershell
python scripts/compare_qml_models.py
python scripts/analyze_qml_regimes.py
```

The comparison now rebuilds eight qubit inputs per outer split from the
classical tuner's training-only selected features on the expanded universe. It
also tunes neighbor interaction re-uploading exclusively inside training and
saves the source-feature/qubit audit manifest alongside the comparison report.

See [docs/qml_model_comparison.md](docs/qml_model_comparison.md) for the
training-only QSVM selection protocol, saved audit artifacts, confidence
intervals, and the initial model-selection conclusion.

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
