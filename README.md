# QML Market Predictor

[![Tests](https://github.com/noahdanehebdon/QML_Market_Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/noahdanehebdon/QML_Market_Predictor/actions/workflows/tests.yml)

A research platform for testing whether quantum machine-learning models add
useful signal beyond strong classical baselines when ranking US equities by
future performance relative to `SPY`.

## Research snapshot

| Question | Current evidence |
| --- | --- |
| Do QML models beat strong classical baselines? | No demonstrated quantum advantage; classical models lead classification overall and across documented regimes. |
| Is evaluation leakage-aware? | Point-in-time joins, train-only preprocessing, horizon purging, nested chronological validation, and an untouched 252-day locked test. |
| Are backtests execution-aware? | Non-overlapping return windows, configurable transaction costs, turnover, drawdown, and horizon-correct annualization. |
| What is the latest target result? | Development-only research favors 10-day classification and cross-sectional-ranking candidates; locked-test rows inspected: zero. |

The project is best read as an auditable empirical research system, including
negative and superseded experiments—not as a production trading strategy. Start
with the [current result status](docs/results_status.md), then review the
[research methodology](docs/methodology.md) and
[QML findings](docs/qml_experiments.md).
The decision rules for the final two-lane evaluation are documented in the
[definitive comparison protocol](docs/definitive_qml_comparison.md).
The complete documentation map is available at [docs/README.md](docs/README.md).

## Five-minute overview

The project turns daily market, macroeconomic, and SEC filing information into
point-in-time features, trains models on chronological walk-forward windows,
and evaluates their out-of-sample predictions after transaction costs. Its
purpose is comparative research—not live trading or a claim of quantum
advantage.

### Prediction targets

The primary classification target, `outperform_spy_5d`, is 1 when a stock's
next five-trading-day return exceeds `SPY` and 0 otherwise. Ranking models use
the continuous five-day excess return, `forward_excess_return_5d`. Labels are
purged at split boundaries to prevent overlapping future-return windows from
leaking into validation.

### Data sources

- Alpaca daily OHLCV bars for equities and `SPY`;
- Federal Reserve Board H.15 and G.17 releases;
- Bureau of Labor Statistics CPI and unemployment series; and
- SEC EDGAR submissions and XBRL company facts.

Users supply their own credentials and permissions. Provider-derived data,
predictions, reports, and fitted models are excluded from Git and GitHub Actions
artifacts. Automated runs place reproducible snapshots in a private Cloudflare
R2 bucket.

### Model suite

- Classical classification: logistic regression, random forest, and gradient
  boosting.
- Classical ranking/regression: ridge, elastic net, Huber, random forest, and
  standard or train-only-tuned gradient boosting.
- Quantum experiments: VQC, quantum-kernel SVM, and an eight-qubit QCNN, all
  currently executed with local simulators.

Every comparison uses chronological splits and train-only preprocessing. The
walk-forward runner records predictions, metrics, portfolio results, and a
traceable fitted-model bundle for each model and split.

### Project status

The ingestion, feature engineering, classical baseline, QML/QCNN simulation,
walk-forward evaluation, reporting, private data versioning, and scheduled
retraining workflows are implemented. Real quantum-hardware execution is not
yet implemented. For the quantum methodology and current findings, see
[docs/qml_experiments.md](docs/qml_experiments.md).

## Local Environment

Create and activate a Python 3.10 or newer virtual environment, then install the
project in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`;
the remaining commands are the same.

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

### Quick start

Run a credential-free demonstration using deterministic synthetic prices:

```powershell
python -m scripts.generate_demo_prices
python -m scripts.research_prediction_targets `
  --prices data/processed/demo_prices.parquet `
  --sector-column sector
```

The demo exercises multi-horizon label construction, point-in-time
cross-sectional and sector-relative targets, purging, locked-test isolation, and
nested development selection. Its outputs are synthetic research artifacts and
remain ignored by Git.

For the provider-backed workflow, preview the configured pipeline without making
API requests or writing data:

```powershell
ingest-prices --dry-run
ingest-macro --dry-run
ingest-sec --dry-run
build-features --dry-run
backtest --dry-run
```

To execute the complete local workflow after configuring credentials:

```powershell
ingest-prices
ingest-macro
ingest-sec
build-features
backtest
```

Run `train` before `report` when you want the standalone latest-date signal
report. All commands should be run from the repository root. Generated outputs
land under `data/`, `artifacts/`, and `reports/` and remain ignored by Git.

## GitHub Secrets

For GitHub Actions or other hosted workflows, add the same secret names in the repository settings under GitHub Secrets:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `BLS_API_KEY`
- `SEC_USER_AGENT`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_ACCOUNT_ID`
- `R2_BUCKET_NAME`

## Data ingestion

The ingestion layer normalizes:

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

## Features and labels

The feature pipeline implements:

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

Run the same lint and formatting checks enforced in CI before opening a pull
request:

```powershell
python -m ruff check .
python -m ruff format --check .
```

Apply Ruff's automatic fixes and formatter with:

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
See the [quality-gate guide](docs/quality_gates.md) for coverage, strict typing,
dependency auditing, and distribution-verification commands enforced in CI.

## Command-Line Interface

Installing the project creates seven workflow commands. Each reads
an embedded default workflow; `--config` can select a workspace-specific YAML
file. Relative configuration, data, report, and artifact paths resolve from
`--workspace-root`, then `MARKET_QML_WORKSPACE`, then the current directory:

```powershell
ingest-prices
ingest-macro
ingest-sec
build-features
train
backtest
report
```

For an installed wheel used outside a repository checkout:

```powershell
$env:MARKET_QML_WORKSPACE = "C:\research\qml-market"
build-features --dry-run
backtest --workspace-root "C:\research\qml-market" --dry-run
```

Repository-root development remains supported. Pass `--config configs/cli.yaml`
when you want to edit workflow steps locally. Runtime installation excludes
test and build tools; install `.[dev]` to run pytest, Ruff, and package builds.

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
market-aligned macro table, creates a SHA-256 manifest, and uploads only
`data/processed/` to a private Cloudflare R2 snapshot. Raw provider responses
are not uploaded. Configure the four bucket-scoped secrets `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID`, and `R2_BUCKET_NAME`. Generated data
remains excluded by `.gitignore`, and the workflow never commits data files to
the repository. A failed run emits a GitHub error annotation and identifies the
failed step in its log.

### Weekly model retraining

The `Weekly model retraining` GitHub Actions workflow runs Saturdays at 08:00
UTC, after the final scheduled weekday refresh. It downloads and checksum-
verifies the latest private R2 processed-data snapshot, rebuilds features
and walk-forward splits, and retrains the tuned gradient-boosting regressor by
default. The workflow uploads its generated reports to the private R2 bucket
under `reports/runs/<workflow-run-id>/`; it does not create GitHub Actions
artifacts or upload source datasets to GitHub.

Each weekly run also saves a fitted artifact bundle for every model and
walk-forward split. Prediction rows contain an `artifact_id` that maps to the
bundle manifest. Bundles include the fitted estimator, train-only preprocessor,
QML PCA transformer and explicit parameters when applicable, configuration
snapshot, Git commit, training timestamp, and train/validation date ranges.
These files follow the reports into private R2 and are never uploaded as
GitHub Actions artifacts or committed to Git.

The workflow can also be started from the repository's **Actions** tab. Manual
runs can select a different classical baseline and optionally enable VQC with
the `run_qml` input. Scheduled runs keep QML disabled so the slower quantum
workflow remains explicitly opt-in.

See [docs/data_versioning.md](docs/data_versioning.md) for the private-data
retention policy, snapshot identity, regeneration procedure, and Git safeguards.
See [docs/point_in_time_universe.md](docs/point_in_time_universe.md) for the
prospective liquid-equity universe, survivorship limits, and provider controls.
See [docs/feature_audit.md](docs/feature_audit.md) for leakage-safe feature quality,
drift, stability, redundancy, exposure, and family-ablation diagnostics.
See [docs/strong_baselines.md](docs/strong_baselines.md) for the XGBoost classifier,
date-grouped LambdaMART ranker, naive controls, and evidence threshold.
See [docs/ensembles.md](docs/ensembles.md) for chronological calibration, constrained
stacking, turnover/stability penalties, and constituent-removal sensitivity.

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

## Classical evaluation and backtesting

The evaluation layer provides chronological walk-forward splits, train-only preprocessing,
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

Portfolio evaluation uses non-overlapping five-trading-day returns, rebalances
every five prediction dates, and derives annualization as `252 / 5 = 50.4`
periods per year. Results produced before Issue #153 used 252 periods per year
and therefore overstated annualized volatility and Sharpe ratios; those earlier
portfolio risk figures are superseded.

Validation follows a locked-test protocol. The most recent 252 trading days are
excluded from routine split generation, with a five-day embargo between
development and the final test. Hyperparameters are selected across three
expanding chronological inner folds with label-horizon purging. Routine model
comparisons report date-block bootstrap intervals, paired sign-permutation
tests, Holm-adjusted p-values, observed effect sizes, and a 0.02 practical
ROC-AUC threshold. Do not inspect the locked period during feature or model
research. Final access must be deliberate and audited first:

```powershell
python -m scripts.log_locked_test_access `
  --reason "Milestone 6 final evaluation; configurations frozen"
```

The audit file authorizes evaluation; it does not copy market data into Git.

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

## Quantum machine-learning experiments

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

## Paper-trading preparation

The broker-independent trade-intent workflow converts current signals from an
explicitly promoted model into deterministic, long-only target holdings and a
dry-run JSON record. It validates freshness, finite values, portfolio constraints,
turnover, artifact lineage, and locked-test separation. It cannot contact a broker
or submit an order.

See [docs/trade_intents.md](docs/trade_intents.md) for the private promotion
manifest, required input schemas, conservative configuration, and reproducible
dry-run command.

The next guarded layer can validate those intents against an Alpaca paper account and,
only after three explicit safety gates, submit paper-only day limit orders. See
[docs/alpaca_paper_execution.md](docs/alpaca_paper_execution.md) for credential
separation, dry-run usage, risk checks, kill-switch behavior, and the opt-in paper
integration test.

Paper submissions can be backed by a private, restart-safe SQLite lifecycle journal.
The reconciliation workflow compares intended orders with Alpaca orders, fills, and
positions; reports slippage and residual exposure; records stale cancellations; and
enforces the five-trading-day rebalance cadence across separate runs. See
[docs/execution_reconciliation.md](docs/execution_reconciliation.md).

Before submission is enabled, staged validation archives broker-independent shadow
decisions, compares shadow, paper, and backtest evidence, and requires human approval
bound to an eligible report. See
[docs/shadow_paper_validation.md](docs/shadow_paper_validation.md). Live trading is
not supported.

## Limitations

- Results are backtests on a limited equity universe, not evidence of future
  profitability or a production trading strategy.
- Market regimes, transaction-cost assumptions, data revisions, survivorship,
  and the selected date range can materially change conclusions.
- The quantum models run on idealized local simulators. They do not measure
  queueing, shot noise, calibration drift, or errors from real hardware.
- Small QML samples and expensive circuit evaluation limit statistical power;
  no current result establishes quantum advantage over the classical baselines.
- The pipeline does not place orders, manage capital, or provide investment
  advice. Users remain responsible for data-provider terms and permissions.

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

## Data rights

Provider-derived data remain subject to their respective provider and exchange
terms and are not redistributed by this repository.
