# QML Market Predictor

[![Tests](https://github.com/noahdanehebdon/QML_Market_Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/noahdanehebdon/QML_Market_Predictor/actions/workflows/tests.yml)

An auditable research platform for testing whether quantum machine-learning
(QML) models add useful signal beyond strong classical baselines when ranking
US equities by future performance relative to `SPY`.

The project treats quantum advantage as a hypothesis to test, not a premise.
Current evidence does **not** demonstrate quantum advantage: classical models
lead the documented comparisons. The repository is designed to make that result
reproducible through point-in-time data alignment, chronological validation,
train-only preprocessing, cost-aware backtests, and traceable artifacts.

> This is a research system, not a production trading strategy or investment
> advice. Live trading is not supported.

## Research snapshot

| Question | Current evidence |
| --- | --- |
| Do QML models beat strong classical baselines? | No demonstrated quantum advantage. |
| Is evaluation leakage-aware? | Point-in-time joins, horizon purging, nested chronological validation, and an untouched 252-day locked test. |
| Are backtests execution-aware? | Non-overlapping return windows, configurable costs, turnover, drawdown, and horizon-correct annualization. |
| Does the project support quantum hardware? | Yes, for bounded inference of pre-trained VQC parameters through IBM Quantum Runtime; training and primary comparisons remain local and reproducible. |
| What is the latest research conclusion? | Development evidence favors classical models; locked-test rows inspected: zero. |

Start with the [current results](docs/results_status.md),
[methodology](docs/methodology.md), and
[QML experiment record](docs/qml_experiments.md). The
[documentation index](docs/README.md) maps the complete research surface.

## Five-minute overview

The pipeline converts daily market, macroeconomic, and SEC filing information
into point-in-time features, builds benchmark-relative forward targets, trains
models on expanding chronological windows, and evaluates out-of-sample
predictions after transaction costs.

### Prediction targets

- Classification: whether an equity outperforms `SPY` over a future horizon.
- Ranking/regression: the corresponding continuous forward excess return.
- Research variants: multiple horizons, volatility normalization, and a neutral
  zone for small excess returns.

Labels are purged at split boundaries so overlapping future-return windows do
not leak into validation.

### Data sources

- Alpaca daily OHLCV bars for equities and `SPY`.
- Federal Reserve Board H.15 and G.17 releases.
- Bureau of Labor Statistics CPI and unemployment series.
- SEC EDGAR submissions and XBRL company facts.

Users supply their own credentials and data permissions. Provider-derived data,
predictions, models, and reports are excluded from Git. See
[data sources and rights](docs/data_sources.md) for provider-specific guidance.

### Model suite

- Classical classification: logistic regression, random forest, gradient
  boosting, and XGBoost.
- Classical ranking/regression: ridge, elastic net, Huber, random forest,
  gradient boosting, and LambdaMART.
- Quantum research: VQC, quantum-kernel SVM, and an eight-qubit QCNN.

### Project status

Implemented capabilities include ingestion, point-in-time feature engineering,
classical and QML modeling, nested walk-forward evaluation, portfolio
backtesting, experiment reporting, private artifact storage, guarded paper
execution, and scheduled retraining. IBM Quantum integration supports explicit,
bounded hardware smoke runs with locally trained VQC parameters.

## Architecture

```text
Provider APIs
    -> normalized point-in-time data
    -> canonical features and forward labels
    -> chronological train / validation / locked-test splits
    -> classical and quantum model families
    -> aligned metrics and cost-aware portfolio simulation
    -> reports, model bundles, and guarded paper-trading intents
```

| Path | Purpose |
| --- | --- |
| `src/market_qml/` | Reusable ingestion, feature, model, QML, backtest, reporting, and execution code. |
| `scripts/` | Reproducible research and operational entry points. |
| `configs/` | Versioned experiment and workflow configuration. |
| `tests/` | Offline unit, integration, workflow, and repository-policy tests. |
| `docs/` | Methodology, evidence, subsystem design, and operational guides. |
| `data/` | Git-ignored local data areas; only directory placeholders are tracked. |

## Local Environment

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable ".[dev]"
Get-Command ingest-prices
```

The final command confirms that the virtual environment's console scripts are
available. If it cannot find `ingest-prices`, reactivate `.venv`; do not rely on
commands installed into an unrelated global Python environment. On macOS or
Linux, activate with `source .venv/bin/activate` and verify with
`command -v ingest-prices`.

Copy `.env.example` to `.env` and add only the credentials needed for your
workflow. Never commit `.env`.

### Quick start

The synthetic demo requires no provider credentials:

```powershell
python -m scripts.generate_demo_prices
python -m scripts.research_prediction_targets `
  --prices data/processed/demo_prices.parquet `
  --sector-column sector
```

Preview the configured provider-backed pipeline without API requests or writes:

```powershell
ingest-prices --dry-run
ingest-macro --dry-run
ingest-sec --dry-run
build-features --dry-run
train --dry-run
backtest --dry-run
report --dry-run
```

Run the core workflow after configuring credentials:

```powershell
ingest-prices
ingest-macro
ingest-sec
build-features
train
backtest
report
```

These seven commands are installed with the package. Each accepts `--config`
for an alternate workflow YAML and `--workspace-root` for paths outside the
checkout. Generated outputs remain under ignored `data/`, `artifacts/`, and
`reports/` directories.

## Reproducible research workflows

Run a small classical walk-forward backtest:

```powershell
python -m scripts.run_walk_forward_backtest `
  --models gradient_boosting_regressor tuned_gradient_boosting_regressor `
  --max-splits 1 `
  --transaction-cost-bps 5 `
  --disable-mlflow
```

Run a local VQC smoke backtest:

```powershell
python -m scripts.run_walk_forward_backtest `
  --models vqc `
  --max-splits 1 `
  --disable-mlflow
```

Run bounded IBM Quantum inference after installing the quantum dependencies and
setting `IBM_QUANTUM_API_KEY` and `IBM_QUANTUM_INSTANCE`:

```powershell
python -m pip install --editable ".[quantum]"
python -m scripts.run_ibm_vqc_smoke --backend least_busy
```

The hardware path does not train on a quantum processor. It submits a deliberately
small inference workload using fixed, locally trained parameters, enforces shot
and circuit limits, and requires explicit credentials. See the
[IBM Quantum backend guide](docs/ibm_quantum_backend.md).

The primary comparison and analysis entry points are:

```powershell
python -m scripts.compare_qml_models
python -m scripts.analyze_qml_regimes
python -m scripts.build_model_comparison_report
python -m scripts.build_backtest_charts
```

Results must be interpreted through the documented
[validation protocol](docs/methodology.md) and
[claim status](docs/results_status.md), not isolated headline metrics.

## Automation and secrets

GitHub Actions provides three distinct paths:

- `tests.yml` runs credential-free quality gates on pushes and pull requests.
- `nightly-data-refresh.yml` and `weekly-retraining.yml` use repository secrets
  and store snapshots and reports in a private Cloudflare R2 bucket.
- `ibm-quantum-smoke.yml` is manual-only and uses repository secrets for a
  bounded IBM Quantum hardware check.

Required secret names are documented in `.env.example` and the relevant
workflow guides. Secrets belong in local `.env` files or GitHub Actions secrets,
never in source, configuration, command output, or issue discussions.

## Testing

Run the local quality gates:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src scripts
python -m pytest
python -m scripts.check_git_data_policy
```

Build and validate the distribution separately:

```powershell
python -m build
python -m twine check dist/*
```

See [quality gates](docs/quality_gates.md) for the CI contract, coverage floor,
dependency audit, and wheel-install verification.

## Paper-trading boundary

The execution layer can convert promoted research signals into deterministic
trade intents, validate them against an Alpaca paper account, and reconcile
paper orders through a restart-safe journal. Submission requires explicit
safety gates and paper-only credentials. Live brokerage endpoints and live-order
commands are intentionally absent.

See [trade intents](docs/trade_intents.md),
[paper execution](docs/alpaca_paper_execution.md),
[reconciliation](docs/execution_reconciliation.md), and
[shadow validation](docs/shadow_paper_validation.md).

## Limitations

- Results are historical backtests on a limited equity universe and do not
  establish future profitability.
- Regimes, costs, revisions, survivorship, universe construction, and date
  selection can materially change conclusions.
- Most quantum evidence comes from idealized local simulation. The IBM path
  enables hardware inference but does not make existing simulator results into
  hardware evidence or establish quantum advantage.
- Small QML samples and expensive circuit evaluation limit statistical power.
- Provider data remain subject to provider and exchange terms.
- The project does not manage capital or provide financial advice.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing model behavior or
methodology. Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md); do not place credentials or private data in a public
issue.
