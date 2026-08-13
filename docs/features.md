# Feature Set

This document describes the modeling feature table produced by the project.
The canonical feature table is written to:

```text
data/features/feature_table.parquet
```

Labels are intentionally stored separately at:

```text
data/labels/forward_return_labels.parquet
```

Every feature-table row represents one symbol on one trading date and is keyed
by:

```text
symbol, date
```

## Build Order

Run the pipeline from the repository root after configuring `.env` and the YAML
files under `configs/`.

```powershell
python -m scripts.ingest_alpaca_prices
python -m scripts.pull_macro
python -m scripts.build_macro_daily --prices data/processed/prices.parquet
python -m scripts.build_sec_ticker_cik_lookup
python -m scripts.ingest_sec_submissions
python -m scripts.ingest_sec_company_facts
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

`scripts.ingest_sec_company_facts` skips symbols whose SEC companyfacts endpoint
returns `404`. This can happen for non-operating entities such as ETFs. Those
symbols remain in the market feature table with missing fundamental values.

## Price Columns

The canonical table keeps the original daily OHLCV-style market columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `trade_count`
- `vwap`

These columns preserve the base market state used to derive technical features.
Keeping them in the canonical table gives later modeling and analysis steps the
option to use raw price, liquidity, or execution context directly.

## Return Features

Backward-looking return features are computed by symbol:

- `return_1d`
- `return_5d`
- `return_10d`
- `return_20d`
- `return_60d`

Each return is computed as:

```text
close[t] / close[t - window] - 1
```

These features capture short-term and medium-term momentum. They use only past
and current prices.

Missing values are expected at the start of each symbol history when there is
not enough lookback history.

## Volatility Features

Rolling realized volatility features are computed from `return_1d`:

- `realized_vol_5d`
- `realized_vol_20d`
- `realized_vol_60d`

These are annualized rolling standard deviations through the current row. They
capture recent risk and turbulence over roughly weekly, monthly, and quarterly
windows.

Missing values are expected until each symbol has enough observations for the
requested window.

## Volume And Liquidity Features

Volume features include:

- `dollar_volume`
- `avg_volume_5d`
- `volume_shock_5d`
- `avg_dollar_volume_5d`
- `avg_volume_20d`
- `volume_shock_20d`
- `avg_dollar_volume_20d`
- `avg_volume_60d`
- `volume_shock_60d`
- `avg_dollar_volume_60d`

These features describe trading activity, liquidity, and unusual volume
conditions. Volume shock compares current volume with its rolling average.

Missing values are expected at the start of each rolling window.

## Benchmark-Relative Features

## Price-path and liquidity signals

The canonical builder adds trailing-only reversal, drawdown, downside-risk,
positive/zero-return share, range-volatility, volume-confirmation, Amihud
illiquidity, and rolling market-residual momentum signals. Selected signals also
receive same-date robust ranks. These quantities are stationary ratios or returns;
future rows never participate in their rolling windows.

Benchmark-relative features compare each symbol with the configured benchmark,
currently `SPY`:

- `excess_return_1d_vs_spy`
- `excess_return_5d_vs_spy`
- `excess_return_20d_vs_spy`
- `excess_return_60d_vs_spy`
- `rolling_corr_20d_vs_spy`
- `rolling_beta_20d_vs_spy`
- `relative_vol_20d_vs_spy`
- `relative_momentum_20d_vs_spy`
- `rolling_corr_60d_vs_spy`
- `rolling_beta_60d_vs_spy`
- `relative_vol_60d_vs_spy`
- `relative_momentum_60d_vs_spy`

These features provide market-relative context. The project predicts equity
outperformance, so relative behavior versus the benchmark is often more useful
than standalone price movement.

The `excess_return_*_vs_spy` columns are backward-looking feature columns. They
are distinct from the forward-looking label columns, which live only in the
label table.

## Cross-Sectional Features

For the core momentum, volatility, liquidity, and benchmark-relative signals,
the canonical table includes same-date percentile ranks, robust median/MAD
z-scores clipped to `[-5, 5]`, and—when point-in-time sector metadata is
available—within-sector percentile ranks. These transforms remove broad market
and sector level effects without using information from another date.

The canonical builder adds same-date percentile ranks for available momentum,
volatility, volume-shock, and benchmark-relative momentum columns. Each ranked
column receives an `_xs_rank` suffix and a companion `_missing` indicator.
Ranks use only securities observed on that date, so they add relative context
without looking forward in time.

## Macro Features

Macro level features include:

- `treasury_10y`
- `treasury_2y`
- `fed_funds`
- `cpi_all_items_sa`
- `unemployment_rate`
- `industrial_production`
- `yield_spread_10y_2y`

Macro change features include:

- `treasury_10y_change_5d`
- `treasury_2y_change_5d`
- `fed_funds_change_5d`
- `yield_spread_10y_2y_change_5d`
- `treasury_10y_change_20d`
- `treasury_2y_change_20d`
- `fed_funds_change_20d`
- `yield_spread_10y_2y_change_20d`
- `treasury_10y_change_60d`
- `treasury_2y_change_60d`
- `fed_funds_change_60d`
- `yield_spread_10y_2y_change_60d`
- `cpi_inflation_21d`
- `unemployment_rate_change_21d`
- `industrial_production_growth_21d`
- `cpi_inflation_63d`
- `unemployment_rate_change_63d`
- `industrial_production_growth_63d`
- `cpi_inflation_252d`
- `unemployment_rate_change_252d`
- `industrial_production_growth_252d`

These features capture the interest-rate environment, inflation, labor-market
conditions, and industrial activity.

Daily rate observations are aligned to trading dates with an as-of merge. The
macro daily builder also supports `--lag-daily-rates` for a more conservative
setup where same-day rate observations are not available until the following
calendar day.

Monthly macro observations are shifted forward by one month before daily
alignment. This conservative availability assumption avoids using same-month
macro values before they would have been known.

Missing values can occur near the start of the series, before conservative
monthly availability dates, or when a source has not yet published the latest
observation.

## Market Regime Labels

`scripts.build_market_regimes` writes one row per SPY trading date to:

```text
data/features/market_regimes.parquet
```

Regimes are analysis labels rather than forward targets. Each row uses only
information available through that date:

- `volatility_regime` compares annualized 20-day SPY realized volatility with
  the expanding median of prior valid 20-day volatility observations. The
  threshold is shifted by one date, so the current observation cannot alter its
  own high/low classification. Labels are `high_volatility`, `low_volatility`,
  or `normal_volatility` for an exact tie.
- `rate_regime` uses the 20-trading-date change in the average of the 2Y and 10Y
  Treasury yields. Positive, negative, and zero changes map to `rising_rates`,
  `falling_rates`, and `flat_rates`.
- `yield_curve_regime` uses the contemporaneously available 10Y-minus-2Y spread.
  Positive, negative, and zero spreads map to `normal_curve`, `inverted_curve`,
  and `flat_curve`. `--curve-flat-tolerance` can define a symmetric near-zero
  flat band.
- `yield_curve_trend` uses the trailing 20-date spread change and labels it
  `steepening_curve`, `flattening_curve`, or `unchanged_curve`.

The output retains the numeric volatility, threshold, yield level, yield
change, spread, and spread change beside the categorical labels for auditability.
Warm-up rows remain missing until their required history is available.

The regimes inherit the macro table's publication-aware alignment. Use
`build_macro_daily --lag-daily-rates` before building the canonical feature table
when same-day Treasury observations should be treated as unavailable until the
following calendar day.

## SEC Fundamental Features

Fundamental metadata columns include:

- `cik`
- `cik_padded`
- `fiscal_year`
- `fiscal_period`
- `filing_date`
- `form`
- `end_date`
- `accession_number`

Fundamental value and ratio features include:

- `fundamental_revenue`
- `fundamental_net_income`
- `fundamental_assets`
- `fundamental_liabilities`
- `fundamental_stockholders_equity`
- `revenue_growth`
- `net_income_margin`
- `liability_ratio`
- `equity_ratio`
- `filing_recency_days`

These features describe company scale, profitability, balance-sheet structure,
growth, and how stale the most recent known filing is.

The expanded point-in-time family includes operating income and cash flow, capital
expenditure, liquidity, debt, gross profit, R&D, stock compensation, and share
count. Derived signals emphasize same-fiscal-period growth, margins, free cash
flow, accruals, balance-sheet quality, and dilution rather than raw company size.

Fundamentals are merged by `symbol/date` using filing-date-aware as-of
alignment. A market row only receives a fundamental value after the corresponding
SEC filing date.

Missing values are expected for symbols without available SEC companyfacts, for
periods before the first known filing, and for filings that do not contain a
specific concept.

## SEC Filing Event Features

Recent filing metadata features include:

- `sec_last_filing_cik`
- `sec_last_filing_cik_padded`
- `sec_last_filing_form`
- `sec_last_filing_date`
- `sec_last_filing_report_date`
- `sec_last_filing_accession_number`
- `sec_last_filing_primary_document`
- `sec_days_since_last_filing`
- `sec_recent_filing_30d`

Form-specific event features include:

- `sec_last_10k_filing_date`
- `sec_last_10k_accession_number`
- `sec_days_since_last_10k`
- `sec_recent_10k_90d`
- `sec_last_filing_is_10k`
- `sec_last_10q_filing_date`
- `sec_last_10q_accession_number`
- `sec_days_since_last_10q`
- `sec_recent_10q_90d`
- `sec_last_filing_is_10q`
- `sec_last_8k_filing_date`
- `sec_last_8k_accession_number`
- `sec_days_since_last_8k`
- `sec_recent_8k_30d`
- `sec_last_filing_is_8k`

These features capture corporate information events and filing recency. They are
based on SEC submissions metadata and are merged with filing-date-aware as-of
alignment.

Missing values are expected before a symbol's first relevant SEC filing and for
symbols without applicable SEC submissions in the normalized source table.

## Labels

Labels are separate from the canonical feature table. The default label table
contains:

- `symbol`
- `date`
- `label_horizon_days`
- `forward_return_5d`
- `spy_forward_return_5d`
- `forward_excess_return_5d`
- `outperform_spy_5d`

The binary target is:

```text
outperform_spy_5d = 1 if forward_excess_return_5d > 0 else 0
```

The continuous ranking target is:

```text
forward_excess_return_5d
```

The label builder also emits `vol_normalized_excess_return_5d`, which divides
forward excess return by trailing realized excess-return volatility, and a
neutral-zone classification target. The CLI defaults to a 0.5% neutral band;
rows inside that band retain their regression target but receive no neutral-zone
class label.

The label table drops rows where the forward horizon is incomplete. With a
5-trading-day horizon, the last five trading rows for each symbol do not have
complete labels.

## Modeling Dataset Constructor

The modeling dataset constructor joins the canonical feature table and label
table by `symbol/date` only when preparing model inputs. It returns:

- `X`: numeric and boolean feature matrix
- `y`: target label vector
- `metadata`: row identifiers and non-feature context

By default, label columns are excluded from `X`. Date columns, filing forms,
accession numbers, and other non-numeric identifiers remain in metadata unless a
modeling step explicitly encodes them later.

## QML Angle Encoding

QML angle encoding consumes PCA component columns such as `pca_00` or grouped
columns such as `macro_pca_00`. The default encoder expects eight feature
columns for eight-qubit VQC and QCNN experiments.

PCA components are unbounded real values, so they are scaled with:

```text
angle = 2 * atan(component)
```

This maps every input smoothly into `[-pi, pi]`, which is a valid rotation angle
range. The transform is stateless, so it does not fit scaling parameters on train
or validation rows and does not introduce an additional leakage path.

## Leakage Precautions

The project includes dedicated tests for leakage-sensitive behavior:

- Return features use past prices only.
- Volatility features use past and current returns only.
- Monthly macro values are unavailable before conservative availability dates.
- Daily macro rate values can be lagged for stricter no-lookahead assumptions.
- SEC fundamentals are known only after `filing_date`.
- SEC filing events are known only after `filing_date`.
- Label columns are excluded from feature matrices.
- The canonical feature table rejects label-like columns.

These tests live in:

```text
tests/test_feature_leakage.py
```

## Missing Value Handling

Missing values are not globally imputed during feature construction. They are
preserved so modeling and preprocessing steps can make explicit decisions.

Common missing-value sources include:

- insufficient rolling-window history
- unavailable macro observations
- conservative macro availability lags
- symbols without SEC companyfacts
- SEC filings that do not report a specific concept
- dates before a symbol's first known SEC filing

The modeling dataset constructor can optionally drop rows with too many missing
features using `max_missing_feature_fraction`. Later preprocessing steps should
handle remaining missing values in a model-specific way.

The tuned boosted-regression selector runs within each outer training fold. Its
feature score combines training-only target association, sign consistency across
training dates, and usable coverage; outer-validation targets never participate.
