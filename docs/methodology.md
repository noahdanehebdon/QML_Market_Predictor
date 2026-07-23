# Research Methodology

## Purpose and research question

This repository studies whether small quantum machine-learning (QML) models add
out-of-sample information to equity cross-sectional prediction after controlling for
strong classical baselines, leakage, turnover, and transaction costs.

The primary question is deliberately narrower than “can quantum computers predict the
market?” It is:

> On identical chronological validation rows and a controlled input budget, do the
> tested VQC, QSVM, or QCNN implementations improve discrimination, ranking, or a
> simple long-only portfolio relative to classical controls?

This is comparative research, not a claim of deployable alpha. Model selection uses
development data. A final 252-trading-day period is locked from routine research, and
superiority claims require a preregistered locked-test evaluation.

## Research universe and data

The seed universe in `configs/universe.yaml` contains 30 diversified US large-cap
equities plus SPY as the benchmark. Research begins on 2020-01-01. This compact seed
universe makes the full experiment reproducible, but it creates survivorship and
selection limitations.

The repository also supports a larger point-in-time universe constructed from archived
Alpaca asset snapshots. Eligibility on each date requires the configured exchange,
price, liquidity, history, breadth, and sector rules. Membership is never backfilled
before the first observed asset snapshot. Periods before reliable snapshots therefore
remain explicitly survivorship-biased.

The data sources are:

- Alpaca daily adjusted OHLCV, VWAP, and trade-count data for equities and SPY;
- BLS and Federal Reserve macro series, aligned by publication or availability date;
- SEC submissions and company facts for filing events and fundamental observations;
- archived Alpaca asset metadata for point-in-time tradability and universe history.

Raw and derived provider data remain private and are excluded from version control.
Version manifests record file hashes, schema, row/date coverage, configuration, and
lineage without publishing licensed market data.

## Prediction target

For symbol `i` on trading date `t`, the main continuous outcome is five-trading-day
excess return over SPY:

```text
r_i(t, t+5) - r_SPY(t, t+5)
```

The corresponding binary label is one when that excess return is positive and zero
otherwise. The continuous value is retained for rank-IC and portfolio evaluation.

The target protocol also evaluates 5-, 10-, 20-, and 60-day horizons, neutral-zone
labels, volatility-normalized excess returns, cross-sectional ranks, and
sector-relative variants when point-in-time sector data are available. Target selection
is deterministic and development-only. Every candidate uses a purge at least as long
as its forward horizon. See [Prediction target research](prediction_targets.md).

## Feature construction and information timing

Each observation is keyed by `(symbol, date)`. Features and labels are stored in
separate tables. Feature families include:

- lagged returns, realized volatility, volume, dollar volume, and liquidity shocks;
- benchmark-relative momentum, beta, correlation, and relative volatility;
- macro levels and changes aligned to when the value was actually available;
- SEC fundamental values selected by filing/acceptance availability;
- filing-event recency and cadence;
- same-date cross-sectional ranks computed without using future dates;
- volatility, rate, and yield-curve descriptors used for regime analysis.

Rolling calculations operate within symbol using current-or-earlier observations.
Macro releases and SEC facts are not retrospectively assigned before they were public.
Early-window missing values are expected and handled inside training-only
preprocessing. The feature audit checks missingness, finite values, duplicate keys,
constant columns, distribution shift, and suspicious feature-label relationships.

## Chronological experimental design

Random train/test splitting is inappropriate for this question. The standard
walk-forward configuration uses:

| Component | Default |
| --- | ---: |
| Outer training window | 756 trading days |
| Outer validation window | 126 trading days |
| Locked final period | 252 trading days |
| Label purge | 5 trading days |
| Embargo | 5 trading days |
| Inner chronological folds | 3 |

For every outer split:

1. Training dates precede validation dates.
2. Training labels whose five-day outcomes overlap validation are purged.
3. An embargo separates development research from the locked period.
4. Imputation, scaling, feature selection, PCA, calibration, and hyperparameter
   selection fit on outer-training data only.
5. Hyperparameters use inner chronological folds; outer-validation labels are not a
   tuning resource.
6. The fitted pipeline predicts each outer-validation row once and saves row keys,
   model lineage, and artifacts.

The definitive QML comparison has two declared lanes. The **equal-input lane** gives
models the same outer rows and eight train-selected inputs. The **best-available lane**
allows each family its best development-validated representation and bounded compute
budget. Conclusions distinguish these questions rather than mixing their results.

## Preprocessing and dimensionality control

Numeric missing values are replaced using statistics learned from the training window;
non-finite values are rejected or normalized before estimator fitting. Scaling is fit
on training rows only.

Classical models may use a selected wider feature set. Quantum experiments use eight
inputs because the simulated circuits contain eight qubits. Depending on the declared
comparison lane, these are either training-only PCA components or eight training-ranked,
de-correlated source features. Each scalar is bounded for angle encoding by:

```text
angle = 2 * atan(x)
```

This compression is part of the tested method and a material limitation. A result from
eight compressed inputs does not establish how another encoding would behave.

## Classical baselines

The project treats a quantum comparison as meaningful only when it includes credible
classical alternatives:

- majority, prevalence, and simple market controls;
- logistic regression;
- linear and radial-basis-function SVMs;
- random forest and histogram/gradient-boosting classifiers;
- ridge, elastic net, Huber, random-forest, and gradient-boosting regressors;
- tuned gradient boosting and a learning-to-rank baseline;
- constrained probability/rank ensembles selected on chronological data.

Baseline hyperparameters, input counts, preprocessing, and selection metrics are saved
with the run. The strongest baseline is selected from development evidence, not from a
single favorable validation or portfolio statistic.

## Quantum models

All reported quantum models currently run on an exact local NumPy statevector
simulator. No result in this repository is evidence from physical quantum hardware.

### Variational quantum classifier

The VQC applies one `RY` input rotation per qubit, followed by trainable `RY` rotations
and a ring of CNOT entanglers. The output is the probability that the readout qubit is
one. Binary cross-entropy with L2 regularization is optimized primarily by seeded SPSA;
bounded depth, learning-rate, and optimizer choices use training-only validation.

### Quantum-kernel SVM

The QSVM defines similarity as squared state fidelity,
`|<phi(x)|phi(z)>|^2`, and passes the precomputed Gram matrix to a classical SVM.
Feature-map repetitions, interaction scale, and SVM `C` use inner chronological folds.
Kernel dimensions, similarity summaries, support-vector counts, runtime, and the
selected trial are persisted.

### Quantum convolutional neural network

The QCNN uses eight-qubit angle encoding followed by convolution and pooling blocks
that reduce active qubits from `8 → 4 → 2`. Two Pauli-Z expectations form the class
probability. Training uses binary cross-entropy, L2 regularization, mini-batches, and
SPSA. Stability experiments vary seeds and optimization settings and report loss and
gradient variability rather than selecting only the best run.

## Evaluation metrics and statistical comparison

Classification is reported with accuracy, balanced accuracy where applicable,
ROC-AUC, log loss, and Brier score. Because the economic use is cross-sectional,
evaluation also includes Spearman rank information coefficient, top-minus-bottom
spread, turnover, and portfolio results.

Models are compared on identical row keys. The definitive protocol uses block
bootstrap confidence intervals to respect temporal dependence and Holm correction for
multiple pairwise comparisons. Default practical-effect thresholds are 0.02 ROC-AUC
and 0.01 rank IC. Statistical significance without a practically meaningful effect is
not treated as superiority, and a development result is not called final without the
locked test.

Runtime, peak memory, circuit evaluations, kernel size, and parameter count accompany
predictive metrics so any gain can be judged against computational cost.

## Portfolio simulation

Scores form a simple cross-sectional long-only portfolio. The canonical backtest
rebalances every five trading days to match the outcome horizon and uses non-overlapping
return windows. It starts with 100,000 units of capital, applies a default 10-basis-point
transaction cost to turnover, and does not use shorting or leverage.

Outputs include gross/net return, annualized volatility and Sharpe using
`252 / 5 = 50.4` periods per year, maximum drawdown, average turnover, exposure, and
benchmark comparison. This remains stylized: it does not fully model bid-ask dynamics,
market impact, queue position, capacity, taxes, or all corporate actions.

## Regime analysis

Market regimes use contemporaneously available information:

- 20-day SPY realized volatility for low/high volatility;
- the 20-trading-day change in average two- and ten-year Treasury yields for
  falling/rising rates;
- the ten-year-minus-two-year spread for normal/inverted yield curves.

Regime metrics use already-produced out-of-sample predictions; regimes are not another
tuning opportunity. Small slices remain in audit artifacts but are excluded from
headlines. These analyses are descriptive and subject to multiplicity.

## Reproducibility and audit trail

Configuration is versioned under `configs/`. Runs preserve split definitions, exact
row manifests and hashes, random seeds, artifacts, predictions, metrics, timing, and
data lineage. The final-period access logger records any opening of the locked test.
Private data and credentials are not committed.

The recommended audit order is:

1. verify data and point-in-time manifests;
2. inspect feature and leakage diagnostics;
3. confirm chronological splits, purge, embargo, and row equality;
4. review training-only model selection;
5. compare predictive, calibration, ranking, portfolio, and resource metrics together;
6. inspect uncertainty and regime sensitivity;
7. check [current result status](results_status.md) before quoting any number.

## Interpretation rules and limitations

The project does not demonstrate quantum advantage. Current controlled evidence shows
classical models leading classification overall and across documented regimes. Some QML
models produce isolated ranking or portfolio leaders, but inconsistencies between
discrimination, calibration, ranking, and backtest metrics make those insufficient for
a superiority claim.

The main limitations are:

- a small, partly survivorship-biased research universe and limited history;
- bounded samples, six outer comparison splits, and limited statistical power;
- eight-qubit simulation rather than noisy, shot-based physical hardware;
- sensitivity to feature compression, optimizer budget, and initialization;
- model and target multiplicity despite correction and locked-test controls;
- macro revisions and incomplete historical security/sector metadata;
- stylized transaction costs and execution assumptions;
- no causal interpretation of predictive associations;
- simulated paper execution does not validate live performance.

This work supports claims about a reproducible leakage-aware comparison framework and
an informative negative QML result. It does not support claims of future profitability,
production readiness, or a general quantum advantage in finance.

## Related technical documentation

- [Feature definitions and timing](features.md)
- [Point-in-time universe](point_in_time_universe.md)
- [Definitive QML comparison protocol](definitive_qml_comparison.md)
- [QML experiment details and findings](qml_experiments.md)
- [Regime analysis](qml_regime_analysis.md)
- [Strong classical baselines](strong_baselines.md)
- [Current result status](results_status.md)
