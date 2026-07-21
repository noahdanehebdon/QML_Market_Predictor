# Research result status

This page is the canonical guide to which reported results should be treated as
current. The repository preserves older experiments for auditability, including
failed and superseded work.

## Current conclusions

- No experiment establishes quantum advantage over the classical baselines.
- Classical models lead classification ROC-AUC overall and in the documented
  market-regime slices.
- The latest target research selects 10-trading-day classification and
  cross-sectional-ranking candidates on nested development data only.
- The final 252-trading-day locked test has not been inspected for routine
  research or target selection.
- Portfolio annualization uses the return horizon and rebalance cadence rather
  than assuming 252 independent portfolio observations per year.

## Superseded results

Portfolio volatility and Sharpe values generated before issue #153 used an
incorrect 252-period annualization factor for five-day returns. Those values are
retained only as historical evidence and must not be used as current performance
claims. Classification and ranking metrics from those experiments remain useful
only within their documented sample and split limitations.

## Interpretation

All reported values are research backtests on a limited universe. They are not
live-trading results, are not independently audited, and are not evidence of
future returns. See [QML experiments](qml_experiments.md),
[model comparison](qml_model_comparison.md), and
[prediction targets](prediction_targets.md) for the underlying methodology.
