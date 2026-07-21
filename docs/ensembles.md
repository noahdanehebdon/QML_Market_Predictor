# Calibrated chronological ensembles

For outer fold `k`, the ensemble layer can use only out-of-sample predictions from
folds earlier than `k`. It cannot use the current outer validation fold or the locked
test. Binary scores are Platt-calibrated on that prior history. If history is too short,
the report records `equal_weight_insufficient_history` instead of fitting a calibrator
or hiding the limitation.

Three ensemble controls are emitted: calibrated simple average, within-date rank
average, and a nonnegative constrained stack whose weights sum to one. The stack's
chronological objective penalizes score turnover and dispersion in fold performance.
Regime-conditioned weights are permitted only for regimes meeting the configured
minimum sample size; smaller regimes remain pooled.

All ensemble predictions flow through the same classification, rank-IC, net-return,
turnover, and drawdown reports as single models. `ensemble_sensitivity.parquet` removes
each constituent in turn and records score correlation and absolute change. The simple
average is the equal-weight benchmark. A constrained ensemble that does not improve
over it and the strongest constituent remains in the artifacts as a negative result;
no report row is dropped for failing to improve.
