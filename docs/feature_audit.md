# Feature quality and predictive-stability audit

Run the audit only against the development walk-forward splits. The locked test period
is not used for feature selection.

```bash
python -m scripts.audit_feature_quality \
  --membership data/processed/universe_membership.parquet
```

Each run writes machine-readable Parquet tables plus a JSON manifest under the ignored
`reports/feature_audit/` directory:

Automated runs deterministically cap each date at 64 symbols for diagnostics. The
stable hash sample is independent of labels, preserves the full date range, and does
not alter model training or evaluation rows. This bounds quadratic redundancy work as
the feature set grows.

- `quality`: missingness, symbol coverage, staleness, and train-to-validation PSI;
- `predictive`: train and validation daily cross-sectional rank IC and classification
  association for every feature and fold;
- `stability`: median IC, sign agreement, and a conservative stable-evidence flag;
- `redundancy`: training-only feature pairs above the Spearman threshold;
- `ablations`: train-weighted, validation-scored feature-family results and whether
  dropping a family helps, hurts, or has no measurable effect;
- `exposures`: sector/size mean-exposure and missingness-dependence ranges.
- `decisions`: deterministic retain, transform, conditional, remove, or
  insufficient-coverage dispositions with the evidence behind each decision.

The decision table is a research triage artifact, not an automatic production feature
selector. It uses development folds only and exists to make additions and removals
reviewable before an expensive model run.

Feature inclusion must be justified by repeated validation evidence. The implementation
requires at least two folds, at least 60% train/validation sign agreement, and median
absolute validation rank IC of 0.01 for the diagnostic `stable_evidence` flag. That flag
is evidence for review, not an automatic guarantee of economic usefulness.

Timing assumptions are regression-tested: return features use only earlier closes,
Alpaca bars request corporate-action adjustment `all`, macro observations are aligned
by their published availability dates before feature construction, and SEC events are
merged using filing availability rather than report-period dates.
