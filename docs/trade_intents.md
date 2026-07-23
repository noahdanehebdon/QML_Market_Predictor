# Trade Intent Workflow

Trade intents separate research inference from any future broker integration. They are
deterministic dry-run records and cannot submit orders. Alpaca or another broker must not
be imported by this layer.

## Promote a model explicitly

Model selection happens during development, never while an intent is generated. After a
review, create a private promotion manifest alongside the private model artifacts:

```json
{
  "schema_version": 1,
  "artifact_id": "gradient_boosting-split-004",
  "model_name": "gradient_boosting",
  "model_sha256": "<64-character SHA-256>",
  "preprocessor_sha256": "<64-character SHA-256>",
  "feature_version": "canonical-v1",
  "promoted_at_utc": "2026-07-23T14:00:00+00:00",
  "selection_scope": "development_validation"
}
```

The manifest must name a reviewed artifact and the exact model and preprocessor hashes.
An artifact selected from the locked test is rejected. Promotion manifests and fitted
artifacts stay private and are not committed.

## Prepare current inputs

The signals CSV must contain one current cross-section with `date`, `symbol`,
`predicted_outperformance_probability`, `reference_price`, `model_name`,
`artifact_id`, `feature_version`, and optionally `is_benchmark`. The three lineage
columns must match the promotion manifest on every row. Generate these predictions
from the promoted model and its saved, train-fitted preprocessor. Do not include
realized forward returns.

The positions CSV contains `symbol` and non-negative `market_value`. A header-only file
represents an empty portfolio. Reference prices must also be present for every current
holding so exits can be estimated.

## Generate an immutable dry-run intent

```powershell
python -m scripts.generate_trade_intent `
  --config configs/trade_intent.yaml `
  --signals private/current_signals.csv `
  --positions private/current_positions.csv `
  --promotion private/promoted_model.json `
  --account-equity 100000 `
  --as-of 2026-07-23T14:00:00Z `
  --output private/trade_intents/2026-07-23.json
```

The explicit `--as-of` value is part of the reproducible decision. The command refuses
stale, missing, duplicate, nonfinite, short, over-invested, or constraint-violating
inputs. It uses exclusive file creation and will not overwrite an existing intent.

The resulting JSON records current and target holdings, proposed buys and sells,
reference prices, prediction scores, model/data lineage, policy settings, and a
deterministic run ID. `broker_submission_allowed` is always `false`.

The default policy is long-only, equal-weighted, retains 10% cash, caps each position at
20%, and records the same five-trading-day rebalance cadence used by the backtest. The
values are research safety settings, not investment advice.
