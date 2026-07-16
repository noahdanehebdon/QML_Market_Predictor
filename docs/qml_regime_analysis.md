# QML performance by market regime

Issue #52 evaluates the aligned out-of-sample predictions from the controlled
QML comparison within the leakage-safe date labels produced by Issue #51.

Run the analysis with:

```powershell
python scripts/analyze_qml_regimes.py
```

The analysis joins by date and reports row count, chronological split coverage,
accuracy, ROC-AUC, log loss, Brier score, rank information coefficient, and
top-decile excess return for each model within volatility, rate, and yield-curve
regimes. Slices below 50 rows are retained in the parquet audit table but omitted
from the narrative comparison. ROC-AUC is unavailable for single-class slices.

Outputs are written under `reports/qml_regimes/` and include the joined
prediction rows, full metric table, QCNN-versus-each-model differences, and the
generated Markdown report.

## Full-run result

The full comparison contains 1,536 validation predictions per model across six
chronological splits. Classical models lead ROC-AUC in every adequately sized
regime slice; the analysis therefore finds no regime-specific classification
advantage for QCNN.

Interesting descriptive patterns include:

- QCNN is materially weaker in high volatility (`0.4933` ROC-AUC) than in low
  volatility (`0.5372`). RBF SVM leads the high-volatility slice at `0.5746`.
- QCNN is strongest during falling rates (`0.5409`) and nearly random during
  rising rates (`0.5197`). Gradient boosting leads falling rates at `0.5561`;
  RBF SVM leads rising rates at `0.5270`.
- QCNN performs better with an inverted curve (`0.5390`) than a normal curve
  (`0.5200`). Logistic regression leads the inverted-curve slice at `0.5464`,
  while gradient boosting leads the normal-curve slice at `0.5384`.
- Fixed QSVM has the strongest rank IC in the inverted-curve slice (`0.0554`),
  despite trailing logistic regression on classification ROC-AUC. This is a
  ranking-specific behavior, not broad QML outperformance.

The flat-rate slice contains only 21 rows and is excluded by the default
50-row reporting threshold. All regime findings remain descriptive: they should
be treated as hypotheses until they persist in additional chronological data.
