# Controlled QML model comparison

Issue #49 adds one leakage-safe decision experiment for the VQC, fixed QSVM,
training-only tuned QSVM, QCNN, linear SVM, and RBF SVM.

Run it with:

```powershell
python scripts/compare_qml_models.py
```

The default run uses all six chronological outer splits and exactly 128 balanced
training and 128 balanced validation rows per split. Every primary model receives
the same eight `broad_market` PCA components and identical outer row keys. The
sample manifest stores a SHA-256 hash for those keys.

The tuned QSVM searches `C = {0.01, 0.1, 1, 10}`, feature-map repetitions
`{1, 2, 3}`, and three predefined eight-component PCA selections. Selection uses
only an inner chronological partition of the outer training window. The outer
validation period is evaluated once after selection and never influences it.

Outputs under `reports/qml_comparison/` include predictions, per-split metrics,
split-bootstrap confidence intervals, timing and peak traced memory, QSVM kernel
dimensions/similarity/support-vector counts, every tuning trial, selected
configuration, sampled-row hashes, and a Markdown decision report. Generated
reports remain local because market-derived artifacts are excluded from version
control.

## Initial six-split result

| Model | Accuracy | ROC-AUC | Log loss | Brier | IC | Top-decile excess return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 0.6055 | 0.6383 | 0.6695 | 0.2387 | 0.0982 | 0.0167 |
| RBF SVM | 0.5729 | 0.6158 | 0.6723 | 0.2403 | 0.1139 | 0.0092 |
| Tuned QSVM | 0.5742 | 0.5841 | 0.6661 | 0.2373 | 0.0408 | -0.0041 |
| Fixed QSVM | 0.5573 | 0.5787 | 0.6840 | 0.2455 | -0.0375 | -0.0027 |
| VQC | 0.5312 | 0.5450 | 0.6903 | 0.2486 | 0.0406 | 0.0068 |
| QCNN | 0.4310 | 0.4480 | 0.9132 | 0.3301 | 0.0203 | 0.0127 |

The controlled grid improves QSVM mean ROC-AUC from 0.5787 to 0.5841, but the
linear control remains materially stronger at 0.6383. The tuned QSVM also has a
negative mean top-decile excess return. The decision is therefore to keep the
classical models as the current baseline and redesign the quantum kernel before
expanding QSVM tuning. These are small-sample research results, not evidence of
future trading performance.
