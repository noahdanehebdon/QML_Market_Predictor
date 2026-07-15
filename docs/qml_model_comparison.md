# Controlled QML model comparison

Issue #49 adds one leakage-safe decision experiment for the VQC, fixed QSVM,
training-only tuned QSVM, QCNN, linear SVM, and RBF SVM.

Run it with:

```powershell
python scripts/compare_qml_models.py
```

The default run uses all six chronological outer splits and exactly 256 balanced
training and 256 balanced validation rows per split. Every primary model receives
the same eight training-only classical-selected inputs and identical outer row keys. The
sample manifest stores a SHA-256 hash for those keys.

The tuned QSVM searches `C = {0.01, 0.1, 1, 10}`, feature-map repetitions
`{1, 2, 3}`, and interaction strengths `{0, 0.5, 1}`. Selection uses
only an inner chronological partition of the outer training window. The outer
validation period is evaluated once after selection and never influences it.

## Expanded-universe redesign

The current comparison rebuilds inputs from the 30-equity-plus-SPY universe.
For each outer split, it takes the winning feature-count budget from the
classical gradient-boosting tuner, ranks source features against continuous
excess return using outer-training rows only, removes highly correlated
duplicates, and assigns eight surviving standardized features to qubits.

Features are ordered so strongly related inputs are adjacent on the circuit's
ring. The redesigned kernel additionally tests neighbor interaction
re-uploading strengths `{0, 0.5, 1.0}` together with `C` and circuit
repetitions. All choices use an inner chronological partition of outer training.
The selected source feature, target correlation, qubit assignment, neighboring
feature, and neighboring correlation are saved in
`selected_feature_manifest.parquet`.

Outputs under `reports/qml_comparison/` include predictions, per-split metrics,
split-bootstrap confidence intervals, timing and peak traced memory, QSVM kernel
dimensions/similarity/support-vector counts, every tuning trial, selected
configuration, sampled-row hashes, and a Markdown decision report. Generated
reports remain local because market-derived artifacts are excluded from version
control.

## Expanded selected-feature result

The six-split expanded run used 256 balanced training and validation rows per
split. Confidence intervals are split-bootstrap intervals, so the overlap near
0.5 is important when interpreting small mean differences.

| Model | Accuracy | ROC-AUC | Log loss | Brier | IC | Top-decile excess return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QCNN | 0.5169 | 0.5314 | 0.8934 | 0.3135 | 0.0381 | 0.0049 |
| Linear SVM | 0.5091 | 0.5250 | 0.6926 | 0.2497 | 0.0167 | 0.0106 |
| RBF SVM | 0.5228 | 0.5223 | 0.6943 | 0.2505 | 0.0328 | 0.0056 |
| Fixed QSVM | 0.5150 | 0.5215 | 0.6943 | 0.2505 | 0.0079 | -0.0004 |
| Tuned QSVM | 0.5254 | 0.5095 | 0.6961 | 0.2513 | 0.0092 | -0.0015 |
| VQC | 0.4974 | 0.4973 | 0.6941 | 0.2505 | 0.0124 | 0.0007 |

QCNN has the highest mean ROC-AUC, but its 95% split-bootstrap interval
(`0.5005` to `0.5679`) nearly touches random and its calibration is poor. The
interaction-tuned QSVM selected nonzero interaction strength in five splits but
did not generalize: strong inner ROC-AUC values became a mean outer ROC-AUC of
`0.5095` and negative top-decile excess return. This is not evidence of quantum
advantage; it is evidence that the expanded comparison and circuit redesign are
working as an auditable negative experiment.

## Historical PCA result

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
