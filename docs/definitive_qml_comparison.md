# Definitive classical-versus-quantum comparison

The final comparison has two deliberately separate lanes:

- **Equal input:** every model receives the same outer rows and the same eight
  features selected without using outer-validation data.
- **Best available:** each model family uses its strongest configuration and
  validated inputs within the documented compute budget.

VQC, QCNN, and quantum-kernel SVM hyperparameters are selected only with
multi-fold chronological inner validation. The report records classification,
ranking, calibration, runtime, peak memory, split stability, and the corrected
non-overlapping portfolio backtest. Date-block bootstrap intervals, paired sign
permutation tests, and Holm correction prevent a single favorable split or one
of many comparisons from becoming a superiority claim.

Detailed prediction-derived tables remain in the private report store. The
public report contains aggregate conclusions only. A quantum-advantage claim
is disabled unless the locked-test manifest records deliberate access and the
same QML candidate clears practical thresholds for corrected ROC-AUC, rank IC,
and net excess portfolio return. Otherwise the report says plainly that no
advantage was demonstrated and retains the strongest classical system as the
default.

The scheduled full experiment builds both lanes with:

```powershell
python -m scripts.build_definitive_qml_comparison `
  --equal-dir reports/weekly_retraining/qml_tuned_full `
  --best-classical-dir reports/weekly_retraining/classical_full `
  --best-qml-dir reports/weekly_retraining/qml_tuned_full
```
