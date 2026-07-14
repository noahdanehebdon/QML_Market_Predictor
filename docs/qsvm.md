# Quantum Kernel Support Vector Machine

The QSVM baseline combines the simulator-backed quantum feature map with a
classical support vector machine. Quantum circuit state overlaps define the
kernel; the SVM performs the final supervised classification.

## Training flow

For a reduced QML train/validation split, the workflow:

1. Selects eight deterministic PCA-compressed feature columns.
2. Maps every row to an eight-qubit state using RY data re-uploading and ring-CZ
   entanglement.
3. Computes the square training fidelity matrix
   `K_train[i,j] = |<phi(x_i)|phi(x_j)>|^2`.
4. Fits `sklearn.svm.SVC` with `kernel="precomputed"`.
5. Computes the rectangular validation-to-training fidelity matrix.
6. Produces calibrated positive-class probabilities in the repository's
   standard prediction format.

The quantum portion is the state preparation and fidelity evaluation. The SVM
optimization and probability calibration are classical. All circuits currently
run on the exact `numpy_statevector` backend.

## Reduced-sample requirement

Kernel storage and computation grow quadratically with training rows. The
default reproducible sample therefore uses 512 training and 256 validation rows
from split 0. The CLI joins canonical forward-return labels by symbol and date so
predictions remain compatible with ranking and portfolio evaluation.

Run the baseline with:

```powershell
python -m scripts.train_qsvm_baseline
```

The command saves:

- The fitted QSVM model.
- Standard validation predictions.
- Train and validation kernel matrices.
- Kernel diagnostics including dimensions, range, mean, symmetry error,
  diagonal values, and support-vector count.

## Initial reduced-sample result

The initial split-0 smoke experiment used 512 balanced training rows, 256
balanced validation rows, 8 qubits, 2 feature-map repetitions, `C=1.0`, and
random seed 42.

- Training kernel shape: `512 x 512`
- Validation kernel shape: `256 x 512`
- Maximum training symmetry error: `0.0`
- Training diagonal range: `1.0` to `1.0`
- Support vectors: `453`
- Validation accuracy: `0.566406`
- Validation ROC-AUC: `0.582825`
- Validation log loss: `0.664289`

This is a functioning baseline, not evidence of quantum advantage. The large
support-vector count indicates a relatively complex decision boundary, and the
result must be compared on identical data and splits against VQC, QCNN, and
classical baselines in later issues.
