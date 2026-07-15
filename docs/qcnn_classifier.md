# Eight-Qubit QCNN Classifier

The QCNN classifier combines eight leakage-safe PCA components, RY angle
encoding, the `8 → 4 → 2` convolution/pooling architecture, and a two-qubit
expectation readout into a trainable binary classifier.

## Forward pass

For each row:

1. Map each PCA value with `theta = 2 * atan(x)` and apply `RY(theta)` to one of
   eight qubits.
2. Execute the 30-parameter QCNN convolution and pooling circuit.
3. Measure the exact Pauli-Z expectation of the final active qubits `q0` and
   `q4` in the statevector simulator.
4. Average those expectations and map the result to a positive-class score:

```text
score = (1 - mean(<Z_q0>, <Z_q4>)) / 2
```

The score is the mean probability that the two readout qubits would be measured
as one and is clipped to `[0, 1]`.

## Training

The classifier uses binary cross-entropy plus small L2 parameter regularization.
SPSA estimates the gradient with two perturbed circuit evaluations per iteration,
then the post-update objective is recorded on that iteration's mini-batch.

Default settings are 50 iterations, learning rate 0.1, perturbation 0.1, batch
size 32, L2 regularization 0.001, and random seed 42. Because each recorded loss
uses a newly sampled mini-batch, adjacent loss values are noisy and should not be
interpreted as a full-dataset monotonic curve.

## Running the classifier

```powershell
python -m scripts.train_qcnn_classifier
```

The command joins canonical forward-return metadata and saves:

- The fitted QCNN model and 30 learned parameters.
- Standard validation predictions.
- Per-iteration training loss.
- Full-training-sample metrics.
- Validation metrics.

## Initial bounded result

The initial issue-validation run used the reduced split-0 sample with 512
training rows, 256 validation rows, and 10 SPSA iterations.

- Training accuracy: `0.556641`
- Training log loss: `0.775683`
- Validation accuracy: `0.570312`
- Validation ROC-AUC: `0.594299`
- Validation log loss: `0.708286`
- Validation Brier score: `0.254411`

This confirms that the QCNN trains and produces correctly shaped outputs. It is
not yet a stable or well-tuned model: validation log loss is slightly worse than
the approximately `0.693` random-probability baseline, and the stochastic loss
curve fluctuates substantially. Issue 4.11 is specifically responsible for
testing initialization, learning rate, sample size, and gradient behavior before
the later model comparisons.
