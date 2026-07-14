# VQC Architecture and Optimizer Tuning

Issue 4.6 evaluates basic VQC architecture and training choices with a
reproducible grid search. The tuning workflow compares ansatz depth, learning
rate, and optimizer while holding the QML sample, split, qubit count, batch size,
regularization, perturbation, and random seed constant.

## Reproducing the tuning run

From an editable project installation, run:

```powershell
python -m scripts.tune_vqc `
  --ansatz-depths 1 2 `
  --learning-rates 0.05 0.1 `
  --optimizers spsa finite_difference `
  --max-iter 10 `
  --batch-size 32 `
  --split-id 0
```

The command saves:

- The ranked configuration table as Parquet.
- Per-iteration training loss for every configuration as Parquet.
- The selected configuration as JSON.
- A Markdown report containing the complete comparison.

Configurations are ranked by validation log loss, followed by validation Brier
score. Overfitting is flagged when validation log loss exceeds training log loss
by more than the configured threshold, which defaults to `0.05`.

## Optimizers

- `spsa` estimates the full gradient from two randomly perturbed circuit
  evaluations per update. Its cost does not grow with the parameter count.
- `finite_difference` uses centered finite differences for every circuit
  parameter. It is more computationally expensive but provides a useful
  deterministic comparison for these small simulated circuits.

## Initial split-0 result

The initial bounded experiment evaluated eight configurations using 8 qubits,
10 iterations, batch size 32, perturbation 0.1, L2 regularization 0.001, and
random seed 42.

The selected configuration was:

- Ansatz depth: `1`
- Learning rate: `0.1`
- Optimizer: `spsa`
- Training log loss: `0.694929`
- Validation log loss: `0.695313`
- Validation accuracy: `0.493056`
- Validation Brier score: `0.251068`
- Validation-minus-training loss gap: `0.000384`

None of the eight configurations crossed the overfitting threshold. The result
is close to random-classifier performance, so it identifies the best tested
starting configuration rather than evidence of predictive advantage. Deeper
circuits produced slightly higher classification accuracy in this short run but
worse calibrated validation log loss, which is the primary selection metric.

This result is intentionally treated as an initial tuning baseline. Later model
comparison issues should rerun the workflow with longer training and additional
walk-forward splits before drawing performance conclusions.
