# QCNN Training Stability

The QCNN stability workflow compares initialization scale, learning rate, and
training-sample size while recording gradient, parameter, loss, and
generalization behavior. It distinguishes numerical optimization stability from
predictive performance: a stable circuit can still be a poor classifier.

## Diagnostics

Every SPSA iteration records:

- Post-update mini-batch loss.
- SPSA gradient norm.
- Applied parameter-step norm.
- Complete circuit-parameter norm.
- Mini-batch row count.

Every configuration also records full training and validation log loss,
accuracy, Brier score, and the validation-minus-training loss gap.

The current diagnostic thresholds flag:

- `vanishing_gradient`: median gradient norm below `1e-5`.
- `exploding_gradient`: maximum gradient norm above `5.0`.
- `unstable_loss`: standard deviation of adjacent mini-batch loss changes above
  `0.2`.
- `parameter_growth`: parameter norm above `10.0`.
- `overfitting`: validation log loss more than `0.05` above training log loss.
- `non_finite`: any non-finite loss, gradient, or parameter value.

Stable configurations are ranked before unstable configurations, then by
validation log loss and loss volatility.

## Reproducing the bounded study

```powershell
python -m scripts.analyze_qcnn_stability `
  --initialization-scales 0.01 0.1 `
  --learning-rates 0.02 0.05 `
  --train-sample-sizes 128 512 `
  --max-iter 10
```

The workflow saves the ranked configuration table, complete optimization
history, selected configuration JSON, and a Markdown report.

## Selected stable configuration

The initial split-0 study evaluated eight configurations with batch size 32,
perturbation 0.1, L2 regularization 0.001, and random seed 42.

- Initialization scale: `0.1`
- Learning rate: `0.05`
- Training rows: `128`
- Iterations: `10`
- Median gradient norm: `1.432923`
- Maximum gradient norm: `2.784436`
- Maximum parameter norm: `0.442864`
- Mini-batch loss volatility: `0.181532`
- Training accuracy: `0.593750`
- Validation accuracy: `0.578125`
- Training log loss: `0.835787`
- Validation log loss: `0.765287`

No configured failure threshold was crossed for this setup. The initial
mini-batch loss was `0.905015` and the final sampled mini-batch loss was
`0.679376`, although those batches are different and the values should not be
read as a controlled convergence comparison.

## Known failure modes

All four 512-row configurations crossed the mini-batch loss-volatility threshold.
The tested configurations did not show vanishing gradients, exploding gradients,
unbounded parameter growth, non-finite values, or the configured overfitting gap.

The numerically stable configuration is not yet a strong predictor: its
validation log loss remains worse than the approximately `0.693` random-
probability baseline. The next apples-to-apples comparison should use this setup
as the stable QCNN starting point while reporting that optimization stability did
not by itself solve predictive quality.
