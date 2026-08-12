# Quantum machine-learning experiments

> **Superseded portfolio metrics:** historical volatility and Sharpe values in
> this document used 252 annualization periods for non-overlapping five-day
> returns. Issue #153 corrected this to 50.4; portfolio risk figures must be
> recomputed before use.

This document is the self-contained guide to the repository's quantum
machine-learning experiments. It explains what was tested, how data reached the
circuits, how the models were compared, what the experiments found, and which
conclusions are justified.

## Research question

The experiments ask whether small simulated quantum classifiers add useful
signal for predicting whether an equity will outperform SPY over the next five
trading days. The binary target is `outperform_spy_5d`; continuous forward excess
return is retained for ranking and portfolio evaluation.

The project tests three QML families:

- a variational quantum classifier (VQC),
- a quantum-kernel support vector machine (QSVM), and
- an eight-qubit quantum convolutional neural network (QCNN).

These are compared with linear and RBF SVM controls and, in the final classical
comparison, logistic regression and histogram gradient boosting. All current
quantum circuits run on the repository's exact NumPy statevector simulator. A
separate, explicitly invoked IBM Quantum Runtime smoke path supports bounded
hardware inference with fixed, locally trained VQC parameters; the comparisons
reported in this document remain simulator results. No reported comparison
result comes from physical quantum hardware.

## Leakage-safe experimental design

The final controlled run uses six chronological walk-forward splits. Within
each split, every primary model receives the same 256 balanced training rows,
256 balanced validation rows, eight input features, and validation row keys.
The sample manifest records row counts, class balance, symbols, and a SHA-256
hash of the exact symbol/date keys.

The outer validation period is used once for model comparison. QSVM
hyperparameters are selected with an inner chronological partition of the outer
training window; final-period validation labels do not influence selection.
Feature selection and standardization are also fitted on outer-training data
only.

This design matters more than small metric differences. It prevents a model
from appearing better because it saw easier rows, different features, or
validation information during tuning.

## Feature compression and encoding

Eight qubits can directly encode only eight scalar inputs with the chosen angle
encoding. The project therefore reduces the much wider market feature table
before circuit execution.

The historical workflow grouped source features by economic family, fitted PCA
on training rows, and supplied eight PCA components to each model. The expanded
workflow instead uses the feature-count budget selected by the classical
gradient-boosting tuner, ranks source features against continuous excess return
on outer-training rows, removes highly correlated duplicates, standardizes the
survivors with training-only statistics, and assigns eight features to qubits.
Related features are placed next to one another on the circuit ring so local
entanglers connect economically related inputs.

Every selected scalar is mapped to a rotation angle with:

```text
angle = 2 * atan(x)
```

This bounded, stateless transform maps real values into `[-pi, pi]`. It does not
fit parameters on validation data. Compression is an engineering requirement,
but it is also a limitation: potentially useful information is discarded before
the quantum model sees the observation.

## VQC design

The VQC begins in the all-zero eight-qubit state. One `RY` rotation per qubit
encodes the eight inputs. Each variational layer then applies a trainable `RY`
rotation to every qubit followed by a ring of CNOT entanglers. The positive-class
score is the exact probability of measuring qubit zero as one.

Training minimizes binary cross-entropy with small L2 regularization. SPSA is
the default optimizer: two randomly perturbed circuit evaluations estimate the
full gradient regardless of parameter count. Centered finite differences are
available as a more expensive control. Mini-batches and seeded initialization
make experiments reproducible, though per-iteration loss remains noisy.

The bounded architecture study compared ansatz depths 1 and 2, learning rates
0.05 and 0.1, and SPSA versus finite differences. Depth 1, learning rate 0.1,
and SPSA produced the best tested validation log loss (`0.6953`) on the initial
split, essentially random-classifier performance. This configuration was a
starting point, not evidence of predictive value.

## QSVM design

The QSVM uses a quantum circuit to define similarity and a classical SVM to fit
the decision boundary. Its feature map repeats:

1. `RY` rotations containing the encoded inputs,
2. a ring of `CZ` gates between neighboring qubits, and
3. optional neighbor-interaction re-uploading in the redesigned kernel.

For observations `x` and `z`, the kernel is the squared state fidelity:

```text
K(x, z) = |<phi(x) | phi(z)>|^2
```

The simulator constructs a square training kernel and a rectangular
validation-to-training kernel. Scikit-learn's `SVC` then trains with
`kernel="precomputed"` and produces calibrated positive-class probabilities.
The state preparation and overlap are quantum-inspired circuit computations;
SVM fitting and probability calibration are classical.

The controlled grid tests `C` in `{0.01, 0.1, 1, 10}`, feature-map repetitions
in `{1, 2, 3}`, and interaction scales in `{0, 0.5, 1}`. Every trial, kernel
dimension, kernel mean similarity, support-vector count, runtime, and selected
configuration is retained for auditability. Kernel computation grows
quadratically with training rows, which is why the QML experiment uses a bounded
sample.

## QCNN design

The QCNN applies the same eight-input `RY` encoding, then a trainable 30-parameter
convolution-and-pooling circuit. Its hierarchy reduces active qubits from
`8 → 4 → 2`. Repeated local convolution blocks share a common structure, while
pooling transfers information from qubits that are discarded to those retained
for the next stage.

After the final stage, the simulator calculates exact Pauli-Z expectations for
the two active readout qubits. Their mean is converted to a class score:

```text
score = (1 - mean(<Z_q0>, <Z_q4>)) / 2
```

The QCNN uses binary cross-entropy, L2 regularization, mini-batches, and SPSA.
Its initial bounded validation ROC-AUC was `0.5943`, but log loss (`0.7083`) and
training stability showed that the score calibration and optimization were not
yet reliable. Later stability experiments found substantial mini-batch loss and
gradient variability; more circuit depth or iterations alone did not resolve
that behavior.

## Final controlled results

The expanded six-split comparison produced the following mean classification
metrics on identical rows and inputs:

| Model | Accuracy | ROC-AUC | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.5241 | 0.5406 | 0.7022 | 0.2540 |
| Gradient boosting | 0.5150 | 0.5398 | 0.8457 | 0.3018 |
| QCNN | 0.5169 | 0.5314 | 0.8934 | 0.3135 |
| Linear SVM | 0.5091 | 0.5250 | 0.6926 | 0.2497 |
| RBF SVM | 0.5228 | 0.5223 | 0.6943 | 0.2505 |
| Fixed QSVM | 0.5150 | 0.5215 | 0.6943 | 0.2505 |
| Tuned QSVM | 0.5254 | 0.5095 | 0.6961 | 0.2513 |
| VQC | 0.4974 | 0.4973 | 0.6941 | 0.2505 |

Logistic regression is the classification leader. QCNN is the strongest QML
classifier, but its mean ROC-AUC advantage over random is small and its
calibration is poor. QSVM tuning improves inner-training validation scores but
does not generalize to the outer periods. VQC is indistinguishable from random
on mean ROC-AUC.

The downstream results are mixed. Fixed QSVM has the strongest overall rank IC
(`0.0804`), and VQC has the highest long-only net Sharpe in the bounded backtest
(`4.5183`) despite below-random classification ROC-AUC. These inconsistencies
are reasons to investigate score ordering, sample sensitivity, and portfolio
construction—not evidence of quantum advantage.

## Regime findings

Classical models lead ROC-AUC in every volatility, rate, and yield-curve slice
with at least 50 observations. QCNN is stronger in low volatility (`0.5372`),
falling rates (`0.5409`), and inverted curves (`0.5390`) than in high volatility
(`0.4933`), rising rates (`0.5197`), and normal curves (`0.5200`). Fixed QSVM has
the strongest inverted-curve rank IC (`0.0554`). These are descriptive patterns
for future validation, not statistically established conditional advantages.

## Interpretation

The experiments do not demonstrate quantum advantage. On the tested data,
circuits, simulator, and sample budget:

- classical baselines are stronger and simpler for classification;
- controlled QSVM tuning does not produce stable outer-split improvement;
- the redesigned quantum kernel changes behavior but not the overall conclusion;
- QCNN has weak discrimination and poor calibration despite being the strongest
  QML classifier; and
- isolated ranking or portfolio leaders do not establish robust superiority.

The useful result is an auditable negative experiment. The infrastructure makes
data equality, nested tuning, uncertainty, resource cost, and downstream model
behavior visible instead of selecting a favorable isolated score.

## Limitations

- **Simulation only:** exact statevectors omit device noise, finite shots,
  compilation constraints, queue time, and hardware cost.
- **Small circuits and samples:** eight qubits and bounded balanced samples are
  computational compromises, not a complete test of QML at scale.
- **Compression dependence:** conclusions apply to the selected eight-feature
  representations; other leakage-safe encodings may behave differently.
- **Optimization budget:** short SPSA runs can underfit, while deeper searches
  increase selection risk and simulation cost.
- **Uncertainty:** six walk-forward splits provide limited power. Overlapping
  bootstrap intervals near 0.5 make small mean differences inconclusive.
- **Calibration:** several QML scores have poor log loss or Brier score even when
  rank metrics appear useful.
- **Backtest realism:** results depend on a small universe, simplified execution,
  overlapping forward-return horizons, turnover assumptions, and fixed costs.
- **Regime multiplicity:** slicing creates many comparisons; observed differences
  may be sampling noise.
- **No causal claim:** predictive associations do not imply that quantum circuit
  structure captures a causal market mechanism.

## Reproducing the experiments

From the repository root, after building the feature table, labels, splits, and
classical selection diagnostics:

```powershell
python scripts/compare_qml_models.py
python scripts/build_market_regimes.py
python scripts/analyze_qml_regimes.py
```

The comparison writes predictions, split metrics, bootstrap summaries, ranking
metrics, portfolio metrics, resource usage, tuning trials, selected
configurations, and row manifests under `reports/qml_comparison/`. The regime
analysis writes joined predictions, slice metrics, QCNN pairwise differences,
and a generated report under `reports/qml_regimes/`.

For implementation-level detail, see:

- [Quantum feature map](quantum_feature_map.md)
- [VQC tuning](vqc_tuning.md)
- [QSVM](qsvm.md)
- [QCNN blocks](qcnn_blocks.md)
- [QCNN classifier](qcnn_classifier.md)
- [QCNN stability](qcnn_stability.md)
- [Controlled model comparison](qml_model_comparison.md)
- [Regime analysis](qml_regime_analysis.md)
