# Quantum Kernel Feature Map

The quantum kernel feature map converts each PCA-compressed market observation
into a normalized quantum state. Later QSVM work can compare observations with
the fidelity kernel

\[
K(x,z) = |\langle \phi(x) | \phi(z) \rangle|^2.
\]

## Circuit choice

The default circuit uses 8 qubits and 2 data-reuploading repetitions. In each
repetition it:

1. Maps each of the eight selected PCA components to `[-pi, pi]` with
   `2 * atan(x)`.
2. Applies one `RY` rotation to each qubit using the corresponding angle.
3. Applies a ring of `CZ` gates to entangle neighboring qubits.

The map has no trained parameters. Identical input rows therefore produce
identical states, and the same circuit is applied to training and validation
rows. PCA must already have been fitted on training data only by the existing
QML compression workflow.

## Simulator backend

The selected backend is `numpy_statevector`, the repository's batched exact
statevector simulator. An 8-qubit row is represented by 256 complex amplitudes.
The simulator applies unitary RY and CZ gates, so state norms remain one. Exact
state overlaps produce deterministic kernel values between zero and one without
finite-shot noise.

## Building circuit outputs

From an editable installation, run:

```powershell
python -m scripts.build_quantum_feature_map --split-id 0
```

The command uses the first eight deterministic, sorted PCA component columns and
saves:

- Compressed NumPy state arrays for training and validation rows.
- Training and validation metadata preserving row alignment.
- An ordered circuit-operation table documenting every RY and CZ gate.

These state arrays are the circuit outputs required to construct the train and
validation kernel matrices in issue 4.8. The current backend is an exact local
simulator; shot-based and real-hardware execution are intentionally deferred to
the separate hardware-backend issue.
