# Minimal Eight-Qubit QCNN Blocks

The QCNN architecture reduces an eight-qubit encoded market observation to two
active readout qubits using alternating convolution and pooling stages. This
issue constructs and executes the circuit without training it; optimization and
classification are handled in the following QCNN issue.

## Active-qubit flow

```text
Input:    q0  q1  q2  q3  q4  q5  q6  q7
           \ /     \ /     \ /     \ /
Stage 0:   q0      q2      q4      q6
             \     /         \     /
Stage 1:       q0              q4
```

At each stage, adjacent active qubits first pass through a convolution block.
Pooling then transfers information from the second qubit of each pair into the
first, and the second qubit is removed from the active-qubit list. The underlying
eight-qubit statevector remains unitary; “inactive” means later QCNN blocks no
longer operate on that wire.

The exact active flow is:

```text
(0, 1, 2, 3, 4, 5, 6, 7)
          -> (0, 2, 4, 6)
          -> (0, 4)
```

## Two-qubit convolution

For active qubits `a` and `b`, the trainable convolution is:

```text
RY(theta0) a
RY(theta1) b
CNOT a -> b
RY(theta2) b
CNOT b -> a
RY(theta3) a
```

The bidirectional CNOTs allow information to flow in both directions. Each
convolution block has parameter shape `(4,)`.

## Pooling

For retiring source `b` and retained target `a`, pooling applies:

```text
CNOT b -> a
RY(phi0) a
```

Each pooling block has parameter shape `(1,)`.

## Complete parameter layout

- Stage 0: four convolutions (`16` parameters) and four pools (`4`).
- Stage 1: two convolutions (`8` parameters) and two pools (`2`).
- Complete flat parameter vector: shape `(30,)`.

Parameters are initialized reproducibly from a uniform interval, `[-0.1, 0.1]`
by default. The constructed operation table stores every gate in execution order
and maps each trainable RY gate to a unique position in the flat vector.

Both the standalone blocks and complete architecture execute on the shared exact
NumPy statevector backend and preserve state normalization.
