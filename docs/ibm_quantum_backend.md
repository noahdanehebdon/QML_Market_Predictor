# IBM Quantum backend

VQC parameters are trained with the repository's exact local simulator. Only a
small, fixed validation subset is sent to IBM Quantum for shot-based evaluation.

Install the optional dependencies:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[quantum]"
```

Use a clean environment. Modern Qiskit cannot coexist with legacy
`qiskit-terra` or `qiskit-ibmq-provider` installations.

Set credentials locally or through GitHub repository secrets. Never commit them:

```text
IBM_QUANTUM_API_KEY=<IBM Cloud API key>
IBM_QUANTUM_INSTANCE=<Qiskit Runtime service instance CRN>
```

Run an explicitly bounded smoke evaluation:

```bash
python -m scripts.run_ibm_vqc_smoke \
  --model artifacts/models/vqc.pkl \
  --sample data/processed/qml_sample.parquet \
  --output-dir artifacts/qml/ibm_smoke \
  --rows 4 \
  --shots 1024 \
  --max-total-shots 4096
```

Omit `--backend` to select the least-busy operational device with enough qubits,
or pass a device name explicitly. The command refuses requests exceeding the
circuit or total-shot limits. It saves the Runtime job ID, device, UTC timestamp,
shots, transpiled depth, gate counts, client-side timing, configuration, and
scores. Hardware is noisy and queued; submitted work can consume an IBM plan's
allocated execution time.

The adapter follows IBM's current `ibm_quantum_platform`, `SamplerV2`, and preset
pass-manager workflow. Tests use mocked provider objects and require no account.

## GitHub Actions

The **IBM Quantum hardware smoke** workflow is manual-only. Open Actions, select
the workflow, choose a backend or leave it blank for least-busy selection, and
select the bounded circuit and shot counts. Repository secrets are injected only
into that job. Pull requests and pushes never receive them and never submit QPU
work. The workflow uses a deterministic four-row-or-smaller fixture so exact and
hardware scores are directly comparable and uploads the result bundle.

The submission ID is written before polling. If a runner is interrupted, retrieve
the job with `QiskitRuntimeService.job(job_id)`; the adapter's
`collect_ibm_vqc()` function supports the same resumption path. IBM Runtime job
metrics and available backend calibration properties are added after completion.
