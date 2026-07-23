# Contributing

## Scope

Contributions should preserve the repository's central research guarantees:
chronological evaluation, point-in-time information, train-only preprocessing,
auditable artifacts, honest reporting, and paper-only execution.

Open an issue before a large methodology or architecture change. Keep refactoring
separate from changes to targets, model behavior, or reported results.

## Development workflow

1. Create a focused branch from current `main`.
2. Install the project with `python -m pip install --editable ".[dev]"`.
3. Add or update focused tests before changing established behavior.
4. Run the checks below.
5. Explain methodology changes, artifact compatibility, and limitations in the PR.

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m scripts.check_git_data_policy
python -m build
python -m twine check dist/*
```

## Data and security

Never commit credentials, `.env`, provider responses, market datasets, account data,
trade intents, execution journals, model artifacts, or private reports. Use synthetic
fixtures for tests. If a secret is exposed, revoke it immediately and follow
[SECURITY.md](SECURITY.md); deleting it from the newest commit is insufficient.

## Research changes

A model or methodology PR should state:

- the hypothesis and comparison baseline;
- the data, target, split, purge, and embargo used;
- which choices were selected on inner training data;
- whether locked-test data were accessed;
- predictive, calibration, ranking, portfolio, and resource effects;
- uncertainty, negative results, and known limitations.

Do not describe development-only improvements as final, simulated QML as hardware
evidence, or paper execution as evidence of live performance.
