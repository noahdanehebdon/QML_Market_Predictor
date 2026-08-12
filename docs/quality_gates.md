# Quality gates

CI enforces the same commands contributors can run locally:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src/market_qml/cli.py scripts/check_git_data_policy.py scripts/verify_wheel_install.py
python -m pytest -q --cov=market_qml --cov-report=term-missing:skip-covered --cov-report=json:coverage.json --cov-fail-under=82
python -m build
python -m twine check dist/*
$wheel = Get-ChildItem dist -Filter "*.whl" | Select-Object -First 1 -ExpandProperty FullName
python -m scripts.verify_wheel_install $wheel
python -m pip_audit . --strict
```

PowerShell does not expand `*.whl` for native commands, so the wheel path is
resolved explicitly above. In Bash, `python -m scripts.verify_wheel_install
dist/*.whl` remains valid.

## Coverage baseline

The initial July 2026 baseline is 82.8% branch-aware source coverage on the local
suite with three unavailable optional-binary tests omitted. Clean CI executes
the complete suite. The enforced minimum is 82%, close enough to expose a
meaningful regression while allowing small platform-specific differences.
Each matrix job writes its coverage table to the GitHub Actions job summary;
generated coverage files are not published as repository artifacts.

## Incremental typing

Strict mypy checking begins with the package CLI, tracked-data policy, and
installed-wheel verifier. Expand this list as modules acquire complete type
annotations rather than weakening strict mode globally.

## Dependency and distribution checks

`pip-audit` resolves the dependencies declared by this project and fails on a
known vulnerability or an incomplete audit. It does not audit unrelated
packages from a developer's global environment. `build`, `twine check`, and the
wheel verifier validate the source distribution, wheel metadata, clean
installation, and all seven console entrypoints.
