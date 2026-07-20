# Data versioning and reproducibility

## Decision

Generated and provider-derived datasets are not versioned in Git. The project
uses local working directories for development and a private Cloudflare R2
bucket for automated snapshots. DVC is intentionally deferred: R2 snapshot
paths and checksum manifests provide the required reproducibility without
adding another dependency. This decision can be revisited if dataset branching
or multi-user data lineage becomes necessary.

The repository may be public, but the R2 bucket must remain private. Disable
the `r2.dev` public URL and custom domains, scope the API token to the project
bucket, and apply a 30-day object lifecycle rule.

## What must not be committed

- raw or processed provider data;
- feature tables, labels, predictions, and backtest result tables;
- model, preprocessor, PCA, and quantum artifacts;
- MLflow databases and experiment runs;
- credentials, `.env` files, API responses, or request headers; and
- any non-fixture file larger than 5 MiB.

Small deterministic fixtures under `tests/fixtures/` are allowed up to 1 MiB.
The `.gitignore` rules provide the first safeguard. The CI data-policy check
examines tracked files, so `git add -f` cannot silently bypass the policy.

## Automated snapshot identity

Each successful nightly refresh creates `data/processed/data_manifest.json`.
The manifest records the workflow run ID, Git commit, creation time, byte size,
and SHA-256 digest of every processed file. The snapshot is uploaded to:

```text
processed/runs/<workflow-run-id>/
```

The private `processed/latest-run-id.txt` pointer identifies the newest
complete snapshot. It is updated only after upload succeeds. Weekly retraining
downloads that snapshot and verifies every file against its manifest before
building features.

Weekly retraining outputs are stored under `reports/runs/<workflow-run-id>/`
in the same private bucket. Workflows do not use GitHub Actions artifacts for
provider data, processed data, predictions, reports, or model outputs.

## Local regeneration

Configure `.env` as described in the README, then run from the repository root:

```powershell
ingest-prices
python -m scripts.pull_macro --config configs/data_sources.yaml --start-year 2020
python -m scripts.build_macro_daily --prices data/processed/prices.parquet
ingest-sec
build-features
python -m scripts.data_manifest create
python -m scripts.data_manifest verify
```

Record the Git commit, configuration files, and generated manifest with any
published experiment. Other users must obtain data under their own provider
accounts and terms; this repository does not redistribute market datasets.

## Recovery and retention

R2 stores processed data and generated weekly reports. Raw provider responses
stay on the temporary runner and are deleted with it. The bucket lifecycle
removes snapshots and reports after 30 days. Recreate an expired snapshot by
checking out its recorded Git commit, using the same configuration, and
rerunning ingestion and feature generation.
