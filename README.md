# QML_Market_Predictor
A market prediction platform comparing classical ML, standard QML models, and a QCNN architecture for regime-aware equity outperformance prediction.

## Local Environment

Create a local `.env` file from `.env.example` and fill in your own API credentials:

```text
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
FRED_API_KEY=
SEC_USER_AGENT=
```

Do not commit `.env`. It is ignored by Git and should contain only local secrets.

The SEC requires a descriptive User-Agent for automated requests. Use a value that identifies the project and provides contact information, such as `QML Market Predictor contact@example.com`.

## GitHub Secrets

For GitHub Actions or other hosted workflows, add the same secret names in the repository settings under GitHub Secrets:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `FRED_API_KEY`
- `SEC_USER_AGENT`
