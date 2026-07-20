"""Data loading and preparation helpers for the local results dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_REPORTS_DIR = Path("reports")


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet report table, returning an empty frame if absent."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dashboard table format: {path.suffix}")


def latest_signal_report(reports_dir: str | Path = DEFAULT_REPORTS_DIR) -> pd.DataFrame:
    """Load and order the generated daily signal report."""
    signals = load_table(Path(reports_dir) / "daily_signal.csv")
    if signals.empty:
        return signals
    required = {"symbol", "rank", "predicted_outperformance_probability"}
    _require_columns(signals, required, "Daily signal report")
    if "date" in signals:
        signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    return signals.sort_values("rank").reset_index(drop=True)


def model_comparison(reports_dir: str | Path = DEFAULT_REPORTS_DIR) -> pd.DataFrame:
    """Load the unified classical model comparison table."""
    return load_table(Path(reports_dir) / "backtests" / "classical_baseline_comparison.parquet")


def portfolio_series(reports_dir: str | Path = DEFAULT_REPORTS_DIR) -> pd.DataFrame:
    """Load QML comparison returns, falling back to classical backtests."""
    reports_dir = Path(reports_dir)
    data = load_table(reports_dir / "qml_comparison" / "portfolio_returns.parquet")
    if data.empty:
        data = load_table(reports_dir / "backtests" / "portfolio_backtest.parquet")
    if data.empty:
        return data
    _require_columns(data, {"model_name", "date", "net_return"}, "Portfolio returns")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values(["model_name", "date"]).copy()
    data["cumulative_net_return"] = data.groupby("model_name")["net_return"].transform(
        lambda values: (1 + values).cumprod() - 1
    )
    wealth = 1 + data["cumulative_net_return"]
    peaks = wealth.groupby(data["model_name"]).cummax()
    data["drawdown"] = wealth / peaks - 1
    return data.reset_index(drop=True)


def top_ranked_stocks(
    reports_dir: str | Path = DEFAULT_REPORTS_DIR, *, limit: int = 10
) -> pd.DataFrame:
    """Return daily signals or latest-date model scores when signals are absent."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    signals = latest_signal_report(reports_dir)
    if not signals.empty:
        equities = (
            signals.loc[~signals["is_benchmark"].astype(bool)]
            if "is_benchmark" in signals
            else signals
        )
        return equities.head(limit).reset_index(drop=True)

    predictions = load_table(Path(reports_dir) / "backtests" / "predictions.parquet")
    if predictions.empty:
        return predictions
    _require_columns(predictions, {"symbol", "date", "y_score", "model_name"}, "Predictions")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    latest_date = predictions["date"].max()
    latest = predictions.loc[predictions["date"].eq(latest_date)].copy()
    latest = latest.sort_values(["model_name", "y_score"], ascending=[True, False])
    latest["rank"] = latest.groupby("model_name").cumcount() + 1
    columns = ["date", "model_name", "rank", "symbol", "y_score"]
    return latest.loc[latest["rank"].le(limit), columns].reset_index(drop=True)


def qml_experiment_summary(reports_dir: str | Path = DEFAULT_REPORTS_DIR) -> pd.DataFrame:
    """Pivot aggregate QML experiment metrics into a model comparison table."""
    metrics = load_table(Path(reports_dir) / "qml_comparison" / "aggregate_metrics.parquet")
    if metrics.empty:
        return metrics
    _require_columns(metrics, {"model_name", "metric", "mean"}, "QML aggregate metrics")
    return (
        metrics.pivot_table(index="model_name", columns="metric", values="mean", aggfunc="first")
        .reset_index()
        .rename_axis(columns=None)
        .sort_values("model_name")
        .reset_index(drop=True)
    )


def _require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: " + ", ".join(sorted(missing)))
