"""Simple portfolio backtests from standard model prediction tables."""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


PORTFOLIO_RETURN_COLUMNS = [
    "model_name",
    "split_id",
    "date",
    "selected_count",
    "turnover",
    "transaction_cost",
    "gross_return",
    "net_return",
    "benchmark_return",
    "gross_excess_return",
    "net_excess_return",
    "cumulative_gross_return",
    "cumulative_net_return",
    "benchmark_cumulative_return",
    "cumulative_gross_excess_return",
    "cumulative_net_excess_return",
]


def run_portfolio_backtest(
    predictions: pd.DataFrame,
    *,
    top_k: int | None = None,
    top_fraction: float = 0.1,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Run an equal-weight long-only backtest from model scores."""
    _validate_prediction_table(predictions)
    _validate_selection(top_k=top_k, top_fraction=top_fraction)
    _validate_transaction_cost(transaction_cost_bps)

    rows = []
    grouped = predictions.groupby(["model_name", "split_id"], sort=True)
    for (model_name, split_id), model_split_predictions in grouped:
        previous_weights: dict[str, float] = {}
        for date, group in model_split_predictions.groupby("date", sort=True):
            selected = _select_names(group, top_k=top_k, top_fraction=top_fraction)
            current_weights = _equal_weights(selected["symbol"])
            turnover = _portfolio_turnover(previous_weights, current_weights)
            transaction_cost = turnover * (transaction_cost_bps / 10_000)
            gross_return = float(selected["forward_return"].mean())
            net_return = gross_return - transaction_cost
            benchmark_return = float(
                (selected["forward_return"] - selected["forward_excess_return"]).mean()
            )
            rows.append(
                {
                    "model_name": str(model_name),
                    "split_id": int(split_id),
                    "date": pd.Timestamp(date).normalize(),
                    "selected_count": len(selected),
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "gross_excess_return": gross_return - benchmark_return,
                    "net_excess_return": net_return - benchmark_return,
                }
            )
            previous_weights = current_weights

    result = pd.DataFrame(rows).sort_values(["model_name", "split_id", "date"])
    if result.empty:
        raise ValueError("No portfolio return rows were produced.")

    result["cumulative_gross_return"] = result.groupby(["model_name", "split_id"])[
        "gross_return"
    ].transform(lambda returns: (1 + returns).cumprod() - 1)
    result["cumulative_net_return"] = result.groupby(["model_name", "split_id"])[
        "net_return"
    ].transform(lambda returns: (1 + returns).cumprod() - 1)
    result["benchmark_cumulative_return"] = result.groupby(["model_name", "split_id"])[
        "benchmark_return"
    ].transform(lambda returns: (1 + returns).cumprod() - 1)
    result["cumulative_gross_excess_return"] = (
        result["cumulative_gross_return"] - result["benchmark_cumulative_return"]
    )
    result["cumulative_net_excess_return"] = (
        result["cumulative_net_return"] - result["benchmark_cumulative_return"]
    )

    return result[PORTFOLIO_RETURN_COLUMNS].reset_index(drop=True)


def load_prediction_tables(prediction_paths: list[str | Path]) -> pd.DataFrame:
    """Load and combine standard prediction parquet files for portfolio backtests."""
    if not prediction_paths:
        raise ValueError("At least one prediction path is required.")

    frames = []
    for prediction_path in prediction_paths:
        predictions = pd.read_parquet(Path(prediction_path))
        _validate_prediction_columns(predictions)
        frames.append(predictions)

    return pd.concat(frames, ignore_index=True)


def save_portfolio_returns(portfolio_returns: pd.DataFrame, output_path: str | Path) -> None:
    """Save portfolio return series to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_returns.to_parquet(output_path, index=False)


def _select_names(
    predictions: pd.DataFrame,
    *,
    top_k: int | None,
    top_fraction: float,
) -> pd.DataFrame:
    ordered = predictions.sort_values(["y_score", "symbol"], ascending=[False, True])
    count = top_k if top_k is not None else math.ceil(len(ordered) * top_fraction)
    count = max(1, min(count, len(ordered)))
    return ordered.head(count)


def _equal_weights(symbols: pd.Series) -> dict[str, float]:
    symbol_list = symbols.astype(str).tolist()
    weight = 1 / len(symbol_list)
    return {symbol: weight for symbol in symbol_list}


def _portfolio_turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    if not previous_weights:
        return 1.0

    symbols = set(previous_weights) | set(current_weights)
    return 0.5 * sum(
        abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
        for symbol in symbols
    )


def _validate_prediction_table(predictions: pd.DataFrame) -> None:
    _validate_prediction_columns(predictions)
    if predictions.empty:
        raise ValueError("Prediction table is empty.")

    numeric_columns = ["y_score", "forward_return", "forward_excess_return"]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Portfolio backtest requires numeric, non-missing scores and returns.")


def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
    missing_columns = set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def _validate_selection(*, top_k: int | None, top_fraction: float) -> None:
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be greater than 0 and at most 1.")


def _validate_transaction_cost(transaction_cost_bps: float) -> None:
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative.")
