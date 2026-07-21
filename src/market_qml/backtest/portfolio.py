"""Simple portfolio backtests from standard model prediction tables."""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RETURN_HORIZON_DAYS = 5
DEFAULT_REBALANCE_FREQUENCY = 5
DEFAULT_TRANSACTION_COST_BPS = 10.0

PORTFOLIO_RETURN_COLUMNS = [
    "model_name",
    "split_id",
    "date",
    "return_horizon_days",
    "rebalance_frequency",
    "transaction_cost_bps",
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

PORTFOLIO_RISK_COLUMNS = [
    "model_name",
    "scope",
    "split_id",
    "rows",
    "return_horizon_days",
    "rebalance_frequency",
    "periods_per_year",
    "transaction_cost_bps",
    "cumulative_gross_return",
    "cumulative_net_return",
    "benchmark_cumulative_return",
    "cumulative_net_excess_return",
    "gross_volatility",
    "net_volatility",
    "benchmark_volatility",
    "net_excess_volatility",
    "gross_sharpe",
    "net_sharpe",
    "benchmark_sharpe",
    "net_excess_sharpe",
    "gross_max_drawdown",
    "net_max_drawdown",
    "benchmark_max_drawdown",
    "hit_rate",
    "excess_hit_rate",
    "average_turnover",
    "total_transaction_cost",
]


def run_portfolio_backtest(
    predictions: pd.DataFrame,
    *,
    top_k: int | None = None,
    top_fraction: float = 0.1,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY,
    return_horizon_days: int = DEFAULT_RETURN_HORIZON_DAYS,
) -> pd.DataFrame:
    """Run an equal-weight long-only backtest from model scores."""
    _validate_prediction_table(predictions)
    _validate_selection(top_k=top_k, top_fraction=top_fraction)
    _validate_transaction_cost(transaction_cost_bps)
    _validate_rebalance_frequency(rebalance_frequency)
    if return_horizon_days <= 0:
        raise ValueError("return_horizon_days must be positive.")
    if rebalance_frequency < return_horizon_days:
        raise ValueError(
            "rebalance_frequency must be at least return_horizon_days to avoid "
            "overlapping forward returns."
        )

    rows = []
    grouped = predictions.groupby(["model_name", "split_id"], sort=True)
    for (model_name, split_id), model_split_predictions in grouped:
        previous_weights: dict[str, float] = {}
        date_groups = list(model_split_predictions.groupby("date", sort=True))
        for date_index, (date, group) in enumerate(date_groups):
            if date_index % rebalance_frequency != 0:
                continue

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
                    "return_horizon_days": return_horizon_days,
                    "rebalance_frequency": rebalance_frequency,
                    "transaction_cost_bps": transaction_cost_bps,
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


def summarize_portfolio_risk(
    portfolio_returns: pd.DataFrame,
    *,
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    """Summarize risk-adjusted performance by split and overall."""
    _validate_portfolio_returns(portfolio_returns)
    settings = _portfolio_settings(portfolio_returns)
    inferred_periods = TRADING_DAYS_PER_YEAR / settings["rebalance_frequency"]
    if periods_per_year is None:
        periods_per_year = inferred_periods
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    rows = []
    for (model_name, split_id), group in portfolio_returns.groupby(
        ["model_name", "split_id"],
        sort=True,
    ):
        rows.append(
            _risk_row(
                group,
                model_name=str(model_name),
                scope="split",
                split_id=int(split_id),
                periods_per_year=periods_per_year,
            )
        )

    for model_name, group in portfolio_returns.groupby("model_name", sort=True):
        rows.append(
            _risk_row(
                group,
                model_name=str(model_name),
                scope="overall",
                split_id=pd.NA,
                periods_per_year=periods_per_year,
            )
        )

    return pd.DataFrame(rows, columns=PORTFOLIO_RISK_COLUMNS)


def save_portfolio_risk_metrics(
    risk_metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save portfolio risk summary metrics to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    risk_metrics.to_parquet(output_path, index=False)


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


def _risk_row(
    portfolio_returns: pd.DataFrame,
    *,
    model_name: str,
    scope: str,
    split_id,
    periods_per_year: int,
) -> dict:
    ordered = portfolio_returns.sort_values(["split_id", "date"])
    gross = ordered["gross_return"]
    net = ordered["net_return"]
    benchmark = ordered["benchmark_return"]
    net_excess = ordered["net_excess_return"]

    return {
        "model_name": model_name,
        "scope": scope,
        "split_id": split_id,
        "rows": len(ordered),
        "return_horizon_days": int(ordered["return_horizon_days"].iloc[0]),
        "rebalance_frequency": int(ordered["rebalance_frequency"].iloc[0]),
        "periods_per_year": periods_per_year,
        "transaction_cost_bps": float(ordered["transaction_cost_bps"].iloc[0]),
        "cumulative_gross_return": _cumulative_return(gross),
        "cumulative_net_return": _cumulative_return(net),
        "benchmark_cumulative_return": _cumulative_return(benchmark),
        "cumulative_net_excess_return": _cumulative_return(net)
        - _cumulative_return(benchmark),
        "gross_volatility": _annualized_volatility(gross, periods_per_year),
        "net_volatility": _annualized_volatility(net, periods_per_year),
        "benchmark_volatility": _annualized_volatility(benchmark, periods_per_year),
        "net_excess_volatility": _annualized_volatility(net_excess, periods_per_year),
        "gross_sharpe": _annualized_sharpe(gross, periods_per_year),
        "net_sharpe": _annualized_sharpe(net, periods_per_year),
        "benchmark_sharpe": _annualized_sharpe(benchmark, periods_per_year),
        "net_excess_sharpe": _annualized_sharpe(net_excess, periods_per_year),
        "gross_max_drawdown": _max_drawdown(gross),
        "net_max_drawdown": _max_drawdown(net),
        "benchmark_max_drawdown": _max_drawdown(benchmark),
        "hit_rate": float((net > 0).mean()),
        "excess_hit_rate": float((net_excess > 0).mean()),
        "average_turnover": float(ordered["turnover"].mean()),
        "total_transaction_cost": float(ordered["transaction_cost"].sum()),
    }


def _cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def _annualized_volatility(returns: pd.Series, periods_per_year: float):
    if len(returns) < 2:
        return pd.NA
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def _annualized_sharpe(returns: pd.Series, periods_per_year: float):
    if len(returns) < 2:
        return pd.NA
    volatility = returns.std(ddof=1)
    if volatility == 0:
        return pd.NA
    return float((returns.mean() / volatility) * math.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def _validate_prediction_table(predictions: pd.DataFrame) -> None:
    _validate_prediction_columns(predictions)
    if predictions.empty:
        raise ValueError("Prediction table is empty.")

    numeric_columns = ["y_score", "forward_return", "forward_excess_return"]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Portfolio backtest requires numeric, non-missing scores and returns.")


def _validate_portfolio_returns(portfolio_returns: pd.DataFrame) -> None:
    missing_columns = set(PORTFOLIO_RETURN_COLUMNS) - set(portfolio_returns.columns)
    if missing_columns:
        raise ValueError(
            "Portfolio returns table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns table is empty.")


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


def _validate_rebalance_frequency(rebalance_frequency: int) -> None:
    if rebalance_frequency <= 0:
        raise ValueError("rebalance_frequency must be positive.")


def _portfolio_settings(portfolio_returns: pd.DataFrame) -> dict[str, float]:
    columns = [
        "return_horizon_days",
        "rebalance_frequency",
        "transaction_cost_bps",
    ]
    settings = {}
    for column in columns:
        values = portfolio_returns[column].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Portfolio returns must have one consistent {column}.")
        settings[column] = float(values[0])
    return settings
