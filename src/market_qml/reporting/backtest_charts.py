"""Reproducible static charts for model and portfolio results."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_qml.backtest.portfolio import TRADING_DAYS_PER_YEAR


FIGURE_FILENAMES = {
    "cumulative_returns": "cumulative_returns.png",
    "drawdowns": "drawdowns.png",
    "rolling_sharpe": "rolling_sharpe.png",
    "model_comparison": "model_comparison_metrics.png",
    "regime_performance": "regime_performance.png",
}
_COLORS = plt.get_cmap("tab10").colors
_LINE_STYLES = ("-", "--", "-.", ":")


def generate_backtest_charts(
    *,
    portfolio_returns: pd.DataFrame,
    model_summary: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    output_dir: str | Path = "reports/figures",
    rolling_window: int = 20,
    periods_per_year: float | None = None,
) -> dict[str, Path]:
    """Generate the complete deterministic chart set and Markdown index."""
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(portfolio_returns)
    _validate_inputs(
        portfolio_returns,
        model_summary,
        regime_metrics,
        rolling_window=rolling_window,
        periods_per_year=periods_per_year,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    paths = {
        name: output_dir / filename for name, filename in FIGURE_FILENAMES.items()
    }
    plot_cumulative_returns(portfolio_returns, paths["cumulative_returns"])
    plot_drawdowns(portfolio_returns, paths["drawdowns"])
    plot_rolling_sharpe(
        portfolio_returns,
        paths["rolling_sharpe"],
        rolling_window=rolling_window,
        periods_per_year=periods_per_year,
    )
    plot_model_comparison(model_summary, paths["model_comparison"])
    plot_regime_performance(regime_metrics, paths["regime_performance"])
    return paths


def plot_cumulative_returns(data: pd.DataFrame, output_path: str | Path) -> None:
    """Plot chained net portfolio growth and the benchmark."""
    series = _portfolio_series(data)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for index, (model, group) in enumerate(series.groupby("model_name", sort=True)):
        axis.plot(
            group["date"],
            group["cumulative_net_return"],
            label=model,
            **_line_style(index),
        )
    benchmark = _benchmark_series(series)
    axis.plot(
        benchmark["date"],
        benchmark["benchmark_cumulative_return"],
        color="black",
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        label="benchmark",
    )
    _finish_time_plot(
        fig,
        axis,
        title="Cumulative Net Return",
        ylabel="Cumulative return",
        zero_line=True,
    )
    _save_figure(fig, output_path)


def plot_drawdowns(data: pd.DataFrame, output_path: str | Path) -> None:
    """Plot portfolio drawdown from each model's running net-return peak."""
    series = _portfolio_series(data)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for index, (model, group) in enumerate(series.groupby("model_name", sort=True)):
        axis.plot(
            group["date"],
            group["net_drawdown"],
            label=model,
            **_line_style(index),
        )
    _finish_time_plot(
        fig,
        axis,
        title="Net Portfolio Drawdown",
        ylabel="Drawdown",
        zero_line=True,
    )
    _save_figure(fig, output_path)


def plot_rolling_sharpe(
    data: pd.DataFrame,
    output_path: str | Path,
    *,
    rolling_window: int,
    periods_per_year: float,
) -> None:
    """Plot annualized rolling Sharpe using net portfolio returns."""
    series = _portfolio_series(data)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for index, (model, group) in enumerate(series.groupby("model_name", sort=True)):
        returns = group["net_return"]
        minimum = min(5, rolling_window)
        mean = returns.rolling(rolling_window, min_periods=minimum).mean()
        volatility = returns.rolling(rolling_window, min_periods=minimum).std(ddof=1)
        sharpe = (mean / volatility.replace(0, np.nan)) * np.sqrt(periods_per_year)
        axis.plot(
            group["date"],
            sharpe,
            label=model,
            **_line_style(index),
        )
    _finish_time_plot(
        fig,
        axis,
        title=f"Rolling {rolling_window}-Period Net Sharpe",
        ylabel="Annualized Sharpe",
        zero_line=True,
    )
    _save_figure(fig, output_path)


def plot_model_comparison(data: pd.DataFrame, output_path: str | Path) -> None:
    """Plot classification, ranking, and portfolio metrics as aligned facets."""
    metrics = [
        ("classification_roc_auc", "Mean ROC-AUC", 0.5),
        ("ranking_rank_information_coefficient", "Overall rank IC", 0.0),
        ("portfolio_net_sharpe", "Net Sharpe", 0.0),
    ]
    ordered = data.dropna(subset=["model_name"]).sort_values("model_name")
    fig, axes = plt.subplots(1, 3, figsize=(13, max(5, len(ordered) * 0.42)))
    positions = np.arange(len(ordered))
    colors = [_COLORS[index % len(_COLORS)] for index in range(len(ordered))]
    for axis, (column, title, reference) in zip(axes, metrics):
        values = pd.to_numeric(ordered[column], errors="coerce")
        axis.barh(positions, values, color=colors, alpha=0.85)
        axis.axvline(reference, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_yticks(positions, ordered["model_name"])
    for axis in axes[1:]:
        axis.set_yticks(positions, [])
    for axis in axes:
        axis.invert_yaxis()
    fig.suptitle("Model Comparison Metrics")
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_regime_performance(data: pd.DataFrame, output_path: str | Path) -> None:
    """Plot valid regime ROC-AUC values as one heatmap per regime family."""
    valid = data.loc[data["meets_minimum_rows"].astype(bool)].copy()
    regime_types = sorted(valid["regime_type"].dropna().astype(str).unique())
    fig, axes = plt.subplots(
        len(regime_types),
        1,
        figsize=(11, max(3.5, len(regime_types) * 3.5)),
        squeeze=False,
    )
    for axis, regime_type in zip(axes[:, 0], regime_types):
        subset = valid.loc[valid["regime_type"].eq(regime_type)]
        matrix = subset.pivot_table(
            index="model_name", columns="regime", values="roc_auc", aggfunc="first"
        ).sort_index()
        image = axis.imshow(matrix.to_numpy(float), aspect="auto", vmin=0.4, vmax=0.65)
        axis.set_title(regime_type.replace("_", " ").title())
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        for row in range(len(matrix.index)):
            for column in range(len(matrix.columns)):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    axis.text(column, row, f"{value:.3f}", ha="center", va="center")
        fig.colorbar(image, ax=axis, label="ROC-AUC", fraction=0.025, pad=0.02)
    fig.suptitle("Regime-Specific Classification Performance")
    fig.tight_layout()
    _save_figure(fig, output_path)


def render_chart_report(paths: dict[str, Path], *, report_path: str | Path) -> str:
    """Render relative figure links for inclusion in project reports."""
    report_path = Path(report_path)
    titles = {
        "cumulative_returns": "Cumulative Return",
        "drawdowns": "Drawdown",
        "rolling_sharpe": "Rolling Sharpe",
        "model_comparison": "Model Comparison Metrics",
        "regime_performance": "Regime-Specific Results",
    }
    lines = [
        "# Backtest and Model Charts",
        "",
        "Charts use aligned out-of-sample comparison results and transaction-cost-aware "
        "portfolio returns. They are research summaries, not evidence of future returns.",
        "",
    ]
    for name in FIGURE_FILENAMES:
        relative = Path(os.path.relpath(paths[name], start=report_path.parent))
        lines.extend([f"## {titles[name]}", "", f"![{titles[name]}]({relative.as_posix()})", ""])
    return "\n".join(lines)


def save_chart_report(markdown: str, output_path: str | Path) -> None:
    """Save a chart index that can be included in the final report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def _portfolio_series(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered = ordered.sort_values(["model_name", "split_id", "date"]).reset_index(
        drop=True
    )
    ordered["cumulative_net_return"] = ordered.groupby("model_name")[
        "net_return"
    ].transform(lambda values: (1 + values).cumprod() - 1)
    ordered["benchmark_cumulative_return"] = ordered.groupby("model_name")[
        "benchmark_return"
    ].transform(lambda values: (1 + values).cumprod() - 1)
    wealth = 1 + ordered["cumulative_net_return"]
    peaks = wealth.groupby(ordered["model_name"]).cummax()
    ordered["net_drawdown"] = wealth / peaks - 1
    return ordered


def _benchmark_series(data: pd.DataFrame) -> pd.DataFrame:
    first_model = sorted(data["model_name"].unique())[0]
    return data.loc[data["model_name"].eq(first_model)]


def _validate_inputs(
    portfolio_returns: pd.DataFrame,
    model_summary: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    *,
    rolling_window: int,
    periods_per_year: float,
) -> None:
    requirements = [
        (
            portfolio_returns,
            {"model_name", "split_id", "date", "net_return", "benchmark_return"},
            "Portfolio returns",
        ),
        (
            model_summary,
            {
                "model_name",
                "classification_roc_auc",
                "ranking_rank_information_coefficient",
                "portfolio_net_sharpe",
            },
            "Model summary",
        ),
        (
            regime_metrics,
            {
                "regime_type",
                "regime",
                "model_name",
                "meets_minimum_rows",
                "roc_auc",
            },
            "Regime metrics",
        ),
    ]
    for data, required, name in requirements:
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{name} are missing columns: " + ", ".join(sorted(missing)))
        if data.empty:
            raise ValueError(f"{name} must be non-empty.")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if not regime_metrics["meets_minimum_rows"].astype(bool).any():
        raise ValueError("At least one regime slice must meet the minimum row threshold.")


def _infer_periods_per_year(portfolio_returns: pd.DataFrame) -> float:
    if "rebalance_frequency" not in portfolio_returns.columns:
        raise ValueError(
            "Portfolio returns must include rebalance_frequency when "
            "periods_per_year is not provided."
        )
    frequencies = pd.to_numeric(
        portfolio_returns["rebalance_frequency"], errors="coerce"
    ).dropna().unique()
    if len(frequencies) != 1 or frequencies[0] <= 0:
        raise ValueError("Portfolio returns must have one positive rebalance_frequency.")
    return TRADING_DAYS_PER_YEAR / float(frequencies[0])


def _line_style(index: int) -> dict:
    return {
        "color": _COLORS[index % len(_COLORS)],
        "linestyle": _LINE_STYLES[(index // len(_COLORS)) % len(_LINE_STYLES)],
        "linewidth": 1.6,
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def _finish_time_plot(
    fig,
    axis,
    *,
    title: str,
    ylabel: str,
    zero_line: bool,
) -> None:
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if zero_line:
        axis.axhline(0, color="black", linewidth=0.8)
    axis.legend(ncol=2, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()


def _save_figure(fig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"Software": "market-qml-predictor"},
    )
    plt.close(fig)
