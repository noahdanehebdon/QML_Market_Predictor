"""Daily cross-sectional signal report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from market_qml.models.preprocessing import FittedPreprocessor, transform_features


DISCLAIMER = (
    "This report is for research and educational purposes only and is not "
    "financial advice."
)
SIGNAL_COLUMNS = [
    "date",
    "rank",
    "symbol",
    "predicted_outperformance_probability",
    "is_benchmark",
    "model_name",
]


@dataclass(frozen=True)
class DailySignalReport:
    """Ranked latest-date signals and their rendered report."""

    signals: pd.DataFrame
    markdown: str
    signal_date: pd.Timestamp
    benchmark: str
    model_name: str


def load_model(path: str | Path) -> Any:
    """Load a trusted local model artifact with probability predictions."""
    path = Path(path)
    with path.open("rb") as model_file:
        model = pickle.load(model_file)

    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Model artifact must provide a callable predict_proba method.")
    if not hasattr(model, "classes_"):
        raise TypeError("Model artifact must provide fitted classes_ metadata.")
    return model


def build_daily_signal_report(
    features: pd.DataFrame,
    *,
    model: Any,
    preprocessor: FittedPreprocessor,
    model_name: str = "logistic_regression",
    benchmark: str = "SPY",
    top_n: int = 5,
    bottom_n: int = 5,
) -> DailySignalReport:
    """Score and rank the latest feature cross-section."""
    if top_n < 1 or bottom_n < 1:
        raise ValueError("top_n and bottom_n must both be positive.")

    latest = _latest_feature_cross_section(features)
    benchmark = benchmark.strip().upper()
    if benchmark not in set(latest["symbol"]):
        raise ValueError(f"Benchmark '{benchmark}' is absent from the latest feature date.")

    transformed = transform_features(latest, preprocessor)
    positive_class_index = _positive_class_index(model)
    probabilities = model.predict_proba(transformed)[:, positive_class_index]

    signals = latest[["date", "symbol"]].copy()
    signals["predicted_outperformance_probability"] = pd.to_numeric(
        pd.Series(probabilities, index=signals.index), errors="coerce"
    )
    if signals["predicted_outperformance_probability"].isna().any():
        raise ValueError("Model produced missing or non-numeric signal scores.")

    signals["is_benchmark"] = signals["symbol"].eq(benchmark)
    signals["model_name"] = model_name
    signals = signals.sort_values(
        ["predicted_outperformance_probability", "symbol"],
        ascending=[False, True],
    ).reset_index(drop=True)
    signals.insert(1, "rank", range(1, len(signals) + 1))
    signals = signals[SIGNAL_COLUMNS]

    signal_date = pd.Timestamp(signals["date"].iloc[0]).normalize()
    markdown = render_daily_signal_report(
        signals,
        signal_date=signal_date,
        benchmark=benchmark,
        model_name=model_name,
        top_n=top_n,
        bottom_n=bottom_n,
    )
    return DailySignalReport(
        signals=signals,
        markdown=markdown,
        signal_date=signal_date,
        benchmark=benchmark,
        model_name=model_name,
    )


def render_daily_signal_report(
    signals: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    benchmark: str,
    model_name: str,
    top_n: int,
    bottom_n: int,
) -> str:
    """Render ranked signals as a concise Markdown research report."""
    benchmark_row = signals.loc[signals["symbol"].eq(benchmark)].iloc[0]
    equities = signals.loc[~signals["is_benchmark"]]
    top = equities.head(top_n)
    bottom = equities.tail(bottom_n).sort_values("rank", ascending=False)

    lines = [
        "# Daily Equity Signal Report",
        "",
        f"**Signal date:** {signal_date.date().isoformat()}",
        f"**Model:** `{model_name}`",
        "",
        f"> **Disclaimer:** {DISCLAIMER}",
        "",
        "## Benchmark Context",
        "",
        f"{benchmark} ranks **{int(benchmark_row['rank'])} of {len(signals)}** with a "
        "predicted outperformance probability of "
        f"**{float(benchmark_row['predicted_outperformance_probability']):.2%}**.",
        "",
        "Scores are model probabilities, not expected returns or trading recommendations.",
        "",
        f"## Top {min(top_n, len(top))} Names",
        "",
        _render_signal_table(top),
        "",
        f"## Bottom {min(bottom_n, len(bottom))} Names",
        "",
        _render_signal_table(bottom),
        "",
        "## Method Note",
        "",
        "The report applies the saved train-fitted preprocessor and model to the latest "
        "available feature cross-section, then ranks symbols by predicted probability of "
        "benchmark outperformance. Realized forward returns are not used to create the report.",
        "",
    ]
    return "\n".join(lines)


def save_daily_signal_report(
    report: DailySignalReport,
    *,
    markdown_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Save Markdown and machine-readable CSV report artifacts."""
    markdown_path = Path(markdown_path)
    csv_path = Path(csv_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(report.markdown, encoding="utf-8")
    report.signals.to_csv(csv_path, index=False)


def _latest_feature_cross_section(features: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(
            "Feature table is missing required columns: " + ", ".join(sorted(missing))
        )
    if features.empty:
        raise ValueError("Feature table is empty.")

    latest = features.copy()
    latest["symbol"] = latest["symbol"].astype(str).str.strip().str.upper()
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce").dt.normalize()
    if latest["date"].isna().any():
        raise ValueError("Feature table contains invalid dates.")
    latest = latest.loc[latest["date"].eq(latest["date"].max())].copy()
    if latest["symbol"].duplicated().any():
        raise ValueError("Latest feature date contains duplicate symbols.")
    return latest.sort_values("symbol").reset_index(drop=True)


def _positive_class_index(model: Any) -> int:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model classes do not include positive class 1.")
    return classes.index(1)


def _render_signal_table(signals: pd.DataFrame) -> str:
    rows = ["| Rank | Symbol | Probability |", "|---:|:---|---:|"]
    for row in signals.itertuples(index=False):
        rows.append(
            f"| {row.rank} | {row.symbol} | "
            f"{row.predicted_outperformance_probability:.2%} |"
        )
    return "\n".join(rows)
