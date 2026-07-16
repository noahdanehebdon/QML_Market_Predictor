"""Regime-specific analysis of aligned QML and classical predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


REGIME_COLUMNS = ["volatility_regime", "rate_regime", "yield_curve_regime"]
QML_MODELS = {"qcnn", "vqc", "qsvm", "qsvm_tuned"}


@dataclass(frozen=True)
class RegimeAnalysisResult:
    joined_predictions: pd.DataFrame
    metrics: pd.DataFrame
    model_differences: pd.DataFrame


def analyze_predictions_by_regime(
    predictions: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    minimum_rows: int = 50,
) -> RegimeAnalysisResult:
    """Compute comparable classification and ranking metrics within each regime."""
    _validate_inputs(predictions, regimes, minimum_rows)
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.normalize()
    regime_table = regimes[["date", *REGIME_COLUMNS]].copy()
    regime_table["date"] = pd.to_datetime(regime_table["date"], errors="coerce").dt.normalize()
    if regime_table["date"].duplicated().any():
        raise ValueError("Regime table must contain one row per date")
    joined = predictions.merge(regime_table, on="date", how="left", validate="many_to_one")

    rows = []
    for regime_type in REGIME_COLUMNS:
        available = joined.dropna(subset=[regime_type])
        for (regime, model), group in available.groupby([regime_type, "model_name"], sort=True):
            rows.append(_metric_row(group, regime_type, str(regime), str(model), minimum_rows))
    metrics = pd.DataFrame(rows)
    differences = _qcnn_differences(metrics)
    return RegimeAnalysisResult(joined, metrics, differences)


def save_regime_analysis(result: RegimeAnalysisResult, output_dir: str | Path) -> dict[str, Path]:
    """Save auditable tables and a Markdown interpretation report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "joined_predictions": output / "regime_predictions.parquet",
        "metrics": output / "regime_metrics.parquet",
        "model_differences": output / "qcnn_model_differences.parquet",
        "report": output / "regime_analysis.md",
    }
    result.joined_predictions.to_parquet(paths["joined_predictions"], index=False)
    result.metrics.to_parquet(paths["metrics"], index=False)
    result.model_differences.to_parquet(paths["model_differences"], index=False)
    paths["report"].write_text(render_regime_report(result), encoding="utf-8")
    return paths


def render_regime_report(result: RegimeAnalysisResult) -> str:
    valid = result.metrics[result.metrics["meets_minimum_rows"]].copy()
    lines = [
        "# QML performance by market regime", "",
        "All slices reuse the aligned out-of-sample prediction rows. Regime labels are date-keyed and computed only from information available through each date.", "",
    ]
    for regime_type in REGIME_COLUMNS:
        lines.extend([f"## {regime_type}", ""])
        group = valid[valid.regime_type == regime_type]
        if group.empty:
            lines.extend(["No slice met the minimum row requirement.", ""])
            continue
        display = group[["regime", "model_name", "rows", "splits", "roc_auc", "accuracy", "brier_score",
                         "rank_information_coefficient", "top_decile_return"]]
        lines.extend([_markdown_table(display), ""])
        for regime, slice_metrics in group.groupby("regime", sort=True):
            roc = slice_metrics.dropna(subset=["roc_auc"]).sort_values("roc_auc", ascending=False)
            rank = slice_metrics.dropna(subset=["rank_information_coefficient"]).sort_values(
                "rank_information_coefficient", ascending=False
            )
            if not roc.empty:
                lines.append(f"- **{regime}:** ROC-AUC leader is `{roc.iloc[0].model_name}` ({roc.iloc[0].roc_auc:.4f}).")
            if not rank.empty:
                lines.append(f"  Rank-IC leader is `{rank.iloc[0].model_name}` ({rank.iloc[0].rank_information_coefficient:.4f}).")
        lines.append("")
    qcnn = valid[valid.model_name == "qcnn"]
    if not qcnn.empty:
        best = qcnn.dropna(subset=["roc_auc"]).sort_values("roc_auc", ascending=False)
        worst = qcnn.dropna(subset=["roc_auc"]).sort_values("roc_auc")
        lines.extend(["## QCNN pattern", ""])
        if not best.empty:
            lines.append(f"QCNN's strongest ROC-AUC slice is **{best.iloc[0].regime}** ({best.iloc[0].roc_auc:.4f}); its weakest is **{worst.iloc[0].regime}** ({worst.iloc[0].roc_auc:.4f}).")
        lines.extend(["", "Treat regime gaps as descriptive unless they persist across multiple chronological splits and adequate sample sizes.", ""])
    return "\n".join(lines)


def _metric_row(group, regime_type, regime, model_name, minimum_rows):
    y = pd.to_numeric(group["y_true"], errors="coerce").astype(int)
    score = pd.to_numeric(group["y_score"], errors="coerce").clip(1e-12, 1 - 1e-12)
    returns = pd.to_numeric(group["forward_excess_return"], errors="coerce")
    enough = len(group) >= minimum_rows
    auc = roc_auc_score(y, score) if enough and y.nunique() == 2 else np.nan
    rank_ic = score.corr(returns, method="spearman") if enough and score.nunique() > 1 and returns.nunique() > 1 else np.nan
    top_count = max(1, int(np.ceil(len(group) * 0.1)))
    top_return = group.assign(_score=score.to_numpy()).nlargest(top_count, "_score")["forward_excess_return"].mean()
    return {
        "regime_type": regime_type, "regime": regime, "model_name": model_name,
        "model_family": "qml" if model_name in QML_MODELS else "classical",
        "rows": len(group), "splits": group["split_id"].nunique(),
        "positive_rate": y.mean(), "meets_minimum_rows": enough,
        "accuracy": accuracy_score(y, score >= 0.5) if enough else np.nan,
        "roc_auc": auc,
        "log_loss": log_loss(y, score, labels=[0, 1]) if enough else np.nan,
        "brier_score": brier_score_loss(y, score) if enough else np.nan,
        "rank_information_coefficient": rank_ic,
        "top_decile_return": top_return if enough else np.nan,
    }


def _qcnn_differences(metrics):
    qcnn = metrics[metrics.model_name == "qcnn"]
    others = metrics[metrics.model_name != "qcnn"]
    merged = qcnn.merge(others, on=["regime_type", "regime"], suffixes=("_qcnn", "_comparison"))
    for metric in ["roc_auc", "accuracy", "rank_information_coefficient", "top_decile_return"]:
        merged[f"{metric}_difference"] = merged[f"{metric}_qcnn"] - merged[f"{metric}_comparison"]
    return merged


def _markdown_table(data):
    headers = list(data.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data.itertuples(index=False, name=None):
        values = ["NA" if pd.isna(value) else f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _validate_inputs(predictions, regimes, minimum_rows):
    missing_predictions = set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_predictions:
        raise ValueError("Predictions are missing columns: " + ", ".join(sorted(missing_predictions)))
    missing_regimes = {"date", *REGIME_COLUMNS} - set(regimes.columns)
    if missing_regimes:
        raise ValueError("Regimes are missing columns: " + ", ".join(sorted(missing_regimes)))
    if predictions.empty or regimes.empty:
        raise ValueError("Predictions and regimes must be non-empty")
    if minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive")
