"""Definitive two-lane classical-versus-quantum decision reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from market_qml.backtest.portfolio import (
    run_portfolio_backtest,
    summarize_portfolio_risk,
)
from market_qml.backtest.ranking_metrics import evaluate_ranking_metrics
from market_qml.backtest.validation import (
    paired_model_comparisons,
    prediction_date_block_metrics,
)

QML_MODELS = {"vqc", "qcnn", "qsvm", "qsvm_tuned"}


@dataclass(frozen=True)
class DefinitiveComparisonResult:
    aggregate_metrics: pd.DataFrame
    paired_comparisons: pd.DataFrame
    resource_summary: pd.DataFrame
    portfolio_summary: pd.DataFrame
    conclusion: dict[str, object]


def build_definitive_comparison(
    equal_input_predictions: pd.DataFrame,
    best_available_predictions: pd.DataFrame,
    *,
    equal_resources: pd.DataFrame | None = None,
    best_resources: pd.DataFrame | None = None,
    locked_test_manifest: dict[str, object] | None = None,
    bootstrap_iterations: int = 2000,
    return_horizon_days: int,
    rebalance_frequency: int | None = None,
) -> DefinitiveComparisonResult:
    """Compare equal-input and best-available lanes without overstating evidence."""
    _validate_equal_input(equal_input_predictions)
    lane_frames = {
        "equal_input": equal_input_predictions,
        "best_available": best_available_predictions,
    }
    aggregate_rows, paired_frames, portfolio_frames = [], [], []
    lane_leaders = {}
    for lane, predictions in lane_frames.items():
        metrics = _aggregate_metrics(predictions)
        metrics.insert(0, "lane", lane)
        aggregate_rows.append(metrics)
        leader = _strongest_classical(metrics)
        lane_leaders[lane] = leader
        blocks = prediction_date_block_metrics(_binary_predictions(predictions))
        classification_leader = _strongest_classical(
            metrics.loc[metrics["roc_auc"].notna()]
        )
        if not blocks.empty and classification_leader in set(blocks["model_name"]):
            paired = paired_model_comparisons(
                blocks,
                metric="roc_auc",
                baseline_model=classification_leader,
                bootstrap_iterations=bootstrap_iterations,
                practical_threshold=0.02,
            )
            paired.insert(0, "lane", lane)
            paired_frames.append(paired)
        portfolio = summarize_portfolio_risk(
            run_portfolio_backtest(
                predictions,
                top_fraction=0.1,
                return_horizon_days=return_horizon_days,
                rebalance_frequency=rebalance_frequency or return_horizon_days,
                sector_neutral=True,
            )
        )
        portfolio = portfolio.loc[portfolio["scope"].eq("overall")].copy()
        portfolio.insert(0, "lane", lane)
        portfolio_frames.append(portfolio)
    aggregate = pd.concat(aggregate_rows, ignore_index=True)
    paired = (
        pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    )
    portfolio = pd.concat(portfolio_frames, ignore_index=True)
    resources = _resource_summary(equal_resources, best_resources)
    locked = bool((locked_test_manifest or {}).get("locked_test_accessed", False))
    defensible = _defensible_qml_claim(aggregate, paired, portfolio, locked)
    strongest = lane_leaders["best_available"]
    conclusion = {
        "equal_input_leader": _overall_leader(aggregate, "equal_input"),
        "best_available_leader": _overall_leader(aggregate, "best_available"),
        "strongest_classical_system": strongest,
        "locked_test_accessed": locked,
        "quantum_advantage_demonstrated": defensible,
        "decision": (
            "A statistically and practically defensible locked-test quantum advantage was found."
            if defensible
            else "No statistically and practically defensible quantum advantage is demonstrated; the strongest classical system remains the default."
        ),
    }
    return DefinitiveComparisonResult(
        aggregate, paired, resources, portfolio, conclusion
    )


def save_definitive_comparison(result, private_output_dir, public_output_dir):
    """Keep detailed tables private and publish aggregate-only conclusions."""
    private, public = Path(private_output_dir), Path(public_output_dir)
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    for name in [
        "aggregate_metrics",
        "paired_comparisons",
        "resource_summary",
        "portfolio_summary",
    ]:
        getattr(result, name).to_parquet(private / f"{name}.parquet", index=False)
    (private / "conclusion.json").write_text(
        json.dumps(result.conclusion, indent=2) + "\n", encoding="utf-8"
    )
    public_summary = {
        "comparison_lanes": ["equal_input", "best_available"],
        **result.conclusion,
    }
    (public / "definitive_summary.json").write_text(
        json.dumps(public_summary, indent=2) + "\n", encoding="utf-8"
    )
    (public / "definitive_summary.md").write_text(
        _render_summary(public_summary), encoding="utf-8"
    )


def _aggregate_metrics(predictions):
    rows = []
    ranking = (
        evaluate_ranking_metrics(predictions)
        .query("scope == 'overall'")
        .set_index("model_name")
    )
    for model, frame in predictions.groupby("model_name"):
        binary = set(frame["y_true"].unique()) <= {0, 1}
        rows.append(
            {
                "model_name": model,
                "model_family": "qml" if model in QML_MODELS else "classical",
                "roc_auc": roc_auc_score(frame["y_true"], frame["y_score"])
                if binary and frame["y_true"].nunique() > 1
                else np.nan,
                "log_loss": log_loss(
                    frame["y_true"],
                    np.clip(frame["y_score"], 1e-6, 1 - 1e-6),
                    labels=[0, 1],
                )
                if binary
                else np.nan,
                "brier_score": brier_score_loss(frame["y_true"], frame["y_score"])
                if binary
                else np.nan,
                "rank_ic": ranking.loc[model, "rank_information_coefficient"],
            }
        )
    return pd.DataFrame(rows)


def _validate_equal_input(predictions):
    required = {"model_name", "split_id", "symbol", "date"}
    if required - set(predictions):
        raise ValueError("Equal-input predictions are missing required columns.")
    keys = (
        predictions.assign(
            _key=predictions["symbol"].astype(str)
            + "|"
            + predictions["date"].astype(str)
        )
        .groupby(["split_id", "model_name"])["_key"]
        .apply(lambda values: tuple(sorted(values)))
    )
    if not keys.groupby(level=0).nunique().eq(1).all():
        raise ValueError("Equal-input models do not contain identical outer rows.")


def _binary_predictions(predictions):
    frames = [
        frame
        for _, frame in predictions.groupby("model_name")
        if set(frame["y_true"].unique()) <= {0, 1}
    ]
    return (
        pd.concat(frames, ignore_index=True) if frames else predictions.iloc[0:0].copy()
    )


def _strongest_classical(metrics):
    classical = metrics.loc[metrics["model_family"].eq("classical")].copy()
    if classical.empty:
        raise ValueError("Each comparison lane requires at least one classical model.")
    classical["score"] = classical["roc_auc"].fillna(0) + classical["rank_ic"].fillna(0)
    return str(classical.sort_values("score", ascending=False).iloc[0]["model_name"])


def _overall_leader(metrics, lane):
    selected = metrics.loc[metrics["lane"].eq(lane)].copy()
    selected["score"] = selected["roc_auc"].fillna(0) + selected["rank_ic"].fillna(0)
    return str(selected.sort_values("score", ascending=False).iloc[0]["model_name"])


def _defensible_qml_claim(aggregate, paired, portfolio, locked):
    """Require concordant locked-test evidence across three outcome families."""
    if not locked or paired.empty:
        return False
    qml = paired.loc[
        paired["candidate_model"].isin(QML_MODELS)
        & paired["decision"].eq("material_difference")
        & paired["mean_difference"].gt(0)
    ]
    lane_metrics = aggregate.loc[aggregate["lane"].eq("best_available")].set_index(
        "model_name"
    )
    lane_portfolio = portfolio.loc[portfolio["lane"].eq("best_available")].set_index(
        "model_name"
    )
    for row in qml.itertuples():
        candidate = row.candidate_model
        baseline = row.baseline_model
        if not {candidate, baseline} <= set(lane_metrics.index):
            continue
        if not {candidate, baseline} <= set(lane_portfolio.index):
            continue
        rank_gain = (
            lane_metrics.loc[candidate, "rank_ic"]
            - lane_metrics.loc[baseline, "rank_ic"]
        )
        portfolio_gain = (
            lane_portfolio.loc[candidate, "cumulative_net_excess_return"]
            - lane_portfolio.loc[baseline, "cumulative_net_excess_return"]
        )
        if rank_gain >= 0.01 and portfolio_gain > 0:
            return True
    return False


def _resource_summary(equal, best):
    frames = []
    for lane, frame in [("equal_input", equal), ("best_available", best)]:
        if frame is not None and not frame.empty:
            summary = frame.groupby("model_name", as_index=False).agg(
                runtime_seconds=("runtime_seconds", "mean"),
                peak_memory_mb=("peak_memory_mb", "mean"),
            )
            summary.insert(0, "lane", lane)
            frames.append(summary)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["lane", "model_name", "runtime_seconds", "peak_memory_mb"]
        )
    )


def _render_summary(summary):
    return "\n".join(
        [
            "# Definitive classical-versus-quantum comparison",
            "",
            "Two conclusions are reported separately: equal eight-feature inputs and best available validated inputs.",
            "",
            f"**Decision:** {summary['decision']}",
            "",
            f"Locked test accessed: `{summary['locked_test_accessed']}`.",
            "",
            "No model is labeled superior from a single metric, fold, or portfolio realization.",
            "",
        ]
    )
