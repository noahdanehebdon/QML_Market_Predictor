"""Chronological calibration and leakage-safe prediction ensembles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


KEYS = ["symbol", "date", "split_id", "y_true", "forward_return", "forward_excess_return"]


@dataclass(frozen=True)
class EnsembleResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    sensitivity: pd.DataFrame


def eligible_regime_weights(regimes: pd.DataFrame, *, regime_column: str, min_rows: int = 100) -> pd.DataFrame:
    """Report regimes eligible for conditioning; small samples remain pooled."""
    if regime_column not in regimes:
        raise ValueError(f"Regime table is missing column: {regime_column}")
    counts = regimes.groupby(regime_column, dropna=True).size().rename("rows").reset_index()
    counts["minimum_rows"] = min_rows
    counts["eligible_for_conditioning"] = counts["rows"] >= min_rows
    return counts


def build_chronological_ensembles(predictions: pd.DataFrame, *, min_history_rows=20, turnover_penalty=0.01, instability_penalty=0.1) -> EnsembleResult:
    """Learn calibration and blend weights using earlier outer folds only."""
    base = predictions.loc[~predictions["model_name"].str.contains("ensemble")].copy()
    binary_models = {
        name for name, frame in base.groupby("model_name")
        if set(pd.to_numeric(frame["y_true"]).dropna().unique()) <= {0, 1}
    }
    all_predictions, diagnostics, sensitivities = [], [], []
    for task, models in [("classification", binary_models), ("ranking", set(base["model_name"]) - binary_models)]:
        if len(models) < 2:
            continue
        task_data = base.loc[base["model_name"].isin(models)]
        for split_id in sorted(task_data["split_id"].unique()):
            history = task_data.loc[task_data["split_id"] < split_id]
            current = task_data.loc[task_data["split_id"] == split_id]
            history_wide = _wide(history, sorted(models))
            current_wide = _wide(current, sorted(models))
            if current_wide.empty:
                continue
            model_columns = [m for m in sorted(models) if m in current_wide and current_wide[m].notna().all()]
            if len(model_columns) < 2:
                continue
            calibrated_current = current_wide[model_columns].copy()
            calibrated_history = history_wide.reindex(columns=history_wide.columns.union(model_columns))
            if task == "classification":
                for model in model_columns:
                    calibrated_current[model], calibrated_history[model] = _calibrate(
                        history_wide, current_wide, model, min_history_rows
                    )
            learned, source = _learn_weights(calibrated_history, model_columns, task, min_history_rows, turnover_penalty, instability_penalty)
            methods = {
                "simple_average_ensemble": calibrated_current[model_columns].mean(axis=1),
                "rank_average_ensemble": calibrated_current.groupby(current_wide["date"])[model_columns].rank(pct=True).mean(axis=1),
                "constrained_stack_ensemble": calibrated_current[model_columns].to_numpy() @ learned,
            }
            for method, scores in methods.items():
                name = f"{task}_{method}"
                all_predictions.append(_prediction_frame(current_wide, scores, name))
            diagnostics.extend(
                {"task": task, "split_id": int(split_id), "model_name": model, "weight": float(weight), "weight_source": source, "turnover_penalty": turnover_penalty, "instability_penalty": instability_penalty}
                for model, weight in zip(model_columns, learned)
            )
            full_score = calibrated_current[model_columns].to_numpy() @ learned
            for removed in model_columns:
                kept = [m for m in model_columns if m != removed]
                removal = calibrated_current[kept].mean(axis=1)
                sensitivities.append({"task": task, "split_id": int(split_id), "removed_model": removed, "score_correlation": float(pd.Series(full_score).corr(pd.Series(removal), method="spearman")), "mean_absolute_change": float(np.mean(np.abs(full_score - removal)))})
    combined = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame(columns=REQUIRED_PREDICTION_COLUMNS)
    return EnsembleResult(combined, pd.DataFrame(diagnostics), pd.DataFrame(sensitivities))


def _wide(data, models):
    if data.empty:
        return pd.DataFrame(columns=KEYS + models)
    return data.pivot(index=KEYS, columns="model_name", values="y_score").reset_index()


def _calibrate(history, current, model, minimum):
    valid = history.dropna(subset=[model, "y_true"])
    if len(valid) < minimum or valid["y_true"].nunique() < 2:
        return current[model].clip(1e-6, 1 - 1e-6), history.get(model, pd.Series(index=history.index, dtype=float)).clip(1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(random_state=42).fit(valid[[model]], valid["y_true"].astype(int))
    current_scores = calibrator.predict_proba(current[[model]])[:, 1]
    history_scores = pd.Series(np.nan, index=history.index)
    mask = history[model].notna()
    history_scores.loc[mask] = calibrator.predict_proba(history.loc[mask, [model]])[:, 1]
    return pd.Series(current_scores, index=current.index), history_scores


def _learn_weights(history, models, task, minimum, turnover_penalty, instability_penalty):
    usable = history.dropna(subset=models + ["y_true"])
    equal = np.full(len(models), 1 / len(models))
    if len(usable) < minimum:
        return equal, "equal_weight_insufficient_history"
    X, y = usable[models].to_numpy(), usable["y_true"].to_numpy()
    def objective(weights):
        score = X @ weights
        if task == "classification":
            base = log_loss(y.astype(int), np.clip(score, 1e-6, 1 - 1e-6), labels=[0, 1])
            fold_values = usable.assign(_score=score).groupby("split_id").apply(lambda f: log_loss(f["y_true"].astype(int), np.clip(f["_score"], 1e-6, 1 - 1e-6), labels=[0, 1]), include_groups=False)
        else:
            fold_values = usable.assign(_score=score).groupby("split_id").apply(lambda f: -f["_score"].corr(f["forward_excess_return"], method="spearman"), include_groups=False).dropna()
            base = float(fold_values.mean()) if len(fold_values) else 0.0
        ordered = usable.assign(_score=score).sort_values(["symbol", "date"])
        turnover_proxy = ordered.groupby("symbol")["_score"].diff().abs().mean()
        return base + instability_penalty * float(fold_values.std(ddof=0)) + turnover_penalty * float(turnover_proxy if pd.notna(turnover_proxy) else 0)
    result = minimize(objective, equal, method="SLSQP", bounds=[(0, 1)] * len(models), constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return (result.x if result.success else equal), ("chronological_constrained" if result.success else "equal_weight_optimizer_failure")


def _prediction_frame(wide, scores, name):
    result = wide[KEYS].copy()
    result["y_score"] = np.asarray(scores)
    result["model_name"] = name
    return result[REQUIRED_PREDICTION_COLUMNS]
