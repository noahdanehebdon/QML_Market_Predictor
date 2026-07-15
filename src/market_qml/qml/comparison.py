"""Leakage-safe, apples-to-apples QML and classical model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.svm import SVC

from market_qml.models.predictions import build_prediction_table
from market_qml.qml.interface import QMLDataset, QMLTrainValidation, build_qml_train_validation
from market_qml.qml.qcnn import train_qcnn
from market_qml.qml.qsvm import QuantumKernelSVM
from market_qml.qml.interface import QMLModelConfig
from market_qml.qml.vqc import train_vqc


DEFAULT_FEATURE_SELECTIONS = {
    "broad_market": [
        "raw_price_pca_00", "returns_momentum_pca_00", "volatility_pca_00",
        "volume_liquidity_pca_00", "benchmark_relative_pca_00", "macro_pca_00",
        "fundamentals_pca_00", "other_pca_00",
    ],
    "market_dynamics": [
        "raw_price_pca_00", "raw_price_pca_01", "returns_momentum_pca_00",
        "returns_momentum_pca_01", "returns_momentum_pca_02", "volatility_pca_00",
        "volume_liquidity_pca_00", "benchmark_relative_pca_00",
    ],
    "benchmark_macro": [
        "benchmark_relative_pca_00", "benchmark_relative_pca_01",
        "benchmark_relative_pca_02", "macro_pca_00", "macro_pca_01",
        "returns_momentum_pca_00", "volatility_pca_00", "fundamentals_pca_00",
    ],
}
DEFAULT_SELECTED_FEATURES = [f"selected_feature_{index:02d}" for index in range(8)]


@dataclass(frozen=True)
class ComparisonConfig:
    train_rows: int = 128
    validation_rows: int = 128
    random_state: int = 42
    vqc_iterations: int = 10
    qcnn_iterations: int = 10
    qsvm_c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    qsvm_repetitions: tuple[int, ...] = (1, 2, 3)
    feature_selection_names: tuple[str, ...] = ("classical_selected",)
    interaction_scales: tuple[float, ...] = (0.0, 0.5, 1.0)
    bootstrap_iterations: int = 2000


@dataclass(frozen=True)
class ComparisonResult:
    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    resource_usage: pd.DataFrame
    qsvm_tuning_trials: pd.DataFrame
    qsvm_selected_configs: pd.DataFrame
    sample_manifest: pd.DataFrame


def run_model_comparison(data: pd.DataFrame, config: ComparisonConfig = ComparisonConfig()) -> ComparisonResult:
    """Compare all models on identical outer rows and tune QSVM on train rows only."""
    _validate_inputs(data, config)
    predictions, resources, trials, selected, manifests = [], [], [], [], []
    for split_id in sorted(data["split_id"].unique()):
        sampled = _sample_split(data, int(split_id), config)
        manifests.append(_sample_manifest(sampled, int(split_id)))
        primary_columns = _feature_columns(sampled, config.feature_selection_names[0])
        primary = build_qml_train_validation(
            sampled, split_id=int(split_id), feature_columns=primary_columns
        )
        chosen, split_trials = _select_qsvm(sampled, int(split_id), config)
        trials.append(split_trials)
        selected.append(chosen)

        runners = {
            "vqc": lambda: train_vqc(primary, max_iter=config.vqc_iterations,
                                      random_state=config.random_state + int(split_id)).predictions,
            "qcnn": lambda: train_qcnn(primary, max_iter=config.qcnn_iterations,
                                        learning_rate=0.05, initialization_scale=0.1,
                                        random_state=config.random_state + int(split_id)).predictions,
            "qsvm": lambda: _qsvm_predictions(primary, 1.0, 2, 0.0, "qsvm", config.random_state),
            "linear_svm": lambda: _classical_predictions(primary, "linear", "linear_svm", config.random_state),
            "rbf_svm": lambda: _classical_predictions(primary, "rbf", "rbf_svm", config.random_state),
        }
        tuned_data = build_qml_train_validation(
            sampled, split_id=int(split_id), feature_columns=_feature_columns(sampled, chosen["feature_selection"])
        )
        runners["qsvm_tuned"] = lambda d=tuned_data, c=chosen: _qsvm_predictions(
            d, float(c["C"]), int(c["repetitions"]), float(c["interaction_scale"]),
            "qsvm_tuned", config.random_state
        )
        for model_name, runner in runners.items():
            frame, seconds, peak_mb = _measure(runner)
            predictions.append(frame)
            resources.append({"model_name": model_name, "split_id": int(split_id),
                              "runtime_seconds": seconds, "peak_memory_mb": peak_mb,
                              **frame.attrs})

    prediction_table = pd.concat(predictions, ignore_index=True)
    split_metrics = _split_metrics(prediction_table)
    return ComparisonResult(
        predictions=prediction_table,
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_split_metrics(split_metrics, config.bootstrap_iterations, config.random_state),
        resource_usage=pd.DataFrame(resources),
        qsvm_tuning_trials=pd.concat(trials, ignore_index=True),
        qsvm_selected_configs=pd.DataFrame(selected),
        sample_manifest=pd.concat(manifests, ignore_index=True),
    )


def aggregate_split_metrics(metrics: pd.DataFrame, bootstrap_iterations: int = 2000,
                            random_state: int = 42) -> pd.DataFrame:
    """Aggregate across chronological splits with split-bootstrap intervals."""
    rng = np.random.default_rng(random_state)
    rows = []
    metric_columns = [c for c in metrics.columns if c not in {"model_name", "split_id"}]
    for model_name, group in metrics.groupby("model_name", sort=True):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            boot = np.asarray([rng.choice(values, len(values), replace=True).mean()
                               for _ in range(bootstrap_iterations)])
            rows.append({"model_name": model_name, "metric": metric, "splits": len(values),
                         "mean": values.mean(), "median": np.median(values),
                         "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                         "ci_lower": np.quantile(boot, 0.025), "ci_upper": np.quantile(boot, 0.975)})
    return pd.DataFrame(rows)


def save_comparison_result(result: ComparisonResult, output_dir: str | Path) -> dict[str, Path]:
    """Persist auditable tables and a concise decision report."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    tables = {
        "predictions": result.predictions, "split_metrics": result.split_metrics,
        "aggregate_metrics": result.aggregate_metrics, "resource_usage": result.resource_usage,
        "qsvm_tuning_trials": result.qsvm_tuning_trials,
        "qsvm_selected_configs": result.qsvm_selected_configs,
        "sample_manifest": result.sample_manifest,
    }
    paths = {}
    for name, table in tables.items():
        paths[name] = output / f"{name}.parquet"; table.to_parquet(paths[name], index=False)
    paths["report"] = output / "comparison_report.md"
    paths["report"].write_text(render_comparison_report(result), encoding="utf-8")
    return paths


def render_comparison_report(result: ComparisonResult) -> str:
    pivot = result.aggregate_metrics.pivot(index="model_name", columns="metric", values="mean")
    ranked = pivot.sort_values("roc_auc", ascending=False)
    best = ranked.index[0]
    qsvm_auc = float(pivot.loc["qsvm_tuned", "roc_auc"])
    classical_auc = float(pivot.loc[["linear_svm", "rbf_svm"], "roc_auc"].max())
    if classical_auc > qsvm_auc + 0.02:
        decision = "Classical controls are currently stronger; keep them as the production baseline and redesign the quantum kernel before wider QSVM tuning."
    elif qsvm_auc <= 0.52:
        decision = "The tuned QSVM remains near random ranking performance; redesign the quantum kernel before spending on a larger hyperparameter search."
    else:
        decision = "The QSVM is competitive enough to justify a deeper, still training-only tuning study."
    display = ranked[["accuracy", "roc_auc", "log_loss", "brier_score", "information_coefficient", "top_decile_return"]]
    headers = ["model"] + list(display.columns)
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    table.extend("| " + " | ".join([str(index)] + [f"{value:.4f}" for value in row]) + " |"
                 for index, row in display.iterrows())
    return "\n".join([
        "# QML model comparison", "", "All models used identical outer validation rows. QSVM choices were made only from an inner chronological portion of each training window.", "",
        *table, "", f"Best mean ROC-AUC: **{best}**.", "", f"Decision: {decision}", "",
        "Uncertainty is recorded in `aggregate_metrics.parquet`; runtime, peak traced memory, selected configurations, tuning trials, and exact sampled-row hashes are retained beside this report.",
    ])


def _select_qsvm(sampled: pd.DataFrame, split_id: int, config: ComparisonConfig):
    train = sampled[sampled.sample_role == "train"].copy().sort_values(["date", "symbol"])
    dates = np.asarray(sorted(pd.to_datetime(train.date).unique()))
    cutoff = dates[max(1, int(len(dates) * 0.8)) - 1]
    train["sample_role"] = np.where(pd.to_datetime(train.date) <= cutoff, "train", "validation")
    if train.groupby("sample_role").target.nunique().min() < 2:
        # Deterministic fallback only for tiny synthetic tests; real runs remain date-separated.
        ordered = train.sort_values(["date", "symbol"]).reset_index(drop=True)
        ordered["sample_role"] = "train"
        ordered.loc[ordered.index >= max(2, int(len(ordered) * .8)), "sample_role"] = "validation"
        train = ordered
    trial_rows = []
    for selection in config.feature_selection_names:
        inner = build_qml_train_validation(train, split_id=split_id,
                                           feature_columns=_feature_columns(train, selection))
        for reps in config.qsvm_repetitions:
            for C in config.qsvm_c_values:
                for interaction_scale in config.interaction_scales:
                    started = time.perf_counter()
                    pred = _qsvm_predictions(
                        inner,
                        C,
                        reps,
                        interaction_scale,
                        "qsvm_candidate",
                        config.random_state,
                    )
                    score = roc_auc_score(pred.y_true, pred.y_score)
                    trial_rows.append({"split_id": split_id, "feature_selection": selection,
                                       "repetitions": reps, "C": C,
                                       "interaction_scale": interaction_scale,
                                       "inner_roc_auc": score,
                                       "runtime_seconds": time.perf_counter() - started,
                                       "inner_train_end": pd.to_datetime(inner.train.metadata.date).max(),
                                       "inner_validation_start": pd.to_datetime(inner.validation.metadata.date).min()})
    trials = pd.DataFrame(trial_rows).sort_values(
        ["inner_roc_auc", "feature_selection", "repetitions", "C", "interaction_scale"],
        ascending=[False, True, True, True, True], kind="stable")
    chosen = trials.iloc[0].to_dict()
    return chosen, trials.reset_index(drop=True)


def _qsvm_predictions(data, C, repetitions, interaction_scale, model_name, seed):
    model = QuantumKernelSVM(QMLModelConfig(model_name=model_name, seed=seed,
        params={"C": C, "n_qubits": 8, "repetitions": repetitions,
                "interaction_scale": interaction_scale})).fit(data.train)
    scores = model.predict_scores(data.validation)
    result = build_prediction_table(metadata=data.validation.metadata, y_true=data.validation.y,
                                    y_score=scores, model_name=model_name, split_id=data.split_id)
    kernel = model.train_kernel_
    result.attrs = {"train_kernel_rows": kernel.shape[0], "train_kernel_columns": kernel.shape[1],
                    "validation_kernel_rows": model.last_prediction_kernel_.shape[0],
                    "validation_kernel_columns": model.last_prediction_kernel_.shape[1],
                    "kernel_mean_similarity": float(kernel.mean()),
                    "support_vectors": int(model.estimator_.support_.size)}
    return result


def _classical_predictions(data, kernel, model_name, seed):
    model = SVC(C=1.0, kernel=kernel, probability=True, random_state=seed).fit(data.train.X, data.train.y)
    scores = model.predict_proba(data.validation.X)[:, int(np.where(model.classes_ == 1)[0][0])]
    result = build_prediction_table(metadata=data.validation.metadata, y_true=data.validation.y,
                                    y_score=scores, model_name=model_name, split_id=data.split_id)
    result.attrs = {"train_kernel_rows": np.nan, "train_kernel_columns": np.nan,
                    "validation_kernel_rows": np.nan, "validation_kernel_columns": np.nan,
                    "kernel_mean_similarity": np.nan, "support_vectors": int(model.support_.size)}
    return result


def _measure(function):
    tracemalloc.start(); started = time.perf_counter()
    try:
        result = function(); _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, time.perf_counter() - started, peak / (1024 ** 2)


def _split_metrics(predictions):
    rows = []
    for (model, split_id), group in predictions.groupby(["model_name", "split_id"], sort=True):
        y, score = group.y_true.to_numpy(int), np.clip(group.y_score.to_numpy(float), 1e-12, 1 - 1e-12)
        count = max(1, int(np.ceil(len(group) * .1)))
        top = group.nlargest(count, "y_score")
        information_coefficient = (
            group.y_score.corr(group.forward_excess_return, method="spearman")
            if group.y_score.nunique() > 1
            and group.forward_excess_return.nunique() > 1
            else np.nan
        )
        rows.append({"model_name": model, "split_id": split_id,
                     "accuracy": accuracy_score(y, score >= .5), "roc_auc": roc_auc_score(y, score),
                     "log_loss": log_loss(y, score, labels=[0, 1]), "brier_score": brier_score_loss(y, score),
                     "information_coefficient": information_coefficient,
                     "top_decile_return": top.forward_excess_return.mean()})
    return pd.DataFrame(rows)


def _sample_split(data, split_id, config):
    frames = []
    for role, limit in [("train", config.train_rows), ("validation", config.validation_rows)]:
        group = data[(data.split_id == split_id) & (data.sample_role == role)]
        per_class = limit // 2
        chosen = [part.sample(n=min(per_class, len(part)), random_state=config.random_state + split_id * 100 + int(target))
                  for target, part in group.groupby("target", sort=True)]
        frames.append(pd.concat(chosen).sort_values(["date", "symbol"]))
    return pd.concat(frames, ignore_index=True)


def _sample_manifest(sampled, split_id):
    rows = []
    for role, group in sampled.groupby("sample_role"):
        keys = group.sort_values(["date", "symbol"])[["symbol", "date"]].astype(str)
        import hashlib
        digest = hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()
        rows.append({"split_id": split_id, "sample_role": role, "rows": len(group),
                     "positive_rate": group.target.mean(),
                     "unique_symbols": group.symbol.nunique(),
                     "symbols": ",".join(sorted(group.symbol.astype(str).unique())),
                     "row_key_sha256": digest})
    return pd.DataFrame(rows)


def _validate_inputs(data, config):
    required = {"symbol", "date", "split_id", "sample_role", "target", "forward_return_5d", "forward_excess_return_5d"}
    for name in config.feature_selection_names:
        required.update(_feature_columns(data, name))
    missing = required - set(data.columns)
    if missing: raise ValueError("Comparison data is missing: " + ", ".join(sorted(missing)))
    if config.bootstrap_iterations <= 0: raise ValueError("bootstrap_iterations must be positive")
    if not config.interaction_scales or any(value < 0 for value in config.interaction_scales):
        raise ValueError("interaction_scales must contain non-negative values")


def _feature_columns(data: pd.DataFrame, selection: str) -> list[str]:
    if selection == "classical_selected":
        missing = set(DEFAULT_SELECTED_FEATURES) - set(data.columns)
        if missing:
            raise ValueError("Classical-selected QML inputs are missing: " + ", ".join(sorted(missing)))
        return DEFAULT_SELECTED_FEATURES
    if selection not in DEFAULT_FEATURE_SELECTIONS:
        raise ValueError(f"Unknown feature selection: {selection}")
    return DEFAULT_FEATURE_SELECTIONS[selection]
