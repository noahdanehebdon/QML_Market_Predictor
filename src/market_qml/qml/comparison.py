"""Leakage-safe, apples-to-apples QML and classical model comparison."""

from __future__ import annotations

import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from market_qml.backtest.portfolio import (
    run_portfolio_backtest,
    summarize_portfolio_risk,
)
from market_qml.backtest.ranking_metrics import evaluate_ranking_metrics
from market_qml.backtest.validation import (
    paired_model_comparisons,
    prediction_date_block_metrics,
)
from market_qml.models.predictions import build_prediction_table
from market_qml.qml.comparison_reporting import (
    config_text,
    render_comparison_report,
    save_comparison_result,
)
from market_qml.qml.comparison_types import (
    DEFAULT_FEATURE_SELECTIONS,
    DEFAULT_SELECTED_FEATURES,
    ComparisonConfig,
    ComparisonResult,
)
from market_qml.qml.encoding import AngleEncodingConfig, angle_encode_dataset
from market_qml.qml.feature_map import QuantumFeatureMapConfig, QuantumKernelFeatureMap
from market_qml.qml.interface import QMLModelConfig, build_qml_train_validation
from market_qml.qml.qcnn import train_qcnn
from market_qml.qml.qsvm import (
    CachedStateFidelityKernel,
    QuantumKernelSVM,
    calibrated_svc,
)
from market_qml.qml.vqc import train_vqc

__all__ = [
    "DEFAULT_FEATURE_SELECTIONS",
    "ComparisonConfig",
    "ComparisonResult",
    "aggregate_split_metrics",
    "config_text",
    "render_comparison_report",
    "run_model_comparison",
    "save_comparison_result",
]


def run_model_comparison(
    data: pd.DataFrame, config: ComparisonConfig = ComparisonConfig()
) -> ComparisonResult:
    """Compare all models on identical outer rows and tune QSVM on train rows only."""
    _validate_inputs(data, config)
    predictions, resources, trials, selected, manifests = [], [], [], [], []
    vqc_trials, vqc_selected, qcnn_trials, qcnn_selected = [], [], [], []
    for split_id in sorted(data["split_id"].unique()):
        sampled = _sample_split(data, int(split_id), config)
        manifests.append(_sample_manifest(sampled, int(split_id)))
        primary_columns = _feature_columns(sampled, config.feature_selection_names[0])
        primary = build_qml_train_validation(
            sampled, split_id=int(split_id), feature_columns=primary_columns
        )
        inner_folds = _prepared_inner_folds(sampled, int(split_id), config)
        angle_cache = _prepared_angle_cache(
            inner_folds[config.feature_selection_names[0]]
        )
        chosen, split_trials = _select_qsvm(
            sampled, int(split_id), config, prepared_folds=inner_folds
        )
        trials.append(split_trials)
        selected.append(chosen)
        chosen_vqc, split_vqc_trials = _select_vqc(
            sampled,
            int(split_id),
            config,
            prepared_folds=inner_folds,
            angle_cache=angle_cache,
        )
        vqc_trials.append(split_vqc_trials)
        vqc_selected.append(chosen_vqc)
        chosen_qcnn, split_qcnn_trials = _select_qcnn(
            sampled,
            int(split_id),
            config,
            prepared_folds=inner_folds,
            angle_cache=angle_cache,
        )
        qcnn_trials.append(split_qcnn_trials)
        qcnn_selected.append(chosen_qcnn)

        runners = {
            "vqc": lambda c=chosen_vqc: (
                train_vqc(
                    primary,
                    max_iter=config.vqc_iterations,
                    ansatz_depth=int(c["ansatz_depth"]),
                    learning_rate=float(c["learning_rate"]),
                    optimizer=str(c["optimizer"]),
                    random_state=config.random_state + int(split_id),
                ).predictions
            ),
            "qcnn": lambda c=chosen_qcnn: (
                train_qcnn(
                    primary,
                    max_iter=config.qcnn_iterations,
                    learning_rate=float(c["learning_rate"]),
                    initialization_scale=float(c["initialization_scale"]),
                    random_state=config.random_state + int(split_id),
                ).predictions
            ),
            "qsvm": lambda: _qsvm_predictions(
                primary, 1.0, 2, 0.0, "qsvm", config.random_state
            ),
            "linear_svm": lambda: _classical_predictions(
                primary, "linear", "linear_svm", config.random_state
            ),
            "rbf_svm": lambda: _classical_predictions(
                primary, "rbf", "rbf_svm", config.random_state
            ),
            "logistic_regression": lambda: _logistic_predictions(
                primary, config.random_state
            ),
            "gradient_boosting": lambda: _gradient_boosting_predictions(
                primary, config.random_state
            ),
        }
        tuned_data = build_qml_train_validation(
            sampled,
            split_id=int(split_id),
            feature_columns=_feature_columns(sampled, chosen["feature_selection"]),
        )
        runners["qsvm_tuned"] = lambda d=tuned_data, c=chosen: _qsvm_predictions(
            d,
            float(c["C"]),
            int(c["repetitions"]),
            float(c["interaction_scale"]),
            "qsvm_tuned",
            config.random_state,
        )
        for model_name, runner in runners.items():
            frame, seconds, peak_mb = _measure(runner)
            predictions.append(frame)
            resources.append(
                {
                    "model_name": model_name,
                    "split_id": int(split_id),
                    "runtime_seconds": seconds,
                    "peak_memory_mb": peak_mb,
                    **frame.attrs,
                }
            )

    prediction_table = pd.concat(predictions, ignore_index=True)
    split_metrics = _split_metrics(prediction_table)
    ranking_metrics = evaluate_ranking_metrics(
        prediction_table, top_fraction=config.portfolio_top_fraction
    )
    portfolio_returns = run_portfolio_backtest(
        prediction_table,
        top_fraction=config.portfolio_top_fraction,
        transaction_cost_bps=config.transaction_cost_bps,
        rebalance_frequency=config.rebalance_frequency,
        return_horizon_days=config.return_horizon_days,
        sector_neutral=True,
    )
    portfolio_metrics = summarize_portfolio_risk(portfolio_returns)
    date_block_metrics = prediction_date_block_metrics(
        prediction_table, block_days=config.bootstrap_block_days
    )
    paired_comparisons = paired_model_comparisons(
        date_block_metrics,
        metric="roc_auc",
        baseline_model="logistic_regression",
        bootstrap_iterations=config.bootstrap_iterations,
        practical_threshold=config.practical_auc_threshold,
        random_state=config.random_state,
    )
    return ComparisonResult(
        predictions=prediction_table,
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_split_metrics(
            split_metrics, config.bootstrap_iterations, config.random_state
        ),
        resource_usage=pd.DataFrame(resources),
        qsvm_tuning_trials=pd.concat(trials, ignore_index=True),
        qsvm_selected_configs=pd.DataFrame(selected),
        vqc_tuning_trials=pd.concat(vqc_trials, ignore_index=True),
        vqc_selected_configs=pd.DataFrame(vqc_selected),
        qcnn_tuning_trials=pd.concat(qcnn_trials, ignore_index=True),
        qcnn_selected_configs=pd.DataFrame(qcnn_selected),
        sample_manifest=pd.concat(manifests, ignore_index=True),
        ranking_metrics=ranking_metrics,
        portfolio_returns=portfolio_returns,
        portfolio_metrics=portfolio_metrics,
        paired_comparisons=paired_comparisons,
        date_block_metrics=date_block_metrics,
    )


def aggregate_split_metrics(
    metrics: pd.DataFrame, bootstrap_iterations: int = 2000, random_state: int = 42
) -> pd.DataFrame:
    """Aggregate across chronological splits with split-bootstrap intervals."""
    rng = np.random.default_rng(random_state)
    rows = []
    metric_columns = [c for c in metrics.columns if c not in {"model_name", "split_id"}]
    for model_name, group in metrics.groupby("model_name", sort=True):
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            boot = np.asarray(
                [
                    rng.choice(values, len(values), replace=True).mean()
                    for _ in range(bootstrap_iterations)
                ]
            )
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "splits": len(values),
                    "mean": values.mean(),
                    "median": np.median(values),
                    "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "ci_lower": np.quantile(boot, 0.025),
                    "ci_upper": np.quantile(boot, 0.975),
                }
            )
    return pd.DataFrame(rows)


def _select_qsvm(
    sampled: pd.DataFrame,
    split_id: int,
    config: ComparisonConfig,
    *,
    prepared_folds: dict[str, list] | None = None,
):
    folds = prepared_folds or _prepared_inner_folds(sampled, split_id, config)
    state_cache = {}
    tasks = []
    for selection in config.feature_selection_names:
        for reps in config.qsvm_repetitions:
            for interaction_scale in config.interaction_scales:
                feature_map = QuantumKernelFeatureMap(
                    QuantumFeatureMapConfig(
                        n_qubits=8,
                        repetitions=reps,
                        interaction_scale=interaction_scale,
                    )
                )
                for fold_id, inner in enumerate(folds[selection]):
                    key = (selection, reps, interaction_scale, fold_id)
                    train_states = feature_map.transform(inner.train).states
                    state_cache[key] = (
                        train_states,
                        feature_map.transform(inner.validation).states,
                        CachedStateFidelityKernel(train_states),
                    )
                    for C in config.qsvm_c_values:
                        tasks.append(
                            (selection, reps, C, interaction_scale, fold_id, inner, key)
                        )

    def evaluate(task):
        selection, reps, C, interaction_scale, fold_id, inner, key = task
        started = time.perf_counter()
        pred = _qsvm_predictions_from_states(
            inner,
            state_cache[key],
            C,
            reps,
            interaction_scale,
            "qsvm_candidate",
            config.random_state,
        )
        return {
            "split_id": split_id,
            "inner_fold_id": fold_id,
            "feature_selection": selection,
            "repetitions": reps,
            "C": C,
            "interaction_scale": interaction_scale,
            "inner_roc_auc": roc_auc_score(pred.y_true, pred.y_score),
            "runtime_seconds": time.perf_counter() - started,
            "inner_train_end": pd.to_datetime(inner.train.metadata.date).max(),
            "inner_validation_start": pd.to_datetime(
                inner.validation.metadata.date
            ).min(),
        }

    trial_rows = _ordered_map(evaluate, tasks, config.max_workers)
    trials = pd.DataFrame(trial_rows)
    keys = ["feature_selection", "repetitions", "C", "interaction_scale"]
    means = (
        trials.groupby(keys, as_index=False)
        .inner_roc_auc.mean()
        .rename(columns={"inner_roc_auc": "mean_inner_roc_auc"})
    )
    trials = trials.merge(means, on=keys).sort_values(
        ["mean_inner_roc_auc", *keys],
        ascending=[False, True, True, True, True],
        kind="stable",
    )
    chosen = trials.iloc[0].to_dict()
    return chosen, trials.reset_index(drop=True)


def _select_vqc(
    sampled: pd.DataFrame,
    split_id: int,
    config: ComparisonConfig,
    *,
    prepared_folds: dict[str, list] | None = None,
    angle_cache: list[tuple[np.ndarray, np.ndarray]] | None = None,
):
    rows = []
    folds = prepared_folds or _prepared_inner_folds(sampled, split_id, config)
    cached_angles = angle_cache or _prepared_angle_cache(
        folds[config.feature_selection_names[0]]
    )
    for depth, learning_rate, optimizer in product(
        config.vqc_ansatz_depths,
        config.vqc_learning_rates,
        config.vqc_optimizers,
    ):
        for fold_id, inner in enumerate(folds[config.feature_selection_names[0]]):
            started = time.perf_counter()
            predictions = train_vqc(
                inner,
                ansatz_depth=depth,
                learning_rate=learning_rate,
                optimizer=optimizer,
                max_iter=config.vqc_iterations,
                random_state=config.random_state + split_id,
                train_angles=cached_angles[fold_id][0],
                validation_angles=cached_angles[fold_id][1],
            ).predictions
            rows.append(
                {
                    "split_id": split_id,
                    "inner_fold_id": fold_id,
                    "ansatz_depth": depth,
                    "learning_rate": learning_rate,
                    "optimizer": optimizer,
                    "inner_roc_auc": roc_auc_score(
                        predictions.y_true, predictions.y_score
                    ),
                    "runtime_seconds": time.perf_counter() - started,
                    "inner_train_end": pd.to_datetime(inner.train.metadata.date).max(),
                    "inner_validation_start": pd.to_datetime(
                        inner.validation.metadata.date
                    ).min(),
                }
            )
    trials = pd.DataFrame(rows)
    keys = ["ansatz_depth", "learning_rate", "optimizer"]
    means = (
        trials.groupby(keys, as_index=False)
        .inner_roc_auc.mean()
        .rename(columns={"inner_roc_auc": "mean_inner_roc_auc"})
    )
    trials = trials.merge(means, on=keys).sort_values(
        ["mean_inner_roc_auc", *keys],
        ascending=[False, True, True, True],
        kind="stable",
    )
    return trials.iloc[0].to_dict(), trials.reset_index(drop=True)


def _select_qcnn(
    sampled: pd.DataFrame,
    split_id: int,
    config: ComparisonConfig,
    *,
    prepared_folds: dict[str, list] | None = None,
    angle_cache: list[tuple[np.ndarray, np.ndarray]] | None = None,
):
    rows = []
    folds = prepared_folds or _prepared_inner_folds(sampled, split_id, config)
    cached_angles = angle_cache or _prepared_angle_cache(
        folds[config.feature_selection_names[0]]
    )
    for learning_rate, initialization_scale in product(
        config.qcnn_learning_rates,
        config.qcnn_initialization_scales,
    ):
        for fold_id, inner in enumerate(folds[config.feature_selection_names[0]]):
            started = time.perf_counter()
            predictions = train_qcnn(
                inner,
                learning_rate=learning_rate,
                initialization_scale=initialization_scale,
                max_iter=config.qcnn_iterations,
                random_state=config.random_state + split_id,
                train_angles=cached_angles[fold_id][0],
                validation_angles=cached_angles[fold_id][1],
            ).predictions
            rows.append(
                {
                    "split_id": split_id,
                    "inner_fold_id": fold_id,
                    "learning_rate": learning_rate,
                    "initialization_scale": initialization_scale,
                    "inner_roc_auc": roc_auc_score(
                        predictions.y_true, predictions.y_score
                    ),
                    "runtime_seconds": time.perf_counter() - started,
                    "inner_train_end": pd.to_datetime(inner.train.metadata.date).max(),
                    "inner_validation_start": pd.to_datetime(
                        inner.validation.metadata.date
                    ).min(),
                }
            )
    trials = pd.DataFrame(rows)
    keys = ["learning_rate", "initialization_scale"]
    means = (
        trials.groupby(keys, as_index=False)
        .inner_roc_auc.mean()
        .rename(columns={"inner_roc_auc": "mean_inner_roc_auc"})
    )
    trials = trials.merge(means, on=keys).sort_values(
        ["mean_inner_roc_auc", *keys],
        ascending=[False, True, True],
        kind="stable",
    )
    return trials.iloc[0].to_dict(), trials.reset_index(drop=True)


def _inner_training_folds(
    sampled: pd.DataFrame, config: ComparisonConfig
) -> list[pd.DataFrame]:
    train = (
        sampled[sampled.sample_role == "train"].copy().sort_values(["date", "symbol"])
    )
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(train.date).unique()))
    if config.inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    minimum_train = max(2, len(dates) // (config.inner_folds + 1))
    validation_blocks = np.array_split(dates[minimum_train:], config.inner_folds)
    folds = []
    for block in validation_blocks:
        if not len(block):
            continue
        validation_start = pd.Timestamp(block[0])
        earlier_dates = dates[dates < validation_start]
        usable_train_dates = earlier_dates[
            : max(0, len(earlier_dates) - config.inner_purge_days)
        ]
        fold = train.loc[
            train.date.isin(usable_train_dates) | train.date.isin(block)
        ].copy()
        fold["sample_role"] = np.where(fold.date.isin(block), "validation", "train")
        if (fold.groupby("sample_role").target.nunique() >= 2).all():
            folds.append(fold)
    if len(folds) < 2:
        raise ValueError(
            "Training data cannot form at least two valid chronological inner folds"
        )
    return folds


def _prepared_inner_folds(sampled, split_id, config):
    """Build chronological folds and their QML frames once per outer split."""
    raw_folds = _inner_training_folds(sampled, config)
    return {
        selection: [
            build_qml_train_validation(
                fold,
                split_id=split_id,
                feature_columns=_feature_columns(fold, selection),
            )
            for fold in raw_folds
        ]
        for selection in config.feature_selection_names
    }


def _prepared_angle_cache(folds):
    """Encode each fold role once for reuse by VQC and QCNN candidates."""
    config = AngleEncodingConfig(n_qubits=8)
    return [
        (
            angle_encode_dataset(
                fold.train,
                config=config,
                feature_columns=list(fold.train.X.columns),
            ).X.to_numpy(dtype=float),
            angle_encode_dataset(
                fold.validation,
                config=config,
                feature_columns=list(fold.validation.X.columns),
            ).X.to_numpy(dtype=float),
        )
        for fold in folds
    ]


def _ordered_map(function, items, max_workers):
    """Evaluate independent tasks concurrently while preserving input order."""
    if max_workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(function, items))


def _qsvm_predictions(data, C, repetitions, interaction_scale, model_name, seed):
    model = QuantumKernelSVM(
        QMLModelConfig(
            model_name=model_name,
            seed=seed,
            params={
                "C": C,
                "n_qubits": 8,
                "repetitions": repetitions,
                "interaction_scale": interaction_scale,
            },
        )
    ).fit(data.train)
    scores = model.predict_scores(data.validation)
    result = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=scores,
        model_name=model_name,
        split_id=data.split_id,
    )
    kernel = model.train_kernel_
    result.attrs = {
        "train_kernel_rows": kernel.shape[0],
        "train_kernel_columns": kernel.shape[1],
        "validation_kernel_rows": model.last_prediction_kernel_.shape[0],
        "validation_kernel_columns": model.last_prediction_kernel_.shape[1],
        "kernel_mean_similarity": float(kernel.mean()),
        "support_vectors": model.support_vector_count,
    }
    return result


def _qsvm_predictions_from_states(
    data,
    states,
    C,
    repetitions,
    interaction_scale,
    model_name,
    seed,
):
    """Run a QSVM candidate from cached train/validation circuit states."""
    train_states, validation_states, kernel = states
    model = QuantumKernelSVM(
        QMLModelConfig(
            model_name=model_name,
            seed=seed,
            params={
                "C": C,
                "n_qubits": 8,
                "repetitions": repetitions,
                "interaction_scale": interaction_scale,
            },
        )
    ).fit_states(train_states, data.train.y, kernel=kernel)
    scores = model.predict_state_scores(validation_states)
    return build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=scores,
        model_name=model_name,
        split_id=data.split_id,
    )


def _classical_predictions(data, kernel, model_name, seed):
    model = calibrated_svc(
        C=1.0,
        kernel=kernel,
        y=data.train.y,
        random_state=seed,
    ).fit(data.train.X, data.train.y)
    scores = model.predict_proba(data.validation.X)[
        :, int(np.where(model.classes_ == 1)[0][0])
    ]
    result = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=scores,
        model_name=model_name,
        split_id=data.split_id,
    )
    result.attrs = {
        "train_kernel_rows": np.nan,
        "train_kernel_columns": np.nan,
        "validation_kernel_rows": np.nan,
        "validation_kernel_columns": np.nan,
        "kernel_mean_similarity": np.nan,
        "support_vectors": int(
            model.calibrated_classifiers_[0].estimator.support_.size
        ),
    }
    return result


def _logistic_predictions(data, seed):
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(
        data.train.X, data.train.y
    )
    scores = model.predict_proba(data.validation.X)[
        :, int(np.where(model.classes_ == 1)[0][0])
    ]
    result = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=scores,
        model_name="logistic_regression",
        split_id=data.split_id,
    )
    result.attrs = _non_kernel_resource_attrs()
    return result


def _gradient_boosting_predictions(data, seed):
    model = HistGradientBoostingClassifier(random_state=seed).fit(
        data.train.X, data.train.y
    )
    scores = model.predict_proba(data.validation.X)[
        :, int(np.where(model.classes_ == 1)[0][0])
    ]
    result = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=scores,
        model_name="gradient_boosting",
        split_id=data.split_id,
    )
    result.attrs = _non_kernel_resource_attrs()
    return result


def _non_kernel_resource_attrs():
    return {
        "train_kernel_rows": np.nan,
        "train_kernel_columns": np.nan,
        "validation_kernel_rows": np.nan,
        "validation_kernel_columns": np.nan,
        "kernel_mean_similarity": np.nan,
        "support_vectors": np.nan,
    }


def _measure(function):
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, time.perf_counter() - started, peak / (1024**2)


def _split_metrics(predictions):
    rows = []
    for (model, split_id), group in predictions.groupby(
        ["model_name", "split_id"], sort=True
    ):
        y, score = (
            group.y_true.to_numpy(int),
            np.clip(group.y_score.to_numpy(float), 1e-12, 1 - 1e-12),
        )
        count = max(1, int(np.ceil(len(group) * 0.1)))
        top = group.nlargest(count, "y_score")
        information_coefficient = (
            group.y_score.corr(group.forward_excess_return, method="spearman")
            if group.y_score.nunique() > 1 and group.forward_excess_return.nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "model_name": model,
                "split_id": split_id,
                "accuracy": accuracy_score(y, score >= 0.5),
                "roc_auc": roc_auc_score(y, score),
                "log_loss": log_loss(y, score, labels=[0, 1]),
                "brier_score": brier_score_loss(y, score),
                "information_coefficient": information_coefficient,
                "top_decile_return": top.forward_excess_return.mean(),
            }
        )
    return pd.DataFrame(rows)


def _sample_split(data, split_id, config):
    frames = []
    for role, limit in [
        ("train", config.train_rows),
        ("validation", config.validation_rows),
    ]:
        group = data[(data.split_id == split_id) & (data.sample_role == role)]
        per_class = limit // 2
        chosen = [
            part.sample(
                n=min(per_class, len(part)),
                random_state=config.random_state + split_id * 100 + int(target),
            )
            for target, part in group.groupby("target", sort=True)
        ]
        frames.append(pd.concat(chosen).sort_values(["date", "symbol"]))
    return pd.concat(frames, ignore_index=True)


def _sample_manifest(sampled, split_id):
    rows = []
    for role, group in sampled.groupby("sample_role"):
        keys = group.sort_values(["date", "symbol"])[["symbol", "date"]].astype(str)
        import hashlib

        digest = hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()
        rows.append(
            {
                "split_id": split_id,
                "sample_role": role,
                "rows": len(group),
                "positive_rate": group.target.mean(),
                "unique_symbols": group.symbol.nunique(),
                "symbols": ",".join(sorted(group.symbol.astype(str).unique())),
                "row_key_sha256": digest,
            }
        )
    return pd.DataFrame(rows)


def _validate_inputs(data, config):
    required = {
        "symbol",
        "date",
        "split_id",
        "sample_role",
        "target",
        f"forward_return_{config.return_horizon_days}d",
        f"forward_excess_return_{config.return_horizon_days}d",
    }
    for name in config.feature_selection_names:
        required.update(_feature_columns(data, name))
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Comparison data is missing: " + ", ".join(sorted(missing)))
    if config.bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if config.inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    if config.inner_purge_days < 0:
        raise ValueError("inner_purge_days cannot be negative")
    if config.bootstrap_block_days <= 0:
        raise ValueError("bootstrap_block_days must be positive")
    if config.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not config.interaction_scales or any(
        value < 0 for value in config.interaction_scales
    ):
        raise ValueError("interaction_scales must contain non-negative values")
    if not 0 < config.portfolio_top_fraction <= 0.5:
        raise ValueError(
            "portfolio_top_fraction must be greater than 0 and at most 0.5"
        )
    if config.transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if config.rebalance_frequency <= 0:
        raise ValueError("rebalance_frequency must be positive")
    if config.return_horizon_days <= 0:
        raise ValueError("return_horizon_days must be positive")
    if (
        not config.vqc_ansatz_depths
        or not config.vqc_learning_rates
        or not config.vqc_optimizers
    ):
        raise ValueError("VQC tuning grids must not be empty")
    if not config.qcnn_learning_rates or not config.qcnn_initialization_scales:
        raise ValueError("QCNN tuning grids must not be empty")


def _feature_columns(data: pd.DataFrame, selection: str) -> list[str]:
    if selection == "classical_selected":
        missing = set(DEFAULT_SELECTED_FEATURES) - set(data.columns)
        if missing:
            raise ValueError(
                "Classical-selected QML inputs are missing: "
                + ", ".join(sorted(missing))
            )
        return DEFAULT_SELECTED_FEATURES
    if selection not in DEFAULT_FEATURE_SELECTIONS:
        raise ValueError(f"Unknown feature selection: {selection}")
    return DEFAULT_FEATURE_SELECTIONS[selection]
