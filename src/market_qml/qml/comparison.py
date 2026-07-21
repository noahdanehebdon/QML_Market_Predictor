"""Leakage-safe, apples-to-apples QML and classical model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.svm import SVC

from market_qml.backtest.portfolio import (
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_RETURN_HORIZON_DAYS,
    DEFAULT_TRANSACTION_COST_BPS,
    TRADING_DAYS_PER_YEAR,
    run_portfolio_backtest,
    summarize_portfolio_risk,
)
from market_qml.backtest.ranking_metrics import evaluate_ranking_metrics
from market_qml.models.predictions import build_prediction_table
from market_qml.qml.interface import build_qml_train_validation
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
    vqc_ansatz_depths: tuple[int, ...] = (1, 2)
    vqc_learning_rates: tuple[float, ...] = (0.05, 0.1)
    vqc_optimizers: tuple[str, ...] = ("spsa", "finite_difference")
    qcnn_learning_rates: tuple[float, ...] = (0.03, 0.05, 0.1)
    qcnn_initialization_scales: tuple[float, ...] = (0.05, 0.1)
    qsvm_c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    qsvm_repetitions: tuple[int, ...] = (1, 2, 3)
    feature_selection_names: tuple[str, ...] = ("classical_selected",)
    interaction_scales: tuple[float, ...] = (0.0, 0.5, 1.0)
    bootstrap_iterations: int = 2000
    portfolio_top_fraction: float = 0.1
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY
    return_horizon_days: int = DEFAULT_RETURN_HORIZON_DAYS


@dataclass(frozen=True)
class ComparisonResult:
    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    resource_usage: pd.DataFrame
    qsvm_tuning_trials: pd.DataFrame
    qsvm_selected_configs: pd.DataFrame
    vqc_tuning_trials: pd.DataFrame
    vqc_selected_configs: pd.DataFrame
    qcnn_tuning_trials: pd.DataFrame
    qcnn_selected_configs: pd.DataFrame
    sample_manifest: pd.DataFrame
    ranking_metrics: pd.DataFrame
    portfolio_returns: pd.DataFrame
    portfolio_metrics: pd.DataFrame


def run_model_comparison(data: pd.DataFrame, config: ComparisonConfig = ComparisonConfig()) -> ComparisonResult:
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
        chosen, split_trials = _select_qsvm(sampled, int(split_id), config)
        trials.append(split_trials)
        selected.append(chosen)
        chosen_vqc, split_vqc_trials = _select_vqc(sampled, int(split_id), config)
        vqc_trials.append(split_vqc_trials)
        vqc_selected.append(chosen_vqc)
        chosen_qcnn, split_qcnn_trials = _select_qcnn(sampled, int(split_id), config)
        qcnn_trials.append(split_qcnn_trials)
        qcnn_selected.append(chosen_qcnn)

        runners = {
            "vqc": lambda c=chosen_vqc: train_vqc(
                primary, max_iter=config.vqc_iterations,
                ansatz_depth=int(c["ansatz_depth"]),
                learning_rate=float(c["learning_rate"]),
                optimizer=str(c["optimizer"]),
                random_state=config.random_state + int(split_id)).predictions,
            "qcnn": lambda c=chosen_qcnn: train_qcnn(
                primary, max_iter=config.qcnn_iterations,
                learning_rate=float(c["learning_rate"]),
                initialization_scale=float(c["initialization_scale"]),
                random_state=config.random_state + int(split_id)).predictions,
            "qsvm": lambda: _qsvm_predictions(primary, 1.0, 2, 0.0, "qsvm", config.random_state),
            "linear_svm": lambda: _classical_predictions(primary, "linear", "linear_svm", config.random_state),
            "rbf_svm": lambda: _classical_predictions(primary, "rbf", "rbf_svm", config.random_state),
            "logistic_regression": lambda: _logistic_predictions(primary, config.random_state),
            "gradient_boosting": lambda: _gradient_boosting_predictions(primary, config.random_state),
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
    ranking_metrics = evaluate_ranking_metrics(
        prediction_table, top_fraction=config.portfolio_top_fraction
    )
    portfolio_returns = run_portfolio_backtest(
        prediction_table,
        top_fraction=config.portfolio_top_fraction,
        transaction_cost_bps=config.transaction_cost_bps,
        rebalance_frequency=config.rebalance_frequency,
        return_horizon_days=config.return_horizon_days,
    )
    portfolio_metrics = summarize_portfolio_risk(portfolio_returns)
    return ComparisonResult(
        predictions=prediction_table,
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_split_metrics(split_metrics, config.bootstrap_iterations, config.random_state),
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
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "predictions": result.predictions, "split_metrics": result.split_metrics,
        "aggregate_metrics": result.aggregate_metrics, "resource_usage": result.resource_usage,
        "qsvm_tuning_trials": result.qsvm_tuning_trials,
        "qsvm_selected_configs": result.qsvm_selected_configs,
        "vqc_tuning_trials": result.vqc_tuning_trials,
        "vqc_selected_configs": result.vqc_selected_configs,
        "qcnn_tuning_trials": result.qcnn_tuning_trials,
        "qcnn_selected_configs": result.qcnn_selected_configs,
        "sample_manifest": result.sample_manifest,
        "ranking_metrics": result.ranking_metrics,
        "portfolio_returns": result.portfolio_returns,
        "portfolio_metrics": result.portfolio_metrics,
    }
    paths = {}
    for name, table in tables.items():
        paths[name] = output / f"{name}.parquet"
        table.to_parquet(paths[name], index=False)
    paths["report"] = output / "comparison_report.md"
    paths["report"].write_text(render_comparison_report(result), encoding="utf-8")
    return paths


def render_comparison_report(result: ComparisonResult) -> str:
    pivot = result.aggregate_metrics.pivot(index="model_name", columns="metric", values="mean")
    ranked = pivot.sort_values("roc_auc", ascending=False)
    best = ranked.index[0]
    qml_models = ["vqc", "qcnn", "qsvm", "qsvm_tuned"]
    classical_models = ["logistic_regression", "gradient_boosting"]
    best_qml = pivot.loc[qml_models, "roc_auc"].idxmax()
    best_classical = pivot.loc[classical_models, "roc_auc"].idxmax()
    qml_auc = float(pivot.loc[best_qml, "roc_auc"])
    classical_auc = float(pivot.loc[best_classical, "roc_auc"])
    if classical_auc > qml_auc + 0.02:
        decision = f"QML underperforms the requested classical baselines on mean ROC-AUC ({qml_auc:.4f} versus {classical_auc:.4f})."
    elif qml_auc > classical_auc + 0.02:
        decision = f"QML outperforms the requested classical baselines on mean ROC-AUC ({qml_auc:.4f} versus {classical_auc:.4f}); confirm the result on additional chronological splits."
    else:
        decision = f"QML and the requested classical baselines behave similarly on mean ROC-AUC ({qml_auc:.4f} versus {classical_auc:.4f}); ranking and portfolio metrics should drive interpretation."
    overall_ranking = result.ranking_metrics.query("scope == 'overall'").set_index("model_name")
    overall_portfolio = result.portfolio_metrics.query("scope == 'overall'").set_index("model_name")
    display = ranked[["accuracy", "roc_auc", "log_loss", "brier_score"]].join(
        overall_ranking[["rank_information_coefficient", "long_short_spread"]]
    ).join(
        overall_portfolio[["cumulative_net_return", "cumulative_net_excess_return", "net_sharpe"]]
    )
    ranking_winner = _metric_leader(display["rank_information_coefficient"])
    portfolio_winner = _metric_leader(display["net_sharpe"])
    headers = ["model"] + list(display.columns)
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    table.extend("| " + " | ".join([str(index)] + [_format_report_value(value) for value in row]) + " |"
                 for index, row in display.iterrows())
    return "\n".join([
        "# QML model comparison", "", "All models used identical outer validation rows. VQC, QCNN, and QSVM choices were made only from an inner chronological portion of each training window.", "",
        f"Portfolio assumptions: {config_text(result.portfolio_returns)}.", "",
        *table, "", f"Classification leader (mean ROC-AUC): **{best}**. Ranking leader (overall rank IC): **{ranking_winner}**. Portfolio leader (overall net Sharpe): **{portfolio_winner}**.", "",
        f"Best QML: **{best_qml}**; best requested classical baseline: **{best_classical}**.", "", f"Decision: {decision}", "",
        "Classification and split-bootstrap uncertainty are recorded in `split_metrics.parquet` and `aggregate_metrics.parquet`. Date-level ranking results are in `ranking_metrics.parquet`; transaction-cost-aware returns and risk metrics are in `portfolio_returns.parquet` and `portfolio_metrics.parquet`.", "",
        "Runtime, peak traced memory, selected configurations, tuning trials, and exact sampled-row hashes are retained beside this report.",
    ])


def config_text(portfolio_returns: pd.DataFrame) -> str:
    row = portfolio_returns.iloc[0]
    periods = TRADING_DAYS_PER_YEAR / float(row["rebalance_frequency"])
    return (
        f"{int(row['return_horizon_days'])}-trading-day returns, rebalance every "
        f"{int(row['rebalance_frequency'])} prediction dates, {periods:g} periods/year, "
        f"and {float(row['transaction_cost_bps']):g} bps one-way costs"
    )


def _metric_leader(values: pd.Series) -> str:
    available = pd.to_numeric(values, errors="coerce").dropna()
    if available.empty:
        return "not available"
    maximum = available.max()
    winners = [str(name) for name, value in available.items() if np.isclose(value, maximum)]
    return ", ".join(winners)


def _format_report_value(value) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}"


def _select_qsvm(sampled: pd.DataFrame, split_id: int, config: ComparisonConfig):
    train = _inner_training_frame(sampled)
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


def _select_vqc(sampled: pd.DataFrame, split_id: int, config: ComparisonConfig):
    inner = build_qml_train_validation(
        _inner_training_frame(sampled),
        split_id=split_id,
        feature_columns=_feature_columns(sampled, config.feature_selection_names[0]),
    )
    rows = []
    for depth, learning_rate, optimizer in product(
        config.vqc_ansatz_depths,
        config.vqc_learning_rates,
        config.vqc_optimizers,
    ):
        started = time.perf_counter()
        predictions = train_vqc(
            inner,
            ansatz_depth=depth,
            learning_rate=learning_rate,
            optimizer=optimizer,
            max_iter=config.vqc_iterations,
            random_state=config.random_state + split_id,
        ).predictions
        rows.append(
            {
                "split_id": split_id,
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
    trials = pd.DataFrame(rows).sort_values(
        ["inner_roc_auc", "ansatz_depth", "learning_rate", "optimizer"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    return trials.iloc[0].to_dict(), trials.reset_index(drop=True)


def _select_qcnn(sampled: pd.DataFrame, split_id: int, config: ComparisonConfig):
    inner = build_qml_train_validation(
        _inner_training_frame(sampled),
        split_id=split_id,
        feature_columns=_feature_columns(sampled, config.feature_selection_names[0]),
    )
    rows = []
    for learning_rate, initialization_scale in product(
        config.qcnn_learning_rates,
        config.qcnn_initialization_scales,
    ):
        started = time.perf_counter()
        predictions = train_qcnn(
            inner,
            learning_rate=learning_rate,
            initialization_scale=initialization_scale,
            max_iter=config.qcnn_iterations,
            random_state=config.random_state + split_id,
        ).predictions
        rows.append(
            {
                "split_id": split_id,
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
    trials = pd.DataFrame(rows).sort_values(
        ["inner_roc_auc", "learning_rate", "initialization_scale"],
        ascending=[False, True, True],
        kind="stable",
    )
    return trials.iloc[0].to_dict(), trials.reset_index(drop=True)


def _inner_training_frame(sampled: pd.DataFrame) -> pd.DataFrame:
    train = sampled[sampled.sample_role == "train"].copy().sort_values(
        ["date", "symbol"]
    )
    dates = np.asarray(sorted(pd.to_datetime(train.date).unique()))
    cutoff = dates[max(1, int(len(dates) * 0.8)) - 1]
    train["sample_role"] = np.where(
        pd.to_datetime(train.date) <= cutoff, "train", "validation"
    )
    if train.groupby("sample_role").target.nunique().min() < 2:
        # Deterministic fallback only for tiny synthetic tests.
        ordered = train.sort_values(["date", "symbol"]).reset_index(drop=True)
        ordered["sample_role"] = "train"
        ordered.loc[
            ordered.index >= max(2, int(len(ordered) * 0.8)), "sample_role"
        ] = "validation"
        train = ordered
    return train


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


def _logistic_predictions(data, seed):
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(data.train.X, data.train.y)
    scores = model.predict_proba(data.validation.X)[:, int(np.where(model.classes_ == 1)[0][0])]
    result = build_prediction_table(metadata=data.validation.metadata, y_true=data.validation.y,
                                    y_score=scores, model_name="logistic_regression", split_id=data.split_id)
    result.attrs = _non_kernel_resource_attrs()
    return result


def _gradient_boosting_predictions(data, seed):
    model = HistGradientBoostingClassifier(random_state=seed).fit(data.train.X, data.train.y)
    scores = model.predict_proba(data.validation.X)[:, int(np.where(model.classes_ == 1)[0][0])]
    result = build_prediction_table(metadata=data.validation.metadata, y_true=data.validation.y,
                                    y_score=scores, model_name="gradient_boosting", split_id=data.split_id)
    result.attrs = _non_kernel_resource_attrs()
    return result


def _non_kernel_resource_attrs():
    return {"train_kernel_rows": np.nan, "train_kernel_columns": np.nan,
            "validation_kernel_rows": np.nan, "validation_kernel_columns": np.nan,
            "kernel_mean_similarity": np.nan, "support_vectors": np.nan}


def _measure(function):
    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
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
    if missing:
        raise ValueError("Comparison data is missing: " + ", ".join(sorted(missing)))
    if config.bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if not config.interaction_scales or any(value < 0 for value in config.interaction_scales):
        raise ValueError("interaction_scales must contain non-negative values")
    if not 0 < config.portfolio_top_fraction <= 0.5:
        raise ValueError("portfolio_top_fraction must be greater than 0 and at most 0.5")
    if config.transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if config.rebalance_frequency <= 0:
        raise ValueError("rebalance_frequency must be positive")
    if not config.vqc_ansatz_depths or not config.vqc_learning_rates or not config.vqc_optimizers:
        raise ValueError("VQC tuning grids must not be empty")
    if not config.qcnn_learning_rates or not config.qcnn_initialization_scales:
        raise ValueError("QCNN tuning grids must not be empty")


def _feature_columns(data: pd.DataFrame, selection: str) -> list[str]:
    if selection == "classical_selected":
        missing = set(DEFAULT_SELECTED_FEATURES) - set(data.columns)
        if missing:
            raise ValueError("Classical-selected QML inputs are missing: " + ", ".join(sorted(missing)))
        return DEFAULT_SELECTED_FEATURES
    if selection not in DEFAULT_FEATURE_SELECTIONS:
        raise ValueError(f"Unknown feature selection: {selection}")
    return DEFAULT_FEATURE_SELECTIONS[selection]
