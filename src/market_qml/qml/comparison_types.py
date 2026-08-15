"""Configuration and result contracts for controlled QML comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_qml.backtest.portfolio import (
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_RETURN_HORIZON_DAYS,
    DEFAULT_TRANSACTION_COST_BPS,
)

DEFAULT_FEATURE_SELECTIONS = {
    "broad_market": [
        "raw_price_pca_00",
        "returns_momentum_pca_00",
        "volatility_pca_00",
        "volume_liquidity_pca_00",
        "benchmark_relative_pca_00",
        "macro_pca_00",
        "fundamentals_pca_00",
        "other_pca_00",
    ],
    "market_dynamics": [
        "raw_price_pca_00",
        "raw_price_pca_01",
        "returns_momentum_pca_00",
        "returns_momentum_pca_01",
        "returns_momentum_pca_02",
        "volatility_pca_00",
        "volume_liquidity_pca_00",
        "benchmark_relative_pca_00",
    ],
    "benchmark_macro": [
        "benchmark_relative_pca_00",
        "benchmark_relative_pca_01",
        "benchmark_relative_pca_02",
        "macro_pca_00",
        "macro_pca_01",
        "returns_momentum_pca_00",
        "volatility_pca_00",
        "fundamentals_pca_00",
    ],
}
DEFAULT_SELECTED_FEATURES = [f"selected_feature_{index:02d}" for index in range(8)]


@dataclass(frozen=True)
class ComparisonConfig:
    train_rows: int = 128
    validation_rows: int = 128
    sample_dates_per_role: int = 8
    random_state: int = 42
    vqc_iterations: int = 10
    qcnn_iterations: int = 10
    vqc_ansatz_depths: tuple[int, ...] = (1, 2)
    vqc_learning_rates: tuple[float, ...] = (0.05, 0.1)
    vqc_optimizers: tuple[str, ...] = ("spsa", "finite_difference")
    vqc_seeds: tuple[int, ...] = (42, 43, 44)
    qcnn_learning_rates: tuple[float, ...] = (0.03, 0.05, 0.1)
    qcnn_initialization_scales: tuple[float, ...] = (0.05, 0.1)
    qsvm_c_values: tuple[float, ...] = (0.3, 1.0, 3.0)
    qsvm_repetitions: tuple[int, ...] = (1, 2)
    feature_selection_names: tuple[str, ...] = ("classical_selected",)
    interaction_scales: tuple[float, ...] = (0.0, 0.25, 0.5)
    qsvm_min_positive_fold_share: float = 2 / 3
    qsvm_min_tuning_improvement: float = 0.005
    qsvm_kernel_concentration_threshold: float = 0.8
    qsvm_support_fraction_threshold: float = 0.9
    bootstrap_iterations: int = 2000
    portfolio_top_fraction: float = 0.1
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY
    return_horizon_days: int = DEFAULT_RETURN_HORIZON_DAYS
    inner_folds: int = 3
    inner_purge_days: int = DEFAULT_RETURN_HORIZON_DAYS
    practical_auc_threshold: float = 0.02
    bootstrap_block_days: int = 20
    max_workers: int = 1


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
    paired_comparisons: pd.DataFrame
    date_block_metrics: pd.DataFrame
