"""Build train-only, classical-selected inputs for expanded-universe QML."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_qml.features.audit import feature_family
from market_qml.models.dataset import build_train_validation_datasets
from market_qml.models.preprocessing import fit_transform_train_validation
from market_qml.utils.statistics import (
    absolute_correlation_matrix,
    safe_correlation,
)


@dataclass(frozen=True)
class SelectedQMLFeatureResult:
    features: pd.DataFrame
    manifest: pd.DataFrame


def build_selected_qml_features(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    selection_diagnostics: pd.DataFrame,
    universe_membership: pd.DataFrame | None = None,
    target_horizon_days: int = 5,
    n_qubits: int = 8,
    correlation_threshold: float = 0.9,
) -> SelectedQMLFeatureResult:
    """Select diverse source features using outer-training rows only.

    The winning classical tuner trial supplies the candidate feature-count
    budget. Features are ranked against continuous excess return and greedily
    de-correlated before being assigned to qubits. No outer-validation target is
    consulted.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if target_horizon_days <= 0:
        raise ValueError("target_horizon_days must be positive.")
    if not 0 <= correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be between zero and one.")

    output_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for split in splits.sort_values("split_id").itertuples(index=False):
        split_id = int(split.split_id)
        best = selection_diagnostics[
            (selection_diagnostics["split_id"] == split_id)
            & (selection_diagnostics["rank"] == 1)
        ]
        if len(best) != 1:
            raise ValueError(
                f"Expected one rank-1 classical selection for split {split_id}."
            )
        candidate_count = int(best.iloc[0]["feature_count"])
        return_column = f"forward_return_{target_horizon_days}d"
        excess_column = f"forward_excess_return_{target_horizon_days}d"
        residual_column = f"residualized_{excess_column}"
        selection_target = (
            residual_column if residual_column in labels.columns else excess_column
        )
        target_column = f"outperform_spy_{target_horizon_days}d"
        datasets = build_train_validation_datasets(
            features=features,
            labels=labels,
            universe_membership=universe_membership,
            target_column=selection_target,
            train_start_date=split.train_start_date,
            train_end_date=split.train_end_date,
            validation_start_date=split.validation_start_date,
            validation_end_date=split.validation_end_date,
        )
        data = fit_transform_train_validation(datasets)
        target = pd.to_numeric(data.train.y, errors="coerce")
        train_dates = pd.to_datetime(data.train.metadata["date"])
        correlations = pd.Series(
            {
                column: _stable_daily_score(data.train.X[column], target, train_dates)
                for column in data.train.X
            }
        ).sort_values(ascending=False)
        candidates = correlations.head(candidate_count).index.tolist()
        selected = _select_diverse(
            data.train.X[candidates],
            candidates,
            n_qubits=n_qubits,
            threshold=correlation_threshold,
        )
        if len(selected) < n_qubits:
            raise ValueError(
                f"Split {split_id} has fewer than {n_qubits} usable features."
            )
        selected = _order_by_family(data.train.X[selected], selected)
        qml_columns = [f"selected_feature_{index:02d}" for index in range(n_qubits)]
        for role, dataset in (("train", data.train), ("validation", data.validation)):
            metadata_columns = ["symbol", "date", return_column, excess_column]
            metadata_columns += [
                column
                for column in ["sector", "size_bucket"]
                if column in dataset.metadata
            ]
            frame = dataset.metadata[metadata_columns].copy()
            frame["split_id"] = split_id
            frame["sample_role"] = role
            continuous_target = pd.to_numeric(dataset.y, errors="coerce")
            frame["target"] = continuous_target.gt(0).astype("int8").to_numpy()
            frame["ranking_target"] = continuous_target.to_numpy()
            transformed = dataset.X[selected].copy()
            transformed = _cross_sectional_rank_encode(
                transformed, pd.to_datetime(dataset.metadata["date"])
            )
            transformed.columns = qml_columns
            frame = pd.concat(
                [frame.reset_index(drop=True), transformed.reset_index(drop=True)],
                axis=1,
            )
            output_frames.append(frame)
        source_corr = absolute_correlation_matrix(data.train.X[selected])
        for qubit, source in enumerate(selected):
            neighbor = selected[(qubit + 1) % len(selected)]
            manifest_rows.append(
                {
                    "split_id": split_id,
                    "qubit": qubit,
                    "qml_feature": qml_columns[qubit],
                    "source_feature": source,
                    "target_abs_correlation": float(correlations[source]),
                    "target_column": selection_target,
                    "classification_target": target_column,
                    "return_column": excess_column,
                    "source_feature_family": feature_family(source),
                    "encoding": "same_date_rank_linear",
                    "target_horizon_days": target_horizon_days,
                    "next_qubit_source_feature": neighbor,
                    "neighbor_abs_correlation": float(
                        source_corr.loc[source, neighbor]
                    ),
                    "classical_candidate_feature_count": candidate_count,
                    "train_start_date": pd.Timestamp(split.train_start_date),
                    "train_end_date": pd.Timestamp(split.train_end_date),
                    "validation_start_date": pd.Timestamp(split.validation_start_date),
                }
            )
    return SelectedQMLFeatureResult(
        features=pd.concat(output_frames, ignore_index=True),
        manifest=pd.DataFrame(manifest_rows),
    )


def _select_diverse(
    X: pd.DataFrame,
    ranked: list[str],
    *,
    n_qubits: int,
    threshold: float,
) -> list[str]:
    correlations = absolute_correlation_matrix(X)
    selected: list[str] = []
    for feature in ranked:
        if not selected or all(
            correlations.loc[feature, existing] < threshold for existing in selected
        ):
            selected.append(feature)
        if len(selected) == n_qubits:
            return selected
    for feature in ranked:
        if feature not in selected:
            selected.append(feature)
        if len(selected) == n_qubits:
            break
    return selected


def _order_by_relationship(X: pd.DataFrame, selected: list[str]) -> list[str]:
    """Order selected features so ring neighbors have meaningful relationships."""
    correlations = absolute_correlation_matrix(X)
    ordered = [selected[0]]
    remaining = set(selected[1:])
    while remaining:
        current = ordered[-1]
        next_feature = max(
            remaining,
            key=lambda candidate: (correlations.loc[current, candidate], candidate),
        )
        ordered.append(next_feature)
        remaining.remove(next_feature)
    return ordered


def _order_by_family(X: pd.DataFrame, selected: list[str]) -> list[str]:
    """Place related economic families together, then order within each family."""
    families: dict[str, list[str]] = {}
    for feature in selected:
        families.setdefault(feature_family(feature), []).append(feature)
    ordered: list[str] = []
    for family in sorted(families):
        ordered.extend(_order_by_relationship(X, families[family]))
    return ordered


def _stable_daily_score(
    feature: pd.Series, target: pd.Series, dates: pd.Series
) -> float:
    daily = []
    frame = pd.DataFrame({"feature": feature, "target": target, "date": dates})
    for _, day in frame.groupby("date", sort=False):
        correlation = safe_correlation(day["feature"], day["target"])
        if pd.notna(correlation):
            daily.append(float(correlation))
    if not daily:
        return 0.0
    median = float(pd.Series(daily).median())
    sign_share = float((pd.Series(daily) * median >= 0).mean()) if median else 0.0
    coverage = float(pd.to_numeric(feature, errors="coerce").notna().mean())
    return abs(median) * sign_share * coverage


def _cross_sectional_rank_encode(
    features: pd.DataFrame, dates: pd.Series
) -> pd.DataFrame:
    """Map each contemporaneous cross-section to [-1, 1] without fitting."""
    result = features.apply(pd.to_numeric, errors="coerce")
    for column in result:
        ranks = (
            result[column]
            .groupby(dates.reset_index(drop=True))
            .rank(method="average", pct=True)
        )
        result[column] = (2.0 * ranks - 1.0).fillna(0.0)
    return result
