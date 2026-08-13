"""Leakage-safe feature quality, drift, and predictive-stability audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

KEYS = ["symbol", "date"]
FAMILY_PREFIXES = {
    "conditional": ("rate_", "curve_", "inflation_", "stress_"),
    "benchmark": ("spy_", "benchmark_", "beta_", "excess_", "relative_"),
    "macro": ("treasury_", "fed_", "cpi_", "unemployment_", "industrial_", "yield_"),
    "fundamentals": (
        "fundamental_",
        "revenue",
        "income",
        "assets",
        "liabilities",
        "equity",
        "eps",
        "pe_",
        "market_cap",
    ),
    "filing_events": ("sec_", "filing_"),
}


@dataclass(frozen=True)
class FeatureAuditResult:
    quality: pd.DataFrame
    predictive: pd.DataFrame
    stability: pd.DataFrame
    redundancy: pd.DataFrame
    ablations: pd.DataFrame
    exposures: pd.DataFrame
    decisions: pd.DataFrame


def deterministic_cross_section_sample(
    frame: pd.DataFrame, max_rows_per_date: int | None
) -> pd.DataFrame:
    """Bound audit cost with a stable symbol sample inside each date."""
    if max_rows_per_date is None:
        return frame.copy()
    if max_rows_per_date <= 0:
        raise ValueError("max_rows_per_date must be positive.")
    if not set(KEYS).issubset(frame):
        raise ValueError("Sampled frames require symbol and date columns.")
    sampled = frame.copy()
    stable_key = sampled["symbol"].astype(str).str.upper()
    sampled["_audit_order"] = pd.util.hash_pandas_object(
        stable_key, index=False
    ).to_numpy()
    sampled = (
        sampled.sort_values(["date", "_audit_order", "symbol"])
        .groupby("date", sort=False, group_keys=False)
        .head(max_rows_per_date)
        .drop(columns="_audit_order")
    )
    return sampled.sort_values(KEYS).reset_index(drop=True)


def audit_features(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    return_target: str = "forward_excess_return_5d",
    classification_target: str = "outperform_spy_5d",
    membership: pd.DataFrame | None = None,
    redundancy_threshold: float = 0.9,
) -> FeatureAuditResult:
    """Audit features using training estimates and forward validation evidence."""
    data = _prepare_data(
        features, labels, membership, return_target, classification_target
    )
    feature_columns = _feature_columns(data, return_target, classification_target)
    quality_rows, predictive_rows, ablation_rows = [], [], []
    for split in splits.itertuples(index=False):
        train = _between(data, split.train_start_date, split.train_end_date)
        validation = _between(
            data, split.validation_start_date, split.validation_end_date
        )
        for feature in feature_columns:
            family = feature_family(feature)
            quality_rows.append(
                _quality_row(train, validation, feature, family, split.split_id)
            )
            predictive_rows.extend(
                _predictive_rows(
                    train,
                    validation,
                    feature,
                    family,
                    split.split_id,
                    return_target,
                    classification_target,
                )
            )
        ablation_rows.extend(
            _family_ablation_rows(
                train, validation, feature_columns, split.split_id, return_target
            )
        )
    predictive = pd.DataFrame(predictive_rows)
    quality = pd.DataFrame(quality_rows)
    ablations = pd.DataFrame(ablation_rows)
    stability = _stability_summary(predictive)
    redundancy = _redundancy_report(data, splits, feature_columns, redundancy_threshold)
    exposures = _exposure_report(data, feature_columns)
    return FeatureAuditResult(
        quality=quality,
        predictive=predictive,
        stability=stability,
        redundancy=redundancy,
        ablations=ablations,
        exposures=exposures,
        decisions=_feature_decisions(
            quality, stability, redundancy, ablations, exposures
        ),
    )


def _feature_decisions(quality, stability, redundancy, ablations, exposures):
    """Create an auditable research disposition from development-only evidence."""
    if stability.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "family",
                "decision",
                "reason",
                "validation_median_ic",
                "sign_agreement",
                "validation_missing_rate",
                "median_psi",
                "redundant_pair_count",
            ]
        )
    quality_summary = quality.groupby(["feature", "family"], as_index=False).agg(
        validation_missing_rate=("validation_missing_rate", "median"),
        median_psi=("distribution_psi", "median"),
    )
    pair_counts: dict[str, int] = {}
    for column in ["feature_a", "feature_b"]:
        if column in redundancy:
            for feature, count in redundancy[column].value_counts().items():
                pair_counts[str(feature)] = pair_counts.get(str(feature), 0) + int(
                    count
                )
    family_value = (
        ablations.groupby("family")["drop_family_delta_ic"].median()
        if not ablations.empty
        else pd.Series(dtype=float)
    )
    exposure_summary = (
        exposures.groupby("feature")["mean_exposure_range"].max()
        if not exposures.empty
        else pd.Series(dtype=float)
    )
    decisions = stability.merge(quality_summary, on=["feature", "family"], how="left")
    decisions["redundant_pair_count"] = (
        decisions["feature"].map(pair_counts).fillna(0).astype(int)
    )
    decisions["family_delta_ic"] = decisions["family"].map(family_value)
    decisions["max_group_exposure_range"] = decisions["feature"].map(exposure_summary)

    def classify(row):
        if row.validation_missing_rate > 0.5:
            return "insufficient_coverage", "median validation missingness exceeds 50%"
        if row.stable_evidence and row.redundant_pair_count == 0:
            return "retain", "stable validation evidence without high redundancy"
        if row.stable_evidence:
            return "transform", "stable evidence but materially redundant"
        if row.sign_agreement >= 0.6 and abs(row.validation_median_ic) >= 0.005:
            return "conditional", "weak evidence merits regime or interaction testing"
        return "remove", "no stable incremental development evidence"

    classified = decisions.apply(classify, axis=1)
    decisions["decision"] = classified.map(lambda value: value[0])
    decisions["reason"] = classified.map(lambda value: value[1])
    return (
        decisions[
            [
                "feature",
                "family",
                "decision",
                "reason",
                "validation_median_ic",
                "sign_agreement",
                "validation_missing_rate",
                "median_psi",
                "redundant_pair_count",
                "family_delta_ic",
                "max_group_exposure_range",
            ]
        ]
        .sort_values(["decision", "family", "feature"])
        .reset_index(drop=True)
    )


def feature_family(feature: str) -> str:
    lowered = feature.lower()
    for family, prefixes in FAMILY_PREFIXES.items():
        if lowered.startswith(prefixes):
            return family
    return "technical"


def _prepare_data(features, labels, membership, return_target, classification_target):
    for name, frame, required in [
        ("Features", features, set(KEYS)),
        ("Labels", labels, set(KEYS + [return_target, classification_target])),
    ]:
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} are missing: " + ", ".join(sorted(missing)))
    left, right = (
        features.copy(),
        labels[KEYS + [return_target, classification_target]].copy(),
    )
    for frame in [left, right]:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    data = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if membership is not None:
        members = membership.copy()
        members["symbol"] = members["symbol"].astype(str).str.upper()
        members["date"] = pd.to_datetime(
            members["date"], errors="coerce"
        ).dt.normalize()
        columns = (
            KEYS
            + ["is_member"]
            + [c for c in ["sector", "size_bucket"] if c in members]
        )
        members = members.loc[members["is_member"].eq(True), columns].drop(
            columns="is_member"
        )
        data = data.merge(
            members,
            on=KEYS,
            how="inner",
            validate="one_to_one",
            suffixes=("", "_universe"),
        )
    return data.sort_values(KEYS).reset_index(drop=True)


def _feature_columns(data, return_target, classification_target):
    excluded = set(
        KEYS + [return_target, classification_target, "sector", "size_bucket"]
    )
    return [
        c
        for c in data
        if c not in excluded and (is_numeric_dtype(data[c]) or is_bool_dtype(data[c]))
    ]


def _between(data, start, end):
    return data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end))]


def _quality_row(train, validation, feature, family, split_id):
    train_numeric = pd.to_numeric(train[feature], errors="coerce").astype(float)
    valid_numeric = pd.to_numeric(validation[feature], errors="coerce").astype(float)
    train_non_finite = train_numeric.notna() & ~np.isfinite(train_numeric)
    valid_non_finite = valid_numeric.notna() & ~np.isfinite(valid_numeric)
    train_values = train_numeric.mask(train_non_finite)
    valid_values = valid_numeric.mask(valid_non_finite)
    previous = train.sort_values(KEYS).groupby("symbol")[feature].shift()
    comparable = train[feature].notna() & previous.notna()
    stale = train.loc[comparable, feature].eq(previous.loc[comparable]).mean()
    revision_column = f"{feature}_revision_count"
    revision_tracking = revision_column in train
    return {
        "split_id": int(split_id),
        "feature": feature,
        "family": family,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_missing_rate": float(train_values.isna().mean()),
        "validation_missing_rate": float(valid_values.isna().mean()),
        "train_non_finite_rate": float(train_non_finite.mean()),
        "validation_non_finite_rate": float(valid_non_finite.mean()),
        "train_symbol_coverage": int(
            train.loc[train_values.notna(), "symbol"].nunique()
        ),
        "validation_symbol_coverage": int(
            validation.loc[valid_values.notna(), "symbol"].nunique()
        ),
        "train_stale_rate": float(stale) if pd.notna(stale) else np.nan,
        "revision_tracking_available": revision_tracking,
        "train_revision_count": (
            float(pd.to_numeric(train[revision_column], errors="coerce").sum())
            if revision_tracking
            else np.nan
        ),
        "distribution_psi": _psi(train_values, valid_values),
    }


def _psi(train, validation, bins=10):
    train, validation = train.dropna(), validation.dropna()
    if train.nunique() < 2 or validation.empty:
        return np.nan
    edges = np.unique(train.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    expected = pd.cut(train, edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    actual = pd.cut(validation, edges, include_lowest=True).value_counts(
        normalize=True, sort=False
    )
    expected, actual = expected.clip(lower=1e-6), actual.clip(lower=1e-6)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def _predictive_rows(
    train, validation, feature, family, split_id, return_target, classification_target
):
    rows = []
    for period, frame in [("train", train), ("validation", validation)]:
        rows.append(
            {
                "split_id": int(split_id),
                "period": period,
                "feature": feature,
                "family": family,
                "rank_ic": _daily_rank_ic(frame, feature, return_target),
                "classification_association": _correlation(
                    frame[feature], frame[classification_target]
                ),
                "observations": int(frame[[feature, return_target]].dropna().shape[0]),
            }
        )
    return rows


def _daily_rank_ic(frame, feature, target):
    values = []
    for _, day in frame.groupby("date"):
        value = _correlation(day[feature], day[target])
        if pd.notna(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _correlation(left, right):
    pair = pd.DataFrame(
        {
            "left": _finite_numeric(left),
            "right": _finite_numeric(right),
        }
    ).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return float(pair["left"].rank().corr(pair["right"].rank()))


def _stability_summary(predictive):
    if predictive.empty:
        return pd.DataFrame()
    pivot = predictive.pivot(
        index=["split_id", "feature", "family"], columns="period", values="rank_ic"
    ).reset_index()
    pivot["same_sign"] = np.sign(pivot.get("train")) == np.sign(pivot.get("validation"))
    summary = (
        pivot.groupby(["feature", "family"])
        .agg(
            folds=("split_id", "nunique"),
            train_median_ic=("train", "median"),
            validation_median_ic=("validation", "median"),
            sign_agreement=("same_sign", "mean"),
        )
        .reset_index()
    )
    summary["stable_evidence"] = (
        (summary["folds"] >= 2)
        & (summary["sign_agreement"] >= 0.6)
        & summary["validation_median_ic"].abs().ge(0.01)
    )
    return summary


def _redundancy_report(data, splits, features, threshold):
    rows = []
    for split in splits.itertuples(index=False):
        train = _between(data, split.train_start_date, split.train_end_date)[
            features
        ].apply(_finite_numeric)
        train = train.corr(method="spearman")
        for i, left in enumerate(features):
            for right in features[i + 1 :]:
                corr = train.at[left, right]
                if pd.notna(corr) and abs(corr) >= threshold:
                    rows.append(
                        {
                            "split_id": int(split.split_id),
                            "feature_a": left,
                            "feature_b": right,
                            "spearman_correlation": float(corr),
                        }
                    )
    return pd.DataFrame(
        rows, columns=["split_id", "feature_a", "feature_b", "spearman_correlation"]
    )


def _family_ablation_rows(train, validation, features, split_id, target):
    families = sorted({feature_family(feature) for feature in features})
    scores = {
        family: _family_validation_ic(
            train,
            validation,
            [f for f in features if feature_family(f) == family],
            target,
        )
        for family in families
    }
    all_score = _family_validation_ic(train, validation, features, target)
    rows = []
    for family, score in scores.items():
        without = _family_validation_ic(
            train,
            validation,
            [f for f in features if feature_family(f) != family],
            target,
        )
        delta = (
            all_score - without if pd.notna(all_score) and pd.notna(without) else np.nan
        )
        verdict = (
            "no_measurable_value"
            if pd.isna(delta) or abs(delta) < 0.005
            else ("helps" if delta > 0 else "hurts")
        )
        rows.append(
            {
                "split_id": int(split_id),
                "family": family,
                "family_only_validation_ic": score,
                "all_features_validation_ic": all_score,
                "drop_family_delta_ic": delta,
                "verdict": verdict,
            }
        )
    return rows


def _family_validation_ic(train, validation, features, target):
    weights = {f: _daily_rank_ic(train, f, target) for f in features}
    weights = {f: w for f, w in weights.items() if pd.notna(w) and w != 0}
    if not weights:
        return np.nan
    scored = validation[["date", target] + list(weights)].copy()
    scored[list(weights)] = scored[list(weights)].apply(_finite_numeric)
    components = [
        scored.groupby("date")[f].rank(pct=True).mul(w) for f, w in weights.items()
    ]
    scored["score"] = pd.concat(components, axis=1).sum(axis=1, min_count=1) / sum(
        abs(w) for w in weights.values()
    )
    return _daily_rank_ic(scored, "score", target)


def _exposure_report(data, features):
    rows = []
    for group in ["sector", "size_bucket"]:
        if group not in data:
            continue
        for feature in features:
            values = _finite_numeric(data[feature])
            means = (
                data.assign(_value=values).groupby(group, dropna=True)["_value"].mean()
            )
            missing = (
                data.assign(_missing=values.isna())
                .groupby(group, dropna=True)["_missing"]
                .mean()
            )
            rows.append(
                {
                    "feature": feature,
                    "family": feature_family(feature),
                    "grouping": group,
                    "mean_exposure_range": float(means.max() - means.min())
                    if len(means)
                    else np.nan,
                    "missingness_rate_range": float(missing.max() - missing.min())
                    if len(missing)
                    else np.nan,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "family",
            "grouping",
            "mean_exposure_range",
            "missingness_rate_range",
        ],
    )


def _finite_numeric(values):
    """Coerce values to numbers and represent positive/negative infinity as missing."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric))
