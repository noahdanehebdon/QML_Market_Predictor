"""Leakage-safe feature quality, drift, and predictive-stability audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


KEYS = ["symbol", "date"]
FAMILY_PREFIXES = {
    "benchmark": ("spy_", "benchmark_", "beta_", "excess_", "relative_"),
    "macro": ("treasury_", "fed_", "cpi_", "unemployment_", "industrial_", "yield_"),
    "fundamentals": ("fundamental_", "revenue", "income", "assets", "liabilities", "equity", "eps", "pe_", "market_cap"),
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
    data = _prepare_data(features, labels, membership, return_target, classification_target)
    feature_columns = _feature_columns(data, return_target, classification_target)
    quality_rows, predictive_rows, ablation_rows = [], [], []
    for split in splits.itertuples(index=False):
        train = _between(data, split.train_start_date, split.train_end_date)
        validation = _between(data, split.validation_start_date, split.validation_end_date)
        for feature in feature_columns:
            family = feature_family(feature)
            quality_rows.append(_quality_row(train, validation, feature, family, split.split_id))
            predictive_rows.extend(
                _predictive_rows(train, validation, feature, family, split.split_id, return_target, classification_target)
            )
        ablation_rows.extend(
            _family_ablation_rows(train, validation, feature_columns, split.split_id, return_target)
        )
    predictive = pd.DataFrame(predictive_rows)
    return FeatureAuditResult(
        quality=pd.DataFrame(quality_rows),
        predictive=predictive,
        stability=_stability_summary(predictive),
        redundancy=_redundancy_report(data, splits, feature_columns, redundancy_threshold),
        ablations=pd.DataFrame(ablation_rows),
        exposures=_exposure_report(data, feature_columns),
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
    left, right = features.copy(), labels[KEYS + [return_target, classification_target]].copy()
    for frame in [left, right]:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    data = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if membership is not None:
        members = membership.copy()
        members["symbol"] = members["symbol"].astype(str).str.upper()
        members["date"] = pd.to_datetime(members["date"], errors="coerce").dt.normalize()
        columns = KEYS + ["is_member"] + [c for c in ["sector", "size_bucket"] if c in members]
        members = members.loc[members["is_member"].eq(True), columns].drop(columns="is_member")
        data = data.merge(members, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_universe"))
    return data.sort_values(KEYS).reset_index(drop=True)


def _feature_columns(data, return_target, classification_target):
    excluded = set(KEYS + [return_target, classification_target, "sector", "size_bucket"])
    return [c for c in data if c not in excluded and (is_numeric_dtype(data[c]) or is_bool_dtype(data[c]))]


def _between(data, start, end):
    return data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end))]


def _quality_row(train, validation, feature, family, split_id):
    train_values = pd.to_numeric(train[feature], errors="coerce").astype(float)
    valid_values = pd.to_numeric(validation[feature], errors="coerce").astype(float)
    previous = train.sort_values(KEYS).groupby("symbol")[feature].shift()
    comparable = train[feature].notna() & previous.notna()
    stale = train.loc[comparable, feature].eq(previous.loc[comparable]).mean()
    revision_column = f"{feature}_revision_count"
    revision_tracking = revision_column in train
    return {
        "split_id": int(split_id), "feature": feature, "family": family,
        "train_rows": len(train), "validation_rows": len(validation),
        "train_missing_rate": float(train_values.isna().mean()),
        "validation_missing_rate": float(valid_values.isna().mean()),
        "train_symbol_coverage": int(train.loc[train_values.notna(), "symbol"].nunique()),
        "validation_symbol_coverage": int(validation.loc[valid_values.notna(), "symbol"].nunique()),
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
    expected = pd.cut(train, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    actual = pd.cut(validation, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    expected, actual = expected.clip(lower=1e-6), actual.clip(lower=1e-6)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def _predictive_rows(train, validation, feature, family, split_id, return_target, classification_target):
    rows = []
    for period, frame in [("train", train), ("validation", validation)]:
        rows.append({
            "split_id": int(split_id), "period": period, "feature": feature, "family": family,
            "rank_ic": _daily_rank_ic(frame, feature, return_target),
            "classification_association": _correlation(frame[feature], frame[classification_target]),
            "observations": int(frame[[feature, return_target]].dropna().shape[0]),
        })
    return rows


def _daily_rank_ic(frame, feature, target):
    values = []
    for _, day in frame.groupby("date"):
        value = _correlation(day[feature], day[target])
        if pd.notna(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _correlation(left, right):
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return float(pair["left"].rank().corr(pair["right"].rank()))


def _stability_summary(predictive):
    if predictive.empty:
        return pd.DataFrame()
    pivot = predictive.pivot(index=["split_id", "feature", "family"], columns="period", values="rank_ic").reset_index()
    pivot["same_sign"] = np.sign(pivot.get("train")) == np.sign(pivot.get("validation"))
    summary = pivot.groupby(["feature", "family"]).agg(
        folds=("split_id", "nunique"), train_median_ic=("train", "median"),
        validation_median_ic=("validation", "median"), sign_agreement=("same_sign", "mean"),
    ).reset_index()
    summary["stable_evidence"] = (summary["folds"] >= 2) & (summary["sign_agreement"] >= 0.6) & summary["validation_median_ic"].abs().ge(0.01)
    return summary


def _redundancy_report(data, splits, features, threshold):
    rows = []
    for split in splits.itertuples(index=False):
        train = _between(data, split.train_start_date, split.train_end_date)[features].corr(method="spearman")
        for i, left in enumerate(features):
            for right in features[i + 1:]:
                corr = train.at[left, right]
                if pd.notna(corr) and abs(corr) >= threshold:
                    rows.append({"split_id": int(split.split_id), "feature_a": left, "feature_b": right, "spearman_correlation": float(corr)})
    return pd.DataFrame(rows, columns=["split_id", "feature_a", "feature_b", "spearman_correlation"])


def _family_ablation_rows(train, validation, features, split_id, target):
    families = sorted({feature_family(feature) for feature in features})
    scores = {family: _family_validation_ic(train, validation, [f for f in features if feature_family(f) == family], target) for family in families}
    all_score = _family_validation_ic(train, validation, features, target)
    rows = []
    for family, score in scores.items():
        without = _family_validation_ic(train, validation, [f for f in features if feature_family(f) != family], target)
        delta = all_score - without if pd.notna(all_score) and pd.notna(without) else np.nan
        verdict = "no_measurable_value" if pd.isna(delta) or abs(delta) < 0.005 else ("helps" if delta > 0 else "hurts")
        rows.append({"split_id": int(split_id), "family": family, "family_only_validation_ic": score, "all_features_validation_ic": all_score, "drop_family_delta_ic": delta, "verdict": verdict})
    return rows


def _family_validation_ic(train, validation, features, target):
    weights = {f: _daily_rank_ic(train, f, target) for f in features}
    weights = {f: w for f, w in weights.items() if pd.notna(w) and w != 0}
    if not weights:
        return np.nan
    scored = validation[["date", target] + list(weights)].copy()
    components = [scored.groupby("date")[f].rank(pct=True).mul(w) for f, w in weights.items()]
    scored["score"] = pd.concat(components, axis=1).sum(axis=1, min_count=1) / sum(abs(w) for w in weights.values())
    return _daily_rank_ic(scored, "score", target)


def _exposure_report(data, features):
    rows = []
    for group in ["sector", "size_bucket"]:
        if group not in data:
            continue
        for feature in features:
            values = pd.to_numeric(data[feature], errors="coerce")
            means = data.assign(_value=values).groupby(group, dropna=True)["_value"].mean()
            missing = data.assign(_missing=values.isna()).groupby(group, dropna=True)["_missing"].mean()
            rows.append({"feature": feature, "family": feature_family(feature), "grouping": group, "mean_exposure_range": float(means.max() - means.min()) if len(means) else np.nan, "missingness_rate_range": float(missing.max() - missing.min()) if len(missing) else np.nan})
    return pd.DataFrame(rows, columns=["feature", "family", "grouping", "mean_exposure_range", "missingness_rate_range"])
