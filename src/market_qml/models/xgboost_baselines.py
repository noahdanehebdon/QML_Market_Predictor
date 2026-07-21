"""Task-aligned XGBoost classification and LambdaMART ranking baselines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from xgboost import XGBClassifier, XGBRanker

from market_qml.models.predictions import build_prediction_table
from market_qml.models.preprocessing import PreprocessedTrainValidation


CLASSIFIER_NAME = "xgboost_classifier"
RANKER_NAME = "xgboost_ranker"
RANK_TARGET = "forward_excess_return_5d"


@dataclass(frozen=True)
class XGBoostResult:
    model: object
    predictions: pd.DataFrame
    parameters: dict[str, object]
    selection_diagnostics: pd.DataFrame


def train_xgboost_classifier(data: PreprocessedTrainValidation, *, split_id=0, model_name=CLASSIFIER_NAME, random_state=42):
    inner_train, inner_valid = _inner_masks(data.train.metadata)
    weights = _date_weights(data.train.metadata.loc[inner_train, "date"])
    model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=30, random_state=random_state, n_jobs=1)
    model.fit(data.train.X.loc[inner_train], data.train.y.loc[inner_train].astype(int), sample_weight=weights, eval_set=[(data.train.X.loc[inner_valid], data.train.y.loc[inner_valid].astype(int))], verbose=False)
    score = model.predict_proba(data.validation.X)[:, 1]
    predictions = build_prediction_table(metadata=data.validation.metadata, y_true=data.validation.y, y_score=score, model_name=model_name, split_id=split_id)
    best = int(model.best_iteration)
    diagnostics = pd.DataFrame([{"split_id": split_id, "objective": "logloss", "best_iteration": best, "inner_validation_logloss": float(model.best_score)}])
    return XGBoostResult(model, predictions, {"model": model_name, "best_iteration": best, "date_aware_weights": True}, diagnostics)


def train_xgboost_ranker(data: PreprocessedTrainValidation, *, split_id=0, model_name=RANKER_NAME, random_state=42):
    inner_train, inner_valid = _inner_masks(data.train.metadata)
    train = _rank_frame(data.train.X.loc[inner_train], data.train.y.loc[inner_train], data.train.metadata.loc[inner_train])
    valid = _rank_frame(data.train.X.loc[inner_valid], data.train.y.loc[inner_valid], data.train.metadata.loc[inner_valid])
    model = XGBRanker(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="rank:ndcg", eval_metric="ndcg", early_stopping_rounds=30, lambdarank_num_pair_per_sample=8, random_state=random_state, n_jobs=1)
    model.fit(train[0], train[1], qid=train[2], eval_set=[(valid[0], valid[1])], eval_qid=[valid[2]], verbose=False)
    score = model.predict(data.validation.X)
    predictions = build_prediction_table(metadata=data.validation.metadata, y_true=data.validation.y, y_score=score, model_name=model_name, split_id=split_id)
    best = int(model.best_iteration)
    diagnostics = pd.DataFrame([{"split_id": split_id, "objective": "rank:ndcg", "best_iteration": best, "inner_validation_ndcg": float(model.best_score)}])
    return XGBoostResult(model, predictions, {"model": model_name, "best_iteration": best, "grouped_by": "date"}, diagnostics)


def _inner_masks(metadata, fraction=0.2):
    dates = pd.DatetimeIndex(pd.to_datetime(metadata["date"]).unique()).sort_values()
    if len(dates) < 3:
        raise ValueError("Training data has too few dates for early stopping.")
    cutoff = dates[max(1, int(len(dates) * (1 - fraction)))]
    normalized = pd.to_datetime(metadata["date"])
    return normalized.lt(cutoff), normalized.ge(cutoff)


def _date_weights(dates):
    dates = pd.to_datetime(dates)
    counts = dates.value_counts()
    weights = dates.map(1 / counts)
    return (weights / weights.mean()).to_numpy()


def _rank_frame(X, y, metadata):
    order = pd.to_datetime(metadata["date"]).sort_values(kind="stable").index
    ordered_y = pd.to_numeric(y.loc[order], errors="coerce")
    relevance = ordered_y.groupby(pd.to_datetime(metadata.loc[order, "date"])).rank(pct=True).mul(4).round().astype(int)
    qid = pd.factorize(pd.to_datetime(metadata.loc[order, "date"]), sort=True)[0]
    return X.loc[order], relevance, qid
