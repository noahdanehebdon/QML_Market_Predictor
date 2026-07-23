"""Reproducible controls for judging complex cross-sectional rankers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from market_qml.models.predictions import build_prediction_table

TARGET = "forward_excess_return_5d"


@dataclass(frozen=True)
class NaiveRankResult:
    model: object
    predictions: pd.DataFrame
    parameters: dict[str, object]


def train_momentum_rank(data, *, split_id=0):
    feature = next(
        (
            c
            for c in ["return_20d", "return_10d", "return_5d", "return_1d"]
            if c in data.validation.X
        ),
        data.validation.X.columns[0],
    )
    return _result(
        data,
        data.validation.X[feature],
        "momentum_rank",
        split_id,
        {"feature": feature},
    )


def train_sign_rank(data, *, split_id=0):
    feature = next(
        (c for c in ["return_20d", "return_5d", "return_1d"] if c in data.validation.X),
        data.validation.X.columns[0],
    )
    return _result(
        data,
        np.sign(data.validation.X[feature]),
        "sign_rank",
        split_id,
        {"feature": feature},
    )


def train_random_rank(data, *, split_id=0):
    keys = (
        data.validation.metadata["symbol"].astype(str)
        + "|"
        + data.validation.metadata["date"].astype(str)
        + f"|{split_id}"
    )
    scores = keys.map(
        lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:12], 16) / 16**12
    )
    return _result(data, scores, "random_rank", split_id, {"seed": "sha256"})


def train_linear_rank(data, *, split_id=0):
    model = Ridge(alpha=1.0).fit(data.train.X, data.train.y)
    return _result(
        data,
        model.predict(data.validation.X),
        "linear_rank",
        split_id,
        {"alpha": 1.0},
        model,
    )


def train_sector_neutral_rank(data, *, split_id=0):
    base = train_linear_rank(data, split_id=split_id)
    frame = data.validation.metadata[["date"]].copy()
    frame["sector"] = (
        data.validation.metadata.get("sector", "unknown").fillna("unknown")
        if "sector" in data.validation.metadata
        else "unknown"
    )
    frame["score"] = base.predictions["y_score"].to_numpy()
    neutral = frame["score"] - frame.groupby(["date", "sector"])["score"].transform(
        "mean"
    )
    return _result(
        data,
        neutral,
        "sector_neutral_rank",
        split_id,
        {"base": "linear_rank"},
        base.model,
    )


def _result(data, score, name, split_id, parameters, model=None):
    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=score,
        model_name=name,
        split_id=split_id,
    )
    return NaiveRankResult(model, predictions, {"model": name, **parameters})
