"""Practical and statistical evidence versus naive ranking controls."""

import numpy as np
import pandas as pd

NAIVE_MODELS = {"sign_rank", "momentum_rank", "random_rank", "sector_neutral_rank", "linear_rank"}


def compare_to_naive_baselines(predictions, *, practical_margin=0.01, bootstrap_samples=2000, seed=42):
    metrics = []
    for (model, split), frame in predictions.groupby(["model_name", "split_id"]):
        daily = frame.groupby("date").apply(_rank_ic, include_groups=False)
        metrics.append({"model_name": model, "split_id": split, "rank_ic": daily.mean()})
    metrics = pd.DataFrame(metrics)
    columns = ["model_name", "naive_model", "mean_rank_ic_margin", "ci_lower", "ci_upper", "practical_margin", "beats_naive"]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    naive = metrics.loc[metrics["model_name"].isin(NAIVE_MODELS)]
    if naive.empty:
        return pd.DataFrame(columns=columns)
    benchmark = naive.groupby("model_name")["rank_ic"].mean().idxmax()
    benchmark_rows = naive.loc[naive["model_name"].eq(benchmark), ["split_id", "rank_ic"]].rename(columns={"rank_ic": "naive_ic"})
    rng, rows = np.random.default_rng(seed), []
    for model in sorted(set(metrics["model_name"]) - NAIVE_MODELS):
        paired = metrics.loc[metrics["model_name"].eq(model), ["split_id", "rank_ic"]].merge(benchmark_rows, on="split_id").dropna()
        delta = (paired["rank_ic"] - paired["naive_ic"]).to_numpy()
        if len(delta):
            samples = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
            mean, (lower, upper) = delta.mean(), np.quantile(samples, [0.025, 0.975])
        else:
            mean = lower = upper = np.nan
        rows.append({"model_name": model, "naive_model": benchmark, "mean_rank_ic_margin": mean, "ci_lower": lower, "ci_upper": upper, "practical_margin": practical_margin, "beats_naive": bool(pd.notna(mean) and mean >= practical_margin and lower > 0)})
    return pd.DataFrame(rows, columns=columns)


def _rank_ic(day):
    if day["y_score"].nunique() < 2 or day["forward_excess_return"].nunique() < 2:
        return np.nan
    return day["y_score"].corr(day["forward_excess_return"], method="spearman")
