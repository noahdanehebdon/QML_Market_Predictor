import pandas as pd
import pytest

from market_qml.reporting.model_comparison import (
    build_model_comparison_report,
    save_model_comparison_report,
)

MODELS = ["logistic_regression", "gradient_boosting", "vqc", "qsvm", "qcnn"]


def _aggregate_metrics() -> pd.DataFrame:
    rows = []
    aucs = dict(zip(MODELS, [0.62, 0.60, 0.55, 0.57, 0.58]))
    for model in MODELS:
        values = {
            "accuracy": aucs[model] - 0.05,
            "roc_auc": aucs[model],
            "log_loss": 1.0 - aucs[model],
            "brier_score": 0.5 - aucs[model] / 3,
        }
        for metric, mean in values.items():
            rows.append(
                {
                    "model_name": model,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": mean - 0.03,
                    "ci_upper": mean + 0.03,
                }
            )
    return pd.DataFrame(rows)


def _ranking_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": MODELS,
            "scope": ["overall"] * len(MODELS),
            "rank_information_coefficient": [0.10, 0.08, 0.04, 0.07, 0.05],
            "long_short_spread": [0.03, 0.025, 0.01, 0.02, 0.015],
        }
    )


def _portfolio_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": MODELS,
            "scope": ["overall"] * len(MODELS),
            "cumulative_net_return": [0.30, 0.25, 0.12, 0.20, 0.18],
            "cumulative_net_excess_return": [0.12, 0.10, -0.01, 0.05, 0.03],
            "net_sharpe": [1.4, 1.2, 0.7, 1.0, 0.9],
            "net_max_drawdown": [-0.08, -0.10, -0.18, -0.12, -0.14],
        }
    )


def _regime_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime_type": ["volatility_regime"] * 3,
            "regime": ["high_volatility"] * 3,
            "model_name": ["logistic_regression", "qsvm", "qcnn"],
            "rows": [120] * 3,
            "meets_minimum_rows": [True] * 3,
            "roc_auc": [0.61, 0.57, 0.59],
            "rank_information_coefficient": [0.04, 0.08, 0.06],
        }
    )


def test_build_model_comparison_identifies_leaders_and_limitations() -> None:
    report = build_model_comparison_report(
        aggregate_metrics=_aggregate_metrics(),
        ranking_metrics=_ranking_metrics(),
        portfolio_metrics=_portfolio_metrics(),
        regime_metrics=_regime_metrics(),
        expected_models=MODELS,
    )

    assert report.strongest_model == "logistic_regression"
    assert report.summary.iloc[0]["model_name"] == "logistic_regression"
    assert set(report.summary["model_family"]) == {"classical", "qml"}
    regime = report.regime_leaders.iloc[0]
    assert regime["classification_leader"] == "logistic_regression"
    assert regime["ranking_leader"] == "qsvm"
    assert "Strongest QML model" in report.markdown
    assert "## Limitations" in report.markdown
    assert "does not show a QML classification advantage" in report.markdown


def test_model_comparison_outputs_can_be_saved(tmp_path) -> None:
    report = build_model_comparison_report(
        aggregate_metrics=_aggregate_metrics(),
        ranking_metrics=_ranking_metrics(),
        portfolio_metrics=_portfolio_metrics(),
        regime_metrics=_regime_metrics(),
        expected_models=MODELS,
    )
    markdown = tmp_path / "comparison.md"
    summary = tmp_path / "comparison.csv"
    regimes = tmp_path / "regimes.csv"

    save_model_comparison_report(
        report,
        markdown_path=markdown,
        summary_path=summary,
        regime_path=regimes,
    )

    assert "Strongest model" in markdown.read_text(encoding="utf-8")
    assert pd.read_csv(summary)["model_name"].iloc[0] == "logistic_regression"
    assert pd.read_csv(regimes)["ranking_leader"].iloc[0] == "qsvm"


def test_small_regime_slices_are_not_reported_as_leaders() -> None:
    regimes = _regime_metrics()
    regimes["meets_minimum_rows"] = False
    report = build_model_comparison_report(
        aggregate_metrics=_aggregate_metrics(),
        ranking_metrics=_ranking_metrics(),
        portfolio_metrics=_portfolio_metrics(),
        regime_metrics=regimes,
        expected_models=MODELS,
    )

    assert report.regime_leaders.empty
    assert "No regime slices met" in report.markdown


def test_missing_metric_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_model_comparison_report(
            aggregate_metrics=_aggregate_metrics().drop(columns="ci_lower"),
            ranking_metrics=_ranking_metrics(),
            portfolio_metrics=_portfolio_metrics(),
            regime_metrics=_regime_metrics(),
            expected_models=MODELS,
        )
