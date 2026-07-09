import pandas as pd

from market_qml.reporting.classical_baselines import (
    build_classical_baseline_comparison,
    render_classical_baseline_report,
    save_classical_baseline_report,
    strongest_baseline,
)


def _classification() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["logistic_regression", "random_forest"],
            "scope": ["overall", "overall"],
            "roc_auc": [0.62, 0.68],
            "average_precision": [0.60, 0.66],
        }
    )


def _ranking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [
                "logistic_regression",
                "random_forest",
                "ridge_regression",
            ],
            "scope": ["overall", "overall", "overall"],
            "long_short_spread": [0.01, 0.03, 0.025],
            "rank_information_coefficient": [0.05, 0.10, 0.08],
        }
    )


def _portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [
                "logistic_regression",
                "random_forest",
                "ridge_regression",
            ],
            "scope": ["overall", "overall", "overall"],
            "cumulative_net_return": [0.12, 0.20, 0.18],
            "cumulative_net_excess_return": [0.03, 0.08, 0.07],
            "net_sharpe": [0.9, 1.2, 1.1],
            "net_max_drawdown": [-0.20, -0.10, -0.12],
        }
    )


def test_build_classical_baseline_comparison_adds_task_metadata():
    comparison = build_classical_baseline_comparison(
        classification_metrics=_classification(),
        ranking_metrics=_ranking(),
        portfolio_risk_metrics=_portfolio(),
        expected_models=["logistic_regression", "random_forest", "ridge_regression"],
    )

    assert comparison["model_name"].tolist() == [
        "random_forest",
        "ridge_regression",
        "logistic_regression",
    ]
    assert strongest_baseline(comparison) == "random_forest"
    assert comparison.loc[0, "classification_roc_auc"] == 0.68
    assert comparison.loc[0, "portfolio_net_sharpe"] == 1.2
    ridge = comparison[comparison["model_name"] == "ridge_regression"].iloc[0]
    assert ridge["prediction_task"] == "regression_ranking"
    assert ridge["target_name"] == "forward_excess_return_5d"
    assert ridge["score_used_for_ranking"] == "predicted_forward_excess_return"
    assert pd.isna(ridge["classification_roc_auc"])


def test_render_classical_baseline_report_warns_about_missing_expected_models():
    comparison = build_classical_baseline_comparison(
        classification_metrics=_classification().iloc[[0]],
        ranking_metrics=_ranking().iloc[[0]],
        portfolio_risk_metrics=_portfolio().iloc[[0]],
    )

    report = render_classical_baseline_report(comparison)

    assert "Strongest available baseline: **logistic_regression**" in report
    assert "Missing expected baselines" in report
    assert "random_forest" in report
    assert "elastic_net" in report
    assert "classifier-only metrics are shown as `NA`" in report
    assert "| model_name |" in report


def test_render_classical_baseline_report_marks_not_applicable_metrics():
    comparison = build_classical_baseline_comparison(
        classification_metrics=_classification(),
        ranking_metrics=_ranking(),
        portfolio_risk_metrics=_portfolio(),
        expected_models=["ridge_regression"],
    )

    report = render_classical_baseline_report(
        comparison,
        expected_models=["ridge_regression"],
    )

    assert "regression_ranking" in report
    assert "| ridge_regression |" in report
    assert "| ridge_regression | linear | regression_ranking" in report
    assert "NA" in report


def test_save_classical_baseline_report_outputs_files(tmp_path):
    comparison = build_classical_baseline_comparison(
        classification_metrics=_classification(),
        ranking_metrics=_ranking(),
        portfolio_risk_metrics=_portfolio(),
        expected_models=["logistic_regression", "random_forest", "ridge_regression"],
    )
    markdown = render_classical_baseline_report(comparison)
    comparison_path = tmp_path / "comparison.parquet"
    markdown_path = tmp_path / "comparison.md"

    save_classical_baseline_report(
        comparison=comparison,
        markdown=markdown,
        comparison_output=comparison_path,
        markdown_output=markdown_path,
    )

    saved = pd.read_parquet(comparison_path)

    assert comparison_path.exists()
    assert markdown_path.exists()
    assert saved["model_name"].tolist() == comparison["model_name"].tolist()
    assert markdown_path.read_text(encoding="utf-8") == markdown
