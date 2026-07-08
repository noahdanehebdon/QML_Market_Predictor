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
            "model_name": ["logistic_regression", "random_forest"],
            "scope": ["overall", "overall"],
            "long_short_spread": [0.01, 0.03],
            "rank_information_coefficient": [0.05, 0.10],
        }
    )


def _portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["logistic_regression", "random_forest"],
            "scope": ["overall", "overall"],
            "cumulative_net_return": [0.12, 0.20],
            "cumulative_net_excess_return": [0.03, 0.08],
            "net_sharpe": [0.9, 1.2],
            "net_max_drawdown": [-0.20, -0.10],
        }
    )


def test_build_classical_baseline_comparison_identifies_strongest_baseline():
    comparison = build_classical_baseline_comparison(
        classification_metrics=_classification(),
        ranking_metrics=_ranking(),
        portfolio_risk_metrics=_portfolio(),
        expected_models=["logistic_regression", "random_forest"],
    )

    assert comparison["model_name"].tolist() == ["random_forest", "logistic_regression"]
    assert strongest_baseline(comparison) == "random_forest"
    assert comparison.loc[0, "classification_roc_auc"] == 0.68
    assert comparison.loc[0, "portfolio_net_sharpe"] == 1.2


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
    assert "| model_name |" in report


def test_save_classical_baseline_report_outputs_files(tmp_path):
    comparison = build_classical_baseline_comparison(
        classification_metrics=_classification(),
        ranking_metrics=_ranking(),
        portfolio_risk_metrics=_portfolio(),
        expected_models=["logistic_regression", "random_forest"],
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
