import pickle

import numpy as np
import pandas as pd
import pytest

from market_qml.models.preprocessing import fit_preprocessor
from market_qml.reporting.daily_signal import (
    DISCLAIMER,
    SIGNAL_COLUMNS,
    build_daily_signal_report,
    load_model,
    save_daily_signal_report,
)


class DemoModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probabilities = 1 / (1 + np.exp(-features["momentum"].to_numpy()))
        return np.column_stack([1 - probabilities, probabilities])


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "SPY", "AAPL", "MSFT", "NVDA", "SPY"],
            "date": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-05",
                ]
            ),
            "momentum": [-1.0, 0.0, 2.0, -2.0, 1.0, 0.0],
            "volatility": [0.2, 0.3, 0.2, 0.4, 0.3, 0.3],
        }
    )


def test_daily_signal_uses_latest_date_and_ranks_scores() -> None:
    features = _features()
    preprocessor = fit_preprocessor(features[["momentum", "volatility"]])

    report = build_daily_signal_report(
        features,
        model=DemoModel(),
        preprocessor=preprocessor,
        top_n=2,
        bottom_n=2,
    )

    assert report.signal_date == pd.Timestamp("2026-01-05")
    assert list(report.signals.columns) == SIGNAL_COLUMNS
    assert report.signals["symbol"].tolist() == ["AAPL", "NVDA", "SPY", "MSFT"]
    assert report.signals["rank"].tolist() == [1, 2, 3, 4]
    assert report.signals.loc[report.signals["symbol"] == "SPY", "is_benchmark"].item()
    assert "AAPL" in report.markdown
    assert "MSFT" in report.markdown
    assert "3 of 4" in report.markdown
    assert DISCLAIMER in report.markdown


def test_daily_signal_report_saves_markdown_and_csv(tmp_path) -> None:
    features = _features()
    report = build_daily_signal_report(
        features,
        model=DemoModel(),
        preprocessor=fit_preprocessor(features[["momentum", "volatility"]]),
    )
    markdown_path = tmp_path / "daily_signal.md"
    csv_path = tmp_path / "daily_signal.csv"

    save_daily_signal_report(
        report,
        markdown_path=markdown_path,
        csv_path=csv_path,
    )

    assert markdown_path.read_text(encoding="utf-8") == report.markdown
    saved = pd.read_csv(csv_path)
    assert saved["symbol"].tolist() == report.signals["symbol"].tolist()


def test_load_model_accepts_fitted_probability_model(tmp_path) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(DemoModel(), model_file)

    loaded = load_model(model_path)

    assert loaded.classes_.tolist() == [0, 1]
    assert callable(loaded.predict_proba)


@pytest.mark.parametrize("top_n,bottom_n", [(0, 1), (1, 0), (-1, 1)])
def test_daily_signal_report_rejects_invalid_display_counts(top_n, bottom_n) -> None:
    features = _features()
    with pytest.raises(ValueError, match="must both be positive"):
        build_daily_signal_report(
            features,
            model=DemoModel(),
            preprocessor=fit_preprocessor(features[["momentum", "volatility"]]),
            top_n=top_n,
            bottom_n=bottom_n,
        )


def test_daily_signal_report_requires_benchmark_on_latest_date() -> None:
    features = _features()
    features = features.loc[
        ~(
            features["symbol"].eq("SPY")
            & features["date"].eq(pd.Timestamp("2026-01-05"))
        )
    ]
    with pytest.raises(ValueError, match="absent"):
        build_daily_signal_report(
            features,
            model=DemoModel(),
            preprocessor=fit_preprocessor(features[["momentum", "volatility"]]),
        )
