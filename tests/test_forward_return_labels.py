import pandas as pd
import pytest

from market_qml.labels.forward_returns import (
    build_forward_return_label_table,
    build_forward_return_labels,
)


def _price_rows(symbol: str, closes: list[float]) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "close": close,
        }
        for i, close in enumerate(closes)
    ]


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        _price_rows("AAPL", [100, 110, 120, 130, 140, 150])
        + _price_rows("MSFT", [200, 210, 220, 230, 240, 250])
        + _price_rows("SPY", [400, 404, 408, 412, 416, 420])
    )


def test_build_forward_return_labels_computes_excess_return_and_binary_label():
    result = build_forward_return_labels(_prices(), horizon=5)
    aapl = result[result["symbol"] == "AAPL"].iloc[0]
    msft = result[result["symbol"] == "MSFT"].iloc[0]
    spy = result[result["symbol"] == "SPY"].iloc[0]

    assert aapl["forward_return_5d"] == pytest.approx(0.50)
    assert aapl["spy_forward_return_5d"] == pytest.approx(0.05)
    assert aapl["forward_excess_return_5d"] == pytest.approx(0.45)
    assert aapl["outperform_spy_5d"] == 1
    assert msft["forward_return_5d"] == pytest.approx(0.25)
    assert msft["outperform_spy_5d"] == 1
    assert spy["forward_excess_return_5d"] == pytest.approx(0.0)
    assert spy["outperform_spy_5d"] == 0


def test_build_forward_return_labels_drops_incomplete_horizon_by_default():
    result = build_forward_return_labels(_prices(), horizon=5)

    assert result["date"].unique().tolist() == [pd.Timestamp("2024-01-01")]
    assert len(result) == 3


def test_build_forward_return_labels_can_keep_missing_future_rows():
    result = build_forward_return_labels(_prices(), horizon=5, drop_missing=False)

    assert len(result) == 18
    last_aapl = result[result["symbol"] == "AAPL"].iloc[-1]
    assert pd.isna(last_aapl["forward_return_5d"])
    assert pd.isna(last_aapl["outperform_spy_5d"])


def test_build_forward_return_labels_supports_configurable_horizon():
    result = build_forward_return_labels(_prices(), horizon=2)
    aapl = result[
        (result["symbol"] == "AAPL")
        & (result["date"] == pd.Timestamp("2024-01-01"))
    ].iloc[0]

    assert aapl["label_horizon_days"] == 2
    assert aapl["forward_return_2d"] == pytest.approx(0.20)
    assert aapl["spy_forward_return_2d"] == pytest.approx(0.02)
    assert aapl["forward_excess_return_2d"] == pytest.approx(0.18)
    assert aapl["outperform_spy_2d"] == 1


def test_build_forward_return_labels_requires_schema_and_benchmark():
    with pytest.raises(ValueError, match="missing required columns"):
        build_forward_return_labels(pd.DataFrame({"symbol": ["AAPL"]}))

    with pytest.raises(ValueError, match="Benchmark symbol not found"):
        build_forward_return_labels(
            pd.DataFrame(_price_rows("AAPL", [100, 110])),
            horizon=1,
        )


def test_build_forward_return_labels_requires_positive_horizon():
    with pytest.raises(ValueError, match="Label horizon must be positive"):
        build_forward_return_labels(_prices(), horizon=0)


def test_build_forward_return_label_table_saves_output(tmp_path):
    price_path = tmp_path / "prices.parquet"
    output_path = tmp_path / "forward_return_labels.parquet"
    _prices().to_parquet(price_path, index=False)

    result = build_forward_return_label_table(
        price_path=price_path,
        output_path=output_path,
        horizon=5,
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert "forward_excess_return_5d" in saved.columns
