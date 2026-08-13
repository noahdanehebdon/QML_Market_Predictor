import pandas as pd

from market_qml.labels.integrity import add_return_integrity_flags


def test_return_integrity_flags_extremes_without_clipping_values():
    labels = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "date": pd.to_datetime(["2024-01-01"] * 4),
            "forward_return_10d": [0.02, 0.03, 0.01, 4.0],
        }
    )

    result = add_return_integrity_flags(labels, return_column="forward_return_10d")

    assert result.loc[result.symbol.eq("D"), "return_integrity_valid"].item() is False
    assert result.loc[result.symbol.eq("D"), "forward_return_10d"].item() == 4.0
    assert (
        result.loc[result.symbol.eq("D"), "return_integrity_status"].item()
        == "extreme_absolute_return"
    )
