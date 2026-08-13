import pandas as pd

from market_qml.labels.residualized import build_residualized_target


def test_residualized_target_removes_same_date_sector_and_beta_exposure():
    rows = []
    for index in range(30):
        rows.append(
            {
                "symbol": f"S{index:02d}",
                "date": pd.Timestamp("2024-01-01"),
                "forward_excess_return_10d": index / 100,
                "rolling_beta_20d_vs_spy": index / 10,
                "sector": "tech" if index < 15 else "finance",
            }
        )
    labels = pd.DataFrame(rows)[["symbol", "date", "forward_excess_return_10d"]]
    exposures = pd.DataFrame(rows)[
        ["symbol", "date", "rolling_beta_20d_vs_spy", "sector"]
    ]

    result = build_residualized_target(
        labels,
        exposures,
        target_column="forward_excess_return_10d",
        numeric_exposures=("rolling_beta_20d_vs_spy",),
        categorical_exposures=("sector",),
    )

    residual = result["residualized_forward_excess_return_10d"]
    assert residual.notna().all()
    assert abs(residual.mean()) < 1e-10
    assert residual.abs().max() < 1e-10
