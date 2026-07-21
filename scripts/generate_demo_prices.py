"""Generate deterministic synthetic prices for the credential-free public demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path("data/processed/demo_prices.parquet")
DEMO_SYMBOLS = ("ALFA", "BRAV", "CHAR", "DELT", "ECHO", "FOXT", "SPY")


def generate_demo_prices(
    *, days: int = 1_260, seed: int = 42, start: str = "2020-01-02"
) -> pd.DataFrame:
    """Return a reproducible panel with market, sector, and idiosyncratic moves."""
    if days < 400:
        raise ValueError("days must be at least 400 for purged locked-test research.")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=days)
    market = rng.normal(0.00025, 0.009, days)
    sector_shocks = {
        "technology": rng.normal(0.00008, 0.004, days),
        "industrial": rng.normal(0.00004, 0.0035, days),
        "healthcare": rng.normal(0.00006, 0.003, days),
        "benchmark": np.zeros(days),
    }
    definitions = {
        "ALFA": ("technology", 1.10, 0.00010),
        "BRAV": ("technology", 0.95, 0.00006),
        "CHAR": ("industrial", 1.05, 0.00005),
        "DELT": ("industrial", 0.90, 0.00003),
        "ECHO": ("healthcare", 0.80, 0.00007),
        "FOXT": ("healthcare", 1.00, 0.00004),
        "SPY": ("benchmark", 1.00, 0.0),
    }
    rows: list[pd.DataFrame] = []
    for symbol in DEMO_SYMBOLS:
        sector, beta, drift = definitions[symbol]
        noise_scale = 0.001 if symbol == "SPY" else 0.006
        returns = (
            beta * market
            + sector_shocks[sector]
            + drift
            + rng.normal(0.0, noise_scale, days)
        )
        close = 100 * np.exp(np.cumsum(np.log1p(np.clip(returns, -0.95, None))))
        rows.append(
            pd.DataFrame(
                {"symbol": symbol, "sector": sector, "date": dates, "close": close}
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=1_260)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = generate_demo_prices(days=args.days, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(args.output, index=False)
    print(f"Saved {len(prices):,} synthetic rows to {args.output}")
    print("Synthetic data only: no provider or live-market observations are included.")


if __name__ == "__main__":
    main()
