"""SEC fundamental feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_MARKET_COLUMNS = {"symbol", "date"}
REQUIRED_FUNDAMENTAL_COLUMNS = {
    "symbol",
    "cik",
    "cik_padded",
    "fiscal_year",
    "fiscal_period",
    "filing_date",
    "form",
    "concept",
    "value",
    "end_date",
    "accession_number",
}
FUNDAMENTAL_CONCEPTS = [
    "revenue",
    "net_income",
    "assets",
    "liabilities",
    "stockholders_equity",
    "operating_income",
    "operating_cash_flow",
    "capital_expenditure",
    "cash",
    "current_assets",
    "current_liabilities",
    "debt",
    "gross_profit",
    "interest_expense",
    "research_and_development",
    "stock_based_compensation",
    "shares_outstanding",
]


def build_filing_fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Convert long SEC companyfacts rows into filing-level fundamental features."""
    missing_columns = REQUIRED_FUNDAMENTAL_COLUMNS - set(fundamentals.columns)
    if missing_columns:
        raise ValueError(
            "Fundamentals table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    facts = fundamentals.copy()
    if "earliest_tradable_date" in facts:
        facts["filing_date"] = facts["earliest_tradable_date"]
    facts["symbol"] = facts["symbol"].astype(str).str.upper()
    facts["filing_date"] = pd.to_datetime(
        facts["filing_date"], errors="coerce"
    ).dt.normalize()
    facts["end_date"] = pd.to_datetime(
        facts["end_date"], errors="coerce"
    ).dt.normalize()
    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")
    facts = facts.dropna(subset=["symbol", "filing_date", "concept", "value"])
    facts = facts[facts["concept"].isin(FUNDAMENTAL_CONCEPTS)]

    if facts.empty:
        return _empty_filing_features()

    index_columns = [
        "symbol",
        "cik",
        "cik_padded",
        "fiscal_year",
        "fiscal_period",
        "filing_date",
        "form",
        "end_date",
        "accession_number",
    ]

    facts = facts.sort_values(index_columns + ["concept"])
    facts = facts.drop_duplicates(
        subset=index_columns + ["concept"],
        keep="last",
    )

    filing_features = facts.pivot_table(
        index=index_columns,
        columns="concept",
        values="value",
        aggfunc="last",
    ).reset_index()
    filing_features.columns.name = None

    for concept in FUNDAMENTAL_CONCEPTS:
        if concept not in filing_features.columns:
            filing_features[concept] = pd.NA

    filing_features = filing_features.rename(
        columns={
            "revenue": "fundamental_revenue",
            "net_income": "fundamental_net_income",
            "assets": "fundamental_assets",
            "liabilities": "fundamental_liabilities",
            "stockholders_equity": "fundamental_stockholders_equity",
            **{
                concept: f"fundamental_{concept}"
                for concept in FUNDAMENTAL_CONCEPTS
                if concept
                not in {
                    "revenue",
                    "net_income",
                    "assets",
                    "liabilities",
                    "stockholders_equity",
                }
            },
        }
    )

    filing_features = filing_features.sort_values(
        ["symbol", "filing_date", "end_date", "accession_number"]
    ).reset_index(drop=True)

    comparison_groups = ["symbol", "fiscal_period"]
    for concept in [
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "assets",
        "debt",
        "shares_outstanding",
    ]:
        column = f"fundamental_{concept}"
        filing_features[f"{concept}_growth"] = filing_features.groupby(
            comparison_groups, dropna=False, group_keys=False
        )[column].apply(_pct_change_non_null)
    filing_features["net_income_margin"] = (
        filing_features["fundamental_net_income"]
        / filing_features["fundamental_revenue"]
    )
    filing_features["liability_ratio"] = (
        filing_features["fundamental_liabilities"]
        / filing_features["fundamental_assets"]
    )
    filing_features["equity_ratio"] = (
        filing_features["fundamental_stockholders_equity"]
        / filing_features["fundamental_assets"]
    )
    filing_features["gross_margin"] = (
        filing_features["fundamental_gross_profit"]
        / filing_features["fundamental_revenue"]
    )
    filing_features["operating_margin"] = (
        filing_features["fundamental_operating_income"]
        / filing_features["fundamental_revenue"]
    )
    filing_features["free_cash_flow_margin"] = (
        filing_features["fundamental_operating_cash_flow"]
        - filing_features["fundamental_capital_expenditure"]
    ) / filing_features["fundamental_revenue"]
    filing_features["cash_flow_quality"] = (
        filing_features["fundamental_operating_cash_flow"]
        / filing_features["fundamental_net_income"]
    )
    filing_features["accrual_ratio"] = (
        filing_features["fundamental_net_income"]
        - filing_features["fundamental_operating_cash_flow"]
    ) / filing_features["fundamental_assets"]
    filing_features["current_ratio"] = (
        filing_features["fundamental_current_assets"]
        / filing_features["fundamental_current_liabilities"]
    )
    filing_features["debt_ratio"] = (
        filing_features["fundamental_debt"] / filing_features["fundamental_assets"]
    )
    filing_features["research_and_development_intensity"] = (
        filing_features["fundamental_research_and_development"]
        / filing_features["fundamental_revenue"]
    )
    filing_features["stock_based_compensation_intensity"] = (
        filing_features["fundamental_stock_based_compensation"]
        / filing_features["fundamental_revenue"]
    )

    return filing_features[_filing_feature_columns()]


def merge_fundamental_features(
    market_features: pd.DataFrame,
    filing_features: pd.DataFrame,
) -> pd.DataFrame:
    """As-of merge filing-level fundamentals into market features by symbol/date."""
    missing_columns = REQUIRED_MARKET_COLUMNS - set(market_features.columns)
    if missing_columns:
        raise ValueError(
            "Market feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if "filing_date" not in filing_features.columns:
        raise ValueError("Filing feature table is missing required column: filing_date")

    market = market_features.copy()
    market["symbol"] = market["symbol"].astype(str).str.upper()
    market["date"] = (
        pd.to_datetime(market["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    filings = filing_features.copy()
    filings["symbol"] = filings["symbol"].astype(str).str.upper()
    filings["filing_date"] = (
        pd.to_datetime(
            filings["filing_date"],
            errors="coerce",
        )
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    if market["date"].isna().any():
        raise ValueError("Market feature table contains invalid dates.")

    if filings["filing_date"].isna().any():
        raise ValueError("Filing feature table contains invalid filing dates.")

    merged_frames: list[pd.DataFrame] = []
    for symbol, symbol_market in market.groupby("symbol", sort=False):
        symbol_filings = filings[filings["symbol"] == symbol].sort_values("filing_date")
        symbol_market = symbol_market.sort_values("date")

        if symbol_filings.empty:
            merged = symbol_market.copy()
        else:
            merged = pd.merge_asof(
                symbol_market,
                symbol_filings,
                left_on="date",
                right_on="filing_date",
                by="symbol",
                direction="backward",
            )

        merged_frames.append(merged)

    result = pd.concat(merged_frames, ignore_index=True)
    for column in _filing_feature_columns():
        if column != "symbol" and column not in result.columns:
            result[column] = pd.NA

    if "filing_date" not in result.columns:
        result["filing_date"] = pd.NaT

    result["filing_date"] = pd.to_datetime(result["filing_date"], errors="coerce")
    result["filing_recency_days"] = (result["date"] - result["filing_date"]).dt.days
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_fundamental_feature_table(
    feature_path: str | Path,
    fundamentals_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Load market/fundamental inputs, merge SEC fundamental features, and save."""
    feature_path = Path(feature_path)
    fundamentals_path = Path(fundamentals_path)
    output_path = Path(output_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Market feature file not found: {feature_path}. "
            "Run python -m scripts.build_macro_features first."
        )

    if not fundamentals_path.exists():
        raise FileNotFoundError(
            f"Fundamentals file not found: {fundamentals_path}. "
            "Run python -m scripts.ingest_sec_company_facts first."
        )

    market_features = pd.read_parquet(feature_path)
    fundamentals = pd.read_parquet(fundamentals_path)
    filing_features = build_filing_fundamental_features(fundamentals)
    result = merge_fundamental_features(
        market_features=market_features,
        filing_features=filing_features,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result


def _filing_feature_columns() -> list[str]:
    return [
        "symbol",
        "cik",
        "cik_padded",
        "fiscal_year",
        "fiscal_period",
        "filing_date",
        "form",
        "end_date",
        "accession_number",
        "fundamental_revenue",
        "fundamental_net_income",
        "fundamental_assets",
        "fundamental_liabilities",
        "fundamental_stockholders_equity",
        "fundamental_operating_income",
        "fundamental_operating_cash_flow",
        "fundamental_capital_expenditure",
        "fundamental_cash",
        "fundamental_current_assets",
        "fundamental_current_liabilities",
        "fundamental_debt",
        "fundamental_gross_profit",
        "fundamental_interest_expense",
        "fundamental_research_and_development",
        "fundamental_stock_based_compensation",
        "fundamental_shares_outstanding",
        "revenue_growth",
        "net_income_growth",
        "operating_income_growth",
        "operating_cash_flow_growth",
        "assets_growth",
        "debt_growth",
        "shares_outstanding_growth",
        "net_income_margin",
        "liability_ratio",
        "equity_ratio",
        "gross_margin",
        "operating_margin",
        "free_cash_flow_margin",
        "cash_flow_quality",
        "accrual_ratio",
        "current_ratio",
        "debt_ratio",
        "research_and_development_intensity",
        "stock_based_compensation_intensity",
    ]


def _empty_filing_features() -> pd.DataFrame:
    return pd.DataFrame(columns=_filing_feature_columns())


def _pct_change_non_null(values: pd.Series) -> pd.Series:
    result = pd.Series(index=values.index, dtype="float64")
    valid = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    result.loc[valid.index] = valid.pct_change(fill_method=None)
    return result
