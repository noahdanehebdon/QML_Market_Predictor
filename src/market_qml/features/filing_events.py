"""SEC filing event feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILING_FORMS = ("10-K", "10-Q", "8-K")
RECENT_FILING_WINDOW_DAYS = 30
FORM_RECENT_WINDOWS_DAYS = {
    "10-K": 90,
    "10-Q": 90,
    "8-K": 30,
}
REQUIRED_MARKET_COLUMNS = {"symbol", "date"}
REQUIRED_SUBMISSION_COLUMNS = {
    "symbol",
    "cik",
    "cik_padded",
    "form",
    "filing_date",
    "report_date",
    "accession_number",
    "primary_document",
}


def build_filing_event_features(submissions: pd.DataFrame) -> pd.DataFrame:
    """Normalize SEC submissions into filing-event rows for as-of merging."""
    missing_columns = REQUIRED_SUBMISSION_COLUMNS - set(submissions.columns)
    if missing_columns:
        raise ValueError(
            "SEC submissions table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    events = submissions.copy()
    if "earliest_tradable_date" in events:
        events["filing_date"] = events["earliest_tradable_date"]
    events["symbol"] = events["symbol"].astype(str).str.upper()
    events["form"] = events["form"].astype(str).str.upper()
    events["filing_date"] = pd.to_datetime(
        events["filing_date"],
        errors="coerce",
    ).dt.normalize()
    events["report_date"] = pd.to_datetime(
        events["report_date"],
        errors="coerce",
    ).dt.normalize()

    events = events.dropna(subset=["symbol", "form", "filing_date"])
    events = events[events["form"].isin(FILING_FORMS)]

    if events.empty:
        return _empty_event_features()

    events = events.sort_values(["symbol", "filing_date", "form", "accession_number"])
    events = events.drop_duplicates(
        subset=["symbol", "form", "filing_date", "accession_number"],
        keep="last",
    )

    return events[_event_feature_columns()].reset_index(drop=True)


def merge_filing_event_features(
    market_features: pd.DataFrame,
    filing_events: pd.DataFrame,
    *,
    recent_window_days: int = RECENT_FILING_WINDOW_DAYS,
    form_recent_windows_days: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Merge SEC filing-event features into market rows by symbol/date."""
    missing_columns = REQUIRED_MARKET_COLUMNS - set(market_features.columns)
    if missing_columns:
        raise ValueError(
            "Market feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    event_columns = {"symbol", "form", "filing_date"}
    missing_event_columns = event_columns - set(filing_events.columns)
    if missing_event_columns:
        raise ValueError(
            "Filing event table is missing required columns: "
            + ", ".join(sorted(missing_event_columns))
        )

    form_windows = form_recent_windows_days or FORM_RECENT_WINDOWS_DAYS
    market = market_features.copy()
    market["symbol"] = market["symbol"].astype(str).str.upper()
    market["date"] = (
        pd.to_datetime(market["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    if market["date"].isna().any():
        raise ValueError("Market feature table contains invalid dates.")

    events = filing_events.copy()
    events["symbol"] = events["symbol"].astype(str).str.upper()
    events["form"] = events["form"].astype(str).str.upper()
    events["filing_date"] = (
        pd.to_datetime(
            events["filing_date"],
            errors="coerce",
        )
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    events = events.dropna(subset=["symbol", "form", "filing_date"])
    events = events[events["form"].isin(FILING_FORMS)]

    result = _merge_last_filing(market, events)
    result["sec_days_since_last_filing"] = (
        result["date"] - result["sec_last_filing_date"]
    ).dt.days
    result["sec_recent_filing_30d"] = (
        result["sec_days_since_last_filing"].le(recent_window_days).fillna(False)
    )

    for form in FILING_FORMS:
        suffix = _form_suffix(form)
        result = _merge_last_form_filing(result, events, form=form, suffix=suffix)
        days_column = f"sec_days_since_last_{suffix}"
        result[days_column] = (
            result["date"] - result[f"sec_last_{suffix}_filing_date"]
        ).dt.days
        result[f"sec_recent_{suffix}_{form_windows[form]}d"] = (
            result[days_column].le(form_windows[form]).fillna(False)
        )
        result[f"sec_last_filing_is_{suffix}"] = (
            result["sec_last_filing_form"].eq(form).fillna(False)
        )

    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_filing_event_feature_table(
    feature_path: str | Path,
    submissions_path: str | Path,
    output_path: str | Path,
    *,
    recent_window_days: int = RECENT_FILING_WINDOW_DAYS,
    form_recent_windows_days: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Load market/submission inputs, merge SEC filing events, and save."""
    feature_path = Path(feature_path)
    submissions_path = Path(submissions_path)
    output_path = Path(output_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Market feature file not found: {feature_path}. "
            "Run python -m scripts.build_fundamental_features first."
        )

    if not submissions_path.exists():
        raise FileNotFoundError(
            f"SEC submissions file not found: {submissions_path}. "
            "Run python -m scripts.ingest_sec_submissions first."
        )

    market_features = pd.read_parquet(feature_path)
    submissions = pd.read_parquet(submissions_path)
    filing_events = build_filing_event_features(submissions)
    result = merge_filing_event_features(
        market_features=market_features,
        filing_events=filing_events,
        recent_window_days=recent_window_days,
        form_recent_windows_days=form_recent_windows_days,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result


def _merge_last_filing(market: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    renamed = events.rename(
        columns={
            "cik": "sec_last_filing_cik",
            "cik_padded": "sec_last_filing_cik_padded",
            "form": "sec_last_filing_form",
            "filing_date": "sec_last_filing_date",
            "report_date": "sec_last_filing_report_date",
            "accession_number": "sec_last_filing_accession_number",
            "primary_document": "sec_last_filing_primary_document",
        }
    )
    columns = [
        "symbol",
        "sec_last_filing_cik",
        "sec_last_filing_cik_padded",
        "sec_last_filing_form",
        "sec_last_filing_date",
        "sec_last_filing_report_date",
        "sec_last_filing_accession_number",
        "sec_last_filing_primary_document",
    ]
    return _asof_by_symbol(
        market=market,
        events=renamed[columns],
        event_date_column="sec_last_filing_date",
    )


def _merge_last_form_filing(
    market: pd.DataFrame,
    events: pd.DataFrame,
    *,
    form: str,
    suffix: str,
) -> pd.DataFrame:
    form_events = events[events["form"] == form].rename(
        columns={
            "filing_date": f"sec_last_{suffix}_filing_date",
            "accession_number": f"sec_last_{suffix}_accession_number",
        }
    )
    columns = [
        "symbol",
        f"sec_last_{suffix}_filing_date",
        f"sec_last_{suffix}_accession_number",
    ]
    return _asof_by_symbol(
        market=market,
        events=form_events[columns],
        event_date_column=f"sec_last_{suffix}_filing_date",
    )


def _asof_by_symbol(
    market: pd.DataFrame,
    events: pd.DataFrame,
    *,
    event_date_column: str,
) -> pd.DataFrame:
    merged_frames: list[pd.DataFrame] = []

    for symbol, symbol_market in market.groupby("symbol", sort=False):
        symbol_market = symbol_market.sort_values("date")
        symbol_events = events[events["symbol"] == symbol].sort_values(
            event_date_column
        )

        if symbol_events.empty:
            merged = symbol_market.copy()
        else:
            merged = pd.merge_asof(
                symbol_market,
                symbol_events,
                left_on="date",
                right_on=event_date_column,
                by="symbol",
                direction="backward",
            )

        merged_frames.append(merged)

    result = pd.concat(merged_frames, ignore_index=True)
    for column in events.columns:
        if column != "symbol" and column not in result.columns:
            result[column] = pd.NA

    return result


def _form_suffix(form: str) -> str:
    return form.lower().replace("-", "")


def _event_feature_columns() -> list[str]:
    return [
        "symbol",
        "cik",
        "cik_padded",
        "form",
        "filing_date",
        "report_date",
        "accession_number",
        "primary_document",
    ]


def _empty_event_features() -> pd.DataFrame:
    return pd.DataFrame(columns=_event_feature_columns())
