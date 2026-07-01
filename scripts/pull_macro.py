"""
Pull macroeconomic data from primary public sources.

Sources:
- BLS Public Data API:
  - CPI-U, all items, seasonally adjusted: CUSR0000SA0
  - Unemployment rate, seasonally adjusted: LNS14000000

- Federal Reserve Board Data Download Program:
  - 10-year Treasury yield: RIFLGFCY10_N.B
  - 2-year Treasury yield: RIFLGFCY02_N.B
  - Federal funds effective rate: RIFSPFF_N.B
  - Industrial production total index: IP.B50001.S

Outputs:
- data/raw/bls_macro.parquet
- data/raw/fed_macro.parquet
- data/raw/macro_sources.parquet
- data/processed/macro.parquet

Run from repo root:

    python -m scripts.pull_macro

Optional:

    python -m scripts.pull_macro --start-year 2000
"""

import argparse
import csv
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

BLS_RAW_OUTPUT_PATH = RAW_DIR / "bls_macro.parquet"
FED_RAW_OUTPUT_PATH = RAW_DIR / "fed_macro.parquet"
COMBINED_RAW_OUTPUT_PATH = RAW_DIR / "macro_sources.parquet"
PROCESSED_OUTPUT_PATH = PROCESSED_DIR / "macro.parquet"


# ---------------------------------------------------------------------
# BLS configuration
# ---------------------------------------------------------------------

BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

BLS_SERIES = {
    "cpi_all_items_sa": "CUSR0000SA0",
    "unemployment_rate": "LNS14000000",
}


# ---------------------------------------------------------------------
# Federal Reserve DDP configuration
# ---------------------------------------------------------------------

FED_DDP_SERIES = {
    "treasury_10y": {
        "series_id": "RIFLGFCY10_N.B",
        "url": (
            "https://www.federalreserve.gov/datadownload/Output.aspx"
            "?rel=H15"
            "&series=bf17364827e38702b42a58cf8eaa3f78"
            "&lastObs="
            "&from="
            "&to="
            "&filetype=csv"
            "&label=include"
            "&layout=seriescolumn"
        ),
        "source": "federal_reserve_h15",
    },
    "treasury_2y": {
        "series_id": "RIFLGFCY02_N.B",
        "url": (
            "https://www.federalreserve.gov/datadownload/Output.aspx"
            "?rel=H15"
            "&series=bf17364827e38702b42a58cf8eaa3f78"
            "&lastObs="
            "&from="
            "&to="
            "&filetype=csv"
            "&label=include"
            "&layout=seriescolumn"
        ),
        "source": "federal_reserve_h15",
    },
    "fed_funds": {
        "series_id": "RIFSPFF_N.B",
        "url": (
            "https://www.federalreserve.gov/datadownload/Output.aspx"
            "?rel=H15"
            "&series=646250c87b1afd04cc6774796fc0cec8"
            "&lastObs="
            "&from="
            "&to="
            "&filetype=csv"
            "&label=include"
            "&layout=seriescolumn"
        ),
        "source": "federal_reserve_h15",
    },
    "industrial_production": {
        "series_id": "IP.B50001.S",
        "url": (
            "https://www.federalreserve.gov/datadownload/Output.aspx"
            "?rel=G17"
            "&series=403cc340ea3a19566b30770b9e4793ab"
            "&lastObs="
            "&from="
            "&to="
            "&filetype=csv"
            "&label=include"
            "&layout=seriescolumn"
            "&type=package"
        ),
        "source": "federal_reserve_g17",
    },
}


EXPECTED_COLUMNS = [
    "treasury_10y",
    "treasury_2y",
    "fed_funds",
    "cpi_all_items_sa",
    "unemployment_rate",
    "industrial_production",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def safe_numeric(value) -> float | None:
    """
    Convert a value to float.

    Returns None for missing values such as:
    - "-"
    - "ND"
    - "."
    - empty strings
    """
    if value is None:
        return None

    value_str = str(value).strip()

    if value_str in {"", "-", "ND", "N/A", "NA", "."}:
        return None

    parsed = pd.to_numeric(value_str, errors="coerce")

    if pd.isna(parsed):
        return None

    return float(parsed)


def ensure_output_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# BLS functions
# ---------------------------------------------------------------------

def get_bls_api_key() -> str:
    """Load the BLS API key from .env."""
    load_dotenv()

    api_key = os.getenv("BLS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing BLS_API_KEY. Add it to your .env file like this:\n"
            "BLS_API_KEY=your_actual_key_here"
        )

    return api_key


def fetch_bls_chunk(
    series_map: dict[str, str],
    start_year: int,
    end_year: int,
    api_key: str,
) -> pd.DataFrame:
    """
    Fetch one BLS API chunk.

    Registered BLS API users can request up to 20 years per query,
    so longer histories should be split into chunks.
    """
    payload = {
        "seriesid": list(series_map.values()),
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": api_key,
    }

    response = requests.post(BLS_URL, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API request failed:\n{data}")

    id_to_column = {series_id: column for column, series_id in series_map.items()}
    rows = []
    retrieved_at = pd.Timestamp.utcnow()

    for series in data["Results"]["series"]:
        series_id = series["seriesID"]
        column = id_to_column.get(series_id)

        if column is None:
            continue

        for obs in series["data"]:
            period = obs.get("period")

            # Skip annual average rows like M13 if returned.
            if period == "M13" or not str(period).startswith("M"):
                continue

            value = safe_numeric(obs.get("value"))

            if value is None:
                continue

            month = int(period[1:])
            year = int(obs["year"])
            date = pd.Timestamp(year=year, month=month, day=1)

            rows.append(
                {
                    "date": date,
                    "series_id": series_id,
                    "column": column,
                    "value": value,
                    "source": "bls_api",
                    "retrieved_at": retrieved_at,
                }
            )

    return pd.DataFrame(rows)


def fetch_bls_history(
    series_map: dict[str, str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch BLS data over a potentially long historical window."""
    api_key = get_bls_api_key()
    frames = []

    for chunk_start in range(start_year, end_year + 1, 20):
        chunk_end = min(chunk_start + 19, end_year)

        print(f"Pulling BLS data for {chunk_start}-{chunk_end}...")

        chunk = fetch_bls_chunk(
            series_map=series_map,
            start_year=chunk_start,
            end_year=chunk_end,
            api_key=api_key,
        )

        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        raise RuntimeError("No BLS data were returned.")

    raw = pd.concat(frames, ignore_index=True)

    raw = raw.dropna(subset=["value"])
    raw = raw.drop_duplicates(subset=["date", "series_id"], keep="last")
    raw = raw.sort_values(["series_id", "date"]).reset_index(drop=True)

    return raw


# ---------------------------------------------------------------------
# Federal Reserve DDP functions
# ---------------------------------------------------------------------

def find_ddp_header(reader: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """
    Find the DDP observation header row.

    Fed DDP CSVs usually include several metadata rows, then a row containing
    "Time Period". Observation rows follow that row.
    """
    for i, row in enumerate(reader):
        cleaned = [str(cell).strip() for cell in row]

        if "Time Period" in cleaned:
            return cleaned, reader[i + 1 :]

    raise RuntimeError("Could not find a 'Time Period' row in Fed DDP response.")


def find_series_column_index(header: list[str], series_id: str) -> int:
    """
    Find the column for a requested Fed series.

    Depending on the DDP layout, the column may be named:
    - RIFLGFCY10_N.B
    - H15/H15/RIFLGFCY10_N.B
    - something containing the series ID
    """
    for i, name in enumerate(header):
        cleaned_name = str(name).strip()

        if cleaned_name == series_id:
            return i

        if cleaned_name.endswith(series_id):
            return i

        if series_id in cleaned_name:
            return i

    raise RuntimeError(
        f"Could not find series {series_id} in Fed DDP response.\n"
        f"Available columns were:\n{header}"
    )


def fetch_fed_ddp_series(
    column: str,
    series_id: str,
    url: str,
    source: str,
    start_year: int,
) -> pd.DataFrame:
    """
    Fetch one Federal Reserve DDP series.

    The Fed DDP CSV contains metadata rows before the actual observations.
    This parser finds the row containing "Time Period", then reads the
    requested series column from the observation rows below it.
    """
    print(f"Pulling Fed series: {column}...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    text = response.text

    if "temporarily unavailable" in text.lower():
        raise RuntimeError(
            "The Federal Reserve Data Download Program is temporarily unavailable. "
            "Try running the script again later."
        )

    if "<html" in text.lower() and "time period" not in text.lower():
        raise RuntimeError(
            "The Federal Reserve returned an HTML page instead of a CSV data file. "
            f"Check this URL:\n{url}"
        )

    reader = list(csv.reader(StringIO(text)))

    try:
        header, data_rows = find_ddp_header(reader)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\nCould not parse Fed DDP response for {column}.\n"
            f"URL checked:\n{url}"
        ) from exc

    date_idx = header.index("Time Period")
    value_idx = find_series_column_index(header, series_id)

    rows = []
    retrieved_at = pd.Timestamp.utcnow()

    for row in data_rows:
        if len(row) <= max(date_idx, value_idx):
            continue

        raw_date = str(row[date_idx]).strip()
        raw_value = str(row[value_idx]).strip()

        date = pd.to_datetime(raw_date, errors="coerce")

        if pd.isna(date):
            continue

        if date.year < start_year:
            continue

        value = safe_numeric(raw_value)

        if value is None:
            continue

        rows.append(
            {
                "date": date,
                "series_id": series_id,
                "column": column,
                "value": value,
                "source": source,
                "retrieved_at": retrieved_at,
            }
        )

    if not rows:
        raise RuntimeError(
            f"No usable observations returned for Fed series {column}. "
            f"Check URL:\n{url}"
        )

    return pd.DataFrame(rows)


def fetch_fed_history(start_year: int) -> pd.DataFrame:
    """Fetch all configured Federal Reserve series."""
    frames = []

    for column, info in FED_DDP_SERIES.items():
        frame = fetch_fed_ddp_series(
            column=column,
            series_id=info["series_id"],
            url=info["url"],
            source=info["source"],
            start_year=start_year,
        )
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["date", "series_id"], keep="last")
    raw = raw.sort_values(["series_id", "date"]).reset_index(drop=True)

    return raw


# ---------------------------------------------------------------------
# Cleaning and saving
# ---------------------------------------------------------------------

def clean_macro_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format macro data into wide-format macro features.

    Important:
    This keeps the original observation-date index. It does not forward-fill
    monthly CPI, unemployment, or industrial production across daily rows.

    For model training, release-date alignment should happen later to avoid
    look-ahead bias.
    """
    clean = raw.pivot_table(
        index="date",
        columns="column",
        values="value",
        aggfunc="last",
    )

    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()

    # Remove column index name created by pivot_table.
    clean.columns.name = None

    for column in EXPECTED_COLUMNS:
        if column not in clean.columns:
            clean[column] = pd.NA

    clean = clean[EXPECTED_COLUMNS]

    return clean


def save_outputs(
    bls_raw: pd.DataFrame,
    fed_raw: pd.DataFrame,
    combined_raw: pd.DataFrame,
    clean: pd.DataFrame,
) -> None:
    """Save raw and cleaned macro data."""
    ensure_output_dirs()

    bls_raw.to_parquet(BLS_RAW_OUTPUT_PATH, index=False)
    fed_raw.to_parquet(FED_RAW_OUTPUT_PATH, index=False)
    combined_raw.to_parquet(COMBINED_RAW_OUTPUT_PATH, index=False)
    clean.to_parquet(PROCESSED_OUTPUT_PATH)

    print(f"\nSaved raw BLS data to {BLS_RAW_OUTPUT_PATH}")
    print(f"Saved raw Fed data to {FED_RAW_OUTPUT_PATH}")
    print(f"Saved combined raw macro data to {COMBINED_RAW_OUTPUT_PATH}")
    print(f"Saved cleaned macro data to {PROCESSED_OUTPUT_PATH}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    current_year = pd.Timestamp.today().year

    parser = argparse.ArgumentParser(
        description="Pull macroeconomic data from BLS and Federal Reserve sources."
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="First year of macro data to pull.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=current_year,
        help="Final year of BLS data to pull.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Starting macro pull for {args.start_year}-{args.end_year}...\n")

    bls_raw = fetch_bls_history(
        series_map=BLS_SERIES,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    fed_raw = fetch_fed_history(start_year=args.start_year)

    combined_raw = pd.concat([bls_raw, fed_raw], ignore_index=True)
    combined_raw = combined_raw.drop_duplicates(
        subset=["date", "series_id"],
        keep="last",
    )
    combined_raw = combined_raw.sort_values(["series_id", "date"]).reset_index(drop=True)

    clean = clean_macro_raw(combined_raw)

    save_outputs(
        bls_raw=bls_raw,
        fed_raw=fed_raw,
        combined_raw=combined_raw,
        clean=clean,
    )

    print("\nCleaned macro data tail:")
    print(clean.tail(10))

    print("\nColumns:")
    print(list(clean.columns))

    print("\nMissing values by column:")
    print(clean.isna().sum())


if __name__ == "__main__":
    main()