"""Universe builder - constructs the initial company universe from SEC EDGAR."""

import pandas as pd

from clinical_alpha.sec.edgar import fetch_sec_healthcare_universe


def build_initial_universe() -> pd.DataFrame:
    """Build initial healthcare company universe.

    Returns DataFrame with columns:
        ticker, cik, name, exchange
    """
    df = fetch_sec_healthcare_universe()
    return df


def load_universe(path: str | None = None) -> pd.DataFrame:
    """Load universe from parquet or rebuild."""
    if path:
        return pd.read_parquet(path)
    return build_initial_universe()


def save_universe(df: pd.DataFrame, path: str):
    """Save universe to parquet."""
    df.to_parquet(path, index=False)
