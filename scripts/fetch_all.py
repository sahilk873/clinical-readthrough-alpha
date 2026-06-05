"""Script to fetch all raw data sources."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_alpha.config import Settings
from clinical_alpha.fda.fetcher import fetch_all_fda_data
from clinical_alpha.sec.edgar import fetch_sec_healthcare_universe
from clinical_alpha.universe.builder import save_universe

settings = Settings()


def fetch_all():
    """Fetch all data sources and save raw data."""
    print("Fetching healthcare universe from SEC EDGAR...")
    universe = fetch_sec_healthcare_universe()
    save_universe(universe, str(settings.RAW_DIR / "healthcare_universe.parquet"))
    print(f"  {len(universe)} companies")

    print("Fetching FDA data...")
    fda = fetch_all_fda_data()
    for name, df in fda.items():
        if df is not None:
            df.to_parquet(settings.RAW_DIR / f"fda_{name}.parquet", index=False)
            print(f"  FDA {name}: {len(df)} records")
        else:
            print(f"  FDA {name}: (unavailable)")

    print(f"\nRaw data saved to {settings.RAW_DIR}")


if __name__ == "__main__":
    fetch_all()
