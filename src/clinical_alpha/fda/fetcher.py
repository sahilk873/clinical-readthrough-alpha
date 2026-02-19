"""FDA Drugs@FDA data fetcher.

Downloads and parses FDA approval events and drug metadata
from the publicly available Drugs@FDA database.
"""

import io
import zipfile
from typing import Optional

import httpx
import pandas as pd

from clinical_alpha.config import Settings

settings = Settings()

FDA_BASE = "https://www.accessdata.fda.gov"
FDA_DRUG_LIST = f"{FDA_BASE}/cder/drugsatfda_datafiles/druglist.zip"
FDA_PRODUCTS = f"{FDA_BASE}/cder/drugsatfda_datafiles/product.zip"
FDA_APPLICATIONS = f"{FDA_BASE}/cder/drugsatfda_datafiles/apps.zip"
FDA_APP_DOCS = f"{FDA_BASE}/cder/drugsatfda_datafiles/appdocepa.zip"
FDA_TEALTH = f"{FDA_BASE}/cder/drugsatfda_datafiles/tealth.zip"


def _download_zip(url: str) -> Optional[zipfile.ZipFile]:
    """Download a ZIP file from FDA and return as ZipFile object."""
    with httpx.Client(timeout=120) as client:
        try:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(resp.content))
        except Exception:
            return None


def _read_csv_from_zip(z: zipfile.ZipFile, filename: str) -> Optional[pd.DataFrame]:
    """Read a CSV file from a ZIP archive."""
    try:
        with z.open(filename) as f:
            return pd.read_csv(f, encoding="latin1", low_memory=False)
    except Exception:
        for name in z.namelist():
            if filename.lower() in name.lower():
                with z.open(name) as f:
                    return pd.read_csv(f, encoding="latin1", low_memory=False)
    return None


def fetch_drug_list() -> Optional[pd.DataFrame]:
    """Fetch the FDA drug list (drug names, application numbers)."""
    z = _download_zip(FDA_DRUG_LIST)
    if z is None:
        return None
    return _read_csv_from_zip(z, "druglist.txt")


def fetch_products() -> Optional[pd.DataFrame]:
    """Fetch FDA product data (dosage forms, routes, ingredients)."""
    z = _download_zip(FDA_PRODUCTS)
    if z is None:
        return None
    return _read_csv_from_zip(z, "product.txt")


def fetch_applications() -> Optional[pd.DataFrame]:
    """Fetch FDA application data (sponsor, approval dates)."""
    z = _download_zip(FDA_APPLICATIONS)
    if z is None:
        return None
    return _read_csv_from_zip(z, "apps.txt")


def fetch_application_docs() -> Optional[pd.DataFrame]:
    """Fetch FDA application documents (review docs, approval letters)."""
    z = _download_zip(FDA_APP_DOCS)
    if z is None:
        return None
    return _read_csv_from_zip(z, "appdocepa.txt")


def parse_approval_events(
    applications: Optional[pd.DataFrame] = None,
    products: Optional[pd.DataFrame] = None,
    drug_list: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Parse FDA approval events from raw data.

    Returns DataFrame with columns:
        appl_no, drug_name, sponsor, approval_date, type, route, ingredient
    """
    if applications is None:
        applications = fetch_applications()
    if products is None:
        products = fetch_products()
    if drug_list is None:
        drug_list = fetch_drug_list()

    # Merge applications with products on ApplNo
    events = None
    if applications is not None and products is not None:
        applications["ApplNo"] = applications["ApplNo"].astype(str)
        products["ApplNo"] = products["ApplNo"].astype(str)
        events = applications.merge(products, on="ApplNo", how="left", suffixes=("_app", "_prod"))

        # Filter to approval actions
        if "ActionType" in events.columns:
            events = events[
                events["ActionType"].str.contains("APPR|APPROVAL", na=False, case=False)
            ].copy()

        events["approval_date"] = pd.to_datetime(
            events.get("ActionDate", events.get("DocDate", pd.NaT)),
            errors="coerce",
        )

        events.rename(
            columns={
                "ApplNo": "appl_no",
                "DrugName": "drug_name",
                "SponsorName": "sponsor",
                "ActiveIngredient": "ingredient",
                "Route": "route",
                "ProductNo": "product_no",
            },
            inplace=True,
        )

        # Merge drug list for additional names
        if drug_list is not None:
            drug_list["ApplNo"] = drug_list["ApplNo"].astype(str)
            events = events.merge(
                drug_list[["ApplNo", "DrugName"]],
                on="ApplNo",
                how="left",
                suffixes=("", "_list"),
            )
            if "drug_name_list" in events.columns:
                events["drug_name"] = events["drug_name"].fillna(events["drug_name_list"])

    if events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "appl_no",
                "drug_name",
                "sponsor",
                "approval_date",
                "type",
                "route",
                "ingredient",
            ]
        )

    result_cols = [
        "appl_no",
        "drug_name",
        "sponsor",
        "approval_date",
        "type",
        "route",
        "ingredient",
    ]
    available = [c for c in result_cols if c in events.columns]
    return events[available].copy()


def fetch_fda_approval_events(max_records: int = 1000) -> pd.DataFrame:
    """High-level function to fetch and parse FDA approval events."""
    apps = fetch_applications()
    prods = fetch_products()
    drugs = fetch_drug_list()
    events = parse_approval_events(apps, prods, drugs)
    if len(events) > max_records:
        events = events.head(max_records)
    return events


def fetch_all_fda_data() -> dict[str, Optional[pd.DataFrame]]:
    """Fetch all FDA datasets and return as a dict."""
    return {
        "drug_list": fetch_drug_list(),
        "products": fetch_products(),
        "applications": fetch_applications(),
        "app_docs": fetch_application_docs(),
    }
