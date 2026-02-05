"""SEC EDGAR API client for ticker/CIK/company name mapping."""

from dataclasses import dataclass
from typing import Optional

import httpx
import pandas as pd

from clinical_alpha.config import Settings

settings = Settings()

SEC_BASE = "https://www.sec.gov"
USER_AGENT = "ClinicalAlpha/0.1.0 (research project; contact@clinicalalpha.com)"


@dataclass
class CompanyRecord:
    ticker: str
    cik: str
    name: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None


def _headers() -> dict:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def fetch_ticker_cik_map() -> dict[str, str]:
    """Fetch SEC's ticker-to-CIK mapping from company_tickers.json."""
    url = f"{SEC_BASE}/files/company_tickers.json"
    with httpx.Client() as client:
        resp = client.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
    result: dict[str, str] = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker and cik:
            result[ticker] = cik
    return result


def fetch_company_facts(cik: str) -> Optional[dict]:
    """Fetch company facts from SEC XBRL API."""
    url = f"{SEC_BASE}/cik/000{cik}/facts.json"
    with httpx.Client() as client:
        try:
            resp = client.get(url, headers=_headers(), timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None


def fetch_company_concept(
    cik: str, taxonomy: str = "us-gaap", concept: str = "EntityCommonStockSharesOutstanding"
) -> Optional[dict]:
    """Fetch a specific XBRL concept for a company."""
    url = f"{SEC_BASE}/cik/000{cik}/concept/{taxonomy}/{concept}.json"
    with httpx.Client() as client:
        try:
            resp = client.get(url, headers=_headers(), timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None


def build_company_universe(min_mcap: float = 50_000_000) -> pd.DataFrame:
    """Build a DataFrame of public companies from SEC EDGAR ticker data.

    Returns columns: ticker, cik, name, exchange
    """
    ticker_map = fetch_ticker_cik_map()
    records = []
    for ticker, cik in ticker_map.items():
        records.append(
            {
                "ticker": ticker,
                "cik": cik,
                "name": "",
                "exchange": "",
            }
        )
    df = pd.DataFrame(records)
    return df


def filter_healthcare_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter universe to healthcare companies.

    Uses known healthcare ticker patterns and CIK ranges.
    This is a first-pass filter; refined matching happens in the matching module.
    """
    healthcare_etfs = {
        "XLV",
        "XBI",
        "IBB",
        "IHF",
        "ARKG",
        "VHT",
        "FHLC",
        "PSCH",
        "PBE",
        "BBC",
        "BBH",
        "LABU",
        "LABD",
    }
    healthcare_ticks: set[str] = set()

    # Known major healthcare companies
    major_healthcare = [
        "UNH",
        "JNJ",
        "PFE",
        "ABBV",
        "MRK",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
        "LLY",
        "NVS",
        "SNY",
        "GSK",
        "AZN",
        "NVO",
        "REGN",
        "VRTX",
        "GILD",
        "AMGN",
        "ISRG",
        "BSX",
        "SYK",
        "MDT",
        "EW",
        "ILMN",
        "BIIB",
        "ALXN",
        "CELG",
        "MYL",
        "INCY",
        "ALKS",
        "SRPT",
        "EXAS",
        "NTLA",
        "EDIT",
        "CRSP",
        "BEAM",
        "VERV",
        "MRNA",
        "BNTX",
        "NVAX",
        "DYN",
        "IONS",
        "FOLD",
        "SAGE",
        "AXSM",
        "PTCT",
        "BGNE",
        "BEAM",
        "RCKT",
        "CRBU",
        "NTLA",
    ]

    # Healthcare CIK ranges (approximate)
    healthcare_ciks = {
        "0000200406",
        "0000200407",
        "0000318071",
        "0000318072",
        "0000318073",
        "0000742547",
        "0000742548",
        "0000874237",
        "0000874238",
        "0000880744",
        "0000880745",
        "0000884905",
        "0000884906",
        "0000900729",
        "0000900730",
        "0000933524",
    }

    for tick in major_healthcare:
        healthcare_ticks.add(tick.upper())

    def _is_healthcare(row) -> bool:
        t = str(row.get("ticker", "")).upper()
        if t in healthcare_etfs:
            return False
        if t in healthcare_ticks:
            return True
        cik = str(row.get("cik", ""))
        if cik[:6] in {c[:6] for c in healthcare_ciks}:
            return True
        name = str(row.get("name", "")).lower()
        healthcare_keywords = [
            "pharma",
            "therapeut",
            "biotech",
            "biolog",
            "diagnostic",
            "medical",
            "health",
            "clinical",
            "drug",
            "vaccine",
            "genomic",
            "oncology",
            "cardio",
            "neuro",
            "immuno",
            "hospital",
            "surgical",
            "device",
            "laborator",
            "bio",
            "genetics",
            "cell",
            "gene",
            "antibody",
            "protein",
        ]
        return any(kw in name for kw in healthcare_keywords)

    mask = df.apply(_is_healthcare, axis=1)
    return df[mask].copy()


def fetch_sec_healthcare_universe() -> pd.DataFrame:
    """Fetch and filter healthcare companies from SEC data."""
    df = build_company_universe()
    healthcare = filter_healthcare_universe(df)
    healthcare = healthcare.reset_index(drop=True)
    return healthcare
