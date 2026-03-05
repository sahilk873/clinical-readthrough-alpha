"""Normalization and fuzzy matching for company names, drug names, conditions.

All low-confidence fuzzy matches are exported for manual review.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd

from clinical_alpha.config import Settings

settings = Settings()


def normalize_company_name(name: str) -> str:
    """Normalize a company name for matching."""
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r"[,\-\(\)\.\'\"&]", " ", n)
    n = re.sub(r"\s+", " ", n)
    # Remove common suffixes
    suffixes = [
        r"\binc\b",
        r"\bcorp\b",
        r"\bltd\b",
        r"\bllc\b",
        r"\blp\b",
        r"\bplc\b",
        r"\bgmbh\b",
        r"\bco\b",
        r"\bholdings\b",
        r"\bgroup\b",
        r"\bthe\b",
        r"\bpharmaceuticals\b",
        r"\btherapeutics\b",
        r"\bbiotech\b",
        r"\blaboratories\b",
        r"\bdiagnostics\b",
        r"\btechnologies\b",
    ]
    for suffix in suffixes:
        n = re.sub(suffix, "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def normalize_drug_name(name: str) -> str:
    """Normalize a drug/intervention name for matching."""
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r"[\(\)\[\]\{\}]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def normalize_condition_name(name: str) -> str:
    """Normalize a medical condition name for matching."""
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r"[\(\)\[\]\{\}]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def fuzzy_match_score(a: str, b: str) -> float:
    """Compute a fuzzy match score between two strings."""
    a_norm = normalize_company_name(a)
    b_norm = normalize_company_name(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def fuzzy_match_drug_score(a: str, b: str) -> float:
    """Compute fuzzy match score for drug names."""
    a_norm = normalize_drug_name(a)
    b_norm = normalize_drug_name(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def match_sponsor_to_company(
    sponsor_name: str,
    company_df: pd.DataFrame,
    threshold: float = 0.75,
) -> list[dict]:
    """Fuzzy match a clinical trial sponsor name to known companies.

    Returns list of dicts with match info. Matches below threshold
    are returned for manual review.
    """
    results = []
    sponsor_norm = normalize_company_name(sponsor_name)
    for _, row in company_df.iterrows():
        company_name = row.get("name", "")
        ticker = row.get("ticker", "")
        score = fuzzy_match_score(sponsor_norm, company_name)
        is_high_conf = score >= threshold
        results.append(
            {
                "sponsor_raw": sponsor_name,
                "company_name": company_name,
                "ticker": ticker,
                "score": round(score, 4),
                "high_confidence": is_high_conf,
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def match_drug_name(
    drug_name_raw: str,
    fda_drugs: list[str],
    threshold: float = 0.8,
) -> list[dict]:
    """Fuzzy match a clinical trial intervention name to FDA drug names."""
    results = []
    drug_norm = normalize_drug_name(drug_name_raw)
    for fda_drug in fda_drugs:
        score = fuzzy_match_drug_score(drug_norm, fda_drug)
        is_high_conf = score >= threshold
        results.append(
            {
                "trial_drug_raw": drug_name_raw,
                "fda_drug_name": fda_drug,
                "score": round(score, 4),
                "high_confidence": is_high_conf,
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)  # type: ignore[arg-type, return-value]
    return results


def export_low_confidence_matches(
    matches: list[dict],
    output_path: str | Path,
    threshold: float = 0.75,
):
    """Export low-confidence fuzzy matches to CSV for manual review."""
    low_conf = [m for m in matches if m.get("score", 0) < threshold]
    if not low_conf:
        return
    pd.DataFrame(low_conf).to_csv(output_path, index=False)


def resolve_best_match(
    matches: list[dict],
    min_score: float = 0.7,
) -> Optional[dict]:
    """Resolve the best high-confidence match from a list of matches."""
    high_conf = [
        m for m in matches if m.get("high_confidence", False) and m.get("score", 0) >= min_score
    ]
    if high_conf:
        return high_conf[0]
    return None


def create_matched_sponsor_map(
    sponsor_names: list[str],
    company_df: pd.DataFrame,
    threshold: float = 0.75,
    low_conf_export_path: Optional[str] = None,
) -> dict[str, Optional[dict]]:
    """Create a mapping from raw sponsor names to matched company info.

    Exports low-confidence matches for manual review.
    """
    mapping: dict[str, Optional[dict]] = {}
    all_review: list[dict] = []

    for sponsor in sponsor_names:
        if not sponsor:
            continue
        matches = match_sponsor_to_company(sponsor, company_df, threshold)
        best = resolve_best_match(matches, min_score=threshold)
        mapping[sponsor] = best
        all_review.extend(matches)

    if low_conf_export_path:
        export_low_confidence_matches(all_review, low_conf_export_path, threshold)

    return mapping
