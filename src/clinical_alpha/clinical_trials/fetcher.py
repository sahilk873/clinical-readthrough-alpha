"""ClinicalTrials.gov API v2 fetcher.

Fetches trial records by sponsor/company, drug/intervention, and condition.
Treats ClinicalTrials.gov primarily as a graph-construction source.
"""

import time
from typing import Any, Optional

import httpx
import pandas as pd

from clinical_alpha.config import Settings

settings = Settings()

CT_BASE = "https://clinicaltrials.gov/api/v2"
CT_STUDIES_URL = f"{CT_BASE}/studies"
CT_FIELD_QUERY = (
    "NCTId,briefTitle,officialTitle,status,phase,sponsorName,"
    "leadSponsorName,interventionName,interventionType,conditionName,"
    "primaryCompletionDate,completionDate,startDate,resultFirstPostDate,"
    "overallStatus,whyStopped,lastUpdatePostDate"
)


def _flatten_study(study: dict) -> dict:
    """Flatten nested clinical trial study JSON into a single record."""
    protocol = study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    arms_module = protocol.get("armsInterventionsModule", {})
    design_module = protocol.get("designModule", {})

    nct_id = id_module.get("nctId", "")
    brief_title = id_module.get("briefTitle", "")
    official_title = id_module.get("officialTitle", "")

    overall_status = status_module.get("overallStatus", "")
    start_date = (
        status_module.get("startDateStruct", {}).get("date", "")
        if isinstance(status_module.get("startDateStruct"), dict)
        else ""
    )
    primary_completion = (
        status_module.get("primaryCompletionDateStruct", {}).get("date", "")
        if isinstance(status_module.get("primaryCompletionDateStruct"), dict)
        else ""
    )
    completion_date = (
        status_module.get("completionDateStruct", {}).get("date", "")
        if isinstance(status_module.get("completionDateStruct"), dict)
        else ""
    )
    result_post = (
        status_module.get("resultsFirstPostDateStruct", {}).get("date", "")
        if isinstance(status_module.get("resultsFirstPostDateStruct"), dict)
        else ""
    )
    last_update = (
        status_module.get("lastUpdatePostDateStruct", {}).get("date", "")
        if isinstance(status_module.get("lastUpdatePostDateStruct"), dict)
        else ""
    )

    lead_sponsor = (
        sponsor_module.get("leadSponsor", {}).get("name", "")
        if isinstance(sponsor_module.get("leadSponsor"), dict)
        else ""
    )
    collaborators = sponsor_module.get("collaborators", [])
    sponsor_names = [lead_sponsor]
    for c in collaborators:
        if isinstance(c, dict):
            sponsor_names.append(c.get("name", ""))
    sponsors = "; ".join(filter(None, sponsor_names))

    conditions = conditions_module.get("conditions", [])
    condition_str = "; ".join(filter(None, conditions)) if conditions else ""

    phase = design_module.get("phases", [])
    phase_str = "; ".join(filter(None, phase)) if phase else ""

    interventions = arms_module.get("interventions", [])
    intervention_names = []
    intervention_types = []
    for i in interventions:
        if isinstance(i, dict):
            intervention_names.append(i.get("name", ""))
            intervention_types.append(i.get("type", ""))
    intervention_name_str = "; ".join(filter(None, intervention_names))
    intervention_type_str = "; ".join(filter(None, intervention_types))

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "overall_status": overall_status,
        "phase": phase_str,
        "sponsors": sponsors,
        "lead_sponsor": lead_sponsor,
        "intervention_name": intervention_name_str,
        "intervention_type": intervention_type_str,
        "conditions": condition_str,
        "start_date": start_date,
        "primary_completion_date": primary_completion,
        "completion_date": completion_date,
        "result_first_post_date": result_post,
        "last_update_post_date": last_update,
    }


def fetch_studies(
    query: Optional[str] = None,
    sponsor: Optional[str] = None,
    intervention: Optional[str] = None,
    condition: Optional[str] = None,
    max_results: int = 500,
    page_size: int = 100,
) -> list[dict]:
    """Fetch studies from ClinicalTrials.gov API v2.

    Supports filtering by sponsor, intervention, and/or condition.
    """
    params: dict[str, Any] = {
        "format": "json",
        "pageSize": min(page_size, max_results),
        "fields": CT_FIELD_QUERY,
    }

    query_parts = []
    if query:
        query_parts.append(f"AREA[SearchTerm]{query}")
    if sponsor:
        query_parts.append(f"AREA[LeadSponsorName]{sponsor}")
    if intervention:
        query_parts.append(f"AREA[InterventionName]{intervention}")
    if condition:
        query_parts.append(f"AREA[ConditionName]{condition}")

    if query_parts:
        params["query.term"] = " AND ".join(query_parts)
    else:
        params["query.term"] = (
            "AREA[OverallStatus]RECRUITING OR AREA[OverallStatus]ACTIVE_NOT_RECRUITING"
        )

    all_studies: list[dict] = []
    next_page_token: Optional[str] = None

    with httpx.Client(timeout=60) as client:
        while len(all_studies) < max_results:
            if next_page_token:
                params["pageToken"] = next_page_token
            try:
                resp = client.get(CT_STUDIES_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                break

            studies = data.get("studies", [])
            for s in studies:
                all_studies.append(_flatten_study(s))

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
            time.sleep(0.3)

    return all_studies[:max_results]


def fetch_studies_for_companies(
    company_names: list[str],
    max_per_company: int = 200,
) -> pd.DataFrame:
    """Fetch clinical trials sponsored by a list of companies."""
    all_records = []
    for name in company_names:
        if not name:
            continue
        try:
            records = fetch_studies(sponsor=name, max_results=max_per_company)
            all_records.extend(records)
        except Exception:
            continue
        time.sleep(0.35)
    return pd.DataFrame(all_records)


def fetch_studies_by_drug(drug_names: list[str], max_per_drug: int = 200) -> pd.DataFrame:
    """Fetch clinical trials by drug/intervention name."""
    all_records = []
    for drug in drug_names:
        if not drug:
            continue
        try:
            records = fetch_studies(intervention=drug, max_results=max_per_drug)
            all_records.extend(records)
        except Exception:
            continue
        time.sleep(0.35)
    return pd.DataFrame(all_records)


def fetch_studies_by_condition(conditions: list[str], max_per_condition: int = 200) -> pd.DataFrame:
    """Fetch clinical trials by medical condition."""
    all_records = []
    for cond in conditions:
        if not cond:
            continue
        try:
            records = fetch_studies(condition=cond, max_results=max_per_condition)
            all_records.extend(records)
        except Exception:
            continue
        time.sleep(0.35)
    return pd.DataFrame(all_records)
