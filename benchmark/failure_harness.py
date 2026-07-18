from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sources import DataSourceError, fetch_openfda_label
from event_analysis import build_event_search_plan
from models import CmsUsage, DrugMember, LabelEvidence, Settings
from spl_parser import DailyMedError, enrich_label_with_dailymed, parse_spl_xml
from structured_evidence import extract_structured_event_evidence


MOCK_SET_ID = "00000000-0000-0000-0000-000000000000"


def _usage() -> CmsUsage:
    return CmsUsage(
        member=DrugMember(rxcui="20352", name="carvedilol", tty="IN"),
        cms_generic_names=["carvedilol"],
        total_claims=0,
        data_year=2024,
        match_quality="benchmark-mock",
    )


def _label(*, sections: dict[str, str] | None = None) -> LabelEvidence:
    return LabelEvidence(
        rxcui="20352",
        requested_name="carvedilol",
        query_field="openfda.rxcui",
        matched_label_count=1,
        selected_spl_set_id=MOCK_SET_ID,
        selection_reason="benchmark mock",
        label_match_score=100,
        label_match_confidence="high",
        sections=sections or {"warnings": "Fallback warning text."},
    )


def safe_dailymed_enrichment(label: LabelEvidence) -> tuple[LabelEvidence, str]:
    try:
        enriched, _ = enrich_label_with_dailymed(label, timeout_seconds=0.1)
        return enriched, "dailymed_spl_xml"
    except DailyMedError as exc:
        warning = f"DailyMed unavailable; openFDA fallback retained. {exc}"
        return (
            label.model_copy(
                update={
                    "extraction_source": "openfda_fallback",
                    "dailymed_warning": warning,
                }
            ),
            "openfda_fallback",
        )


def safe_openfda_fetch(
    usage: CmsUsage, settings: Settings
) -> tuple[LabelEvidence | None, str]:
    try:
        label, _ = fetch_openfda_label(usage, settings)
        return label, "available" if label else "insufficient_label_data"
    except DataSourceError:
        return None, "insufficient_label_data"


def safe_parse_spl(xml_bytes: bytes) -> tuple[object | None, str]:
    try:
        parsed = parse_spl_xml(
            xml_bytes,
            expected_set_id=MOCK_SET_ID,
            xml_url="https://example.invalid/mock.xml",
        )
        return parsed, "parsed"
    except DailyMedError:
        return None, "openfda_fallback"


def run_mock_failure_checks() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}

    timeout = httpx.ReadTimeout(
        "mock DailyMed timeout",
        request=httpx.Request("GET", "https://example.invalid"),
    )
    with patch("spl_parser.httpx.Client.get", side_effect=timeout):
        label, status = safe_dailymed_enrichment(_label())
    results["dailymed_timeout"] = {
        "passed": status == "openfda_fallback" and label.dailymed_warning is not None,
        "outcome": status,
    }

    with patch(
        "data_sources._request_json",
        side_effect=DataSourceError("openFDA", "mock openFDA timeout"),
    ):
        label, status = safe_openfda_fetch(_usage(), Settings())
    results["openfda_timeout"] = {
        "passed": label is None and status == "insufficient_label_data",
        "outcome": status,
    }

    parsed, status = safe_parse_spl(b"<document><malformed></document>")
    results["malformed_spl_xml"] = {
        "passed": parsed is None and status == "openfda_fallback",
        "outcome": status,
    }

    plan = build_event_search_plan("bradycardia")
    empty_label = _label(sections={}).model_copy(
        update={"extraction_source": "dailymed_spl_xml", "spl_chunks": []}
    )
    extracted = extract_structured_event_evidence(
        selected_drugs=[_usage()],
        labels=[empty_label],
        normalized_event=plan.normalized_event,
        direct_terms=plan.direct_terms,
        related_terms=plan.related_terms,
    )
    empty_status = extracted.items[0].evidence_status if extracted.items else "missing"
    results["empty_safety_sections"] = {
        "passed": empty_status == "insufficient_label_data",
        "outcome": empty_status,
    }
    return results
