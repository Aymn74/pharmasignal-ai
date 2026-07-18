from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data_sources import (
    DataSourceError,
    enrich_class_candidates,
    fetch_rxclass_catalog,
    fetch_cms_usage,
    fetch_openfda_label,
    get_classes_by_drug,
    get_class_members,
    load_drug_class_catalog,
    normalize_drug_name,
    normalize_name,
    search_class_catalog,
    search_drug_candidates,
    store_analysis_run,
    sync_drug_class_catalog,
    supabase_configuration_status,
    test_openfda_connection,
    test_supabase,
)
from event_analysis import (
    ABSENCE_LIMITATION,
    ANALYSIS_RULES_VERSION,
    build_structured_analysis,
    build_event_search_plan,
)
from models import (
    AnalysisResult,
    ClassCandidate,
    CmsUsage,
    DrugCandidate,
    DrugMember,
    LabelEvidence,
    Settings,
)
from spl_parser import DailyMedError, enrich_label_with_dailymed
from structured_evidence import (
    EXTRACTION_RULES_VERSION,
    extract_structured_event_evidence,
)


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env", override=False)

st.set_page_config(
    page_title="PharmaSignal AI",
    page_icon="💊",
    layout="wide",
)

SETTINGS = Settings(_env_file=APP_DIR / ".env")


@st.cache_data(ttl=21600, show_spinner=False)
def cached_class_catalog():
    if supabase_configuration_status(SETTINGS) != "Not configured":
        try:
            catalog, detail = load_drug_class_catalog(SETTINGS)
            return catalog, [detail], "Loaded from Supabase cache"
        except DataSourceError:
            pass
    catalog, details = fetch_rxclass_catalog(SETTINGS)
    status = "Live RxClass catalog active"
    if supabase_configuration_status(SETTINGS) != "Not configured":
        try:
            details.append(sync_drug_class_catalog(catalog, SETTINGS))
            status = "Synced to Supabase from live RxClass"
        except DataSourceError as exc:
            status = f"Live RxClass active; cache sync unavailable: {exc.user_message}"
    return catalog, details, status


@st.cache_data(ttl=3600, show_spinner=False)
def cached_enriched_classes(candidates_json: str, limit: int):
    candidates = [ClassCandidate.model_validate(item) for item in json.loads(candidates_json)]
    return enrich_class_candidates(candidates, SETTINGS, limit=limit)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_drug_search(query: str):
    return search_drug_candidates(query, SETTINGS)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_drug_normalization(query: str):
    return normalize_drug_name(query, SETTINGS)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_classes_by_drug(drug_json: str):
    drug = DrugCandidate.model_validate_json(drug_json)
    classes, details = get_classes_by_drug(drug, SETTINGS)
    enriched, member_details = enrich_class_candidates(classes, SETTINGS, limit=12)
    return enriched, details + member_details


@st.cache_data(ttl=3600, show_spinner=False)
def cached_class_members(candidate_json: str):
    return get_class_members(ClassCandidate.model_validate_json(candidate_json), SETTINGS)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_cms_usage(members_json: str):
    raw = json.loads(members_json)
    members = [DrugMember.model_validate(item) for item in raw]
    return fetch_cms_usage(members, SETTINGS)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_openfda_label(usage_json: str):
    return fetch_openfda_label(CmsUsage.model_validate_json(usage_json), SETTINGS)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_dailymed_label(label_json: str):
    return enrich_label_with_dailymed(
        LabelEvidence.model_validate_json(label_json),
        timeout_seconds=SETTINGS.request_timeout_seconds,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_supabase_test():
    return test_supabase(SETTINGS)


@st.cache_data(ttl=300, show_spinner=False)
def cached_openfda_connection_test():
    return test_openfda_connection(SETTINGS)


def initialize_state() -> None:
    defaults = {
        "class_candidates": [],
        "possible_class_candidates": [],
        "drug_candidates": [],
        "drug_class_candidates": [],
        "class_query_drug_hint": None,
        "analysis": None,
        "selected_class": None,
        "confirmed_class": None,
        "class_members": [],
        "cms_usage": [],
        "selected_drugs": [],
        "labels": [],
        "analysis_result": None,
        "export_payload": None,
        "download_json": None,
        "adverse_event_query": "",
        "drug_event_evidence": [],
        "class_level_summary": None,
        "class_assessment": "",
        "search_details": [],
        "search_terms": [],
        "last_class_query": "",
        "last_drug_query": "",
        "synonym_expansion": None,
        "catalog_status": "Not loaded",
        "discovery_method": "Search by class name",
        "openfda_connection_detail": None,
        "ui_error": "",
        "source_status": {
            "RxClass": "Not checked",
            "CMS": "Not checked",
            "openFDA": "Connection failed",
            "DailyMed": "Not run",
            "Supabase": "Not configured",
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ANALYSIS_STATE_DEFAULTS = {
    "analysis": None,
    "selected_class": None,
    "confirmed_class": None,
    "class_members": [],
    "cms_usage": [],
    "selected_drugs": [],
    "labels": [],
    "analysis_result": None,
    "export_payload": None,
    "download_json": None,
    "adverse_event_query": "",
    "drug_event_evidence": [],
    "class_level_summary": None,
    "class_assessment": "",
}

SELECTION_WIDGET_KEYS = [
    "class_name_result",
    "class_name_recommended_result",
    "class_name_technical_result",
    "possible_class_result",
    "possible_class_recommended_result",
    "possible_class_technical_result",
    "drug_result",
    "drug_class_result",
    "drug_class_recommended_result",
    "drug_class_mechanistic_result",
    "drug_class_other_result",
    "drug_class_combination_result",
    "drug_class_technical_result",
    "browse_recommended_result",
    "browse_technical_result",
    "browse_class_result",
    "exceptional_class_confirmation",
    "adverse_event_input",
]


def clear_analysis_state() -> None:
    for key, value in ANALYSIS_STATE_DEFAULTS.items():
        st.session_state[key] = list(value) if isinstance(value, list) else value


def clear_selection_widgets(*, except_keys: set[str] | None = None) -> None:
    keep = except_keys or set()
    for key in SELECTION_WIDGET_KEYS:
        if key not in keep:
            st.session_state.pop(key, None)


def reset_discovery_context() -> None:
    clear_analysis_state()
    clear_selection_widgets()
    st.session_state.class_candidates = []
    st.session_state.possible_class_candidates = []
    st.session_state.drug_candidates = []
    st.session_state.drug_class_candidates = []
    st.session_state.class_query_drug_hint = None
    st.session_state.search_details = []
    st.session_state.search_terms = []
    st.session_state.synonym_expansion = None
    st.session_state.ui_error = ""


def handle_drug_selection_change() -> None:
    clear_analysis_state()
    st.session_state.drug_class_candidates = []
    clear_selection_widgets(except_keys={"drug_result"})


def handle_class_selection_change(active_key: str, peer_keys: list[str]) -> None:
    clear_analysis_state()
    st.session_state.pop("exceptional_class_confirmation", None)
    for key in peer_keys:
        if key != active_key:
            st.session_state.pop(key, None)


def reset_browse_selection() -> None:
    clear_analysis_state()
    for key in ["browse_recommended_result", "browse_technical_result", "exceptional_class_confirmation"]:
        st.session_state.pop(key, None)


def move_class_query_to_drug_search() -> None:
    query = st.session_state.last_class_query
    reset_discovery_context()
    st.session_state.discovery_method = "Search by drug name"
    st.session_state.drug_query_input = query
    st.session_state.last_drug_query = ""


def refresh_live_data() -> None:
    cached_class_catalog.clear()
    cached_enriched_classes.clear()
    cached_drug_search.clear()
    cached_drug_normalization.clear()
    cached_classes_by_drug.clear()
    cached_class_members.clear()
    cached_cms_usage.clear()
    cached_openfda_label.clear()
    cached_dailymed_label.clear()
    cached_openfda_connection_test.clear()
    cached_supabase_test.clear()
    reset_discovery_context()


def status_value(source: str) -> str:
    value = st.session_state.source_status.get(source, "Not checked")
    if value == "Connected":
        return "● Connected"
    if value.startswith("Connected —"):
        return f"● {value}"
    if value == "Not configured":
        return "○ Not configured"
    if value == "Not checked":
        return "○ Not checked"
    if value == "Not run":
        return "○ Not run"
    return f"● {value}"


def run_analysis(selected: ClassCandidate, adverse_event_query: str) -> None:
    source_details = list(st.session_state.search_details)
    if st.session_state.openfda_connection_detail:
        source_details.append(st.session_state.openfda_connection_detail)
    warnings: list[str] = [
        "CMS results describe Medicare Part D beneficiaries, not the entire population.",
        "CMS beneficiary counts are sums of matched published rows, not patient-level deduplication.",
    ]
    errors: list[str] = []
    statuses = dict(st.session_state.source_status)
    members: list[DrugMember] = []
    usage: list[CmsUsage] = []
    selected_drugs: list[CmsUsage] = []
    labels = []
    cms_identity = None

    try:
        members, selected, details = cached_class_members(selected.model_dump_json())
        source_details.extend(details)
        statuses["RxClass"] = "Connected"
    except DataSourceError as exc:
        statuses["RxClass"] = "Error"
        errors.append(exc.user_message)
        st.session_state.source_status = statuses
        st.session_state.ui_error = exc.user_message
        return

    try:
        members_json = json.dumps([member.model_dump(mode="json") for member in members])
        usage, cms_identity, details = cached_cms_usage(members_json)
        source_details.extend(details)
        statuses["CMS"] = "Connected"
        selected_drugs = usage[: max(1, SETTINGS.default_representative_count)]
        if not usage:
            warnings.append("CMS returned no defensible national generic-name matches for these RxClass ingredients.")
        approximate_count = sum(item.approximate_match for item in usage)
        if approximate_count:
            warnings.append(
                f"CMS includes {approximate_count} salt/combination-normalized match(es). "
                "They are marked approximate in the table and must be reviewed."
            )
    except DataSourceError as exc:
        statuses["CMS"] = "Error"
        errors.append(exc.user_message)

    if selected_drugs:
        openfda_failed = False
        dailymed_success_count = 0
        dailymed_fallback_count = 0
        for item in selected_drugs:
            try:
                label, details = cached_openfda_label(item.model_dump_json())
                source_details.extend(details)
                if label:
                    try:
                        label, dailymed_detail = cached_dailymed_label(label.model_dump_json())
                        source_details.append(dailymed_detail)
                        dailymed_success_count += 1
                    except DailyMedError as exc:
                        dailymed_fallback_count += 1
                        fallback_warning = (
                            f'DailyMed SPL XML unavailable for "{item.member.name}"; '
                            f"using current openFDA section text as fallback. {exc}"
                        )
                        warnings.append(fallback_warning)
                        label = label.model_copy(
                            update={
                                "extraction_source": "openfda_fallback",
                                "dailymed_warning": fallback_warning,
                            }
                        )
                    labels.append(label)
                    if label.label_match_confidence == "low":
                        warnings.append(
                            f'LOW-CONFIDENCE openFDA label match for "{item.member.name}": '
                            + " ".join(label.label_match_reasons)
                        )
                    elif label.label_match_confidence == "moderate":
                        warnings.append(
                            f'MODERATE-CONFIDENCE openFDA label match for "{item.member.name}". '
                            "Review the selected product and SPL effective time before interpreting evidence."
                        )
                else:
                    warnings.append(f'openFDA returned no label for "{item.member.name}".')
            except DataSourceError as exc:
                openfda_failed = True
                errors.append(exc.user_message)
        if openfda_failed:
            statuses["openFDA"] = "Connection failed"
        if dailymed_success_count and not dailymed_fallback_count:
            statuses["DailyMed"] = "Connected"
        elif dailymed_success_count:
            statuses["DailyMed"] = "Partial fallback"
        elif dailymed_fallback_count:
            statuses["DailyMed"] = "openFDA fallback"

    if not SETTINGS.openfda_api_key.strip():
        warnings.append("OPENFDA_API_KEY is not configured. Public access works at lower rate limits; a key is recommended.")

    event_plan = build_event_search_plan(adverse_event_query)
    structured_evidence = extract_structured_event_evidence(
        selected_drugs=selected_drugs,
        labels=labels,
        normalized_event=event_plan.normalized_event,
        direct_terms=event_plan.direct_terms,
        related_terms=event_plan.related_terms,
    )
    structured_items = structured_evidence.items
    drug_event_evidence, class_level_summary = build_structured_analysis(
        selected_class=selected.class_name,
        class_member_count=len(members),
        selected_drugs=selected_drugs,
        labels=labels,
        plan=event_plan,
        items=structured_items,
    )

    result = AnalysisResult(
        selected_class=selected,
        cms_dataset=cms_identity,
        class_members=members,
        cms_usage=usage,
        selected_drugs=selected_drugs,
        labels=labels,
        source_status=statuses,
        source_details=source_details,
        warnings=list(dict.fromkeys(warnings)),
        errors=errors,
        metadata={
            "cms_population": "Medicare Part D beneficiaries",
            "causality_notice": "Label content does not establish causality; absence does not mean no effect.",
        },
        adverse_event_query=adverse_event_query.strip(),
        normalized_event=event_plan.normalized_event,
        searched_terms=event_plan.searched_terms,
        drug_event_evidence=drug_event_evidence,
        class_level_summary=class_level_summary,
        class_assessment=class_level_summary.class_assessment,
        analysis_rules_version=ANALYSIS_RULES_VERSION,
        event_evidence_items=structured_items,
        evidence_item_count_before_merge=structured_evidence.evidence_count_before_merge,
        evidence_item_count=len(structured_items),
        unique_positive_evidence_count=sum(
            item.evidence_status == "explicit_positive" for item in structured_items
        ),
        negated_evidence_count=sum(
            item.evidence_status == "negated" for item in structured_items
        ),
        interaction_dependent_count=sum(
            item.evidence_status == "interaction_dependent" for item in structured_items
        ),
        extraction_rules_version=EXTRACTION_RULES_VERSION,
    )

    if statuses.get("Supabase") == "Connected":
        try:
            detail = store_analysis_run(result, SETTINGS)
            result.source_details.append(detail)
        except DataSourceError as exc:
            statuses["Supabase"] = "Error"
            result.source_status["Supabase"] = "Error"
            result.errors.append(exc.user_message)

    st.session_state.source_status = statuses
    st.session_state.analysis = result
    st.session_state.analysis_result = result
    st.session_state.selected_class = selected
    st.session_state.confirmed_class = selected
    st.session_state.class_members = list(result.class_members)
    st.session_state.cms_usage = list(result.cms_usage)
    st.session_state.selected_drugs = list(result.selected_drugs)
    st.session_state.labels = list(result.labels)
    st.session_state.export_payload = result.model_dump(mode="json")
    st.session_state.download_json = result.model_dump_json(indent=2)
    st.session_state.adverse_event_query = result.adverse_event_query
    st.session_state.drug_event_evidence = list(result.drug_event_evidence)
    st.session_state.class_level_summary = result.class_level_summary
    st.session_state.class_assessment = result.class_assessment
    st.session_state.ui_error = ""


def section_text(label, keys: list[str]) -> str:
    values = [label.sections[key] for key in keys if label.sections.get(key)]
    return "\n\n".join(values)


def display_list(values: list[str] | None, unavailable: str = "Unavailable") -> str:
    cleaned = [str(value).strip() for value in (values or []) if str(value).strip()]
    return ", ".join(cleaned) if cleaned else unavailable


def class_option_label(candidate: ClassCandidate) -> str:
    relation = ", ".join(candidate.relations) or candidate.rela or "All / unspecified"
    sources = ", ".join(candidate.rela_sources) or candidate.rela_source
    badge = " | Recommended for class-level analysis" if candidate.recommended_for_analysis else ""
    return (
        f"{candidate.class_name} | {candidate.class_id} | {candidate.class_type} | "
        f"{sources} | {relation}{badge}"
    )


def is_technical_candidate(candidate: ClassCandidate) -> bool:
    return candidate.class_type.upper() == "CHEM" or (
        candidate.member_count is not None and candidate.member_count < 2
    )


def partition_class_candidates(
    candidates: list[ClassCandidate],
) -> tuple[list[ClassCandidate], list[ClassCandidate]]:
    recommended = [item for item in candidates if not is_technical_candidate(item)]
    technical = [item for item in candidates if is_technical_candidate(item)]
    return recommended, technical


def likely_drug_hint(candidates: list[DrugCandidate]) -> DrugCandidate | None:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.tty.upper() in {"IN", "PIN", "MIN", "BN"}
        ),
        None,
    )


def render_class_results(
    candidates: list[ClassCandidate],
    *,
    heading: str,
    key: str,
    peer_keys: list[str] | None = None,
    show_drug_context: bool = False,
) -> ClassCandidate | None:
    if not candidates:
        return None
    st.markdown(f"#### {heading}")
    rows = []
    for item in candidates:
        row = {
            "Class name": item.class_name,
            "Class ID": item.class_id,
            "Type": item.class_type,
            "Sources": ", ".join(item.rela_sources) or item.rela_source,
            "Preferred source": item.preferred_source or item.rela_source,
            "Additional sources": ", ".join(item.additional_sources) or "None",
            "Relations": ", ".join(item.relations) or item.rela or "All / unspecified",
            "Members": str(item.member_count) if item.member_count is not None else "On selection",
            "Examples": ", ".join(item.example_members[:3]) or "On selection",
        }
        if show_drug_context:
            row["Why shown"] = item.why_shown or "Direct ingredient membership"
            row["Status"] = (
                "Recommended for class-level analysis"
                if item.recommended_for_analysis
                else "Additional classification"
            )
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(460, 84 + 35 * len(rows)))
    selected_index = st.selectbox(
        "Select a drug class",
        options=list(range(len(candidates))),
        index=None,
        placeholder="Choose a class; no result is selected automatically",
        format_func=lambda index: class_option_label(candidates[index]),
        key=key,
        on_change=handle_class_selection_change,
        args=(key, peer_keys or []),
    )
    return candidates[selected_index] if selected_index is not None else None


def render_grouped_class_results(
    candidates: list[ClassCandidate], *, recommended_heading: str, key_prefix: str
) -> ClassCandidate | None:
    recommended, technical = partition_class_candidates(candidates)
    recommended_key = f"{key_prefix}_recommended_result"
    technical_key = f"{key_prefix}_technical_result"
    selected: ClassCandidate | None = None
    if recommended:
        selected = render_class_results(
            recommended,
            heading=recommended_heading,
            key=recommended_key,
            peer_keys=[technical_key],
        )
    if technical:
        st.warning(
            "These records are chemical or single-ingredient classifications and are not "
            "recommended for class-level analysis."
        )
        technical_selected = render_class_results(
            technical,
            heading="Technical or single-ingredient classifications",
            key=technical_key,
            peer_keys=[recommended_key],
        )
        if technical_selected is not None:
            selected = technical_selected
    return selected


def render_drug_class_results(candidates: list[ClassCandidate]) -> ClassCandidate | None:
    recommended = [item for item in candidates if item.recommended_for_analysis]
    recommended_keys = {(item.class_id, item.class_type) for item in recommended}
    mechanistic = [
        item
        for item in candidates
        if (item.class_id, item.class_type) not in recommended_keys
        and item.membership_category == "other-mechanistic"
    ]
    mechanistic_keys = {(item.class_id, item.class_type) for item in mechanistic}
    combination = [
        item
        for item in candidates
        if (item.class_id, item.class_type) not in recommended_keys
        and (item.class_id, item.class_type) not in mechanistic_keys
        and item.membership_category == "combination"
    ]
    combination_keys = {(item.class_id, item.class_type) for item in combination}
    technical = [
        item
        for item in candidates
        if (item.class_id, item.class_type) not in recommended_keys
        and (item.class_id, item.class_type) not in mechanistic_keys
        and (item.class_id, item.class_type) not in combination_keys
        and (
            item.membership_category == "technical"
            or item.class_type.upper() in {"CHEM", "VA", "STRUCT", "DISPOS", "SCHEDULE", "PK"}
            or (item.member_count is not None and item.member_count < 2)
        )
    ]
    technical_keys = {(item.class_id, item.class_type) for item in technical}
    other = [
        item
        for item in candidates
        if (item.class_id, item.class_type) not in recommended_keys
        and (item.class_id, item.class_type) not in mechanistic_keys
        and (item.class_id, item.class_type) not in combination_keys
        and (item.class_id, item.class_type) not in technical_keys
    ]
    all_keys = [
        "drug_class_recommended_result",
        "drug_class_mechanistic_result",
        "drug_class_other_result",
        "drug_class_combination_result",
        "drug_class_technical_result",
    ]
    selected: ClassCandidate | None = None
    if recommended:
        st.success("Recommended for class-level analysis")
        selected = render_class_results(
            recommended,
            heading="Analysis-ready classes",
            key="drug_class_recommended_result",
            peer_keys=all_keys,
            show_drug_context=True,
        )
    else:
        st.info("No direct multi-member EPC, MOA, or ATC class was confirmed for this drug.")

    if mechanistic or other or combination or technical:
        with st.expander("Show additional technical and combination classifications"):
            if mechanistic:
                mechanistic_selected = render_class_results(
                    mechanistic,
                    heading="Other mechanistic classifications",
                    key="drug_class_mechanistic_result",
                    peer_keys=all_keys,
                    show_drug_context=True,
                )
                selected = mechanistic_selected or selected
            if other:
                other_selected = render_class_results(
                    other,
                    heading="Related indications",
                    key="drug_class_other_result",
                    peer_keys=all_keys,
                    show_drug_context=True,
                )
                selected = other_selected or selected
            if combination:
                combination_selected = render_class_results(
                    combination,
                    heading="Combination-product classifications",
                    key="drug_class_combination_result",
                    peer_keys=all_keys,
                    show_drug_context=True,
                )
                selected = combination_selected or selected
            if technical:
                technical_selected = render_class_results(
                    technical,
                    heading="Technical classifications",
                    key="drug_class_technical_result",
                    peer_keys=all_keys,
                    show_drug_context=True,
                )
                selected = technical_selected or selected
    return selected


def render_confirmation(candidate: ClassCandidate) -> None:
    try:
        with st.spinner("Loading official RxClass members for confirmation..."):
            members, resolved, details = cached_class_members(candidate.model_dump_json())
        resolved = resolved.model_copy(
            update={
                "member_count": len(members),
                "example_members": [member.name for member in members[:5]],
            }
        )
        existing_details = {
            (detail.source, detail.query) for detail in st.session_state.search_details
        }
        st.session_state.search_details = list(st.session_state.search_details) + [
            detail for detail in details if (detail.source, detail.query) not in existing_details
        ]
        st.session_state.source_status["RxClass"] = "Connected"
    except DataSourceError:
        st.info(
            "This class could not be resolved to ingredient members for the current analysis. "
            "Choose another result or search by a drug name."
        )
        return

    st.session_state.selected_class = resolved
    technical_classification = is_technical_candidate(resolved)

    st.markdown("### Confirm this drug class")
    with st.container(border=True):
        fields = st.columns(3)
        fields[0].metric("Official class", resolved.class_name)
        fields[1].metric("Class ID", resolved.class_id)
        fields[2].metric("Class type", resolved.class_type)
        second = st.columns(3)
        second[0].metric("Relation source", resolved.rela_source)
        second[1].metric("Relation", resolved.rela or "All / unspecified")
        second[2].metric("Member count", resolved.member_count or 0)
        st.markdown("**Five example members:** " + display_list(resolved.example_members[:5]))
        st.caption(
            "EPC describes a pharmacologic class used in product labeling. MOA describes how a drug acts. "
            "They answer different questions, so confirm the type you intend to analyze."
        )
        if resolved.discovered_by_drug and not resolved.direct_membership:
            if resolved.combination_membership:
                st.warning(
                    "This classification appears through combination products and may not represent "
                    "the active ingredient alone. Choose a direct class for class-level analysis."
                )
            else:
                st.warning(
                    "This technical or product-derived classification is not a direct ingredient "
                    "membership and may not represent the active ingredient alone. Choose a direct class for analysis."
                )
            return
        exceptional_confirmation = True
        if technical_classification:
            st.warning(
                "This classification contains only one ingredient and is not suitable for "
                "class-level analysis."
                if (resolved.member_count or 0) < 2
                else "This is a technical chemical classification and is not recommended for class-level analysis."
            )
            exceptional_confirmation = st.checkbox(
                "I understand this is a technical or single-ingredient classification and want to analyze it exceptionally.",
                key="exceptional_class_confirmation",
            )
        adverse_event_query = st.text_input(
            "Adverse event or safety outcome",
            placeholder="Examples: rhabdomyolysis, bradycardia, hypersensitivity",
            key="adverse_event_input",
        )
        if adverse_event_query.strip():
            event_plan = build_event_search_plan(adverse_event_query)
            st.caption(f"Normalized event: {event_plan.normalized_event}")
            st.markdown("**Searched terms:** " + display_list(event_plan.searched_terms))
        else:
            st.caption("Enter an adverse event or safety outcome before starting analysis.")
        if st.button(
            "Analyze this classification and event"
            if technical_classification
            else "Analyze this drug class and event",
            type="primary",
            key="use_confirmed_drug_class",
            disabled=not exceptional_confirmation or not adverse_event_query.strip(),
        ):
            clear_analysis_state()
            st.session_state.selected_class = resolved
            st.session_state.confirmed_class = resolved
            with st.spinner("Retrieving live RxClass, filtered CMS, and openFDA evidence..."):
                run_analysis(resolved, adverse_event_query)
            st.rerun()


initialize_state()

if supabase_configuration_status(SETTINGS) == "Not configured":
    st.session_state.source_status["Supabase"] = "Not configured"
else:
    try:
        supabase_status, _ = cached_supabase_test()
        st.session_state.source_status["Supabase"] = supabase_status
    except DataSourceError as exc:
        st.session_state.source_status["Supabase"] = "Error"
        if not st.session_state.ui_error:
            st.session_state.ui_error = exc.user_message

try:
    openfda_status, openfda_detail = cached_openfda_connection_test()
    st.session_state.source_status["openFDA"] = openfda_status
    st.session_state.openfda_connection_detail = openfda_detail
except DataSourceError:
    st.session_state.source_status["openFDA"] = "Connection failed"
    st.session_state.openfda_connection_detail = None

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #475569;
        padding: .8rem;
        border-radius: .75rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {color: #cbd5e1 !important;}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        min-height: 2.4rem;
        white-space: normal !important;
        overflow: visible !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.15 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
        color: #f8fafc !important;
        font-size: 1.35rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, refresh_col = st.columns([5, 1])
with title_col:
    st.title("PharmaSignal AI")
    st.subheader("Live Drug Class Evidence Explorer")
with refresh_col:
    st.write("")
    if st.button("Refresh Live Data", type="secondary", width="stretch"):
        refresh_live_data()
        st.rerun()

if not SETTINGS.openfda_api_key.strip():
    st.info("openFDA is available without a key at public rate limits. Configure OPENFDA_API_KEY for more reliable repeated use.")

status_columns = st.columns(4)
for column, source in zip(status_columns, ["RxClass", "CMS", "openFDA", "Supabase"]):
    with column:
        st.metric(source, status_value(source))

st.sidebar.header("Source status")
for source in ["RxClass", "CMS", "openFDA", "Supabase"]:
    st.sidebar.markdown(f"**{source}:** {st.session_state.source_status.get(source, 'Not checked')}")

if st.session_state.ui_error:
    st.error(st.session_state.ui_error)

st.markdown("## Find a drug class")
search_mode = st.segmented_control(
    "Discovery method",
    options=["Search by class name", "Search by drug name", "Browse drug classes"],
    selection_mode="single",
    key="discovery_method",
    on_change=reset_discovery_context,
)

selected_candidate: ClassCandidate | None = None

if search_mode == "Search by class name":
    with st.form("class_name_search_form"):
        query = st.text_input(
            "Class name or common term",
            placeholder="Examples: beta, beta blockers, statins",
            key="class_query_input",
        )
        search_clicked = st.form_submit_button("Search classes", type="primary")

    if search_clicked:
        reset_discovery_context()
        st.session_state.last_class_query = query
        try:
            with st.spinner("Searching the official RxClass catalog..."):
                catalog, details, catalog_status = cached_class_catalog()
                strong, possible, expansion, terms = search_class_catalog(query, catalog)
                if strong:
                    payload = json.dumps([item.model_dump(mode="json") for item in strong])
                    strong, member_details = cached_enriched_classes(payload, 12)
                elif possible:
                    payload = json.dumps([item.model_dump(mode="json") for item in possible])
                    possible, member_details = cached_enriched_classes(payload, 8)
                else:
                    member_details = []
                drug_hint = None
                drug_details = []
                if not strong:
                    try:
                        drug_candidates, drug_details = cached_drug_normalization(query)
                        drug_hint = likely_drug_hint(drug_candidates)
                    except DataSourceError:
                        drug_hint = None
                    if drug_hint is not None:
                        possible = []
            st.session_state.class_candidates = strong
            st.session_state.possible_class_candidates = possible
            st.session_state.class_query_drug_hint = drug_hint
            st.session_state.search_details = details + member_details + drug_details
            st.session_state.search_terms = terms
            st.session_state.synonym_expansion = expansion
            st.session_state.catalog_status = catalog_status
            st.session_state.source_status["RxClass"] = "Connected"
        except DataSourceError as exc:
            st.session_state.source_status["RxClass"] = "Error"
            st.session_state.ui_error = exc.user_message
        st.rerun()

    if st.session_state.synonym_expansion:
        st.info(
            "Synonym assist used: "
            f'"{st.session_state.last_class_query}" '
            f'was expanded to "{st.session_state.synonym_expansion}". '
            "The selectable records below are still official RxClass results."
        )

    strong_candidates: list[ClassCandidate] = st.session_state.class_candidates
    possible_candidates: list[ClassCandidate] = st.session_state.possible_class_candidates
    drug_hint: DrugCandidate | None = st.session_state.class_query_drug_hint
    if strong_candidates:
        selected_candidate = render_grouped_class_results(
            strong_candidates,
            recommended_heading="Official class matches",
            key_prefix="class_name",
        )
    elif drug_hint is not None:
        st.info("This appears to be a drug name. Use Search by drug name.")
        st.caption(
            f"RxNorm normalized this entry to {drug_hint.name} "
            f"(RXCUI {drug_hint.rxcui}, {drug_hint.tty})."
        )
        st.button(
            "Continue with Search by drug name",
            type="primary",
            on_click=move_class_query_to_drug_search,
        )
    elif possible_candidates:
        st.warning("No strong class match found")
        selected_candidate = render_grouped_class_results(
            possible_candidates,
            recommended_heading="Possible matches",
            key_prefix="possible_class",
        )
    elif st.session_state.search_terms:
        st.info(
            "No class matched those words. Try searching by a drug name or use a shorter part of the class name."
        )
        st.caption("Search words used: " + ", ".join(st.session_state.search_terms))

elif search_mode == "Search by drug name":
    with st.form("drug_name_search_form"):
        drug_query = st.text_input(
            "Generic name, brand name, or RXCUI",
            placeholder="Examples: metoprolol, Lopressor, 6918",
            key="drug_query_input",
        )
        drug_search_clicked = st.form_submit_button("Search drugs", type="primary")

    if drug_search_clicked:
        reset_discovery_context()
        st.session_state.last_drug_query = drug_query
        try:
            with st.spinner("Searching live RxNorm concepts..."):
                drugs, details = cached_drug_search(drug_query)
            st.session_state.drug_candidates = drugs
            st.session_state.search_details = details
            st.session_state.source_status["RxClass"] = "Connected"
        except DataSourceError as exc:
            st.session_state.drug_candidates = []
            st.session_state.source_status["RxClass"] = "Error"
            st.session_state.ui_error = exc.user_message
        st.rerun()

    drugs: list[DrugCandidate] = st.session_state.drug_candidates
    if drugs:
        drug_index = st.selectbox(
            "Select the intended RxNorm drug",
            options=list(range(len(drugs))),
            index=None,
            placeholder="Choose the intended drug; ambiguous matches are not auto-selected",
            format_func=lambda index: (
                f"{drugs[index].name} | RXCUI {drugs[index].rxcui} | "
                f"{drugs[index].tty or 'type unavailable'}"
            ),
            key="drug_result",
            on_change=handle_drug_selection_change,
        )
        selected_drug = drugs[drug_index] if drug_index is not None else None
        if st.button(
            "Find classes for this drug",
            type="primary",
            disabled=selected_drug is None,
        ):
            clear_analysis_state()
            for key in [
                "drug_class_recommended_result",
                "drug_class_mechanistic_result",
                "drug_class_other_result",
                "drug_class_combination_result",
                "drug_class_technical_result",
                "exceptional_class_confirmation",
            ]:
                st.session_state.pop(key, None)
            try:
                with st.spinner("Loading official RxClass relationships for the selected drug..."):
                    classes, details = cached_classes_by_drug(selected_drug.model_dump_json())
                st.session_state.drug_class_candidates = classes
                st.session_state.search_details = list(st.session_state.search_details) + details
                st.session_state.source_status["RxClass"] = "Connected"
            except DataSourceError as exc:
                st.session_state.drug_class_candidates = []
                st.session_state.ui_error = exc.user_message
            st.rerun()
    elif not st.session_state.ui_error and st.session_state.last_drug_query:
        st.info("No RxNorm drug matched that entry. Try a generic name, brand name, or RXCUI.")

    drug_classes: list[ClassCandidate] = st.session_state.drug_class_candidates
    if drug_classes:
        selected_candidate = render_drug_class_results(drug_classes)

else:
    try:
        with st.spinner("Loading the official RxClass catalog..."):
            catalog, details, catalog_status = cached_class_catalog()
        st.session_state.catalog_status = catalog_status
        st.session_state.search_details = details
        st.session_state.source_status["RxClass"] = "Connected"
    except DataSourceError as exc:
        catalog = []
        st.session_state.ui_error = exc.user_message
        st.session_state.source_status["RxClass"] = "Error"

    if catalog:
        st.caption(
            f"{len(catalog):,} official RxClass records. Member details are fetched only after selection. "
            f"Catalog: {st.session_state.catalog_status}."
        )
        if st.button("Refresh catalog from RxClass", type="secondary"):
            reset_browse_selection()
            try:
                with st.spinner("Refreshing the official RxClass catalog..."):
                    live_catalog, live_details = fetch_rxclass_catalog(SETTINGS)
                    if supabase_configuration_status(SETTINGS) != "Not configured":
                        live_details.append(sync_drug_class_catalog(live_catalog, SETTINGS))
                cached_class_catalog.clear()
                st.session_state.search_details = live_details
                st.session_state.catalog_status = (
                    "Synced to Supabase from live RxClass"
                    if supabase_configuration_status(SETTINGS) != "Not configured"
                    else "Refreshed from live RxClass; Supabase not configured"
                )
                st.success(st.session_state.catalog_status)
            except DataSourceError as exc:
                st.session_state.ui_error = exc.user_message
        filter_columns = st.columns(5)
        keyword = filter_columns[0].text_input(
            "Search word", key="browse_keyword", on_change=reset_browse_selection
        )
        class_types = sorted({item.class_type for item in catalog if item.class_type})
        type_filter = filter_columns[1].selectbox(
            "Class type", ["All"] + class_types, on_change=reset_browse_selection
        )
        relation_sources = sorted({item.rela_source for item in catalog if item.rela_source})
        source_filter = filter_columns[2].selectbox(
            "Relation source", ["All"] + relation_sources, on_change=reset_browse_selection
        )
        letters = sorted({item.class_name[0].upper() for item in catalog if item.class_name})
        letter_filter = filter_columns[3].selectbox(
            "First letter", ["All"] + letters, on_change=reset_browse_selection
        )
        count_filter = filter_columns[4].selectbox(
            "Member count",
            ["All", "Known only", "Unknown", "1-9", "10-49", "50+"],
            on_change=reset_browse_selection,
        )

        normalized_keyword = normalize_name(keyword)
        filtered = []
        for item in catalog:
            if normalized_keyword and normalized_keyword not in normalize_name(item.class_name):
                continue
            if type_filter != "All" and item.class_type != type_filter:
                continue
            if source_filter != "All" and item.rela_source != source_filter:
                continue
            if letter_filter != "All" and not item.class_name.upper().startswith(letter_filter):
                continue
            count = item.member_count
            if count_filter == "Known only" and count is None:
                continue
            if count_filter == "Unknown" and count is not None:
                continue
            if count_filter == "1-9" and not (count is not None and 1 <= count <= 9):
                continue
            if count_filter == "10-49" and not (count is not None and 10 <= count <= 49):
                continue
            if count_filter == "50+" and not (count is not None and count >= 50):
                continue
            filtered.append(item)

        st.caption(f"{len(filtered):,} matches; showing the first 200.")
        selected_candidate = render_grouped_class_results(
            filtered[:200],
            recommended_heading="Recommended drug classes",
            key_prefix="browse",
        )

if selected_candidate is not None:
    render_confirmation(selected_candidate)

result: AnalysisResult | None = st.session_state.analysis
if result:
    if result.errors:
        for error in result.errors:
            st.error(error)
    for warning in result.warnings:
        st.warning(warning)

    with st.expander("SPL extraction diagnostics"):
        if not result.labels:
            st.info("No selected labels are available for SPL extraction diagnostics.")
        for label in result.labels:
            diagnostics = label.spl_diagnostics
            st.markdown(f"#### {label.requested_name}")
            diagnostic_row = {
                "SET ID": label.selected_spl_set_id or "Unavailable",
                "SPL version": label.spl_version or "Unavailable",
                "Section count": diagnostics.section_count if diagnostics else 0,
                "Total chunks": diagnostics.total_chunks if diagnostics else len(label.spl_chunks),
                "Unique chunks": diagnostics.unique_chunks if diagnostics else len(label.spl_chunks),
                "Duplicate rate": (
                    f"{diagnostics.duplicate_rate:.2%}" if diagnostics else "Unavailable"
                ),
                "Median chunk length": (
                    round(diagnostics.median_characters, 1) if diagnostics else 0
                ),
                "Maximum chunk length": diagnostics.maximum_characters if diagnostics else 0,
                "Extraction source": label.extraction_source,
            }
            st.dataframe(pd.DataFrame([diagnostic_row]), hide_index=True, width="stretch")
            sample_rows = [
                {
                    "section_code": chunk.section_code,
                    "section_title": chunk.section_title,
                    "subsection_title": chunk.subsection_title or "",
                    "chunk_type": chunk.chunk_type,
                    "character_count": chunk.character_count,
                    "source_path": chunk.source_path,
                    "chunk_hash": chunk.chunk_hash,
                    "text preview": (
                        chunk.text[:237].rstrip() + "..."
                        if len(chunk.text) > 240
                        else chunk.text
                    ),
                }
                for chunk in label.spl_chunks[:10]
            ]
            if sample_rows:
                st.caption("First 10 unique chunks in source order")
                st.dataframe(pd.DataFrame(sample_rows), hide_index=True, width="stretch")
            else:
                st.info("No DailyMed SPL XML chunks were available; openFDA fallback remains active.")

    with st.expander("Structured event evidence"):
        structured_rows = [
            {
                "Drug": item.drug_name,
                "Status": item.evidence_status,
                "Assertion": item.assertion,
                "Subject": item.subject,
                "Context": display_list(item.evidence_context),
                "Frequency": item.frequency_text or "Unavailable",
                "Comparator": item.comparator_text or "Unavailable",
                "Population": display_list(item.population_context),
                "Interaction": display_list(item.interaction_context),
                "Temporal context": display_list(item.temporal_context),
                "Section": display_list(item.mentioned_in_sections),
                "Supporting quote": item.supporting_quote or "Unavailable",
                "Source path": display_list(item.source_paths),
            }
            for item in result.event_evidence_items
        ]
        if structured_rows:
            st.caption(
                f"Deterministic extraction: {result.evidence_item_count_before_merge} item(s) "
                f"before merge; {result.evidence_item_count} after merge."
            )
            st.dataframe(pd.DataFrame(structured_rows), hide_index=True, width="stretch")
        else:
            st.info("No structured event evidence items were extracted.")

    st.markdown("### Drug-level event evidence")
    st.markdown("**Searched terms:** " + display_list(result.searched_terms))
    evidence_rows = [
        {
            "Drug": item.drug_name,
            "CMS rank": item.cms_rank,
            "Evidence status": item.evidence_status,
            "Matched sections": ", ".join(item.matched_sections) or "None",
            "Label confidence": item.label_match_confidence,
            "Extraction source": item.extraction_source,
            "SPL version": item.spl_version or "Unavailable",
            "SPL identifier": item.selected_spl_set_id or "Unavailable",
            "Effective time": item.label_effective_time or "Unavailable",
        }
        for item in result.drug_event_evidence
    ]
    if evidence_rows:
        st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, width="stretch")
    else:
        st.info("No CMS-ranked drugs were available for event-level label review.")
    st.warning(ABSENCE_LIMITATION)

    for item in result.drug_event_evidence:
        with st.expander(f"{item.drug_name} — {item.evidence_status}"):
            trace_fields = st.columns(3)
            trace_fields[0].metric("RXCUI", item.rxcui)
            trace_fields[1].metric("CMS rank", item.cms_rank or "Unavailable")
            trace_fields[2].metric("Label confidence", item.label_match_confidence)
            st.markdown(
                "**Selected SPL set identifier:** "
                + (item.selected_spl_set_id or "Unavailable")
            )
            st.markdown("**Extraction source:** " + item.extraction_source)
            st.markdown("**SPL version:** " + (item.spl_version or "Unavailable"))
            st.markdown("**Effective time:** " + (item.label_effective_time or "Unavailable"))
            st.markdown("**Searched terms:** " + display_list(item.searched_terms))
            if item.evidence_snippets:
                st.markdown("**Evidence snippets:**")
                for snippet in item.evidence_snippets:
                    st.markdown(f"- `{snippet.section}` — matched **{snippet.matched_term}**")
                    st.caption(snippet.text)
                    st.caption(
                        " → ".join(
                            [
                                item.selected_spl_set_id or "SET ID unavailable",
                                f"version {snippet.spl_version or item.spl_version or 'unavailable'}",
                                f"LOINC {snippet.section_code or 'unavailable'}",
                                snippet.subsection_title or "No subsection title",
                                snippet.chunk_type,
                                snippet.source_path or "Source path unavailable",
                            ]
                        )
                    )
            else:
                st.info("No matching snippet was found in the reviewed safety sections.")
            st.markdown("**Limitations:**")
            for limitation in item.limitations:
                st.markdown(f"- {limitation}")

    summary = result.class_level_summary
    if summary:
        st.markdown("### Class-level synthesis")
        summary_metrics = st.columns(4)
        summary_metrics[0].metric("Selected class", summary.selected_class)
        summary_metrics[1].metric("Adverse event", summary.adverse_event)
        summary_metrics[2].metric("Class members", summary.class_member_count)
        summary_metrics[3].metric("Drugs analyzed", summary.drugs_analyzed)
        st.markdown("**Evidence-status distribution:**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Evidence status": status, "Count": count}
                    for status, count in summary.evidence_distribution.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown(f"**Class assessment:** `{summary.class_assessment}`")
        st.write(summary.interpretation)
        st.markdown("**Limitations:**")
        for limitation in summary.limitations:
            st.markdown(f"- {limitation}")

    st.markdown("### RxClass members")
    member_rows = [
        {"RXCUI": item.rxcui, "RxClass base ingredient": item.name, "TTY": item.tty}
        for item in result.class_members
    ]
    st.dataframe(pd.DataFrame(member_rows), hide_index=True, width="stretch")

    st.markdown("### Highest-use CMS ingredients")
    if result.cms_usage:
        usage_rows = [
            {
                "Rank": item.rank,
                "RxClass base ingredient": item.member.name,
                "RXCUI": item.member.rxcui,
                "CMS matched products or salts": ", ".join(item.cms_generic_names),
                "Total claims": item.total_claims,
                "Summed beneficiary count — not deduplicated": item.total_beneficiaries,
                "Year": item.data_year,
                "CMS match quality": item.match_quality,
                "Approximate": item.approximate_match,
            }
            for item in result.cms_usage
        ]
        st.dataframe(pd.DataFrame(usage_rows), hide_index=True, width="stretch")
        st.caption(
            "This value is the sum of beneficiary counts reported across matched CMS rows. "
            "It is not a deduplicated count of unique patients. CMS covers Medicare Part D "
            "beneficiaries, not the whole population."
        )
    else:
        st.info("No CMS-ranked ingredients are available for this analysis. No substitute data were generated.")

    if result.selected_drugs:
        st.markdown("### Selected drug evidence cards")
        labels_by_rxcui = {label.rxcui: label for label in result.labels}
        for usage_item in result.selected_drugs:
            label = labels_by_rxcui.get(usage_item.member.rxcui)
            with st.container(border=True):
                st.markdown(f"#### #{usage_item.rank} — {usage_item.member.name}")
                metrics = st.columns(3)
                metrics[0].metric("Rank", usage_item.rank)
                metrics[1].metric("Total claims", f"{usage_item.total_claims:,}")
                metrics[2].metric(
                    "Summed beneficiary count — not deduplicated",
                    f"{usage_item.total_beneficiaries:,}"
                    if usage_item.total_beneficiaries is not None
                    else "Unavailable",
                )
                secondary_metrics = st.columns(3)
                secondary_metrics[0].metric("CMS year", usage_item.data_year)
                secondary_metrics[1].metric("CMS match quality", usage_item.match_quality)
                secondary_metrics[2].metric(
                    "Matched FDA labels", label.matched_label_count if label else 0
                )

                levels = st.columns(3)
                with levels[0]:
                    st.markdown("**RxClass base ingredient**")
                    st.write(usage_item.member.name)
                    st.caption(f"Base RXCUI: {usage_item.member.rxcui}")
                with levels[1]:
                    st.markdown("**CMS matched products or salts**")
                    st.write(display_list(usage_item.cms_generic_names))
                    st.caption("Usage is aggregated across these matched CMS rows.")
                with levels[2]:
                    st.markdown("**Selected FDA label product**")
                    if label:
                        st.write(display_list(label.generic_names))
                        st.caption(
                            "Brand: " + display_list(label.brand_names, "Brand name unavailable")
                        )
                    else:
                        st.write("No selected openFDA label")

                if usage_item.approximate_match:
                    st.warning("CMS approximate-match warning: " + usage_item.match_note)

                if label:
                    st.info(
                        "The CMS use figure is aggregated at the RxClass base-ingredient level. "
                        "The selected FDA label applies only to the product/salt shown below and "
                        "does not automatically represent every CMS product or salt aggregated above."
                    )

                    if label.label_match_confidence == "low":
                        st.error(
                            "Low-confidence label match. Treat this record as uncertain; it is not "
                            "confirmed to represent the base ingredient or all aggregated CMS products."
                        )

                    label_metrics = st.columns(3)
                    label_metrics[0].metric("Label match score", f"{label.label_match_score}/100")
                    label_metrics[1].metric(
                        "Label match confidence", label.label_match_confidence.upper()
                    )
                    label_metrics[2].metric(
                        "Effective time", label.effective_time or "Unavailable"
                    )

                    st.markdown("**Selected FDA generic name:** " + display_list(label.generic_names))
                    st.markdown(
                        "**Selected FDA brand name:** "
                        + display_list(label.brand_names, "Brand name unavailable")
                    )
                    st.markdown(
                        "**Selected FDA substance / salt:** "
                        + display_list(label.substance_names, "Substance or salt unavailable")
                    )
                    st.markdown(
                        "**Selected FDA RXCUI values:** "
                        + display_list(label.label_rxcuis, "FDA RXCUI unavailable")
                    )
                    product = label.selected_label_product
                    if product.get("dosage_forms") or product.get("routes"):
                        st.markdown(
                            "**Selected FDA form / route:** "
                            + display_list(product.get("dosage_forms"), "Form unavailable")
                            + " / "
                            + display_list(product.get("routes"), "Route unavailable")
                        )
                    st.markdown(
                        "**Selected SPL set identifier:** "
                        + (
                            label.selected_spl_set_id
                            if label.selected_spl_set_id
                            else "SPL identifier unavailable in the selected openFDA record"
                        )
                    )
                    st.markdown("**Why this label was selected:** " + label.selection_reason)
                    st.markdown("**Label match reasons:**")
                    for match_reason in label.label_match_reasons:
                        st.markdown(f"- {match_reason}")
                    with st.expander("Warnings"):
                        warning_text = section_text(
                            label, ["boxed_warning", "warnings", "warnings_and_cautions"]
                        )
                        st.write(warning_text or "Not present in the selected label; absence is not evidence of no effect.")
                    with st.expander("Adverse reactions"):
                        adverse = section_text(label, ["adverse_reactions"])
                        st.write(adverse or "Not present in the selected label; absence is not evidence of no effect.")
                    with st.expander("Drug interactions"):
                        interactions = section_text(label, ["drug_interactions"])
                        st.write(interactions or "Not present in the selected label; absence is not evidence of no effect.")
                else:
                    st.info("No matching openFDA label was found. No label text was inferred or substituted.")

    with st.expander("Raw Source Details"):
        detail_rows = [
            {
                "Source": detail.source,
                "Query (keys redacted)": detail.query,
                "Retrieved at (UTC)": detail.retrieved_at.isoformat(),
                "Records": detail.record_count,
                "Note": detail.note,
            }
            for detail in result.source_details
        ]
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")

    result_json = result.model_dump_json(indent=2)
    st.download_button(
        "Download Results as JSON",
        data=result_json,
        file_name="pharmasignal_analysis.json",
        mime="application/json",
    )

st.divider()
st.error(
    "This application is a research proof of concept.\n\n"
    "It does not establish causality and must not be used as an independent clinical decision tool."
)
