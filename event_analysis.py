from __future__ import annotations

import re
from dataclasses import dataclass

from models import (
    ClassLevelSummary,
    CmsUsage,
    DrugEventEvidence,
    EventEvidenceItem,
    EventEvidenceSnippet,
    LabelEvidence,
    SPLSectionChunk,
)
from spl_parser import section_key_for_code
from structured_evidence import extract_structured_event_evidence


ANALYSIS_RULES_VERSION = "event-label-rules-v1"
ABSENCE_LIMITATION = (
    "Not found in the reviewed label sections does not establish absence of risk."
)
REVIEWED_SECTIONS = [
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "contraindications",
    "drug_interactions",
]
REVIEWED_DAILYMED_SECTIONS = [
    "boxed_warning",
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "contraindications",
    "drug_interactions",
    "use_in_specific_populations",
]
SEARCHABLE_SPL_CHUNK_TYPES = {"paragraph", "list_item", "table_row"}


EVENT_TERMS: dict[str, dict[str, list[str]]] = {
    "rhabdomyolysis": {
        "direct": ["rhabdomyolysis", "muscle breakdown"],
        "related": ["myopathy"],
    },
    "bradycardia": {
        "direct": ["bradycardia", "slow heart rate", "reduced heart rate"],
        "related": [],
    },
    "hypersensitivity": {
        "direct": [
            "hypersensitivity",
            "anaphylaxis",
            "anaphylactic reaction",
            "allergic reaction",
        ],
        "related": [],
    },
    "serotonin syndrome": {
        "direct": ["serotonin syndrome"],
        "related": ["serotonergic syndrome", "serotonin toxicity"],
    },
    "hypotension": {
        "direct": ["hypotension", "low blood pressure"],
        "related": ["hypotensive episode"],
    },
    "qt prolongation": {
        "direct": ["qt prolongation", "prolonged qt", "qt interval prolongation"],
        "related": [],
    },
    "torsades de pointes": {
        "direct": ["torsades de pointes", "torsade de pointes"],
        "related": ["qt prolongation", "prolonged qt", "qt interval prolongation"],
    },
}


@dataclass(frozen=True)
class EventSearchPlan:
    event_query: str
    normalized_event: str
    direct_terms: list[str]
    related_terms: list[str]

    @property
    def searched_terms(self) -> list[str]:
        return list(dict.fromkeys(self.direct_terms + self.related_terms))


def _normalize_term(value: str) -> str:
    normalized = value.casefold().strip()
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    return " ".join(normalized.replace("-", " ").split())


def build_event_search_plan(event_query: str) -> EventSearchPlan:
    clean_query = event_query.strip()
    normalized_query = _normalize_term(clean_query)
    for canonical, groups in EVENT_TERMS.items():
        known_terms = groups["direct"] + groups["related"]
        if normalized_query in {_normalize_term(term) for term in known_terms}:
            return EventSearchPlan(
                event_query=clean_query,
                normalized_event=canonical,
                direct_terms=list(groups["direct"]),
                related_terms=list(groups["related"]),
            )
    return EventSearchPlan(
        event_query=clean_query,
        normalized_event=normalized_query,
        direct_terms=[normalized_query] if normalized_query else [],
        related_terms=[],
    )


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _short_snippet(text: str, start: int, end: int, *, maximum: int = 300) -> str:
    collapsed = " ".join(text.split())
    left = max(0, start - 120)
    right = min(len(collapsed), end + 160)
    sentence_left = max(
        collapsed.rfind(". ", left, start),
        collapsed.rfind("; ", left, start),
        collapsed.rfind(": ", left, start),
    )
    if sentence_left >= left:
        left = sentence_left + 2
    sentence_candidates = [
        position
        for position in [
            collapsed.find(". ", end, right),
            collapsed.find("; ", end, right),
        ]
        if position >= 0
    ]
    if sentence_candidates:
        right = min(sentence_candidates) + 1
    snippet = collapsed[left:right].strip()
    if len(snippet) > maximum:
        snippet = snippet[: maximum - 1].rstrip() + "…"
    if left > 0:
        snippet = "…" + snippet
    if right < len(collapsed):
        snippet += "…"
    return snippet


def _find_term_matches(
    section: str, text: str, terms: list[str]
) -> list[EventEvidenceSnippet]:
    collapsed = " ".join(text.split())
    snippets: list[EventEvidenceSnippet] = []
    for term in terms:
        match = _term_pattern(term).search(collapsed)
        if not match:
            continue
        snippets.append(
            EventEvidenceSnippet(
                section=section,
                matched_term=term,
                text=_short_snippet(collapsed, match.start(), match.end()),
                extraction_source="openfda_fallback",
                chunk_type="section_text",
                source_path=f"/openfda/sections/{section}",
            )
        )
    return snippets


def _find_chunk_matches(
    chunk: SPLSectionChunk, terms: list[str]
) -> list[EventEvidenceSnippet]:
    if chunk.chunk_type not in SEARCHABLE_SPL_CHUNK_TYPES:
        return []
    collapsed = " ".join(chunk.text.split())
    snippets: list[EventEvidenceSnippet] = []
    for term in terms:
        match = _term_pattern(term).search(collapsed)
        if not match:
            continue
        snippets.append(
            EventEvidenceSnippet(
                section=section_key_for_code(chunk.section_code),
                matched_term=term,
                text=_short_snippet(collapsed, match.start(), match.end()),
                extraction_source="dailymed_spl_xml",
                spl_version=chunk.version,
                section_code=chunk.section_code,
                subsection_title=chunk.subsection_title,
                chunk_type=chunk.chunk_type,
                source_path=chunk.source_path,
            )
        )
    return snippets


def build_drug_event_evidence(
    usage: CmsUsage,
    label: LabelEvidence | None,
    plan: EventSearchPlan,
) -> DrugEventEvidence:
    limitations = [
        "Only the selected openFDA label and the specified safety sections were reviewed."
    ]
    if label is None:
        limitations.extend(
            [
                "No selected openFDA label was available for this drug.",
                ABSENCE_LIMITATION,
            ]
        )
        return DrugEventEvidence(
            drug_name=usage.member.name,
            rxcui=usage.member.rxcui,
            cms_rank=usage.rank,
            event_query=plan.event_query,
            searched_terms=plan.searched_terms,
            evidence_status="insufficient_label_data",
            label_match_confidence="unavailable",
            extraction_source="openfda_fallback",
            limitations=limitations,
        )

    direct_snippets: list[EventEvidenceSnippet] = []
    related_snippets: list[EventEvidenceSnippet] = []
    use_dailymed = (
        label.extraction_source == "dailymed_spl_xml" and bool(label.spl_chunks)
    )
    if use_dailymed:
        searchable_chunks = [
            chunk
            for chunk in label.spl_chunks
            if chunk.chunk_type in SEARCHABLE_SPL_CHUNK_TYPES
            and section_key_for_code(chunk.section_code) in REVIEWED_DAILYMED_SECTIONS
        ]
        for chunk in searchable_chunks:
            direct_snippets.extend(_find_chunk_matches(chunk, plan.direct_terms))
            related_snippets.extend(_find_chunk_matches(chunk, plan.related_terms))
        has_reviewable_content = bool(searchable_chunks)
        extraction_source = "dailymed_spl_xml"
        limitations.append(
            "The event search used structured DailyMed SPL XML chunks selected by LOINC section code."
        )
    else:
        available_sections = {
            section: label.sections[section]
            for section in REVIEWED_SECTIONS
            if label.sections.get(section)
        }
        for section, text in available_sections.items():
            direct_snippets.extend(_find_term_matches(section, text, plan.direct_terms))
            related_snippets.extend(_find_term_matches(section, text, plan.related_terms))
        has_reviewable_content = bool(available_sections)
        extraction_source = "openfda_fallback"
        limitations.append(
            "DailyMed SPL XML was unavailable; the event search used current openFDA section text as fallback."
        )
        if label.dailymed_warning:
            limitations.append(label.dailymed_warning)

    evidence_snippets = direct_snippets + related_snippets
    section_order = REVIEWED_DAILYMED_SECTIONS if use_dailymed else REVIEWED_SECTIONS
    matched_sections = [
        section
        for section in section_order
        if any(snippet.section == section for snippet in evidence_snippets)
    ]
    low_confidence = label.label_match_confidence.casefold() == "low"
    if low_confidence or not has_reviewable_content:
        evidence_status = "insufficient_label_data"
        if low_confidence:
            limitations.append("The selected label match has low confidence.")
        if not has_reviewable_content:
            limitations.append("None of the specified safety sections were available in the selected label.")
        limitations.append(ABSENCE_LIMITATION)
    elif direct_snippets:
        evidence_status = "explicit_positive"
        limitations.append(
            "A label mention does not establish causality or a class-wide adverse-event relationship."
        )
    elif related_snippets:
        evidence_status = "related_but_not_explicit"
        limitations.append(
            "A related concept was found, but the requested event was not explicitly identified."
        )
        limitations.append(ABSENCE_LIMITATION)
    else:
        evidence_status = "not_found"
        limitations.append(ABSENCE_LIMITATION)

    return DrugEventEvidence(
        drug_name=usage.member.name,
        rxcui=usage.member.rxcui,
        cms_rank=usage.rank,
        selected_spl_set_id=label.selected_spl_set_id,
        label_effective_time=label.spl_effective_time or label.effective_time,
        event_query=plan.event_query,
        searched_terms=plan.searched_terms,
        evidence_status=evidence_status,
        matched_sections=matched_sections,
        evidence_snippets=evidence_snippets[:10],
        label_match_confidence=label.label_match_confidence,
        extraction_source=extraction_source,
        spl_version=label.spl_version,
        limitations=limitations,
    )


STRUCTURED_STATUS_PRIORITY = {
    "explicit_positive": 0,
    "interaction_dependent": 1,
    "related_but_not_explicit": 2,
    "negated": 3,
    "historical_or_preexisting": 4,
    "comparator_only": 5,
    "not_found": 6,
    "insufficient_label_data": 7,
}


def _structured_drug_status(items: list[EventEvidenceItem]) -> str:
    if not items:
        return "not_found"
    return min(
        (item.evidence_status for item in items),
        key=lambda status: STRUCTURED_STATUS_PRIORITY.get(status, 99),
    )


def _structured_snippet(
    item: EventEvidenceItem,
    *,
    extraction_source: str,
    spl_version: str | None,
) -> EventEvidenceSnippet | None:
    if not item.supporting_quote:
        return None
    return EventEvidenceSnippet(
        section=section_key_for_code(item.section_code) or item.section_title,
        matched_term=item.matched_term,
        text=item.supporting_quote,
        extraction_source=extraction_source,
        spl_version=spl_version,
        section_code=item.section_code,
        subsection_title=item.subsection_title,
        chunk_type=item.chunk_type or "structured_evidence_item",
        source_path=item.source_path or (item.source_paths[0] if item.source_paths else ""),
        chunk_hash=item.chunk_hash or (item.chunk_hashes[0] if item.chunk_hashes else ""),
    )


def build_structured_drug_event_evidence(
    *,
    selected_drugs: list[CmsUsage],
    labels: list[LabelEvidence],
    event_query: str,
    searched_terms: list[str],
    items: list[EventEvidenceItem],
) -> list[DrugEventEvidence]:
    labels_by_rxcui = {label.rxcui: label for label in labels}
    items_by_rxcui: dict[str, list[EventEvidenceItem]] = {}
    for item in items:
        items_by_rxcui.setdefault(item.rxcui, []).append(item)

    evidence: list[DrugEventEvidence] = []
    for usage in selected_drugs:
        label = labels_by_rxcui.get(usage.member.rxcui)
        drug_items = items_by_rxcui.get(usage.member.rxcui, [])
        evidence_status = _structured_drug_status(drug_items)
        extraction_source = label.extraction_source if label else "openfda_fallback"
        snippets = [
            snippet
            for item in drug_items
            if (
                snippet := _structured_snippet(
                    item,
                    extraction_source=extraction_source,
                    spl_version=label.spl_version if label else None,
                )
            )
            is not None
        ]
        matched_sections = list(
            dict.fromkeys(snippet.section for snippet in snippets if snippet.section)
        )
        limitations = [
            "Drug-level status is derived only from deterministic structured SPL evidence rules."
        ]
        if label is None:
            limitations.append("No selected openFDA label was available for this drug.")
        elif label.label_match_confidence.casefold() == "low":
            limitations.append("The selected label match has low confidence.")
        elif label.extraction_source != "dailymed_spl_xml" or not label.spl_chunks:
            limitations.append(
                "DailyMed SPL XML was unavailable; the same deterministic rules were applied to openFDA fallback sections."
            )
        if evidence_status == "explicit_positive":
            limitations.append(
                "A label mention does not establish causality or a class-wide adverse-event relationship."
            )
        elif evidence_status == "related_but_not_explicit":
            limitations.append(
                "A related concept was found, but the requested event was not explicitly identified."
            )
        if evidence_status in {
            "negated",
            "historical_or_preexisting",
            "comparator_only",
            "not_found",
            "insufficient_label_data",
        }:
            limitations.append(ABSENCE_LIMITATION)

        evidence.append(
            DrugEventEvidence(
                drug_name=usage.member.name,
                rxcui=usage.member.rxcui,
                cms_rank=usage.rank,
                selected_spl_set_id=label.selected_spl_set_id if label else None,
                label_effective_time=(
                    label.spl_effective_time or label.effective_time if label else None
                ),
                event_query=event_query,
                searched_terms=searched_terms,
                evidence_status=evidence_status,
                matched_sections=matched_sections,
                evidence_snippets=snippets[:10],
                label_match_confidence=(
                    label.label_match_confidence if label else "unavailable"
                ),
                extraction_source=extraction_source,
                spl_version=label.spl_version if label else None,
                limitations=list(dict.fromkeys(limitations)),
            )
        )
    return evidence


def build_class_level_summary(
    *,
    selected_class: str,
    class_member_count: int,
    plan: EventSearchPlan,
    evidence: list[DrugEventEvidence],
) -> ClassLevelSummary:
    distribution = {
        status: sum(item.evidence_status == status for item in evidence)
        for status in STRUCTURED_STATUS_PRIORITY
    }
    drugs_analyzed = len(evidence)
    explicit_count = distribution["explicit_positive"]
    related_count = distribution["related_but_not_explicit"]
    not_found_count = distribution["not_found"]
    insufficient_count = distribution["insufficient_label_data"]

    if drugs_analyzed == 0 or insufficient_count > drugs_analyzed / 2:
        assessment = "insufficient_class_evidence"
        interpretation = (
            "Most selected drugs lacked sufficient matched label data for this rule-based review."
        )
    elif explicit_count >= 2 and explicit_count / drugs_analyzed >= 2 / 3:
        assessment = "consistent_label_evidence"
        interpretation = (
            "The event was explicitly mentioned in at least two thirds of the analyzed drug labels, "
            "with at least two explicit positives."
        )
    elif explicit_count >= 1:
        assessment = "partial_or_mixed_label_evidence"
        interpretation = (
            "At least one selected drug label explicitly mentioned the event, but the pattern was "
            "not consistent across the analyzed drugs."
        )
    else:
        assessment = "no_explicit_mentions_found"
        interpretation = (
            "No explicit event mention was found in the reviewed sections of the analyzed labels. "
            "This does not establish absence of risk."
        )

    return ClassLevelSummary(
        selected_class=selected_class,
        adverse_event=plan.normalized_event,
        class_member_count=class_member_count,
        drugs_analyzed=drugs_analyzed,
        explicit_positive_count=explicit_count,
        related_count=related_count,
        not_found_count=not_found_count,
        insufficient_count=insufficient_count,
        evidence_distribution=distribution,
        class_assessment=assessment,
        interpretation=interpretation,
        limitations=[
            ABSENCE_LIMITATION,
            "Only one selected label per analyzed drug was reviewed.",
            (
                f"This synthesis covers {drugs_analyzed} CMS-ranked selected drug(s) "
                f"out of {class_member_count} RxClass member(s); it is not a full-class census."
            ),
            "This synthesis is label-text evidence only and does not establish causality or incidence.",
        ],
    )


def build_structured_analysis(
    *,
    selected_class: str,
    class_member_count: int,
    selected_drugs: list[CmsUsage],
    labels: list[LabelEvidence],
    plan: EventSearchPlan,
    items: list[EventEvidenceItem],
) -> tuple[list[DrugEventEvidence], ClassLevelSummary]:
    evidence = build_structured_drug_event_evidence(
        selected_drugs=selected_drugs,
        labels=labels,
        event_query=plan.event_query,
        searched_terms=plan.searched_terms,
        items=items,
    )
    summary = build_class_level_summary(
        selected_class=selected_class,
        class_member_count=class_member_count,
        plan=plan,
        evidence=evidence,
    )
    return evidence, summary


def analyze_event_for_selected_drugs(
    *,
    selected_class: str,
    class_member_count: int,
    selected_drugs: list[CmsUsage],
    labels: list[LabelEvidence],
    event_query: str,
) -> tuple[EventSearchPlan, list[DrugEventEvidence], ClassLevelSummary]:
    plan = build_event_search_plan(event_query)
    structured = extract_structured_event_evidence(
        selected_drugs=selected_drugs,
        labels=labels,
        normalized_event=plan.normalized_event,
        direct_terms=plan.direct_terms,
        related_terms=plan.related_terms,
    )
    evidence, summary = build_structured_analysis(
        selected_class=selected_class,
        class_member_count=class_member_count,
        selected_drugs=selected_drugs,
        labels=labels,
        plan=plan,
        items=structured.items,
    )
    return plan, evidence, summary
