from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.failure_harness import run_mock_failure_checks
from event_analysis import (
    STRUCTURED_STATUS_PRIORITY,
    analyze_event_for_selected_drugs,
    build_event_search_plan,
)
from models import CmsUsage, DrugMember, EventEvidenceItem, LabelEvidence
from spl_parser import fetch_dailymed_spl
from structured_evidence import (
    EXTRACTION_RULES_VERSION,
    extract_structured_event_evidence,
)


BENCHMARK_VERSION = "pharmasignal-v0.2-validation-benchmark-v1"
STATUS_LABELS = [
    "explicit_positive",
    "related_but_not_explicit",
    "negated",
    "historical_or_preexisting",
    "comparator_only",
    "interaction_dependent",
    "not_found",
    "insufficient_label_data",
]
PASS_THRESHOLDS = {
    "explicit_positive_precision": 0.90,
    "explicit_positive_recall": 0.85,
    "status_macro_f1": 0.85,
    "negation_accuracy": 0.90,
    "context_accuracy": 0.90,
    "frequency_exact_match_accuracy": 0.90,
    "comparator_exact_match_accuracy": 0.90,
    "supporting_quote_accuracy": 1.00,
    "traceability_completeness": 1.00,
    "deterministic_output_rate": 1.00,
}
EXPECTED_ITEM_CASES = 30
EXPECTED_CLASS_SCENARIOS = 6
REQUIRED_COVERAGE_AXES = [
    "explicit_positive",
    "interaction_or_conditional",
    "negation_historical_or_comparator_only",
    "frequency_or_comparator",
    "not_found",
]


class FixedSPLVersionUnavailable(RuntimeError):
    def __init__(self, code: str, message: str, identity: dict[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.identity = identity


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _usage(drug_name: str, rxcui: str) -> CmsUsage:
    return CmsUsage(
        member=DrugMember(rxcui=rxcui, name=drug_name, tty="IN"),
        cms_generic_names=[drug_name],
        total_claims=0,
        data_year=2024,
        match_quality="fixed-validation-benchmark",
    )


def _normalize_effective_time(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value).strip())
    return digits[:8] if len(digits) >= 8 else None


def _fixed_spl_metadata(case: dict[str, Any]) -> tuple[str | None, str | None]:
    version = case.get("fixed_spl_version")
    if version is None:
        version = case.get("spl_version")
    effective_time = case.get("fixed_spl_effective_time")
    normalized_version = str(version).strip() if version is not None else ""
    return normalized_version or None, _normalize_effective_time(effective_time)


def _validate_fixed_spl_identity(
    case: dict[str, Any],
    *,
    actual_version: Any = None,
    actual_effective_time: Any = None,
    verify_actual: bool = False,
) -> dict[str, Any]:
    fixed_version, fixed_effective_time = _fixed_spl_metadata(case)
    actual_version_text = str(actual_version).strip() if actual_version is not None else ""
    actual_effective_text = _normalize_effective_time(actual_effective_time)
    verify_actual = verify_actual or actual_version is not None or actual_effective_time is not None
    effective_time_match: bool | None = None
    effective_time_failure_reason: str | None = None
    if verify_actual and fixed_effective_time is not None:
        if actual_effective_text is None:
            effective_time_match = False
            effective_time_failure_reason = "fixed_spl_effective_time_unavailable"
        elif actual_effective_text != fixed_effective_time:
            effective_time_match = False
            effective_time_failure_reason = "fixed_spl_effective_time_mismatch"
        else:
            effective_time_match = True

    def identity(valid: bool, status: str) -> dict[str, Any]:
        return {
            "valid": valid,
            "status": status,
            "fixed_spl_version": fixed_version,
            "fixed_effective_time": fixed_effective_time,
            "actual_spl_version": actual_version_text or None,
            "actual_effective_time": actual_effective_text,
            "effective_time_match": effective_time_match,
            "effective_time_failure_reason": effective_time_failure_reason,
        }

    if fixed_version is None:
        missing_version = identity(False, "missing_fixed_spl_version")
        missing_version.update(
            {
                "actual_spl_version": None,
                "actual_effective_time": None,
                "effective_time_match": None,
                "effective_time_failure_reason": None,
            }
        )
        return missing_version
    if verify_actual and actual_version_text != fixed_version:
        return identity(False, "fixed_spl_version_unavailable")
    if effective_time_failure_reason is not None:
        return identity(False, effective_time_failure_reason)
    return identity(
        True,
        "fixed_spl_version_verified" if verify_actual else "fixed_spl_version_declared",
    )


def _fixed_label(case: dict[str, Any]) -> LabelEvidence:
    declared_identity = _validate_fixed_spl_identity(case)
    if not declared_identity["valid"]:
        raise FixedSPLVersionUnavailable(
            declared_identity["status"],
            f"{case.get('case_id', case.get('drug_name', 'case'))}: fixed SPL version is missing.",
            declared_identity,
        )
    try:
        parsed, _ = fetch_dailymed_spl(case["spl_set_id"], timeout_seconds=60)
    except Exception as exc:
        raise FixedSPLVersionUnavailable(
            "fixed_spl_version_unavailable",
            f"{case.get('case_id', case.get('drug_name', 'case'))}: fixed SPL could not be fetched.",
            declared_identity,
        ) from exc
    verified_identity = _validate_fixed_spl_identity(
        case,
        actual_version=parsed.version,
        actual_effective_time=parsed.effective_time,
        verify_actual=True,
    )
    if not verified_identity["valid"]:
        raise FixedSPLVersionUnavailable(
            verified_identity["status"],
            f"{case.get('case_id', case.get('drug_name', 'case'))}: requested fixed SPL version is unavailable.",
            verified_identity,
        )
    return LabelEvidence(
        rxcui=case["rxcui"],
        requested_name=case["drug_name"],
        query_field="benchmark.fixed_spl_set_id",
        matched_label_count=1,
        selected_spl_set_id=case["spl_set_id"],
        effective_time=parsed.effective_time,
        selection_reason="Fixed manually reviewed benchmark SPL SET ID.",
        label_match_score=100,
        label_match_confidence="high",
        extraction_source="dailymed_spl_xml",
        spl_version=parsed.version,
        spl_effective_time=parsed.effective_time,
        spl_chunks=parsed.chunks,
        spl_diagnostics=parsed.diagnostics,
        dailymed_xml_url=parsed.xml_url,
    )


def _extract_case(case: dict[str, Any], label: LabelEvidence) -> list[EventEvidenceItem]:
    plan = build_event_search_plan(case["event_query"])
    result = extract_structured_event_evidence(
        selected_drugs=[_usage(case["drug_name"], case["rxcui"])],
        labels=[label],
        normalized_event=plan.normalized_event,
        direct_terms=plan.direct_terms,
        related_terms=plan.related_terms,
    )
    return result.items


def _without_timestamps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timestamps(item)
            for key, item in sorted(value.items())
            if key not in {"created_at", "retrieved_at", "timestamp"}
        }
    if isinstance(value, list):
        return [_without_timestamps(item) for item in value]
    return value


def _canonical_items(items: list[EventEvidenceItem]) -> str:
    payload = [_without_timestamps(item.model_dump(mode="json")) for item in items]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _select_item(
    event_query: str,
    items: list[EventEvidenceItem],
) -> EventEvidenceItem | None:
    if not items:
        return None
    plan = build_event_search_plan(event_query)
    direct_terms = {term.casefold() for term in plan.direct_terms}
    confidence_priority = {"high": 0, "medium": 1, "low": 2}
    subject_priority = {
        "selected_drug": 0,
        "general_class_statement": 1,
        "concomitant_drug": 2,
        "comparator": 3,
        "patient_history": 4,
        "unclear": 5,
    }
    context_priority = {
        "boxed_warning": 0,
        "warning": 1,
        "adverse_reaction": 2,
        "contraindication": 3,
        "drug_interaction": 4,
        "general_safety_statement": 5,
    }

    def source_path(item: EventEvidenceItem) -> str:
        return item.source_path or (item.source_paths[0] if item.source_paths else "")

    def quote_completeness_priority(item: EventEvidenceItem) -> int:
        quote = item.supporting_quote.strip()
        if not quote:
            return 1
        if quote[0].islower() or quote[0] in ")]},;:":
            return 1
        if quote.count(")") > quote.count("(") or quote.count("]") > quote.count("["):
            return 1
        return 0

    def direct_statement_priority(item: EventEvidenceItem) -> int:
        quote = item.supporting_quote
        drug_pattern = re.compile(
            rf"(?<!\w){re.escape(item.drug_name).replace(r'\ ', r'\s+')}(?!\w)",
            re.IGNORECASE,
        )
        direct_patterns = [
            re.compile(
                rf"(?<!\w){re.escape(term).replace(r'\ ', r'\s+')}(?!\w)",
                re.IGNORECASE,
            )
            for term in plan.direct_terms
        ]
        clauses = re.split(r"(?<=[.!?;])\s+", quote)
        if any(
            drug_pattern.search(clause)
            and any(pattern.search(clause) for pattern in direct_patterns)
            for clause in clauses
        ):
            return 0
        if (
            item.evidence_status == "interaction_dependent"
            or item.assertion == "conditional"
            or bool(item.interaction_context)
        ):
            return 2
        return 1

    def spl_source_order(item: EventEvidenceItem) -> tuple[int, ...]:
        indices = tuple(int(value) for value in re.findall(r"\[(\d+)\]", source_path(item)))
        return indices or (sys.maxsize,)

    def quote_source_position(item: EventEvidenceItem) -> int:
        normalized_source = " ".join(item.source_text.casefold().split())
        normalized_quote = " ".join(item.supporting_quote.casefold().split())
        if not normalized_source or not normalized_quote:
            return sys.maxsize
        position = normalized_source.find(normalized_quote)
        return position if position >= 0 else sys.maxsize

    def chunk_index(item: EventEvidenceItem) -> int:
        explicit_index = getattr(item, "chunk_index", None)
        if isinstance(explicit_index, int):
            return explicit_index
        indices = re.findall(r"\[(\d+)\]", source_path(item))
        return int(indices[-1]) if indices else sys.maxsize

    def selection_key(indexed_item: tuple[int, EventEvidenceItem]) -> tuple[object, ...]:
        original_index, item = indexed_item
        traceability_fields = [
            bool(item.supporting_quote.strip()),
            bool(item.source_path or item.source_paths),
            bool(item.chunk_hash or item.chunk_hashes),
            any(term.casefold() in item.supporting_quote.casefold() for term in plan.searched_terms),
            _classification_supported(item),
        ]
        return (
            STRUCTURED_STATUS_PRIORITY.get(item.evidence_status, 99),
            0 if item.matched_term.casefold() in direct_terms else 1,
            confidence_priority.get(item.extraction_confidence, 99),
            subject_priority.get(item.subject, 99),
            min(
                (context_priority.get(context, 99) for context in item.evidence_context),
                default=99,
            ),
            -sum(traceability_fields),
            quote_completeness_priority(item),
            direct_statement_priority(item),
            spl_source_order(item),
            quote_source_position(item),
            chunk_index(item),
            source_path(item),
            original_index,
        )

    return min(enumerate(items), key=selection_key)[1]


def _prediction_fields(
    selected: EventEvidenceItem | None,
    *,
    evidence_item_count: int,
) -> dict[str, Any]:
    return {
        "predicted_assertion": selected.assertion if selected else None,
        "predicted_subject": selected.subject if selected else None,
        "predicted_context": selected.evidence_context if selected else [],
        "predicted_section": (selected.section_title or None) if selected else None,
        "predicted_subsection": selected.subsection_title if selected else None,
        "predicted_frequency_text": selected.frequency_text if selected else None,
        "predicted_frequency_value": selected.frequency_value if selected else None,
        "predicted_frequency_unit": selected.frequency_unit if selected else None,
        "predicted_comparator_text": selected.comparator_text if selected else None,
        "matching_evidence_item_count": evidence_item_count,
    }


def _prediction_status(selected: EventEvidenceItem | None) -> str:
    return selected.evidence_status if selected else "not_found"


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _exact(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        return set(left or []) == set(right or [])
    return (left or None) == (right or None)


def _normalize_supporting_quote(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _supporting_quote_match(expected: Any, predicted: Any) -> dict[str, Any]:
    expected_text = str(expected or "")
    predicted_text = str(predicted or "")
    normalized_expected = _normalize_supporting_quote(expected_text)
    normalized_predicted = _normalize_supporting_quote(predicted_text)
    if not normalized_expected:
        return {
            "quote_scored": False,
            "quote_match": None,
            "expected_supporting_quote": expected if expected is not None else None,
            "predicted_supporting_quote": predicted if predicted is not None else None,
            "quote_mismatch_reason": "not_scored_no_expected_supporting_quote",
        }
    if not normalized_predicted:
        return {
            "quote_scored": True,
            "quote_match": False,
            "expected_supporting_quote": expected_text,
            "predicted_supporting_quote": predicted if predicted is not None else None,
            "quote_mismatch_reason": "predicted_supporting_quote_missing",
        }
    matched = normalized_expected in normalized_predicted
    return {
        "quote_scored": True,
        "quote_match": matched,
        "expected_supporting_quote": expected_text,
        "predicted_supporting_quote": predicted_text,
        "quote_mismatch_reason": None if matched else "expected_quote_not_fully_contained",
    }


def _records_for_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("evaluation_eligible") is True]


def _coverage_axes(
    cases: list[dict[str, Any]],
    eligible_case_ids: set[str],
) -> dict[str, int]:
    coverage = {axis: 0 for axis in REQUIRED_COVERAGE_AXES}
    for case in cases:
        if case.get("case_id") not in eligible_case_ids:
            continue
        status = case.get("expected_status")
        assertion = case.get("expected_assertion")
        group = case.get("benchmark_group")
        if status == "explicit_positive":
            coverage["explicit_positive"] += 1
        if status == "interaction_dependent" or assertion == "conditional":
            coverage["interaction_or_conditional"] += 1
        if status in {"negated", "historical_or_preexisting", "comparator_only"}:
            coverage["negation_historical_or_comparator_only"] += 1
        if case.get("score_frequency") is True or case.get("score_comparator") is True:
            coverage["frequency_or_comparator"] += 1
        if status == "not_found" or group == "not_found":
            coverage["not_found"] += 1
    return coverage


def _benchmark_completeness(
    cases: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    records: list[dict[str, Any]],
    class_results: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_case_ids = {
        case["case_id"] for case in cases if case.get("gold_review_complete") is True
    }
    eligible_case_ids = {
        record["case_id"] for record in _records_for_metrics(records)
    } & completed_case_ids
    completed_scenario_ids = {
        scenario["scenario_id"]
        for scenario in scenarios
        if scenario.get("gold_review_complete") is True
    }
    evaluated_scenario_ids = {
        result["scenario_id"]
        for result in class_results
        if result.get("evaluated") is True
    }
    coverage = _coverage_axes(cases, eligible_case_ids)
    missing_axes = [axis for axis, count in coverage.items() if count == 0]
    missing_case_slots = max(EXPECTED_ITEM_CASES - len(cases), 0)
    missing_scenario_slots = max(EXPECTED_CLASS_SCENARIOS - len(scenarios), 0)
    cases_needing_review = len(cases) - len(completed_case_ids)
    scenarios_needing_review = len(scenarios) - len(completed_scenario_ids)
    completed_cases_not_evaluated = len(completed_case_ids - eligible_case_ids)
    completed_scenarios_not_evaluated = len(
        completed_scenario_ids - evaluated_scenario_ids
    )
    complete = all(
        [
            len(cases) == EXPECTED_ITEM_CASES,
            len(completed_case_ids) == EXPECTED_ITEM_CASES,
            len(eligible_case_ids) == EXPECTED_ITEM_CASES,
            len(scenarios) == EXPECTED_CLASS_SCENARIOS,
            len(completed_scenario_ids) == EXPECTED_CLASS_SCENARIOS,
            len(evaluated_scenario_ids) == EXPECTED_CLASS_SCENARIOS,
            not missing_axes,
        ]
    )
    return {
        "complete": complete,
        "expected_item_cases": EXPECTED_ITEM_CASES,
        "item_case_slots": len(cases),
        "missing_item_case_slots": missing_case_slots,
        "completed_item_cases": len(completed_case_ids),
        "item_cases_needing_review": cases_needing_review,
        "evaluated_item_cases": len(eligible_case_ids),
        "completed_item_cases_not_evaluated": completed_cases_not_evaluated,
        "expected_class_scenarios": EXPECTED_CLASS_SCENARIOS,
        "class_scenario_slots": len(scenarios),
        "missing_class_scenario_slots": missing_scenario_slots,
        "completed_class_scenarios": len(completed_scenario_ids),
        "class_scenarios_needing_review": scenarios_needing_review,
        "evaluated_class_scenarios": len(evaluated_scenario_ids),
        "completed_class_scenarios_not_evaluated": completed_scenarios_not_evaluated,
        "coverage": coverage,
        "missing_coverage_axes": missing_axes,
    }


def _normalized_section(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value.casefold()).split()
    )


def _section_matches(item: EventEvidenceItem, expected_section: Any) -> bool:
    if not expected_section:
        return False
    expected = _normalized_section(str(expected_section))
    actual = _normalized_section(
        " ".join(
            [
                item.section_title,
                item.subsection_title or "",
                *item.mentioned_in_sections,
            ]
        )
    )
    return bool(expected and expected in actual)


def _classification_supported(item: EventEvidenceItem) -> bool:
    text = item.supporting_quote.casefold()
    if item.evidence_status == "negated":
        return bool("no cases" in text or "not observed" in text or "none observed" in text)
    if item.evidence_status == "historical_or_preexisting":
        return any(term in text for term in ["pre-existing", "preexisting", "history of"])
    if item.evidence_status == "comparator_only":
        return any(term in text for term in ["placebo", "comparator", "control"])
    if item.evidence_status == "interaction_dependent":
        return bool(item.interaction_context) or "drug_interaction" in item.evidence_context
    if item.evidence_status == "explicit_positive":
        return item.assertion == "present" and bool(item.matched_term)
    if item.evidence_status == "related_but_not_explicit":
        return item.assertion == "uncertain" and bool(item.matched_term)
    return True


def _traceability_checks(
    item: EventEvidenceItem,
    *,
    set_id: str,
    searched_terms: list[str],
) -> dict[str, bool]:
    quote = item.supporting_quote.casefold()
    return {
        "supporting_quote": bool(item.supporting_quote.strip()),
        "source_path": bool(item.source_path or item.source_paths),
        "chunk_hash": bool(item.chunk_hash or item.chunk_hashes),
        "set_id": bool(set_id.strip()),
        "event_or_synonym": any(term.casefold() in quote for term in searched_terms),
        "classification_supported": _classification_supported(item),
    }


def _status_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    matrix = {expected: {predicted: 0 for predicted in STATUS_LABELS} for expected in STATUS_LABELS}
    for record in records:
        expected = record["expected_status"]
        predicted = record["predicted_status"]
        if expected in matrix and predicted in matrix[expected]:
            matrix[expected][predicted] += 1

    per_status: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    for status in STATUS_LABELS:
        tp = matrix[status][status]
        fp = sum(matrix[other][status] for other in STATUS_LABELS if other != status)
        fn = sum(matrix[status][other] for other in STATUS_LABELS if other != status)
        support = sum(matrix[status].values())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        if support and f1 is not None:
            f1_values.append(f1)
        per_status[status] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(record["expected_status"] == record["predicted_status"] for record in records)
    micro = _safe_divide(correct, len(records))
    return {
        "precision": micro,
        "recall": micro,
        "f1": micro,
        "status_macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_status": per_status,
    }, matrix


def _accuracy(records: list[dict[str, Any]], predicate, field: str) -> float | None:
    selected = [record for record in records if predicate(record)]
    return _safe_divide(sum(record[field] for record in selected), len(selected))


def evaluate_cases(cases: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completed = [case for case in cases if case.get("gold_review_complete") is True]
    records: list[dict[str, Any]] = []
    deterministic_passes = 0
    trace_checks: list[bool] = []
    quote_checks: list[bool] = []
    path_checks: list[bool] = []
    unsupported_positive_count = 0

    for case in completed:
        try:
            label = _fixed_label(case)
        except FixedSPLVersionUnavailable as exc:
            quote_result = _supporting_quote_match(
                case.get("expected_supporting_quote"),
                None,
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "evaluation_eligible": False,
                    "evaluation_incomplete_reason": exc.code,
                    "spl_version_validation": exc.identity,
                    "expected_status": case.get("expected_status"),
                    "predicted_status": None,
                    "status_correct": False,
                    "assertion_correct": False,
                    "subject_correct": False,
                    "context_correct": False,
                    "frequency_correct": False,
                    "comparator_correct": False,
                    "section_correct": False,
                    "deterministic": None,
                    "traceability": None,
                    "predicted_item": None,
                    **quote_result,
                }
            )
            continue
        runs = [_extract_case(case, label) for _ in range(3)]
        deterministic = len({_canonical_items(items) for items in runs}) == 1
        deterministic_passes += int(deterministic)
        items = runs[0]
        selected = _select_item(case["event_query"], items)
        predicted_status = _prediction_status(selected)
        plan = build_event_search_plan(case["event_query"])
        checks = (
            _traceability_checks(
                selected,
                set_id=case["spl_set_id"],
                searched_terms=plan.searched_terms,
            )
            if selected and predicted_status not in {"not_found", "insufficient_label_data"}
            else None
        )
        if checks:
            trace_checks.append(all(checks.values()))
            quote_checks.append(checks["supporting_quote"])
            path_checks.append(checks["source_path"])
        for item in items:
            if item.evidence_status != "explicit_positive":
                continue
            item_checks = _traceability_checks(
                item,
                set_id=case["spl_set_id"],
                searched_terms=plan.searched_terms,
            )
            unsupported_positive_count += int(not all(item_checks.values()))

        quote_result = _supporting_quote_match(
            case.get("expected_supporting_quote"),
            selected.supporting_quote if selected else None,
        )

        record = {
            "case_id": case["case_id"],
            "evaluation_eligible": True,
            "evaluation_incomplete_reason": None,
            "spl_version_validation": _validate_fixed_spl_identity(
                case,
                actual_version=label.spl_version,
                actual_effective_time=label.spl_effective_time,
            ),
            "expected_status": case["expected_status"],
            "predicted_status": predicted_status,
            "status_correct": predicted_status == case["expected_status"],
            **_prediction_fields(selected, evidence_item_count=len(items)),
            "assertion_correct": _exact(
                selected.assertion if selected else None,
                case["expected_assertion"],
            ),
            "subject_correct": _exact(
                selected.subject if selected else None,
                case["expected_subject"],
            ),
            "context_correct": _exact(
                selected.evidence_context if selected else [],
                case["expected_context"],
            ),
            "frequency_correct": _exact(
                selected.frequency_text if selected else None,
                case["expected_frequency_text"],
            ),
            "comparator_correct": _exact(
                selected.comparator_text if selected else None,
                case["expected_comparator_text"],
            ),
            "section_correct": bool(
                _section_matches(selected, case.get("expected_section"))
                if selected
                else case.get("expected_section") is None
            ),
            "deterministic": deterministic,
            "traceability": checks,
            "predicted_item": selected.model_dump(mode="json") if selected else None,
            **quote_result,
        }
        records.append(record)

    scored_records = _records_for_metrics(records)
    completed_by_id = {case["case_id"]: case for case in completed}
    status_metrics, confusion_matrix = _status_metrics(scored_records)
    explicit = status_metrics["per_status"]["explicit_positive"]
    metrics = {
        **status_metrics,
        "explicit_positive_precision": explicit["precision"],
        "explicit_positive_recall": explicit["recall"],
        "negation_accuracy": _accuracy(
            scored_records,
            lambda record: record["expected_status"] == "negated",
            "status_correct",
        ),
        "interaction_dependent_accuracy": _accuracy(
            scored_records,
            lambda record: record["expected_status"] == "interaction_dependent",
            "status_correct",
        ),
        "context_accuracy": _accuracy(scored_records, lambda record: True, "context_correct"),
        "frequency_exact_match_accuracy": _accuracy(
            scored_records,
            lambda record: completed_by_id[record["case_id"]].get("score_frequency", False),
            "frequency_correct",
        ),
        "comparator_exact_match_accuracy": _accuracy(
            scored_records,
            lambda record: completed_by_id[record["case_id"]].get("score_comparator", False),
            "comparator_correct",
        ),
        "section_accuracy": _accuracy(
            scored_records,
            lambda record: bool(completed_by_id[record["case_id"]].get("expected_section")),
            "section_correct",
        ),
        "supporting_quote_accuracy": _accuracy(
            scored_records,
            lambda record: record["quote_scored"],
            "quote_match",
        ),
        "source_path_completeness": _safe_divide(sum(path_checks), len(path_checks)),
        "supporting_quote_completeness": _safe_divide(sum(quote_checks), len(quote_checks)),
        "traceability_completeness": _safe_divide(sum(trace_checks), len(trace_checks)),
        "unsupported_positive_count": unsupported_positive_count,
        "deterministic_output_rate": _safe_divide(deterministic_passes, len(scored_records)),
        "manually_completed_case_count": len(completed),
        "evaluated_case_count": len(scored_records),
        "version_incomplete_case_count": len(completed) - len(scored_records),
        "confusion_matrix": confusion_matrix,
    }
    return metrics, records


def evaluate_class_scenarios(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.get("gold_review_complete") is not True:
            continue
        usages: list[CmsUsage] = []
        labels: list[LabelEvidence] = []
        version_error: FixedSPLVersionUnavailable | None = None
        for drug in scenario.get("drugs") or []:
            case = dict(drug)
            case["case_id"] = f"{scenario['scenario_id']}:{drug.get('drug_name', 'drug')}"
            usages.append(_usage(drug["drug_name"], drug["rxcui"]))
            try:
                labels.append(_fixed_label(case))
            except FixedSPLVersionUnavailable as exc:
                version_error = exc
                break
        if version_error is not None:
            results.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "evaluated": False,
                    "evaluation_incomplete_reason": version_error.code,
                    "spl_version_validation": version_error.identity,
                    "expected_class_assessment": scenario.get("expected_class_assessment"),
                    "predicted_class_assessment": None,
                    "correct": False,
                }
            )
            continue
        _, _, summary = analyze_event_for_selected_drugs(
            selected_class=scenario["class_name"],
            class_member_count=scenario["class_member_count"],
            selected_drugs=usages,
            labels=labels,
            event_query=scenario["event_query"],
        )
        results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "evaluated": True,
                "evaluation_incomplete_reason": None,
                "expected_class_assessment": scenario["expected_class_assessment"],
                "predicted_class_assessment": summary.class_assessment,
                "correct": summary.class_assessment == scenario["expected_class_assessment"],
            }
        )
    return results


def _metric_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _scientific_criteria_passed(
    metrics: dict[str, Any],
    failures: dict[str, dict[str, Any]],
    class_results: list[dict[str, Any]],
) -> bool:
    if any(metrics.get(name) is None for name in PASS_THRESHOLDS):
        return False
    if any(metrics[name] < threshold for name, threshold in PASS_THRESHOLDS.items()):
        return False
    if metrics.get("unsupported_positive_count") != 0:
        return False
    if not all(result["passed"] for result in failures.values()):
        return False
    return bool(class_results) and all(
        result.get("evaluated") is True and result.get("correct") is True
        for result in class_results
    )


def _validation_outcome(
    metrics: dict[str, Any],
    failures: dict[str, dict[str, Any]],
    completeness: dict[str, Any],
    class_results: list[dict[str, Any]],
) -> tuple[bool, str]:
    if not completeness["complete"]:
        return False, "FAIL — validation incomplete"
    if not _scientific_criteria_passed(metrics, failures, class_results):
        return False, "FAIL — performance criteria not met"
    return True, "PASS"


def _case_error_messages(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[str]:
    cases_by_id = {case["case_id"]: case for case in cases}
    messages: list[str] = []
    for record in records:
        case = cases_by_id[record["case_id"]]
        if record.get("evaluation_eligible") is False:
            messages.append(
                f"{record['case_id']}: validation incomplete — "
                f"{record.get('evaluation_incomplete_reason', 'not evaluated')}"
            )
            continue
        mismatches: list[str] = []
        if not record["status_correct"]:
            mismatches.append(
                f"status expected {record['expected_status']}, predicted {record['predicted_status']}"
            )
        for field in ["assertion", "subject", "context"]:
            if not record[f"{field}_correct"]:
                mismatches.append(f"{field} mismatch")
        if case.get("expected_section") and not record["section_correct"]:
            mismatches.append("section mismatch")
        if case.get("score_frequency") and not record["frequency_correct"]:
            mismatches.append("frequency mismatch")
        if case.get("score_comparator") and not record["comparator_correct"]:
            mismatches.append("comparator mismatch")
        if record.get("quote_scored") and not record.get("quote_match"):
            mismatches.append(
                "supporting quote mismatch "
                f"({record.get('quote_mismatch_reason', 'unspecified')})"
            )
        if mismatches:
            messages.append(f"{record['case_id']}: " + "; ".join(mismatches))
    return messages


def render_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    matrix = metrics["confusion_matrix"]
    matrix_header = "| Expected \\ Predicted | " + " | ".join(STATUS_LABELS) + " |"
    matrix_separator = "|---|" + "---:|" * len(STATUS_LABELS)
    matrix_rows = [
        "| " + expected + " | " + " | ".join(str(matrix[expected][predicted]) for predicted in STATUS_LABELS) + " |"
        for expected in STATUS_LABELS
    ]
    metric_names = [
        "precision",
        "recall",
        "f1",
        "status_macro_f1",
        "explicit_positive_precision",
        "explicit_positive_recall",
        "negation_accuracy",
        "interaction_dependent_accuracy",
        "context_accuracy",
        "frequency_exact_match_accuracy",
        "comparator_exact_match_accuracy",
        "section_accuracy",
        "source_path_completeness",
        "supporting_quote_completeness",
        "supporting_quote_accuracy",
        "traceability_completeness",
        "deterministic_output_rate",
    ]
    metric_rows = "\n".join(f"| {name} | {_metric_text(metrics.get(name))} |" for name in metric_names)
    failure_rows = "\n".join(
        f"| {name} | {'PASS' if result['passed'] else 'FAIL'} | {result['outcome']} |"
        for name, result in payload["failure_tests"].items()
    )
    threshold_rows = "\n".join(
        f"| {name} | {threshold:.2f} | {_metric_text(metrics.get(name))} | "
        f"{'PASS' if metrics.get(name) is not None and metrics[name] >= threshold else 'FAIL'} |"
        for name, threshold in PASS_THRESHOLDS.items()
    )
    errors = payload["case_errors"] or [
        (
            "No scored discrepancies were found in the manually completed cases."
            if payload["completed_cases"]
            else "No manually completed cases were available for case-level error analysis."
        )
    ]
    error_lines = "\n".join(f"- {error}" for error in errors)
    identity_rows = "\n".join(
        "| "
        + " | ".join(
            [
                record["case_id"],
                _metric_text(record.get("spl_version_validation", {}).get("fixed_effective_time")),
                _metric_text(record.get("spl_version_validation", {}).get("actual_effective_time")),
                _metric_text(record.get("spl_version_validation", {}).get("effective_time_match")),
                _metric_text(
                    record.get("spl_version_validation", {}).get(
                        "effective_time_failure_reason"
                    )
                ),
            ]
        )
        + " |"
        for record in payload["records"]
    ) or "| None | N/A | N/A | N/A | N/A |"
    completeness = payload["validation_completeness"]
    missing_axes = ", ".join(completeness["missing_coverage_axes"]) or "None"
    outcome = payload["validation_status"]
    return f"""# PharmaSignal AI v0.2 Validation Benchmark

## Code version

- Application version: PharmaSignal AI v0.2
- Extraction rules: `{EXTRACTION_RULES_VERSION}`
- Benchmark version: `{BENCHMARK_VERSION}`
- Git version: uncommitted local workspace; no stable commit identifier is available

## Evaluation date

{payload['evaluation_date']}

## Test-set description

- Item-level slots: {payload['total_cases']} total; {payload['completed_cases']} manually completed; {payload['cases_needing_review']} awaiting manual review.
- Item-level evaluations eligible for metrics: {completeness['evaluated_item_cases']}; completed but not evaluable because fixed SPL identity is incomplete or unavailable: {completeness['completed_item_cases_not_evaluated']}.
- Distribution: 8 explicit-positive, 5 interaction/conditional, 5 frequency/comparator, 6 negated/historical/comparator-only, and 6 not-found slots.
- End-to-end class scenarios: 6 total; {payload['completed_class_scenarios']} manually completed; {payload['class_scenarios_needing_review']} awaiting review.
- Class scenarios evaluated: {completeness['evaluated_class_scenarios']}; completed but not evaluable: {completeness['completed_class_scenarios_not_evaluated']}.
- Missing measurable coverage axes: {missing_axes}.
- Gold answers are never generated by the runner. Cases require both `gold_review_complete=true` and a verified fixed SPL identity before entering metric calculations.
- A completed case enters metrics only after its fixed SPL version (and fixed effective time when supplied) matches the fetched SPL. Missing versions are never inferred from predictions.
- Every eligible item case runs three times against its fixed SPL SET ID and fixed SPL version.

## Fixed SPL effective-time verification

| Case | Fixed effective time | Actual effective time | Match | Failure reason |
|---|---:|---:|---|---|
{identity_rows}

## Overall results

| Metric | Result |
|---|---:|
{metric_rows}
| unsupported_positive_count | {metrics['unsupported_positive_count']} |

Metrics are `N/A` when their manually reviewed denominator is zero. A zero unsupported-positive count over zero evaluated cases is not sufficient for PASS.

## Validation completeness gate

- Required item cases: {completeness['expected_item_cases']}; manually complete: {completeness['completed_item_cases']}; missing review: {completeness['item_cases_needing_review']}.
- Required class scenarios: {completeness['expected_class_scenarios']}; manually complete: {completeness['completed_class_scenarios']}; missing review: {completeness['class_scenarios_needing_review']}.
- Completed item cases lacking verified fixed SPL identity: {completeness['completed_item_cases_not_evaluated']}.
- Completed class scenarios not evaluated: {completeness['completed_class_scenarios_not_evaluated']}.
- Missing measurable coverage axes: {missing_axes}.
- Completeness gate: {'PASS' if completeness['complete'] else 'FAIL'}.

## Confusion matrix

{matrix_header}
{matrix_separator}
{chr(10).join(matrix_rows)}

## Case-level errors

{error_lines}

## Failure-mode tests

| Mocked failure | Result | Explicit outcome |
|---|---|---|
{failure_rows}

## Success criteria

| Criterion | Threshold | Result | Status |
|---|---:|---:|---|
{threshold_rows}
| unsupported_positive_count | 0 | {metrics['unsupported_positive_count']} | {'PASS' if metrics['unsupported_positive_count'] == 0 and completeness['evaluated_item_cases'] else 'FAIL'} |
| all 30 item cases complete and version-verified | 30/30 | {completeness['evaluated_item_cases']}/30 | {'PASS' if completeness['evaluated_item_cases'] == 30 and completeness['completed_item_cases'] == 30 else 'FAIL'} |
| all 6 class scenarios complete and evaluated | 6/6 | {completeness['evaluated_class_scenarios']}/6 | {'PASS' if completeness['evaluated_class_scenarios'] == 6 and completeness['completed_class_scenarios'] == 6 else 'FAIL'} |
| all required evidence axes measurable | 5/5 | {5 - len(completeness['missing_coverage_axes'])}/5 | {'PASS' if not completeness['missing_coverage_axes'] else 'FAIL'} |
| all failure-mode tests | 4/4 | {sum(result['passed'] for result in payload['failure_tests'].values())}/4 | {'PASS' if all(result['passed'] for result in payload['failure_tests'].values()) else 'FAIL'} |

## Final result

**{outcome}**

The current report is expected to remain FAIL until the fixed gold-standard fields are completed and approved manually. No extraction rule was changed or tuned during this benchmark run.

## Limitations

- No item-level performance estimate is valid before manual gold-standard completion.
- Empty templates prove benchmark structure and runner behavior, not clinical extraction quality.
- Fixed SPL SET IDs and versions improve reproducibility; cases without an explicit fixed version remain validation-incomplete.
- The four failure tests are mocked resilience tests and do not measure real service availability.
- Class-level scenarios are reported separately and do not contribute to item-level status metrics.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PharmaSignal AI v0.2 validation benchmark.")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "benchmark" / "benchmark_cases.json")
    parser.add_argument("--class-scenarios", type=Path, default=PROJECT_ROOT / "benchmark" / "class_scenarios.json")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "benchmark" / "benchmark_results.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "VALIDATION_REPORT.md")
    args = parser.parse_args()

    cases = _read_json(args.cases)["cases"]
    scenarios = _read_json(args.class_scenarios)["scenarios"]
    metrics, records = evaluate_cases(cases)
    class_results = evaluate_class_scenarios(scenarios)
    failures = run_mock_failure_checks()
    completed_cases = sum(case.get("gold_review_complete") is True for case in cases)
    completed_scenarios = sum(scenario.get("gold_review_complete") is True for scenario in scenarios)
    case_errors = _case_error_messages(records, cases)
    completeness = _benchmark_completeness(cases, scenarios, records, class_results)
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "extraction_rules_version": EXTRACTION_RULES_VERSION,
        "evaluation_date": date.today().isoformat(),
        "total_cases": len(cases),
        "completed_cases": completed_cases,
        "cases_needing_review": len(cases) - completed_cases,
        "total_class_scenarios": len(scenarios),
        "completed_class_scenarios": completed_scenarios,
        "class_scenarios_needing_review": len(scenarios) - completed_scenarios,
        "metrics": metrics,
        "records": records,
        "class_scenario_results": class_results,
        "failure_tests": failures,
        "case_errors": case_errors,
        "validation_completeness": completeness,
    }
    payload["passed"], payload["validation_status"] = _validation_outcome(
        metrics,
        failures,
        completeness,
        class_results,
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.results, payload)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({
        "completed_cases": completed_cases,
        "cases_needing_review": len(cases) - completed_cases,
        "failure_tests": failures,
        "passed": payload["passed"],
        "validation_status": payload["validation_status"],
        "validation_completeness": completeness,
        "report": str(args.report.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
