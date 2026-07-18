from __future__ import annotations

import hashlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark.run_validation_benchmark as benchmark_runner

from benchmark.run_validation_benchmark import (
    FixedSPLVersionUnavailable,
    PASS_THRESHOLDS,
    _benchmark_completeness,
    _case_error_messages,
    _fixed_label,
    _prediction_fields,
    _prediction_status,
    _records_for_metrics,
    _select_item,
    _status_metrics,
    _supporting_quote_match,
    _validate_fixed_spl_identity,
    _validation_outcome,
)
from models import EventEvidenceItem


class BenchmarkPredictionSelectionIntegrityTests(unittest.TestCase):
    def item(
        self,
        *,
        status: str,
        source_path: str,
        assertion: str = "present",
        subject: str = "selected_drug",
        quote: str = "Bradycardia was reported.",
        drug_name: str = "metoprolol",
        normalized_event: str = "bradycardia",
        interaction_context: list[str] | None = None,
        source_text: str = "",
    ) -> EventEvidenceItem:
        return EventEvidenceItem(
            drug_name=drug_name,
            rxcui="6918",
            normalized_event=normalized_event,
            matched_term=normalized_event,
            evidence_status=status,
            assertion=assertion,
            subject=subject,
            section_code="43685-7",
            section_title="Warnings and Precautions",
            evidence_context=["warning"],
            interaction_context=interaction_context or [],
            source_text=source_text,
            supporting_quote=quote,
            chunk_type="paragraph",
            source_path=source_path,
            chunk_hash=("a" if source_path.endswith("[1]") else "b") * 64,
            source_paths=[source_path],
        )

    def test_empty_not_found_prediction_serializes_null_semantics(self) -> None:
        self.assertEqual(_prediction_status(None), "not_found")
        fields = _prediction_fields(None, evidence_item_count=0)
        self.assertIsNone(fields["predicted_assertion"])
        self.assertIsNone(fields["predicted_subject"])
        self.assertEqual(fields["predicted_context"], [])
        self.assertIsNone(fields["predicted_section"])
        self.assertIsNone(fields["predicted_subsection"])
        self.assertIsNone(fields["predicted_frequency_text"])
        self.assertIsNone(fields["predicted_frequency_value"])
        self.assertIsNone(fields["predicted_frequency_unit"])
        self.assertIsNone(fields["predicted_comparator_text"])
        self.assertEqual(fields["matching_evidence_item_count"], 0)

    def test_selector_api_cannot_receive_gold_standard_fields(self) -> None:
        self.assertEqual(
            list(inspect.signature(_select_item).parameters),
            ["event_query", "items"],
        )
        self.assertNotIn("expected_", inspect.getsource(_select_item))

    def test_selection_uses_production_status_priority_not_gold_answer(self) -> None:
        explicit = self.item(
            status="explicit_positive",
            source_path="section[2]",
        )
        negated = self.item(
            status="negated",
            assertion="absent",
            source_path="section[1]",
            quote="No cases of bradycardia were observed.",
        )
        selected = _select_item("bradycardia", [negated, explicit])
        self.assertIs(selected, explicit)

    def test_equal_predictions_use_stable_source_order(self) -> None:
        later = self.item(
            status="explicit_positive",
            source_path="section[2]",
        )
        earlier = self.item(
            status="explicit_positive",
            source_path="section[1]",
        )
        selected = _select_item("bradycardia", [later, earlier])
        self.assertIs(selected, earlier)

    def test_partial_johns_wort_quote_is_not_selected_as_primary(self) -> None:
        partial = self.item(
            status="explicit_positive",
            source_path="section[2]/subsection[1]/paragraph[0]",
            drug_name="trazodone",
            normalized_event="serotonin syndrome",
            quote="John's Wort) and other drugs. Serotonin syndrome can also occur.",
        )
        complete = self.item(
            status="explicit_positive",
            source_path="section[2]/subsection[1]/paragraph[0]",
            drug_name="trazodone",
            normalized_event="serotonin syndrome",
            quote=(
                "Serotonin-norepinephrine reuptake inhibitors (SNRIs) and SSRIs, "
                "including trazodone, can precipitate serotonin syndrome."
            ),
        )
        self.assertIs(_select_item("serotonin syndrome", [partial, complete]), complete)

    def test_tie_break_uses_earlier_source_position_not_alphabetic_quote(self) -> None:
        source_text = (
            "Zeta statement: metoprolol caused bradycardia. "
            "Alpha statement: metoprolol caused bradycardia."
        )
        later_alphabetic = self.item(
            status="explicit_positive",
            source_path="section[2]/paragraph[0]",
            quote="Alpha statement: metoprolol caused bradycardia.",
            source_text=source_text,
        )
        earlier_non_alphabetic = self.item(
            status="explicit_positive",
            source_path="section[2]/paragraph[0]",
            quote="Zeta statement: metoprolol caused bradycardia.",
            source_text=source_text,
        )
        selected = _select_item("bradycardia", [later_alphabetic, earlier_non_alphabetic])
        self.assertIs(selected, earlier_non_alphabetic)

    def test_ep01_like_tie_selects_direct_trazodone_statement(self) -> None:
        secondary = self.item(
            status="explicit_positive",
            source_path="section[2]/subsection[1]/paragraph[0]",
            drug_name="trazodone",
            normalized_event="serotonin syndrome",
            quote=(
                "The risk is increased with concomitant serotonergic drugs. "
                "Serotonin syndrome can also occur when these drugs are used alone."
            ),
        )
        direct = self.item(
            status="explicit_positive",
            source_path="section[2]/subsection[1]/paragraph[0]",
            drug_name="trazodone",
            normalized_event="serotonin syndrome",
            quote=(
                "Serotonin-norepinephrine reuptake inhibitors (SNRIs) and SSRIs, "
                "including trazodone, can precipitate serotonin syndrome, a potentially "
                "life-threatening condition."
            ),
        )
        self.assertIs(_select_item("serotonin syndrome", [secondary, direct]), direct)

    def test_ep03_like_tie_selects_direct_fluoxetine_statement(self) -> None:
        secondary = self.item(
            status="explicit_positive",
            source_path="section[2]/subsection[1]/paragraph[0]",
            drug_name="fluoxetine",
            normalized_event="serotonin syndrome",
            quote="Serotonin syndrome can also occur when these drugs are used alone.",
        )
        direct = self.item(
            status="explicit_positive",
            source_path="section[2]/subsection[1]/paragraph[0]",
            drug_name="fluoxetine",
            normalized_event="serotonin syndrome",
            quote=(
                "Selective serotonin reuptake inhibitors (SSRIs), including fluoxetine, "
                "can precipitate serotonin syndrome, a potentially life-threatening condition."
            ),
        )
        self.assertIs(_select_item("serotonin syndrome", [secondary, direct]), direct)

    def test_expected_quote_changes_cannot_change_prediction_selection(self) -> None:
        direct = self.item(
            status="explicit_positive",
            source_path="section[1]/paragraph[0]",
            quote="Metoprolol caused bradycardia.",
        )
        secondary = self.item(
            status="explicit_positive",
            source_path="section[2]/paragraph[0]",
            quote="Bradycardia was also discussed.",
        )
        expected_supporting_quotes = [
            "Metoprolol caused bradycardia.",
            "A deliberately different gold quote.",
        ]
        selections = [
            _select_item("bradycardia", [secondary, direct])
            for _expected_supporting_quote in expected_supporting_quotes
        ]
        self.assertEqual(selections, [direct, direct])

    def test_changing_any_expected_field_cannot_affect_prediction_selection(self) -> None:
        direct = self.item(
            status="explicit_positive",
            source_path="section[1]/paragraph[0]",
            quote="Metoprolol caused bradycardia.",
        )
        secondary = self.item(
            status="explicit_positive",
            source_path="section[2]/paragraph[0]",
            quote="Bradycardia was discussed.",
        )
        expected_fields = {
            "expected_status": "negated",
            "expected_assertion": "absent",
            "expected_subject": "comparator",
            "expected_context": ["contraindication"],
            "expected_supporting_quote": "A different quote.",
        }
        for field, value in expected_fields.items():
            with self.subTest(field=field):
                case_gold = {field: value}
                self.assertEqual(case_gold[field], value)
                self.assertIs(_select_item("bradycardia", [secondary, direct]), direct)

    def test_primary_selection_is_deterministic_across_repeated_runs(self) -> None:
        direct = self.item(
            status="explicit_positive",
            source_path="section[2]/paragraph[0]",
            quote="Metoprolol caused bradycardia.",
        )
        other = self.item(
            status="explicit_positive",
            source_path="section[2]/paragraph[0]",
            quote="Bradycardia was discussed.",
        )
        selections = [_select_item("bradycardia", [other, direct]) for _ in range(20)]
        self.assertTrue(all(selected is direct for selected in selections))

    def test_warning_evidence_precedes_other_sections_without_gold_input(self) -> None:
        contraindication = self.item(
            status="explicit_positive",
            source_path="section[1]",
        ).model_copy(
            update={"evidence_context": ["contraindication"]}
        )
        warning = self.item(
            status="explicit_positive",
            source_path="section[2]",
        ).model_copy(
            update={"evidence_context": ["warning"]}
        )
        selected = _select_item("bradycardia", [contraindication, warning])
        self.assertIs(selected, warning)

    def test_case_error_report_includes_non_status_mismatches(self) -> None:
        record = {
            "case_id": "EP-X",
            "expected_status": "explicit_positive",
            "predicted_status": "explicit_positive",
            "status_correct": True,
            "assertion_correct": True,
            "subject_correct": True,
            "context_correct": False,
            "section_correct": False,
            "frequency_correct": True,
            "comparator_correct": True,
        }
        case = {
            "case_id": "EP-X",
            "expected_section": "Warnings and Precautions",
            "score_frequency": False,
            "score_comparator": False,
        }
        self.assertEqual(
            _case_error_messages([record], [case]),
            ["EP-X: context mismatch; section mismatch"],
        )


class BenchmarkValidationGateIntegrityTests(unittest.TestCase):
    def complete_cases(self) -> list[dict[str, object]]:
        cases: list[dict[str, object]] = []
        for index in range(30):
            cases.append(
                {
                    "case_id": f"CASE-{index + 1:02d}",
                    "benchmark_group": "explicit_positive",
                    "expected_status": "explicit_positive",
                    "expected_assertion": "present",
                    "score_frequency": False,
                    "score_comparator": False,
                    "gold_review_complete": True,
                }
            )
        cases[1].update(
            benchmark_group="interaction_or_conditional",
            expected_status="interaction_dependent",
            expected_assertion="conditional",
        )
        cases[2].update(
            benchmark_group="negated_historical_or_comparator_only",
            expected_status="negated",
            expected_assertion="absent",
        )
        cases[3].update(
            benchmark_group="frequency_or_comparator",
            score_frequency=True,
        )
        cases[4].update(
            benchmark_group="not_found",
            expected_status="not_found",
            expected_assertion=None,
        )
        return cases

    def complete_scenarios(self) -> list[dict[str, object]]:
        return [
            {"scenario_id": f"SCENARIO-{index + 1}", "gold_review_complete": True}
            for index in range(6)
        ]

    def eligible_records(self, cases: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {"case_id": case["case_id"], "evaluation_eligible": True}
            for case in cases
            if case["gold_review_complete"] is True
        ]

    def class_results(self, scenarios: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {"scenario_id": scenario["scenario_id"], "evaluated": True, "correct": True}
            for scenario in scenarios
            if scenario["gold_review_complete"] is True
        ]

    def passing_metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = dict(PASS_THRESHOLDS)
        metrics["unsupported_positive_count"] = 0
        return metrics

    def passing_failures(self) -> dict[str, dict[str, object]]:
        return {"mock": {"passed": True, "outcome": "expected"}}

    def test_29_of_30_completed_cases_cannot_pass(self) -> None:
        cases = self.complete_cases()
        cases[-1]["gold_review_complete"] = False
        scenarios = self.complete_scenarios()
        completeness = _benchmark_completeness(
            cases,
            scenarios,
            self.eligible_records(cases),
            self.class_results(scenarios),
        )
        passed, status = _validation_outcome(
            self.passing_metrics(),
            self.passing_failures(),
            completeness,
            self.class_results(scenarios),
        )
        self.assertFalse(passed)
        self.assertEqual(status, "FAIL — validation incomplete")
        self.assertEqual(completeness["item_cases_needing_review"], 1)

    def test_all_cases_with_one_incomplete_class_scenario_cannot_pass(self) -> None:
        cases = self.complete_cases()
        scenarios = self.complete_scenarios()
        scenarios[-1]["gold_review_complete"] = False
        completeness = _benchmark_completeness(
            cases,
            scenarios,
            self.eligible_records(cases),
            self.class_results(scenarios),
        )
        passed, status = _validation_outcome(
            self.passing_metrics(),
            self.passing_failures(),
            completeness,
            self.class_results(scenarios),
        )
        self.assertFalse(passed)
        self.assertEqual(status, "FAIL — validation incomplete")
        self.assertEqual(completeness["class_scenarios_needing_review"], 1)

    def test_missing_interaction_or_frequency_coverage_prevents_pass(self) -> None:
        for missing_axis in ["interaction_or_conditional", "frequency_or_comparator"]:
            with self.subTest(missing_axis=missing_axis):
                cases = self.complete_cases()
                if missing_axis == "interaction_or_conditional":
                    cases[1].update(
                        benchmark_group="explicit_positive",
                        expected_status="explicit_positive",
                        expected_assertion="present",
                    )
                else:
                    cases[3]["score_frequency"] = False
                    cases[3]["score_comparator"] = False
                scenarios = self.complete_scenarios()
                completeness = _benchmark_completeness(
                    cases,
                    scenarios,
                    self.eligible_records(cases),
                    self.class_results(scenarios),
                )
                passed, _ = _validation_outcome(
                    self.passing_metrics(),
                    self.passing_failures(),
                    completeness,
                    self.class_results(scenarios),
                )
                self.assertFalse(passed)
                self.assertIn(missing_axis, completeness["missing_coverage_axes"])

    def test_different_existing_supporting_quote_fails_match(self) -> None:
        result = _supporting_quote_match(
            "Serotonin syndrome can occur.",
            "A supporting quote exists, but says something else.",
        )
        self.assertTrue(result["quote_scored"])
        self.assertFalse(result["quote_match"])
        self.assertEqual(
            result["quote_mismatch_reason"],
            "expected_quote_not_fully_contained",
        )

    def test_longer_prediction_containing_full_gold_quote_passes(self) -> None:
        result = _supporting_quote_match(
            "Serotonin syndrome can occur.",
            "Warning:   SEROTONIN syndrome can occur. Seek urgent care.",
        )
        self.assertTrue(result["quote_scored"])
        self.assertTrue(result["quote_match"])
        self.assertIsNone(result["quote_mismatch_reason"])

    def test_fixed_spl_version_mismatch_is_detected(self) -> None:
        result = _validate_fixed_spl_identity(
            {"spl_version": "4", "spl_effective_time": "20250101"},
            actual_version="5",
            actual_effective_time="20250101",
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "fixed_spl_version_unavailable")
        self.assertEqual(result["fixed_spl_version"], "4")
        self.assertEqual(result["actual_spl_version"], "5")

    def test_matching_fixed_effective_time_is_verified(self) -> None:
        result = _validate_fixed_spl_identity(
            {
                "fixed_spl_version": "2",
                "fixed_spl_effective_time": "20260630",
            },
            actual_version="2",
            actual_effective_time="20260630",
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["fixed_effective_time"], "20260630")
        self.assertEqual(result["actual_effective_time"], "20260630")
        self.assertTrue(result["effective_time_match"])
        self.assertIsNone(result["effective_time_failure_reason"])

    def test_effective_time_mismatch_prevents_case_evaluation(self) -> None:
        case = {
            "case_id": "EP-X",
            "spl_set_id": "fixed-set-id",
            "fixed_spl_version": "2",
            "fixed_spl_effective_time": "20260630",
        }
        parsed = SimpleNamespace(version="2", effective_time="20260701")
        with patch(
            "benchmark.run_validation_benchmark.fetch_dailymed_spl",
            return_value=(parsed, None),
        ):
            with self.assertRaises(FixedSPLVersionUnavailable) as raised:
                _fixed_label(case)
        self.assertEqual(raised.exception.code, "fixed_spl_effective_time_mismatch")
        self.assertFalse(raised.exception.identity["valid"])
        self.assertFalse(raised.exception.identity["effective_time_match"])

    def test_unavailable_actual_effective_time_prevents_case_evaluation(self) -> None:
        case = {
            "case_id": "EP-X",
            "spl_set_id": "fixed-set-id",
            "fixed_spl_version": "2",
            "fixed_spl_effective_time": "20260630",
        }
        parsed = SimpleNamespace(version="2", effective_time=None)
        with patch(
            "benchmark.run_validation_benchmark.fetch_dailymed_spl",
            return_value=(parsed, None),
        ):
            with self.assertRaises(FixedSPLVersionUnavailable) as raised:
                _fixed_label(case)
        self.assertEqual(raised.exception.code, "fixed_spl_effective_time_unavailable")
        self.assertFalse(raised.exception.identity["effective_time_match"])

    def test_missing_fixed_effective_time_is_not_autofilled(self) -> None:
        result = _validate_fixed_spl_identity(
            {
                "fixed_spl_version": "2",
                "spl_effective_time": "20260630",
                "effective_time": "20260630",
            },
            actual_version="2",
            actual_effective_time="20260701",
        )
        self.assertTrue(result["valid"])
        self.assertIsNone(result["fixed_effective_time"])
        self.assertEqual(result["actual_effective_time"], "20260701")
        self.assertIsNone(result["effective_time_match"])
        self.assertIsNone(result["effective_time_failure_reason"])

    def test_effective_time_formats_normalize_to_same_date(self) -> None:
        result = _validate_fixed_spl_identity(
            {
                "fixed_spl_version": "2",
                "fixed_spl_effective_time": "2026-06-30",
            },
            actual_version="2",
            actual_effective_time="20260630",
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["fixed_effective_time"], "20260630")
        self.assertEqual(result["actual_effective_time"], "20260630")
        self.assertTrue(result["effective_time_match"])

    def test_benchmark_cases_gold_file_sha256_is_unchanged(self) -> None:
        cases_path = Path(__file__).resolve().parent / "benchmark_cases.json"
        before = hashlib.sha256(cases_path.read_bytes()).hexdigest().upper()
        class_scenarios_path = cases_path.with_name("class_scenarios.json")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            arguments = [
                "run_validation_benchmark.py",
                "--cases",
                str(cases_path),
                "--class-scenarios",
                str(class_scenarios_path),
                "--results",
                str(temporary_path / "results.json"),
                "--report",
                str(temporary_path / "report.md"),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(benchmark_runner, "evaluate_cases", return_value=({}, [])),
                patch.object(benchmark_runner, "evaluate_class_scenarios", return_value=[]),
                patch.object(benchmark_runner, "run_mock_failure_checks", return_value={}),
                patch.object(benchmark_runner, "_case_error_messages", return_value=[]),
                patch.object(benchmark_runner, "_benchmark_completeness", return_value={}),
                patch.object(
                    benchmark_runner,
                    "_validation_outcome",
                    return_value=(False, "FAIL — validation incomplete"),
                ),
                patch.object(benchmark_runner, "render_report", return_value="# Test report\n"),
            ):
                self.assertEqual(benchmark_runner.main(), 0)
        after = hashlib.sha256(cases_path.read_bytes()).hexdigest().upper()
        self.assertEqual(after, before)

    def test_missing_fixed_spl_version_is_incomplete_and_not_autofilled(self) -> None:
        result = _validate_fixed_spl_identity(
            {"spl_set_id": "fixed-set-id"},
            actual_version="9",
            actual_effective_time="20250101",
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "missing_fixed_spl_version")
        self.assertIsNone(result["fixed_spl_version"])
        self.assertIsNone(result["actual_spl_version"])

    def test_prediction_selector_still_cannot_use_expected_fields(self) -> None:
        self.assertEqual(
            list(inspect.signature(_select_item).parameters),
            ["event_query", "items"],
        )
        self.assertNotIn("expected_", inspect.getsource(_select_item))

    def test_incomplete_cases_are_excluded_from_metrics_but_block_pass(self) -> None:
        records = [
            {
                "case_id": "COMPLETE",
                "evaluation_eligible": True,
                "expected_status": "explicit_positive",
                "predicted_status": "explicit_positive",
            },
            {
                "case_id": "INCOMPLETE",
                "evaluation_eligible": False,
                "expected_status": "explicit_positive",
                "predicted_status": "negated",
            },
        ]
        scored = _records_for_metrics(records)
        metrics, _ = _status_metrics(scored)
        self.assertEqual(len(scored), 1)
        self.assertEqual(metrics["precision"], 1.0)

        cases = self.complete_cases()
        scenarios = self.complete_scenarios()
        eligible = self.eligible_records(cases)[:-1]
        completeness = _benchmark_completeness(
            cases,
            scenarios,
            eligible,
            self.class_results(scenarios),
        )
        passed, status = _validation_outcome(
            self.passing_metrics(),
            self.passing_failures(),
            completeness,
            self.class_results(scenarios),
        )
        self.assertFalse(passed)
        self.assertEqual(status, "FAIL — validation incomplete")
        self.assertEqual(completeness["completed_item_cases_not_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
