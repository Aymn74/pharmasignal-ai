from __future__ import annotations

import unittest

from models import CmsUsage, DrugMember, LabelEvidence, SPLSectionChunk
from structured_evidence import (
    _frequency_tokens,
    _sentences,
    extract_items_from_chunk,
    extract_structured_event_evidence,
    merge_evidence_items,
)


class StructuredEvidenceSentenceSegmentationRegression(unittest.TestCase):
    def test_st_johns_wort_straight_apostrophe_is_not_split_after_st(self) -> None:
        sentences = _sentences(
            "Risk increases with St. John's Wort. Serotonin syndrome can occur."
        )
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "Risk increases with St. John's Wort.")

    def test_st_johns_wort_curly_apostrophe_is_not_split_after_st(self) -> None:
        sentences = _sentences(
            "Risk increases with St. John’s Wort. Serotonin syndrome can occur."
        )
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "Risk increases with St. John’s Wort.")


class StructuredEventEvidenceNegativeControls(unittest.TestCase):
    def setUp(self) -> None:
        self.usage = CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name="test drug", tty="IN"),
            cms_generic_names=[],
            total_claims=0,
            data_year=2024,
            match_quality="unit-test",
        )

    def extract(self, text: str, *, event: str = "bradycardia"):
        chunk = SPLSectionChunk(
            set_id="test-set-id",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[0]/paragraph[0]",
            chunk_hash="a" * 64,
        )
        items = extract_items_from_chunk(
            usage=self.usage,
            chunk=chunk,
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )
        self.assertEqual(len(items), 1)
        return items[0]

    def test_negated_phrase(self) -> None:
        item = self.extract("No cases of bradycardia were observed")
        self.assertEqual(item.evidence_status, "negated")
        self.assertEqual(item.assertion, "absent")

    def test_preexisting_phrase(self) -> None:
        item = self.extract("Patients with pre-existing bradycardia")
        self.assertEqual(item.evidence_status, "historical_or_preexisting")
        self.assertEqual(item.assertion, "historical")
        self.assertEqual(item.subject, "patient_history")

    def test_comparator_only_phrase(self) -> None:
        item = self.extract("Bradycardia occurred more frequently with placebo")
        self.assertEqual(item.evidence_status, "comparator_only")
        self.assertEqual(item.subject, "comparator")

    def test_interaction_dependent_phrase(self) -> None:
        item = self.extract(
            "Concomitant verapamil may increase the risk of bradycardia"
        )
        self.assertEqual(item.evidence_status, "interaction_dependent")
        self.assertEqual(item.assertion, "conditional")
        self.assertEqual(item.subject, "concomitant_drug")


class StructuredEventEvidenceConcomitantScopeRegression(unittest.TestCase):
    def extract(self, text: str, *, event: str, drug_name: str):
        usage = CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name=drug_name, tty="IN"),
            cms_generic_names=[drug_name],
            total_claims=0,
            data_year=2024,
            match_quality="regression-test",
        )
        chunk = SPLSectionChunk(
            set_id="test-set-id",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[0]/paragraph[0]",
            chunk_hash="c" * 64,
        )
        return extract_items_from_chunk(
            usage=usage,
            chunk=chunk,
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )

    def test_associated_with_concomitant_use_is_interaction_dependent(self) -> None:
        item = self.extract(
            "Serotonin syndrome associated with the concomitant use of "
            "linezolid and serotonergic agents has been reported.",
            event="serotonin syndrome",
            drug_name="linezolid",
        )[0]
        self.assertEqual(item.evidence_status, "interaction_dependent")
        self.assertEqual(item.assertion, "conditional")
        self.assertEqual(item.subject, "selected_drug")
        self.assertIn("warning", item.evidence_context)

    def test_reported_with_concomitant_use_is_interaction_dependent(self) -> None:
        item = self.extract(
            "Rhabdomyolysis has been reported with concomitant use of "
            "simvastatin and strong CYP3A4 inhibitors.",
            event="rhabdomyolysis",
            drug_name="simvastatin",
        )[0]
        self.assertEqual(item.evidence_status, "interaction_dependent")
        self.assertEqual(item.assertion, "conditional")

    def test_reported_during_coadministration_is_interaction_dependent(self) -> None:
        item = self.extract(
            "Bradycardia has been reported during coadministration with verapamil.",
            event="bradycardia",
            drug_name="metoprolol",
        )[0]
        self.assertEqual(item.evidence_status, "interaction_dependent")
        self.assertEqual(item.assertion, "conditional")

    def test_independent_drug_event_statement_remains_explicit_positive(self) -> None:
        item = self.extract(
            "Linezolid may cause serotonin syndrome.",
            event="serotonin syndrome",
            drug_name="linezolid",
        )[0]
        self.assertEqual(item.evidence_status, "explicit_positive")
        self.assertEqual(item.assertion, "present")

    def test_direct_clause_is_not_overridden_by_secondary_interaction(self) -> None:
        items = self.extract(
            "Linezolid may cause serotonin syndrome, and the risk is increased "
            "with serotonergic agents.",
            event="serotonin syndrome",
            drug_name="linezolid",
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].evidence_status, "explicit_positive")
        self.assertEqual(items[0].assertion, "present")
        self.assertTrue(items[0].interaction_context)

    def test_event_symptom_description_is_not_a_direct_drug_positive(self) -> None:
        item = self.extract(
            "Symptoms associated with serotonin syndrome may include agitation.",
            event="serotonin syndrome",
            drug_name="linezolid",
        )[0]
        self.assertEqual(item.evidence_status, "related_but_not_explicit")
        self.assertEqual(item.assertion, "uncertain")

    def test_conditional_management_instruction_is_not_a_direct_drug_positive(self) -> None:
        item = self.extract(
            "If signs or symptoms of serotonin syndrome occur, consider "
            "discontinuing treatment.",
            event="serotonin syndrome",
            drug_name="linezolid",
        )[0]
        self.assertEqual(item.evidence_status, "related_but_not_explicit")
        self.assertEqual(item.assertion, "uncertain")


class StructuredEventEvidenceBloodPressureRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.usage = CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name="test drug", tty="IN"),
            cms_generic_names=[],
            total_claims=0,
            data_year=2024,
            match_quality="regression-test",
        )

    def extract(self, text: str, *, event: str):
        chunk = SPLSectionChunk(
            set_id="test-set-id",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[0]/paragraph[0]",
            chunk_hash="b" * 64,
        )
        items = extract_items_from_chunk(
            usage=self.usage,
            chunk=chunk,
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )
        self.assertEqual(len(items), 1)
        return items[0]

    def assert_no_frequency(self, item) -> None:
        self.assertIsNone(item.frequency_text)
        self.assertIsNone(item.frequency_value)
        self.assertIsNone(item.frequency_unit)

    def test_a_baseline_hypotension_ranges_are_history_not_frequency(self) -> None:
        text = (
            "hypotension (resting systolic blood pressure of <90 mmHg); "
            "uncontrolled hypertension (>170/110 mmHg)"
        )
        item = self.extract(text, event="hypotension")
        self.assert_no_frequency(item)
        self.assertEqual(_frequency_tokens(text), [])
        self.assertEqual(item.evidence_status, "historical_or_preexisting")
        self.assertEqual(item.assertion, "historical")
        self.assertEqual(item.subject, "patient_history")
        self.assertNotEqual(item.evidence_status, "explicit_positive")

    def test_b_bp_ranges_in_patient_history_are_not_frequency(self) -> None:
        text = (
            "Patients with resting hypotension (BP <90/50 mmHg) or "
            "hypertension (BP >170/110 mmHg)"
        )
        item = self.extract(text, event="hypotension")
        self.assert_no_frequency(item)
        self.assertEqual(_frequency_tokens(text), [])
        self.assertEqual(item.evidence_status, "historical_or_preexisting")
        self.assertEqual(item.assertion, "historical")
        self.assertEqual(item.subject, "patient_history")

    def test_c_drug_caused_bp_decrease_is_preserved_without_frequency(self) -> None:
        text = (
            "Sildenafil has systemic vasodilatory properties that resulted in a mean "
            "maximum decrease of 8.4/5.5 mmHg in blood pressure."
        )
        item = self.extract(text, event="blood pressure")
        self.assert_no_frequency(item)
        self.assertEqual(_frequency_tokens(text), [])
        self.assertEqual(item.evidence_status, "explicit_positive")
        self.assertEqual(item.assertion, "present")
        self.assertEqual(item.subject, "selected_drug")
        self.assertIn("resulted in a mean maximum decrease", item.supporting_quote)

    def test_d_real_case_and_patient_year_frequencies_remain(self) -> None:
        item = self.extract(
            "Bradycardia was reported in 2 cases per 1000 patients.",
            event="bradycardia",
        )
        self.assertEqual(item.frequency_text, "2 cases per 1000 patients")
        self.assertEqual(item.frequency_value, 2.0)
        self.assertEqual(item.frequency_unit, "per 1000 patients")
        self.assertEqual(
            _frequency_tokens("Bradycardia occurred in 3 events per 100 patient-years."),
            ["3 events per 100 patient-years"],
        )

    def test_e_interaction_hypotension_remains_conditional(self) -> None:
        item = self.extract(
            "Concomitant use with alpha-blockers may lead to symptomatic hypotension.",
            event="hypotension",
        )
        self.assertEqual(item.evidence_status, "interaction_dependent")
        self.assertEqual(item.assertion, "conditional")

    def test_f_clinical_trial_frequency_and_comparator_remain(self) -> None:
        item = self.extract(
            "Hypotension 2% versus placebo 1%.",
            event="hypotension",
        )
        self.assertEqual(item.frequency_text, "2%")
        self.assertEqual(item.frequency_value, 2.0)
        self.assertEqual(item.frequency_unit, "percent")
        self.assertEqual(item.comparator_text, "Placebo: 1%")

    def test_tadalafil_list_item_inherits_patient_group_context(self) -> None:
        usage = CmsUsage(
            member=DrugMember(rxcui="358263", name="tadalafil", tty="IN"),
            cms_generic_names=["tadalafil"],
            total_claims=0,
            data_year=2024,
            match_quality="regression-test",
        )
        introduction = SPLSectionChunk(
            set_id="tadalafil-test-set",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="paragraph",
            chunk_index=0,
            text=(
                "Tadalafil is not recommended for the following groups of patients "
                "who were not included in clinical trials:"
            ),
            source_path="section[0]/paragraph[0]",
            chunk_hash="c" * 64,
        )
        list_item = SPLSectionChunk(
            set_id="tadalafil-test-set",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="list_item",
            chunk_index=1,
            text=(
                "uncontrolled arrhythmias, hypotension (less than 90/50 mm Hg), "
                "or uncontrolled hypertension"
            ),
            source_path="section[0]/list[0]/item[0]",
            chunk_hash="d" * 64,
        )
        label = LabelEvidence(
            rxcui="358263",
            requested_name="tadalafil",
            query_field="regression-test",
            matched_label_count=1,
            selected_spl_set_id="tadalafil-test-set",
            selection_reason="regression-test",
            label_match_score=100,
            label_match_confidence="high",
            extraction_source="dailymed_spl_xml",
            spl_chunks=[introduction, list_item],
        )
        result = extract_structured_event_evidence(
            selected_drugs=[usage],
            labels=[label],
            normalized_event="hypotension",
            direct_terms=["hypotension"],
            related_terms=[],
        )
        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assert_no_frequency(item)
        self.assertEqual(item.evidence_status, "historical_or_preexisting")
        self.assertEqual(item.assertion, "historical")
        self.assertEqual(item.subject, "patient_history")
        self.assertIn("not recommended for the following groups", item.supporting_quote)

    def test_tadalafil_drug_caused_supine_bp_decrease_remains_explicit(self) -> None:
        text = (
            "tadalafil 20 mg resulted in a mean maximal decrease in supine blood "
            "pressure of 1.6/0.8 mm Hg"
        )
        item = self.extract(text, event="blood pressure")
        self.assert_no_frequency(item)
        self.assertEqual(_frequency_tokens(text), [])
        self.assertEqual(item.evidence_status, "explicit_positive")
        self.assertEqual(item.assertion, "present")
        self.assertEqual(item.subject, "selected_drug")
        self.assertIn("resulted in a mean maximal decrease", item.supporting_quote)


class StructuredEventEvidenceTableFrequencyRegression(unittest.TestCase):
    def table_chunk(
        self,
        *,
        row_label: str,
        headers: list[str],
        cells: list[str],
        index: int = 0,
    ) -> SPLSectionChunk:
        text = " | ".join(
            f"{header}: {cell}" for header, cell in zip(headers, cells)
        )
        return SPLSectionChunk(
            set_id="table-test-set",
            section_code="34084-4",
            section_title="6 ADVERSE REACTIONS",
            loinc_display_name="Adverse reactions section",
            subsection_title="6.1 Clinical Trials Experience",
            chunk_type="table_row",
            chunk_index=index,
            text=text,
            table_id="table-test",
            row_index=index,
            column_headers=headers,
            row_cells=cells,
            source_path=f"section[0]/table[0]/row[{index}]",
            chunk_hash=f"{index + 1:064x}",
        )

    def extract(
        self,
        *,
        headers: list[str],
        cells: list[str],
        drug_name: str,
        event: str = "nausea",
    ):
        usage = CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name=drug_name, tty="IN"),
            cms_generic_names=[drug_name],
            total_claims=0,
            data_year=2024,
            match_quality="regression-test",
        )
        items = extract_items_from_chunk(
            usage=usage,
            chunk=self.table_chunk(
                row_label=cells[0],
                headers=headers,
                cells=cells,
            ),
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )
        self.assertEqual(len(items), 1)
        return items[0]

    def test_fc01_header_cell_pairs_supply_frequency_and_comparator(self) -> None:
        item = self.extract(
            headers=["Column 1", "Amlodipine besylate (%)", "Placebo (%)"],
            cells=["Nausea", "2.9", "1.9"],
            drug_name="amlodipine",
        )
        self.assertEqual(item.frequency_text, "2.9%")
        self.assertEqual(item.frequency_value, 2.9)
        self.assertEqual(item.frequency_unit, "percent")
        self.assertEqual(item.comparator_text, "Placebo 1.9%")
        self.assertEqual(item.source_path, "section[0]/table[0]/row[0]")

    def test_numeric_cell_under_percent_header_gains_percent_unit(self) -> None:
        item = self.extract(
            headers=["Event", "Drug (%)"],
            cells=["Nausea", "4.2"],
            drug_name="drug",
        )
        self.assertEqual(item.frequency_text, "4.2%")
        self.assertEqual(item.frequency_value, 4.2)
        self.assertEqual(item.frequency_unit, "percent")

    def test_table_without_comparator_extracts_frequency_only(self) -> None:
        item = self.extract(
            headers=["Event", "Metoprolol (%)"],
            cells=["Nausea", "3.1"],
            drug_name="metoprolol",
        )
        self.assertEqual(item.frequency_text, "3.1%")
        self.assertIsNone(item.comparator_text)

    def test_common_comparator_headers_are_recognized(self) -> None:
        for header, expected in [
            ("Control (%)", "Control 1.2%"),
            ("Comparator (%)", "Comparator 1.2%"),
            ("Active control (%)", "Active control 1.2%"),
        ]:
            with self.subTest(header=header):
                item = self.extract(
                    headers=["Event", "Metoprolol (%)", header],
                    cells=["Nausea", "3.1", "1.2"],
                    drug_name="metoprolol",
                )
                self.assertEqual(item.frequency_text, "3.1%")
                self.assertEqual(item.comparator_text, expected)

    def test_dose_in_header_is_not_mistaken_for_frequency(self) -> None:
        item = self.extract(
            headers=["Event", "Metoprolol 10 mg (%)"],
            cells=["Nausea", "2.5"],
            drug_name="metoprolol",
        )
        self.assertEqual(item.frequency_text, "2.5%")
        self.assertNotIn("10 mg", item.frequency_text)

    def test_sample_size_in_header_is_not_mistaken_for_frequency(self) -> None:
        item = self.extract(
            headers=["Event", "Metoprolol (N=200)"],
            cells=["Nausea", "6%"],
            drug_name="metoprolol",
        )
        self.assertEqual(item.frequency_text, "6%")
        self.assertNotIn("200", item.frequency_text)

    def test_only_the_matching_event_row_is_extracted(self) -> None:
        usage = CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name="amlodipine", tty="IN"),
            cms_generic_names=["amlodipine"],
            total_claims=0,
            data_year=2024,
            match_quality="regression-test",
        )
        headers = ["Event", "Amlodipine (%)", "Placebo (%)"]
        nausea = self.table_chunk(
            row_label="Nausea",
            headers=headers,
            cells=["Nausea", "2.9", "1.9"],
            index=0,
        )
        headache = self.table_chunk(
            row_label="Headache",
            headers=headers,
            cells=["Headache", "7.3", "7.8"],
            index=1,
        )
        items = []
        for chunk in [nausea, headache]:
            items.extend(
                extract_items_from_chunk(
                    usage=usage,
                    chunk=chunk,
                    normalized_event="nausea",
                    direct_terms=["nausea"],
                    related_terms=[],
                )
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].frequency_text, "2.9%")
        self.assertEqual(items[0].comparator_text, "Placebo 1.9%")

    def test_ambiguous_numeric_table_without_linked_headers_returns_null(self) -> None:
        item = self.extract(
            headers=["Column 1", "Column 2", "Column 3"],
            cells=["Nausea", "2.9", "1.9"],
            drug_name="amlodipine",
        )
        self.assertIsNone(item.frequency_text)
        self.assertIsNone(item.comparator_text)

class StructuredEventEvidenceFrequencyScopeRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.usage = CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name="simvastatin", tty="IN"),
            cms_generic_names=[],
            total_claims=0,
            data_year=2024,
            match_quality="regression-test",
        )

    def extract(self, text: str, *, event: str = "rhabdomyolysis"):
        chunk = SPLSectionChunk(
            set_id="test-set-id",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[0]/paragraph[0]",
            chunk_hash="e" * 64,
        )
        return extract_items_from_chunk(
            usage=self.usage,
            chunk=chunk,
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )

    def assert_no_frequency(self, item) -> None:
        self.assertIsNone(item.frequency_text)
        self.assertIsNone(item.frequency_value)
        self.assertIsNone(item.frequency_unit)

    def test_a_rare_fatalities_do_not_set_rhabdomyolysis_frequency(self) -> None:
        for statin in ("Simvastatin", "Atorvastatin", "Rosuvastatin"):
            with self.subTest(statin=statin):
                text = (
                    f"{statin} may cause myopathy and rhabdomyolysis. Acute kidney "
                    "injury secondary to myoglobinuria and rare fatalities have "
                    "occurred as a result of rhabdomyolysis."
                )
                items = self.extract(text)
                self.assertEqual(len(items), 2)
                for item in items:
                    self.assertEqual(item.evidence_status, "explicit_positive")
                    self.assert_no_frequency(item)

    def test_b_reported_rarely_remains_qualitative_frequency(self) -> None:
        item = self.extract(
            "Rhabdomyolysis has been reported rarely in patients receiving simvastatin."
        )[0]
        self.assertEqual(item.frequency_text, "rarely")
        self.assertIsNone(item.frequency_value)
        self.assertEqual(item.frequency_unit, "qualitative")

    def test_c_rare_cases_remain_qualitative_frequency(self) -> None:
        item = self.extract("Rare cases of rhabdomyolysis have been reported.")[0]
        self.assertEqual(item.evidence_status, "explicit_positive")
        self.assertEqual(item.frequency_text, "rare")
        self.assertIsNone(item.frequency_value)
        self.assertEqual(item.frequency_unit, "qualitative")

    def test_d_true_percentage_remains_frequency(self) -> None:
        item = self.extract("Rhabdomyolysis occurred in 0.1% of patients.")[0]
        self.assertEqual(item.frequency_text, "0.1%")
        self.assertEqual(item.frequency_value, 0.1)
        self.assertEqual(item.frequency_unit, "percent")

    def test_e_combination_dose_is_not_frequency(self) -> None:
        text = "Patients received ezetimibe/simvastatin 10/40 mg/day."
        self.assertEqual(_frequency_tokens(text), [])

    def test_f_other_slash_doses_are_not_frequency(self) -> None:
        dose_examples = [
            "amlodipine/benazepril 5/20 mg",
            "hydrocodone/acetaminophen 5/325 mg tablet",
            "valsartan/hydrochlorothiazide 160/12.5 mg daily",
        ]
        for text in dose_examples:
            with self.subTest(text=text):
                self.assertEqual(_frequency_tokens(text), [])

    def test_g_real_per_population_frequency_is_preserved(self) -> None:
        self.assertEqual(
            _frequency_tokens("2 cases per 1000 patients"),
            ["2 cases per 1000 patients"],
        )


class StructuredEventEvidenceNegationAndRatioRegression(unittest.TestCase):
    def extract(self, text: str, *, event: str, drug_name: str = "test drug"):
        usage = CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name=drug_name, tty="IN"),
            cms_generic_names=[drug_name],
            total_claims=0,
            data_year=2024,
            match_quality="regression-test",
        )
        chunk = SPLSectionChunk(
            set_id="test-set-id",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[0]/paragraph[0]",
            chunk_hash="9" * 64,
        )
        items = extract_items_from_chunk(
            usage=usage,
            chunk=chunk,
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )
        self.assertEqual(len(items), 1)
        return items[0]

    def assert_no_frequency(self, item) -> None:
        self.assertIsNone(item.frequency_text)
        self.assertIsNone(item.frequency_value)
        self.assertIsNone(item.frequency_unit)

    def test_a_negated_event_precedes_secondary_interaction(self) -> None:
        item = self.extract(
            "Serotonin syndrome was not observed; concomitant use with MAOIs "
            "increases the risk.",
            event="serotonin syndrome",
            drug_name="sertraline",
        )
        self.assertEqual(item.evidence_status, "negated")
        self.assertEqual(item.assertion, "absent")
        self.assertNotEqual(item.evidence_status, "explicit_positive")
        self.assertTrue(item.interaction_context)

    def test_b_not_reported_event_is_negated(self) -> None:
        item = self.extract("Bradycardia was not reported.", event="bradycardia")
        self.assertEqual(item.evidence_status, "negated")
        self.assertEqual(item.assertion, "absent")
        self.assertNotEqual(item.evidence_status, "explicit_positive")

    def test_c_negation_precedes_general_class_detection(self) -> None:
        item = self.extract(
            "Unlike statins, this treatment did not cause rhabdomyolysis.",
            event="rhabdomyolysis",
            drug_name="test treatment",
        )
        self.assertEqual(item.evidence_status, "negated")
        self.assertEqual(item.assertion, "absent")
        self.assertNotEqual(item.subject, "general_class_statement")
        self.assertNotEqual(item.evidence_status, "explicit_positive")

    def test_d_population_ratio_is_frequency_not_blood_pressure(self) -> None:
        text = "Hypotension occurred in 2/100 patients."
        item = self.extract(text, event="hypotension")
        self.assertEqual(item.frequency_text, "2/100 patients")
        self.assertEqual(item.frequency_value, 2.0)
        self.assertEqual(item.frequency_unit, "per 100 patients")
        self.assertEqual(_frequency_tokens(text), ["2/100 patients"])

    def test_e_population_ratio_survives_separate_dose_context(self) -> None:
        text = "Rhabdomyolysis occurred in 2/100 patients at a dose of 40 mg."
        item = self.extract(text, event="rhabdomyolysis")
        self.assertEqual(item.frequency_text, "2/100 patients")
        self.assertEqual(item.frequency_value, 2.0)
        self.assertEqual(item.frequency_unit, "per 100 patients")
        self.assertEqual(_frequency_tokens(text), ["2/100 patients"])

    def test_f_blood_pressure_ratio_remains_excluded(self) -> None:
        text = "Blood pressure was 90/50 mm Hg."
        item = self.extract(text, event="blood pressure")
        self.assert_no_frequency(item)
        self.assertEqual(_frequency_tokens(text), [])

    def test_g_combination_dose_ratio_remains_excluded(self) -> None:
        text = "Patients received ezetimibe/simvastatin 10/40 mg/day."
        self.assertEqual(_frequency_tokens(text), [])


class StructuredEventEvidenceNumericRoleRegression(unittest.TestCase):
    def extract(
        self,
        text: str,
        *,
        event: str,
        drug_name: str = "test drug",
        subsection_title: str | None = None,
    ):
        usage = CmsUsage(
            member=DrugMember(rxcui="numeric-role-rxcui", name=drug_name, tty="IN"),
            cms_generic_names=[drug_name],
            total_claims=0,
            data_year=2024,
            match_quality="numeric-role-regression-test",
        )
        chunk = SPLSectionChunk(
            set_id="numeric-role-set",
            section_code="43685-7",
            section_title="5 WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            subsection_title=subsection_title,
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[0]/paragraph[0]",
            chunk_hash="8" * 64,
        )
        items = extract_items_from_chunk(
            usage=usage,
            chunk=chunk,
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )
        self.assertEqual(len(items), 1)
        return items[0]

    def assert_no_frequency(self, text: str) -> None:
        self.assertEqual(_frequency_tokens(text), [], msg=text)

    def test_product_strength_percentages_are_not_event_frequency(self) -> None:
        examples = [
            "topical dapsone gel, 5% treatment",
            "dapsone gel 5%",
            "5% dapsone gel",
            "hydrocortisone cream 1%",
            "ointment 0.1%",
            "ophthalmic solution 0.3%",
            "0.9% sodium chloride",
            "5% dextrose",
            "10% lipid emulsion",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assert_no_frequency(text)

    def test_mass_volume_and_solid_strengths_are_not_event_frequency(self) -> None:
        examples = [
            "oral solution 5 mg/5 mL",
            "injection 10 mg/mL",
            "ointment 20 mg/g",
            "inhaler 90 mcg/actuation",
            "insulin 100 units/mL",
            "potassium chloride 2 mEq/mL",
            "5 mg tablet",
            "4 mg/0.1 mL injection",
            "40 mg per vial",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assert_no_frequency(text)

    def test_administered_doses_are_not_event_frequency(self) -> None:
        examples = [
            "Patients received 10 mg daily.",
            "The dose was 10 mg/kg/day.",
            "Infusion was administered at 2 mg/hour.",
            "Patients received one 10 mg tablet twice daily.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assert_no_frequency(text)

    def test_sample_sizes_are_not_event_frequency(self) -> None:
        examples = [
            "N=200",
            "Two hundred patients received treatment.",
            "Of 200 patients, 2 developed rash.",
            "A total of 600 participants received treatment.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assert_no_frequency(text)

    def test_laboratory_and_vital_values_are_not_event_frequency(self) -> None:
        examples = [
            "Blood pressure was 90/50 mmHg.",
            "QTc exceeded 500 ms.",
            "Oxygen saturation fell to 88%.",
            "Glucose was 70 mg/dL.",
            "Ejection fraction decreased to 35%.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assert_no_frequency(text)

    def test_true_event_frequency_expressions_are_preserved(self) -> None:
        expected = {
            "Nausea occurred in 5% of patients.": ["5%"],
            "5% of subjects reported nausea.": ["5%"],
            "The incidence of nausea was 3.2%.": ["3.2%"],
            "Two cases per 1,000 patient-years were observed.": [
                "Two cases per 1,000 patient-years"
            ],
            "Rash occurred in 4 of 100 patients.": ["4 of 100 patients"],
            "Nausea was reported by 12 patients.": ["12 patients"],
        }
        for text, tokens in expected.items():
            with self.subTest(text=text):
                self.assertEqual(_frequency_tokens(text), tokens)

    def test_mixed_product_strength_and_event_frequency_uses_event_only(self) -> None:
        examples = {
            "Patients received gel 5%; nausea occurred in 2% of patients.": ["2%"],
            "Hydrocortisone cream 1% was used in 200 subjects; burning occurred in 3%.": ["3%"],
            "The 100 units/mL formulation was administered to 450 patients; hypoglycemia occurred in 8%.": ["8%"],
            "The cream contains 0.1% active ingredient. Burning occurred in 5%.": ["5%"],
            "Patients received 5 mg/5 mL oral solution twice daily. Vomiting occurred in 6%.": ["6%"],
        }
        for text, tokens in examples.items():
            with self.subTest(text=text):
                self.assertEqual(_frequency_tokens(text), tokens)

    def test_mixed_dose_frequency_and_comparator_remain_separate(self) -> None:
        item = self.extract(
            "Patients received one 10 mg tablet daily; nausea occurred in 4.5% "
            "versus 2% with placebo.",
            event="nausea",
        )
        self.assertEqual(item.frequency_text, "4.5%")
        self.assertEqual(item.frequency_value, 4.5)
        self.assertEqual(item.frequency_unit, "percent")
        self.assertEqual(item.comparator_text, "Placebo 2%")

    def test_nc01_product_strength_does_not_populate_frequency(self) -> None:
        item = self.extract(
            "No events of peripheral neuropathy were observed in clinical trials "
            "with topical dapsone gel, 5% treatment.",
            event="peripheral neuropathy",
            drug_name="Dapsone gel 5%",
            subsection_title="5.3 Peripheral Neuropathy",
        )
        self.assertEqual(item.evidence_status, "negated")
        self.assertEqual(item.assertion, "absent")
        self.assertEqual(item.subject, "selected_drug")
        self.assertIsNone(item.frequency_text)
        self.assertIsNone(item.frequency_value)
        self.assertIsNone(item.frequency_unit)
        self.assertEqual(item.route_context, ["topical", "gel", "matching_route"])


class StructuredEventEvidenceRouteApplicabilityRegression(unittest.TestCase):
    event = "peripheral neuropathy"

    def extract(
        self,
        text: str,
        *,
        drug_name: str = "Dapsone gel 5%",
        rxcui: str = "3108",
        subsection_title: str | None = "5.3 Peripheral Neuropathy",
    ):
        usage = CmsUsage(
            member=DrugMember(rxcui=rxcui, name=drug_name, tty="IN"),
            cms_generic_names=[drug_name],
            total_claims=0,
            data_year=2024,
            match_quality="route-regression-test",
        )
        chunk = SPLSectionChunk(
            set_id="test-set-id",
            section_code="43685-7",
            section_title="5 WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            subsection_title=subsection_title,
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[1]/subsection[2]/paragraph[0]",
            chunk_hash="f" * 64,
        )
        return extract_items_from_chunk(
            usage=usage,
            chunk=chunk,
            normalized_event=self.event,
            direct_terms=[self.event],
            related_terms=[],
        )

    def test_oral_positive_and_topical_negative_selects_topical_negation(self) -> None:
        from benchmark.run_validation_benchmark import _select_item

        items = self.extract(
            "Peripheral neuropathy has been reported with oral treatment. "
            "No events of peripheral neuropathy were observed in clinical trials "
            "with topical gel treatment."
        )
        self.assertEqual(len(items), 2)
        oral = next(item for item in items if "route_mismatch" in item.route_context)
        topical = next(item for item in items if "matching_route" in item.route_context)
        self.assertEqual(oral.assertion, "present")
        self.assertEqual(oral.subject, "unclear")
        self.assertEqual(oral.evidence_status, "insufficient_label_data")
        self.assertEqual(oral.route_context, ["oral", "route_mismatch"])
        self.assertEqual(topical.evidence_status, "negated")
        self.assertEqual(topical.assertion, "absent")
        self.assertEqual(topical.subject, "selected_drug")
        self.assertEqual(
            topical.supporting_quote,
            "No events of peripheral neuropathy were observed in clinical trials "
            "with topical gel treatment.",
        )
        self.assertIs(_select_item(self.event, items), topical)

    def test_oral_negative_and_topical_positive_selects_topical_positive(self) -> None:
        from benchmark.run_validation_benchmark import _select_item

        items = self.extract(
            "No events of peripheral neuropathy were observed with oral treatment. "
            "Peripheral neuropathy has been reported with topical gel treatment."
        )
        topical = next(item for item in items if "matching_route" in item.route_context)
        oral = next(item for item in items if "route_mismatch" in item.route_context)
        self.assertEqual(oral.assertion, "absent")
        self.assertEqual(oral.evidence_status, "insufficient_label_data")
        self.assertEqual(topical.evidence_status, "explicit_positive")
        self.assertEqual(topical.assertion, "present")
        self.assertIs(_select_item(self.event, items), topical)

    def test_two_event_sentences_create_separate_evidence_items(self) -> None:
        items = self.extract(
            "Peripheral neuropathy has been reported with oral treatment. "
            "Peripheral neuropathy was not observed with topical gel treatment."
        )
        self.assertEqual(len(items), 2)
        self.assertNotEqual(items[0].supporting_quote, items[1].supporting_quote)
        self.assertTrue(
            all(
                item.supporting_quote.casefold().count("peripheral neuropathy") == 1
                for item in items
            )
        )

    def test_oral_subsection_supplies_route_when_sentence_has_none(self) -> None:
        item = self.extract(
            "Peripheral neuropathy has been reported.",
            subsection_title="6.2 Experience with Oral Use of Dapsone",
        )[0]
        self.assertEqual(item.route_context, ["oral", "route_mismatch"])
        self.assertEqual(item.evidence_status, "insufficient_label_data")

    def test_local_topical_route_overrides_oral_subsection(self) -> None:
        item = self.extract(
            "Peripheral neuropathy was not observed with topical gel treatment.",
            subsection_title="6.2 Experience with Oral Use of Dapsone",
        )[0]
        self.assertEqual(item.route_context, ["topical", "gel", "matching_route"])
        self.assertEqual(item.evidence_status, "negated")

    def test_selected_gel_does_not_treat_oral_evidence_as_matching(self) -> None:
        item = self.extract(
            "Peripheral neuropathy has been reported after oral administration."
        )[0]
        self.assertIn("route_mismatch", item.route_context)
        self.assertNotEqual(item.subject, "selected_drug")

    def test_same_ingredient_rxcui_does_not_override_route_mismatch(self) -> None:
        text = "Peripheral neuropathy has been reported with oral treatment."
        topical_item = self.extract(text, drug_name="Dapsone gel 5%", rxcui="3108")[0]
        oral_item = self.extract(text, drug_name="Dapsone tablet", rxcui="3108")[0]
        self.assertIn("route_mismatch", topical_item.route_context)
        self.assertIn("matching_route", oral_item.route_context)
        self.assertEqual(oral_item.evidence_status, "explicit_positive")

    def test_missing_selected_route_keeps_conflicting_evidence_conservative(self) -> None:
        items = self.extract(
            "Peripheral neuropathy has been reported with oral treatment. "
            "Peripheral neuropathy was not observed with topical treatment.",
            drug_name="Dapsone",
        )
        self.assertEqual(len(items), 2)
        self.assertTrue(all("ambiguous_route" in item.route_context for item in items))
        self.assertTrue(all(item.evidence_status == "related_but_not_explicit" for item in items))
        self.assertTrue(all(item.assertion == "uncertain" for item in items))

    def test_primary_quote_is_only_the_matching_sentence(self) -> None:
        items = self.extract(
            "Peripheral neuropathy has been reported with oral dapsone treatment. "
            "No events of peripheral neuropathy were observed in clinical trials "
            "with topical dapsone gel, 5% treatment."
        )
        topical = next(item for item in items if "matching_route" in item.route_context)
        self.assertEqual(
            topical.supporting_quote,
            "No events of peripheral neuropathy were observed in clinical trials "
            "with topical dapsone gel, 5% treatment.",
        )
        self.assertNotIn("oral dapsone", topical.supporting_quote.casefold())

    def test_duplicate_evidence_with_same_traceability_is_merged(self) -> None:
        item = self.extract(
            "Peripheral neuropathy was not observed with topical gel treatment."
        )[0]
        merged = merge_evidence_items([item, item.model_copy(deep=True)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_path, item.source_path)
        self.assertEqual(merged[0].chunk_hash, item.chunk_hash)


if __name__ == "__main__":
    unittest.main()
