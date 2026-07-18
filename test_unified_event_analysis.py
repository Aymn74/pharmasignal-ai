from __future__ import annotations

import unittest

from event_analysis import analyze_event_for_selected_drugs, build_event_search_plan
from models import CmsUsage, DrugMember, LabelEvidence, SPLSectionChunk
from structured_evidence import extract_items_from_chunk, extract_structured_event_evidence


class UnifiedEventAnalysisRegression(unittest.TestCase):
    def usage(self, drug_name: str = "sertraline") -> CmsUsage:
        return CmsUsage(
            member=DrugMember(rxcui="test-rxcui", name=drug_name, tty="IN"),
            cms_generic_names=[drug_name],
            total_claims=100,
            data_year=2024,
            match_quality="regression-test",
            rank=1,
        )

    def chunk(self, text: str, *, chunk_hash: str = "f" * 64) -> SPLSectionChunk:
        return SPLSectionChunk(
            set_id="test-set-id",
            version="1",
            effective_time="20260717",
            section_code="43685-7",
            section_title="WARNINGS AND PRECAUTIONS",
            loinc_display_name="Warnings and precautions",
            chunk_type="paragraph",
            chunk_index=0,
            text=text,
            source_path="section[0]/paragraph[0]",
            chunk_hash=chunk_hash,
        )

    def label(
        self,
        text: str,
        *,
        confidence: str = "high",
    ) -> LabelEvidence:
        return LabelEvidence(
            rxcui="test-rxcui",
            requested_name="test drug",
            query_field="regression-test",
            matched_label_count=1,
            selected_spl_set_id="test-set-id",
            effective_time="20260717",
            selection_reason="regression-test",
            label_match_score=100,
            label_match_confidence=confidence,
            extraction_source="dailymed_spl_xml",
            spl_version="1",
            spl_effective_time="20260717",
            spl_chunks=[self.chunk(text)],
        )

    def extract_one(self, text: str, *, drug_name: str, event: str):
        items = extract_items_from_chunk(
            usage=self.usage(drug_name),
            chunk=self.chunk(text),
            normalized_event=event,
            direct_terms=[event],
            related_terms=[],
        )
        self.assertEqual(len(items), 1)
        return items[0]

    def test_event_dictionary_expands_without_merging_torsades_into_qt(self) -> None:
        serotonin = build_event_search_plan("serotonin toxicity")
        self.assertEqual(serotonin.normalized_event, "serotonin syndrome")
        self.assertIn("serotonin syndrome", serotonin.direct_terms)
        self.assertIn("serotonin toxicity", serotonin.related_terms)

        hypotension = build_event_search_plan("low blood pressure")
        self.assertEqual(hypotension.normalized_event, "hypotension")
        self.assertIn("hypotensive episode", hypotension.related_terms)

        torsades = build_event_search_plan("torsades de pointes")
        self.assertEqual(torsades.normalized_event, "torsades de pointes")
        self.assertIn("torsades de pointes", torsades.direct_terms)

    def test_general_class_statement_is_not_beta_blocker_specific(self) -> None:
        examples = [
            ("Beta-blockers can cause bradycardia.", "metoprolol", "bradycardia"),
            (
                "SSRIs can precipitate serotonin syndrome.",
                "sertraline",
                "serotonin syndrome",
            ),
            ("Statins can cause rhabdomyolysis.", "simvastatin", "rhabdomyolysis"),
        ]
        for text, drug_name, event in examples:
            with self.subTest(text=text):
                item = self.extract_one(text, drug_name=drug_name, event=event)
                self.assertEqual(item.evidence_status, "explicit_positive")
                self.assertEqual(item.subject, "general_class_statement")

    def test_plural_noun_alone_is_not_a_drug_class_statement(self) -> None:
        item = self.extract_one(
            "Patients can experience bradycardia.",
            drug_name="metoprolol",
            event="bradycardia",
        )
        self.assertEqual(item.subject, "selected_drug")

    def test_secondary_interaction_does_not_override_primary_drug_effect(self) -> None:
        item = self.extract_one(
            "Sertraline can precipitate serotonin syndrome; concomitant use with "
            "MAOIs increases the risk.",
            drug_name="sertraline",
            event="serotonin syndrome",
        )
        self.assertEqual(item.evidence_status, "explicit_positive")
        self.assertEqual(item.assertion, "present")
        self.assertEqual(item.subject, "selected_drug")
        self.assertIn("concomitant use", [value.casefold() for value in item.interaction_context])

    def test_interaction_that_governs_event_remains_conditional(self) -> None:
        item = self.extract_one(
            "Concomitant use with MAOIs can precipitate serotonin syndrome.",
            drug_name="sertraline",
            event="serotonin syndrome",
        )
        self.assertEqual(item.evidence_status, "interaction_dependent")
        self.assertEqual(item.assertion, "conditional")

    def test_negated_structured_status_controls_drug_and_class_summary(self) -> None:
        usage = self.usage("metoprolol")
        plan, drug_evidence, summary = analyze_event_for_selected_drugs(
            selected_class="beta-Adrenergic Blocker",
            class_member_count=12,
            selected_drugs=[usage],
            labels=[self.label("No cases of bradycardia were observed.")],
            event_query="bradycardia",
        )
        self.assertEqual(plan.normalized_event, "bradycardia")
        self.assertEqual(drug_evidence[0].evidence_status, "negated")
        self.assertEqual(summary.explicit_positive_count, 0)
        self.assertEqual(summary.evidence_distribution["negated"], 1)
        self.assertEqual(summary.class_assessment, "no_explicit_mentions_found")
        self.assertTrue(drug_evidence[0].evidence_snippets[0].chunk_hash)

    def test_low_confidence_label_is_insufficient_everywhere(self) -> None:
        _, drug_evidence, summary = analyze_event_for_selected_drugs(
            selected_class="beta-Adrenergic Blocker",
            class_member_count=12,
            selected_drugs=[self.usage("metoprolol")],
            labels=[
                self.label(
                    "Metoprolol can cause bradycardia.",
                    confidence="low",
                )
            ],
            event_query="bradycardia",
        )
        self.assertEqual(drug_evidence[0].evidence_status, "insufficient_label_data")
        self.assertEqual(summary.insufficient_count, 1)
        self.assertEqual(summary.class_assessment, "insufficient_class_evidence")

    def test_completed_search_without_match_is_not_found_without_evidence_items(self) -> None:
        usage = self.usage("rivaroxaban")
        label = self.label(
            "Major bleeding may involve the pancreas. Vomiting was reported."
        )
        plan = build_event_search_plan("pancreatitis")
        structured = extract_structured_event_evidence(
            selected_drugs=[usage],
            labels=[label],
            normalized_event=plan.normalized_event,
            direct_terms=plan.direct_terms,
            related_terms=plan.related_terms,
        )
        self.assertEqual(structured.items, [])
        self.assertEqual(structured.evidence_count_before_merge, 0)

        _, drug_evidence, summary = analyze_event_for_selected_drugs(
            selected_class="Factor Xa Inhibitor",
            class_member_count=1,
            selected_drugs=[usage],
            labels=[label],
            event_query="pancreatitis",
        )
        result = drug_evidence[0]
        self.assertEqual(result.evidence_status, "not_found")
        self.assertNotEqual(result.evidence_status, "insufficient_label_data")
        self.assertEqual(result.matched_sections, [])
        self.assertEqual(result.evidence_snippets, [])
        self.assertTrue(
            any("reviewed label sections" in limitation for limitation in result.limitations)
        )
        self.assertEqual(summary.not_found_count, 1)
        self.assertEqual(summary.insufficient_count, 0)
        self.assertEqual(summary.evidence_distribution["not_found"], 1)
        self.assertEqual(summary.class_assessment, "no_explicit_mentions_found")

    def test_real_positive_is_not_affected_by_empty_not_found_handling(self) -> None:
        _, drug_evidence, summary = analyze_event_for_selected_drugs(
            selected_class="Factor Xa Inhibitor",
            class_member_count=1,
            selected_drugs=[self.usage("rivaroxaban")],
            labels=[self.label("Rivaroxaban can cause pancreatitis.")],
            event_query="pancreatitis",
        )
        self.assertEqual(drug_evidence[0].evidence_status, "explicit_positive")
        self.assertEqual(len(drug_evidence[0].evidence_snippets), 1)
        self.assertEqual(summary.explicit_positive_count, 1)

    def test_openfda_fallback_uses_same_negation_rules(self) -> None:
        label = self.label("No cases of bradycardia were observed.").model_copy(
            update={
                "extraction_source": "openfda_fallback",
                "spl_chunks": [],
                "sections": {
                    "warnings_and_cautions": "No cases of bradycardia were observed."
                },
            }
        )
        _, drug_evidence, summary = analyze_event_for_selected_drugs(
            selected_class="beta-Adrenergic Blocker",
            class_member_count=12,
            selected_drugs=[self.usage("metoprolol")],
            labels=[label],
            event_query="bradycardia",
        )
        self.assertEqual(drug_evidence[0].evidence_status, "negated")
        self.assertEqual(drug_evidence[0].extraction_source, "openfda_fallback")
        self.assertTrue(drug_evidence[0].evidence_snippets[0].chunk_hash)
        self.assertEqual(summary.explicit_positive_count, 0)

    def test_distinct_sections_keep_distinct_evidence_quotes(self) -> None:
        warning = self.chunk(
            "Metoprolol can cause bradycardia.",
            chunk_hash="1" * 64,
        )
        contraindication = self.chunk(
            "Metoprolol is contraindicated when bradycardia is present.",
            chunk_hash="2" * 64,
        ).model_copy(
            update={
                "section_code": "34070-3",
                "section_title": "CONTRAINDICATIONS",
                "source_path": "section[1]/paragraph[0]",
            }
        )
        label = self.label("unused").model_copy(
            update={"spl_chunks": [warning, contraindication]}
        )
        _, drug_evidence, _ = analyze_event_for_selected_drugs(
            selected_class="beta-Adrenergic Blocker",
            class_member_count=12,
            selected_drugs=[self.usage("metoprolol")],
            labels=[label],
            event_query="bradycardia",
        )
        snippets = drug_evidence[0].evidence_snippets
        self.assertEqual(len(snippets), 2)
        self.assertEqual(
            {snippet.section for snippet in snippets},
            {"warnings_and_cautions", "contraindications"},
        )


if __name__ == "__main__":
    unittest.main()
