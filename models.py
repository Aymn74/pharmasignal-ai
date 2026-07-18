from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openfda_api_key: str = ""
    cms_dataset_id: str = ""
    cms_dataset_version_id: str = ""
    cms_data_year: int = 2024
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_secret_key: str = ""
    request_timeout_seconds: float = 30
    default_representative_count: int = 3


class SourceDetail(BaseModel):
    source: str
    query: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    record_count: int = 0
    note: str = ""


class ClassCandidate(BaseModel):
    class_name: str
    class_id: str
    class_type: str
    rela_source: str
    rela: str | None = None
    member_count: int | None = None
    example_members: list[str] = Field(default_factory=list)
    match_score: float | None = None
    match_strength: str = "official"
    matched_via_synonym: bool = False
    synonym_expansion: str | None = None
    relations: list[str] = Field(default_factory=list)
    why_shown: str = ""
    membership_category: str = ""
    direct_membership: bool = False
    combination_membership: bool = False
    recommended_for_analysis: bool = False
    discovered_by_drug: bool = False
    membership_examples: list[str] = Field(default_factory=list)
    member_rxcuis: list[str] = Field(default_factory=list, exclude=True)
    rela_sources: list[str] = Field(default_factory=list)
    preferred_source: str = ""
    additional_sources: list[str] = Field(default_factory=list)


class DrugCandidate(BaseModel):
    rxcui: str
    name: str
    tty: str = ""
    synonym: str = ""
    score: float | None = None
    rank: int | None = None


class DrugMember(BaseModel):
    rxcui: str
    name: str
    tty: str


class CmsDatasetIdentity(BaseModel):
    dataset_type_id: str
    dataset_version_id: str
    year: int


class CmsUsage(BaseModel):
    member: DrugMember
    cms_generic_names: list[str]
    total_claims: int
    total_beneficiaries: int | None = None
    data_year: int
    match_quality: str
    approximate_match: bool = False
    match_note: str = ""
    rank: int | None = None


class SPLSectionChunk(BaseModel):
    set_id: str
    version: str | None = None
    effective_time: str | None = None
    section_code: str
    section_title: str
    loinc_display_name: str
    subsection_title: str | None = None
    chunk_type: str
    chunk_index: int
    text: str
    table_id: str | None = None
    row_index: int | None = None
    column_headers: list[str] = Field(default_factory=list)
    row_cells: list[str] = Field(default_factory=list)
    source_path: str
    character_count: int = 0
    chunk_hash: str = ""


class SPLExtractionDiagnostics(BaseModel):
    qa_version: str = "spl-extraction-qa-v1"
    set_id: str
    spl_version: str | None = None
    section_count: int
    total_chunks: int
    unique_chunks: int
    duplicate_chunks: int
    duplicate_rate: float
    minimum_characters: int
    median_characters: float
    p90_characters: float
    maximum_characters: int
    chunks_over_2000_characters: int
    chunks_over_5000_characters: int
    empty_chunks: int
    extraction_source: str = "dailymed_spl_xml"


class LabelEvidence(BaseModel):
    rxcui: str
    requested_name: str
    query_field: str
    matched_label_count: int
    selected_spl_set_id: str | None = None
    effective_time: str | None = None
    selection_reason: str
    label_match_score: int
    label_match_confidence: str
    label_match_reasons: list[str] = Field(default_factory=list)
    selected_label_product: dict[str, Any] = Field(default_factory=dict)
    generic_names: list[str] = Field(default_factory=list)
    brand_names: list[str] = Field(default_factory=list)
    substance_names: list[str] = Field(default_factory=list)
    label_rxcuis: list[str] = Field(default_factory=list)
    pharm_class_epc: list[str] = Field(default_factory=list)
    pharm_class_moa: list[str] = Field(default_factory=list)
    sections: dict[str, str] = Field(default_factory=dict)
    extraction_source: str = "openfda_fallback"
    spl_version: str | None = None
    spl_effective_time: str | None = None
    spl_chunks: list[SPLSectionChunk] = Field(default_factory=list)
    spl_diagnostics: SPLExtractionDiagnostics | None = None
    dailymed_xml_url: str | None = None
    dailymed_warning: str | None = None


class EventEvidenceSnippet(BaseModel):
    section: str
    matched_term: str
    text: str
    extraction_source: str = "openfda_fallback"
    spl_version: str | None = None
    section_code: str | None = None
    subsection_title: str | None = None
    chunk_type: str = "section_text"
    source_path: str = ""
    chunk_hash: str = ""


class EventEvidenceItem(BaseModel):
    drug_name: str
    rxcui: str
    normalized_event: str
    matched_term: str
    evidence_status: Literal[
        "explicit_positive",
        "related_but_not_explicit",
        "negated",
        "historical_or_preexisting",
        "comparator_only",
        "interaction_dependent",
        "not_found_in_reviewed_sections",
        "insufficient_label_data",
    ]
    assertion: Literal["present", "absent", "uncertain", "conditional", "historical"]
    subject: Literal[
        "selected_drug",
        "concomitant_drug",
        "comparator",
        "patient_history",
        "general_class_statement",
        "unclear",
    ]
    section_code: str = ""
    section_title: str = ""
    subsection_title: str | None = None
    evidence_context: list[str] = Field(default_factory=list)
    frequency_text: str | None = None
    frequency_value: float | None = None
    frequency_unit: str | None = None
    comparator_text: str | None = None
    population_context: list[str] = Field(default_factory=list)
    dose_context: list[str] = Field(default_factory=list)
    interaction_context: list[str] = Field(default_factory=list)
    temporal_context: list[str] = Field(default_factory=list)
    route_context: list[str] = Field(default_factory=list)
    postmarketing_only: bool = False
    source_text: str = ""
    supporting_quote: str = ""
    chunk_type: str = ""
    source_path: str = ""
    chunk_hash: str = ""
    extraction_method: str = "deterministic_rules_v1"
    extraction_confidence: Literal["high", "medium", "low"] = "high"
    mentioned_in_sections: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    chunk_hashes: list[str] = Field(default_factory=list)


class DrugEventEvidence(BaseModel):
    drug_name: str
    rxcui: str
    cms_rank: int | None = None
    selected_spl_set_id: str | None = None
    label_effective_time: str | None = None
    event_query: str
    searched_terms: list[str] = Field(default_factory=list)
    evidence_status: str
    matched_sections: list[str] = Field(default_factory=list)
    evidence_snippets: list[EventEvidenceSnippet] = Field(default_factory=list)
    label_match_confidence: str
    extraction_source: str = "openfda_fallback"
    spl_version: str | None = None
    limitations: list[str] = Field(default_factory=list)


class ClassLevelSummary(BaseModel):
    selected_class: str
    adverse_event: str
    class_member_count: int
    drugs_analyzed: int
    explicit_positive_count: int
    related_count: int
    not_found_count: int
    insufficient_count: int
    evidence_distribution: dict[str, int] = Field(default_factory=dict)
    class_assessment: str
    interpretation: str
    limitations: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    selected_class: ClassCandidate
    cms_dataset: CmsDatasetIdentity | None = None
    class_members: list[DrugMember] = Field(default_factory=list)
    cms_usage: list[CmsUsage] = Field(default_factory=list)
    selected_drugs: list[CmsUsage] = Field(default_factory=list)
    labels: list[LabelEvidence] = Field(default_factory=list)
    source_status: dict[str, str] = Field(default_factory=dict)
    source_details: list[SourceDetail] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    adverse_event_query: str = ""
    normalized_event: str = ""
    searched_terms: list[str] = Field(default_factory=list)
    drug_event_evidence: list[DrugEventEvidence] = Field(default_factory=list)
    class_level_summary: ClassLevelSummary | None = None
    class_assessment: str = ""
    analysis_rules_version: str = ""
    event_evidence_items: list[EventEvidenceItem] = Field(default_factory=list)
    evidence_item_count_before_merge: int = 0
    evidence_item_count: int = 0
    unique_positive_evidence_count: int = 0
    negated_evidence_count: int = 0
    interaction_dependent_count: int = 0
    extraction_rules_version: str = ""
