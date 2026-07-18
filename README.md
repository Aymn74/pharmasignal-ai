# PharmaSignal AI v0.2

**Deterministic, auditable extraction of safety-event evidence from fixed DailyMed SPL labeling.**

> [!IMPORTANT]
> PharmaSignal AI is a **limited-validation research prototype**. It is not a clinical decision-support system, does not establish causality or real-world incidence, and requires human review of every result. The full validation benchmark remains **FAIL — validation incomplete**.

## What this project does

PharmaSignal AI is a local Streamlit application that lets a researcher:

1. discover an official drug class or start from a drug name;
2. review and confirm the class before analysis;
3. select representative ingredients using CMS Medicare Part D 2024 utilization data;
4. enter a safety event such as `serotonin syndrome` or `bradycardia`;
5. retrieve a fixed drug label and its SPL identity;
6. extract structured evidence from safety sections, paragraphs, lists, and tables;
7. classify each drug-event record using explicit deterministic rules; and
8. synthesize the drug-level records into a traceable class-level assessment.

The project does **not** use an LLM, OpenAI API, RAG, embeddings, or FAERS at runtime. It does not ship fabricated drug lists, synthetic evidence, or fallback CSV results.

## Evidence pipeline

```text
Drug class + safety event
        │
        ├─ RxNorm / RxClass: normalize drug identity and resolve class membership
        ├─ CMS Medicare Part D: rank representative ingredients
        ├─ openFDA: select the label record and obtain the SPL SET ID
        ├─ DailyMed SPL XML: extract structured safety sections and chunks
        ├─ LOINC: identify label sections by standard codes
        └─ Deterministic rules: assertion, subject, context, frequency,
                                comparator, quotation, and source traceability
        │
        ├─ Drug-level event evidence
        └─ Class-level synthesis
```

Each supporting result is designed to remain traceable to the drug, RXCUI, SPL SET ID, label version, effective time, section, subsection, source path, chunk hash, and supporting quotation.

## Organizations, standards, and data sources

PharmaSignal AI is an independent project. The organizations below provide public data, terminology, or infrastructure; they do not endorse this application or its results.

| Organization or resource | What it is | How PharmaSignal AI uses it | Important boundary |
|---|---|---|---|
| [U.S. National Library of Medicine (NLM)](https://www.nlm.nih.gov/about/index.html) | An institute of the U.S. National Institutes of Health and a major biomedical information organization. NLM produces or operates RxNorm, RxClass, and DailyMed. | Supplies normalized drug identity, class relationships, and structured labeling access. | NLM does not review PharmaSignal AI results and does not endorse the project. |
| [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/overview.html) | NLM's normalized naming system for generic and branded drugs. It assigns identifiers such as the RXCUI and supports interoperability between drug vocabularies. | Normalizes drug names, resolves ingredient-level identity, and records RXCUI values. | Name normalization is identity support, not clinical equivalence or proof that two products have identical labeling. |
| [RxClass](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxClassAPIs.html) | An NLM service for navigating between drug classes and their RxNorm drug members. It exposes classes from sources such as FDA Structured Product Labeling, ATC, and other terminologies. | Discovers classes, retrieves members, and records class type, source, and relationship. | Class sources and relationship types have different meanings; a technical or single-ingredient classification is not automatically suitable for class-level analysis. |
| [DailyMed](https://dailymed.nlm.nih.gov/dailymed/about-dailymed.cfm) | NLM's public database of labeling submitted to FDA by companies and currently in use. Labels are available in formats including structured XML. | Provides the fixed SPL XML used to extract sections, paragraphs, lists, and table rows. | DailyMed explains that “in-use” labeling may differ from the latest FDA-approved labeling and that NLM does not review SPL content before publication. |
| [U.S. Food and Drug Administration (FDA) / openFDA](https://open.fda.gov/about/) | FDA is the U.S. federal agency responsible for regulating drugs and other products. openFDA is an FDA initiative that exposes public structured datasets and APIs, including drug labeling. | Uses the existing deterministic matching process to select a label record, obtain the SPL SET ID, and provide label text if DailyMed XML is unavailable. | openFDA label data is source material, not an FDA assessment of PharmaSignal AI's conclusions. This project does not use the openFDA adverse-event dataset or FAERS. |
| [Centers for Medicare & Medicaid Services (CMS)](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers) | A U.S. federal agency that administers Medicare and Medicaid programs. Its Medicare Part D Prescribers datasets summarize prescription claims paid under Part D. | Uses 2024 aggregated claims to rank representative class ingredients. | CMS data represents Medicare Part D activity, not the entire population, clinical importance, safety risk, prevalence, or causality. |
| [LOINC](https://loinc.org/about/) | Logical Observation Identifiers Names and Codes, a terminology maintained by Regenstrief Institute and the LOINC Committee for identifying health measurements, observations, and documents. | Identifies SPL sections by standard section codes rather than title text alone. | LOINC identifies document sections; it does not classify the safety evidence or validate the extracted conclusion. |
| [Supabase](https://supabase.com/) | An optional hosted data platform used only when the operator enables storage and a refreshable class-catalog cache. | Can store server-side project records and cache RxClass discovery results. | Supabase is optional infrastructure, not a scientific or regulatory data source. Live RxClass remains authoritative. |

Detailed source terms, attribution language, and redistribution notes are maintained in [LICENSES.md](LICENSES.md).

## What the evidence statuses mean

| Status | Meaning in this project |
|---|---|
| `explicit_positive` | The event or an approved direct synonym is explicitly stated in a reviewed safety section for the selected drug/product context. |
| `interaction_dependent` | The event is explicitly constrained by concomitant use, coadministration, or another interaction condition. |
| `related_but_not_explicit` | A related concept is present, but the requested event is not stated directly enough for an explicit-positive result. |
| `negated` | The reviewed text explicitly states that the event was not observed, reported, or identified in the applicable context. |
| `historical_or_preexisting` | The event describes patient history or a condition present before treatment rather than an effect attributed to the selected drug. |
| `comparator_only` | The event appears only for placebo or another comparator, not for the selected treatment arm. |
| `not_found` | No matching evidence was found in the available reviewed sections of the fixed label. This does **not** establish absence of risk. |
| `insufficient_label_data` | The label is unavailable, incomplete, or insufficiently matched for a reliable reviewed-section result. |

## Current validation status

The frozen `v0.2.0` release includes a manually reviewed minimum-validation set:

- **105/105** automated tests passed;
- **7/30** drug-event benchmark cases completed and evaluated;
- **1/6** class-level scenarios completed and evaluated;
- **4/4** mocked failure-mode tests passed;
- supporting-quote accuracy for scored completed cases: **1.00**;
- frequency exact-match accuracy for scored cases: **1.00**;
- comparator exact-match accuracy for scored cases: **1.00**;
- deterministic-output rate for completed cases: **1.00**; and
- overall benchmark status: **FAIL — validation incomplete**.

The limited set covers explicit-positive evidence, interaction dependence, negation, route/formulation applicability, table frequency and comparator extraction, product-strength separation, not-found semantics, fixed SPL identity, and one class-level aggregation scenario.

See:

- [Minimum Validation Report](MINIMUM_VALIDATION_REPORT.md)
- [Full Validation Benchmark Report](VALIDATION_REPORT.md)
- [Release Notes](RELEASE_NOTES_v0.2.md)
- [Release Manifest](RELEASE_MANIFEST_v0.2.txt)
- [v0.2.0 release package](https://github.com/Aymn74/pharmasignal-ai/releases/tag/v0.2.0)

## Installation and local run

Requirements:

- Python 3.12
- Internet access to the official upstream services used by the selected workflow

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open <http://localhost:8501/>.

An openFDA key is optional; public access works at lower rate limits. Runtime settings can be supplied through environment variables or a local `.env` file, which is ignored by Git. Supported settings include:

- `OPENFDA_API_KEY`
- `CMS_DATASET_ID`
- `CMS_DATASET_VERSION_ID`
- `CMS_DATA_YEAR`
- `REQUEST_TIMEOUT_SECONDS`
- `DEFAULT_REPRESENTATIVE_COUNT`
- optional Supabase settings: `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, and `SUPABASE_SECRET_KEY`

Do not commit `.env`, service-role keys, API keys, or other credentials.

## Typical workflow

1. Choose **Search by class name**, **Search by drug name**, or class browsing.
2. Review official, possible, technical, and combination-product classifications separately.
3. Confirm an analysis-ready class.
4. Review CMS-ranked class members and choose the drugs to analyze.
5. Enter an adverse event or safety outcome.
6. Run the analysis and inspect drug-level evidence, quotations, label identity, and extraction diagnostics.
7. Review the class-level synthesis and limitations.
8. Export JSON only after confirming that the evidence is appropriate for the research question.

## Project structure

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit user interface and workflow state. |
| `models.py` | Typed settings and data models. |
| `data_sources.py` | RxNorm/RxClass, CMS, openFDA, DailyMed, and optional Supabase access. |
| `spl_parser.py` | Structured DailyMed SPL XML parsing. |
| `spl_qa.py` | SPL chunk diagnostics, deduplication, length statistics, and hashes. |
| `structured_evidence.py` | Deterministic event-evidence extraction and classification. |
| `event_analysis.py` | Drug-level analysis and class-level synthesis. |
| `benchmark/` | Fixed manual-gold cases, class scenarios, integrity tests, runner, and results. |
| `test_structured_evidence.py` | Regression tests for evidence rules and numeric-role handling. |
| `test_unified_event_analysis.py` | Unified analysis-path tests. |
| `supabase_schema.sql` | Optional server-side storage and catalog-cache schema. |

## Important limitations

- Label evidence does not prove that a drug caused an event.
- Absence of a term in reviewed sections does not prove absence of risk.
- Labels can differ by manufacturer, formulation, route, version, and effective time.
- A class-level summary is only as representative as the selected drugs and available labels.
- CMS claims are used for transparent representative-drug ranking, not for safety-signal estimation.
- The current class benchmark runner does not independently revalidate RxClass membership, CMS ordering, class identity, or class-level determinism.
- External APIs and datasets can change or become temporarily unavailable.
- Human review remains required before interpreting or communicating any finding.

## Optional Supabase storage

Supabase is not required to run the scientific workflow. If enabled, execute `supabase_schema.sql` in the Supabase SQL editor and configure the server-side variables locally. The schema enables row-level security and does not grant anonymous or authenticated client access. The `drug_class_catalog` table is a refreshable search cache; it does not replace live RxClass as the authority.

## Release statement

> PharmaSignal AI v0.2 is a deterministic, auditable research prototype for structured extraction and comparison of safety-event evidence from fixed DailyMed SPL labeling. Limited validation has been completed across seven drug-event cases and one class-level scenario. Human review remains required.
