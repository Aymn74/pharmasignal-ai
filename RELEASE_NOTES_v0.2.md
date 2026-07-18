# PharmaSignal AI v0.2 — Limited-validation research prototype

Freeze date: **2026-07-18**

## Core functions

- Discover official drug classes through RxNorm/RxClass relationships.
- Select representative class members using CMS 2024 prescribing data.
- Preserve the existing openFDA label-selection path and fixed SPL SET identity.
- Retrieve and parse structured DailyMed SPL XML using section metadata and LOINC codes.
- Extract deterministic event evidence with assertion, subject, context, frequency, comparator, quotation, and source traceability.
- Aggregate reviewed drug-level evidence into an explicit class-level assessment.
- Export structured analysis records and benchmark results.

## Data sources

- RxNorm and RxClass for drug and class identity.
- CMS Medicare Part D 2024 data for representative-drug ordering.
- openFDA drug labeling for the existing label-selection workflow and fallback text.
- DailyMed SPL XML for structured evidence extraction.
- LOINC section codes for section identification.

Source terms and attribution notes are documented in `LICENSES.md`.

## Verification at freeze

- Test suite: **105/105 passed**.
- Manually completed drug-event cases: **7/30**.
- Manually completed class scenarios: **1/6**.
- Mocked failure-mode tests: **4/4 passed**.
- Overall Validation Benchmark: **FAIL — validation incomplete**.

The completed minimum-validation set covers explicit-positive, interaction-dependent, negated, frequency/comparator, route/formulation, product-strength, not-found, and class-aggregation behavior. Detailed results and denominators are in `MINIMUM_VALIDATION_REPORT.md` and `VALIDATION_REPORT.md`.

## Main completed corrections

- Conservative interaction-scope classification for evidence constrained by concomitant use.
- Header-to-cell frequency and comparator extraction from structured SPL tables.
- Clause-local negation and preservation of conditional interaction evidence.
- Route/formulation applicability so evidence from another administration route does not outrank evidence for the selected product.
- Numeric-role classification separating event frequency from blood pressure, dose, product strength, sample size, duration, and other non-frequency values.
- Clean `not_found` output with null assertion, subject, section, quote, frequency, and comparator when no matching evidence exists.
- Benchmark-integrity gates for manual completion, fixed SPL version/effective-time identity, deterministic runs, supporting-quote matching, and immutable gold files during execution.

## Known limitations and research warning

- This is a limited-validation research prototype. Human review is required.
- The full 30-case and six-scenario benchmark is incomplete and remains FAIL.
- Findings are limited to fixed DailyMed SPL versions and reviewed safety sections.
- Label text does not establish causality or real-world incidence.
- `not_found` does not establish absence of risk.
- The current class runner does not independently revalidate RxClass membership, CMS order, class identifier, or class-level determinism.
- Current deterministic rules do not cover every route, formulation, linguistic construction, table layout, or clinical concept.
- The application depends on external RxNorm/RxClass, CMS, openFDA, and DailyMed availability unless a documented fallback applies.

## Run

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The local application is normally available at `http://localhost:8501/`.
