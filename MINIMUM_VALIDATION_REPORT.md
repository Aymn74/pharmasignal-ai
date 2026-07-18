# PharmaSignal AI v0.2 — Minimum Validation Report

## Release status

**Limited-validation research prototype**

- Freeze date: 2026-07-18
- Extraction rules: `structured-event-evidence-v1`
- Full benchmark status: **FAIL — validation incomplete**
- Completed drug-event cases: 7 of 30
- Completed class-level scenarios: 1 of 6

## System description

PharmaSignal AI v0.2 is a deterministic system for structured extraction and comparison of safety-event evidence from fixed DailyMed SPL labeling. Its reviewed path is:

`drug class + safety event → representative drugs → fixed SPL evidence → assertion/subject/context/frequency/comparator → supporting quote and traceability`

The application uses openFDA to select labeling and determine the SPL SET ID, then uses the corresponding DailyMed SPL XML for structured section, paragraph, list, and table extraction. The current evidence rules do not use an LLM, RAG, or embeddings.

## Completed drug-event cases

| Case ID | Drug | Event | Expected result | Actual result | SPL version | Effective time | Deterministic | Result |
|---|---|---|---|---|---:|---:|---|---|
| EP-01 | trazodone | serotonin syndrome | `explicit_positive` | `explicit_positive` | 4 | 20260714 | Yes | PASS |
| EP-02 | sertraline | serotonin syndrome | `explicit_positive` | `explicit_positive` | 2 | 20260630 | Yes | PASS |
| EP-03 | fluoxetine | serotonin syndrome | `explicit_positive` | `explicit_positive` | 1 | 20260622 | Yes | PASS |
| IC-01 | linezolid | serotonin syndrome | `interaction_dependent / conditional` | `interaction_dependent / conditional` | 17 | 20260715 | Yes | PASS |
| FC-01 | amlodipine | nausea | `explicit_positive`; frequency `2.9%`; comparator `Placebo 1.9%` | `explicit_positive`; frequency `2.9%`; comparator `Placebo 1.9%` | 16 | 20260701 | Yes | PASS |
| NC-01 | Dapsone gel 5% | peripheral neuropathy | `negated / absent`; frequency `null`; topical evidence applicable | `negated / absent`; frequency `null`; primary evidence marked topical/gel/matching route | 1 | 20260709 | Yes | PASS |
| NF-01 | rivaroxaban | pancreatitis | `not_found`; assertion and subject `null`; zero evidence items | `not_found`; assertion and subject `null`; zero evidence items | 1 | 20260706 | Yes | PASS |

All seven fixed SPL versions and effective times matched their manually reviewed identities. All scored supporting quotes matched their gold quotations.

## Completed class-level scenario

| Scenario ID | Class | Event | Drugs | Expected assessment | Actual assessment | Result |
|---|---|---|---|---|---|---|
| CLASS-POS-01 | Serotonin Reuptake Inhibitor | serotonin syndrome | trazodone, sertraline, fluoxetine | `consistent_label_evidence` | `consistent_label_evidence` | PASS |

The reference scenario is based on three of three manually reviewed `explicit_positive` drug-level cases. Completing this scenario does not satisfy the full six-scenario validation gate.

## Minimum-validation coverage

The completed cases exercise:

- explicit-positive evidence;
- interaction-dependent evidence;
- frequency and comparator extraction from a table;
- negation;
- route/formulation applicability;
- separation of product strength from event frequency;
- constrained not-found semantics;
- class-level aggregation;
- fixed SPL identity and effective-time verification;
- deterministic output;
- failure-mode handling.

## Failure-mode tests

| Mocked failure | Expected explicit outcome | Result |
|---|---|---|
| DailyMed timeout | `openfda_fallback` | PASS |
| openFDA timeout | `insufficient_label_data` | PASS |
| Malformed SPL XML | `openfda_fallback` | PASS |
| Empty safety sections | `insufficient_label_data` | PASS |

## Numerical results

| Measure | Result |
|---|---:|
| pytest | 105/105 passed |
| Completed drug cases | 7/30 |
| Completed class scenarios | 1/6 |
| Failure-mode tests | 4/4 passed |
| Supporting-quote accuracy for scored completed cases | 1.00 |
| Frequency exact-match accuracy for scored cases | 1.00 |
| Comparator exact-match accuracy for scored cases | 1.00 |
| Deterministic-output rate for completed cases | 1.00 |
| Unsupported-positive count | 0 |
| Traceability completeness under the full benchmark metric | 0.8333 |

Metrics with limited denominators describe only the manually completed cases. They are not estimates of performance across the unreviewed benchmark slots or outside the fixed labels.

## Validation limits

- The 30-case item benchmark is incomplete: 23 cases still require manual review.
- The six class-level scenarios are incomplete: five scenarios still require manual review.
- The full benchmark remains **FAIL — validation incomplete**.
- No claim of clinical validation or readiness for production use is made.
- The system does not establish causality or real-world event incidence.
- `not_found` means only that a matching event was not found in the reviewed sections of the fixed label; it does not establish absence of risk.
- Label findings are constrained by the fixed SPL version, effective time, reviewed sections, and available extraction chunks.
- Human review remains required for every result.
- The current class runner does not automatically verify RxClass membership, CMS ordering, the class identifier, or class-level determinism.
- Route, formulation, language, table, and numeric-pattern coverage is limited to the current rules and tests.
- Failure-mode results are mocked resilience tests and do not measure external-service availability.

## Approved release statement

“PharmaSignal AI v0.2 is a deterministic, auditable research prototype for structured extraction and comparison of safety-event evidence from fixed DailyMed SPL labeling. Limited validation has been completed across seven drug-event cases and one class-level scenario. Human review remains required.”
