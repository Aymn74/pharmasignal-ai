# PharmaSignal AI v0.2 Validation Benchmark

The benchmark is intentionally separate from application runtime and extraction rules.

## Manual gold-standard workflow

1. Assign a fixed drug, RXCUI, event, and DailyMed SPL SET ID to each slot in `benchmark_cases.json`.
2. Review the source SPL manually and enter every `expected_*` field. Null frequency or comparator values are allowed when absence is the reviewed answer.
3. Set `score_frequency` or `score_comparator` only when that field should enter its exact-match denominator.
4. Add reviewer notes and set `gold_review_complete` to `true` only after human approval.
5. Complete the six class scenarios independently in `class_scenarios.json`; each `drugs` entry must contain `drug_name`, `rxcui`, and `spl_set_id`.

The runner never fills expected answers and ignores records whose `gold_review_complete` value is not `true`.

## Run

```powershell
.\.venv\Scripts\python.exe benchmark\run_validation_benchmark.py
.\.venv\Scripts\python.exe -m unittest -v benchmark.test_failure_modes
```

Outputs are written to `benchmark/benchmark_results.json` and `VALIDATION_REPORT.md`.
