# PharmaSignal AI

`PharmaSignal AI` is a local Streamlit proof of concept that connects directly to the official NLM RxClass, CMS Medicare Part D, and openFDA drug-label APIs. It does not ship sample drug lists, synthetic data, fallback CSV files, or fabricated results.

## Run

Double-click `run.bat`, or run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Then open <http://localhost:8501>.

The discovery area supports partial/fuzzy class-name search, a small disclosed synonym assist, generic/brand/RXCUI drug search through RxNorm followed by official RxClass relationships, and lazy browsing of the official RxClass catalog. Selecting a result only loads RxClass member details; CMS and openFDA do not run until **Use this drug class** is pressed.

Copy `.env.example` to `.env` only when you want to configure an openFDA key, fixed CMS identifiers, Supabase, or different timeouts. When CMS identifiers are blank, the app discovers the requested year's current identifiers live from the official `https://data.cms.gov/data.json` catalog. On 2026-07-17, the verified 2024 identifiers were:

- Dataset Type Identifier: `c8ea3f8e-3a09-4fea-86f2-8902fb4b0920`
- Dataset Version Identifier: `9b4c142c-69cc-4a96-a09a-7cf2ba7f5816`

## Optional Supabase storage and catalog cache

Run `supabase_schema.sql` in the Supabase SQL editor, then set `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in `.env`. The publishable key is accepted for configuration completeness but is not used for writes. Both tables have RLS enabled, grant no anonymous or authenticated access, and are written only by the server-side secret key. `drug_class_catalog` is only a refreshable search cache of records fetched from RxClass; live RxClass remains the authority and the app falls back to it when the cache is unavailable.

## Evidence cautions

- CMS measures prescribing for Medicare Part D beneficiaries, not the whole population.
- Beneficiary totals shown across multiple matching CMS brand rows are sums of published rows and are not patient-level deduplication.
- Salt-normalized matches are marked approximate; arbitrary substring matches are rejected.
- Drug-label text does not establish causality, and absence of a term does not mean no effect.
- The NLM does not endorse this application or any result produced from its publicly available data.
