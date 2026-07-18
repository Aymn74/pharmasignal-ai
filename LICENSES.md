# Data sources, terms, and attribution notes

Terms last reviewed: **2026-07-17**. This inventory is operational documentation, not legal advice. Source-specific terms can change; review the linked official terms before redistributing source data or deploying the application commercially.

## openFDA

- **Purpose in this project:** Retrieve candidate drug-label records, select one label using the existing deterministic matching algorithm, obtain the SPL SET ID, and provide label-section text when DailyMed XML is unavailable.
- **Official terms:** https://open.fda.gov/terms/
- **Attribution:** Not required for material covered by CC0, but FDA asks users to credit openFDA. Suggested credit: `Data provided by the U.S. Food and Drug Administration (https://open.fda.gov)`.
- **Redistribution:** openFDA content is generally public domain and offered under CC0. Some records can contain third-party copyrighted content and are not covered by CC0; relevant dataset warnings and third-party rights must be respected.
- **Terms reviewed:** 2026-07-17.

## DailyMed

- **Purpose in this project:** Retrieve the current official SPL XML document corresponding to the SET ID selected through openFDA and extract structured safety-section paragraphs, lists, and tables.
- **Official API documentation:** https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
- **Official NLM copyright policy:** https://www.nlm.nih.gov/web_policies.html#copyright
- **Attribution:** NLM requests acknowledgement for U.S. government works, such as `Courtesy of the National Library of Medicine` or `Source: National Library of Medicine`.
- **Redistribution:** U.S. government works can generally be reused in the United States without permission. DailyMed can also expose labeling and media supplied by private entities; those items may remain copyrighted, and redistribution beyond applicable law requires permission from the rights holder. NLM does not guarantee the copyright status of every item.
- **Terms reviewed:** 2026-07-17.

## RxNorm

- **Purpose in this project:** Normalize drug names and use RXCUI identifiers for drug identity and source matching.
- **Official terms:** https://www.nlm.nih.gov/research/umls/rxnorm/docs/termsofservice.html
- **Attribution:** NLM requests acknowledgement. Recommended statement: `This product uses publicly available data courtesy of the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.`
- **Redistribution:** NLM-created normalized names and RXCUI identifiers are public domain. Full RxNorm releases also contain proprietary source vocabularies with source-specific restrictions; this project uses the public API and does not redistribute a full RxNorm release.
- **Terms reviewed:** 2026-07-17.

## RxClass

- **Purpose in this project:** Discover drug classes and retrieve class-member relationships for the class-level workflow.
- **Official API and terms:** https://lhncbc.nlm.nih.gov/RxNav/APIs/RxClassAPIs.html and https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html
- **Attribution:** NLM requests the acknowledgement statement shown in the RxNorm section above.
- **Redistribution:** No license is required merely to use the RxClass API. Returned class content can originate from third-party terminologies and remains subject to the applicable source-specific terms; API access does not grant broader redistribution rights for those terminologies.
- **Terms reviewed:** 2026-07-17.

## CMS

- **Purpose in this project:** Rank RxClass ingredients using the public Medicare Part D prescriber dataset and report aggregated claims and beneficiary counts.
- **Official API reuse guidance:** https://data.cms.gov/sites/default/files/2022-08/API%20FAQ%20v1_0.pdf
- **Attribution:** Permission is generally not required for U.S. government public data; attribution to CMS as the source is appreciated.
- **Redistribution:** Public CMS government data can generally be reused. Dataset-specific notices and third-party rights still apply. Transformed or reprocessed values must be identified as project-derived rather than represented as unchanged CMS data.
- **Terms reviewed:** 2026-07-17.

## LOINC

- **Purpose in this project:** Identify SPL safety sections by their LOINC codes and retain the corresponding display names alongside those codes.
- **Official license:** https://loinc.org/license
- **Attribution:** Required. The project uses the following notice:

  > This material contains content from LOINC (http://loinc.org). LOINC is copyright © Regenstrief Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the license at http://loinc.org/license. LOINC® is a registered United States trademark of Regenstrief Institute, Inc.

- **Redistribution:** Commercial and non-commercial use and redistribution are permitted under the LOINC license. Users must retain the required notice, keep extracted information associated with its LOINC identifier and display name, comply with third-party notices, and must not alter LOINC content or use it to create a competing terminology standard.
- **Terms reviewed:** 2026-07-17.
