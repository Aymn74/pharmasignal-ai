from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from models import (
    AnalysisResult,
    ClassCandidate,
    CmsDatasetIdentity,
    CmsUsage,
    DrugCandidate,
    DrugMember,
    LabelEvidence,
    Settings,
    SourceDetail,
)


RXCLASS_SEARCH_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byName.json"
RXCLASS_MEMBERS_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json"
RXCLASS_ALL_CLASSES_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/allClasses.json"
RXCLASS_BY_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
RXNORM_APPROXIMATE_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
RXNORM_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNORM_PROPERTIES_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
CMS_CATALOG_URL = "https://data.cms.gov/data.json"
CMS_DATASET_TITLE = "Medicare Part D Prescribers - by Geography and Drug"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
ALLOWED_TTYS = {"IN", "PIN", "MIN"}


class DataSourceError(RuntimeError):
    def __init__(self, source: str, user_message: str):
        super().__init__(user_message)
        self.source = source
        self.user_message = user_message


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _record_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return len(results)
    return 0


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    redacted = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in {"api_key", "apikey", "key", "token", "authorization"}:
            value = "***REDACTED***"
        redacted.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted), parts.fragment))


def _safe_error_text(text: str, settings: Settings) -> str:
    for secret in (settings.openfda_api_key, settings.supabase_secret_key):
        if secret:
            text = text.replace(secret, "***REDACTED***")
    return text


def _request_json(
    source: str,
    url: str,
    settings: Settings,
    *,
    params: dict[str, Any] | None = None,
    allow_404: bool = False,
) -> tuple[Any | None, SourceDetail]:
    attempts = 3
    last_network_error: Exception | None = None

    for attempt in range(attempts):
        try:
            with httpx.Client(
                timeout=settings.request_timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "PharmaSignal-AI-PoC/1.0"},
            ) as client:
                response = client.get(url, params=params)

            query = _redact_url(str(response.request.url))
            detail = SourceDetail(source=source, query=query)

            if response.status_code == 404 and allow_404:
                detail.note = "No records found (HTTP 404)."
                return None, detail

            if response.status_code in {400, 404}:
                message = "The request was rejected" if response.status_code == 400 else "No matching resource was found"
                raise DataSourceError(source, f"{source}: {message} (HTTP {response.status_code}).")

            if response.status_code == 429:
                if attempt < attempts - 1:
                    retry_after = response.headers.get("Retry-After", "1")
                    try:
                        delay = min(float(retry_after), 3.0)
                    except ValueError:
                        delay = 1.0
                    time.sleep(delay)
                    continue
                raise DataSourceError(source, f"{source}: API rate limit exceeded (HTTP 429).")

            if 500 <= response.status_code < 600:
                if attempt < attempts - 1:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise DataSourceError(source, f"{source}: temporary upstream error (HTTP {response.status_code}).")

            response.raise_for_status()
            payload = response.json()
            detail.record_count = _record_count(payload)
            return payload, detail

        except DataSourceError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_network_error = exc
            if attempt < attempts - 1:
                time.sleep(0.6 * (attempt + 1))
                continue
        except httpx.HTTPError as exc:
            raise DataSourceError(source, f"{source}: HTTP connection failed ({type(exc).__name__}).") from exc
        except ValueError as exc:
            raise DataSourceError(source, f"{source}: returned an invalid JSON response.") from exc

    error_name = type(last_network_error).__name__ if last_network_error else "NetworkError"
    raise DataSourceError(source, f"{source}: internet connection failed or timed out ({error_name}).")


RELATION_CANDIDATES: dict[str, list[tuple[str, str | None]]] = {
    "EPC": [("FDASPL", "has_EPC"), ("DAILYMED", "has_EPC"), ("MEDRT", "has_EPC")],
    "MOA": [("MEDRT", "has_MoA"), ("FDASPL", "has_MoA"), ("DAILYMED", "has_MoA")],
    "PE": [("MEDRT", "has_PE"), ("FDASPL", "has_PE"), ("DAILYMED", "has_PE")],
    "CHEM": [("MEDRT", "has_Chemical_Structure"), ("FDASPL", "has_chemical_structure")],
    "DISPOS": [("SNOMEDCT", "isa_disposition")],
    "VA": [("VA", "has_VAClass")],
    "ATC": [("ATC", None)],
    "ATC1-4": [("ATC", None)],
}

CLASS_SEARCH_SYNONYMS = {
    "beta blockers": "beta-Adrenergic Blocker",
    "beta antagonists": "Adrenergic beta-Antagonists",
    "ace inhibitors": "Angiotensin-Converting Enzyme Inhibitors",
    "statins": "HMG-CoA Reductase Inhibitors",
}

FALLBACK_RELA_SOURCES = ["FDASPL", "DAILYMED", "MEDRT", "VA", "ATC", "ATCPROD", "SNOMEDCT"]


def _relation_options(class_type: str) -> list[tuple[str, str | None]]:
    normalized = class_type.upper()
    if normalized.startswith("ATC"):
        return [("ATC", None)]
    return RELATION_CANDIDATES.get(normalized, [(source, None) for source in FALLBACK_RELA_SOURCES])


def fetch_rxclass_catalog(
    settings: Settings,
) -> tuple[list[ClassCandidate], list[SourceDetail]]:
    payload, detail = _request_json("RxClass", RXCLASS_ALL_CLASSES_URL, settings)
    raw_classes = _as_list(
        (payload or {}).get("rxclassMinConceptList", {}).get("rxclassMinConcept")
    )
    candidates: list[ClassCandidate] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_classes:
        class_id = str(item.get("classId") or "").strip()
        class_name = str(item.get("className") or "").strip()
        class_type = str(item.get("classType") or "").strip()
        if not class_id or not class_name:
            continue
        key = (class_id, class_type)
        if key in seen:
            continue
        seen.add(key)
        relation_options = _relation_options(class_type)
        source, rela = relation_options[0]
        if class_type.upper() not in RELATION_CANDIDATES and not class_type.upper().startswith("ATC"):
            source = "Resolve on selection"
            rela = None
        candidates.append(
            ClassCandidate(
                class_name=class_name,
                class_id=class_id,
                class_type=class_type,
                rela_source=source,
                rela=rela,
            )
        )
    candidates.sort(key=lambda item: (item.class_name.casefold(), item.class_type, item.class_id))
    detail.record_count = len(candidates)
    detail.note = "Official RxClass allClasses catalog; member details are loaded only after selection."
    return candidates, [detail]


def _class_match(candidate: ClassCandidate, query: str) -> tuple[int, float] | None:
    name = normalize_name(candidate.class_name)
    if not name or not query:
        return None
    if name == query:
        return (0, 1.0)
    if name.startswith(query):
        return (1, 0.99)
    if query in name:
        return (2, 0.95)

    query_tokens = query.split()
    name_tokens = name.split()
    if query_tokens and all(token in name_tokens for token in query_tokens):
        return (2, 0.93)

    if len(query) < 4 or not name_tokens:
        return None
    phrase_length = len(query_tokens)
    phrases = [
        " ".join(name_tokens[index : index + phrase_length])
        for index in range(0, len(name_tokens) - phrase_length + 1)
    ]
    fuzzy_scores = [
        SequenceMatcher(None, query, phrase).ratio()
        for phrase in phrases
        if phrase
        and query[0] == phrase[0]
        and abs(len(query) - len(phrase)) <= max(2, phrase_length)
    ]
    ratio = max(fuzzy_scores, default=0.0)
    if ratio >= 0.88:
        return (3, ratio)
    return None


def search_class_catalog(
    query: str,
    catalog: list[ClassCandidate],
) -> tuple[list[ClassCandidate], list[ClassCandidate], str | None, list[str]]:
    normalized_query = normalize_name(query)
    if not normalized_query:
        raise DataSourceError("RxClass", "Enter part of a drug class name before searching.")

    expansion = CLASS_SEARCH_SYNONYMS.get(normalized_query)
    official_query = normalize_name(expansion) if expansion else normalized_query
    ranked: list[tuple[int, float, ClassCandidate]] = []
    for candidate in catalog:
        if expansion and normalize_name(candidate.class_name) != official_query:
            continue
        match = _class_match(candidate, official_query)
        if match is None:
            continue
        bucket, score = match
        ranked.append(
            (
                bucket,
                score,
                candidate.model_copy(
                    update={
                        "match_score": round(score, 4),
                        "match_strength": "official" if bucket <= 2 else "possible",
                        "matched_via_synonym": bool(expansion),
                        "synonym_expansion": expansion,
                    }
                ),
            )
        )

    ranked.sort(
        key=lambda row: (
            row[0],
            -row[1],
            {"EPC": 0, "MOA": 1, "PE": 2, "TC": 3, "VA": 4, "ATC1-4": 5, "ATC": 5, "CHEM": 6}.get(
                row[2].class_type.upper(), 7
            ),
            len(normalize_name(row[2].class_name)),
            row[2].class_name.casefold(),
            row[2].class_id,
        )
    )
    strong = [row[2] for row in ranked if row[0] <= 2][:40]
    possible = [row[2] for row in ranked if row[0] == 3][:20]
    return strong, possible, expansion, normalized_query.split()


def enrich_class_candidates(
    candidates: list[ClassCandidate], settings: Settings, *, limit: int = 12
) -> tuple[list[ClassCandidate], list[SourceDetail]]:
    if not candidates or limit <= 0:
        return candidates, []
    enriched = list(candidates)
    details: list[SourceDetail] = []

    def load(index: int, item: ClassCandidate):
        members, resolved, item_details = get_class_members(item, settings)
        member_count = len(members)
        recommended = (
            resolved.discovered_by_drug
            and resolved.direct_membership
            and not resolved.combination_membership
            and _is_recommended_relationship(resolved)
            and member_count >= 2
        )
        category = resolved.membership_category
        if recommended:
            category = "recommended"
        elif resolved.discovered_by_drug and (
            member_count < 2 or resolved.class_type.upper() in TECHNICAL_CLASS_TYPES
        ):
            category = "technical"
        return index, resolved.model_copy(
            update={
                "member_count": member_count,
                "example_members": [member.name for member in members[:5]],
                "member_rxcuis": [member.rxcui for member in members],
                "recommended_for_analysis": recommended,
                "membership_category": category,
            }
        ), item_details

    with ThreadPoolExecutor(max_workers=min(6, limit)) as executor:
        futures = {
            executor.submit(load, index, item): index
            for index, item in enumerate(candidates[:limit])
        }
        for future in as_completed(futures):
            try:
                index, candidate, item_details = future.result()
            except DataSourceError:
                continue
            enriched[index] = candidate
            details.extend(item_details)
    if any(item.discovered_by_drug for item in enriched):
        enriched = _refine_moa_recommendations(enriched)
    return enriched, details


PHARMACOLOGIC_STOP_WORDS = {
    "agent",
    "agents",
    "antagonist",
    "antagonists",
    "agonist",
    "agonists",
    "blocker",
    "blockers",
    "class",
    "drug",
    "drugs",
    "inhibitor",
    "inhibitors",
    "receptor",
    "receptors",
    "selective",
    "excluding",
    "excl",
    "other",
}

SECONDARY_MOA_PATTERNS = {
    "cytochrome p450",
    "cyp2",
    "cyp3",
    "enzyme inhibition",
    "metabolic enzyme",
    "drug transporter",
    "transporters",
    "p glycoprotein",
    "uptake transporter",
    "efflux transporter",
}


def _pharmacologic_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalize_name(value).split()
        if token not in PHARMACOLOGIC_STOP_WORDS
        and (len(token) >= 5 or any(character.isdigit() for character in token))
    }


def _refine_moa_recommendations(
    candidates: list[ClassCandidate],
) -> list[ClassCandidate]:
    primary_classes = [
        item
        for item in candidates
        if item.direct_membership
        and item.member_count is not None
        and item.member_count >= 2
        and (item.class_type.upper() == "EPC" or item.class_type.upper().startswith("ATC"))
    ]
    primary_tokens = set().union(
        *(_pharmacologic_tokens(item.class_name) for item in primary_classes)
    ) if primary_classes else set()
    primary_members = [set(item.member_rxcuis) for item in primary_classes if item.member_rxcuis]

    refined: list[ClassCandidate] = []
    for item in candidates:
        if item.class_type.upper() != "MOA" or not item.discovered_by_drug:
            refined.append(item)
            continue
        normalized_name = normalize_name(item.class_name)
        secondary = any(pattern in normalized_name for pattern in SECONDARY_MOA_PATTERNS)
        name_overlap = bool(_pharmacologic_tokens(item.class_name) & primary_tokens)
        member_set = set(item.member_rxcuis)
        member_overlap = max(
            (len(member_set & primary_set) for primary_set in primary_members),
            default=0,
        )
        analysis_ready = (
            item.direct_membership
            and not item.combination_membership
            and item.member_count is not None
            and item.member_count >= 2
            and "has_moa" in {relation.casefold() for relation in item.relations}
            and not secondary
            and (name_overlap or member_overlap >= 2)
        )
        if analysis_ready:
            refined.append(
                item.model_copy(
                    update={
                        "recommended_for_analysis": True,
                        "membership_category": "recommended",
                        "why_shown": "Mechanism classification",
                    }
                )
            )
        else:
            reason = (
                "Secondary enzyme-inhibition classification"
                if "cytochrome p450" in normalized_name or normalized_name.startswith("cyp")
                else "Other mechanistic classification"
            )
            refined.append(
                item.model_copy(
                    update={
                        "recommended_for_analysis": False,
                        "membership_category": "other-mechanistic",
                        "why_shown": reason,
                    }
                )
            )
    return refined


def _get_rxnorm_properties(
    rxcui: str, settings: Settings
) -> tuple[DrugCandidate | None, SourceDetail]:
    payload, detail = _request_json(
        "RxNorm", RXNORM_PROPERTIES_URL.format(rxcui=rxcui), settings, allow_404=True
    )
    properties = (payload or {}).get("properties") or {}
    name = str(properties.get("name") or "").strip()
    if not name:
        return None, detail
    return (
        DrugCandidate(
            rxcui=str(properties.get("rxcui") or rxcui),
            name=name,
            tty=str(properties.get("tty") or ""),
            synonym=str(properties.get("synonym") or ""),
        ),
        detail,
    )


def search_drug_candidates(
    query: str, settings: Settings
) -> tuple[list[DrugCandidate], list[SourceDetail]]:
    clean_query = query.strip()
    if not clean_query:
        raise DataSourceError("RxNorm", "Enter a generic name, brand name, or RXCUI before searching.")
    if clean_query.isdigit():
        candidate, detail = _get_rxnorm_properties(clean_query, settings)
        return ([candidate] if candidate else []), [detail]

    payload, detail = _request_json(
        "RxNorm",
        RXNORM_APPROXIMATE_URL,
        settings,
        params={"term": clean_query, "maxEntries": 50, "option": 1},
    )
    raw = _as_list((payload or {}).get("approximateGroup", {}).get("candidate"))
    best_by_rxcui: dict[str, dict[str, Any]] = {}
    for item in raw:
        rxcui = str(item.get("rxcui") or "").strip()
        if not rxcui:
            continue
        current = best_by_rxcui.get(rxcui)
        if current is None or str(item.get("source") or "") == "RXNORM":
            best_by_rxcui[rxcui] = item

    ordered = sorted(
        best_by_rxcui.items(),
        key=lambda pair: (
            int(pair[1].get("rank") or 999999),
            -float(pair[1].get("score") or 0),
            pair[0],
        ),
    )[:15]
    details = [detail]
    candidates: list[DrugCandidate] = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(ordered)))) as executor:
        futures = {
            executor.submit(_get_rxnorm_properties, rxcui, settings): (rxcui, item)
            for rxcui, item in ordered
        }
        resolved: dict[str, DrugCandidate] = {}
        for future in as_completed(futures):
            rxcui, item = futures[future]
            try:
                candidate, item_detail = future.result()
                details.append(item_detail)
            except DataSourceError:
                candidate = None
            if candidate:
                resolved[rxcui] = candidate.model_copy(
                    update={
                        "score": float(item.get("score") or 0),
                        "rank": int(item.get("rank") or 0) or None,
                    }
                )
        candidates = [resolved[rxcui] for rxcui, _ in ordered if rxcui in resolved]
        candidates.sort(
            key=lambda item: (
                item.rank or 999999,
                0 if item.tty.upper() == "IN" else 1 if item.tty.upper() in {"PIN", "MIN"} else 2,
                -(item.score or 0),
                item.name.casefold(),
                item.rxcui,
            )
        )
    detail.record_count = len(candidates)
    return candidates, details


def normalize_drug_name(
    query: str, settings: Settings
) -> tuple[list[DrugCandidate], list[SourceDetail]]:
    clean_query = query.strip()
    if not clean_query:
        return [], []
    payload, detail = _request_json(
        "RxNorm",
        RXNORM_RXCUI_URL,
        settings,
        params={"name": clean_query, "search": 2, "allsrc": 0},
    )
    rxcuis = [str(value).strip() for value in _as_list((payload or {}).get("idGroup", {}).get("rxnormId"))]
    rxcuis = [value for value in dict.fromkeys(rxcuis) if value]
    details = [detail]
    candidates: list[DrugCandidate] = []
    for rxcui in rxcuis[:10]:
        try:
            candidate, property_detail = _get_rxnorm_properties(rxcui, settings)
            details.append(property_detail)
        except DataSourceError:
            candidate = None
        if candidate:
            candidates.append(candidate.model_copy(update={"rank": 1, "score": 100.0}))
    candidates.sort(
        key=lambda item: (
            0 if item.tty.upper() == "IN" else 1 if item.tty.upper() in {"PIN", "MIN"} else 2,
            item.name.casefold(),
            item.rxcui,
        )
    )
    detail.record_count = len(candidates)
    detail.note = "RxNorm exact-or-normalized name lookup (search=2)."
    return candidates, details


TECHNICAL_CLASS_TYPES = {"CHEM", "VA", "STRUCT", "DISPOS", "SCHEDULE", "PK"}
PRODUCT_TTYS = {"SCD", "SBD", "SCDG", "SBDG", "SCDF", "SBDF", "GPCK", "BPCK"}


def _is_recommended_relationship(candidate: ClassCandidate) -> bool:
    normalized = candidate.class_type.upper()
    relations = {value.casefold() for value in candidate.relations}
    if normalized == "EPC":
        return "has_epc" in relations
    if normalized == "MOA":
        return "has_moa" in relations
    if normalized.startswith("ATC"):
        return candidate.rela_source.upper() == "ATC"
    return False


def _is_combination_member(concept: dict[str, Any]) -> bool:
    tty = str(concept.get("tty") or "").upper()
    raw_name = str(concept.get("name") or "")
    return tty == "MIN" or tty in {"GPCK", "BPCK"} or " / " in raw_name


def _is_direct_ingredient_member(concept: dict[str, Any], drug: DrugCandidate) -> bool:
    rxcui = str(concept.get("rxcui") or "").strip()
    tty = str(concept.get("tty") or "").upper()
    if rxcui == drug.rxcui:
        return True
    if tty not in {"IN", "PIN"}:
        return False
    drug_tokens = set(normalize_name(drug.name).split())
    member_tokens = set(normalize_name(str(concept.get("name") or "")).split())
    return bool(drug_tokens) and drug_tokens.issubset(member_tokens)


def _why_class_is_shown(
    class_type: str, relations: list[str], *, direct: bool, combination: bool
) -> str:
    normalized = class_type.upper()
    normalized_relations = {relation.casefold() for relation in relations}
    if normalized == "PE":
        return "Physiologic effect relationship"
    if normalized == "STRUCT":
        return "SNOMED structural relationship"
    if normalized == "DISPOS":
        return "SNOMED disposition relationship"
    if normalized == "CHEM":
        return "Chemical classification"
    if normalized == "VA":
        return "Extended VA relationship"
    if combination:
        return "Appears through combination products"
    if normalized == "DISEASE":
        labels = [
            label
            for relation, label in [
                ("may_treat", "Treatment indication relationship"),
                ("may_prevent", "Prevention relationship"),
                ("ci_with", "Contraindication-related relationship"),
            ]
            if relation in normalized_relations
        ]
        if labels:
            return "; ".join(labels)
    if normalized == "MOA" and any(
        relation.casefold() == "has_moa" for relation in relations
    ):
        return "Mechanism classification"
    if normalized.startswith("ATC"):
        return "ATC therapeutic class"
    if any(relation.casefold() == "has_vaclass_extended" for relation in relations):
        return "Extended VA relationship"
    if direct:
        return "Direct ingredient membership"
    return "Extended VA relationship" if normalized == "VA" else "Direct ingredient membership"


RELA_SOURCE_PRIORITY = {
    "FDASPL": 0,
    "DAILYMED": 1,
    "MEDRT": 2,
    "ATC": 3,
    "ATCPROD": 4,
    "VA": 5,
    "SNOMEDCT": 6,
}


def _rela_source_sort_key(source: str) -> tuple[int, str]:
    return (RELA_SOURCE_PRIORITY.get(source.upper(), 50), source.casefold())


def get_classes_by_drug(
    drug: DrugCandidate, settings: Settings
) -> tuple[list[ClassCandidate], list[SourceDetail]]:
    payload, detail = _request_json(
        "RxClass",
        RXCLASS_BY_RXCUI_URL,
        settings,
        params={"rxcui": drug.rxcui},
    )
    raw = _as_list((payload or {}).get("rxclassDrugInfoList", {}).get("rxclassDrugInfo"))
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in raw:
        class_item = item.get("rxclassMinConceptItem") or {}
        concept = item.get("minConcept") or {}
        class_id = str(class_item.get("classId") or "").strip()
        class_name = str(class_item.get("className") or "").strip()
        class_type = str(class_item.get("classType") or "").strip()
        rela_source = str(item.get("relaSource") or "").strip()
        rela = str(item.get("rela") or "").strip() or None
        key = (class_id, class_type)
        if not class_id or not class_name:
            continue
        group = grouped.setdefault(
            key,
            {
                "class_name": class_name,
                "relations": set(),
                "sources": set(),
                "source_relations": {},
                "direct": False,
                "combination_count": 0,
                "single_product_count": 0,
                "direct_examples": [],
                "single_examples": [],
                "combination_examples": [],
            },
        )
        if rela_source:
            group["sources"].add(rela_source)
        if rela:
            group["relations"].add(rela)
            group["source_relations"].setdefault(rela_source, set()).add(rela)
        member_name = str(concept.get("name") or "").strip()
        direct = _is_direct_ingredient_member(concept, drug)
        combination = _is_combination_member(concept)
        tty = str(concept.get("tty") or "").upper()
        if direct:
            group["direct"] = True
            if member_name and member_name not in group["direct_examples"]:
                group["direct_examples"].append(member_name)
        elif combination:
            group["combination_count"] += 1
            if member_name and member_name not in group["combination_examples"]:
                group["combination_examples"].append(member_name)
        elif tty in PRODUCT_TTYS:
            group["single_product_count"] += 1
            if member_name and member_name not in group["single_examples"]:
                group["single_examples"].append(member_name)

    candidates: list[ClassCandidate] = []
    for (class_id, class_type), group in grouped.items():
        sources = sorted(group["sources"], key=_rela_source_sort_key)
        preferred_source = "FDASPL" if "FDASPL" in sources else (sources[0] if sources else "")
        additional_sources = [source for source in sources if source != preferred_source]
        relations = sorted(
            group["relations"],
            key=lambda value: (value.casefold().endswith("extended"), value.casefold()),
        )
        direct = bool(group["direct"])
        combination_only = (
            not direct
            and group["combination_count"] > 0
            and group["single_product_count"] == 0
        )
        normalized_type = class_type.upper()
        if combination_only:
            category = "combination"
        elif normalized_type in TECHNICAL_CLASS_TYPES:
            category = "technical"
        else:
            category = "other"
        examples = (
            group["direct_examples"]
            + group["single_examples"]
            + group["combination_examples"]
        )[:5]
        preferred_relations = sorted(
            group["source_relations"].get(preferred_source, set()),
            key=lambda value: (value.casefold().endswith("extended"), value.casefold()),
        )
        candidates.append(
            ClassCandidate(
                class_name=group["class_name"],
                class_id=class_id,
                class_type=class_type,
                rela_source=preferred_source,
                rela=(preferred_relations[0] if preferred_relations else relations[0])
                if relations
                else None,
                relations=relations,
                rela_sources=sources,
                preferred_source=preferred_source,
                additional_sources=additional_sources,
                why_shown=_why_class_is_shown(
                    class_type, relations, direct=direct, combination=combination_only
                ),
                membership_category=category,
                direct_membership=direct,
                combination_membership=combination_only,
                recommended_for_analysis=False,
                discovered_by_drug=True,
                membership_examples=examples,
                match_strength="official drug relationship",
            )
        )
    candidates.sort(
        key=lambda item: (
            0
            if item.direct_membership and item.class_type.upper() == "EPC"
            else 1
            if item.direct_membership and item.class_type.upper() == "MOA"
            else 2
            if item.direct_membership and item.class_type.upper().startswith("ATC")
            else 3
            if item.direct_membership and item.class_type.upper() not in TECHNICAL_CLASS_TYPES
            else 4
            if item.membership_category == "technical"
            else 5
            if item.membership_category == "combination"
            else 4,
            item.class_name.casefold(),
            item.rela_source,
            ",".join(item.relations),
        )
    )
    detail.record_count = len(candidates)
    return candidates, [detail]


def search_drug_classes(
    class_name: str, settings: Settings
) -> tuple[list[ClassCandidate], list[SourceDetail]]:
    clean_name = class_name.strip()
    if not clean_name:
        raise DataSourceError("RxClass", "Enter a drug class name before searching.")

    payload, detail = _request_json(
        "RxClass",
        RXCLASS_SEARCH_URL,
        settings,
        params={"className": clean_name},
    )
    raw_candidates = _as_list(
        (payload or {}).get("rxclassMinConceptList", {}).get("rxclassMinConcept")
    )

    candidates: list[ClassCandidate] = []
    for item in raw_candidates:
        class_type = str(item.get("classType") or "")
        source, rela = _relation_options(class_type)[0]
        candidates.append(
            ClassCandidate(
                class_name=str(item.get("className") or ""),
                class_id=str(item.get("classId") or ""),
                class_type=class_type,
                rela_source=str(item.get("relaSource") or source),
                rela=rela,
            )
        )

    detail.record_count = len(candidates)
    if not candidates:
        raise DataSourceError(
            "RxClass",
            f'RxClass returned no class for "{clean_name}". Try the official class name or a different spelling.',
        )
    return candidates, [detail]


def get_class_members(
    candidate: ClassCandidate, settings: Settings
) -> tuple[list[DrugMember], ClassCandidate, list[SourceDetail]]:
    attempts: list[tuple[str, str | None]] = []
    if candidate.rela_source and candidate.rela_source != "Resolve on selection":
        attempts.append((candidate.rela_source, candidate.rela))
    for option in _relation_options(candidate.class_type):
        if option not in attempts:
            attempts.append(option)

    details: list[SourceDetail] = []
    last_error: DataSourceError | None = None
    for rela_source, rela in attempts:
        params: dict[str, Any] = {
            "classId": candidate.class_id,
            "relaSource": rela_source,
            "ttys": "IN PIN MIN",
        }
        if rela:
            params["rela"] = rela
        try:
            payload, detail = _request_json(
                "RxClass", RXCLASS_MEMBERS_URL, settings, params=params
            )
        except DataSourceError as exc:
            last_error = exc
            continue

        raw_members = _as_list((payload or {}).get("drugMemberGroup", {}).get("drugMember"))
        members: list[DrugMember] = []
        seen: set[tuple[str, str]] = set()
        for item in raw_members:
            concept = item.get("minConcept") or {}
            tty = str(concept.get("tty") or "").upper()
            rxcui = str(concept.get("rxcui") or "").strip()
            name = str(concept.get("name") or "").strip()
            if tty not in ALLOWED_TTYS or not rxcui or not name:
                continue
            key = (rxcui, normalize_name(name))
            if key in seen:
                continue
            seen.add(key)
            members.append(DrugMember(rxcui=rxcui, name=name, tty=tty))

        detail.record_count = len(members)
        details.append(detail)
        if members:
            resolved = candidate.model_copy(
                update={"rela_source": rela_source, "rela": rela}
            )
            members.sort(key=lambda member: (member.name.casefold(), member.rxcui))
            return members, resolved, details

    if last_error and not details:
        raise last_error
    raise DataSourceError(
        "RxClass",
        f'RxClass returned no IN, PIN, or MIN ingredients for "{candidate.class_name}".',
    )


UUID_PATTERN = re.compile(
    r"/dataset/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:/|$)",
    re.IGNORECASE,
)


def _extract_dataset_uuid(url: str) -> str:
    match = UUID_PATTERN.search(url or "")
    return match.group(1) if match else ""


def resolve_cms_dataset(
    settings: Settings,
) -> tuple[CmsDatasetIdentity, list[SourceDetail]]:
    configured_type = settings.cms_dataset_id.strip()
    configured_version = settings.cms_dataset_version_id.strip()
    if configured_type and configured_version:
        identity = CmsDatasetIdentity(
            dataset_type_id=configured_type,
            dataset_version_id=configured_version,
            year=settings.cms_data_year,
        )
        detail = SourceDetail(
            source="CMS",
            query="Environment configuration (CMS identifiers; no secret values)",
            record_count=1,
            note="Dataset identifiers loaded from .env.",
        )
        return identity, [detail]

    payload, detail = _request_json("CMS", CMS_CATALOG_URL, settings)
    datasets = _as_list((payload or {}).get("dataset"))
    dataset = next(
        (item for item in datasets if item.get("title") == CMS_DATASET_TITLE),
        None,
    )
    if not dataset:
        raise DataSourceError("CMS", f'CMS catalog does not contain "{CMS_DATASET_TITLE}".')

    dataset_type_id = configured_type or _extract_dataset_uuid(str(dataset.get("identifier") or ""))
    version_candidates: list[str] = []
    for distribution in _as_list(dataset.get("distribution")):
        if str(distribution.get("format") or "").upper() != "API":
            continue
        temporal = str(distribution.get("temporal") or "")
        if not temporal.startswith(str(settings.cms_data_year)):
            continue
        candidate_id = _extract_dataset_uuid(str(distribution.get("accessURL") or ""))
        if candidate_id and candidate_id != dataset_type_id:
            version_candidates.append(candidate_id)

    dataset_version_id = configured_version or (version_candidates[0] if version_candidates else "")
    if not dataset_type_id or not dataset_version_id:
        raise DataSourceError(
            "CMS",
            f"CMS catalog did not expose both dataset identifiers for {settings.cms_data_year}.",
        )

    detail.record_count = 1
    detail.note = (
        f"Resolved live: type={dataset_type_id}, version={dataset_version_id}, "
        f"year={settings.cms_data_year}."
    )
    return (
        CmsDatasetIdentity(
            dataset_type_id=dataset_type_id,
            dataset_version_id=dataset_version_id,
            year=settings.cms_data_year,
        ),
        [detail],
    )


def normalize_name(value: str) -> str:
    value = value.casefold().replace("–", "-").replace("—", "-")
    value = re.sub(r"[-_/,+()]", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(value.split())


SALT_WORDS = {
    "acetate",
    "anhydrous",
    "besylate",
    "bitartrate",
    "calcium",
    "citrate",
    "dihydrate",
    "disodium",
    "fumarate",
    "gluconate",
    "hcl",
    "hemifumarate",
    "hemihydrate",
    "hydrobromide",
    "hydrochloride",
    "lactate",
    "maleate",
    "mesylate",
    "monohydrate",
    "pamoate",
    "phosphate",
    "potassium",
    "propanediol",
    "sodium",
    "succinate",
    "sulfate",
    "tartrate",
}


def _match_cms_name(member: DrugMember, cms_name: str) -> tuple[str, bool, str] | None:
    candidate = normalize_name(member.name)
    generic = normalize_name(cms_name)
    if not candidate or not generic:
        return None
    if candidate == generic:
        return "exact", False, "Exact normalized generic-name match."

    candidate_tokens = candidate.split()
    generic_tokens = generic.split()
    candidate_core = [token for token in candidate_tokens if token not in SALT_WORDS]
    generic_core = [token for token in generic_tokens if token not in SALT_WORDS]

    is_cms_combination = "/" in cms_name or "+" in cms_name
    if is_cms_combination and member.tty != "MIN":
        return None

    if member.tty == "IN" and candidate_tokens == candidate_core:
        if candidate_core == generic_core and len(generic_tokens) > len(generic_core):
            return (
                "salt-normalized",
                True,
                f'Approximate salt-normalized match: RxClass "{member.name}" to CMS "{cms_name}".',
            )

    if member.tty == "MIN" and sorted(candidate_core) == sorted(generic_core):
        return (
            "combination-normalized",
            True,
            f'Approximate combination/salt normalization: RxClass "{member.name}" to CMS "{cms_name}".',
        )
    return None


def _to_int(value: Any) -> int:
    try:
        return int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _fetch_cms_candidate(
    member: DrugMember,
    identity: CmsDatasetIdentity,
    settings: Settings,
) -> tuple[DrugMember, list[dict[str, Any]], list[SourceDetail]]:
    url = f"https://data.cms.gov/data-api/v1/dataset/{identity.dataset_version_id}/data"
    page_size = 500
    offset = 0
    rows: list[dict[str, Any]] = []
    details: list[SourceDetail] = []
    while True:
        payload, detail = _request_json(
            "CMS",
            url,
            settings,
            params={
                "keyword": member.name,
                "filter[Prscrbr_Geo_Lvl]": "National",
                "size": page_size,
                "offset": offset,
            },
        )
        page = payload if isinstance(payload, list) else []
        rows.extend(page)
        detail.record_count = len(page)
        details.append(detail)
        if len(page) < page_size:
            break
        offset += page_size
    return member, rows, details


def fetch_cms_usage(
    members: Iterable[DrugMember], settings: Settings
) -> tuple[list[CmsUsage], CmsDatasetIdentity, list[SourceDetail]]:
    member_list = list(members)
    identity, details = resolve_cms_dataset(settings)
    if not member_list:
        return [], identity, details

    fetched: list[tuple[DrugMember, list[dict[str, Any]], list[SourceDetail]]] = []
    max_workers = min(6, len(member_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_cms_candidate, member, identity, settings): member
            for member in member_list
        }
        for future in as_completed(futures):
            fetched.append(future.result())

    usage_rows: list[CmsUsage] = []
    for member, rows, row_details in fetched:
        details.extend(row_details)
        matched: list[tuple[dict[str, Any], tuple[str, bool, str]]] = []
        for row in rows:
            generic_name = str(row.get("Gnrc_Name") or "").strip()
            match = _match_cms_name(member, generic_name)
            if match:
                matched.append((row, match))
        if not matched:
            continue

        generic_names = sorted(
            {str(row.get("Gnrc_Name") or "").strip() for row, _ in matched},
            key=str.casefold,
        )
        total_claims = sum(_to_int(row.get("Tot_Clms")) for row, _ in matched)
        beneficiary_values = [
            _to_int(row.get("Tot_Benes"))
            for row, _ in matched
            if row.get("Tot_Benes") not in (None, "")
        ]
        approximate = any(match[1] for _, match in matched)
        match_notes = sorted({match[2] for _, match in matched})
        qualities = {match[0] for _, match in matched}
        quality = "exact" if qualities == {"exact"} else ", ".join(sorted(qualities))
        usage_rows.append(
            CmsUsage(
                member=member,
                cms_generic_names=generic_names,
                total_claims=total_claims,
                total_beneficiaries=sum(beneficiary_values) if beneficiary_values else None,
                data_year=identity.year,
                match_quality=quality,
                approximate_match=approximate,
                match_note=" ".join(match_notes),
            )
        )

    usage_rows.sort(
        key=lambda item: (
            -item.total_claims,
            -(item.total_beneficiaries or 0),
            item.member.name.casefold(),
        )
    )
    for rank, usage in enumerate(usage_rows, start=1):
        usage.rank = rank
    return usage_rows, identity, details


def _lucene_escape(value: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", value)


def _label_names(label: dict[str, Any], field: str) -> list[str]:
    return [str(value) for value in _as_list((label.get("openfda") or {}).get(field))]


def _join_section(label: dict[str, Any], field: str) -> str:
    values = [str(value).strip() for value in _as_list(label.get(field)) if str(value).strip()]
    return "\n\n".join(values)


def _extract_spl_set_id(label: dict[str, Any]) -> tuple[str | None, str | None]:
    openfda = label.get("openfda") or {}
    locations = [
        ("root.spl_set_id", label.get("spl_set_id")),
        ("root.set_id", label.get("set_id")),
        ("openfda.spl_set_id", openfda.get("spl_set_id")),
        ("openfda.set_id", openfda.get("set_id")),
    ]
    for location, value in locations:
        values = [str(item).strip() for item in _as_list(value) if str(item).strip()]
        if values:
            return values[0], location
    return None, None


def _ingredient_core(value: str) -> str:
    return " ".join(
        token for token in normalize_name(value).split() if token not in SALT_WORDS
    )


OPENFDA_SAFETY_FIELDS = [
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "contraindications",
    "drug_interactions",
]


def _score_openfda_label(
    label: dict[str, Any], usage: CmsUsage
) -> dict[str, Any]:
    target = normalize_name(usage.member.name)
    target_core = _ingredient_core(usage.member.name)
    generic_names = _label_names(label, "generic_name")
    substance_names = _label_names(label, "substance_name")
    generic_normalized = {normalize_name(name) for name in generic_names}
    substance_normalized = {normalize_name(name) for name in substance_names}
    label_product_names = generic_normalized | substance_normalized
    cms_names = {normalize_name(name) for name in usage.cms_generic_names}
    label_rxcuis = set(_label_names(label, "rxcui"))

    exact_rxcui = usage.member.rxcui in label_rxcuis
    exact_generic = target in generic_normalized
    exact_substance = target in substance_normalized
    cms_product_match = bool(cms_names & label_product_names)
    core_match = bool(target_core) and any(
        _ingredient_core(name) == target_core
        for name in generic_names + substance_names
    )
    safety_fields = [field for field in OPENFDA_SAFETY_FIELDS if _join_section(label, field)]
    spl_set_id, spl_location = _extract_spl_set_id(label)

    score = min(
        100,
        (35 if exact_rxcui else 0)
        + (20 if exact_generic else 0)
        + (15 if exact_substance else 0)
        + (25 if cms_product_match else 0)
        + (20 if core_match else 0)
        + round(10 * len(safety_fields) / len(OPENFDA_SAFETY_FIELDS))
        + (5 if spl_set_id else 0),
    )

    reasons: list[str] = []
    reasons.append(
        "Exact RxClass RXCUI match in openFDA."
        if exact_rxcui
        else "No exact base-ingredient RXCUI was present in this product label."
    )
    if exact_generic:
        reasons.append("Exact openFDA generic-name match to the RxClass base ingredient.")
    if exact_substance:
        reasons.append("Exact openFDA substance-name match to the RxClass base ingredient.")
    if cms_product_match:
        matched = sorted(cms_names & label_product_names)
        reasons.append(f"FDA label product matches a CMS product/salt: {', '.join(matched)}.")
    elif core_match:
        reasons.append("Active-ingredient core matches, but CMS salt/product identity is not exact.")
    else:
        reasons.append("Product/salt compatibility with the CMS matches is uncertain.")
    reasons.append(
        f"Safety coverage: {len(safety_fields)}/{len(OPENFDA_SAFETY_FIELDS)} requested sections present"
        + (f" ({', '.join(safety_fields)})." if safety_fields else ".")
    )
    if spl_set_id:
        reasons.append(f"SPL set identifier found at {spl_location}.")
    else:
        reasons.append("SPL set identifier was unavailable in the inspected record fields.")

    strong_identity = exact_rxcui or (core_match and cms_product_match) or (
        exact_generic and exact_substance
    )
    if strong_identity and len(safety_fields) >= 3:
        confidence = "high"
    elif (exact_rxcui or exact_generic or exact_substance or core_match) and len(safety_fields) >= 2:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "sort_key": (
            exact_rxcui,
            exact_generic,
            exact_substance,
            cms_product_match,
            core_match,
            len(safety_fields),
            bool(spl_set_id),
            str(label.get("effective_time") or ""),
        ),
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "spl_set_id": spl_set_id,
        "spl_location": spl_location,
        "safety_fields": safety_fields,
    }


def test_openfda_connection(
    settings: Settings,
) -> tuple[str, SourceDetail]:
    api_key = settings.openfda_api_key.strip()
    payload, detail = _request_json(
        "openFDA",
        OPENFDA_LABEL_URL,
        settings,
        params={
            "limit": 1,
            **({"api_key": api_key} if api_key else {}),
        },
    )
    if not _as_list((payload or {}).get("results")):
        raise DataSourceError("openFDA", "openFDA connection failed: no response records returned.")
    status = (
        "Connected — API key configured"
        if api_key
        else "Connected — unauthenticated, public rate limits apply"
    )
    detail.note = status
    return status, detail


def fetch_openfda_label(
    usage: CmsUsage, settings: Settings
) -> tuple[LabelEvidence | None, list[SourceDetail]]:
    api_key = settings.openfda_api_key.strip()
    search_specs = [
        ("openfda.rxcui", usage.member.rxcui),
        ("openfda.generic_name", usage.member.name),
        ("openfda.substance_name", usage.member.name),
        ("openfda.brand_name", usage.member.name),
    ]
    details: list[SourceDetail] = []
    label_records: dict[str, tuple[dict[str, Any], str, int]] = {}
    target = normalize_name(usage.member.name)

    for field, search_value in search_specs:
        payload, detail = _request_json(
            "openFDA",
            OPENFDA_LABEL_URL,
            settings,
            params={
                "search": f'{field}:"{_lucene_escape(search_value)}"',
                "sort": "effective_time:desc",
                "limit": 100,
                **({"api_key": api_key} if api_key else {}),
            },
            allow_404=True,
        )
        details.append(detail)
        results = _as_list((payload or {}).get("results"))
        if results:
            query_total = _to_int(
                (payload or {}).get("meta", {}).get("results", {}).get("total")
            ) or len(results)
            detail.record_count = len(results)
            for index, record in enumerate(results):
                openfda = record.get("openfda") or {}
                record_key = str(
                    record.get("id")
                    or (_as_list(openfda.get("spl_id")) or [""])[0]
                    or f"{field}:{index}:{record.get('effective_time', '')}"
                )
                label_records.setdefault(record_key, (record, field, query_total))

            if field == "openfda.rxcui":
                break
            if field == "openfda.generic_name" and any(
                target in {normalize_name(name) for name in _label_names(record, "generic_name")}
                for record in results
            ):
                break
            if field == "openfda.substance_name" and any(
                target in {normalize_name(name) for name in _label_names(record, "substance_name")}
                for record in results
            ):
                break
            if field == "openfda.brand_name":
                break

    if not label_records:
        return None, details

    scored = [
        (record, _score_openfda_label(record, usage), field, query_total)
        for record, field, query_total in label_records.values()
    ]
    scored.sort(key=lambda item: item[1]["sort_key"], reverse=True)
    selected, selected_score, query_field, total_matched = scored[0]
    selected_time = str(selected.get("effective_time") or "") or None
    reason = (
        "Selected by ordered evidence priorities: exact RXCUI, exact generic name, exact "
        "substance name, CMS salt/product compatibility, safety-section coverage, SPL set "
        f"identifier, then effective_time. Evaluated {len(label_records)} distinct returned "
        f"record(s); the selected query field has {total_matched} total match(es)."
    )

    section_fields = [
        "boxed_warning",
        "warnings",
        "warnings_and_cautions",
        "adverse_reactions",
        "contraindications",
        "drug_interactions",
        "indications_and_usage",
    ]
    sections = {
        field: text
        for field in section_fields
        if (text := _join_section(selected, field))
    }
    evidence = LabelEvidence(
        rxcui=usage.member.rxcui,
        requested_name=usage.member.name,
        query_field=query_field,
        matched_label_count=total_matched,
        selected_spl_set_id=selected_score["spl_set_id"],
        effective_time=selected_time,
        selection_reason=reason,
        label_match_score=selected_score["score"],
        label_match_confidence=selected_score["confidence"],
        label_match_reasons=selected_score["reasons"],
        selected_label_product={
            "generic_names": _label_names(selected, "generic_name"),
            "brand_names": _label_names(selected, "brand_name"),
            "substance_names": _label_names(selected, "substance_name"),
            "rxcui_values": _label_names(selected, "rxcui"),
            "routes": _label_names(selected, "route"),
            "dosage_forms": _label_names(selected, "dosage_form"),
            "product_ndc_values": _label_names(selected, "product_ndc"),
            "package_ndc_values": _label_names(selected, "package_ndc"),
            "manufacturer_names": _label_names(selected, "manufacturer_name"),
            "spl_set_id_source": selected_score["spl_location"],
        },
        generic_names=_label_names(selected, "generic_name"),
        brand_names=_label_names(selected, "brand_name"),
        substance_names=_label_names(selected, "substance_name"),
        label_rxcuis=_label_names(selected, "rxcui"),
        pharm_class_epc=_label_names(selected, "pharm_class_epc"),
        pharm_class_moa=_label_names(selected, "pharm_class_moa"),
        sections=sections,
    )
    return evidence, details


def supabase_configuration_status(settings: Settings) -> str:
    if not settings.supabase_url or not settings.supabase_secret_key:
        return "Not configured"
    return "Configured"


def test_supabase(settings: Settings) -> tuple[str, SourceDetail | None]:
    if supabase_configuration_status(settings) == "Not configured":
        return "Not configured", None
    try:
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_secret_key)
        response = client.table("analysis_runs").select("id").limit(1).execute()
        count = len(response.data or [])
        return (
            "Connected",
            SourceDetail(
                source="Supabase",
                query=f"{settings.supabase_url.rstrip('/')}/rest/v1/analysis_runs?select=id&limit=1",
                record_count=count,
                note="Connection tested with the server-side secret key; key omitted.",
            ),
        )
    except Exception as exc:  # Supabase wraps transport/PostgREST errors in several classes.
        message = _safe_error_text(str(exc), settings)
        raise DataSourceError("Supabase", f"Supabase connection failed: {message}") from exc


def load_drug_class_catalog(
    settings: Settings,
) -> tuple[list[ClassCandidate], SourceDetail]:
    if supabase_configuration_status(settings) == "Not configured":
        raise DataSourceError("Supabase", "Supabase is not configured.")
    try:
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_secret_key)
        rows: list[dict[str, Any]] = []
        page_size = 1000
        for offset in range(0, 50000, page_size):
            response = (
                client.table("drug_class_catalog")
                .select(
                    "class_id,class_name,class_type,rela_source,rela,member_count,"
                    "example_members,source_updated_at,synced_at"
                )
                .range(offset, offset + page_size - 1)
                .execute()
            )
            page = response.data or []
            rows.extend(page)
            if len(page) < page_size:
                break
        candidates = [
            ClassCandidate(
                class_id=str(row.get("class_id") or ""),
                class_name=str(row.get("class_name") or ""),
                class_type=str(row.get("class_type") or ""),
                rela_source=str(row.get("rela_source") or "Resolve on selection"),
                rela=str(row.get("rela") or "") or None,
                member_count=row.get("member_count"),
                example_members=[str(value) for value in _as_list(row.get("example_members"))],
            )
            for row in rows
            if row.get("class_id") and row.get("class_name")
        ]
        if not candidates:
            raise DataSourceError("Supabase", "The drug class catalog cache is empty.")
        return candidates, SourceDetail(
            source="Supabase",
            query=f"{settings.supabase_url.rstrip('/')}/rest/v1/drug_class_catalog [SELECT; secret omitted]",
            record_count=len(candidates),
            note="Server-side RxClass cache loaded; RxClass remains the authoritative source.",
        )
    except DataSourceError:
        raise
    except Exception as exc:
        message = _safe_error_text(str(exc), settings)
        raise DataSourceError("Supabase", f"Class catalog cache read failed: {message}") from exc


def sync_drug_class_catalog(
    catalog: list[ClassCandidate], settings: Settings
) -> SourceDetail:
    if supabase_configuration_status(settings) == "Not configured":
        raise DataSourceError("Supabase", "Supabase is not configured.")
    try:
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_secret_key)
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "class_type": item.class_type,
                "rela_source": item.rela_source or "Resolve on selection",
                "rela": item.rela or "",
                "member_count": item.member_count,
                "example_members": item.example_members[:5],
                "search_text": " ".join(
                    part
                    for part in [
                        normalize_name(item.class_name),
                        normalize_name(item.class_id),
                        normalize_name(item.class_type),
                        normalize_name(item.rela_source),
                    ]
                    if part
                ),
                "source_updated_at": now,
                "synced_at": now,
            }
            for item in catalog
        ]
        written = 0
        for start in range(0, len(rows), 500):
            batch = rows[start : start + 500]
            response = (
                client.table("drug_class_catalog")
                .upsert(batch, on_conflict="class_id,rela_source,rela")
                .execute()
            )
            written += len(response.data or batch)
        return SourceDetail(
            source="Supabase",
            query=f"{settings.supabase_url.rstrip('/')}/rest/v1/drug_class_catalog [UPSERT; secret omitted]",
            record_count=written,
            note="Official RxClass catalog cache synchronized with the server-side secret key.",
        )
    except Exception as exc:
        message = _safe_error_text(str(exc), settings)
        raise DataSourceError("Supabase", f"Class catalog cache sync failed: {message}") from exc


def store_analysis_run(result: AnalysisResult, settings: Settings) -> SourceDetail:
    if supabase_configuration_status(settings) == "Not configured":
        raise DataSourceError("Supabase", "Supabase is not configured.")
    try:
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_secret_key)
        result_json = result.model_dump(mode="json")
        payload = {
            "class_name": result.selected_class.class_name,
            "class_id": result.selected_class.class_id,
            "cms_year": result.cms_dataset.year if result.cms_dataset else None,
            "selected_drugs": [item.model_dump(mode="json") for item in result.selected_drugs],
            "results_json": result_json,
            "source_status": result.source_status,
        }
        response = client.table("analysis_runs").insert(payload).execute()
        return SourceDetail(
            source="Supabase",
            query=f"{settings.supabase_url.rstrip('/')}/rest/v1/analysis_runs [INSERT; secret omitted]",
            retrieved_at=datetime.now(timezone.utc),
            record_count=len(response.data or []),
            note="Final analysis run stored.",
        )
    except Exception as exc:  # Keep UI free of raw tracebacks and secrets.
        message = _safe_error_text(str(exc), settings)
        raise DataSourceError("Supabase", f"Supabase insert failed: {message}") from exc
