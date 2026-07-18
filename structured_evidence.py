from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from models import CmsUsage, EventEvidenceItem, LabelEvidence, SPLSectionChunk
from spl_parser import TARGET_SECTION_CODES, section_key_for_code
from spl_qa import normalize_chunk_text


EXTRACTION_RULES_VERSION = "structured-event-evidence-v1"
SEARCHABLE_CHUNK_TYPES = {"paragraph", "list_item", "table_row"}
REVIEWED_SECTION_KEYS = {
    "boxed_warning",
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "contraindications",
    "drug_interactions",
    "use_in_specific_populations",
}
OPENFDA_SECTION_CODES = {
    section_key: section_code
    for section_code, (section_key, _) in TARGET_SECTION_CODES.items()
}
OPENFDA_SECTION_CODES["warnings"] = OPENFDA_SECTION_CODES["warnings_and_cautions"]

INTERACTION_PATTERNS = [
    r"\bconcomitant use\b",
    r"\bconcomitant\s+[a-z0-9-]+\b",
    r"\bcoadministration\b",
    r"\bwhen combined with\b",
    r"\bincreases? the risk\b",
    r"\bwith CYP(?:\d\w*)? inhibitors?\b",
    r"\bwith MAOIs?\b",
    r"\bwith other serotonergic drugs?\b",
    r"\bpatients taking\b[^.]{0,100}\band a drug\b",
    r"\bMAOIs?\b",
    r"\bserotonergic drugs?\b",
    r"\bconcomitantly with\b",
    r"\bwith the combination of\b",
    r"\b(?:the\s+)?risk\s+(?:is\s+)?increased with\b",
]
CORE_INTERACTION_PATTERNS = [
    INTERACTION_PATTERNS[index]
    for index in [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13]
]
POPULATION_PATTERNS = [
    r"\belderly patients?\b",
    r"\bpediatric patients?\b",
    r"\b(?:patients|subjects) with heart failure\b",
    r"\brenal impairment\b",
    r"\bhepatic impairment\b",
    r"\bpregnan(?:cy|t women)\b",
    r"\bhypertensive subjects?\b",
    r"\b(?:patients|subjects) with myocardial infarction\b",
]
DOSE_PATTERNS = [
    r"\bdose-related\b",
    r"\bat doses? above\b[^.;|]*",
    r"\bduring dose escalation\b",
]
TEMPORAL_PATTERNS = [
    r"\bafter abrupt withdrawal\b",
    r"\bwithin the first \d+ days?\b",
    r"\bduring (?:[a-z0-9-]+\s+){0,3}initiation\b",
    r"\bpostmarketing\b",
]
ROUTE_CLUE_PATTERNS = [
    ("oral", "oral", r"\boral(?:ly)?\b"),
    ("topical", "topical", r"\btopical(?:ly)?\b"),
    ("gel", "topical", r"\bgels?\b"),
    ("cream", "topical", r"\bcreams?\b"),
    ("ophthalmic", "ophthalmic", r"\bophthalmic(?:ally)?\b"),
    ("inhaled", "inhaled", r"\b(?:inhaled|inhalation)\b"),
    ("intravenous", "intravenous", r"\bintravenous(?:ly)?\b"),
    ("intramuscular", "intramuscular", r"\bintramuscular(?:ly)?\b"),
    ("subcutaneous", "subcutaneous", r"\bsubcutaneous(?:ly)?\b"),
    ("transdermal", "transdermal", r"\btransdermal(?:ly)?\b"),
    ("tablet", "oral", r"\btablets?\b"),
    ("capsule", "oral", r"\bcapsules?\b"),
]
ROUTE_NAMES = {
    "oral",
    "topical",
    "ophthalmic",
    "inhaled",
    "intravenous",
    "intramuscular",
    "subcutaneous",
    "transdermal",
}
FORMULATION_NAMES = {"gel", "cream", "tablet", "capsule"}
FREQUENCY_PATTERN = re.compile(
    r"\b(?:approximately|about)?\s*\d+(?:\.\d+)?\s*%"
    r"|\b\d+\s+of\s+\d[\d,]*\s+patients?\b"
    r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:cases?|events?)\s+per\s+\d[\d,]*\s+(?:patient-years?|person-years?|patients?)\b"
    r"|\b\d[\d,]*\s+(?:treated\s+)?(?:patients?|subjects?|participants?)\b"
    r"|(?<![\d.])\d+(?:\.\d+)?\s*/\s*\d[\d,]*(?:\.\d+)?(?![\d.])"
    r"(?:\s+(?:treated\s+patients?|patients?|subjects?|cases?)\b)?"
    r"|\bfrequency cannot be estimated\b"
    r"|\b(?:common|uncommon|rare|rarely)\b",
    re.IGNORECASE,
)
POPULATION_RATIO_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*/\s*\d[\d,]*(?:\.\d+)?\s+"
    r"(?:treated\s+patients?|patients?|subjects?|cases?)\b",
    re.IGNORECASE,
)
BLOOD_PRESSURE_CONTEXT_PATTERN = re.compile(
    r"(?:\bmm\s*hg\b|\bbp\b|\bblood[ -]pressure\b|\bsystolic\b|\bdiastolic\b"
    r"|\bhypotension\b|\bhypertension\b|\bless\s+than\b|\bgreater\s+than\b|[<>])",
    re.IGNORECASE,
)
DOSE_CONTEXT_PATTERN = re.compile(
    r"\b(?:mg|mcg|g)\b(?:\s*/\s*day|\s+daily)?"
    r"|\b(?:dose|dosage|tablet|capsule)\b"
    r"|\bfixed[ -]dose\s+combination\b"
    r"|\b[a-z][a-z0-9-]*\s*/\s*[a-z][a-z0-9-]*\b",
    re.IGNORECASE,
)
QUALITATIVE_FREQUENCY_TERMS = {"common", "uncommon", "rare", "rarely"}
NUMERIC_ROLES = {
    "product_strength",
    "administered_dose",
    "sample_size",
    "laboratory_or_vital_value",
    "event_frequency",
    "comparator_frequency",
    "ambiguous_numeric_value",
}
DOSAGE_FORM_PATTERN = (
    r"(?:"
    r"(?:chewable|orally\s+disintegrating|extended[- ]release|delayed[- ]release|vaginal)\s+tablet"
    r"|(?:delayed[- ]release|vaginal)\s+capsule"
    r"|(?:oral|topical|ophthalmic|otic|nasal|inhalation|injectable|intravenous)\s+(?:solution|suspension|powder)"
    r"|(?:ophthalmic|vaginal)\s+ointment"
    r"|(?:nasal|transdermal)\s+(?:spray|patch|system)"
    r"|(?:inhalation|rectal)\s+(?:aerosol|foam)"
    r"|(?:metered[- ]dose|dry[- ]powder)\s+inhaler"
    r"|(?:prefilled\s+)?syringe|autoinjector|nebulizer\s+solution"
    r"|tablets?|capsules?|caplets?|granules?|powders?|solutions?|suspensions?"
    r"|syrups?|elixirs?|drops?|concentrates?|creams?|ointments?|gels?|lotions?"
    r"|foams?|shampoos?|rinses?|pastes?|sprays?|aerosols?|inhalers?"
    r"|injections?|infusions?|vials?|ampules?|cartridges?|patches?"
    r"|suppositories?|enemas?|rings?|implants?|films?|lozenges?|troches?"
    r"|medicated\s+swabs?|intrauterine\s+systems?|transdermal\s+systems?"
    r")"
)
EVENT_FREQUENCY_CUE_PATTERN = re.compile(
    r"\b(?:occurred|occurs?|reported|observed|experienced|developed|incidence|frequency|rate|cases?)\b",
    re.IGNORECASE,
)
SAMPLE_SIZE_CUE_PATTERN = re.compile(
    r"\b(?:n\s*=|sample\s+size|study\s+population|total\s+of|randomi[sz]ed|enrolled|received\s+treatment|treated)\b",
    re.IGNORECASE,
)
LAB_OR_VITAL_PATTERN = re.compile(
    r"\b(?:blood[ -]pressure|systolic|diastolic|qtc?|heart\s+rate|pulse|temperature"
    r"|oxygen\s+saturation|spo2|ejection\s+fraction|glucose|creatinine|potassium"
    r"|sodium|hemoglobin|haemoglobin|hematocrit|alt|ast)\b",
    re.IGNORECASE,
)
PRODUCT_COMPOSITION_PATTERN = re.compile(
    r"\b(?:sodium\s+chloride|dextrose|lipid\s+emulsion|active\s+ingredient)\b",
    re.IGNORECASE,
)
HISTORICAL_PARENT_CONTEXT_PATTERN = re.compile(
    r"\bgroups?\s+of\s+patients?\b"
    r"|\bpatients?\s+(?:were\s+)?not\s+included\s+in\s+(?:clinical\s+)?trials?\b"
    r"|\bthe\s+drug\s+is\s+not\s+recommended\s+for\s+the\s+following\s+groups?\b",
    re.IGNORECASE,
)
DRUG_CLASS_STATEMENT_PATTERN = re.compile(
    r"\b(?:ssris?|snris?|statins?|beta[ -]?blockers?|beta[ -]?adrenergic\s+blockers?"
    r"|ace\s+inhibitors?|angiotensin\s+receptor\s+blockers?|arbs?"
    r"|penicillins?|cephalosporins?|opioids?|benzodiazepines?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StructuredExtractionResult:
    items: list[EventEvidenceItem]
    evidence_count_before_merge: int

    @property
    def evidence_count_after_merge(self) -> int:
        return len(self.items)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    protected = collapsed
    substitutions = {
        "e.g.": "e<prd>g<prd>",
        "i.e.": "i<prd>e<prd>",
        "U.S.": "U<prd>S<prd>",
        "Dr.": "Dr<prd>",
    }
    for original, replacement in substitutions.items():
        protected = protected.replace(original, replacement)
    # Protect common title/name abbreviations only when they introduce a
    # capitalized name. This keeps the rule conservative while preventing
    # names such as "St. John's Wort" from being split after the abbreviation.
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.(?=\s+[A-Z][A-Za-z'’\-]*)",
        r"\1<prd>",
        protected,
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    return [part.replace("<prd>", ".").strip() for part in parts if part.strip()]


def _supporting_quote(sentences: list[str], index: int, maximum: int = 1200) -> str:
    # Evidence quotes stay local to the sentence that contains the event. Wider
    # sentence or list context is passed separately for classification when needed.
    return sentences[index]


def _event_clauses(sentence: str) -> list[str]:
    # Semicolons are reliable independent-clause boundaries in label prose.
    # Commas are not split generically because they also delimit adverse-event
    # lists and splitting them would create incomplete evidence quotations.
    parts = re.split(r"\s*;\s*", sentence)
    clauses: list[str] = []
    for part in parts:
        clean = re.sub(
            r"^(?:and|but|while|whereas)\s+",
            "",
            part.strip(),
            flags=re.IGNORECASE,
        )
        if clean:
            clauses.append(clean)
    return clauses


def _matched_sentences(
    text: str,
    terms: list[str],
) -> list[tuple[int, str, list[str], str]]:
    sentences = _sentences(text)
    matches: list[tuple[int, str, list[str], str]] = []
    for index, sentence in enumerate(sentences):
        for clause in _event_clauses(sentence):
            term_matches = [term for term in terms if _term_pattern(term).search(clause)]
            if term_matches:
                matches.append((index, term_matches[0], sentences, clause))
    return matches


def _extract_patterns(text: str, patterns: list[str]) -> list[str]:
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(0).strip() for match in re.finditer(pattern, text, re.IGNORECASE))
    return _unique(values)


def _route_matches(text: str) -> list[tuple[str, str, int, int]]:
    matches: list[tuple[str, str, int, int]] = []
    for clue, route, pattern in ROUTE_CLUE_PATTERNS:
        matches.extend(
            (clue, route, match.start(), match.end())
            for match in re.finditer(pattern, text, re.IGNORECASE)
        )
    return sorted(matches, key=lambda value: (value[2], value[3], value[0]))


def _nearest_route_context(text: str, matched_term: str | None = None) -> list[str]:
    matches = _route_matches(text)
    if not matches:
        return []
    term_match = _term_pattern(matched_term).search(text) if matched_term else None
    if term_match is None:
        routes = _unique(route for _, route, _, _ in matches)
        if len(routes) != 1:
            return []
        selected_route = routes[0]
    else:
        term_center = (term_match.start() + term_match.end()) / 2
        selected_route = min(
            matches,
            key=lambda value: abs(((value[2] + value[3]) / 2) - term_center),
        )[1]
    clues = _unique(clue for clue, route, _, _ in matches if route == selected_route)
    return _unique([selected_route, *clues])


def _scoped_route_context(
    sentence: str,
    *,
    matched_term: str,
    chunk: SPLSectionChunk,
) -> list[str]:
    local = _nearest_route_context(sentence, matched_term)
    if local:
        return local
    subsection = _nearest_route_context(chunk.subsection_title or "")
    if subsection:
        return subsection
    return _nearest_route_context(chunk.section_title)


def _route_applicability(
    evidence_context: list[str],
    selected_product_context: str,
) -> str:
    evidence_routes = set(evidence_context) & ROUTE_NAMES
    selected_context = _nearest_route_context(selected_product_context)
    selected_routes = set(selected_context) & ROUTE_NAMES
    if not evidence_routes:
        return "unrestricted_route"
    if not selected_routes:
        return "ambiguous_route"
    if not evidence_routes.intersection(selected_routes):
        return "route_mismatch"
    evidence_forms = set(evidence_context) & FORMULATION_NAMES
    selected_forms = set(selected_context) & FORMULATION_NAMES
    if evidence_forms and selected_forms and not evidence_forms.intersection(selected_forms):
        return "route_mismatch"
    return "matching_route"


def _interaction_context(text: str, *, drug_interaction_section: bool = False) -> list[str]:
    core_matches = _extract_patterns(text, CORE_INTERACTION_PATTERNS)
    if not core_matches and not drug_interaction_section:
        return []
    return _extract_patterns(text, INTERACTION_PATTERNS)


def _frequency_parts(frequency_text: str | None) -> tuple[float | None, str | None]:
    if not frequency_text:
        return None, None
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", frequency_text)
    if percent:
        return float(percent.group(1)), "percent"
    of_patients = re.search(r"(\d+)\s+of\s+(\d[\d,]*)\s+patients?", frequency_text, re.IGNORECASE)
    if of_patients:
        return float(of_patients.group(1)), f"of {of_patients.group(2)} patients"
    per_population = re.search(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:cases?|events?)\s+per\s+(\d[\d,]*)\s+"
        r"(patient-years?|person-years?|patients?)",
        frequency_text,
        re.IGNORECASE,
    )
    if per_population:
        word_values = {
            "one": 1.0,
            "two": 2.0,
            "three": 3.0,
            "four": 4.0,
            "five": 5.0,
            "six": 6.0,
            "seven": 7.0,
            "eight": 8.0,
            "nine": 9.0,
            "ten": 10.0,
        }
        count_text = per_population.group(1).casefold()
        return (
            word_values.get(count_text, float(count_text) if count_text.isdigit() else None),
            f"per {per_population.group(2)} {per_population.group(3)}",
        )
    patient_count = re.fullmatch(
        r"\s*(\d[\d,]*)\s+((?:treated\s+)?(?:patients?|subjects?|participants?))\s*",
        frequency_text,
        re.IGNORECASE,
    )
    if patient_count:
        return float(patient_count.group(1).replace(",", "")), patient_count.group(2)
    ratio = re.search(
        r"(\d+(?:\.\d+)?)\s*/\s*(\d[\d,]*(?:\.\d+)?)"
        r"(?:\s+(treated\s+patients?|patients?|subjects?|cases?))?",
        frequency_text,
        re.IGNORECASE,
    )
    if ratio:
        population = f" {ratio.group(3)}" if ratio.group(3) else ""
        return float(ratio.group(1)), f"per {ratio.group(2)}{population}"
    if frequency_text.casefold() in {
        "common",
        "uncommon",
        "rare",
        "rarely",
        "frequency cannot be estimated",
    }:
        return None, "qualitative"
    return None, None


def _is_blood_pressure_ratio(text: str, match: re.Match[str]) -> bool:
    if "/" not in match.group(0):
        return False
    if POPULATION_RATIO_PATTERN.search(match.group(0)):
        return False
    left = max(0, match.start() - 80)
    right = min(len(text), match.end() + 80)
    context = text[left:right]
    return bool(BLOOD_PRESSURE_CONTEXT_PATTERN.search(context))


def _is_dose_ratio(text: str, match: re.Match[str]) -> bool:
    if "/" not in match.group(0):
        return False
    if POPULATION_RATIO_PATTERN.search(match.group(0)):
        return False
    left = max(0, match.start() - 80)
    right = min(len(text), match.end() + 40)
    return bool(DOSE_CONTEXT_PATTERN.search(text[left:right]))


def _numeric_clause(text: str, match: re.Match[str]) -> tuple[str, int, int]:
    left_boundary = max(
        text.rfind(".", 0, match.start()),
        text.rfind(";", 0, match.start()),
        text.rfind("|", 0, match.start()),
        text.rfind("\n", 0, match.start()),
    )
    right_candidates = [
        index
        for separator in (".", ";", "|", "\n")
        if (index := text.find(separator, match.end())) >= 0
    ]
    left = left_boundary + 1
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right], match.start() - left, match.end() - left


def _is_product_strength(text: str, match: re.Match[str]) -> bool:
    candidate = match.group(0)
    if "%" not in candidate:
        return False
    clause, start, end = _numeric_clause(text, match)
    before = clause[:start]
    after = clause[end:]
    form_before = re.search(
        rf"\b{DOSAGE_FORM_PATTERN}\b(?:\s*,)?\s*(?:contains?\s+)?$",
        before,
        re.IGNORECASE,
    )
    form_after = re.match(
        rf"\s*(?:[a-z][a-z0-9-]*\s+){{0,3}}{DOSAGE_FORM_PATTERN}\b",
        after,
        re.IGNORECASE,
    )
    composition_after = re.match(r"\s*(?:of\s+)?(?:active\s+ingredient|sodium\s+chloride|dextrose|lipid\s+emulsion)\b", after, re.IGNORECASE)
    composition_before = PRODUCT_COMPOSITION_PATTERN.search(before[-50:])
    return bool(form_before or form_after or composition_after or composition_before)


def _is_administered_dose(text: str, match: re.Match[str]) -> bool:
    if POPULATION_RATIO_PATTERN.search(match.group(0)):
        return False
    if _is_dose_ratio(text, match):
        return True
    clause, start, end = _numeric_clause(text, match)
    local = clause[max(0, start - 45) : min(len(clause), end + 45)]
    return bool(
        re.search(
            r"\b(?:received|administered|treated\s+with|dose(?:d)?|dosage|titrated|infusion)\b"
            r"[^.;]{0,30}\b\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms?|g|units?|mEq|mmol)\b",
            local,
            re.IGNORECASE,
        )
    )


def _is_sample_size(text: str, match: re.Match[str]) -> bool:
    candidate = match.group(0)
    if not re.search(r"\b(?:patients?|subjects?|participants?)\b", candidate, re.IGNORECASE):
        return False
    clause, start, end = _numeric_clause(text, match)
    before = clause[max(0, start - 55) : start]
    after = clause[end : min(len(clause), end + 45)]
    if re.match(
        r"\s*,\s*\d[\d,]*\s+(?:developed|experienced|reported|had)\b",
        after,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b(?:developed|experienced|reported|had)\b", after, re.IGNORECASE):
        return False
    return bool(SAMPLE_SIZE_CUE_PATTERN.search(before) or re.search(r"\breceived\b", after, re.IGNORECASE))


def _is_lab_or_vital_value(text: str, match: re.Match[str]) -> bool:
    clause, start, end = _numeric_clause(text, match)
    local = clause[max(0, start - 70) : min(len(clause), end + 25)]
    return bool(LAB_OR_VITAL_PATTERN.search(local))


def _is_comparator_frequency(text: str, match: re.Match[str]) -> bool:
    clause, start, end = _numeric_clause(text, match)
    before = clause[max(0, start - 35) : start]
    after = clause[end : min(len(clause), end + 35)]
    comparator = r"(?:placebo|active[ -]control|comparator|control)"
    return bool(
        re.search(rf"\b{comparator}\b[^%]{{0,24}}$", before, re.IGNORECASE)
        or re.match(rf"\s*(?:with\s+)?\b{comparator}\b", after, re.IGNORECASE)
    )


def _has_event_frequency_relation(
    text: str,
    match: re.Match[str],
    *,
    matched_term: str | None,
) -> bool:
    candidate = match.group(0)
    if re.search(r"\b(?:cases?|events?)\s+per\b", candidate, re.IGNORECASE):
        return True
    if re.search(r"\b\d+\s+of\s+\d[\d,]*\s+patients?\b", candidate, re.IGNORECASE):
        return True
    if POPULATION_RATIO_PATTERN.search(candidate):
        return True
    clause, start, end = _numeric_clause(text, match)
    before = clause[:start]
    after = clause[end:]
    if re.match(r"\s+of\s+(?:treated\s+)?(?:patients?|subjects?|participants?)\b", after, re.IGNORECASE):
        return True
    if re.match(
        r"\s+of\s+(?:treated\s+)?(?:patients?|subjects?|participants?)\s+"
        r"(?:reported|experienced|developed|had)\b",
        after,
        re.IGNORECASE,
    ):
        return True
    if EVENT_FREQUENCY_CUE_PATTERN.search(before[-100:]):
        return True
    if matched_term:
        term_pattern = _term_pattern(matched_term)
        if term_pattern.search(before[-80:]) or term_pattern.search(after[:80]):
            return True
    if re.search(r"\b(?:reported|experienced|developed|had)\b", after[:55], re.IGNORECASE):
        return True
    return False


def _classify_numeric_role(
    text: str,
    match: re.Match[str],
    *,
    matched_term: str | None = None,
) -> str:
    candidate = match.group(0).casefold()
    if candidate in QUALITATIVE_FREQUENCY_TERMS or candidate == "frequency cannot be estimated":
        return "event_frequency"
    if _is_product_strength(text, match):
        return "product_strength"
    if _is_blood_pressure_ratio(text, match) or _is_lab_or_vital_value(text, match):
        return "laboratory_or_vital_value"
    if _is_administered_dose(text, match):
        return "administered_dose"
    if _is_sample_size(text, match):
        return "sample_size"
    if _is_comparator_frequency(text, match):
        return "comparator_frequency"
    if _has_event_frequency_relation(text, match, matched_term=matched_term):
        return "event_frequency"
    return "ambiguous_numeric_value"


def _qualitative_frequency_applies(
    text: str,
    match: re.Match[str],
    *,
    matched_term: str | None,
) -> bool:
    if match.group(0).casefold() not in QUALITATIVE_FREQUENCY_TERMS or not matched_term:
        return True

    term = re.escape(matched_term).replace(r"\ ", r"\s+")
    qualifier = re.escape(match.group(0))
    qualifier_before_event = re.compile(
        rf"(?<!\w){qualifier}(?!\w)\s+"
        rf"(?:(?:cases?|events?|reports?)\s+of\s+)?(?<!\w){term}(?!\w)",
        re.IGNORECASE,
    )
    event_before_qualifier = re.compile(
        rf"(?<!\w){term}(?!\w)(?:"
        rf"[^.;]{{0,60}}?\b(?:reported|observed|seen|occurred|occurs|occurring)\b"
        rf"[^.;]{{0,20}}?|\s+\b(?:is|are|was|were)\b\s+)"
        rf"(?<!\w){qualifier}(?!\w)",
        re.IGNORECASE,
    )
    return bool(
        qualifier_before_event.search(text)
        or event_before_qualifier.search(text)
    )


def _frequency_matches(
    text: str,
    *,
    matched_term: str | None = None,
) -> list[re.Match[str]]:
    return [
        match
        for match in FREQUENCY_PATTERN.finditer(text)
        if _classify_numeric_role(text, match, matched_term=matched_term) == "event_frequency"
        and _qualitative_frequency_applies(text, match, matched_term=matched_term)
    ]


def _first_frequency(
    text: str,
    *,
    matched_term: str | None = None,
) -> tuple[str | None, float | None, str | None]:
    matches = _frequency_matches(text, matched_term=matched_term)
    if not matches:
        return None, None, None
    match = matches[0]
    frequency_text = " ".join(match.group(0).split())
    if frequency_text.casefold() in QUALITATIVE_FREQUENCY_TERMS:
        frequency_text = frequency_text.casefold()
    value, unit = _frequency_parts(frequency_text)
    return frequency_text, value, unit


def _frequency_tokens(text: str, *, matched_term: str | None = None) -> list[str]:
    return [
        " ".join(match.group(0).split())
        for match in _frequency_matches(text, matched_term=matched_term)
    ]


def _value_at(text: str, index: int) -> str | None:
    values = _frequency_tokens(text)
    if values:
        return values[min(index, len(values) - 1)]
    clean = text.strip()
    if clean in {"-", "—", "–"}:
        return clean
    return None


DRUG_HEADER_IGNORED_TOKENS = {
    "acetate",
    "besylate",
    "calcium",
    "citrate",
    "dihydrate",
    "fumarate",
    "hydrochloride",
    "hydrate",
    "maleate",
    "monohydrate",
    "phosphate",
    "sodium",
    "succinate",
    "sulfate",
    "tartrate",
}


def _drug_header_matches(header: str, drug_names: Iterable[str]) -> bool:
    header_tokens = set(re.findall(r"[a-z0-9]+", header.casefold()))
    if not header_tokens:
        return False
    for drug_name in drug_names:
        name_tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", drug_name.casefold())
            if len(token) > 2 and token not in DRUG_HEADER_IGNORED_TOKENS
        ]
        if name_tokens and all(token in header_tokens for token in name_tokens):
            return True
    return False


def _table_cell_frequency(header: str, cell: str, index: int = 0) -> str | None:
    clean_cell = " ".join(cell.split())
    if re.fullmatch(r"[<>]?\d+(?:\.\d+)?\s*%", clean_cell):
        return clean_cell
    explicit = _value_at(cell, index)
    if explicit:
        return explicit
    header_declares_percent = bool(
        re.search(r"\(\s*%\s*\)|\bpercent(?:age)?\b", header, re.IGNORECASE)
    )
    if header_declares_percent and re.fullmatch(r"[<>]?\d+(?:\.\d+)?", clean_cell):
        return f"{clean_cell}%"
    return None


def _comparator_label(header: str) -> str | None:
    normalized = " ".join(header.casefold().split())
    for pattern, label in [
        (r"\bactive[ -]control\b", "Active control"),
        (r"\bplacebo\b", "Placebo"),
        (r"\bcomparator\b", "Comparator"),
        (r"\bcontrol\b", "Control"),
    ]:
        if re.search(pattern, normalized):
            return label
    return None


def _table_frequency_variants(
    chunk: SPLSectionChunk,
    *,
    drug_names: Iterable[str],
) -> list[tuple[str, float | None, str | None, str | None, list[str]]]:
    if chunk.chunk_type != "table_row" or len(chunk.column_headers) != len(chunk.row_cells):
        return []
    selected_columns = [
        index
        for index, header in enumerate(chunk.column_headers)
        if index > 0 and _drug_header_matches(header, drug_names)
    ]
    if not selected_columns:
        return []

    event_index = 0
    variants: list[tuple[str, float | None, str | None, str | None, list[str]]] = []
    for index in selected_columns:
        frequency_text = _table_cell_frequency(
            chunk.column_headers[index],
            chunk.row_cells[index],
            event_index,
        )
        if not frequency_text:
            continue
        value, unit = _frequency_parts(frequency_text)
        selected_header = chunk.column_headers[index]
        group = selected_header.split("/")[0].strip() if "/" in selected_header else ""
        comparator_text = None
        comparator_candidates = [
            other_index
            for other_index, header in enumerate(chunk.column_headers)
            if other_index != index
            and _comparator_label(header) is not None
            and (not group or header.casefold().startswith(group.casefold()))
        ]
        if not comparator_candidates:
            comparator_candidates = [
                other_index
                for other_index, header in enumerate(chunk.column_headers)
                if other_index != index
                and _comparator_label(header) is not None
            ]
        if comparator_candidates:
            comparator_index = comparator_candidates[0]
            comparator_value = _table_cell_frequency(
                chunk.column_headers[comparator_index],
                chunk.row_cells[comparator_index],
                event_index,
            )
            if comparator_value:
                header = chunk.column_headers[comparator_index]
                comparator_name = _comparator_label(header)
                comparator_text = f"{comparator_name} {comparator_value}"
        is_population_header = bool(
            re.search(
                r"\bHF\b|heart failure|hypertens|patients?|subjects?|pregnan|renal|hepatic|myocardial",
                group,
                re.IGNORECASE,
            )
        )
        population = [group] if group and is_population_header else []
        variants.append((frequency_text, value, unit, comparator_text, population))
    return variants


def _free_text_frequency_variants(
    text: str,
) -> list[tuple[str, float | None, str | None, str | None, list[str]]]:
    population = (
        r"(?:elderly|pediatric|hypertensive) (?:patients|subjects)"
        r"|(?:patients|subjects) with (?:heart failure|myocardial infarction)"
        r"|patients with (?:renal|hepatic) impairment"
    )
    pattern = re.compile(
        rf"(?P<frequency>(?:approximately|about)?\s*\d+(?:\.\d+)?\s*%)"
        rf"\s+of\s+(?P<population>{population})",
        re.IGNORECASE,
    )
    variants = []
    comparator = _free_text_comparator(text)
    for match in pattern.finditer(text):
        frequency_text = " ".join(match.group("frequency").split())
        value, unit = _frequency_parts(frequency_text)
        variants.append(
            (
                frequency_text,
                value,
                unit,
                comparator,
                [" ".join(match.group("population").split())],
            )
        )
    return variants


def _free_text_comparator(text: str) -> str | None:
    pattern = re.compile(
        r"\b(placebo|comparator|control)\b[^.;|]{0,80}?(\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d[\d,]*|-)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        value_start, value_end = match.span(2)
        value_pattern = re.compile(
            r"\d+(?:\.\d+)?\s*%|\d+\s*/\s*\d[\d,]*",
            re.IGNORECASE,
        )
        value_match = value_pattern.match(text, value_start, value_end)
        if match.group(2) == "-" or (
            value_match
            and _classify_numeric_role(
                text,
                value_match,
            ) == "comparator_frequency"
        ):
            return f"{match.group(1).title()}: {' '.join(match.group(2).split())}"
    reverse = re.search(
        r"\bversus\s+(\d+(?:\.\d+)?\s*%)\s+with\s+"
        r"(placebo|comparator|control)\b",
        text,
        re.IGNORECASE,
    )
    if reverse:
        return f"{reverse.group(2).title()} {''.join(reverse.group(1).split())}"
    qualitative = re.search(
        r"\bmore\s+(?:often|frequently)\s+with\s+(placebo|comparator|control)\b",
        text,
        re.IGNORECASE,
    )
    return f"{qualitative.group(1).title()} (frequency not stated)" if qualitative else None


def _evidence_context(chunk: SPLSectionChunk, text: str) -> list[str]:
    context: list[str] = []
    section_key = section_key_for_code(chunk.section_code)
    structural_text = " ".join(
        [chunk.section_title, chunk.subsection_title or "", text]
    ).casefold()
    if section_key == "boxed_warning":
        context.append("boxed_warning")
    elif section_key in {"warnings", "warnings_and_cautions"}:
        context.append("warning")
    elif section_key == "contraindications":
        context.append("contraindication")
    elif section_key == "drug_interactions":
        context.append("drug_interaction")
    elif section_key == "adverse_reactions":
        context.append("adverse_reaction")
    if any(term in structural_text for term in ["clinical trial", "clinical studies", "clinical trials experience"]):
        context.append("clinical_trial")
    if "postmarketing" in structural_text:
        context.append("postmarketing")
    if any(term in structural_text for term in ["observational", "cohort study", "case-control"]):
        context.append("observational_statement")
    if not context:
        context.append("general_safety_statement")
    return _unique(context)


def _is_historical_context(normalized: str, term: str) -> bool:
    if not re.search(rf"\b{term}\b", normalized):
        return False
    patterns = [
        rf"\bpatients?\s+(?:with|who\s+have)\b[^.;]{{0,100}}\b{term}\b",
        rf"\bcontraindicated\s+in\s+patients?\s+with\b[^.;]{{0,100}}\b{term}\b",
        rf"\b(?:pre-existing|preexisting|known)\b[^.;]{{0,50}}\b{term}\b",
        rf"\bhistory\s+of\b[^.;]{{0,80}}\b{term}\b",
        rf"\bresting\b[^.;]{{0,50}}\b{term}\b",
        rf"\b{term}\b\s*\([^)]*\b(?:resting\s+systolic\s+blood\s+pressure|bp)\b",
        rf"\b{term}\b\s*\([^)]*(?:less\s+than|greater\s+than|[<>])[^)]*(?:mm\s*hg|bp|blood[ -]pressure)[^)]*\)",
        rf"\bno\s+controlled\s+clinical\s+data\b[^.;]{{0,160}}\b{term}\b",
    ]
    return "underlying condition" in normalized or any(
        re.search(pattern, normalized) for pattern in patterns
    )


def _historical_parent_context_applies(context: str, drug_name: str) -> bool:
    if HISTORICAL_PARENT_CONTEXT_PATTERN.search(context):
        return True
    escaped_drug = re.escape(drug_name.casefold()).replace(r"\ ", r"\s+")
    return bool(
        re.search(
            rf"\b{escaped_drug}\b\s+is\s+not\s+recommended\s+for\s+the\s+following\s+groups?\b",
            context,
            re.IGNORECASE,
        )
    )


def _secondary_interaction_modifier(
    sentence: str,
    *,
    matched_term: str,
    drug_name: str,
) -> bool:
    interaction_starts = [
        match.start()
        for pattern in CORE_INTERACTION_PATTERNS
        for match in re.finditer(pattern, sentence, re.IGNORECASE)
    ]
    term_match = _term_pattern(matched_term).search(sentence)
    if not interaction_starts or term_match is None:
        return False
    interaction_start = min(interaction_starts)
    if term_match.start() >= interaction_start:
        return False
    between_event_and_interaction = sentence[term_match.end() : interaction_start]
    clause_boundary = re.search(
        r"(?:;\s*|,\s*(?:and|but|while|whereas)\s+)",
        between_event_and_interaction,
        re.IGNORECASE,
    )
    if clause_boundary is None:
        return False
    boundary_start = term_match.end() + clause_boundary.start()
    primary_clause = sentence[:boundary_start].rstrip(" ;,:-")
    escaped_drug = re.escape(drug_name).replace(r"\ ", r"\s+")
    names_selected_drug = bool(
        re.search(rf"(?<!\w){escaped_drug}(?!\w)", primary_clause, re.IGNORECASE)
    )
    states_drug_effect = bool(
        re.search(
            r"\b(?:cause[sd]?|causing|precipitate[sd]?|result(?:ed|s)?\s+in|"
            r"associated\s+with|reported|occurred|observed)\b",
            primary_clause,
            re.IGNORECASE,
        )
    )
    return names_selected_drug or states_drug_effect


def _is_general_class_statement(sentence: str, drug_name: str) -> bool:
    escaped_drug = re.escape(drug_name).replace(r"\ ", r"\s+")
    drug_is_named = bool(
        re.search(rf"(?<!\w){escaped_drug}(?!\w)", sentence, re.IGNORECASE)
    )
    return not drug_is_named and bool(DRUG_CLASS_STATEMENT_PATTERN.search(sentence))


def _event_clause(sentence: str, matched_term: str) -> str:
    term_match = _term_pattern(matched_term).search(sentence)
    if term_match is None:
        return sentence
    left_boundary = sentence.rfind(";", 0, term_match.start())
    right_boundary = sentence.find(";", term_match.end())
    start = left_boundary + 1 if left_boundary >= 0 else 0
    end = right_boundary if right_boundary >= 0 else len(sentence)
    return sentence[start:end].strip()


def _event_is_negated(clause: str, matched_term: str) -> bool:
    normalized = " ".join(clause.casefold().split())
    term = re.escape(matched_term.casefold()).replace(r"\ ", r"\s+")
    patterns = [
        rf"\bno\s+(?:(?:cases?|events?)\s+of\s+)?{term}\b[^.]*\b(?:were|was)?\s*observed\b",
        rf"\b{term}\b[^.;]{{0,80}}\b(?:was|were|is|are|has\s+been|have\s+been)\s+not\s+"
        rf"(?:observed|reported|seen|identified|detected|found)\b",
        rf"\b(?:did|does|do)\s+not\s+(?:cause|result\s+in|produce|precipitate|lead\s+to)\b"
        rf"[^.;]{{0,80}}\b{term}\b",
        rf"\bwithout\s+(?:evidence\s+of\s+)?{term}\b",
    ]
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def _is_descriptive_or_management_reference(normalized: str, term: str) -> bool:
    patterns = [
        rf"\b(?:signs?\s+(?:or|and)\s+)?symptoms?\s+"
        rf"(?:associated\s+with|of)\s+{term}\b",
        rf"\bif\s+(?:signs?\s+(?:or|and)\s+symptoms?\s+of\s+)?{term}\b"
        rf"[^.;]{{0,80}}\b(?:occur|develop|emerge)\b",
    ]
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)


def _classify(
    sentence: str,
    *,
    matched_term: str,
    direct: bool,
    drug_name: str,
    drug_interaction_section: bool,
    context_text: str,
) -> tuple[str, str, str, str]:
    normalized = " ".join(sentence.casefold().split())
    normalized_context = " ".join(context_text.casefold().split())
    term = re.escape(matched_term.casefold()).replace(r"\ ", r"\s+")
    event_clause = _event_clause(sentence, matched_term)
    if _event_is_negated(event_clause, matched_term):
        return "negated", "absent", "selected_drug", "high"
    if re.search(rf"\b{term}\b[^.]*\bmore\s+(?:often|frequently)\s+with\s+(?:placebo|control|comparator)\b", normalized):
        return "comparator_only", "present", "comparator", "high"
    has_interaction = drug_interaction_section or any(
        re.search(pattern, sentence, re.IGNORECASE) for pattern in CORE_INTERACTION_PATTERNS
    )
    secondary_interaction = _secondary_interaction_modifier(
        sentence,
        matched_term=matched_term,
        drug_name=drug_name,
    )
    if has_interaction and not secondary_interaction:
        subject = (
            "concomitant_drug"
            if re.search(r"^\s*concomitant\s+[a-z0-9-]+\s+may\s+increase", sentence, re.IGNORECASE)
            else "selected_drug"
        )
        return "interaction_dependent", "conditional", subject, "high"
    if _is_historical_context(normalized, term) or (
        _historical_parent_context_applies(normalized_context, drug_name)
        and re.search(rf"\b{term}\b", normalized_context)
    ):
        return "historical_or_preexisting", "historical", "patient_history", "high"
    if _is_descriptive_or_management_reference(normalized, term):
        return "related_but_not_explicit", "uncertain", "unclear", "medium"
    if not direct:
        return "related_but_not_explicit", "uncertain", "unclear", "medium"
    if _is_general_class_statement(sentence, drug_name):
        return "explicit_positive", "present", "general_class_statement", "high"
    return "explicit_positive", "present", "selected_drug", "high"


def _make_item(
    *,
    usage: CmsUsage,
    normalized_event: str,
    matched_term: str,
    direct: bool,
    chunk: SPLSectionChunk,
    sentence: str,
    supporting_quote: str,
    context_text: str,
    selected_product_context: str,
    frequency: tuple[str | None, float | None, str | None, str | None, list[str]] | None = None,
) -> EventEvidenceItem:
    drug_interaction_section = section_key_for_code(chunk.section_code) == "drug_interactions"
    status, assertion, subject, confidence = _classify(
        sentence,
        matched_term=matched_term,
        direct=direct,
        drug_name=usage.member.name,
        drug_interaction_section=drug_interaction_section,
        context_text=context_text,
    )
    route_context = _scoped_route_context(
        sentence,
        matched_term=matched_term,
        chunk=chunk,
    )
    applicability = _route_applicability(route_context, selected_product_context)
    route_context = _unique([*route_context, applicability])
    if applicability == "route_mismatch":
        status = "insufficient_label_data"
        subject = "unclear"
        confidence = "low"
    elif applicability == "ambiguous_route" and status in {
        "explicit_positive",
        "negated",
    }:
        status = "related_but_not_explicit"
        assertion = "uncertain"
        subject = "unclear"
        confidence = "low"
    if frequency:
        frequency_text, frequency_value, frequency_unit, comparator_text, table_population = frequency
    else:
        frequency_text, frequency_value, frequency_unit = _first_frequency(
            sentence,
            matched_term=matched_term,
        )
        comparator_text = _free_text_comparator(sentence)
        table_population = []
    population_context = (
        _unique(table_population)
        if table_population
        else _extract_patterns(supporting_quote, POPULATION_PATTERNS)
    )
    interaction_source = context_text
    interaction_context = _interaction_context(
        interaction_source,
        drug_interaction_section=drug_interaction_section,
    )
    evidence_context = _evidence_context(chunk, context_text)
    mentioned_section = chunk.subsection_title or chunk.section_title
    return EventEvidenceItem(
        drug_name=usage.member.name,
        rxcui=usage.member.rxcui,
        normalized_event=normalized_event,
        matched_term=matched_term,
        evidence_status=status,
        assertion=assertion,
        subject=subject,
        section_code=chunk.section_code,
        section_title=chunk.section_title,
        subsection_title=chunk.subsection_title,
        evidence_context=evidence_context,
        frequency_text=frequency_text,
        frequency_value=frequency_value,
        frequency_unit=frequency_unit,
        comparator_text=comparator_text,
        population_context=population_context,
        dose_context=_extract_patterns(supporting_quote, DOSE_PATTERNS),
        interaction_context=interaction_context,
        temporal_context=_extract_patterns(supporting_quote, TEMPORAL_PATTERNS),
        route_context=route_context,
        postmarketing_only=(
            "postmarketing" in evidence_context and "clinical_trial" not in evidence_context
        ),
        source_text=chunk.text,
        supporting_quote=supporting_quote,
        chunk_type=chunk.chunk_type,
        source_path=chunk.source_path,
        chunk_hash=chunk.chunk_hash,
        extraction_method="deterministic_rules_v1",
        extraction_confidence=confidence,
        mentioned_in_sections=[mentioned_section],
        source_paths=[chunk.source_path],
        chunk_hashes=[chunk.chunk_hash],
    )


def extract_items_from_chunk(
    *,
    usage: CmsUsage,
    chunk: SPLSectionChunk,
    normalized_event: str,
    direct_terms: list[str],
    related_terms: list[str],
    parent_context: str | None = None,
    selected_product_context: str | None = None,
) -> list[EventEvidenceItem]:
    direct_matches = _matched_sentences(chunk.text, direct_terms)
    matches = direct_matches or _matched_sentences(chunk.text, related_terms)
    direct = bool(direct_matches)
    items: list[EventEvidenceItem] = []
    for sentence_index, matched_term, sentences, clause in matches:
        source_sentence = sentences[sentence_index]
        sentence = clause
        quote = clause
        clean_parent = " ".join((parent_context or "").split())
        context_text = source_sentence
        if chunk.chunk_type == "list_item" and clean_parent:
            combined_quote = f"{clean_parent} {quote}"
            if len(combined_quote) <= 1200:
                quote = combined_quote
            context_text = f"{clean_parent} {source_sentence}"
        product_context = selected_product_context or usage.member.name
        table_variants = _table_frequency_variants(
            chunk,
            drug_names=[usage.member.name, *usage.cms_generic_names],
        )
        frequency_variants = table_variants or _free_text_frequency_variants(sentence)
        if frequency_variants:
            items.extend(
                _make_item(
                    usage=usage,
                    normalized_event=normalized_event,
                    matched_term=matched_term,
                    direct=direct,
                    chunk=chunk,
                    sentence=sentence,
                    supporting_quote=quote,
                    context_text=context_text,
                    selected_product_context=product_context,
                    frequency=variant,
                )
                for variant in frequency_variants
            )
        else:
            items.append(
                _make_item(
                    usage=usage,
                    normalized_event=normalized_event,
                    matched_term=matched_term,
                    direct=direct,
                    chunk=chunk,
                    sentence=sentence,
                    supporting_quote=quote,
                    context_text=context_text,
                    selected_product_context=product_context,
                )
            )
    return items


def _unique_searchable_chunks(label: LabelEvidence) -> list[SPLSectionChunk]:
    unique: list[SPLSectionChunk] = []
    seen: set[str] = set()
    for chunk in label.spl_chunks:
        if chunk.chunk_type not in SEARCHABLE_CHUNK_TYPES:
            continue
        if section_key_for_code(chunk.section_code) not in REVIEWED_SECTION_KEYS:
            continue
        normalized = normalize_chunk_text(
            chunk.text,
            section_title=chunk.section_title,
            subsection_title=chunk.subsection_title,
        )
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(chunk)
    return unique


def _selected_product_context(usage: CmsUsage, label: LabelEvidence) -> str:
    values = [
        usage.member.name,
        label.requested_name,
        *usage.cms_generic_names,
        *label.generic_names,
        *label.brand_names,
        *label.substance_names,
    ]
    values.extend(str(value) for value in label.selected_label_product.values() if value)
    return " ".join(_unique(" ".join(str(value).split()) for value in values if value))


def _openfda_fallback_chunks(label: LabelEvidence) -> list[SPLSectionChunk]:
    chunks: list[SPLSectionChunk] = []
    set_id = label.selected_spl_set_id or "openfda-set-id-unavailable"
    for index, section_key in enumerate(
        [
            "boxed_warning",
            "warnings",
            "warnings_and_cautions",
            "adverse_reactions",
            "contraindications",
            "drug_interactions",
            "use_in_specific_populations",
        ]
    ):
        text = " ".join((label.sections.get(section_key) or "").split())
        section_code = OPENFDA_SECTION_CODES.get(section_key)
        if not text or not section_code:
            continue
        source_path = f"/openfda/sections/{section_key}"
        normalized = normalize_chunk_text(text)
        chunk_hash = hashlib.sha256(
            f"{set_id}|{section_code}|{source_path}|{normalized}".encode("utf-8")
        ).hexdigest()
        chunks.append(
            SPLSectionChunk(
                set_id=set_id,
                version=label.spl_version,
                effective_time=label.spl_effective_time or label.effective_time,
                section_code=section_code,
                section_title=section_key.replace("_", " ").title(),
                loinc_display_name=TARGET_SECTION_CODES[section_code][1],
                chunk_type="section_text",
                chunk_index=index,
                text=text,
                source_path=source_path,
                character_count=len(text),
                chunk_hash=chunk_hash,
            )
        )
    return chunks


def _list_item_parent_context(
    chunks: list[SPLSectionChunk], index: int
) -> str | None:
    chunk = chunks[index]
    if chunk.chunk_type != "list_item":
        return None
    list_path = chunk.source_path.rsplit("/item[", 1)[0]
    for previous in reversed(chunks[:index]):
        if (
            previous.section_code != chunk.section_code
            or previous.section_title != chunk.section_title
            or previous.subsection_title != chunk.subsection_title
        ):
            break
        if previous.chunk_type == "paragraph":
            return previous.text
        if (
            previous.chunk_type == "list_item"
            and previous.source_path.rsplit("/item[", 1)[0] == list_path
        ):
            continue
        break
    return None


def _placeholder_item(
    usage: CmsUsage,
    *,
    normalized_event: str,
    status: str,
) -> EventEvidenceItem:
    return EventEvidenceItem(
        drug_name=usage.member.name,
        rxcui=usage.member.rxcui,
        normalized_event=normalized_event,
        matched_term=normalized_event,
        evidence_status=status,
        assertion="uncertain",
        subject="unclear",
        extraction_confidence="low",
    )


def _merge_key(item: EventEvidenceItem) -> tuple[object, ...]:
    return (
        item.drug_name.casefold(),
        item.normalized_event.casefold(),
        item.evidence_status,
        item.assertion,
        item.subject,
        tuple(value.casefold() for value in item.route_context),
        item.section_code.casefold(),
        (item.subsection_title or "").casefold(),
        item.source_path.casefold(),
        item.chunk_hash.casefold(),
        " ".join(item.supporting_quote.casefold().split()),
        (item.frequency_text or "").casefold(),
        (item.comparator_text or "").casefold(),
        tuple(value.casefold() for value in item.population_context),
        tuple(value.casefold() for value in item.interaction_context),
    )


def merge_evidence_items(items: list[EventEvidenceItem]) -> list[EventEvidenceItem]:
    merged: dict[tuple[object, ...], EventEvidenceItem] = {}
    for item in items:
        key = _merge_key(item)
        current = merged.get(key)
        if current is None:
            merged[key] = item
            continue
        merged[key] = current.model_copy(
            update={
                "evidence_context": _unique([*current.evidence_context, *item.evidence_context]),
                "dose_context": _unique([*current.dose_context, *item.dose_context]),
                "temporal_context": _unique([*current.temporal_context, *item.temporal_context]),
                "route_context": _unique([*current.route_context, *item.route_context]),
                "mentioned_in_sections": _unique(
                    [*current.mentioned_in_sections, *item.mentioned_in_sections]
                ),
                "source_paths": _unique([*current.source_paths, *item.source_paths]),
                "chunk_hashes": _unique([*current.chunk_hashes, *item.chunk_hashes]),
                "postmarketing_only": current.postmarketing_only and item.postmarketing_only,
            }
        )
    return list(merged.values())


def extract_structured_event_evidence(
    *,
    selected_drugs: list[CmsUsage],
    labels: list[LabelEvidence],
    normalized_event: str,
    direct_terms: list[str],
    related_terms: list[str],
) -> StructuredExtractionResult:
    labels_by_rxcui = {label.rxcui: label for label in labels}
    raw_items: list[EventEvidenceItem] = []
    for usage in selected_drugs:
        label = labels_by_rxcui.get(usage.member.rxcui)
        if (
            label is None
            or label.label_match_confidence.casefold() == "low"
        ):
            raw_items.append(
                _placeholder_item(
                    usage,
                    normalized_event=normalized_event,
                    status="insufficient_label_data",
                )
            )
            continue
        if label.extraction_source == "dailymed_spl_xml":
            chunks = _unique_searchable_chunks(label) if label.spl_chunks else []
        else:
            chunks = _openfda_fallback_chunks(label)
        if not chunks:
            raw_items.append(
                _placeholder_item(
                    usage,
                    normalized_event=normalized_event,
                    status="insufficient_label_data",
                )
            )
            continue
        drug_items: list[EventEvidenceItem] = []
        selected_product_context = _selected_product_context(usage, label)
        for index, chunk in enumerate(chunks):
            drug_items.extend(
                extract_items_from_chunk(
                    usage=usage,
                    chunk=chunk,
                    normalized_event=normalized_event,
                    direct_terms=direct_terms,
                    related_terms=related_terms,
                    parent_context=_list_item_parent_context(chunks, index),
                    selected_product_context=selected_product_context,
                )
            )
        if drug_items:
            raw_items.extend(drug_items)
    return StructuredExtractionResult(
        items=merge_evidence_items(raw_items),
        evidence_count_before_merge=len(raw_items),
    )
