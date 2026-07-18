from __future__ import annotations

import hashlib
import math
import unicodedata

from models import SPLExtractionDiagnostics, SPLSectionChunk


SPL_EXTRACTION_QA_VERSION = "spl-extraction-qa-v1"


def _normalize_characters(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        character if character.isalnum() or character.isspace() or character == "%" else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def normalize_chunk_text(
    text: str,
    *,
    section_title: str = "",
    subsection_title: str | None = None,
) -> str:
    normalized = _normalize_characters(text)
    heading_values = [section_title]
    if subsection_title:
        heading_values.extend([subsection_title, *subsection_title.split(">")])
    headings = sorted(
        {_normalize_characters(value) for value in heading_values if value},
        key=len,
        reverse=True,
    )
    changed = True
    while normalized and changed:
        changed = False
        for heading in headings:
            if normalized == heading:
                normalized = ""
                changed = True
                break
            prefix = heading + " "
            if heading and normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
                changed = True
                break
    return normalized


def chunk_hash(chunk: SPLSectionChunk) -> str:
    normalized_text = normalize_chunk_text(
        chunk.text,
        section_title=chunk.section_title,
        subsection_title=chunk.subsection_title,
    )
    hash_input = "\x1f".join(
        [chunk.set_id.casefold(), chunk.section_code, chunk.source_path, normalized_text]
    )
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def _percentile_90(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = 0.9 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def run_spl_extraction_qa(
    chunks: list[SPLSectionChunk],
    *,
    section_count: int,
    set_id: str,
    spl_version: str | None,
) -> tuple[list[SPLSectionChunk], SPLExtractionDiagnostics]:
    total_chunks = len(chunks)
    unique_chunks: list[SPLSectionChunk] = []
    seen_text: set[str] = set()
    duplicate_chunks = 0
    empty_chunks = 0

    for chunk in chunks:
        normalized_text = normalize_chunk_text(
            chunk.text,
            section_title=chunk.section_title,
            subsection_title=chunk.subsection_title,
        )
        if not normalized_text:
            empty_chunks += 1
            continue
        if normalized_text in seen_text:
            duplicate_chunks += 1
            continue
        seen_text.add(normalized_text)
        prepared = chunk.model_copy(
            update={
                "chunk_index": len(unique_chunks),
                "character_count": len(chunk.text),
                "chunk_hash": chunk_hash(chunk),
            }
        )
        unique_chunks.append(prepared)

    lengths = [chunk.character_count for chunk in unique_chunks]
    diagnostics = SPLExtractionDiagnostics(
        qa_version=SPL_EXTRACTION_QA_VERSION,
        set_id=set_id,
        spl_version=spl_version,
        section_count=section_count,
        total_chunks=total_chunks,
        unique_chunks=len(unique_chunks),
        duplicate_chunks=duplicate_chunks,
        duplicate_rate=(duplicate_chunks / total_chunks if total_chunks else 0.0),
        minimum_characters=min(lengths, default=0),
        median_characters=_median(lengths),
        p90_characters=_percentile_90(lengths),
        maximum_characters=max(lengths, default=0),
        chunks_over_2000_characters=sum(length > 2000 for length in lengths),
        chunks_over_5000_characters=sum(length > 5000 for length in lengths),
        empty_chunks=empty_chunks,
        extraction_source="dailymed_spl_xml",
    )
    return unique_chunks, diagnostics
