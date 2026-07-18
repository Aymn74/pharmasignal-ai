from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import httpx

from models import (
    LabelEvidence,
    SPLExtractionDiagnostics,
    SPLSectionChunk,
    SourceDetail,
)
from spl_qa import run_spl_extraction_qa


DAILYMED_SPL_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
HL7_NAMESPACE = "urn:hl7-org:v3"
NS = {"h": HL7_NAMESPACE}
MAX_XML_BYTES = 25 * 1024 * 1024
SET_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# LOINC codes identify the primary SPL sections. Nested sections inherit the
# primary code while retaining their own title as subsection_title.
TARGET_SECTION_CODES: dict[str, tuple[str, str]] = {
    "34066-1": ("boxed_warning", "FDA package insert Boxed warning section"),
    "43685-7": (
        "warnings_and_cautions",
        "FDA package insert Warnings and precautions section",
    ),
    "34071-1": ("warnings", "FDA package insert Warnings section"),
    "34084-4": ("adverse_reactions", "FDA package insert Adverse reactions section"),
    "34070-3": ("contraindications", "FDA package insert Contraindications section"),
    "34073-7": ("drug_interactions", "FDA package insert Drug interactions section"),
    "43684-0": (
        "use_in_specific_populations",
        "FDA package insert Use in specific populations section",
    ),
}


class DailyMedError(RuntimeError):
    pass


@dataclass(frozen=True)
class SPLParseResult:
    set_id: str
    version: str | None
    effective_time: str | None
    chunks: list[SPLSectionChunk]
    diagnostics: SPLExtractionDiagnostics
    xml_url: str

    @property
    def section_count(self) -> int:
        return self.diagnostics.section_count

    @property
    def loinc_section_count(self) -> int:
        return len({chunk.section_code for chunk in self.chunks})


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _clean_text(" ".join(element.itertext()))


def _attribute(element: ET.Element | None, name: str) -> str | None:
    if element is None:
        return None
    value = str(element.attrib.get(name) or "").strip()
    return value or None


def section_key_for_code(section_code: str) -> str:
    return TARGET_SECTION_CODES.get(section_code, ("other", ""))[0]


def _append_chunk(
    chunks: list[SPLSectionChunk],
    *,
    set_id: str,
    version: str | None,
    effective_time: str | None,
    section_code: str,
    section_title: str,
    loinc_display_name: str,
    subsection_title: str | None,
    chunk_type: str,
    text: str,
    source_path: str,
    table_id: str | None = None,
    row_index: int | None = None,
    column_headers: list[str] | None = None,
    row_cells: list[str] | None = None,
) -> None:
    clean = _clean_text(text)
    if not clean:
        return
    chunks.append(
        SPLSectionChunk(
            set_id=set_id,
            version=version,
            effective_time=effective_time,
            section_code=section_code,
            section_title=section_title,
            loinc_display_name=loinc_display_name,
            subsection_title=subsection_title,
            chunk_type=chunk_type,
            chunk_index=len(chunks),
            text=clean,
            table_id=table_id,
            row_index=row_index,
            column_headers=column_headers or [],
            row_cells=row_cells or [],
            source_path=source_path,
        )
    )


def _table_cells(row: ET.Element) -> list[ET.Element]:
    return [
        element
        for element in list(row)
        if _local_name(element.tag) in {"td", "th"}
    ]


def _expanded_header_values(row: ET.Element) -> list[str]:
    values: list[str] = []
    for cell in _table_cells(row):
        text = _element_text(cell)
        try:
            colspan = max(1, int(cell.attrib.get("colspan") or 1))
        except (TypeError, ValueError):
            colspan = 1
        values.extend([text] * colspan)
    return values


def _column_headers(header_rows: list[ET.Element], column_count: int) -> list[str]:
    if not header_rows:
        return [f"Column {index + 1}" for index in range(column_count)]
    expanded_rows = [_expanded_header_values(row) for row in header_rows]
    headers: list[str] = []
    for column_index in range(column_count):
        parts: list[str] = []
        for row_values in expanded_rows:
            if column_index >= len(row_values):
                continue
            value = row_values[column_index]
            if value and value not in parts:
                parts.append(value)
        headers.append(" / ".join(parts) or f"Column {column_index + 1}")
    return headers


def _cell_has_style(cell: ET.Element, style_name: str) -> bool:
    styles = str(cell.attrib.get("styleCode") or "").casefold().split()
    return style_name.casefold() in styles


def _table_caption_text(table: ET.Element) -> str:
    for child in list(table):
        if _local_name(child.tag) == "caption":
            return _element_text(child)
    return ""


def _format_table_cell(text: str, *, percentage_context: bool) -> str:
    """Preserve table semantics when the caption declares percentage values."""
    if not percentage_context:
        return text
    numeric_tokens = re.fullmatch(
        r"\s*([<>]?\d+(?:\.\d+)?(?:\s+[<>]?\d+(?:\.\d+)?)*)\s*",
        text,
    )
    if not numeric_tokens:
        return text
    return " ".join(f"{token}%" for token in numeric_tokens.group(1).split())


def _detect_table_header_rows(rows: list[ET.Element], table: ET.Element) -> list[ET.Element]:
    thead_rows = {
        id(element)
        for thead in table.iter()
        if _local_name(thead.tag) == "thead"
        for element in thead.iter()
        if _local_name(element.tag) == "tr"
    }
    explicit = [
        row
        for row in rows
        if id(row) in thead_rows
        or (
            _table_cells(row)
            and all(_local_name(cell.tag) == "th" for cell in _table_cells(row))
        )
    ]
    if explicit:
        return explicit

    # DailyMed SPL tables frequently use styled TD cells rather than TH/THEAD.
    # Toprule marks the start of the header block; a following Botrule row often
    # supplies sample sizes or leaf column labels.
    for index, row in enumerate(rows):
        cells = _table_cells(row)
        if not cells or not any(_cell_has_style(cell, "Toprule") for cell in cells):
            continue
        header_rows = [row]
        if index + 1 < len(rows):
            next_row = rows[index + 1]
            next_cells = _table_cells(next_row)
            if next_cells and any(_cell_has_style(cell, "Botrule") for cell in next_cells):
                header_rows.append(next_row)
        return header_rows
    return []


def _is_group_heading_row(row: ET.Element) -> bool:
    cells = _table_cells(row)
    if len(cells) != 1 or not _element_text(cells[0]):
        return False
    try:
        return int(cells[0].attrib.get("colspan") or 1) >= 2
    except (TypeError, ValueError):
        return False


def _extract_grouped_key_value_table(
    rows: list[ET.Element],
    *,
    table_path: str,
    table_id: str,
    percentage_context: bool,
    chunks: list[SPLSectionChunk],
    metadata: dict[str, str | None],
) -> bool:
    """Extract DailyMed interaction tables whose spanning rows name each group."""
    group_row_count = sum(1 for row in rows if _is_group_heading_row(row))
    if group_row_count < 2 or not rows or not _is_group_heading_row(rows[0]):
        return False

    group_heading = ""
    for row_index, row in enumerate(rows):
        if _is_group_heading_row(row):
            group_heading = _element_text(_table_cells(row)[0])
            _append_chunk(
                chunks,
                **metadata,
                chunk_type="table_group",
                text=group_heading,
                source_path=f"{table_path}/group[{row_index}]",
                table_id=table_id,
                row_index=row_index,
                column_headers=["Interaction"],
                row_cells=[group_heading],
            )
            continue
        cells = _table_cells(row)
        values = [
            _format_table_cell(_element_text(cell), percentage_context=percentage_context)
            for cell in cells
        ]
        if not group_heading or len(values) < 2 or not any(values):
            continue
        headers = ["Interaction", "Attribute", "Detail"]
        row_cells = [group_heading, values[0], " | ".join(values[1:])]
        row_text = " | ".join(
            f"{header}: {value or 'Not reported'}"
            for header, value in zip(headers, row_cells)
        )
        _append_chunk(
            chunks,
            **metadata,
            chunk_type="table_row",
            text=row_text,
            source_path=f"{table_path}/row[{row_index}]",
            table_id=table_id,
            row_index=row_index,
            column_headers=headers,
            row_cells=row_cells,
        )
    return True


def _extract_table(
    table: ET.Element,
    *,
    table_path: str,
    chunks: list[SPLSectionChunk],
    metadata: dict[str, str | None],
) -> None:
    table_id = (
        str(table.attrib.get("ID") or table.attrib.get("id") or "").strip()
        or table_path.replace("/", "-").replace("[", "").replace("]", "").strip("-")
    )
    rows = [element for element in table.iter() if _local_name(element.tag) == "tr"]
    caption_text = _table_caption_text(table)
    percentage_context = "%" in caption_text or "percent" in caption_text.casefold()
    if _extract_grouped_key_value_table(
        rows,
        table_path=table_path,
        table_id=table_id,
        percentage_context=percentage_context,
        chunks=chunks,
        metadata=metadata,
    ):
        return
    header_rows = _detect_table_header_rows(rows, table)
    header_row_ids = {id(row) for row in header_rows}
    first_header_index = min(
        (index for index, row in enumerate(rows) if id(row) in header_row_ids),
        default=0,
    )
    for row_index, row in enumerate(rows):
        if id(row) in header_row_ids:
            continue
        cells = _table_cells(row)
        cell_texts = [_element_text(cell) for cell in cells]
        if not any(cell_texts):
            continue
        if header_rows and row_index < first_header_index:
            _append_chunk(
                chunks,
                **metadata,
                chunk_type="table_note",
                text=" | ".join(value for value in cell_texts if value),
                source_path=f"{table_path}/note[{row_index}]",
                table_id=table_id,
                row_index=row_index,
                row_cells=cell_texts,
            )
            continue
        cell_texts = [
            _format_table_cell(value, percentage_context=percentage_context)
            for value in cell_texts
        ]
        headers = _column_headers(header_rows, len(cell_texts))
        row_text = " | ".join(
            f"{header}: {cell_text or 'Not reported'}"
            for header, cell_text in zip(headers, cell_texts)
        )
        row_path = f"{table_path}/row[{row_index}]"
        _append_chunk(
            chunks,
            **metadata,
            chunk_type="table_row",
            text=row_text,
            source_path=row_path,
            table_id=table_id,
            row_index=row_index,
            column_headers=headers,
            row_cells=cell_texts,
        )


def _extract_text_container(
    container: ET.Element,
    *,
    container_path: str,
    chunks: list[SPLSectionChunk],
    metadata: dict[str, str | None],
) -> None:
    paragraph_index = 0
    leading_text = _clean_text(container.text or "")
    if leading_text:
        _append_chunk(
            chunks,
            **metadata,
            chunk_type="paragraph",
            text=leading_text,
            source_path=f"{container_path}/paragraph[{paragraph_index}]",
        )
        paragraph_index += 1

    tag_counts: dict[str, int] = {}
    for child in list(container):
        tag = _local_name(child.tag)
        index = tag_counts.get(tag, 0)
        tag_counts[tag] = index + 1
        if tag == "paragraph":
            child_path = f"{container_path}/paragraph[{paragraph_index}]"
            paragraph_index += 1
            _append_chunk(
                chunks,
                **metadata,
                chunk_type="paragraph",
                text=_element_text(child),
                source_path=child_path,
            )
        elif tag == "list":
            child_path = f"{container_path}/list[{index}]"
            items = [element for element in list(child) if _local_name(element.tag) == "item"]
            for item_index, item in enumerate(items):
                _append_chunk(
                    chunks,
                    **metadata,
                    chunk_type="list_item",
                    text=_element_text(item),
                    source_path=f"{child_path}/item[{item_index}]",
                )
        elif tag == "table":
            child_path = f"{container_path}/table[{index}]"
            _extract_table(
                child,
                table_path=child_path,
                chunks=chunks,
                metadata=metadata,
            )
        else:
            child_path = f"{container_path}/{tag}[{index}]"
            _extract_text_container(
                child,
                container_path=child_path,
                chunks=chunks,
                metadata=metadata,
            )

        tail_text = _clean_text(child.tail or "")
        if tail_text:
            _append_chunk(
                chunks,
                **metadata,
                chunk_type="paragraph",
                text=tail_text,
                source_path=f"{container_path}/paragraph[{paragraph_index}]",
            )
            paragraph_index += 1


def _extract_section_tree(
    section: ET.Element,
    *,
    section_path: str,
    primary_code: str,
    primary_title: str,
    subsection_title: str | None,
    chunks: list[SPLSectionChunk],
    document_metadata: dict[str, str | None],
) -> None:
    chunk_metadata = {
        **document_metadata,
        "section_code": primary_code,
        "section_title": primary_title,
        "loinc_display_name": TARGET_SECTION_CODES[primary_code][1],
        "subsection_title": subsection_title,
    }
    text_node = section.find("./h:text", NS)
    if text_node is not None:
        _extract_text_container(
            text_node,
            container_path=section_path,
            chunks=chunks,
            metadata=chunk_metadata,
        )

    nested_sections = section.findall("./h:component/h:section", NS)
    for nested_index, nested in enumerate(nested_sections):
        nested_title = _element_text(nested.find("./h:title", NS)) or "Untitled subsection"
        combined_title = (
            f"{subsection_title} > {nested_title}" if subsection_title else nested_title
        )
        _extract_section_tree(
            nested,
            section_path=f"{section_path}/subsection[{nested_index}]",
            primary_code=primary_code,
            primary_title=primary_title,
            subsection_title=combined_title,
            chunks=chunks,
            document_metadata=document_metadata,
        )


def _iter_target_sections(root: ET.Element):
    def walk(element: ET.Element, inside_target: bool = False):
        for child in list(element):
            tag = _local_name(child.tag)
            if tag == "section":
                code = _attribute(child.find("./h:code", NS), "code") or ""
                is_target = code in TARGET_SECTION_CODES
                if is_target and not inside_target:
                    yield child
                yield from walk(child, inside_target or is_target)
            else:
                yield from walk(child, inside_target)

    yield from walk(root)


def parse_spl_xml(xml_bytes: bytes, *, expected_set_id: str, xml_url: str) -> SPLParseResult:
    if not xml_bytes:
        raise DailyMedError("DailyMed returned an empty SPL XML document.")
    if len(xml_bytes) > MAX_XML_BYTES:
        raise DailyMedError("DailyMed SPL XML exceeds the 25 MB parser safety limit.")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DailyMedError(f"DailyMed returned invalid SPL XML: {exc}") from exc
    if _local_name(root.tag) != "document":
        raise DailyMedError("DailyMed response is not an HL7 SPL document.")

    parsed_set_id = _attribute(root.find("./h:setId", NS), "root")
    version = _attribute(root.find("./h:versionNumber", NS), "value")
    effective_time = _attribute(root.find("./h:effectiveTime", NS), "value")
    if not parsed_set_id:
        raise DailyMedError("The SPL XML does not contain a document SET ID.")
    if parsed_set_id.casefold() != expected_set_id.casefold():
        raise DailyMedError(
            f"DailyMed SET ID mismatch: requested {expected_set_id}, received {parsed_set_id}."
        )

    chunks: list[SPLSectionChunk] = []
    document_metadata = {
        "set_id": parsed_set_id,
        "version": version,
        "effective_time": effective_time,
    }
    for section_index, section in enumerate(_iter_target_sections(root)):
        code = _attribute(section.find("./h:code", NS), "code") or ""
        if code not in TARGET_SECTION_CODES:
            continue
        xml_title = _element_text(section.find("./h:title", NS))
        loinc_title = TARGET_SECTION_CODES[code][1]
        _extract_section_tree(
            section,
            section_path=f"section[{section_index}]",
            primary_code=code,
            primary_title=xml_title or loinc_title,
            subsection_title=None,
            chunks=chunks,
            document_metadata=document_metadata,
        )

    if not chunks:
        raise DailyMedError(
            "DailyMed SPL XML contained none of the configured LOINC safety sections."
        )
    section_count = len(
        {
            (chunk.section_code, chunk.section_title, chunk.subsection_title)
            for chunk in chunks
        }
    )
    unique_chunks, diagnostics = run_spl_extraction_qa(
        chunks,
        section_count=section_count,
        set_id=parsed_set_id,
        spl_version=version,
    )
    return SPLParseResult(
        set_id=parsed_set_id,
        version=version,
        effective_time=effective_time,
        chunks=unique_chunks,
        diagnostics=diagnostics,
        xml_url=xml_url,
    )


def fetch_dailymed_spl(set_id: str, *, timeout_seconds: float) -> tuple[SPLParseResult, SourceDetail]:
    clean_set_id = set_id.strip().lower()
    if not SET_ID_PATTERN.fullmatch(clean_set_id):
        raise DailyMedError("The selected openFDA SPL SET ID is missing or invalid.")
    url = DAILYMED_SPL_URL.format(set_id=clean_set_id)
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "PharmaSignal-AI/1.0 research prototype"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DailyMedError(
            f"DailyMed SPL XML request failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise DailyMedError(f"DailyMed SPL XML request failed: {exc}") from exc

    parsed = parse_spl_xml(response.content, expected_set_id=clean_set_id, xml_url=url)
    detail = SourceDetail(
        source="DailyMed",
        query=url,
        record_count=parsed.diagnostics.unique_chunks,
        note=(
            f"Fetched official SPL XML version {parsed.version or 'unknown'} and extracted "
            f"{parsed.section_count} structural section(s) across {parsed.loinc_section_count} "
            f"configured LOINC section code(s). SPL Extraction QA v1 retained "
            f"{parsed.diagnostics.unique_chunks} unique chunk(s) from "
            f"{parsed.diagnostics.total_chunks} extracted chunk(s); duplicate rate "
            f"{parsed.diagnostics.duplicate_rate:.2%}."
        ),
    )
    return parsed, detail


def enrich_label_with_dailymed(
    label: LabelEvidence, *, timeout_seconds: float
) -> tuple[LabelEvidence, SourceDetail]:
    if not label.selected_spl_set_id:
        raise DailyMedError("The selected openFDA label has no SPL SET ID for DailyMed lookup.")
    parsed, detail = fetch_dailymed_spl(
        label.selected_spl_set_id,
        timeout_seconds=timeout_seconds,
    )
    enriched = label.model_copy(
        update={
            "extraction_source": "dailymed_spl_xml",
            "spl_version": parsed.version,
            "spl_effective_time": parsed.effective_time,
            "spl_chunks": parsed.chunks,
            "spl_diagnostics": parsed.diagnostics,
            "dailymed_xml_url": parsed.xml_url,
            "dailymed_warning": None,
        }
    )
    return enriched, detail
