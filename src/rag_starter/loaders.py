from __future__ import annotations

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .schema import Document, DocumentSection


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".xml"}
MANIFEST_FIELDS = {
    "doc_id",
    "title",
    "source_type",
    "status",
    "version",
    "published_at",
    "canonical_url",
    "download_url",
    "local_path",
    "license",
    "topics",
    "supersedes",
    "superseded_by",
    "quality_review",
}
SECTION_HEADING = re.compile(
    r"^(?P<id>(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)+|Appendix\s+[A-Z])\.?)"
    r"\s+(?P<title>[A-Za-z][A-Za-z0-9 /,()'&:+\-]{2,100})$",
    flags=re.IGNORECASE,
)
MARKDOWN_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
ADMINISTRATIVE_HEADING_PREFIXES = (
    "explanation:",
    "name of standard:",
    "maintenance agency:",
    "implementation schedule:",
    "waiver procedure:",
    "where to obtain copies:",
)
CITATION_HEADING_PREFIXES = ("sp ", "fips ", "rfc ")


def load_documents(path: str | Path) -> list[Document]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Document path does not exist: {root}")
    if root.is_file() and root.suffix.lower() == ".jsonl":
        return load_manifest_documents(root)

    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    documents: list[Document] = []
    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        document = load_document_file(file_path)
        if document.text.strip():
            documents.append(document)
    return documents


def load_manifest_documents(
    manifest_path: str | Path,
    *,
    allow_missing: bool = False,
    limit: int = 0,
) -> list[Document]:
    entries = load_manifest(manifest_path)
    if limit > 0:
        entries = entries[:limit]

    documents = []
    missing = []
    manifest_root = Path(manifest_path).resolve().parent
    for entry in entries:
        local_path = Path(str(entry["local_path"]))
        if not local_path.is_absolute():
            candidates = [
                Path.cwd() / local_path,
                manifest_root / local_path,
                manifest_root.parent / local_path,
            ]
            local_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not local_path.exists():
            missing.append(str(local_path))
            continue
        document = load_document_file(local_path, metadata=entry)
        if document.text.strip():
            documents.append(document)

    if missing and not allow_missing:
        preview = "\n".join(f"- {path}" for path in missing[:10])
        remainder = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(f"Manifest files are missing:\n{preview}{remainder}")
    return documents


def load_manifest(path: str | Path) -> list[dict]:
    entries = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest row must be an object at {path}:{line_number}")
            for field in ("doc_id", "title", "source_type", "download_url", "local_path"):
                if not entry.get(field):
                    raise ValueError(f"Manifest field {field!r} missing at {path}:{line_number}")
            unknown_fields = sorted(set(entry) - MANIFEST_FIELDS)
            if unknown_fields:
                raise ValueError(
                    f"Unknown manifest fields at {path}:{line_number}: "
                    f"{', '.join(unknown_fields)}"
                )
            doc_id = str(entry["doc_id"])
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate manifest doc_id={doc_id!r} at {path}:{line_number}")
            seen_ids.add(doc_id)
            entries.append(entry)
    return entries


def load_document_file(file_path: str | Path, metadata: dict | None = None) -> Document:
    path = Path(file_path)
    metadata = metadata or {}
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        sections = read_pdf_sections(path)
    elif suffix == ".xml" and metadata.get("source_type") == "rfc":
        sections = read_rfc_xml_sections(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        sections = read_text_sections(
            text,
            require_dotted_headings=metadata.get("source_type") == "rfc",
        )

    text, sections = finalize_sections(sections)
    source = str(metadata.get("doc_id") or path.name)
    return Document(
        text=text,
        source=source,
        doc_id=str(metadata.get("doc_id") or path.name),
        title=str(metadata.get("title") or path.stem),
        source_type=str(metadata.get("source_type") or "local"),
        version=str(metadata.get("version") or ""),
        status=str(metadata.get("status") or ""),
        published_at=str(metadata.get("published_at") or ""),
        canonical_url=str(metadata.get("canonical_url") or metadata.get("landing_url") or ""),
        download_url=str(metadata.get("download_url") or ""),
        sha256=sha256_file(path),
        license=str(metadata.get("license") or ""),
        supersedes=tuple(str(item) for item in metadata.get("supersedes", [])),
        superseded_by=(
            str(metadata["superseded_by"])
            if metadata.get("superseded_by")
            else None
        ),
        topics=tuple(str(item) for item in metadata.get("topics", [])),
        sections=tuple(sections),
        extra={
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "doc_id",
                "title",
                "source_type",
                "version",
                "status",
                "published_at",
                "canonical_url",
                "landing_url",
                "download_url",
                "local_path",
                "license",
                "supersedes",
                "superseded_by",
                "topics",
            }
        },
    )


def read_pdf_sections(path: str | Path) -> list[DocumentSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading PDF files requires pypdf. Run: pip install pypdf") from exc

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            text = page.extract_text() or ""
        pages.append((page_number, clean_extracted_text(text)))
    return sections_from_pdf_pages(pages)


def sections_from_pdf_pages(pages: Iterable[tuple[int, str]]) -> list[DocumentSection]:
    sections = []
    current_id = "front-matter"
    current_title = "Front Matter"
    current_level = 1
    current_lines: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    heading_count = 0

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(
                DocumentSection(
                    section_id=current_id,
                    title=current_title,
                    text=text,
                    level=current_level,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
        current_lines = []

    materialized_pages = list(pages)
    for page_number, page_text in materialized_pages:
        if page_start is None:
            page_start = page_number
        page_end = page_number
        for raw_line in page_text.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                continue
            heading = match_section_heading(line)
            if heading:
                flush()
                heading_count += 1
                current_id = heading.group("id").rstrip(".")
                current_title = heading.group("title").strip()
                current_level = section_level(current_id)
                current_lines = [line]
                page_start = page_number
                page_end = page_number
            else:
                current_lines.append(line)
    flush()

    front_matter = next(
        (section for section in sections if section.section_id == "front-matter"),
        None,
    )
    front_matter_pages = (
        (front_matter.page_end or 0) - (front_matter.page_start or 1) + 1
        if front_matter is not None
        else 0
    )
    # Some NIST PDFs place headings and page furniture in extraction order that
    # makes a numbered list item look like the first real section. If more than
    # 40% of the file would become one "front matter" section, page-level
    # sections are more honest and still provide precise citations.
    if heading_count and (
        not materialized_pages
        or front_matter_pages / len(materialized_pages) <= 0.40
    ):
        return sections
    return page_sections(materialized_pages)


def page_sections(pages: Iterable[tuple[int, str]]) -> list[DocumentSection]:
    return [
        DocumentSection(
            section_id=f"page-{page_number}",
            title=f"Page {page_number}",
            text=page_text,
            page_start=page_number,
            page_end=page_number,
        )
        for page_number, page_text in pages
        if page_text.strip()
    ]


def read_rfc_xml_sections(path: str | Path) -> list[DocumentSection]:
    root = ET.parse(path).getroot()
    sections: list[DocumentSection] = []

    abstract = next((element for element in root.iter() if local_name(element.tag) == "abstract"), None)
    if abstract is not None:
        text = element_text(abstract)
        if text:
            sections.append(DocumentSection("abstract", "Abstract", text))

    middle = next((element for element in root.iter() if local_name(element.tag) == "middle"), None)
    if middle is None:
        middle = root

    def walk(parent: ET.Element, prefix: tuple[int, ...] = ()) -> None:
        child_sections = [child for child in parent if local_name(child.tag) == "section"]
        for position, section in enumerate(child_sections, start=1):
            number = (*prefix, position)
            section_id = str(section.get("anchor") or ".".join(map(str, number)))
            title = str(section.get("name") or section_id)
            text_parts = []
            for child in section:
                if local_name(child.tag) == "section":
                    continue
                text = element_text(child)
                if text:
                    text_parts.append(text)
            text = "\n".join(text_parts).strip()
            if text:
                sections.append(
                    DocumentSection(
                        section_id=section_id,
                        title=title,
                        text=text,
                        level=len(number),
                    )
                )
            walk(section, number)

    walk(middle)
    if sections:
        return sections
    text = element_text(root)
    return [DocumentSection("document", "Document", text)] if text else []


def read_text_sections(
    text: str,
    *,
    require_dotted_headings: bool = False,
) -> list[DocumentSection]:
    lines = text.replace("\r\n", "\n").split("\n")
    sections = []
    current_id = "document"
    current_title = "Document"
    current_level = 1
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(
                DocumentSection(
                    section_id=current_id,
                    title=current_title,
                    text=section_text,
                    level=current_level,
                )
            )
        current_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        markdown = None if require_dotted_headings else MARKDOWN_HEADING.match(line)
        numbered = match_section_heading(
            " ".join(line.split()),
            allow_lowercase_title=require_dotted_headings,
        )
        if (
            numbered is not None
            and require_dotted_headings
            and (
                not numbered.group("id").endswith(".")
                or raw_line[:1].isspace()
            )
        ):
            numbered = None
        if markdown:
            flush()
            current_title = markdown.group("title").strip()
            current_id = slugify(current_title)
            current_level = len(markdown.group("hashes"))
            current_lines = [current_title]
        elif numbered:
            flush()
            current_id = numbered.group("id").rstrip(".")
            current_title = numbered.group("title").strip()
            current_level = section_level(current_id)
            current_lines = [line]
        else:
            current_lines.append(raw_line.rstrip())
    flush()
    return sections


def finalize_sections(sections: Iterable[DocumentSection]) -> tuple[str, list[DocumentSection]]:
    finalized = []
    text_parts = []
    cursor = 0
    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        if text_parts:
            cursor += 2
        start = cursor
        text_parts.append(text)
        cursor += len(text)
        finalized.append(replace(section, text=text, start_char=start, end_char=cursor))
    return "\n\n".join(text_parts), finalized


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = "".join(
        character
        for character in text
        if ord(character) >= 32 or character in "\n\t\r"
    )
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def element_text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def section_level(section_id: str) -> int:
    if section_id.lower().startswith("appendix"):
        return 1
    return section_id.count(".") + 1


def match_section_heading(line: str, *, allow_lowercase_title: bool = False):
    match = SECTION_HEADING.match(line)
    if match is None:
        return None
    identifier = match.group("id").rstrip(".")
    title = match.group("title").strip()
    first_component = identifier.split(".", 1)[0]
    if first_component.isdigit() and int(first_component) > 30:
        return None
    words = re.findall(r"[A-Za-z0-9]+", title)
    if len(words) > 14:
        return None
    if title.endswith((".", ",", ";")):
        return None
    if title.lower().startswith(ADMINISTRATIVE_HEADING_PREFIXES):
        return None
    if title[0].islower() and not allow_lowercase_title:
        return None
    if title.lower().startswith(CITATION_HEADING_PREFIXES):
        return None
    return match


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
