from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DocumentSection:
    section_id: str
    title: str
    text: str
    level: int = 1
    page_start: int | None = None
    page_end: int | None = None
    start_char: int = 0
    end_char: int = 0


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    doc_id: str = ""
    title: str = ""
    source_type: str = "local"
    version: str = ""
    status: str = ""
    published_at: str = ""
    canonical_url: str = ""
    download_url: str = ""
    sha256: str = ""
    license: str = ""
    supersedes: tuple[str, ...] = ()
    superseded_by: str | None = None
    topics: tuple[str, ...] = ()
    sections: tuple[DocumentSection, ...] = ()
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_id: str
    start_char: int
    end_char: int
    doc_id: str = ""
    title: str = ""
    source_type: str = "local"
    version: str = ""
    status: str = ""
    canonical_url: str = ""
    supersedes: tuple[str, ...] = ()
    superseded_by: str | None = None
    topics: tuple[str, ...] = ()
    section_id: str = ""
    section_title: str = ""
    page_start: int | None = None
    page_end: int | None = None

    def citation_label(self) -> str:
        identity = self.title or self.doc_id or self.source
        if self.version:
            identity = f"{identity} ({self.version})"
        locations = []
        if self.section_id:
            section = f"Section {self.section_id}"
            if self.section_title and self.section_title.lower() not in self.section_id.lower():
                section += f" {self.section_title}"
            locations.append(section)
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                locations.append(f"Pages {self.page_start}-{self.page_end}")
            else:
                locations.append(f"Page {self.page_start}")
        return ", ".join([identity, *locations])


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
