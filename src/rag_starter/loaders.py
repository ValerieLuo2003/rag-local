from __future__ import annotations

from pathlib import Path

from .schema import Document


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def load_documents(path: str | Path) -> list[Document]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Document path does not exist: {root}")

    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    documents: list[Document] = []

    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_file(file_path)
        if text.strip():
            documents.append(Document(text=text, source=file_path.name))

    return documents


def _read_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(file_path)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading PDF files requires pypdf. Run: pip install pypdf") from exc

    reader = PdfReader(str(file_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n[page {page_number}]\n{text}")
    return "\n".join(pages)

