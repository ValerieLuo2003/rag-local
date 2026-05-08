from __future__ import annotations

import argparse
import json
import re
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path


SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert BEIR SciFact into this starter format.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory used to store the downloaded BEIR dataset.")
    parser.add_argument("--docs-out", default="data/scifact_docs", help="Output directory for converted Markdown docs.")
    parser.add_argument("--eval-out", default="eval/scifact_eval.jsonl", help="Output JSONL evaluation file.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Qrels split to convert.")
    parser.add_argument("--max-queries", type=int, default=300, help="Limit the number of evaluation queries.")
    parser.add_argument("--max-docs", type=int, default=0, help="Limit converted corpus docs; 0 means full corpus.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete an existing docs output directory.")
    parser.add_argument("--zip-path", help="Use an already downloaded scifact.zip instead of downloading it.")
    parser.add_argument("--insecure-download", action="store_true", help="Download without TLS certificate verification.")
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    dataset_dir = ensure_scifact(raw_root, zip_path=args.zip_path, insecure_download=args.insecure_download)

    corpus = load_jsonl(dataset_dir / "corpus.jsonl")
    queries = {row["_id"]: row["text"] for row in load_jsonl(dataset_dir / "queries.jsonl")}
    qrels = load_qrels(dataset_dir / "qrels" / f"{args.split}.tsv")

    doc_id_to_source = write_markdown_docs(
        corpus=corpus,
        docs_out=Path(args.docs_out),
        max_docs=args.max_docs,
        keep_existing=args.keep_existing,
    )
    written_eval = write_eval_file(
        queries=queries,
        qrels=qrels,
        doc_id_to_source=doc_id_to_source,
        eval_out=Path(args.eval_out),
        max_queries=args.max_queries,
    )

    print(f"SciFact raw data: {dataset_dir}")
    print(f"Converted docs: {args.docs_out} ({len(doc_id_to_source)} docs)")
    print(f"Evaluation file: {args.eval_out} ({written_eval} queries)")


def ensure_scifact(raw_root: Path, zip_path: str | None, insecure_download: bool) -> Path:
    dataset_dir = raw_root / "scifact"
    if (dataset_dir / "corpus.jsonl").exists():
        return dataset_dir

    raw_root.mkdir(parents=True, exist_ok=True)
    archive_path = Path(zip_path) if zip_path else raw_root / "scifact.zip"
    if not archive_path.exists():
        print(f"Downloading {SCIFACT_URL}")
        download_file(SCIFACT_URL, archive_path, insecure=insecure_download)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(raw_root)

    if not (dataset_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Could not find corpus.jsonl after extraction under {dataset_dir}")
    return dataset_dir


def download_file(url: str, output_path: Path, insecure: bool) -> None:
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"User-Agent": "rag-starter/0.1"})
    with urllib.request.urlopen(request, context=context) as response:
        with output_path.open("wb") as file:
            shutil.copyfileobj(response, file)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as file:
        header = next(file, None)
        if header is None:
            return qrels
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, corpus_id, score = parts[:3]
            try:
                relevance = int(score)
            except ValueError:
                continue
            if relevance > 0:
                qrels.setdefault(query_id, []).append(corpus_id)
    return qrels


def write_markdown_docs(
    corpus: list[dict],
    docs_out: Path,
    max_docs: int,
    keep_existing: bool,
) -> dict[str, str]:
    if docs_out.exists() and not keep_existing:
        shutil.rmtree(docs_out)
    docs_out.mkdir(parents=True, exist_ok=True)

    doc_id_to_source: dict[str, str] = {}
    rows = corpus if max_docs <= 0 else corpus[:max_docs]
    for row in rows:
        doc_id = str(row["_id"])
        title = row.get("title") or "Untitled"
        text = row.get("text") or ""
        filename = f"scifact_{sanitize_filename(doc_id)}.md"
        doc_id_to_source[doc_id] = filename
        content = (
            f"# {title}\n\n"
            f"BEIR document id: {doc_id}\n\n"
            f"{text.strip()}\n"
        )
        (docs_out / filename).write_text(content, encoding="utf-8")
    return doc_id_to_source


def write_eval_file(
    queries: dict[str, str],
    qrels: dict[str, list[str]],
    doc_id_to_source: dict[str, str],
    eval_out: Path,
    max_queries: int,
) -> int:
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with eval_out.open("w", encoding="utf-8") as file:
        for query_id, relevant_doc_ids in qrels.items():
            if max_queries > 0 and written >= max_queries:
                break
            relevant_sources = [
                doc_id_to_source[doc_id]
                for doc_id in relevant_doc_ids
                if doc_id in doc_id_to_source
            ]
            if not relevant_sources or query_id not in queries:
                continue
            row = {
                "question": queries[query_id],
                "query_id": query_id,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_sources": relevant_sources,
                "type": "scifact",
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
