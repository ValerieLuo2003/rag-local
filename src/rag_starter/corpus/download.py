from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..loaders import load_manifest


USER_AGENT = "cryptosec-verifiable-rag/0.2 (+https://github.com/ValerieLuo2003/rag-local)"
ALLOWED_HOSTS = {
    "nist": {"nvlpubs.nist.gov"},
    "rfc": {"www.rfc-editor.org"},
}
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a curated NIST/RFC manifest.")
    parser.add_argument("--manifest", default="data/corpus_manifest.jsonl")
    parser.add_argument("--state-file", default="data/corpus_state.json")
    parser.add_argument("--source-type", choices=["nist", "rfc"])
    parser.add_argument("--only-quality-review", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    if args.source_type:
        entries = [entry for entry in entries if entry["source_type"] == args.source_type]
    if args.only_quality_review:
        entries = [entry for entry in entries if entry.get("quality_review") is True]
    if args.limit > 0:
        entries = entries[: args.limit]
    if not entries:
        raise ValueError("No manifest entries matched the requested filters")

    for entry in entries:
        validate_entry_url(entry)

    if args.dry_run:
        for entry in entries:
            print(f"{entry['doc_id']}\t{entry['download_url']}\t{entry['local_path']}")
        print(f"documents={len(entries)}")
        return

    state = load_state(args.state_file)
    failures = []
    for index, entry in enumerate(entries, start=1):
        try:
            record = download_entry(
                entry,
                timeout=args.timeout,
                retries=args.retries,
                force=args.force,
            )
            state[entry["doc_id"]] = record
            write_state(args.state_file, state)
            print(
                f"[{index}/{len(entries)}] {record['action']} {entry['doc_id']} "
                f"bytes={record['bytes']} sha256={record['sha256'][:12]}"
            )
        except Exception as exc:
            failures.append({"doc_id": entry["doc_id"], "error": str(exc)})
            print(f"[{index}/{len(entries)}] FAILED {entry['doc_id']}: {exc}")
        if index < len(entries) and args.delay > 0:
            time.sleep(args.delay)

    print(f"downloaded_or_verified={len(entries) - len(failures)}")
    print(f"failed={len(failures)}")
    print(f"state_file={args.state_file}")
    if failures:
        raise SystemExit(1)


def download_entry(
    entry: dict,
    *,
    timeout: float = 45.0,
    retries: int = 3,
    force: bool = False,
) -> dict:
    validate_entry_url(entry)
    target = Path(entry["local_path"])
    if target.exists() and not force:
        validate_local_payload(target, entry["source_type"])
        return state_record(entry, target, action="verified-existing")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                entry["download_url"],
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/pdf, application/xml, text/xml, text/plain;q=0.9, */*;q=0.1",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final_url = response.geturl()
                validate_download_url(final_url, entry["source_type"])
                headers = {
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type"),
                }
                digest = hashlib.sha256()
                total_bytes = 0
                with temporary.open("wb") as file:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total_bytes += len(block)
                        if total_bytes > MAX_DOCUMENT_BYTES:
                            raise ValueError(
                                f"{entry['doc_id']} exceeds {MAX_DOCUMENT_BYTES} bytes"
                            )
                        digest.update(block)
                        file.write(block)
                validate_local_payload(temporary, entry["source_type"])
                os.replace(temporary, target)
                return state_record(
                    entry,
                    target,
                    action="downloaded",
                    sha256=digest.hexdigest(),
                    headers=headers,
                    final_url=final_url,
                )
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 5))

    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def validate_entry_url(entry: dict) -> None:
    source_type = str(entry.get("source_type"))
    if source_type not in ALLOWED_HOSTS:
        raise ValueError(f"Unsupported source_type={source_type!r}")
    validate_download_url(str(entry.get("download_url", "")), source_type)


def validate_download_url(url: str, source_type: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS downloads are allowed: {url}")
    hostname = (parsed.hostname or "").lower()
    if hostname not in ALLOWED_HOSTS[source_type]:
        raise ValueError(
            f"Host {hostname!r} is not allowed for source_type={source_type!r}"
        )
    if parsed.username or parsed.password:
        raise ValueError("Credentials in manifest URLs are not allowed")


def validate_local_payload(path: str | Path, source_type: str) -> None:
    file_path = Path(path)
    size = file_path.stat().st_size
    if size <= 0:
        raise ValueError(f"Downloaded file is empty: {file_path}")
    with file_path.open("rb") as file:
        prefix = file.read(256).lstrip()
    if source_type == "nist" and not prefix.startswith(b"%PDF"):
        raise ValueError(f"Expected a PDF payload for {file_path}")
    if source_type == "rfc":
        is_xml = (
            prefix.startswith(b"<?xml")
            or prefix.startswith(b"<rfc")
            or b"<rfc " in prefix
        )
        is_text = (
            b"Request for Comments:" in prefix
            or b"Internet Engineering Task Force" in prefix
            or b"Network Working Group" in prefix
        )
        if not (is_xml or is_text):
            raise ValueError(f"Expected an RFC XML/TXT payload for {file_path}")


def state_record(
    entry: dict,
    path: Path,
    *,
    action: str,
    sha256: str | None = None,
    headers: dict | None = None,
    final_url: str | None = None,
) -> dict:
    return {
        "doc_id": entry["doc_id"],
        "source_type": entry["source_type"],
        "download_url": entry["download_url"],
        "final_url": final_url or entry["download_url"],
        "local_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256 or sha256_file(path),
        "etag": (headers or {}).get("etag"),
        "last_modified": (headers or {}).get("last_modified"),
        "content_type": (headers or {}).get("content_type"),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
    }


def load_state(path: str | Path) -> dict[str, dict]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"State file must contain a JSON object: {state_path}")
    return value


def write_state(path: str | Path, state: dict[str, dict]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, state_path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
