"""Per-source-type text extraction + chunking pipeline (build order step 3).

Reads a source version's raw snapshot from disk, extracts a normalized text
representation (the "artifact"), chunks it, and writes chunks to the database
plus the FTS5 index. Re-running with the same chunking parameters is a no-op
(idempotent); running with different parameters creates a new artifact
generation and leaves the old chunks untouched (see "Retention vs. Evidence
Integrity" in PLAN_KB_ARCHITECTURE.md — chunks must stay immutable once
anything might reference them as evidence).
"""

import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from deep_research.config import Config
from deep_research.kb.canonical import sha256_bytes
from deep_research.kb.chunking import chunk_text, chunk_transcript_segments, estimate_tokens
from deep_research.kb.db import KBDatabase
from deep_research.kb.embeddings import embed_texts
from deep_research.kb.storage import SnapshotStore
from deep_research.tools.scrape import _extract_text

CHUNKER_VERSION = "fixed_size_v1"


@dataclass
class ArtifactBuildResult:
    status: str  # "chunked" | "unchanged" | "empty"
    artifact_id: str | None = None
    artifact_created: bool = False
    chunk_count: int = 0
    embedded_count: int = 0


def _chunk_params_hash(chunk_size: int, extractor: str) -> str:
    payload = json.dumps({"chunker": CHUNKER_VERSION, "size": chunk_size, "extractor": extractor}, sort_keys=True)
    return sha256_bytes(payload.encode())


def _extract_web_html(raw_bytes: bytes) -> str:
    _title, text = _extract_text(raw_bytes.decode("utf-8", errors="replace"))
    return text


def _extract_docx(raw_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(raw_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_pdf_pages(raw_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(raw_bytes))
    return [(page.extract_text() or "") for page in reader.pages]


async def build_artifact_for_version(
    kb_db: KBDatabase,
    snapshot_store: SnapshotStore,
    source: dict,
    version: dict,
    config: Config | None = None,
    chunk_size: int = 1200,
) -> ArtifactBuildResult:
    source_type_code = await kb_db.get_source_type_code(source["source_type_id"])
    raw_bytes = Path(version["snapshot_path"]).read_bytes()

    if source_type_code in ("web", "html_file"):
        artifact_type, extractor = "clean_text", "bs4_extract_text_v1"
        pages = None
        text = _extract_web_html(raw_bytes)
        segments = None
    elif source_type_code in ("markdown", "text"):
        artifact_type, extractor = "clean_text", "utf8_decode_v1"
        text = raw_bytes.decode("utf-8", errors="replace")
        pages = None
        segments = None
    elif source_type_code == "docx":
        artifact_type, extractor = "parsed_docx", "python_docx_v1"
        text = _extract_docx(raw_bytes)
        pages = None
        segments = None
    elif source_type_code == "pdf":
        artifact_type, extractor = "parsed_pdf", "pypdf_v1"
        pages = _extract_pdf_pages(raw_bytes)
        text = None
        segments = None
    elif source_type_code == "youtube_video":
        artifact_type, extractor = "transcript", "youtube_transcript_api_v1"
        segments = json.loads(raw_bytes.decode("utf-8"))
        text = None
        pages = None
    else:
        raise ValueError(f"No artifact extractor for source type: {source_type_code!r}")

    chunk_params_hash = _chunk_params_hash(chunk_size, extractor)

    if text is not None:
        content_for_hash = text.encode("utf-8")
    elif pages is not None:
        content_for_hash = "\f".join(pages).encode("utf-8")
    else:
        content_for_hash = json.dumps(segments, sort_keys=True).encode("utf-8")
    content_hash = sha256_bytes(content_for_hash)

    current = await kb_db.get_current_artifact(version["id"], artifact_type)
    if current is not None and current["chunk_params_hash"] == chunk_params_hash:
        existing_chunks = await kb_db.list_chunks(current["id"])
        return ArtifactBuildResult(
            status="unchanged", artifact_id=current["id"],
            artifact_created=False, chunk_count=len(existing_chunks),
        )

    artifact_id = str(uuid.uuid4())
    storage_path = snapshot_store.write_artifact(artifact_id, content_for_hash, ext=".txt")

    artifact_row, created = await kb_db.upsert_artifact(
        artifact_id=artifact_id,
        source_version_id=version["id"],
        artifact_type=artifact_type,
        storage_path=str(storage_path),
        content_hash=content_hash,
        chunk_params_hash=chunk_params_hash,
        title=source.get("title"),
    )
    if not created:
        # Lost a race / already had this exact generation — no new file needed.
        snapshot_store.delete(storage_path)
        existing_chunks = await kb_db.list_chunks(artifact_row["id"])
        return ArtifactBuildResult(
            status="unchanged", artifact_id=artifact_row["id"],
            artifact_created=False, chunk_count=len(existing_chunks),
        )

    chunk_count = 0
    created_chunks = []
    if text is not None:
        for idx, (chunk_str, char_start, char_end) in enumerate(chunk_text(text, chunk_size)):
            created_chunks.append(await kb_db.add_chunk(
                artifact_id, idx, chunk_str, sha256_bytes(chunk_str.encode()),
                char_start=char_start, char_end=char_end,
                token_estimate=estimate_tokens(chunk_str),
            ))
            chunk_count += 1
    elif pages is not None:
        idx = 0
        for page_number, page_text in enumerate(pages, start=1):
            for chunk_str, char_start, char_end in chunk_text(page_text, chunk_size):
                created_chunks.append(await kb_db.add_chunk(
                    artifact_id, idx, chunk_str, sha256_bytes(chunk_str.encode()),
                    char_start=char_start, char_end=char_end,
                    token_estimate=estimate_tokens(chunk_str), page_number=page_number,
                ))
                idx += 1
                chunk_count += 1
    else:
        for idx, (chunk_str, t_start, t_end) in enumerate(chunk_transcript_segments(segments, chunk_size)):
            created_chunks.append(await kb_db.add_chunk(
                artifact_id, idx, chunk_str, sha256_bytes(chunk_str.encode()),
                token_estimate=estimate_tokens(chunk_str),
                time_start_seconds=t_start, time_end_seconds=t_end,
            ))
            chunk_count += 1

    embedded_count = 0
    if created_chunks and config is not None:
        try:
            vectors = await embed_texts(
                [c["chunk_text"] for c in created_chunks],
                config.kb.embedding_base_url, config.kb.embedding_model,
            )
            for chunk_row, vector in zip(created_chunks, vectors):
                await kb_db.set_chunk_embedding(chunk_row["id"], vector)
                embedded_count += 1
        except Exception:
            # Best-effort: embeddings are a retrieval enhancement, not a
            # correctness requirement -- ingestion must still succeed if the
            # embedding backend (Ollama) is unreachable. `backfill-embeddings`
            # picks up anything left NULL here.
            pass

    if chunk_count == 0:
        return ArtifactBuildResult(status="empty", artifact_id=artifact_id, artifact_created=True, chunk_count=0)

    return ArtifactBuildResult(
        status="chunked", artifact_id=artifact_id, artifact_created=True,
        chunk_count=chunk_count, embedded_count=embedded_count,
    )
